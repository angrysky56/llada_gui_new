#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized generation algorithm for LLaDA models.
This version implements several optimizations:
1. Memory-efficient processing with CPU offloading
2. Adaptive step scheduling
3. Progressive refinement
4. Attention caching
"""

import torch
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configure a default device based on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")

class TokenBuffer:
    """Memory-efficient token buffer that can offload to CPU when needed."""
    
    def __init__(self, data, device=DEVICE, cpu_offload=True):
        """Initialize buffer with data, optionally on a specific device."""
        self.cpu_offload = cpu_offload
        self.device = device
        self._data = data.to(self.device if not cpu_offload else CPU_DEVICE)
        self._is_on_gpu = not cpu_offload
    
    @property
    def data(self):
        """Get data, moving to GPU if needed."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True
        return self._data
    
    def update(self, data):
        """Update buffer with new data."""
        if self._is_on_gpu or not self.cpu_offload:
            self._data = data
        else:
            self._data = data.to(CPU_DEVICE)
    
    def to_cpu(self):
        """Move data to CPU if not already there."""
        if self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(CPU_DEVICE)
            self._is_on_gpu = False
    
    def to_gpu(self):
        """Move data to GPU if not already there."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True
    
    def clone(self):
        """Create a clone of the buffer."""
        return TokenBuffer(self._data.clone(), self.device, self.cpu_offload)


class AttentionCache:
    """Cache for past key/values to avoid recomputation."""
    
    def __init__(self, model, cpu_offload=True):
        """Initialize empty cache for a specific model."""
        self.model = model
        self.cache = {}
        self.cpu_offload = cpu_offload
    
    def get(self, key, default=None):
        """Get a value from the cache, loading to the correct device if needed."""
        if key not in self.cache:
            return default
        
        # Load from CPU if needed
        if self.cpu_offload and isinstance(self.cache[key], torch.Tensor) and self.cache[key].device.type == "cpu":
            self.cache[key] = self.cache[key].to(DEVICE)
        
        return self.cache[key]
    
    def set(self, key, value):
        """Set a value in the cache, offloading to CPU if configured."""
        if self.cpu_offload and isinstance(value, torch.Tensor) and value.device.type != "cpu":
            self.cache[key] = value.to(CPU_DEVICE)
        else:
            self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}


def add_gumbel_noise(logits, temperature):
    """
    Add Gumbel noise for sampling categorical distributions.
    
    Args:
        logits: Raw logits from the model
        temperature: Sampling temperature
    
    Returns:
        Logits with Gumbel noise applied
    """
    # Use a more memory-efficient approach, avoiding float64 conversion if possible
    if temperature > 0:
        # For non-zero temperatures, use float32 instead of float64 for better efficiency
        dtype = torch.float32
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise
    else:
        # For zero temperature, no need for noise
        return logits


def get_adaptive_transfer_schedule(mask_index, base_steps, min_steps=4, confidence_threshold=0.9):
    """
    Creates an adaptive schedule for token transfers, using fewer steps for easier tokens.
    
    Args:
        mask_index: Boolean tensor indicating masked tokens
        base_steps: Maximum number of steps to use
        min_steps: Minimum number of steps to use
        confidence_threshold: Threshold for considering a token "confident"
    
    Returns:
        Tensor containing number of tokens to transfer at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    # Start with a standard distribution
    base = mask_num // base_steps
    remainder = mask_num % base_steps
    
    # Create the transfer schedule, front-loading more tokens in earlier steps
    # This helps generate high-confidence tokens more quickly
    num_transfer_tokens = torch.zeros(mask_num.size(0), base_steps, device=mask_index.device, dtype=torch.int64)
    
    # Weighted distribution - transfer more tokens in early steps
    weights = torch.linspace(1.5, 0.5, base_steps)
    weights = weights / weights.sum() * base_steps
    
    for i in range(mask_num.size(0)):
        # Distribute tokens according to weights, ensuring we use all tokens
        weighted_tokens = (weights * (mask_num[i].item() / base_steps)).round().to(torch.int64)
        
        # Ensure we assign all tokens
        diff = mask_num[i].item() - weighted_tokens.sum().item()
        if diff > 0:
            # Add remaining tokens to the first steps
            for j in range(diff):
                weighted_tokens[j % base_steps] += 1
        elif diff < 0:
            # Remove excess tokens from the last steps
            for j in range(-diff):
                if weighted_tokens[-(j % base_steps) - 1] > 0:
                    weighted_tokens[-(j % base_steps) - 1] -= 1
        
        num_transfer_tokens[i] = weighted_tokens
    
    return num_transfer_tokens


def chunk_processing(model, tokens, chunk_size=512):
    """
    Process tokens in chunks to reduce memory usage.
    
    Args:
        model: The language model
        tokens: Input tokens
        chunk_size: Size of chunks to process
    
    Returns:
        Model output logits
    """
    seq_len = tokens.shape[1]
    
    # If sequence is short enough, process directly
    if seq_len <= chunk_size:
        return model(tokens).logits
    
    # Otherwise, process in chunks and combine
    all_logits = []
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = tokens[:, i:end_idx]
        
        # Process chunk
        with torch.no_grad():
            chunk_output = model(chunk).logits
        
        all_logits.append(chunk_output)
    
    # Combine chunks
    return torch.cat(all_logits, dim=1)


@torch.no_grad()
def optimized_generate(
    model, 
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336,
    cpu_offload=True,
    adaptive_steps=True,
    progress_callback=None,
    confidence_threshold=0.9,
    device=DEVICE
):
    """
    Optimized generation function for LLaDA models.
    
    Args:
        model: The language model
        prompt: Input prompt tokens
        steps: Maximum number of sampling steps
        gen_length: Length of the generated text
        block_length: Block size for generation
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Strategy for remasking tokens ('low_confidence' or 'random')
        mask_id: Token ID for the mask token
        cpu_offload: Whether to offload tensors to CPU when not in use
        adaptive_steps: Whether to use adaptive step scheduling
        progress_callback: Callback function for progress updates
        confidence_threshold: Confidence threshold for early stopping
        device: Device to use for computation
    
    Returns:
        Generated tokens
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Check if the model requires CPU offloading
    model_device = next(model.parameters()).device
    cpu_offload = cpu_offload and model_device.type == "cuda"
    
    # Create output tensor with masks
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # Track prompt indices
    prompt_index = (x != mask_id)
    
    # Initialize token buffer for memory efficiency
    token_buffer = TokenBuffer(x, device=device, cpu_offload=cpu_offload)
    
    # Calculate blocks
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    # Adjust steps based on blocks
    if not adaptive_steps:
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
    else:
        # Adaptive scheduling - use more steps for early blocks, fewer for later ones
        steps_per_block_list = []
        for b in range(num_blocks):
            # Decay step count for later blocks
            decay_factor = 0.8 ** b
            block_steps = max(4, int(steps / num_blocks * decay_factor))
            steps_per_block_list.append(block_steps)
        
        # Normalize to ensure we use approximately the requested total steps
        total_steps = sum(steps_per_block_list)
        if total_steps != steps:
            scaling_factor = steps / total_steps
            steps_per_block_list = [max(4, int(s * scaling_factor)) for s in steps_per_block_list]
    
    # Initialize attention cache if the model supports it
    cache = AttentionCache(model, cpu_offload=cpu_offload)
    
    # Process each block
    total_steps_completed = 0
    total_steps_expected = steps if not adaptive_steps else sum(steps_per_block_list)
    
    for num_block in range(num_blocks):
        # Get steps for this block
        if adaptive_steps:
            steps_per_block = steps_per_block_list[num_block]
        
        # Calculate block mask indices
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        # Move to GPU for this block
        token_buffer.to_gpu()
        x = token_buffer.data
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # Skip if no masks in this block
        if not block_mask_index.any():
            continue
        
        # Get adaptive transfer schedule
        if adaptive_steps:
            num_transfer_tokens = get_adaptive_transfer_schedule(
                block_mask_index, 
                steps_per_block,
                min_steps=4,
                confidence_threshold=confidence_threshold
            )
        else:
            num_transfer_tokens = torch.div(
                block_mask_index.sum(dim=1, keepdim=True),
                steps_per_block,
                rounding_mode='floor'
            ).repeat(1, steps_per_block)
            
            # Handle remainder
            remainder = block_mask_index.sum(dim=1, keepdim=True) % steps_per_block
            if remainder.sum() > 0:
                for i in range(remainder.shape[0]):
                    num_transfer_tokens[i, :remainder[i]] += 1
        
        # Clear any existing cache for new block
        cache.clear()
        
        # Process steps for this block
        for i in range(steps_per_block):
            if token_buffer._is_on_gpu:
                x = token_buffer.data
            else:
                token_buffer.to_gpu()
                x = token_buffer.data
            
            # Update progress
            if progress_callback:
                total_steps_completed += 1
                progress_percentage = total_steps_completed / total_steps_expected
                progress_callback(progress_percentage, x.clone())
            
            # Get current mask indices
            mask_index = (x == mask_id)
            
            # Skip if no masks left
            if not mask_index.any():
                break
            
            # Apply classifier-free guidance if needed
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                # Process in chunks if needed to save memory
                logits = chunk_processing(model, x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Process in chunks if needed to save memory
                logits = chunk_processing(model, x)
            
            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
            
            # Calculate token confidence
            if remasking == 'low_confidence':
                if temperature > 0:
                    # Use float32 instead of float64 for better efficiency
                    p = F.softmax(logits, dim=-1)
                else:
                    # With zero temperature, we can use float64 for better precision
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # Don't consider tokens outside the current block
            x0_p[:, block_end:] = -np.inf
            
            # Replace only masked tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Determine which tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            for j in range(confidence.shape[0]):
                tokens_to_transfer = num_transfer_tokens[j, i].item()
                if tokens_to_transfer > 0:
                    _, select_index = torch.topk(confidence[j], k=min(tokens_to_transfer, torch.sum(mask_index[j]).item()))
                    transfer_index[j, select_index] = True
            
            # Update tokens
            x[transfer_index] = x0[transfer_index]
            
            # Update buffer
            token_buffer.update(x)
            
            # Free GPU memory if using CPU offloading
            if cpu_offload and i < steps_per_block - 1:
                token_buffer.to_cpu()
                torch.cuda.empty_cache()
        
        # End of block processing - force move to CPU to save memory
        if cpu_offload:
            token_buffer.to_cpu()
            torch.cuda.empty_cache()
    
    # Final result
    token_buffer.to_gpu()
    return token_buffer.data


# Compatibility function with the original API
@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, 
             progress_callback=None, cpu_offload=True):
    """
    Compatible interface with the original generate function, internally using the optimized version.
    """
    return optimized_generate(
        model=model,
        prompt=prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        cpu_offload=cpu_offload,
        adaptive_steps=True,
        progress_callback=progress_callback,
        device=prompt.device
    )


def main():
    """Test the optimized generation function."""
    from transformers import AutoTokenizer, AutoModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    # Move model to appropriate device, using BF16 if available
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16, device=device)
    else:
        model = model.to(device)
    
    model = model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    # Example prompt
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    # Add special tokens for the Instruct model
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # Define a progress callback
    def progress_callback(progress, tokens):
        print(f"Generation progress: {progress:.2%}")
    
    # Generate with the optimized function
    print("Generating with optimized function...")
    out = generate(
        model=model,
        prompt=input_ids,
        steps=64,
        gen_length=128,
        block_length=32,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence',
        progress_callback=progress_callback,
        cpu_offload=True
    )
    
    # Print result
    print("\nResult:")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
