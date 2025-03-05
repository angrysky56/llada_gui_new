#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory-guided generation algorithm for LLaDA models.

This version extends the standard LLaDA generation process with a memory system
that helps maintain coherence and consistency across the generated text.
"""

import torch
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from generate import (
    TokenBuffer, AttentionCache, add_gumbel_noise, 
    get_adaptive_transfer_schedule, chunk_processing
)

logger = logging.getLogger(__name__)

# Configure a default device based on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


@torch.no_grad()
def memory_guided_generate(
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
    memory_callback=None,
    memory_interface=None,
    memory_weight=0.3,
    confidence_threshold=0.9,
    device=DEVICE
):
    """
    Memory-guided generation function for LLaDA models.
    
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
        memory_callback: Callback function for memory state updates
        memory_interface: Memory interface for guidance
        memory_weight: Weight of memory influence (0-1)
        confidence_threshold: Confidence threshold for early stopping
        device: Device to use for computation
    
    Returns:
        Generated tokens
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Check if memory interface is available
    has_memory = memory_interface is not None and memory_interface.initialized
    
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
    
    # Initialize memory state
    if has_memory:
        try:
            # Reset memory state at the beginning
            memory_interface.reset()
            
            # Convert prompt text to embedding
            # We'll use a simple approach here, but a more sophisticated one could be used
            prompt_tokens = x[0, :prompt.shape[1]].cpu().numpy().tolist()
            prompt_embedding = np.zeros(memory_interface.input_dim)
            
            # Create a simple embedding by hashing token positions
            for i, token in enumerate(prompt_tokens):
                # Use token ID and position to influence the embedding
                idx = (token * 17 + i * 31) % memory_interface.input_dim
                prompt_embedding[idx] += 1.0
            
            # Normalize the embedding
            prompt_embedding = prompt_embedding / max(1.0, np.sum(prompt_embedding))
            
            # Update memory with prompt
            memory_state = memory_interface.forward_pass(prompt_embedding)
            
            # Report initial memory state
            if memory_callback:
                memory_callback(memory_interface.get_memory_state())
                
        except Exception as e:
            logger.error(f"Memory initialization error: {e}")
            has_memory = False
    
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
            
            # Apply memory influence if available
            if has_memory:
                try:
                    # Apply memory influence to logits
                    # This is the key step where memory guides token selection
                    
                    # Extract current non-masked tokens to update memory state
                    if i > 0:  # Don't need to do this on the first step
                        # Get the tokens from the previous step that were unmasked
                        current_tokens = x[0, block_start:block_end].cpu().numpy().tolist()
                        
                        # Create an embedding of the current tokens
                        current_embedding = np.zeros(memory_interface.input_dim)
                        for j, token in enumerate(current_tokens):
                            if token != mask_id:
                                idx = (token * 17 + j * 31) % memory_interface.input_dim
                                current_embedding[idx] += 1.0
                        
                        # Normalize the embedding
                        if np.sum(current_embedding) > 0:
                            current_embedding = current_embedding / np.sum(current_embedding)
                            
                            # Update memory with current token context
                            memory_state = memory_interface.forward_pass(current_embedding)
                    
                    # Get memory prediction for the next token
                    # The predicted vector represents the "memory's expectation"
                    memory_prediction = np.array(memory_state.get("predicted", []))
                    
                    # Apply memory prediction to influence token probabilities
                    # We only want to influence masked positions
                    if len(memory_prediction) > 0:
                        # This implementation is a simplified version 
                        # A more sophisticated approach would map memory embeddings to token space
                        vocab_size = logits.size(-1)
                        
                        # Create a memory bias tensor
                        memory_bias = torch.zeros((1, x.size(1), vocab_size), device=logits.device)
                        
                        # Apply memory influence only to masked positions
                        for j in range(x.size(1)):
                            if mask_index[0, j]:
                                # For each masked position, create a bias based on memory
                                for k, value in enumerate(memory_prediction):
                                    # Calculate the influenced tokens
                                    # This uses a simple hashing approach to spread influence across similar tokens
                                    if value > 0.01:  # Only consider significant values
                                        for offset in range(-2, 3):  # Influence nearby tokens
                                            token_idx = (k * 17 + offset * 31) % vocab_size
                                            memory_bias[0, j, token_idx] += value * (1.0 - abs(offset) * 0.2)
                        
                        # Normalize the bias and apply it
                        max_bias = memory_bias.max().item()
                        if max_bias > 0:
                            memory_bias = memory_bias / max_bias
                            
                            # Blend original logits with memory bias
                            blended_logits = (1 - memory_weight) * logits + memory_weight * memory_bias
                            logits = blended_logits
                    
                    # Report memory state after processing
                    if memory_callback:
                        memory_callback(memory_interface.get_memory_state())
                        
                except Exception as e:
                    logger.error(f"Memory application error: {e}")
            
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
def generate(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    cfg_scale=0., remasking='low_confidence', mask_id=126336, 
    progress_callback=None, memory_callback=None, memory_interface=None, 
    memory_weight=0.3, cpu_offload=True
):
    """
    Memory-guided generation function with an interface compatible with the original generate function.
    """
    return memory_guided_generate(
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
        memory_callback=memory_callback,
        memory_interface=memory_interface,
        memory_weight=memory_weight,
        device=prompt.device
    )
