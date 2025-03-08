#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Optimized generation algorithm for LLaDA models.
This version implements several optimizations:
1. Memory-efficient processing with CPU offloading
2. Adaptive step scheduling
3. Memory guidance integration
"""

import logging
from typing import Optional, Callable, List

import torch
import torch.nn.functional as F
from torch import topk, cuda, stack, bool, log, chunk, no_grad, long, zeros_like, empty, div, \
    int64, cat, where, Tensor, gather, tensor, zeros, argmax, rand_like, device, linspace, \
    full

logger = logging.getLogger(__name__)
torch._dynamo.config.capture_scalar_outputs = True

DEVICE = device("cuda" if cuda.is_available() else "cpu")
CPU_DEVICE = device("cpu")
MASK_TOKEN_ID_DEFAULT = 126336  # Default mask token ID
EPSILON = 1e-10


class TokenBuffer:
    """Memory-efficient token buffer that can offload to CPU when needed."""

    def __init__(self, data, device=DEVICE, cpu_offload=True):
        """Initialize buffer with data on the specified device."""
        self.cpu_offload = cpu_offload
        self.device = device
        self._data = data.to(device if not cpu_offload else CPU_DEVICE)
        self._is_on_gpu = not cpu_offload

    @property
    def data(self) -> Tensor:
        """Get data, moving to GPU if needed."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True
        return self._data

    def update(self, data):
        """Update buffer with new data."""
        self._data = data if self._is_on_gpu or not self.cpu_offload else data.to(CPU_DEVICE)

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

    def clone(self) -> 'TokenBuffer':
        """Create a clone of the buffer."""
        return TokenBuffer(self._data.clone(), self.device, self.cpu_offload)


@torch.compile
def add_gumbel_noise(logits: Tensor, temperature: float, epsilon: float) -> Tensor:
    """
    Add Gumbel noise for sampling categorical distributions.

    Args:
        logits: Raw logits from the model
        temperature: Sampling temperature

    Returns:
        Logits with Gumbel noise applied
    """
    noise = rand_like(logits)

    gumbel_noise = (-log(noise + epsilon)) ** temperature

    return logits / gumbel_noise


@torch.compile
def get_adaptive_transfer_schedule(mask_index: Tensor, steps: int) -> Tensor:
    """
    Create an adaptive schedule for token transfers, front-loading more tokens in earlier steps.

    Args:
        mask_index: Boolean tensor indicating masked tokens
        steps: Number of steps to schedule
        min_steps: Minimum number of steps (default: 4)

    Returns:
        Tensor of shape (batch_size, steps) with number of tokens to transfer per step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = zeros(mask_num.size(0), steps, device=mask_index.device, dtype=int64)

    weights = linspace(1.5, 0.5, steps)
    weights = weights / weights.sum() * steps

    for i in range(mask_num.size(0)):
        maski = mask_num[i]
        mi = maski.item()
        weighted_tokens = (weights * (mi / steps)).round().to(int64)
        diff = mi - weighted_tokens.sum().item()
        if diff > 0:
            for j in range(diff):
                weighted_tokens[j % steps] += 1
        elif diff < 0:
            for j in range(-diff):
                index = -(j % steps) - 1
                if weighted_tokens[index] > 0:
                    weighted_tokens[index] -= 1
        num_transfer_tokens[i] = weighted_tokens

    return num_transfer_tokens


def chunk_processing(model, tokens: Tensor, chunk_size: int = 512) -> Tensor:
    """
    Process tokens in chunks to reduce memory usage for long sequences.

    Args:
        model: Language model
        tokens: Input token tensor
        chunk_size: Size of each chunk (default: 512)

    Returns:
        Concatenated logits from model output
    """
    seq_len = tokens.shape[1]
    if seq_len <= chunk_size:
        return model(tokens).logits

    all_logits = []
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = tokens[:, i:end_idx]
        with no_grad():
            chunk_output = model(chunk).logits
        all_logits.append(chunk_output)
    return cat(all_logits, dim=1)


def _get_model_logits(model, x, cfg_scale, chunk_size, mask_id, prompt_index):
    """Helper function to get logits from the model, with optional CFG."""
    if cfg_scale > 0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        logits, un_logits = chunk(chunk_processing(model, cat([x, un_x], dim=0), chunk_size=chunk_size), 2, dim=0)
        return un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        return chunk_processing(model, x, chunk_size=chunk_size)



def apply_memory_guidance(logits, x, memory_integration, mask_index):
    """
    Apply memory guidance to adjust logits based on external memory.

    Args:
        logits: Model logits
        x: Current sequence tokens
        memory_integration: Memory integration module (optional)
        mask_index: Boolean tensor indicating masked tokens

    Returns:
        Adjusted logits
    """
    if memory_integration and memory_integration.is_initialized():
        token_probs = F.softmax(logits, dim=-1)
        tokenProbSize = token_probs.size(-1)
        token_sequence = x[0].tolist() # Keep as tensor if memory_integration can handle it
        for pos in range(x.size(1)):
            if mask_index[0, pos]:
                pos_probs = token_probs[0, pos] # Keep as tensor if memory_integration can handle it
                guided_probs = memory_integration.apply_memory_guidance(
                    pos_probs, token_sequence, tokenProbSize
                )
                token_probs[0, pos] = guided_probs # Assuming memory_integration returns tensor
        logits = log(token_probs + EPSILON)
    return logits


def _select_tokens_to_transfer_block(confidence_block, mask_index_block, num_transfer_tokens_block, step):
    """Helper function to select tokens to transfer within a block."""
    transfer_index_block = zeros_like(confidence_block, dtype=bool, device=confidence_block.device)
    masked_positions = mask_index_block.nonzero().squeeze(0) # Squeeze for single batch
    num_masked = masked_positions.size(0)
    tokens_to_transfer = num_transfer_tokens_block[step].item()
    if tokens_to_transfer > 0 and num_masked > 0:
        k = min(tokens_to_transfer, num_masked)
        _, select_index = topk(confidence_block[masked_positions], k=k)
        transfer_index_block[masked_positions[select_index]] = True
    return transfer_index_block


def select_tokens_to_transfer(confidence, mask_index, num_transfer_tokens, step):
    """
    Select tokens to transfer based on confidence scores and schedule.

    Args:
        confidence: Confidence scores for each token
        mask_index: Boolean tensor indicating masked tokens
        num_transfer_tokens: Tensor with number of tokens to transfer per step
        step: Current step index

    Returns:
        Boolean tensor indicating tokens to transfer
    """
    batch_size = confidence.shape[0]
    transfer_index = zeros_like(confidence, dtype=bool, device=confidence.device)
    for j in range(batch_size):
        masked_positions = mask_index[j].nonzero().squeeze(1)
        num_masked = masked_positions.size(0)
        tokens_to_transfer = num_transfer_tokens[j, step].item()
        if tokens_to_transfer > 0 and num_masked > 0:
            k = min(tokens_to_transfer, num_masked)
            _, select_index = topk(confidence[j, masked_positions], k=k)
            transfer_index[j, masked_positions[select_index]] = True
    return transfer_index


def get_model_logits(cfg_scale, chunk_size, mask_id, mask_index, memory_integration, model, prompt_index, x):
    logits = _get_model_logits(model, x, cfg_scale, chunk_size, mask_id, prompt_index)
    logits = apply_memory_guidance(logits, x, memory_integration, mask_index)

    #if logits.dtype != torch.float32: logits = logits.to(torch.float32)

    return logits

def logitSample(cfg_scale, chunk_size, mask_id, mask_index, memory_integration, model, prompt_index, remasking, temperature, x):
    # Model processing and guidance
    logits = get_model_logits(cfg_scale, chunk_size, mask_id, mask_index, memory_integration, model, prompt_index, x)

    if temperature>0:
        logits = add_gumbel_noise(logits, temperature, EPSILON)

    x0 = argmax(logits, dim=-1)

    # Compute confidence for remasking
    if remasking == 'low_confidence':
        x0_p = remaskConf(logits, temperature, x0)
    elif remasking == 'random':
        x0_p = rand_like(x0, device=x0.device)
    else:
        raise NotImplementedError(f"Unsupported remasking strategy: {remasking}")
    return x0, x0_p


@torch.compile
def remaskConf(logits:Tensor, temperature:float, x0:Tensor)->Tensor:
    p = F.softmax(logits / temperature if temperature > 0 else logits, dim=-1) # Apply temperature scaling to logits before softmax, or if temperature is <=0, use standard softmax
    return gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)


def generateStep(model, x, cfg_scale, chunk_size, mask_id, prompt_index,
                 memory_integration, temperature, remasking, step,
                 num_transfer_tokens, block_start, block_end, mask_index) -> tuple[Tensor, Tensor, Tensor]:
    """Inner iteration of generate"""

    x0, x0_p = logitSample(cfg_scale, chunk_size, mask_id, mask_index, memory_integration, model, prompt_index,
                           remasking, temperature, x)

    # Restrict confidence to current block
    confidence = where(mask_index, x0_p, tensor(-float('inf'), device=x.device))
    confidence[:, :block_start] = confidence[:, block_end:] = -float('inf')

    # Select and transfer tokens
    transfer_index = select_tokens_to_transfer(confidence, mask_index, num_transfer_tokens, step)
    return x0, transfer_index, confidence # Return confidence


@no_grad()
def generate(
    model,
    prompt: Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = MASK_TOKEN_ID_DEFAULT,
    cpu_offload: bool = True,
    adaptive_steps: bool = True,
    progress_callback: Optional[Callable[[float, Tensor], None]] = None,
    memory_integration=None,
    chunk_size: int = 512,
    device: device = DEVICE
) -> Tensor:
    """
    Generate text using the LLaDA model with optimized memory and step scheduling.

    Args:
        model: Language model
        prompt: Input prompt tensor (shape: [1, seq_len])
        steps: Maximum number of sampling steps (default: 128)
        gen_length: Length of text to generate (default: 128)
        block_length: Size of generation blocks (default: 128)
        temperature: Sampling temperature (default: 0.0)
        cfg_scale: Classifier-free guidance scale (default: 0.0)
        remasking: Remasking strategy ('low_confidence' or 'random', default: 'low_confidence')
        mask_id: Mask token ID (default: MASK_TOKEN_ID_DEFAULT)
        cpu_offload: Enable CPU offloading to save GPU memory (default: True)
        adaptive_steps: Use adaptive step scheduling (default: True)
        progress_callback: Optional callback for progress updates
        memory_integration: Optional memory integration module
        chunk_size: Chunk size for processing long sequences (default: 512)
        device: Device for computation (default: DEVICE)

    Returns:
        Generated token tensor (shape: [1, prompt_len + gen_length])
    """
    model.eval()

    #temperature = f32(temperature, device)

    # Adjust gen_length to be divisible by block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length + block_length // 2) // block_length) * block_length
        logger.warning(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")

    # Initialize sequence and token buffer
    x = full((1, prompt.shape[1] + gen_length), mask_id, dtype=long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    token_buffer = TokenBuffer(x, device=device, cpu_offload=cpu_offload)
    mask_index = (x == mask_id)

    num_blocks = gen_length // block_length

    # Determine steps per block
    if not adaptive_steps:
        steps_per_block = steps // num_blocks
        if steps % num_blocks != 0:
            steps = steps_per_block * num_blocks
            logger.warning(f"Adjusted steps to {steps} to be divisible by num_blocks {num_blocks}")
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
            steps_per_block_list = [max(4, int(s * steps / total_steps)) for s in steps_per_block_list]

    total_steps_expected = steps if not adaptive_steps else sum(steps_per_block_list)
    total_steps_completed = 0
    all_step_confidences: List[Tensor] = [] # List to store step confidences

    for num_block in range(num_blocks):
        steps_per_block = steps_per_block_list[num_block] if adaptive_steps else steps_per_block
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        block_mask_index = mask_index[:, block_start:block_end]

        if not block_mask_index.any():
            continue

        # Compute transfer schedule
        num_transfer_tokens = (
            get_adaptive_transfer_schedule(block_mask_index, steps_per_block)
            if adaptive_steps else
            div(block_mask_index.sum(dim=1, keepdim=True), steps_per_block, rounding_mode='floor').repeat(1, steps_per_block)
        )

        step_confidences_block: List[Tensor] = [] # Store confidences for this block
        for i in range(steps_per_block):
            x = token_buffer.data  # Automatically moves to GPU if needed

            if progress_callback:
                total_steps_completed += 1
                progress_callback(total_steps_completed / total_steps_expected, x.clone())

            if not mask_index.any():
                break

            # Process generation step in helper function
            x0, transfer_index, confidence = generateStep( # Get confidence here
                model, x, cfg_scale, chunk_size, mask_id, prompt_index,
                memory_integration, temperature, remasking, i,
                num_transfer_tokens, block_start, block_end, mask_index
            )
            step_confidences_block.append(confidence.clone().cpu()) # Store confidence for this step

            # Update token buffer and mask
            x[transfer_index] = x0[transfer_index]
            mask_index[transfer_index] = False
            token_buffer.update(x)

            # Early exit if block is fully generated
            if not mask_index[:, block_start:block_end].any():
                break

            # Offload to CPU between steps
            if cpu_offload and i < steps_per_block - 1:
                gpu_offload_to_cpu(token_buffer)

        if cpu_offload:
            gpu_offload_to_cpu(token_buffer)
        all_step_confidences.extend(step_confidences_block) # Collect confidences from each block

    token_buffer.to_gpu()
    # Stack confidences along step dimension after generation
    step_confidences_tensor = stack(all_step_confidences, dim=0) if all_step_confidences else empty(0)
    return token_buffer.data, step_confidences_tensor # Return tokens and confidences




def gpu_offload_to_cpu(token_buffer):
    token_buffer.to_cpu()
    cuda.empty_cache()
