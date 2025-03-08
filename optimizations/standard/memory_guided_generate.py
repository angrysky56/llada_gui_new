#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory-guided generation using the LLaDA diffusion model.

This extends the standard generate.py with memory guidance
to improve generation coherence and continuity.
"""

import logging
from typing import Optional, Callable

import numpy as np
import torch

# Try to import vector database
try:
    from core.memory.vector_db import get_vector_db

    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


def generate(
        model,
        prompt,
        steps=64,
        gen_length=32,
        block_length=32,
        temperature=0.0,
        cfg_scale=1.0,
        remasking="low_confidence",
        progress_callback: Optional[Callable[[float, torch.Tensor], None]] = None,
        memory_callback: Optional[Callable[[np.ndarray], None]] = None,
        memory_interface=None,
        memory_weight=0.3,
        cpu_offload=False,
        mask_id=126336
):
    """
    Generate text using the LLaDA model with memory guidance.
    
    Args:
        model: LLaDA model
        prompt: Input prompt tensor
        steps: Number of diffusion steps
        gen_length: Length of text to generate
        block_length: Block length for parallel generation
        temperature: Temperature for sampling
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ("low_confidence" or "random")
        progress_callback: Optional callback for progress updates
        memory_callback: Optional callback for memory state updates
        memory_interface: Memory interface for guidance
        memory_weight: Weight of memory influence (0-1)
        cpu_offload: Whether to offload tensors to CPU during generation
        mask_id: Token ID for mask token
    
    Returns:
        Generated text tensor
    """
    # Check if memory is available
    memory_available = memory_interface is not None and hasattr(memory_interface,
                                                                'initialized') and memory_interface.initialized

    if memory_available:
        logger.info(f"Memory guidance enabled with weight {memory_weight}")
    else:
        logger.info("Memory guidance not available, proceeding with standard generation")

    # Store the device of the model for later use
    device = next(model.parameters()).device

    # Clone the prompt tensor to avoid modifying the original
    prompt = prompt.clone()

    # Get input length and set total length
    input_len = prompt.shape[1]
    total_len = input_len + gen_length

    # Create the full sequence with mask tokens for the generated part
    full = torch.cat([
        prompt,
        torch.ones(1, gen_length, dtype=torch.long, device=device) * mask_id
    ], dim=1)

    # Latent variables for diffusion process
    time_t = torch.ones(1, device=device)

    # Create a mask for tokens that will be predicted (1 for fixed tokens, 0 for masked tokens)
    mask = torch.cat([
        torch.ones(1, input_len, device=device),  # Prompt tokens are fixed
        torch.zeros(1, gen_length, device=device)  # Generated tokens are masked
    ], dim=1)

    # Initial token remask (used for low confidence remasking)
    token_remask = None

    # Generate in blocks for memory efficiency
    start_idx = input_len
    end_idx = min(input_len + block_length, total_len)
    current_block = 0

    # Memory state variables
    memory_state = None
    memory_dim = 64  # Default memory dimension

    # Determine how many steps to run for each block
    while start_idx < total_len:
        # Get number of tokens in this block
        block_tokens = end_idx - start_idx

        # Run diffusion steps
        for step in range(steps):
            # Calculate progress
            total_progress = (current_block * steps + step) / (
                        ((total_len - input_len - 1) // block_length + 1) * steps)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(total_progress, full)

            # Time step for diffusion (linearly spaced)
            time_t[0] = 1 - step / steps

            # Memory weights - influence decreases over time
            if memory_available:
                # Influence is weighted by both the memory_weight parameter and time
                # Early steps have more memory influence than later steps
                current_memory_weight = memory_weight * (1 - step / steps)

            with torch.no_grad():
                # Forward pass through the model
                logits = model(full, time_t, mask, cfg_guidance_scale=cfg_scale)

                # Apply temperature if needed
                if temperature > 0:
                    logits = logits / temperature

                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Apply memory guidance if available
                if memory_available:
                    try:
                        # Convert current tokens to simplified embeddings for memory
                        current_tokens = full[0, :end_idx].cpu().tolist()

                        # Create a simple embedding based on token frequencies
                        def create_token_embedding(tokens, dim=64):
                            # Create a frequency-based embedding
                            embedding = np.zeros(dim)
                            for i, token in enumerate(tokens):
                                embedding[i % dim] = (token % 100) / 100.0  # Simple hash

                            # Normalize
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                            return embedding

                        # Create an embedding for the current tokens
                        token_embedding = create_token_embedding(current_tokens, memory_interface.input_dim)

                        # Check vector database for similar contexts if available
                        vector_guidance = None
                        if VECTOR_DB_AVAILABLE:
                            try:
                                vector_db = get_vector_db()
                                if vector_db is not None:
                                    # Find similar vectors
                                    similar_indices = vector_db.find_similar(token_embedding, top_k=3, threshold=0.6)

                                    # If we found similar vectors, use them for guidance
                                    if similar_indices:
                                        # Get the vectors and their metadata
                                        similar_vectors = []
                                        for idx in similar_indices:
                                            vec, meta = vector_db.get_vector(idx)
                                            similar_vectors.append((vec, meta))

                                        # Use the most similar vector for guidance
                                        most_similar_vec, meta = similar_vectors[0]

                                        # Print what we found (for debugging)
                                        logger.info(
                                            f"Found similar vector: {meta.get('type', 'unknown')} - {meta.get('text', '')}")

                                        # If this is a prompt, look for its generation
                                        if meta.get('type') == 'prompt':
                                            # Look for generations with this prompt
                                            for idx in range(len(vector_db.vectors)):
                                                vec, gen_meta = vector_db.get_vector(idx)
                                                if gen_meta.get('type') == 'generation' and \
                                                        gen_meta.get('prompt', '') == meta.get('text', ''):
                                                    vector_guidance = vec
                                                    logger.info(
                                                        f"Using guidance from previous generation: {gen_meta.get('text', '')}")
                                                    break
                                        elif meta.get('type') == 'generation':
                                            # Use the generation vector directly
                                            vector_guidance = most_similar_vec
                            except Exception as e:
                                logger.error(f"Error using vector database: {e}")

                        # Get memory prediction
                        memory_result = memory_interface.forward_pass(token_embedding)

                        # Update memory state
                        memory_state = np.array(memory_result["newMemory"])

                        # Call memory callback if provided
                        if memory_callback:
                            memory_callback(memory_state)

                        # Get predicted token probabilities from memory
                        memory_prediction = np.array(memory_result["predicted"])

                        # If we have vector guidance, blend it with the memory prediction
                        if vector_guidance is not None:
                            # Blend memory prediction with vector guidance
                            memory_prediction = memory_prediction * 0.7 + vector_guidance * 0.3

                        # Convert to token probability prediction
                        # Create memory-based token probabilities
                        memory_token_probs = np.zeros(probs.shape[-1])

                        # Map the embedding values to token probabilities
                        # Convert memory prediction to token probability distribution
                        vocab_size = probs.shape[-1]

                        # Project memory prediction onto token space more reliably
                        for i, val in enumerate(memory_prediction):
                            # Map prediction values to token indices with more distribution
                            # This creates a smoother probability distribution centered around the predicted values
                            center_idx = int((val + 1) / 2 * vocab_size // 10) % vocab_size

                            # Add probability mass around the center with a Gaussian-like distribution
                            window_size = min(100, vocab_size // 10)
                            for offset in range(-window_size // 2, window_size // 2 + 1):
                                idx = (center_idx + offset) % vocab_size
                                # Add probability with falloff based on distance from center
                                weight = 0.1 * np.exp(-(offset ** 2) / (2 * (window_size // 4) ** 2))
                                memory_token_probs[idx] += weight

                        # Normalize
                        memory_token_probs_sum = memory_token_probs.sum()
                        if memory_token_probs_sum > 0:
                            memory_token_probs = memory_token_probs / memory_token_probs_sum

                        # Mix with model probabilities
                        for i in range(start_idx, end_idx):
                            if mask[0, i] == 0:  # Only for masked tokens
                                # Mix the probabilities
                                mixed_probs = (1 - current_memory_weight) * probs[0, i].cpu().numpy() + \
                                              current_memory_weight * memory_token_probs

                                # Update probabilities
                                probs[0, i] = torch.tensor(mixed_probs, device=device)

                    except Exception as e:
                        logger.error(f"Error applying memory guidance: {str(e)}")

                # Sample from the probability distribution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Update masked tokens
                for i in range(start_idx, end_idx):
                    if mask[0, i] == 0:  # Only update masked tokens
                        full[0, i] = next_tokens[0, i]

                # Determine which tokens to remask
                if remasking == "low_confidence":
                    # Get confidence scores (probability of selected token)
                    confidence = torch.gather(
                        probs, dim=-1,
                        index=full.unsqueeze(-1).to(device)
                    ).squeeze(-1)

                    # Create new token_remask based on confidence
                    token_remask = torch.zeros_like(mask)

                    # Only consider tokens in the current block
                    for i in range(start_idx, end_idx):
                        if mask[0, i] == 0:  # Only for masked tokens
                            token_remask[0, i] = 1 - confidence[0, i]  # Low confidence = high remask probability
                elif remasking == "random":
                    # Random remasking with decreasing probability over steps
                    mask_prob = (1 - step / steps) * 0.5  # Decrease from 0.5 to 0
                    token_remask = torch.zeros_like(mask)
                    for i in range(start_idx, end_idx):
                        if mask[0, i] == 0:  # Only for masked tokens
                            token_remask[0, i] = mask_prob

                # Dynamically decrease number of masked tokens as generation progresses
                target_mask_count = max(1, int(block_tokens * (1 - step / steps)))

                # Sort tokens by remask probability
                if token_remask is not None:
                    # Get remask values for the current block
                    block_remask = token_remask[0, start_idx:end_idx]

                    # Sort by remask probability
                    sorted_indices = torch.argsort(block_remask, descending=True)

                    # Reset mask for all tokens in block
                    mask[0, start_idx:end_idx] = 1

                    # Mask the top target_mask_count tokens
                    for i in range(min(target_mask_count, len(sorted_indices))):
                        mask[0, start_idx + sorted_indices[i]] = 0

                # Offload tensors to CPU to save memory if requested
                if cpu_offload and device.type == 'cuda':
                    # Move unnecessary tensors to CPU
                    torch.cuda.empty_cache()

        # Move to the next block
        start_idx = end_idx
        end_idx = min(start_idx + block_length, total_len)
        current_block += 1

        # Update progress after completing a block
        if progress_callback:
            progress_callback(current_block / ((total_len - input_len - 1) // block_length + 1), full)

    # Final progress update
    if progress_callback:
        progress_callback(1.0, full)

    # Return the generated sequence
    return full
