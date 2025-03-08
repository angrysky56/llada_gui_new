#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory leak patches for LLaDA GUI.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def apply_patches():
    """Apply all memory leak patches."""
    logger.info("Applying memory leak patches")

    # Patch torch cuda memory management
    patch_cuda_memory()

    # Patch attention implementation if transformers is available
    try:
        patch_attention()
    except Exception as e:
        logger.warning(f"Failed to patch attention: {e}")

    logger.info("Memory leak patches applied")


def patch_cuda_memory():
    """Patch CUDA memory management."""
    # Only apply if CUDA is available
    if not torch.cuda.is_available():
        return

    # Run garbage collection more aggressively
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Enable more aggressive memory management
    if hasattr(torch.cuda, "memory_stats"):
        # Log current memory stats
        try:
            memory_stats = torch.cuda.memory_stats()
            logger.debug(f"CUDA memory stats: {memory_stats}")
        except Exception:
            pass


def patch_attention():
    """Patch attention implementation to be more memory efficient."""
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention

        # Store original forward method
        original_forward = LlamaAttention.forward

        # Define patched forward method
        def patched_forward(self, *args, **kwargs):
            # Call original forward
            output = original_forward(self, *args, **kwargs)

            # Aggressively delete intermediate tensors
            if hasattr(self, 'k_cache'):
                del self.k_cache
            if hasattr(self, 'v_cache'):
                del self.v_cache

            # Clear CUDA cache more aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output

        # Apply the patch
        LlamaAttention.forward = patched_forward

        logger.info("Patched LlamaAttention.forward for memory efficiency")
    except ImportError:
        logger.warning("Could not import LlamaAttention, skipping attention patch")
    except Exception as e:
        logger.warning(f"Failed to patch attention: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Apply patches
    apply_patches()
