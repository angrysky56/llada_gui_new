#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply extreme memory optimizations for LLaDA GUI.
This module provides a simplified interface to the extreme optimizer.
"""

import logging
import os
import sys

import torch

# Set up logging
logger = logging.getLogger(__name__)


def apply_optimizations():
    """Apply extreme memory optimizations for LLaDA GUI."""
    logger.info("Applying extreme memory optimizations for LLaDA GUI")

    try:
        # Import the necessary modules
        from optimizations.extreme.memory_patches import apply_patches

        # Set environment variables for memory efficiency
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        os.environ["OMP_NUM_THREADS"] = "1"  # Limit CPU threads

        # Apply PyTorch memory optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # Lower default precision
            torch.set_float32_matmul_precision('medium')  # Less precise but faster

            # Clear GPU memory
            torch.cuda.empty_cache()

            # Log GPU info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory: {gpu_memory:.2f} GB")

        # Apply memory leak patches
        apply_patches()

        logger.info("Extreme memory optimizations applied successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to apply extreme memory optimizations: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Apply optimizations
    result = apply_optimizations()
    sys.exit(0 if result else 1)
