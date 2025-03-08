#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extreme memory optimizations for the LLaDA model.
These optimizations are suitable for GPUs with limited VRAM (8-12GB).
"""

import gc
import logging
import os

import torch

logger = logging.getLogger(__name__)


def apply_optimizations():
    """Apply extreme memory optimizations to the LLaDA model."""
    logger.info("Applying extreme memory optimizations for GPUs with limited VRAM")

    # First apply standard optimizations
    from optimizations.standard.optimize import apply_optimizations as apply_standard_optimizations
    apply_standard_optimizations()

    # Set environment variables for extreme memory optimizations
    if torch.cuda.is_available():
        # Enable aggressive memory optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

        # Force 4-bit quantization
        os.environ["USE_4BIT_QUANTIZATION"] = "1"

        # Enable gradient checkpointing
        os.environ["USE_GRADIENT_CHECKPOINTING"] = "1"

        # Set very aggressive attention slicing
        os.environ["ATTENTION_SLICE_SIZE"] = "4"

        # Disable some features to save memory
        os.environ["DISABLE_TELEMETRY"] = "1"
        os.environ["DISABLE_ERROR_REPORTING"] = "1"

        # Clean up Python memory
        gc.collect()

        # Force CUDA to compact memory
        try:
            torch.cuda.empty_cache()
            torch.cuda.memory_stats()  # Force memory compaction
        except:
            pass

        # Log available GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory after optimization: {gpu_memory:.2f}GB")
        except:
            pass

    # Apply transformer-specific extreme optimizations
    try:
        from config_extreme import EXTREME_CONFIG
        for key, value in EXTREME_CONFIG.items():
            os.environ[key] = str(value)
            logger.debug(f"Set {key}={value}")
    except ImportError:
        logger.warning("Extreme configuration not found, using defaults")

    logger.info("Extreme memory optimizations applied")
    return True


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Apply optimizations
    apply_optimizations()
