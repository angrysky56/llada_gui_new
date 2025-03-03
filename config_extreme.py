#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extreme memory optimization settings for LLaDA GUI.
"""

# Extreme memory optimization constants
OPTIMIZED_GPU_MEMORY = True
CACHE_PRECISION = "float16"  # Use float16 for extreme memory savings
ENABLE_ATTENTION_SLICING = True  # Enable attention slicing with size 1
ENABLE_FLASH_ATTENTION = True  # Use flash attention if available
DISABLE_KV_CACHE = True  # Disable KV cache completely for memory savings
ENABLE_CPU_OFFLOAD = True  # Enable CPU offloading for memory-intensive operations
BLOCK_SIZE = 16  # Smaller blocks for less memory usage
ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Use memory-efficient attention
ENABLE_GRADIENT_CHECKPOINTING = True  # Use gradient checkpointing even for inference

# Advanced settings
ADVANCED_SETTINGS = {
    # PyTorch memory settings
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
    
    # Model loading
    "PROGRESSIVE_LOADING": True,
    "PROGRESSIVE_LOADING_BLOCK_SIZE": 2,
    "QUANTIZATION_BITS": 4,  # Using 4-bit quantization for maximum memory savings
    
    # Attention optimization
    "ATTENTION_SLICE_SIZE": 1,
    "ATTENTION_HEAD_PRUNE_PERCENT": 25,  # Prune 25% of attention heads
    
    # Layer optimization
    "FFN_PRUNE_PERCENT": 20,  # Prune 20% of FFN neurons
    
    # Generation parameters
    "DEFAULT_GEN_LENGTH": 32,
    "DEFAULT_STEPS": 32,
    "DEFAULT_BLOCK_LENGTH": 16,
    
    # Memory monitoring
    "MEMORY_CHECK_INTERVAL": 0.5,  # Check memory more frequently
    "MEMORY_WARNING_THRESHOLD": 85,  # Lower warning threshold
    "MEMORY_CAUTION_THRESHOLD": 70,  # Lower caution threshold
    "CRITICAL_GPU_MEMORY_THRESHOLD": 0.2,  # Lower critical threshold
}

# Apply these settings at import time
import os
for key, value in ADVANCED_SETTINGS.items():
    if key.startswith("PYTORCH_") or key.startswith("CUDA_"):
        os.environ[key] = str(value)

# Apply PyTorch memory optimizations
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')  # Less precise but faster

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llada_extreme")
logger.info("Loaded extreme memory optimization settings")
