#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Low memory configuration options for the LLaDA GUI.
These options are used when running on systems with limited GPU memory.
"""

# Ultra low memory mode
ULTRA_LOW_MEMORY = True

# Use 4-bit quantization instead of 8-bit
USE_4BIT = True

# Enable CPU offloading for some model layers
USE_CPU_OFFLOAD = True

# Device map for CPU/GPU hybrid operation
CPU_OFFLOAD_DEVICE_MAP = {
    # These layers are more memory-intensive and can be offloaded to CPU
    "lm_head": "cpu",
    "model.embed_tokens": "cpu",
    "model.norm": "cpu",
    "transformer.word_embeddings": "cpu",
    # Keep these on GPU for performance
    # "layers": "cuda:0"  # This will be auto-distributed
}

# Reduce cache size to save memory
REDUCED_CACHE_SIZE = True
MAX_CACHE_SIZE = 100  # Maximum number of past key/value pairs to store
