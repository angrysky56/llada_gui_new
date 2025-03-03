#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration and constants for the LLaDA GUI application.
"""

import os
import sys

# Base paths for the application
APP_PATH = os.path.dirname(os.path.abspath(__file__))
LLADA_REPO_PATH = "/home/ty/Repositories/ai_workspace/llada_gui"

# Model information
DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
LOCAL_MODEL_PATH = os.path.join(LLADA_REPO_PATH, "GSAI-ML_LLaDA-8B-Instruct")

# Add LLaDA repository to Python path
if os.path.exists(LLADA_REPO_PATH):
    sys.path.append(LLADA_REPO_PATH)
else:
    # Fall back to original path if the model hasn't been moved yet
    sys.path.append("/home/ty/Repositories/LLaDA")

# Default generation parameters
DEFAULT_PARAMS = {
    # Optimized parameters for better performance and memory usage
    'gen_length': 64,
    'steps': 64,
    'block_length': 32,
    'temperature': 0.0,
    'cfg_scale': 0.0,
    'remasking': 'low_confidence',
}

# Memory-related constants
# Memory optimization constants
OPTIMIZED_GPU_MEMORY = True
CACHE_PRECISION = "bfloat16"  # Use bfloat16 for better performance with minimal precision loss
ENABLE_ATTENTION_SLICING = True  # Slice attention for lower memory usage
ENABLE_FLASH_ATTENTION = True  # Use flash attention if available

MEMORY_CHECK_INTERVAL = 1.0  # seconds
MEMORY_WARNING_THRESHOLD = 90  # percentage
MEMORY_CAUTION_THRESHOLD = 75  # percentage
CRITICAL_GPU_MEMORY_THRESHOLD = 0.3  # GB

# UI-related constants
WINDOW_TITLE = "LLaDA GUI - Large Language Diffusion Model"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
SPLITTER_RATIO = [300, 600]  # Relative sizes of the top and bottom sections
