#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration and constants for the LLaDA GUI application.
"""

import os
import sys

# Base paths for the application
APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLADA_REPO_PATH = "/home/ty/Repositories/ai_workspace/llada_gui" # TODO: Remove hardcoded path

# Model information
DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
LOCAL_MODEL_PATH = os.path.join(LLADA_REPO_PATH, "GSAI-ML_LLaDA-8B-Instruct")

# Add paths to Python path - Ensure core and gui are always accessible
sys.path.insert(0, APP_PATH) # Add root path
sys.path.insert(0, os.path.join(APP_PATH, "core")) # Add core path
sys.path.insert(0, os.path.join(APP_PATH, "gui")) # Add gui path


# Default generation parameters - More descriptive names and grouped
DEFAULT_GENERATION_PARAMS = {
    'gen_length': 128,
    'steps': 128,
    'block_length': 32,
    'temperature': 0,
    'cfg_scale': 0,
    'remasking': 'low_confidence',
}

# Memory optimization constants - Grouped memory related constants
MEMORY_OPTIMIZATION_SETTINGS = {
    'optimized_gpu_memory': True,
    'cache_precision': "bfloat16",
    'enable_attention_slicing': True,
    'enable_flash_attention': True,
}

# Memory monitoring constants - Grouped monitoring related constants
MEMORY_MONITORING_SETTINGS = {
    'memory_check_interval': 1.0,
    'memory_warning_threshold': 90,
    'memory_caution_threshold': 75,
    'critical_gpu_memory_threshold': 0.3, # GB - clearer unit
}

# Memory integration constants - Grouped memory integration related constants
MEMORY_INTEGRATION_SETTINGS = {
    'memory_enabled': False,
    'memory_server_port': 3000,
    'memory_data_dir': os.path.join(APP_PATH, "data", "memory"),
}

# UI-related constants - Grouped UI related constants
UI_SETTINGS = {
    'window_title': "LLaDA GUI - Large Language Diffusion Model",
    'window_width': 1100,
    'window_height': 800,
    'splitter_ratio': [250, 550],
}

# Create necessary directories - Ensure memory data dir exists
os.makedirs(MEMORY_INTEGRATION_SETTINGS['memory_data_dir'], exist_ok=True)
