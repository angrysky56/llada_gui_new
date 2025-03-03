#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the LLaDA GUI application.
"""

import os
import gc
import torch
import traceback
import logging
from config import LOCAL_MODEL_PATH, DEFAULT_MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device_status():
    """
    Get the current status of available devices.
    
    Returns:
        dict: Information about available devices and memory
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cpu_available': True,
        'device_count': 0,
        'gpu_info': []
    }
    
    # Add CPU info
    import psutil
    memory = psutil.virtual_memory()
    info['cpu_memory_total'] = memory.total / (1024**3)  # GB
    info['cpu_memory_used'] = memory.used / (1024**3)    # GB
    info['cpu_memory_percent'] = memory.percent
    
    # Add GPU info if available
    if info['cuda_available']:
        info['device_count'] = torch.cuda.device_count()
        
        for i in range(info['device_count']):
            device_info = {
                'name': torch.cuda.get_device_name(i),
                'index': i,
                'total_memory': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                'used_memory': (torch.cuda.memory_allocated(i) + torch.cuda.memory_reserved(i)) / (1024**3)  # GB
            }
            device_info['free_memory'] = device_info['total_memory'] - device_info['used_memory']
            device_info['used_percent'] = (device_info['used_memory'] / device_info['total_memory']) * 100
            
            info['gpu_info'].append(device_info)
    
    return info

def cleanup_gpu_memory():
    """
    Clean up GPU memory by clearing CUDA cache and running garbage collection.
    """
    logger.info("Cleaning up GPU memory")
    
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        try:
            # Set environment variables for better memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Trigger another garbage collection after emptying cache
            gc.collect()
            
            # Get current memory usage for logging
            used_memory = torch.cuda.memory_allocated() / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory usage after cleanup: {used_memory:.2f}GB / {total_memory:.2f}GB")
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")

def optimize_model_memory(model):
    """
    Apply memory optimizations to a loaded model.
    
    Args:
        model: The model to optimize
        
    Returns:
        The optimized model
    """
    if not torch.cuda.is_available():
        return model
    
    logger.info("Applying model memory optimizations")
    
    try:
        # Disable caching where possible
        if hasattr(model, "config"):
            model.config.use_cache = False
        
        # Apply attention slicing if available
        if hasattr(model, "enable_attention_slicing"):
            logger.info("Enabling attention slicing")
            model.enable_attention_slicing(1)
        
        # Apply specific optimizations for the LLaDA model
        model = _apply_llada_optimizations(model)
        
        # Clear memory again
        cleanup_gpu_memory()
        
        return model
    except Exception as e:
        logger.warning(f"Error during model memory optimization: {e}")
        return model

def _apply_llada_optimizations(model):
    """
    Apply LLaDA-specific memory optimizations.
    
    Args:
        model: The LLaDA model
        
    Returns:
        The optimized model
    """
    try:
        # Some potential LLaDA-specific optimizations could go here
        # For example, removing unnecessary model components
        
        return model
    except Exception as e:
        logger.warning(f"Error applying LLaDA-specific optimizations: {e}")
        return model

def get_model_path():
    """
    Get the path to the model files.
    
    Returns:
        str: Path to the model files
    """
    # Check if model is in the new repository
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    else:
        # Fall back to the original path
        return DEFAULT_MODEL_PATH

def format_error(exception):
    """
    Format an exception for display.
    
    Args:
        exception: The exception to format
        
    Returns:
        str: Formatted error message with traceback
    """
    tb = traceback.format_exc()
    error_msg = f"Error: {str(exception)}\n\n{tb}"
    return error_msg

def format_memory_info(stats):
    """
    Format memory statistics for display.
    
    Args:
        stats (dict): Memory statistics
        
    Returns:
        tuple: (system_memory_text, gpu_memory_text)
    """
    # Format system memory
    system_text = f"{stats['system_used']:.2f} / {stats['system_total']:.2f} GB ({stats['system_percent']:.1f}%)"
    
    # Format GPU memory if available
    if stats.get('gpu_available', False):
        gpu_text = f"{stats['gpu_used']:.2f} / {stats['gpu_total']:.2f} GB ({stats['gpu_percent']:.1f}%)"
    else:
        gpu_text = "Not available"
    
    return system_text, gpu_text

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
