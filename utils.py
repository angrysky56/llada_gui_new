#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the LLaDA GUI application.
"""

import os
import gc
import torch
import traceback
from config import LOCAL_MODEL_PATH, DEFAULT_MODEL_PATH

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

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
