#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration of performance optimizations with LLaDA GUI.

This module provides functions to integrate the performance optimizations
with the main LLaDA GUI components.
"""

import logging
import os
import sys

from .enhanced_worker import enhance_llada_worker
# Import performance components
from .model_cache import get_model_cache

# Configure logging
logger = logging.getLogger(__name__)


def integrate_performance_optimizations():
    """
    Integrate performance optimizations with the LLaDA GUI.
    
    This function replaces the standard LLaDA worker with an enhanced version
    that incorporates model caching and streaming output.
    
    Returns:
        bool: True if integration was successful, False otherwise
    """
    try:
        logger.info("Integrating performance optimizations")

        # Initialize the model cache
        model_cache = get_model_cache()

        # Enhance the LLaDA worker
        from core.llada_worker import LLaDAWorker

        # Replace the original worker class with enhanced version
        enhanced_worker = enhance_llada_worker(LLaDAWorker)
        sys.modules['core.llada_worker'].LLaDAWorker = enhanced_worker

        logger.info("LLaDA worker enhanced with performance optimizations")

        # Register API functions
        register_performance_api()

        return True

    except Exception as e:
        logger.error(f"Error integrating performance optimizations: {e}")
        return False


def register_performance_api():
    """
    Register performance optimization functions in the system API.
    
    Returns:
        bool: True if registration was successful, False otherwise
    """
    try:
        # Add to core utils
        import core.utils as utils

        # Register core functions
        utils.get_model_cache = get_model_cache
        utils.enable_streaming = enable_streaming

        logger.info("Registered performance optimization API")
        return True

    except Exception as e:
        logger.error(f"Error registering performance API: {e}")
        return False


def enable_streaming(gui_instance):
    """
    Enable streaming output in the GUI.
    
    Args:
        gui_instance: The LLaDA GUI instance
        
    Returns:
        bool: True if enabled successfully, False otherwise
    """
    try:
        # Create update function to handle partial results
        def update_output(partial_output):
            if hasattr(gui_instance, 'output_text'):
                gui_instance.output_text.setText(partial_output)

        # Keep reference to update function
        if not hasattr(gui_instance, '_streaming_outputs'):
            gui_instance._streaming_outputs = []

        gui_instance._streaming_outputs.append(update_output)

        logger.info("Streaming output enabled")
        return True

    except Exception as e:
        logger.error(f"Error enabling streaming output: {e}")
        return False


def get_performance_metrics(worker):
    """
    Get performance metrics from a worker.
    
    Args:
        worker: LLaDA worker instance
        
    Returns:
        dict: Performance metrics
    """
    metrics = {
        'cached_model': getattr(worker, 'cached_model', False),
        'load_time': getattr(worker, 'load_time', 0),
        'total_time': 0
    }

    return metrics


def optimize_generation_speed():
    """
    Apply system-wide optimizations for generation speed.
    
    Returns:
        bool: True if optimizations were applied, False otherwise
    """
    try:
        # Set PyTorch settings for speed
        if hasattr(torch, 'set_grad_enabled'):
            torch.set_grad_enabled(False)

        # Set environment variables for better PyTorch performance
        os.environ["OMP_NUM_THREADS"] = str(min(os.cpu_count(), 8))
        os.environ["MKL_NUM_THREADS"] = str(min(os.cpu_count(), 8))

        logger.info("Applied generation speed optimizations")
        return True

    except Exception as e:
        logger.error(f"Error applying speed optimizations: {e}")
        return False
