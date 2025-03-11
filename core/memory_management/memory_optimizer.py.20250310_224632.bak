#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory optimization utilities for LLaDA model.

This module provides specific optimizations for the LLaDA model to reduce memory usage
and prevent out-of-memory errors during initialization and generation.
"""

import gc
import logging
import os

import torch

# Configure logging
logger = logging.getLogger(__name__)


def apply_model_optimizations(model, config):
    """
    Apply memory optimizations to a loaded model.
    
    Args:
        model: The model to optimize
        config: Generation configuration with optimization settings
        
    Returns:
        The optimized model
    """
    logger.info("Applying model memory optimizations")

    # Apply quantization if requested
    model = apply_quantization(model, config)

    # Apply additional optimizations
    model = apply_additional_optimizations(model, config)

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def apply_quantization(model, config):
    """
    Apply quantization to the model based on configuration.
    
    Args:
        model: The model to quantize
        config: Generation configuration with quantization settings
        
    Returns:
        The quantized model
    """
    # Check if quantization is requested
    if not config.get('use_4bit', False) and not config.get('use_8bit', False):
        logger.info("No quantization requested, using full precision")
        return model

    # Check if CUDA is available (quantization only works on CUDA)
    if not torch.cuda.is_available() or config.get('device', 'cuda') != 'cuda':
        logger.warning("Quantization requested but not using CUDA, skipping")
        return model

    try:
        # Import quantization library (bitsandbytes)
        import bitsandbytes as bnb

        # Check for 4-bit quantization
        if config.get('use_4bit', False):
            logger.info("Applying 4-bit quantization")

            # Apply 4-bit quantization
            # This is a placeholder - actual implementation depends on the model architecture
            # For LLaMA-like models, this would convert linear layers to 4-bit
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
                    # Skip specific layers if needed
                    if not any(skip in name for skip in ['output', 'gate']):
                        # Replace with 4-bit quantized version
                        model._modules[name] = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=torch.bfloat16
                        )

            # Convert to bfloat16 for compatible operations with 4-bit
            model = model.bfloat16()

        # Check for 8-bit quantization (if 4-bit not enabled)
        elif config.get('use_8bit', False):
            logger.info("Applying 8-bit quantization")

            # Apply 8-bit quantization
            # This is a placeholder - actual implementation depends on the model architecture
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
                    # Skip specific layers if needed
                    if not any(skip in name for skip in ['output', 'gate']):
                        # Replace with 8-bit quantized version
                        model._modules[name] = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None
                        )

        logger.info("Quantization applied successfully")

    except ImportError:
        logger.warning("bitsandbytes library not available, skipping quantization")
    except Exception as e:
        logger.error(f"Error applying quantization: {e}")

    return model


def apply_additional_optimizations(model, config):
    """
    Apply additional memory optimizations beyond quantization.
    
    Args:
        model: The model to optimize
        config: Generation configuration
        
    Returns:
        The optimized model
    """
    # Enable gradient checkpointing if available
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    except Exception as e:
        logger.warning(f"Error enabling gradient checkpointing: {e}")

    # Apply attention slicing if available
    try:
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing(1)
            logger.info("Attention slicing enabled")
    except Exception as e:
        logger.warning(f"Error enabling attention slicing: {e}")

    # Apply flash attention if available and requested
    try:
        if config.get('use_flash_attention', True) and hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
            logger.info("Flash attention enabled")
    except Exception as e:
        logger.warning(f"Error enabling flash attention: {e}")

    # Disable caching if not needed
    if hasattr(model, "config"):
        model.config.use_cache = False
        logger.info("Model caching disabled")

    # Extreme mode optimizations (for GPUs with very limited memory)
    if config.get('extreme_mode', False):
        logger.info("Applying extreme memory optimizations")
        model = apply_extreme_optimizations(model, config)

    return model


def apply_extreme_optimizations(model, config):
    """
    Apply extreme memory optimizations for GPUs with very limited memory.
    
    Args:
        model: The model to optimize
        config: Generation configuration
        
    Returns:
        The optimized model
    """
    # Force CPU offloading for certain model components
    # This is a placeholder - actual implementation depends on model architecture

    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    # Offload embedding layer to CPU if very low memory
    try:
        if hasattr(model, "get_input_embeddings"):
            embedding = model.get_input_embeddings()
            embedding = embedding.cpu()

            # Create a wrapper that moves tensors to the right device
            class DeviceAwareEmbedding(torch.nn.Module):
                def __init__(self, embedding, device):
                    super().__init__()
                    self.embedding = embedding
                    self.device = device

                def forward(self, input_ids):
                    # Move input to CPU, get embeddings, then move back to GPU
                    input_cpu = input_ids.cpu()
                    embeddings = self.embedding(input_cpu)
                    return embeddings.to(self.device)

            # Replace with wrapped version
            model.set_input_embeddings(DeviceAwareEmbedding(embedding, model.device))
            logger.info("Embedded layer offloaded to CPU to save GPU memory")
    except Exception as e:
        logger.warning(f"Error offloading embedding layer: {e}")

    # Use smaller attention blocks
    try:
        # This is highly model-specific and would need to be implemented
        # based on the LLaDA architecture
        pass
    except Exception as e:
        logger.warning(f"Error applying attention block optimization: {e}")

    return model


def dynamic_memory_management(generation_func):
    """
    Decorator to apply dynamic memory management during generation.
    
    Args:
        generation_func: The generation function to wrap
        
    Returns:
        Wrapped function with dynamic memory management
    """

    def wrapper(self, *args, **kwargs):
        logger.info("Applying dynamic memory management for generation")

        # Force garbage collection before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory before generation
        if torch.cuda.is_available():
            before_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info(f"GPU memory before generation: {before_memory:.2f}GB")

        try:
            # Setup CUDA allocation logging if enabled
            should_log_cuda = os.environ.get("LLADA_LOG_CUDA_ALLOC", "0") == "1"
            if should_log_cuda and torch.cuda.is_available():
                torch.cuda.memory._record_memory_history(max_entries=10000)

            # Run the generation function
            result = generation_func(self, *args, **kwargs)

            # Log memory after generation
            if torch.cuda.is_available():
                after_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"GPU memory after generation: {after_memory:.2f}GB")

            # Clean up memory after generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        except torch.cuda.OutOfMemoryError as e:
            # Handle out-of-memory error
            logger.error(f"CUDA out of memory during generation: {e}")

            # Force clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Attempt to provide guidance on what went wrong
            if hasattr(self, 'config'):
                memory_guidance = "Consider:\n"
                memory_guidance += "1. Reducing generation length\n"
                memory_guidance += "2. Reducing sampling steps\n"
                memory_guidance += "3. Using 4-bit quantization\n"
                memory_guidance += "4. Enabling extreme memory mode\n"
                memory_guidance += "5. Switching to CPU mode (slower but more reliable)"

                error_msg = f"CUDA out of memory error. {memory_guidance}"
            else:
                error_msg = f"CUDA out of memory error: {str(e)}"

            # Re-raise with more helpful message
            raise torch.cuda.OutOfMemoryError(error_msg) from e

    return wrapper


def optimize_inference_params(model, prompt, params):
    """
    Optimize inference parameters based on model, prompt, and available memory.
    
    Args:
        model: The model being used
        prompt: The input prompt
        params: Current generation parameters
        
    Returns:
        dict: Optimized parameters
    """
    # If not using GPU, no optimization needed
    if not torch.cuda.is_available() or params.get('device', 'cuda') != 'cuda':
        return params

    # Clone the parameters
    optimized = dict(params)

    # Get available GPU memory
    try:
        device_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        used_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        available_memory = total_memory - used_memory

        logger.info(f"Available GPU memory for inference: {available_memory:.2f}GB")

        # Adjust parameters based on available memory with progressive scaling
        # The more memory available, the higher quality settings we can use

        # Start with the most memory-critical parameter: generation length
        if available_memory < 2:
            # Extreme memory constraint
            optimized['gen_length'] = min(optimized.get('gen_length', 64), 32)
            optimized['steps'] = min(optimized.get('steps', 64), 32)
            optimized['use_4bit'] = True
            optimized['extreme_mode'] = True

        elif available_memory < 4:
            # Very limited memory
            optimized['gen_length'] = min(optimized.get('gen_length', 64), 48)
            optimized['steps'] = min(optimized.get('steps', 64), 48)

            # Ensure quantization is enabled
            if not optimized.get('use_4bit', False) and not optimized.get('use_8bit', False):
                optimized['use_4bit'] = True

        elif available_memory < 8:
            # Limited memory
            gen_length = optimized.get('gen_length', 64)
            if gen_length > 128:
                optimized['gen_length'] = 128

            # Ensure at least 8-bit quantization for larger generations
            if gen_length > 64 and not optimized.get('use_4bit', False) and not optimized.get('use_8bit', False):
                optimized['use_8bit'] = True

    except Exception as e:
        logger.warning(f"Error optimizing inference parameters: {e}")

    return optimized


def check_memory_usage(func):
    """
    Decorator to check memory usage before and after a function call.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with memory usage logging
    """

    def wrapper(*args, **kwargs):
        # Get memory usage before
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            before_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

            logger.info(f"Memory before {func.__name__}: "
                        f"Allocated: {before_allocated:.2f}GB, "
                        f"Reserved: {before_reserved:.2f}GB")

        # Call the function
        result = func(*args, **kwargs)

        # Get memory usage after
        if torch.cuda.is_available():
            after_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            after_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

            logger.info(f"Memory after {func.__name__}: "
                        f"Allocated: {after_allocated:.2f}GB, "
                        f"Reserved: {after_reserved:.2f}GB, "
                        f"Difference: {after_allocated - before_allocated:.2f}GB")

        return result

    return wrapper


class CudaMemoryTracker:
    """
    Context manager to track CUDA memory usage during a block of code.
    
    Example:
        with CudaMemoryTracker("Model initialization"):
            model = LLaDAModel.from_pretrained(...)
    """

    def __init__(self, operation_name="Operation"):
        """
        Initialize the memory tracker.
        
        Args:
            operation_name: Name of the operation being tracked
        """
        self.operation_name = operation_name
        self.start_allocated = 0
        self.start_reserved = 0

    def __enter__(self):
        """Enter the context manager."""
        # Record starting memory
        if torch.cuda.is_available():
            self.start_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            self.start_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

            logger.info(f"Starting {self.operation_name}: "
                        f"Allocated: {self.start_allocated:.2f}GB, "
                        f"Reserved: {self.start_reserved:.2f}GB")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Record ending memory
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            end_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

            logger.info(f"Completed {self.operation_name}: "
                        f"Allocated: {end_allocated:.2f}GB, "
                        f"Reserved: {end_reserved:.2f}GB, "
                        f"Difference: {end_allocated - self.start_allocated:.2f}GB")

            # If there was a significant increase, log a warning
            if end_allocated - self.start_allocated > 1.0:
                logger.warning(f"Large memory increase during {self.operation_name}: "
                               f"{end_allocated - self.start_allocated:.2f}GB")


def optimize_batch_size(batch_size, available_memory_gb, model_size_gb):
    """
    Optimize batch size based on available memory.
    
    Args:
        batch_size: Initial batch size
        available_memory_gb: Available GPU memory in GB
        model_size_gb: Approximate model size in GB
        
    Returns:
        int: Optimized batch size
    """
    # Calculate memory per batch item (rough estimate)
    memory_per_item = model_size_gb / batch_size

    # Calculate maximum batch size based on available memory with 20% buffer
    max_batch_size = int(available_memory_gb * 0.8 / memory_per_item)

    # Ensure batch size is at least 1
    optimized_batch_size = max(1, min(batch_size, max_batch_size))

    if optimized_batch_size < batch_size:
        logger.warning(f"Reduced batch size from {batch_size} to {optimized_batch_size} "
                       f"due to memory constraints")

    return optimized_batch_size
