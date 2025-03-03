#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory leak patches for LLaDA GUI.
"""

import torch
import gc
import logging

logger = logging.getLogger(__name__)

def apply_patches():
    """Apply all memory leak patches."""
    logger.info("Applying memory leak patches")
    
    # Set PyTorch memory management options
    optimize_memory_settings()
    
    # Patch attention implementation if transformers is available
    try:
        patch_attention()
    except Exception as e:
        logger.warning(f"Failed to patch attention: {e}")
    
    logger.info("Memory leak patches applied")

def optimize_memory_settings():
    """Apply memory optimization settings."""
    # Only apply if CUDA is available
    if not torch.cuda.is_available():
        return
    
    try:
        # Set environment variables for optimized memory usage
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        
        # Disable benchmark mode for more consistent memory usage
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Lower precision for matmul operations
        torch.set_float32_matmul_precision('medium')
        
        # Force aggressive garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Create a hook to clear memory after operations
        def gpu_memory_hook(module, input, output):
            # Run garbage collection
            gc.collect()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            return output
        
        logger.info("Memory optimization settings applied")
    except Exception as e:
        logger.warning(f"Failed to apply memory optimization settings: {e}")

def patch_attention():
    """Patch attention implementation to be more memory efficient."""
    from transformers.models.llama.modeling_llama import LlamaAttention
    
    # Store original forward method
    original_forward = LlamaAttention.forward
    
    # Define patched forward method
    def patched_forward(self, *args, **kwargs):
        # Call original forward
        output = original_forward(self, *args, **kwargs)
        
        # Aggressively delete intermediate tensors
        if hasattr(self, 'k_cache'):
            del self.k_cache
        if hasattr(self, 'v_cache'):
            del self.v_cache
            
        # Clear CUDA cache more aggressively
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return output
    
    # Apply the patch
    LlamaAttention.forward = patched_forward
    
    logger.info("Attention implementation patched")

def patch_tensor_operations():
    """
    Patch tensor operations to be more memory efficient.
    """
    # Patch tensor operations that use a lot of memory
    original_matmul = torch.matmul
    
    def memory_efficient_matmul(input, other, *, out=None):
        # Call original matmul
        result = original_matmul(input, other, out=out)
        
        # Clear inputs if they're large
        if input.numel() > 1000000 and not input.requires_grad:
            input.detach_()
        if other.numel() > 1000000 and not other.requires_grad:
            other.detach_()
            
        # Free memory if under pressure
        if torch.cuda.is_available() and torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.95:
            torch.cuda.empty_cache()
            
        return result
    
    # Apply the patch
    torch.matmul = memory_efficient_matmul
    
    logger.info("Tensor operations patched successfully")
