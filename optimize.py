#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory and performance optimization for LLaDA GUI.

This script provides options to optimize the LLaDA model for better performance
and reduced memory usage without requiring ONNX conversion.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llada_optimize")

def optimize_gpu_memory():
    """
    Apply GPU memory optimizations for PyTorch.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU memory optimizations")
        return False
    
    logger.info("Applying GPU memory optimizations")
    
    # Enable memory efficient attention
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Clear existing GPU cache
    torch.cuda.empty_cache()
    
    # Set PyTorch to release memory more aggressively
    try:
        # Enable gradient checkpointing for GPU memory savings
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        
        # Set to release memory when no longer needed
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        
        logger.info("GPU memory optimizations applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error applying GPU memory optimizations: {e}")
        return False

def patch_config_file():
    """
    Update the config.py file with memory optimizations.
    """
    try:
        config_path = Path("config.py")
        if not config_path.exists():
            logger.error(f"Config file not found at {config_path}")
            return False
        
        logger.info(f"Patching config file at {config_path}")
        
        # Read current content
        content = config_path.read_text()
        
        # Define optimized constants
        optimized_constants = """
# Memory optimization constants
OPTIMIZED_GPU_MEMORY = True
CACHE_PRECISION = "bfloat16"  # Use bfloat16 for better performance with minimal precision loss
ENABLE_ATTENTION_SLICING = True  # Slice attention for lower memory usage
ENABLE_FLASH_ATTENTION = True  # Use flash attention if available
"""
        
        # Check if our constants are already in the file
        if "OPTIMIZED_GPU_MEMORY" in content:
            logger.info("Config file already patched, skipping")
            return True
        
        # Add our optimizations after the memory-related constants
        if "MEMORY_CHECK_INTERVAL" in content:
            new_content = content.replace(
                "# Memory-related constants",
                "# Memory-related constants" + optimized_constants
            )
        else:
            # If we can't find the right section, just append to the end
            new_content = content + "\n" + optimized_constants
        
        # Write back the modified content
        config_path.write_text(new_content)
        
        logger.info("Config file patched successfully")
        return True
    except Exception as e:
        logger.error(f"Error patching config file: {e}")
        return False

def patch_worker_file():
    """
    Update the llada_worker.py file with performance optimizations.
    """
    try:
        worker_path = Path("llada_worker.py")
        if not worker_path.exists():
            logger.error(f"Worker file not found at {worker_path}")
            return False
        
        logger.info(f"Patching worker file at {worker_path}")
        
        # Read current content
        content = worker_path.read_text()
        
        # Define imports to add
        import_patch = """import os
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import CRITICAL_GPU_MEMORY_THRESHOLD
from utils import cleanup_gpu_memory, get_model_path, format_error

# Import memory optimization constants if available
try:
    from config import OPTIMIZED_GPU_MEMORY, CACHE_PRECISION, ENABLE_ATTENTION_SLICING, ENABLE_FLASH_ATTENTION
except ImportError:
    OPTIMIZED_GPU_MEMORY = False
    CACHE_PRECISION = None
    ENABLE_ATTENTION_SLICING = False
    ENABLE_FLASH_ATTENTION = False
"""
        
        # Model loading patch to add memory optimizations
        model_load_patch = """            # Apply memory optimizations if enabled
            if device == 'cuda' and OPTIMIZED_GPU_MEMORY:
                self.progress.emit(16, "Applying memory optimizations...", {})
                
                # Set environment variables for optimized memory usage
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                
                # Use lower precision for better performance
                if CACHE_PRECISION == "bfloat16":
                    load_params["torch_dtype"] = torch.bfloat16
                elif CACHE_PRECISION == "float16":
                    load_params["torch_dtype"] = torch.float16
                
                # Enable attention slicing for lower memory usage
                if ENABLE_ATTENTION_SLICING:
                    # Will be applied after model loading
                    attention_slicing = True
                else:
                    attention_slicing = False
                
                # Enable flash attention if available
                if ENABLE_FLASH_ATTENTION:
                    try:
                        load_params["attn_implementation"] = "flash_attention_2"
                    except:
                        pass
            else:
                attention_slicing = False
"""
        
        # Post-load patch to apply attention slicing
        post_load_patch = """            # Apply attention slicing if enabled
            if device == 'cuda' and attention_slicing:
                try:
                    model.config.use_cache = False  # Disable KV cache for more memory efficiency
                    # Apply attention slicing with a slice size of 1
                    # This splits attention operations to reduce peak memory usage
                    model._apply_model_parallel = None  # Disable model parallelism
                    if hasattr(model, "enable_attention_slicing"):
                        model.enable_attention_slicing(1)
                except Exception as attn_error:
                    self.progress.emit(22, f"Warning: Could not apply attention slicing: {str(attn_error)}", {})
"""
        
        # Check if our patches are already in the file
        if "OPTIMIZED_GPU_MEMORY" in content:
            logger.info("Worker file already patched, skipping")
            return True
        
        # Apply patches
        # 1. Replace imports
        if "import os" in content:
            new_content = content.replace(
                "import os",
                import_patch
            )
        else:
            new_content = import_patch + content
        
        # 2. Add model loading optimizations
        if "load_params[\"torch_dtype\"] = torch.float32" in new_content:
            new_content = new_content.replace(
                "load_params[\"torch_dtype\"] = torch.float32",
                "load_params[\"torch_dtype\"] = torch.float32\n            " + model_load_patch
            )
        else:
            logger.warning("Could not find exact location for model loading patch, skipping")
        
        # 3. Add post-load optimizations
        if "model = model.eval()" in new_content:
            new_content = new_content.replace(
                "model = model.eval()",
                "model = model.eval()\n            " + post_load_patch
            )
        else:
            logger.warning("Could not find exact location for post-load patch, skipping")
        
        # Write back the modified content
        worker_path.write_text(new_content)
        
        logger.info("Worker file patched successfully")
        return True
    except Exception as e:
        logger.error(f"Error patching worker file: {e}")
        return False

def apply_all_optimizations():
    """
    Apply all available optimizations.
    """
    success = True
    
    # Step 1: Apply GPU memory optimizations
    if not optimize_gpu_memory():
        logger.warning("Failed to apply GPU memory optimizations")
        success = False
    
    # Step 2: Patch config file
    if not patch_config_file():
        logger.warning("Failed to patch config file")
        success = False
    
    # Step 3: Patch worker file
    if not patch_worker_file():
        logger.warning("Failed to patch worker file")
        success = False
    
    return success

def main():
    """Run the optimization process."""
    parser = argparse.ArgumentParser(description="Optimize LLaDA GUI performance")
    parser.add_argument("--no-gpu-opt", action="store_true",
                        help="Skip GPU memory optimizations")
    parser.add_argument("--no-config-patch", action="store_true",
                        help="Skip config file patching")
    parser.add_argument("--no-worker-patch", action="store_true",
                        help="Skip worker file patching")
    
    args = parser.parse_args()
    
    if not args.no_gpu_opt and not args.no_config_patch and not args.no_worker_patch:
        # Apply all optimizations
        if apply_all_optimizations():
            logger.info("All optimizations applied successfully!")
            logger.info("Restart the LLaDA GUI to apply the changes.")
            return 0
        else:
            logger.error("Some optimizations failed to apply.")
            return 1
    else:
        # Apply selected optimizations
        success = True
        
        if not args.no_gpu_opt:
            if not optimize_gpu_memory():
                logger.warning("Failed to apply GPU memory optimizations")
                success = False
        
        if not args.no_config_patch:
            if not patch_config_file():
                logger.warning("Failed to patch config file")
                success = False
        
        if not args.no_worker_patch:
            if not patch_worker_file():
                logger.warning("Failed to patch worker file")
                success = False
        
        if success:
            logger.info("Selected optimizations applied successfully!")
            logger.info("Restart the LLaDA GUI to apply the changes.")
            return 0
        else:
            logger.error("Some optimizations failed to apply.")
            return 1

if __name__ == "__main__":
    sys.exit(main())
