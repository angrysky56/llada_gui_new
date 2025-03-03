#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaDA GUI Optimization Installer

This script installs performance optimizations for the LLaDA GUI application.
It modifies the configuration and worker files to use memory-efficient settings.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def main():
    """Install performance optimizations."""
    parser = argparse.ArgumentParser(description="Install performance optimizations for LLaDA GUI")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")
    args = parser.parse_args()
    
    # Current directory
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "llada_gui.py").exists():
        print("Error: This script must be run from the LLaDA GUI directory.")
        print("Please run it from the directory containing llada_gui.py.")
        return 1
    
    # Create backups first
    if not args.no_backup:
        print("Creating backups...")
        for file in ["config.py", "llada_worker.py"]:
            if (current_dir / file).exists():
                backup_path = current_dir / f"{file}.backup"
                shutil.copy2(current_dir / file, backup_path)
                print(f"  - Created backup: {backup_path}")
    
    # Update config.py
    print("Updating config.py...")
    config_path = current_dir / "config.py"
    
    if not config_path.exists():
        print("  - Error: config.py not found.")
        return 1
    
    try:
        # Read config content
        config_content = config_path.read_text()
        
        # Check if already optimized
        if "OPTIMIZED_GPU_MEMORY = True" in config_content:
            print("  - config.py is already optimized.")
        else:
            # Find the right place to add optimizations
            memory_section = "# Memory-related constants"
            
            if memory_section in config_content:
                # Add optimizations after memory constants section
                optimization_code = """
# Memory optimization constants
OPTIMIZED_GPU_MEMORY = True
CACHE_PRECISION = "bfloat16"  # Use bfloat16 for better performance with minimal precision loss
ENABLE_ATTENTION_SLICING = True  # Slice attention for lower memory usage
ENABLE_FLASH_ATTENTION = True  # Use flash attention if available
"""
                new_content = config_content.replace(
                    memory_section,
                    memory_section + optimization_code
                )
                
                # Write updated config
                config_path.write_text(new_content)
                print("  - Successfully updated config.py")
            else:
                print("  - Warning: Could not find memory constants section in config.py.")
                print("  - Adding optimizations at the end of the file.")
                
                # Add at the end
                with open(config_path, 'a') as f:
                    f.write("\n\n# Memory optimization constants\n")
                    f.write("OPTIMIZED_GPU_MEMORY = True\n")
                    f.write("CACHE_PRECISION = \"bfloat16\"  # Use bfloat16 for better performance\n")
                    f.write("ENABLE_ATTENTION_SLICING = True  # Slice attention for lower memory usage\n")
                    f.write("ENABLE_FLASH_ATTENTION = True  # Use flash attention if available\n")
                
                print("  - Successfully added optimizations to config.py")
    
    except Exception as e:
        print(f"  - Error updating config.py: {e}")
        return 1
    
    # Update llada_worker.py
    print("Updating llada_worker.py...")
    worker_path = current_dir / "llada_worker.py"
    
    if not worker_path.exists():
        print("  - Error: llada_worker.py not found.")
        return 1
    
    try:
        # Read worker content
        with open(worker_path, "r") as f:
            worker_content = f.read()
        
        # Check if already optimized
        if "OPTIMIZED_GPU_MEMORY" in worker_content:
            print("  - llada_worker.py is already optimized.")
        else:
            # Prepare the updated worker file with optimizations
            with open(current_dir / "optimized_worker.py", "w") as f:
                # Start with imports
                f.write("#!/usr/bin/env python\n")
                f.write("# -*- coding: utf-8 -*-\n\n")
                f.write('"""Worker thread for running LLaDA model generation."""\n\n')
                
                # Add our imports
                f.write("import os\n")
                f.write("import sys\n")
                f.write("import gc\n")
                f.write("import torch\n")
                f.write("import torch.nn.functional as F\n")
                f.write("import numpy as np\n")
                f.write("from PyQt6.QtCore import QThread, pyqtSignal\n\n")
                
                f.write("from config import CRITICAL_GPU_MEMORY_THRESHOLD\n")
                f.write("from utils import cleanup_gpu_memory, get_model_path, format_error\n\n")
                
                # Add optimization imports
                f.write("# Import memory optimization constants if available\n")
                f.write("try:\n")
                f.write("    from config import OPTIMIZED_GPU_MEMORY, CACHE_PRECISION, ENABLE_ATTENTION_SLICING, ENABLE_FLASH_ATTENTION\n")
                f.write("except ImportError:\n")
                f.write("    OPTIMIZED_GPU_MEMORY = False\n")
                f.write("    CACHE_PRECISION = None\n")
                f.write("    ENABLE_ATTENTION_SLICING = False\n")
                f.write("    ENABLE_FLASH_ATTENTION = False\n\n")
                
                # Continue with the rest of the file after the imports
                if "class LLaDAWorker" in worker_content:
                    # Extract the rest of the file from class definition
                    rest_of_file = worker_content.split("class LLaDAWorker")[1]
                    
                    # Write the class with proper indentation
                    f.write("class LLaDAWorker" + rest_of_file)
                else:
                    print("  - Warning: Could not find LLaDAWorker class in worker file.")
                    # Just copy the original file content after our imports
                    f.write(worker_content)
            
            # Now modify the key parts of the worker file
            with open(current_dir / "optimized_worker.py", "r") as f:
                content = f.read()
            
            # Add optimization code at key points
            if "model_load_params[\"torch_dtype\"] = torch.bfloat16 if device == 'cuda' else torch.float32" in content:
                # Add optimization code after setting torch_dtype
                new_content = content.replace(
                    "model_load_params[\"torch_dtype\"] = torch.bfloat16 if device == 'cuda' else torch.float32",
                    "model_load_params[\"torch_dtype\"] = torch.bfloat16 if device == 'cuda' else torch.float32\n\n                # Apply memory optimizations if enabled\n                attention_slicing = False\n                if device == 'cuda' and OPTIMIZED_GPU_MEMORY:\n                    self.progress.emit(16, \"Applying memory optimizations...\", {})\n                    \n                    # Use lower precision for better performance\n                    if CACHE_PRECISION == \"bfloat16\":\n                        load_params[\"torch_dtype\"] = torch.bfloat16\n                    elif CACHE_PRECISION == \"float16\":\n                        load_params[\"torch_dtype\"] = torch.float16\n                    \n                    # Enable attention slicing for lower memory usage\n                    if ENABLE_ATTENTION_SLICING:\n                        attention_slicing = True\n                    \n                    # Enable flash attention if available\n                    if ENABLE_FLASH_ATTENTION:\n                        try:\n                            load_params[\"attn_implementation\"] = \"flash_attention_2\"\n                        except:\n                            pass"
                )
                
                # Add post-model loading attention slicing
                if "model = model.eval()" in new_content:
                    new_content = new_content.replace(
                        "model = model.eval()",
                        "model = model.eval()\n\n                # Apply attention slicing if enabled\n                if device == 'cuda' and attention_slicing:\n                    try:\n                        model.config.use_cache = False  # Disable KV cache for more memory efficiency\n                        # Apply attention slicing with a slice size of 1\n                        if hasattr(model, \"enable_attention_slicing\"):\n                            model.enable_attention_slicing(1)\n                    except Exception as attn_error:\n                        self.progress.emit(22, f\"Warning: Could not apply attention slicing: {str(attn_error)}\", {})"
                    )
                
                # Write the updated content
                with open(current_dir / "optimized_worker.py", "w") as f:
                    f.write(new_content)
                
                # Replace the original file with our optimized version
                shutil.move(current_dir / "optimized_worker.py", worker_path)
                print("  - Successfully updated llada_worker.py")
            else:
                print("  - Warning: Could not find expected patterns in llada_worker.py.")
                print("  - The worker file might have a different structure than expected.")
                
                # Clean up
                if (current_dir / "optimized_worker.py").exists():
                    os.remove(current_dir / "optimized_worker.py")
                
    except Exception as e:
        print(f"  - Error updating llada_worker.py: {e}")
        # Clean up
        if (current_dir / "optimized_worker.py").exists():
            os.remove(current_dir / "optimized_worker.py")
        return 1
    
    print("\nOptimizations applied successfully!")
    print("Please restart the LLaDA GUI application to use the optimized version.")
    print("You can revert changes by restoring the backup files if needed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
