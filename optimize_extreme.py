#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for LLaDA GUI extreme memory optimization.

This script provides a simple entry point for applying extreme memory optimizations
to allow LLaDA to run on GPUs with as little as 8-12GB VRAM.
"""

import os
import sys
from pathlib import Path

def main():
    """Main function to launch the extreme optimizer GUI."""
    try:
        # Set up PyTorch memory optimization settings before importing
        import torch
        
        # Apply basic memory settings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.set_float32_matmul_precision('medium')
    except Exception as e:
        print(f"Warning: Could not set PyTorch memory settings: {e}")
        
    # Get the path to the optimizer module
    script_dir = Path(__file__).parent
    optimizer_gui_path = script_dir / "optimizations/extreme/extreme_optimizer_gui.py"
    
    if not optimizer_gui_path.exists():
        print(f"Error: Extreme optimizer GUI not found at {optimizer_gui_path}")
        print("Make sure the file exists in the optimizations/extreme directory.")
        return 1
    
    # Launch the GUI
    print(f"Launching Extreme Memory Optimizer GUI from {optimizer_gui_path}")
    
    # Execute the GUI script
    try:
        # Change to the script directory to ensure relative imports work
        os.chdir(script_dir)
        
        # Run the script with the current Python interpreter
        result = os.system(f"{sys.executable} {optimizer_gui_path}")
        
        # Check result
        if result != 0:
            print(f"Error: Optimizer GUI exited with code {result}")
            return result
        
        return 0
    except Exception as e:
        print(f"Error launching optimizer GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
