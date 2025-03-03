#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for LLaDA GUI optimization tools.

This script provides a simple launcher for the optimization tools,
ensuring they're called from the right directory.
"""

import os
import sys
from pathlib import Path

def main():
    """Launch the optimization GUI."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Path to optimization directory
    opt_dir = script_dir / "optimizations"
    
    # Check if optimization directory exists
    if not opt_dir.exists():
        print(f"Error: Optimization directory not found at {opt_dir}")
        print("Please make sure the 'optimizations' directory is present in the LLaDA GUI directory.")
        return 1
    
    # Check for GUI script
    gui_script = opt_dir / "optimize_gui.py"
    
    if not gui_script.exists():
        print(f"Error: Optimization GUI script not found at {gui_script}")
        print("Please make sure the 'optimize_gui.py' file is present in the 'optimizations' directory.")
        return 1
    
    # Launch the GUI
    print(f"Launching optimization GUI from {gui_script}")
    
    # Change to the optimizations directory
    os.chdir(opt_dir)
    
    # Import and run the GUI
    try:
        sys.path.insert(0, str(opt_dir))
        from optimize_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"Error launching optimization GUI: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
