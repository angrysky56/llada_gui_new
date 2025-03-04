#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for the extreme mode limit patch.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the patch from the correct directory."""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Path to the patch script
    patch_script = current_dir / "remove_extreme_limits.py"
    
    if not patch_script.exists():
        print(f"Error: Patch script not found at {patch_script}")
        return 1
    
    # Change to the LLaDA GUI directory
    os.chdir(current_dir)
    print(f"Changed directory to: {os.getcwd()}")
    
    # Run the patch script
    print(f"Running patch script: {patch_script}")
    result = subprocess.run([sys.executable, str(patch_script)], 
                           capture_output=False, 
                           text=True)
    
    # Return the exit code
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
