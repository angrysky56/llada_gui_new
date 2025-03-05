#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for removing extreme mode warnings.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the warning removal script from the correct directory."""
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Path to the script
    script_path = current_dir / "remove_warnings.py"
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        return 1
    
    # Change to the LLaDA GUI directory
    os.chdir(current_dir)
    print(f"Changed directory to: {os.getcwd()}")
    
    # Run the script
    print(f"Running script: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], 
                           capture_output=False, 
                           text=True)
    
    # Return the exit code
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
