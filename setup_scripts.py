#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper script to copy necessary files from the original LLaDA repository.
"""

import os
import shutil
import sys

def copy_file(src, dst):
    """Copy a file and print the result."""
    try:
        shutil.copy2(src, dst)
        print(f"Successfully copied {src} to {dst}")
        return True
    except Exception as e:
        print(f"Failed to copy {src}: {str(e)}")
        return False

def main():
    """Copy necessary script files from original repository."""
    # Define source and destination directories
    src_dir = "/home/ty/Repositories/LLaDA"
    dst_dir = "/home/ty/Repositories/ai_workspace/llada_gui"
    
    # Ensure destination directory exists
    if not os.path.exists(dst_dir):
        print(f"Destination directory {dst_dir} does not exist!")
        return False
    
    # List of files to copy
    files_to_copy = [
        "generate.py",
        "get_log_likelihood.py",
        "chat.py"
    ]
    
    # Copy each file
    success = True
    for file in files_to_copy:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        
        if not os.path.exists(src_file):
            print(f"Source file {src_file} does not exist!")
            success = False
            continue
        
        if not copy_file(src_file, dst_file):
            success = False
    
    if success:
        print("All files copied successfully!")
    else:
        print("Some files could not be copied. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
