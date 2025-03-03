#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cleanup script for LLaDA GUI repository.

This script removes temporary files and makes sure
the repository is in a clean state.
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    """Run the cleanup process."""
    print("Starting cleanup...")
    
    # Get the repository root directory
    repo_dir = Path(__file__).parent.absolute()
    print(f"Repository directory: {repo_dir}")
    
    # Remove cache directories
    cache_dirs = [
        "__pycache__",
        "*/__pycache__",
        "**/__pycache__",
    ]
    
    print("Removing Python cache directories...")
    count = 0
    for pattern in cache_dirs:
        for path in repo_dir.glob(pattern):
            if path.is_dir():
                print(f"  - Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
                count += 1
    
    print(f"Removed {count} cache directories")
    
    # Remove temporary files
    temp_patterns = [
        "*.pyc",
        "*.pyo",
        "*~",
        "*.bak",
        "*.swp",
        "*.tmp",
        "optimized_worker.py"
    ]
    
    print("Removing temporary files...")
    count = 0
    for pattern in temp_patterns:
        for path in repo_dir.glob(pattern):
            if path.is_file():
                print(f"  - Removing {path}")
                path.unlink()
                count += 1
    
    print(f"Removed {count} temporary files")
    
    # Remove .DS_Store files
    print("Removing .DS_Store files...")
    count = 0
    for path in repo_dir.glob("**/.DS_Store"):
        if path.is_file():
            print(f"  - Removing {path}")
            path.unlink()
            count += 1
    
    print(f"Removed {count} .DS_Store files")
    
    print("Cleanup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
