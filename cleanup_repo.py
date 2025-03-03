#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to clean up the repository by removing unnecessary files
and organizing remaining files.
"""

import os
import sys
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Files that are no longer needed
FILES_TO_REMOVE = [
    "optimized_generate.py",
    "optimized_worker.py",
    "fix_visualization.py",
    "update_to_optimized.py",
    "restore_cpu_mode.py",
    "OPTIMIZED_README.md",
    "llada_gui.log",
]

# Files to move to an 'archive' folder
FILES_TO_ARCHIVE = [
    "*.backup",
    "*.backup.opt",
    "*.extreme_backup",
    "config.py.backup",
]

def main():
    """Clean up the repository."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Create an archive directory if it doesn't exist
    archive_dir = os.path.join(current_dir, "archive")
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        logger.info(f"Created archive directory: {archive_dir}")
    
    # Remove unnecessary files
    for file in FILES_TO_REMOVE:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed file: {file}")
    
    # Archive backup files
    for pattern in FILES_TO_ARCHIVE:
        # If the pattern is a glob pattern, get matching files
        if "*" in pattern:
            import glob
            matching_files = glob.glob(os.path.join(current_dir, pattern))
            for file_path in matching_files:
                filename = os.path.basename(file_path)
                archive_path = os.path.join(archive_dir, filename)
                shutil.move(file_path, archive_path)
                logger.info(f"Archived file: {filename}")
        else:
            # Otherwise, move the exact file if it exists
            file_path = os.path.join(current_dir, pattern)
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                archive_path = os.path.join(archive_dir, filename)
                shutil.move(file_path, archive_path)
                logger.info(f"Archived file: {filename}")
    
    # Remove leftover compilation artifacts
    try:
        import glob
        pycache_dirs = glob.glob(os.path.join(current_dir, "**/__pycache__"), recursive=True)
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir)
            logger.info(f"Removed directory: {pycache_dir}")
    except Exception as e:
        logger.error(f"Error removing __pycache__ directories: {e}")
    
    print("\nRepository cleanup complete!")
    print("Unnecessary files have been removed and backup files have been moved to the 'archive' directory.")

if __name__ == "__main__":
    main()
