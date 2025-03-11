#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for QObject import error in memory_monitor.py.

This script corrects the import of QObject, which should be from QtCore instead of QtWidgets.
"""

import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qobject_import_fix")

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fix_memory_monitor():
    """Fix the QObject import in memory_monitor.py."""
    memory_monitor_path = os.path.join(SCRIPT_DIR, "gui", "memory_monitor.py")
    
    if not os.path.exists(memory_monitor_path):
        logger.error(f"Memory monitor file not found: {memory_monitor_path}")
        return False
    
    with open(memory_monitor_path, "r") as f:
        content = f.read()
    
    # Fix the import statement
    incorrect_import = "from PyQt6.QtWidgets import QApplication, QObject"
    if incorrect_import in content:
        # Replace with correct imports
        corrected_content = content.replace(
            incorrect_import, 
            "from PyQt6.QtWidgets import QApplication\nfrom PyQt6.QtCore import QObject"
        )
        
        # Write back the modified content
        with open(memory_monitor_path, "w") as f:
            f.write(corrected_content)
        
        logger.info("Fixed QObject import in memory_monitor.py")
        return True
    else:
        logger.info("QObject import not found or already fixed in memory_monitor.py")
        return False

def main():
    """Main function."""
    print("Fixing QObject import error...")
    
    if fix_memory_monitor():
        print("✅ Successfully fixed QObject import in memory_monitor.py")
    else:
        print("❌ Failed to fix QObject import")
    
    print("\nYou can now run the LLaDA GUI again using ./run_memory.sh")

if __name__ == "__main__":
    main()
