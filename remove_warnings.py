#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to completely remove extreme mode warnings from LLaDA GUI.
"""

import os
import sys
from pathlib import Path

def remove_warnings():
    """Remove all extreme mode warnings from the GUI code."""
    # Path to the GUI file
    file_path = Path("llada_gui.py")
    
    if not file_path.exists():
        print(f"Error: Could not find {file_path}")
        return False
    
    print(f"Reading {file_path}...")
    original_content = file_path.read_text()
    
    # Create backup
    backup_path = file_path.with_suffix(".py.no_warnings_backup")
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}")
        backup_path.write_text(original_content)
    
    # Find the extreme mode section. Look for various possible patterns
    # since the file might have been modified already
    start_marker = "# Add warning for extreme mode"
    alt_start_marker = "# Add warnings for extreme mode"
    
    # Find the section end marker
    end_markers = [
        "# Disable input controls during generation",
        "# Setup progress bar"
    ]
    
    # Find start position
    start_pos = -1
    for marker in [start_marker, alt_start_marker]:
        pos = original_content.find(marker)
        if pos >= 0:
            start_pos = pos
            break
    
    if start_pos < 0:
        print("Could not find the extreme mode warning section.")
        return False
    
    # Find end position
    end_pos = -1
    for marker in end_markers:
        pos = original_content.find(marker, start_pos)
        if pos >= 0:
            if end_pos < 0 or pos < end_pos:
                end_pos = pos
    
    if end_pos < 0:
        print("Could not find the end of the extreme mode warning section.")
        return False
    
    # Extract the section to be removed
    section = original_content[start_pos:end_pos].strip()
    print(f"Found warning section: {len(section)} characters")
    
    # New content just has an empty comment 
    replacement = "        # No warnings or restrictions for extreme mode\n        "
    
    # Replace the section
    new_content = original_content.replace(original_content[start_pos:end_pos], replacement)
    
    # Write the modified file
    print("Writing updated file...")
    file_path.write_text(new_content)
    
    print("Successfully removed all extreme mode warnings!")
    return True

def main():
    """Main function."""
    print("LLaDA GUI Extreme Mode Warning Remover")
    print("======================================")
    print("This script will completely remove extreme mode warnings.")
    
    if remove_warnings():
        print("\nSuccess!")
        print("All extreme mode warnings have been removed.")
        print("You can now use any generation parameters without warnings.")
    else:
        print("\nFailed to remove warnings.")
    
    return 0 if remove_warnings() else 1

if __name__ == "__main__":
    sys.exit(main())
