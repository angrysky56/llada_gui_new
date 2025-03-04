#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch to remove generation length limits in extreme mode for LLaDA GUI.
"""

import os
import sys
from pathlib import Path

def patch_file():
    """Apply the patch to remove generation length limits."""
    file_path = Path("llada_gui.py")
    if not file_path.exists():
        print(f"Error: Could not find {file_path}")
        return False
    
    print(f"Reading {file_path}...")
    original_content = file_path.read_text()
    
    # Create backup
    backup_path = file_path.with_suffix(".py.nolimits_backup")
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}")
        backup_path.write_text(original_content)
    
    # Look for the extreme mode section
    print("Searching for extreme mode section...")
    
    extreme_mode_section = """        # Add warning for extreme mode - only limit generation length, not steps
        if self.extreme_mode.isChecked():
            # Only limit generation length for memory safety
            if config['gen_length'] > 64:
                QMessageBox.warning(
                    self,
                    "Extreme Memory Mode Warning",
                    "Generation length has been reduced to 64 for Extreme Memory Mode. "
                    "Longer generations may cause out-of-memory errors."
                )
                self.gen_length_spin.setValue(64)
                config['gen_length'] = 64
            
            # Ensure block length is appropriate
            if config['block_length'] > 32:
                self.block_length_spin.setValue(32)
                config['block_length'] = 32"""
    
    # Check if this section exists in the file
    if extreme_mode_section in original_content:
        print("Found extreme mode section, replacing with warning-only version...")
        
        # Replace with warning-only version
        replacement = """        # Add warnings for extreme mode settings
        if self.extreme_mode.isChecked():
            warnings = []
            
            if config['gen_length'] > 64:
                warnings.append(f"- Generation Length: {config['gen_length']} (recommended: 64 or less)")
                
            if config['block_length'] > 32:
                warnings.append(f"- Block Length: {config['block_length']} (recommended: 32 or less)")
            
            if warnings:
                result = QMessageBox.warning(
                    self,
                    "Extreme Memory Mode Warning",
                    "You're using parameters that exceed the recommended limits for Extreme Memory Mode:\n\n" +
                    "\n".join(warnings) + 
                    "\n\nThese settings may cause out-of-memory errors on GPUs with limited VRAM (8-12GB)." +
                    "\n\nDo you want to continue with these settings?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if result == QMessageBox.StandardButton.No:
                    # User chose to use recommended values
                    if config['gen_length'] > 64:
                        self.gen_length_spin.setValue(64)
                        config['gen_length'] = 64
                        
                    if config['block_length'] > 32:
                        self.block_length_spin.setValue(32)
                        config['block_length'] = 32"""
        
        # Apply the patch
        new_content = original_content.replace(extreme_mode_section, replacement)
        
        # Write the patched file
        print("Writing patched file...")
        file_path.write_text(new_content)
        
        print("Patch successfully applied!")
        return True
    else:
        print("Could not find the expected extreme mode section. Your file may be different from expected.")
        print("Searching for alternative patterns...")
        
        # More generic search using keywords
        if "if self.extreme_mode.isChecked()" in original_content and "Generation length has been reduced" in original_content:
            print("Found extreme mode code with different format, attempting to patch...")
            
            # Try to find the start and end of the section
            start_marker = "# Add warning for extreme mode"
            end_marker = "# Disable input controls during generation"
            
            start_pos = original_content.find(start_marker)
            end_pos = original_content.find(end_marker)
            
            if start_pos > 0 and end_pos > start_pos:
                # Extract the section
                section = original_content[start_pos:end_pos].strip()
                print(f"Found section to replace: {len(section)} characters")
                
                # Create replacement
                replacement = """        # Add warnings for extreme mode settings without forcing limits
        if self.extreme_mode.isChecked():
            warnings = []
            
            if config['gen_length'] > 64:
                warnings.append(f"- Generation Length: {config['gen_length']} (recommended: 64 or less)")
                
            if config['block_length'] > 32:
                warnings.append(f"- Block Length: {config['block_length']} (recommended: 32 or less)")
            
            if warnings:
                result = QMessageBox.warning(
                    self,
                    "Extreme Memory Mode Warning",
                    "You're using parameters that exceed the recommended limits for Extreme Memory Mode:\\n\\n" +
                    "\\n".join(warnings) + 
                    "\\n\\nThese settings may cause out-of-memory errors on GPUs with limited VRAM (8-12GB)." +
                    "\\n\\nDo you want to continue with these settings?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if result == QMessageBox.StandardButton.No:
                    # User chose to use recommended values
                    if config['gen_length'] > 64:
                        self.gen_length_spin.setValue(64)
                        config['gen_length'] = 64
                        
                    if config['block_length'] > 32:
                        self.block_length_spin.setValue(32)
                        config['block_length'] = 32
                        
        """
                
                # Replace the section
                new_content = original_content.replace(original_content[start_pos:end_pos], replacement)
                
                # Write the patched file
                print("Writing patched file...")
                file_path.write_text(new_content)
                
                print("Patch successfully applied!")
                return True
        
        print("Could not apply patch automatically. Please manually edit the file.")
        print("Look for the section where extreme mode parameters are enforced and replace")
        print("the hard limits with warnings that give the user a choice.")
        return False

if __name__ == "__main__":
    print("LLaDA GUI Extreme Mode Limit Remover")
    print("====================================")
    print("This script will replace hard limits with warnings in extreme mode.")
    
    if patch_file():
        print("\nSuccess!")
        print("You can now use higher generation length with extreme memory mode.")
        print("A warning will be shown, but you can choose to proceed with your settings.")
    else:
        print("\nFailed to apply patch automatically.")
