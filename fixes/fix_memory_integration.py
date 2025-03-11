#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script for memory integration issues in LLaDA GUI.
Specifically fixes the update_memory_info method to handle missing gpu_usage parameter.
"""

import os
import sys

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
memory_integration_path = os.path.join(script_dir, 'core', 'memory', 'memory_integration.py')

# Check if the file exists
if not os.path.exists(memory_integration_path):
    print(f"Error: Memory integration file not found at {memory_integration_path}")
    sys.exit(1)

# Read the current content
with open(memory_integration_path, 'r') as f:
    content = f.read()

# Find the problematic method
target_method = """    def update_memory_info(self, system_usage, gpu_usage):
        \"\"\"Update memory information display.
        
        This method is required by the parent class's memory monitor.
        
        Args:
            system_usage: System memory usage information
            gpu_usage: GPU memory usage information
        \"\"\"
        if hasattr(self, 'system_memory_progress') and hasattr(self, 'system_memory_label'):
            self.system_memory_progress.setValue(system_usage['percent'])
            self.system_memory_label.setText(f"{system_usage['used']:.1f} / {system_usage['total']:.1f} GB ({system_usage['percent']}%)")
                
        if hasattr(self, 'gpu_memory_progress') and hasattr(self, 'gpu_memory_label'):
            self.gpu_memory_progress.setValue(gpu_usage['percent'])
            self.gpu_memory_label.setText(f"{gpu_usage['used']:.1f} / {gpu_usage['total']:.1f} GB ({gpu_usage['percent']}%)")"""

# Fix the method to make gpu_usage optional with a default value of None
fixed_method = """    def update_memory_info(self, system_usage, gpu_usage=None):
        \"\"\"Update memory information display.
        
        This method is required by the parent class's memory monitor.
        
        Args:
            system_usage: System memory usage information
            gpu_usage: GPU memory usage information (optional)
        \"\"\"
        if hasattr(self, 'system_memory_progress') and hasattr(self, 'system_memory_label'):
            try:
                self.system_memory_progress.setValue(system_usage['percent'])
                self.system_memory_label.setText(f"{system_usage['used']:.1f} / {system_usage['total']:.1f} GB ({system_usage['percent']}%)")
            except (KeyError, TypeError) as e:
                # Handle case where system_usage may not have the expected format
                print(f"Warning: Error updating system memory info: {e}")
            
        if gpu_usage is not None and hasattr(self, 'gpu_memory_progress') and hasattr(self, 'gpu_memory_label'):
            try:
                self.gpu_memory_progress.setValue(gpu_usage['percent'])
                self.gpu_memory_label.setText(f"{gpu_usage['used']:.1f} / {gpu_usage['total']:.1f} GB ({gpu_usage['percent']}%)")
            except (KeyError, TypeError) as e:
                # Handle case where gpu_usage may not have the expected format
                print(f"Warning: Error updating GPU memory info: {e}")"""

# Replace all instances of the target method with the fixed version
# There are multiple duplicated definitions in the file, so we need to replace all of them
updated_content = content.replace(target_method, fixed_method)

# Handle the case where the method is defined in a different format
# The error specifically mentions "enhance_llada_gui.<locals>.EnhancedGUI.update_memory_info()"
# So we need to make sure the method is also fixed in that version
if updated_content == content:
    print("First replacement didn't find any matches. Trying alternate formats...")
    
    # Try matching with different indentation or formatting
    import re
    pattern = r'def update_memory_info\(self, system_usage, gpu_usage\):'
    if re.search(pattern, content):
        # Found the method, but with different formatting
        updated_content = re.sub(
            pattern, 
            'def update_memory_info(self, system_usage, gpu_usage=None):', 
            content
        )
        print("Fixed method with pattern matching.")
    else:
        print("Could not find the problematic method. Manual inspection required.")

# Write the updated content
with open(memory_integration_path, 'w') as f:
    f.write(updated_content)

print(f"Updated {memory_integration_path} to make gpu_usage parameter optional.")
print("Please restart the application for changes to take effect.")
