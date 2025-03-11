#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix for memory_integration.py's update_memory_info method.
This script adds proper error handling to prevent crashes when system_usage
doesn't have the expected format or missing keys.
"""

import os
import sys
import re

def fix_memory_integration():
    """
    Add error handling to update_memory_info method in memory_integration.py
    """
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "core", "memory", "memory_integration.py"
    )
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the update_memory_info method
    # Look for all occurrences of the method
    pattern = r'def update_memory_info\(self, system_usage, gpu_usage=None\):[^\n]*\n((?:[ \t]+[^\n]*\n)+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print("Error: update_memory_info method not found in file")
        return False
    
    # Create the fixed method with improved error handling
    fixed_method = '''    def update_memory_info(self, system_usage, gpu_usage=None):
        """Update memory information display.
        
        This method is required by the parent class's memory monitor.
        
        Args:
            system_usage: System memory usage information
            gpu_usage: GPU memory usage information (optional)
        """
        try:
            if hasattr(self, 'system_memory_progress') and system_usage is not None and isinstance(system_usage, dict):
                if 'percent' in system_usage:
                    self.system_memory_progress.setValue(system_usage['percent'])
                
                if hasattr(self, 'system_memory_label'):
                    used = system_usage.get('used', 0)
                    total = system_usage.get('total', 0)
                    percent = system_usage.get('percent', 0)
                    self.system_memory_label.setText(f"{used:.1f} / {total:.1f} GB ({percent}%)")
            
            if gpu_usage is not None and hasattr(self, 'gpu_memory_progress') and isinstance(gpu_usage, dict):
                if 'percent' in gpu_usage:
                    self.gpu_memory_progress.setValue(gpu_usage['percent'])
                
                if hasattr(self, 'gpu_memory_label'):
                    used = gpu_usage.get('used', 0)
                    total = gpu_usage.get('total', 0)
                    percent = gpu_usage.get('percent', 0)
                    self.gpu_memory_label.setText(f"{used:.1f} / {total:.1f} GB ({percent}%)")
        except Exception as e:
            print(f"Error updating memory info: {e}")
'''
    
    # Replace all occurrences of the method
    for match in matches:
        content = content.replace(
            f"def update_memory_info(self, system_usage, gpu_usage=None):\n{match}",
            fixed_method
        )
    
    # Create backup
    backup_path = file_path + ".bak"
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {file_path}")
    print(f"Backup saved to {backup_path}")
    return True

if __name__ == "__main__":
    fix_memory_integration()
