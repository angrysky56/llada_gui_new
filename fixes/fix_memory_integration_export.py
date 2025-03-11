#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script for memory integration export in LLaDA GUI.
This adds a properly exported enhance_llada_gui function to memory_integration.py.
"""

import os
import sys

# Get the project root directory
# Get the direct path to memory_integration.py
memory_integration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_integration.py')

print(f"Fixing file: {memory_integration_path}")

# Check if the file exists
if not os.path.exists(memory_integration_path):
    print(f"Error: Memory integration file not found at {memory_integration_path}")
    sys.exit(1)

# Read the current content
with open(memory_integration_path, 'r') as f:
    content = f.read()

# Define the function to add at module level (at the end of the file)
function_to_add = """
# Function to enhance the LLaDA GUI with memory capabilities
def enhance_llada_gui(llada_gui_class):
    \"\"\"Enhance the LLaDA GUI with memory capabilities.
    
    Args:
        llada_gui_class: The original LLaDA GUI class to enhance
        
    Returns:
        Enhanced class with memory capabilities
    \"\"\"
    # Create an enhanced version of the GUI
    class EnhancedGUI(llada_gui_class):
        \"\"\"Enhanced LLaDA GUI with memory capabilities.\"\"\"
        
        def __init__(self):
            \"\"\"Initialize the enhanced GUI.\"\"\"
            # Call parent init first
            super().__init__()
            
            # Add memory tab if we have the memory adapter
            try:
                from .memory_adapter import add_memory_visualization_tab
                
                # Add memory visualization tab
                self.memory_viz = add_memory_visualization_tab(self)
                
                # Add memory integration checkbox
                self.add_memory_integration_option()
                
                print("Memory integration enabled")
            except ImportError as e:
                print(f"Warning: Could not add memory visualization: {e}")
        
        def add_memory_integration_option(self):
            \"\"\"Add memory integration checkbox to GUI.\"\"\"
            if hasattr(self, 'settings_layout'):
                from PyQt6.QtWidgets import QCheckBox
                
                # Add memory integration checkbox
                self.memory_integration = QCheckBox("Use Memory Integration")
                self.memory_integration.setChecked(False)
                self.memory_integration.setToolTip("Enable memory integration for more coherent generations")
                self.settings_layout.addWidget(self.memory_integration)
        
        def setup_welcome_message(self):
            \"\"\"Set up welcome message in the output text area.\"\"\"
            if hasattr(self, 'output_text'):
                welcome_message = (
                    "<h2>Welcome to LLaDA GUI with Memory Integration</h2>"
                    "<p>This enhanced version integrates cognitive memory capabilities for more coherent and contextual generation.</p>"
                    "<p><b>Key features:</b></p>"
                    "<ul>"
                    "<li>Memory-guided generation for improved coherence</li>"
                    "<li>Memory visualization for understanding the model's internal state</li>"
                    "<li>Trainable memory system that learns from your generations</li>"
                    "</ul>"
                    "<p>Enter your prompt and click 'Generate' to start. Enable memory integration for enhanced results.</p>"
                )
                self.output_text.setHtml(welcome_message)
        
        def update_memory_info(self, system_usage, gpu_usage=None):
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
                    print(f"Warning: Error updating GPU memory info: {e}")
    
    # Return the enhanced class
    return EnhancedGUI
"""

# Check if the function already exists at module level
if "def enhance_llada_gui(llada_gui_class)" in content:
    print("The enhance_llada_gui function already exists at module level. No changes made.")
    sys.exit(0)

# Append the function to the end of the file
updated_content = content + "\n" + function_to_add

# Write the updated content
with open(memory_integration_path, 'w') as f:
    f.write(updated_content)

print(f"Updated {memory_integration_path} to export enhance_llada_gui function.")
print("You should now be able to run the LLaDA GUI with memory integration.")
