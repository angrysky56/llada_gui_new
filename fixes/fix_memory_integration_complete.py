#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete fix for memory integration issues in the enhanced LLaDA GUI.

This script adds all missing methods to the EnhancedGUI class in memory_integration.py
to fix inheritance issues and method calls.
"""

import os
import sys
import re

# Get the file path
memory_integration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'core', 'memory', 'memory_integration.py')

print(f"Fixing file: {memory_integration_path}")

# Ensure the file exists
if not os.path.exists(memory_integration_path):
    print(f"Error: Could not find {memory_integration_path}")
    sys.exit(1)

# Read the current content
with open(memory_integration_path, 'r') as f:
    content = f.read()

# Define the methods to add to the EnhancedGUI class
methods_to_add = """
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
        
        def update_memory_info(self, system_usage, gpu_usage):
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
                self.gpu_memory_label.setText(f"{gpu_usage['used']:.1f} / {gpu_usage['total']:.1f} GB ({gpu_usage['percent']}%)")

        def update_quantization_options(self):
            \"\"\"Update quantization options based on device selection.\"\"\"
            # This method is called from the parent class's init_ui
            # We need to implement it here to ensure it works
            if not hasattr(self, 'use_8bit') or not hasattr(self, 'use_4bit'):
                return
                
            # Check if using GPU
            using_gpu = hasattr(self, 'gpu_radio') and self.gpu_radio.isChecked()
            
            # Enable/disable quantization options
            if hasattr(self, 'use_8bit'):
                self.use_8bit.setEnabled(using_gpu)
            
            if hasattr(self, 'use_4bit'):
                self.use_4bit.setEnabled(using_gpu)
            
            if hasattr(self, 'extreme_mode'):
                self.extreme_mode.setEnabled(using_gpu)
        
        def get_generation_config(self):
            \"\"\"Get generation configuration from UI settings.\"\"\"
            # This method is called from the parent class
            # Ensure it works in our enhanced GUI
            
            config = {}
            
            # Get base parameters if they exist
            if hasattr(self, 'gen_length_spin'):
                config['gen_length'] = self.gen_length_spin.value()
            
            if hasattr(self, 'steps_spin'):
                config['steps'] = self.steps_spin.value()
            
            if hasattr(self, 'block_length_spin'):
                config['block_length'] = self.block_length_spin.value()
            
            if hasattr(self, 'temperature_spin'):
                config['temperature'] = self.temperature_spin.value()
            
            if hasattr(self, 'cfg_scale_spin'):
                config['cfg_scale'] = self.cfg_scale_spin.value()
            
            if hasattr(self, 'remasking_combo'):
                config['remasking'] = self.remasking_combo.currentText()
            
            # Get memory parameters
            if hasattr(self, 'use_memory') and self.use_memory.isChecked():
                config['use_memory'] = True
                
                # Get memory influence if available
                if hasattr(self, 'memory_viz'):
                    config['memory_weight'] = self.memory_viz.get_memory_influence()
            else:
                config['use_memory'] = False
            
            # Get device and optimization options
            if hasattr(self, 'device_group'):
                config['device'] = 'cuda' if hasattr(self, 'gpu_radio') and self.gpu_radio.isChecked() else 'cpu'
            
            if hasattr(self, 'use_8bit') and self.use_8bit.isChecked():
                config['use_8bit'] = True
            else:
                config['use_8bit'] = False
            
            if hasattr(self, 'use_4bit') and self.use_4bit.isChecked():
                config['use_4bit'] = True
            else:
                config['use_4bit'] = False
            
            if hasattr(self, 'extreme_mode') and self.extreme_mode.isChecked():
                config['extreme_mode'] = True
            else:
                config['extreme_mode'] = False
            
            if hasattr(self, 'fast_mode') and self.fast_mode.isChecked():
                config['fast_mode'] = True
            else:
                config['fast_mode'] = False
            
            if hasattr(self, 'memory_integration') and self.memory_integration.isChecked():
                config['memory_integration'] = True
            else:
                config['memory_integration'] = False
            
            return config
"""

# Find the EnhancedGUI class definition
class_pattern = r"class EnhancedGUI\(llada_gui_class\):"
match = re.search(class_pattern, content)

if not match:
    print("Error: Could not find the EnhancedGUI class definition")
    sys.exit(1)

# Find where to insert the methods - after the class definition but before the next method
insertion_point = match.end()
next_method = re.search(r"    def __init__", content[insertion_point:])
if next_method:
    insertion_point += next_method.start()
else:
    # If __init__ not found, look for any indented method
    next_method = re.search(r"    def ", content[insertion_point:])
    if next_method:
        insertion_point += next_method.start()
    else:
        print("Error: Could not find a suitable insertion point")
        sys.exit(1)

# Insert the methods
updated_content = content[:insertion_point] + methods_to_add + content[insertion_point:]

# Write the updated content back to the file
with open(memory_integration_path, 'w') as f:
    f.write(updated_content)

print(f"Successfully updated {memory_integration_path}")
print("Added the following methods to EnhancedGUI class:")
print("- setup_welcome_message")
print("- update_memory_info")
print("- update_quantization_options")
print("- get_generation_config")
print("\nYou can now run ./run_memory.sh again.")

# Also check if init needs to be modified to call setup_welcome_message
if "def __init__" in content and "self.setup_welcome_message()" not in content:
    # Find the __init__ method
    init_pattern = r"def __init__\(self\):.*?super\(\).__init__\(\)"
    init_match = re.search(init_pattern, content, re.DOTALL)
    
    if init_match:
        # Add setup_welcome_message call after super().__init__()
        updated_init = init_match.group(0) + "\n        # Set up welcome message\n        self.setup_welcome_message()"
        updated_content = content.replace(init_match.group(0), updated_init)
        
        # Write the updated content back to the file
        with open(memory_integration_path, 'w') as f:
            f.write(updated_content)
        
        print("\nAlso updated __init__ method to call setup_welcome_message()")
