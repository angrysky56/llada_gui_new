#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI enhancement functions for LLaDA with memory integration.

This module provides functions for enhancing the LLaDA GUI with memory
capabilities.
"""

import time
import logging
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, 
    QMessageBox, QTabWidget
)
from PyQt6.QtCore import QTimer

# Local imports
from ..memory_init import initialize_memory, get_memory_interface, reset_memory
from .visualization import MemoryVisualizationWidget
from .worker import MemoryGuidanceDiffusionWorker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_server.log'
)
logger = logging.getLogger("memory_enhancements")

def enhance_llada_gui(llada_gui_class):
    """
    Function to enhance the LLaDA GUI with memory capabilities.
    
    This takes the original GUI class and adds memory-related features
    without modifying the original code directly.
    
    Args:
        llada_gui_class: The original LLaDAGUI class
        
    Returns:
        Enhanced GUI class with memory capabilities
    """

    class EnhancedGUI(llada_gui_class):
        """Enhanced LLaDA GUI with memory capabilities."""

        def __init__(self):
            # Initialize memory interface first
            self.memory_interface = get_memory_interface() or initialize_memory(False)

            # Call parent constructor
            super().__init__()

            # Modify window title
            self.setWindowTitle(self.windowTitle() + " with Cognitive Memory")

        def init_ui(self):
            """Initialize the UI with memory enhancements."""
            # Call parent method to set up base UI
            super().init_ui()

            # Add memory tab to the output tab widget
            if hasattr(self, 'tab_widget') and isinstance(self.tab_widget, QTabWidget):
                self.memory_viz = MemoryVisualizationWidget(self.memory_interface)
                self.tab_widget.addTab(self.memory_viz, "Memory Visualization")
            else:
                logger.warning("Could not find tab_widget in LLaDA GUI, memory tab not added")

            # Add memory toggle to parameters
            memory_layout = QHBoxLayout()
            self.use_memory = QCheckBox("Use Memory Guidance")
            self.use_memory.setToolTip("Enable cognitive memory guidance for more coherent generation")
            memory_layout.addWidget(self.use_memory)

            # Find the params_layout in the parent GUI and add memory controls
            # This is a bit hacky since we're modifying the existing UI
            for child in self.findChildren(QGroupBox):
                if child.title() == "Generation Parameters":
                    # Assuming the last layout in the parameters box is a grid layout
                    params_layout = child.layout()
                    if params_layout:
                        # Get the row count and add our memory controls
                        row = params_layout.rowCount()
                        params_layout.addWidget(QLabel("Memory:"), row, 0)
                        params_layout.addLayout(memory_layout, row, 1, 1, 3)

            # Try to connect to memory system
            QTimer.singleShot(2000, self.check_memory_connection)

        def check_memory_connection(self):
            """Check if memory system is available and connect if possible."""
            # Try to initialize in background
            try:
                # Auto-start the memory server if memory integration is enabled
                checkbox_enabled = False
                for child in self.findChildren(QCheckBox):
                    if child.text() == "Enable Memory Integration (context-aware generation)":
                        checkbox_enabled = child.isChecked()
                        break

                if checkbox_enabled:
                    result = initialize_memory(True)  # Try to start server
                    if result:
                        self.memory_viz.update_memory_status(True)
                        logger.info("Memory system connected successfully")
            except Exception as e:
                logger.error(f"Error initializing memory integration: {str(e)}")
                logger.info("Proceeding without memory integration.")

        def start_generation(self):
            """Start the generation process with memory support."""
            prompt = self.input_text.toPlainText().strip()

            if not prompt:
                QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt before generating.")
                return

            # Get configuration from UI
            config = self.get_generation_config()

            # Add memory weight if using memory
            if hasattr(self, 'use_memory') and self.use_memory.isChecked():
                config['use_memory'] = True
                config['memory_weight'] = self.memory_viz.get_memory_influence()
            else:
                config['use_memory'] = False

            # Create memory-aware worker if needed
            if config.get('use_memory', False) and self.memory_interface and self.memory_interface.initialized:
                # Disable input controls during generation
                self.set_controls_enabled(False)

                # Setup progress bar
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.status_label.setText("Initializing with memory guidance...")

                # Clear previous output
                self.output_text.clear()

                # Setup visualization for the diffusion process
                if hasattr(self, 'diffusion_viz'):
                    self.diffusion_viz.setup_process(config['gen_length'], config['steps'])

                # Create and start memory-guided worker thread
                self.worker = MemoryGuidanceDiffusionWorker(prompt, config, self.memory_interface)
                self.worker.progress.connect(self.update_progress)

                if hasattr(self, 'update_visualization'):
                    self.worker.step_update.connect(self.update_visualization)

                self.worker.memory_update.connect(self.update_memory_visualization)
                self.worker.finished.connect(self.generation_finished)
                self.worker.error.connect(self.generation_error)
                self.worker.memory_warning.connect(self.display_memory_warning)
                self.worker.start()

                # Enable stop button
                if hasattr(self, 'stop_btn'):
                    self.stop_btn.setEnabled(True)

                # Don't auto-switch to the memory tab - this causes screen switching issues
                # Let the user manually switch tabs if they want to see the memory visualization
            else:
                # Fall back to standard generation
                if config.get('use_memory', False) and (not self.memory_interface or not self.memory_interface.initialized):
                    # Memory was requested but not available
                    QMessageBox.warning(
                        self,
                        "Memory Not Available",
                        "Memory guidance was requested but the memory system is not connected. "
                        "Proceeding with standard generation."
                    )

                # Call the original start_generation
                super().start_generation()

        def generation_finished(self, result=None):
            """Handle completion of generation."""
            # Call parent method
            if hasattr(super(), 'generation_finished'):
                super().generation_finished(result)

            # If using memory, update the memory visualization
            if hasattr(self, 'use_memory') and self.use_memory.isChecked() and hasattr(self, 'memory_viz'):
                # Get the generated text
                generated_text = ""
                if hasattr(self, 'output_text'):
                    generated_text = self.output_text.toPlainText().strip()

                # Get the prompt
                prompt = ""
                if hasattr(self, 'input_text'):
                    prompt = self.input_text.toPlainText().strip()

                # Store for training
                if prompt and generated_text:
                    self.memory_viz.set_generation_data(prompt, generated_text)
                    
                    # Auto-train if enabled
                    if hasattr(self.memory_viz, 'auto_train') and self.memory_viz.auto_train.isChecked():
                        QTimer.singleShot(500, self.memory_viz.train_memory)

        def update_memory_visualization(self, memory_state):
            """Update memory visualization with current state."""
            if hasattr(self, 'memory_viz'):
                self.memory_viz.display_memory_state(memory_state)

        def display_memory_warning(self, warning_msg):
            """Display a memory-related warning."""
            QMessageBox.warning(self, "Memory Warning", warning_msg)
            
        def update_memory_info(self, memory_info):
            """Update memory usage information.
            
            This method is called by the memory monitor to update system and GPU memory info.
            
            Args:
                memory_info: Dictionary with memory usage information
            """
            # Pass the call to the parent class if it has this method
            if hasattr(super(), 'update_memory_info'):
                super().update_memory_info(memory_info)
                
        def update_quantization_options(self, checked):
            """Update quantization options based on device selection.
            
            This method is called when the device radio buttons are toggled.
            
            Args:
                checked: Whether the radio button is checked
            """
            # Pass the call to the parent class if it has this method
            if hasattr(super(), 'update_quantization_options'):
                super().update_quantization_options(checked)
                
        def get_generation_config(self):
            """Get the generation configuration from UI controls.
            
            Returns:
                Dictionary with generation configuration parameters
            """
            # Get the configuration from the parent class
            if hasattr(super(), 'get_generation_config'):
                config = super().get_generation_config()
            else:
                # Fallback if parent doesn't have this method
                config = {
                    'gen_length': self.gen_length_spin.value() if hasattr(self, 'gen_length_spin') else 128,
                    'steps': self.steps_spin.value() if hasattr(self, 'steps_spin') else 64,
                    'block_length': self.block_length_spin.value() if hasattr(self, 'block_length_spin') else 16,
                    'temperature': self.temperature_spin.value() if hasattr(self, 'temperature_spin') else 1.0,
                    'cfg_scale': self.cfg_scale_spin.value() if hasattr(self, 'cfg_scale_spin') else 3.0,
                    'remasking': self.remasking_combo.currentText() if hasattr(self, 'remasking_combo') else 'low_confidence',
                    'device': 'cuda' if hasattr(self, 'gpu_radio') and self.gpu_radio.isChecked() else 'cpu',
                    'use_8bit': hasattr(self, 'use_8bit') and self.use_8bit.isChecked(),
                    'use_4bit': hasattr(self, 'use_4bit') and self.use_4bit.isChecked(),
                    'extreme_mode': hasattr(self, 'extreme_mode') and self.extreme_mode.isChecked(),
                    'fast_mode': hasattr(self, 'fast_mode') and self.fast_mode.isChecked(),
                    'use_memory': False  # Default value, will be updated if memory is enabled
                }
                
            # Add memory configuration if available
            if hasattr(self, 'use_memory'):
                config['use_memory'] = self.use_memory.isChecked()
                
                if config['use_memory'] and hasattr(self, 'memory_viz'):
                    config['memory_weight'] = self.memory_viz.get_memory_influence()
                    
            return config
            
        def setup_welcome_message(self):
            """Set up the welcome message in the output area.
            
            This method is called during initialization to display a welcome message.
            """
            # Pass the call to the parent class if it has this method
            if hasattr(super(), 'setup_welcome_message'):
                super().setup_welcome_message()
            else:
                # Fallback welcome message if the parent doesn't have this method
                if hasattr(self, 'output_text'):
                    welcome_msg = (
                        "<h2>Welcome to LLaDA GUI with Cognitive Memory</h2>"
                        "<p>This enhanced version includes memory integration for context-aware generation.</p>"
                        "<p>To get started:</p>"
                        "<ol>"
                        "<li>Enter your prompt in the text area above</li>"
                        "<li>Adjust generation parameters as needed</li>"
                        "<li>Click the 'Generate' button</li>"
                        "</ol>"
                        "<p>The Memory Visualization tab shows the current memory state and allows you to train the system on your generations.</p>"
                    )
                    self.output_text.setHtml(welcome_msg)

    return EnhancedGUI
