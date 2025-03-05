#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced LLaDA GUI with MCP Titan Memory integration.

This extends the existing LLaDA GUI to incorporate cognitive memory capabilities
using the MCP Titan Memory system. This version automatically starts and manages
the memory server internally, with no external dependencies.
"""

import os
import sys
import torch
import numpy as np
import json
import requests
import subprocess
import time
import signal
import atexit
import logging
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup,
    QSlider, QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the internal memory server module
from memory_server.server import start_server

# Memory server management
class MemoryServerManager:
    """Manages the MCP Titan Memory server process internally."""
    
    def __init__(self, api_url="http://localhost:3000"):
        """Initialize the server manager.
        
        Args:
            api_url: Base URL for the memory server API
        """
        self.api_url = api_url
        self.server_thread = None
        self.is_running = False
        
        # Register the cleanup handler
        atexit.register(self.stop_server)
    
    def start_server(self):
        """Start the memory server in a separate thread."""
        if self.is_running:
            logger.info("Memory server is already running")
            return True
        
        try:
            logger.info("Starting memory server thread")
            # Parse the host and port from the API URL
            parts = self.api_url.split('://')
            if len(parts) > 1:
                host_port = parts[1].split(':')
                host = host_port[0]
                if len(host_port) > 1:
                    port = int(host_port[1].split('/')[0])
                else:
                    port = 80
            else:
                host = "localhost"
                port = 3000
            
            # Start server in a thread
            self.server_thread = threading.Thread(
                target=start_server,
                args=(host, port),
                daemon=True
            )
            self.server_thread.start()
            
            # Wait for server to start (up to 30 seconds)
            start_time = time.time()
            max_wait = 30
            is_ready = False
            
            logger.info("Waiting for memory server to initialize...")
            while time.time() - start_time < max_wait:
                try:
                    # Check if server is running by making a status request
                    response = requests.get(f"{self.api_url}/status", timeout=1)
                    if response.status_code == 200:
                        is_ready = True
                        break
                except requests.exceptions.RequestException:
                    # Server not ready yet, wait and retry
                    time.sleep(1)
            
            if is_ready:
                logger.info("Memory server started successfully")
                self.is_running = True
                return True
            else:
                logger.error("Memory server failed to start in %d seconds", max_wait)
                self.stop_server()
                return False
                
        except Exception as e:
            logger.error("Failed to start memory server: %s", str(e))
            self.stop_server()
            return False
    
    def stop_server(self):
        """Stop the memory server."""
        if self.is_running:
            logger.info("Stopping memory server...")
            self.is_running = False
            # The server will be terminated when the main process exits
            # since we set daemon=True on the thread
            logger.info("Memory server stopped")
    
    def check_server(self):
        """Check if the server is running and responding."""
        if not self.is_running:
            return False
            
        try:
            response = requests.get(f"{self.api_url}/status", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

# Memory interface for MCP Titan
class MCPTitanMemoryInterface:
    """Interface to the MCP Titan Memory system via HTTP API."""
    
    def __init__(self, api_url="http://localhost:3000"):
        """Initialize the memory interface.
        
        Args:
            api_url: URL of the MCP Titan Memory API
        """
        self.api_url = api_url
        self.memory_state = None
        self.input_dim = 64  # Default, will be updated from model
        self.memory_dim = 64  # Default, will be updated from model
        self.initialized = False
        self.server_manager = MemoryServerManager(api_url=api_url)
        
    def initialize(self, input_dim=64, memory_dim=64):
        """Initialize the memory model.
        
        Args:
            input_dim: Dimension of input vectors
            memory_dim: Dimension of memory vectors
            
        Returns:
            True if successful, False otherwise
        """
        # First, make sure the server is running
        if not self.server_manager.check_server():
            if not self.server_manager.start_server():
                logger.error("Failed to start memory server")
                return False
        
        try:
            # Use the init endpoint
            response = requests.post(
                f"{self.api_url}/init", 
                json={"inputDim": input_dim, "outputDim": memory_dim},
                timeout=5  # 5 second timeout
            )
            response.raise_for_status()
            
            self.input_dim = input_dim
            self.memory_dim = memory_dim
            
            # Initialize memory state to zeros
            self.memory_state = np.zeros(memory_dim)
            self.initialized = True
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            return False
    
    def forward_pass(self, input_vector):
        """Run forward pass through the memory model.
        
        Args:
            input_vector: Input vector of shape [input_dim]
            
        Returns:
            dict with predicted, newMemory, and surprise
        """
        if not self.initialized:
            raise ValueError("Memory not initialized. Call initialize() first.")
            
        try:
            # Use the forward endpoint
            response = requests.post(
                f"{self.api_url}/forward",
                json={
                    "x": input_vector.tolist() if isinstance(input_vector, np.ndarray) else input_vector,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Update memory state
            self.memory_state = np.array(result["memory"])
            
            return {
                "predicted": result["predicted"],
                "newMemory": result["memory"],
                "surprise": result["surprise"]
            }
        except Exception as e:
            logger.error(f"Memory forward pass error: {str(e)}")
            # Return default values
            return {
                "predicted": np.zeros(self.input_dim).tolist(),
                "newMemory": self.memory_state.tolist() if isinstance(self.memory_state, np.ndarray) else self.memory_state,
                "surprise": 0.0
            }
    
    def get_memory_state(self):
        """Get the current memory state."""
        return self.memory_state if self.initialized else np.zeros(self.memory_dim)
    
    def reset(self):
        """Reset the memory state."""
        if self.initialized:
            self.memory_state = np.zeros(self.memory_dim)


class MemoryVisualizationWidget(QWidget):
    """Widget for visualizing memory state and influence."""
    
    def __init__(self, memory_interface, parent=None):
        super().__init__(parent)
        self.memory_interface = memory_interface
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Explanation label
        explanation = QLabel(
            "This visualization shows the memory state during diffusion generation. "
            "The memory system provides guidance based on learned patterns, helping ensure consistency and coherence."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Memory state visualization
        memory_group = QGroupBox("Memory State")
        memory_layout = QVBoxLayout(memory_group)
        
        self.memory_state_label = QLabel("Current Memory State:")
        memory_layout.addWidget(self.memory_state_label)
        
        self.memory_state_viz = QTextEdit()
        self.memory_state_viz.setReadOnly(True)
        self.memory_state_viz.setMaximumHeight(120)
        memory_layout.addWidget(self.memory_state_viz)
        
        # Memory influence settings
        influence_layout = QHBoxLayout()
        influence_layout.addWidget(QLabel("Memory Influence:"))
        
        self.memory_slider = QSlider(Qt.Orientation.Horizontal)
        self.memory_slider.setMinimum(0)
        self.memory_slider.setMaximum(100)
        self.memory_slider.setValue(30)  # Default 30%
        self.memory_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.memory_slider.setTickInterval(10)
        influence_layout.addWidget(self.memory_slider)
        
        self.memory_percent = QLabel("30%")
        influence_layout.addWidget(self.memory_percent)
        
        memory_layout.addLayout(influence_layout)
        
        # Connect slider to update label
        self.memory_slider.valueChanged.connect(self.update_memory_influence)
        
        layout.addWidget(memory_group)
        
        # Memory controls
        controls_layout = QHBoxLayout()
        
        # Memory system status indicator
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.Shape.Box)
        self.status_frame.setFixedWidth(20)
        self.status_frame.setFixedHeight(20)
        self.status_frame.setStyleSheet("background-color: red;")  # Default to red (not connected)
        controls_layout.addWidget(self.status_frame)
        
        self.status_label = QLabel("Memory System: Not Connected")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset Memory")
        self.reset_btn.clicked.connect(self.reset_memory)
        controls_layout.addWidget(self.reset_btn)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_memory)
        controls_layout.addWidget(self.connect_btn)
        
        layout.addLayout(controls_layout)
    
    def update_memory_influence(self, value):
        """Update the memory influence display and value."""
        self.memory_percent.setText(f"{value}%")
    
    def get_memory_influence(self):
        """Get the current memory influence value (0-1)."""
        return self.memory_slider.value() / 100.0
    
    def update_memory_status(self, connected):
        """Update the memory system status indicator."""
        if connected:
            self.status_frame.setStyleSheet("background-color: green;")
            self.status_label.setText("Memory System: Connected")
            self.connect_btn.setText("Disconnect")
        else:
            self.status_frame.setStyleSheet("background-color: red;")
            self.status_label.setText("Memory System: Not Connected")
            self.connect_btn.setText("Connect")
    
    def connect_memory(self):
        """Connect or disconnect the memory system."""
        if self.status_label.text() == "Memory System: Connected":
            # Currently connected, disconnect
            self.update_memory_status(False)
            return False
        else:
            # Try to connect
            if self.memory_interface.initialize():
                self.update_memory_status(True)
                return True
            else:
                QMessageBox.warning(
                    self,
                    "Memory Connection Failed",
                    "Could not connect to the MCP Titan Memory system. "
                    "Make sure the server is running and try again."
                )
                return False
    
    def reset_memory(self):
        """Reset the memory state."""
        if self.memory_interface.initialized:
            self.memory_interface.reset()
            self.display_memory_state(self.memory_interface.get_memory_state())
            QMessageBox.information(
                self,
                "Memory Reset",
                "Memory state has been reset to zeros."
            )
    
    def display_memory_state(self, memory_state):
        """Display the current memory state.
        
        Args:
            memory_state: Array of memory state values
        """
        if memory_state is None:
            self.memory_state_viz.setPlainText("No memory state available")
            return
            
        # Display as a heatmap-like visualization
        memory_state = np.array(memory_state)
        
        # Normalize for visualization
        if memory_state.size > 0:
            min_val = np.min(memory_state)
            max_val = np.max(memory_state)
            if min_val == max_val:
                normalized = np.zeros_like(memory_state)
            else:
                normalized = (memory_state - min_val) / (max_val - min_val + 1e-8)
            
            # Create a visual representation using blocks with color intensity
            html = '<div style="font-family: monospace; line-height: 1.0;">'
            
            # Split into chunks for display
            chunk_size = 16  # Display 16 values per line
            for i in range(0, len(normalized), chunk_size):
                chunk = normalized[i:i+chunk_size]
                line = ""
                
                for value in chunk:
                    # Use a gradient from blue to red
                    intensity = int(255 * value)
                    blue = 255 - intensity
                    red = intensity
                    color = f"rgb({red}, 0, {blue})"
                    line += f'<span style="background-color: {color}; color: white; margin: 1px; padding: 2px;">{value:.2f}</span>'
                
                html += line + "<br/>"
            
            html += '</div>'
            self.memory_state_viz.setHtml(html)
        else:
            self.memory_state_viz.setPlainText("No memory state available")


class MemoryGuidanceDiffusionWorker(QThread):
    """Worker thread for memory-guided diffusion generation.
    
    This extends the base LLaDAWorker with memory guidance capabilities.
    """
    
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    memory_update = pyqtSignal(np.ndarray)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    
    def __init__(self, prompt, config, memory_interface=None):
        """Initialize the worker.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            memory_interface: Memory interface for guidance
        """
        super().__init__()
        self.prompt = prompt
        self.config = config
        self.memory_interface = memory_interface
        self.memory_weight = config.get('memory_weight', 0.3)
        self.is_running = True
        
        # This would be the base LLaDA worker initialization with memory additions
    
    def run(self):
        """Run the generation.
        
        For a full implementation, this would modify the standard LLaDA diffusion
        process to incorporate memory guidance at each step.
        """
        try:
            # For now, this is a simplified implementation
            # to demonstrate the integration concepts
            
            # Simulate the generation process
            total_steps = self.config.get('steps', 64)
            
            # Initialize generation (in a real implementation, this would use the LLaDA model)
            # We'll fake the partial output for demonstration purposes
            current_text = self.prompt
            tokens = list(range(100, 100 + len(self.prompt.split())))
            
            # Generate 1-3 new tokens per step
            for step in range(total_steps):
                if not self.is_running:
                    break
                
                # Update progress
                progress = int((step + 1) / total_steps * 100)
                
                # Simulate new tokens with memory guidance
                if self.memory_interface and self.memory_interface.initialized:
                    # In a real implementation, this would query the memory system
                    # and adjust token probabilities based on memory predictions
                    
                    # Fake memory update - in reality this would be based on the model's internal state
                    new_memory = np.random.randn(self.memory_interface.memory_dim) * 0.1
                    if step > 0:
                        # Evolve memory gradually, don't reset each time
                        current_memory = self.memory_interface.get_memory_state()
                        updated_memory = current_memory * 0.9 + new_memory * 0.1
                        self.memory_interface.memory_state = updated_memory
                    else:
                        self.memory_interface.memory_state = new_memory
                    
                    # Emit memory state update
                    self.memory_update.emit(self.memory_interface.get_memory_state())
                
                # Add 1-3 new tokens to the simulation
                new_token_count = np.random.randint(1, 4)
                new_tokens = list(range(200 + step*4, 200 + step*4 + new_token_count))
                tokens.extend(new_tokens)
                
                # Extend the text (in a real implementation, this would decode the tokens)
                words = ["memory", "guided", "diffusion", "process", "cognitive", 
                        "framework", "enhanced", "generation", "coherent", "consistent"]
                new_words = [words[i % len(words)] for i in range(step*2, step*2 + new_token_count)]
                current_text += " " + " ".join(new_words)
                
                # Create fake masks and confidences for visualization
                masks = [0] * len(tokens)  # All unmasked
                confidences = [0.9] * len(tokens)  # High confidence
                
                # Update UI
                self.progress.emit(
                    progress, 
                    f"Step {step+1}/{total_steps}", 
                    {'partial_output': current_text}
                )
                
                self.step_update.emit(
                    step,
                    tokens,
                    masks,
                    confidences
                )
                
                # Pause between steps for visualization
                QThread.msleep(100)
            
            # Emit final result
            if self.is_running:
                self.finished.emit(current_text)
                
        except Exception as e:
            self.error.emit(f"Memory-guided generation error: {str(e)}")
    
    def stop(self):
        """Stop the generation."""
        self.is_running = False


# Import the memory guided worker
from memory_guided_worker import MemoryGuidedWorker


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
            self.memory_interface = MCPTitanMemoryInterface()
            
            # Call parent constructor
            super().__init__()
            
            # Modify window title
            self.setWindowTitle(self.windowTitle() + " with Cognitive Memory")
        
        def init_ui(self):
            """Initialize the UI with memory enhancements."""
            # Call parent method to set up base UI
            super().init_ui()
            
            # Add memory tab to the output tab widget
            self.memory_viz = MemoryVisualizationWidget(self.memory_interface)
            self.tab_widget.addTab(self.memory_viz, "Memory Visualization")
            
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
            result = self.memory_interface.initialize()
            self.memory_viz.update_memory_status(result)
        
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
            if config.get('use_memory', False) and self.memory_interface.initialized:
                # Disable input controls during generation
                self.set_controls_enabled(False)
                
                # Setup progress bar
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(True)
                self.status_label.setText("Initializing with memory guidance...")
                
                # Clear previous output
                self.output_text.clear()
                
                # Setup visualization for the diffusion process
                self.diffusion_viz.setup_process(config['gen_length'], config['steps'])
                
                # Choose the appropriate worker based on availability
                try:
                    # Try to use the full memory-guided worker implementation first
                    self.worker = MemoryGuidedWorker(prompt, config, self.memory_interface)
                    logger.info("Using full MemoryGuidedWorker implementation")
                except (ImportError, AttributeError) as e:
                    # Fall back to the simplified implementation
                    logger.warning(f"Using simplified MemoryGuidanceDiffusionWorker due to: {str(e)}")
                    self.worker = MemoryGuidanceDiffusionWorker(prompt, config, self.memory_interface)
                
                # Connect signals
                self.worker.progress.connect(self.update_progress)
                self.worker.step_update.connect(self.update_visualization)
                self.worker.memory_update.connect(self.update_memory_visualization)
                self.worker.finished.connect(self.generation_finished)
                self.worker.error.connect(self.generation_error)
                
                # Connect memory warning signal if available
                if hasattr(self.worker, 'memory_warning'):
                    self.worker.memory_warning.connect(self.display_memory_warning)
                
                # Start the worker
                self.worker.start()
                
                # Enable stop button
                self.stop_btn.setEnabled(True)
                
                # Switch to the visualization tab
                self.tab_widget.setCurrentIndex(1)
            else:
                # Fall back to standard generation
                if config.get('use_memory', False) and not self.memory_interface.initialized:
                    # Memory was requested but not available
                    QMessageBox.warning(
                        self, 
                        "Memory Not Available", 
                        "Memory guidance was requested but the memory system is not connected. "
                        "Proceeding with standard generation."
                    )
                
                # Call the original start_generation
                super().start_generation()
        
        def update_memory_visualization(self, memory_state):
            """Update memory visualization with current state."""
            if hasattr(self, 'memory_viz'):
                self.memory_viz.display_memory_state(memory_state)
        
        def display_memory_warning(self, warning_msg):
            """Display a memory-related warning."""
            QMessageBox.warning(self, "Memory Warning", warning_msg)
            
        def stop_generation(self):
            """Stop the generation process."""
            if hasattr(self, 'worker') and self.worker is not None:
                self.worker.stop()
                
            # Call parent method
            super().stop_generation()
        
        def closeEvent(self, event):
            """Handle application close event."""
            # Clean up memory server
            if hasattr(self, 'memory_interface') and self.memory_interface is not None:
                if hasattr(self.memory_interface, 'server_manager'):
                    self.memory_interface.server_manager.stop_server()
            
            # Call parent method
            super().closeEvent(event)
    
    return EnhancedGUI


def main():
    """Main function to launch the enhanced GUI."""
    # Import the original LLaDAGUI
    from llada_gui import LLaDAGUI
    
    # Create enhanced version
    EnhancedLLaDAGUI = enhance_llada_gui(LLaDAGUI)
    
    # Launch the application
    app = QApplication(sys.argv)
    window = EnhancedLLaDAGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
