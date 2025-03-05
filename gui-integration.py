#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced LLaDA GUI with MCP Titan Memory integration.

This extends the existing LLaDA GUI to incorporate cognitive memory capabilities
using the MCP Titan Memory system.
"""

import os
import sys
import torch
import numpy as np
import json
import requests
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup,
    QSlider, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor

# Import our modules (assume the original LLaDA GUI components)
from config import WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, SPLITTER_RATIO, DEFAULT_PARAMS
from memory_monitor import MemoryMonitor
from llada_worker import LLaDAWorker
from diffusion_visualization import DiffusionProcessVisualizer
from utils import format_memory_info

# Import our cognitive diffusion components
from cognitive_diffusion import MCPTitanMemoryInterface, CognitiveDiffusionSystem

class MemoryVisualizationWidget(QWidget):
    """Widget for visualizing memory state and influence."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Explanation label
        explanation = QLabel(
            "This visualization shows how the memory system influences the text generation process. "
            "Green highlighting indicates strong memory influence, while red indicates surprise or novelty."
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
        self.memory_state_viz.setMaximumHeight(100)
        memory_layout.addWidget(self.memory_state_viz)
        
        layout.addWidget(memory_group)
        
        # Token influence visualization
        influence_group = QGroupBox("Memory Influence on Tokens")
        influence_layout = QVBoxLayout(influence_group)
        
        self.influence_viz = QTextEdit()
        self.influence_viz.setReadOnly(True)
        influence_layout.addWidget(self.influence_viz)
        
        layout.addWidget(influence_group)
        
        # Controls for visualization
        controls_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze Memory Influence")
        controls_layout.addWidget(self.analyze_btn)
        
        self.reset_btn = QPushButton("Reset Memory")
        controls_layout.addWidget(self.reset_btn)
        
        layout.addLayout(controls_layout)
    
    def display_memory_state(self, memory_state):
        """Display the current memory state.
        
        Args:
            memory_state: Array of memory state values
        """
        # Display as a heatmap-like visualization
        memory_state = np.array(memory_state)
        
        # Normalize for visualization
        if memory_state.size > 0:
            min_val = memory_state.min()
            max_val = memory_state.max()
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
    
    def display_token_influences(self, token_influences, text):
        """Display the influence of memory on token generation.
        
        Args:
            token_influences: List of token influence data
            text: The full text
        """
        html = '<div style="font-family: monospace; line-height: 1.5;">'
        
        # Display each token with color based on memory influence
        for info in token_influences:
            token = info["token"].replace(" ", "&nbsp;")
            # Scale similarity from -1..1 to color
            sim = info["memory_similarity"]  # -1 to 1
            surprise = info["surprise"]
            
            # Use green for memory-aligned tokens (high similarity)
            # Use red for surprising tokens
            if sim > 0.5:  # Strong memory influence
                color = f"rgba(0, 128, 0, {sim})"  # Green with opacity by strength
                html += f'<span style="background-color: {color};" title="Memory influence: {sim:.2f}">{token}</span>'
            elif surprise > 0.5:  # High surprise/novelty
                color = f"rgba(255, 0, 0, {min(1.0, surprise/2)})"  # Red with opacity by strength
                html += f'<span style="background-color: {color};" title="Surprise: {surprise:.2f}">{token}</span>'
            else:
                html += f'<span title="Memory influence: {sim:.2f}, Surprise: {surprise:.2f}">{token}</span>'
        
        html += '</div>'
        self.influence_viz.setHtml(html)


class MemoryGuidedDiffusionWorker(QThread):
    """Worker thread for memory-guided diffusion generation."""
    
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    memory_update = pyqtSignal(list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    
    def __init__(self, prompt, config, memory_interface=None):
        """Initialize the worker.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            memory_interface: Optional memory interface
        """
        super().__init__()
        self.prompt = prompt
        self.config = config
        self.memory_interface = memory_interface
        self.is_running = True
        
        # Create a cognitive diffusion system if memory is enabled
        self.cognitive_system = None
        if config.get('use_memory', False) and memory_interface:
            # This would be created during actual implementation
            pass
    
    def run(self):
        """Run the generation."""
        try:
            # Check if using memory-guided generation
            if self.config.get('use_memory', False) and self.memory_interface:
                self.run_memory_guided_generation()
            else:
                self.run_standard_generation()
        except Exception as e:
            self.error.emit(str(e))
    
    def run_memory_guided_generation(self):
        """Run generation with memory guidance."""
        try:
            # Create CognitiveDiffusionSystem
            from cognitive_diffusion import CognitiveDiffusionSystem
            
            # This is a placeholder - in a real implementation, we'd need to
            # properly initialize and connect to the actual systems
            system = CognitiveDiffusionSystem(
                llada_model_path="path/to/model",
                device="cuda" if self.config['device'] == 'cuda' else "cpu"
            )
            
            # Set system parameters from config
            memory_weight = self.config.get('memory_weight', 0.3)
            
            # Run generation
            result = system.generate(
                prompt=self.prompt,
                gen_length=self.config['gen_length'],
                steps=self.config['steps'],
                block_length=self.config['block_length'],
                temperature=self.config['temperature'],
                cfg_scale=self.config['cfg_scale'],
                memory_weight=memory_weight,
                remasking=self.config['remasking']
            )
            
            # Process steps and emit signals
            for i, step_data in enumerate(result['steps']):
                if not self.is_running:
                    break
                    
                progress = int((i + 1) / self.config['steps'] * 100)
                
                # Emit progress and step update
                self.progress.emit(
                    progress, 
                    f"Step {i+1}/{self.config['steps']}", 
                    {'partial_output': system.tokenizer.decode(step_data['tokens'])}
                )
                
                self.step_update.emit(
                    i,
                    step_data['tokens'],
                    step_data['masks'],
                    step_data['confidences']
                )
                
                # Emit memory state update
                self.memory_update.emit(system.memory.get_memory_state().tolist())
                
                # Simulate step time
                if i < len(result['steps']) - 1:
                    QThread.msleep(50)  # Adjust timing as needed
            
            # Emit final result
            if self.is_running:
                self.finished.emit(result['text'])
        
        except Exception as e:
            self.error.emit(f"Memory-guided generation error: {str(e)}")
    
    def run_standard_generation(self):
        """Run standard LLaDA generation without memory."""
        # This would be the original LLaDAWorker implementation
        pass
    
    def stop(self):
        """Stop the generation."""
        self.is_running = False


class EnhancedLLaDAGUI(QMainWindow):
    """Enhanced LLaDA GUI with memory capabilities."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{WINDOW_TITLE} with Cognitive Memory")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Set up memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.update.connect(self.update_memory_info)
        
        # Worker thread reference
        self.worker = None
        
        # Memory interface
        self.memory_interface = MCPTitanMemoryInterface()
        
        # Memory importance indicator
        self.memory_importance = 0.3  # Default importance (0-1)
        
        # Initialize UI
        self.init_ui()
        
        # Display welcome message
        self.setup_welcome_message()
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        # Try to initialize memory interface
        try:
            self.initialize_memory_interface()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Memory System Warning",
                f"Could not connect to Titan Memory system: {str(e)}\n\n"
                "You can still use LLaDA without memory capabilities."
            )
    
    def initialize_memory_interface(self):
        """Initialize the memory interface."""
        # This would connect to the MCP Titan service
        pass
    
    def init_ui(self):
        """Initialize the user interface."""
        # This would be similar to the original LLaDA GUI init_ui method,
        # but with additional memory-related UI elements
        
        # For demo purposes, let's assume this adds all the basic UI from LLaDAGUI
        # and then adds our memory-specific enhancements
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create a splitter for flexible layout
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # Input area (top section) - similar to original GUI
        