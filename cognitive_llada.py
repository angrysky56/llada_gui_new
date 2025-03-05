#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cognitive LLaDA - Memory-Augmented Diffusion Language Model GUI

This module integrates the LLaDA diffusion model with the Titan Memory system
to create a cognitive language model with memory capabilities.
"""

import os
import sys
import torch
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup,
    QSlider, QLineEdit, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor

# Import LLaDA GUI components
from llada_gui import LLaDAGUI
from config import WINDOW_TITLE, DEFAULT_PARAMS
from memory_monitor import MemoryMonitor
from llada_worker import LLaDAWorker
from diffusion_visualization import DiffusionProcessVisualizer

# Import memory components
from titan_memory import TitanMemorySystem, TitanMemoryConfig
from memory_embeddings import EmbeddingAugmentedMemory

# Directory for memory data
MEMORY_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
os.makedirs(MEMORY_DATA_DIR, exist_ok=True)


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
            "This visualization shows the memory state during diffusion generation. "
            "The memory system provides guidance based on learned patterns in previous generations, "
            "helping ensure consistency and coherence."
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
        
        # Memory training controls
        training_layout = QHBoxLayout()
        training_layout.addWidget(QLabel("Training:"))
        
        self.train_memory = QCheckBox("Train memory on generations")
        self.train_memory.setChecked(True)
        self.train_memory.setToolTip("When enabled, the memory system will learn from generations")
        training_layout.addWidget(self.train_memory)
        
        memory_layout.addLayout(training_layout)
        
        # Connect slider to update label
        self.memory_slider.valueChanged.connect(self.update_memory_influence)
        
        layout.addWidget(memory_group)
        
        # Memory statistics
        stats_group = QGroupBox("Memory Statistics")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("Tokens Processed:"), 0, 0)
        self.tokens_processed = QLabel("0")
        stats_layout.addWidget(self.tokens_processed, 0, 1)
        
        stats_layout.addWidget(QLabel("Average Surprise:"), 0, 2)
        self.avg_surprise = QLabel("0.00")
        stats_layout.addWidget(self.avg_surprise, 0, 3)
        
        stats_layout.addWidget(QLabel("Training Loss:"), 1, 0)
        self.training_loss = QLabel("0.00")
        stats_layout.addWidget(self.training_loss, 1, 1)
        
        stats_layout.addWidget(QLabel("Memory Capacity:"), 1, 2)
        self.memory_capacity = QLabel("0%")
        stats_layout.addWidget(self.memory_capacity, 1, 3)
        
        layout.addWidget(stats_group)
        
        # Memory controls
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset Memory")
        self.reset_btn.setToolTip("Reset the memory state to zeros")
        controls_layout.addWidget(self.reset_btn)
        
        self.save_btn = QPushButton("Save Memory")
        self.save_btn.setToolTip("Save the current memory state to a file")
        controls_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Memory")
        self.load_btn.setToolTip("Load a previously saved memory state")
        controls_layout.addWidget(self.load_btn)
        
        layout.addLayout(controls_layout)
    
    def update_memory_influence(self, value):
        """Update the memory influence display and value."""
        self.memory_percent.setText(f"{value}%")
    
    def get_memory_influence(self):
        """Get the current memory influence value (0-1)."""
        return self.memory_slider.value() / 100.0
    
    def is_training_enabled(self):
        """Check if memory training is enabled."""
        return self.train_memory.isChecked()
    
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
    
    def update_stats(self, tokens_processed, avg_surprise, training_loss):
        """Update memory statistics display.
        
        Args:
            tokens_processed: Number of tokens processed
            avg_surprise: Average surprise value
            training_loss: Training loss value
        """
        self.tokens_processed.setText(str(tokens_processed))
        self.avg_surprise.setText(f"{avg_surprise:.4f}")
        self.training_loss.setText(f"{training_loss:.4f}")
        
        # Estimate memory capacity (simple heuristic based on tokens processed)
        if tokens_processed == 0:
            capacity = 0
        else:
            # Logarithmic scale for capacity
            capacity = min(100, int(20 * np.log2(1 + tokens_processed / 100)))
        
        self.memory_capacity.setText(f"{capacity}%")


class MemoryGuidedDiffusionWorker(QThread):
    """Worker thread for memory-guided diffusion generation."""
    
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    memory_update = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(int, float, float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, 
                prompt, 
                config, 
                memory_system,
                embedding_memory,
                original_worker_class):
        """Initialize the worker.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            memory_system: Titan Memory System instance
            embedding_memory: Embedding Augmented Memory instance
            original_worker_class: Original LLaDAWorker class for base functionality
        """
        super().__init__()
        self.prompt = prompt
        self.config = config
        self.memory_system = memory_system
        self.embedding_memory = embedding_memory
        self.original_worker_class = original_worker_class
        self.memory_weight = config.get('memory_weight', 0.3)
        self.train_memory = config.get('train_memory', True)
        self.is_running = True
        
        # Statistics tracking
        self.tokens_processed = 0
        self.surprise_values = []
        self.training_losses = []
        
        # Create the original worker (but don't start it)
        self.original_worker = original_worker_class(prompt, config)
        
        # Connect signals
        self.original_worker.progress.connect(self.handle_progress)
        self.original_worker.step_update.connect(self.handle_step_update)
        self.original_worker.finished.connect(self.handle_finished)
        self.original_worker.error.connect(self.handle_error)
    
    def run(self):
        """Run the memory-guided generation."""
        try:
            # Start the original worker in this thread
            # We're not actually starting it as a thread, just running its logic
            self.original_worker.run()
        except Exception as e:
            self.error.emit(f"Memory-guided generation error: {str(e)}")
    
    def handle_progress(self, progress, status, data):
        """Handle progress updates from the original worker."""
        if 'partial_output' in data:
            # This is where we would inject memory guidance in a full implementation
            # For now, we'll just pass through the data
            pass
            
        # Pass through the progress signal
        self.progress.emit(progress, status, data)
    
    def handle_step_update(self, step, tokens, masks, confidences):
        """Handle step updates from the original worker and add memory guidance."""
        # In a full implementation, this would be where we:
        # 1. Get the current token sequence
        # 2. Encode it with the embedding system
        # 3. Update the memory state
        # 4. Use memory predictions to influence the next step
        
        # For demo purposes, update memory with sample data
        if not tokens:
            self.step_update.emit(step, tokens, masks, confidences)
            return
            
        # Simulate memory update (in a real implementation, use actual tokens)
        if self.memory_system:
            # Convert tokens to memory-compatible vectors
            # In a full implementation, use actual tokens from the model
            token_idx = len(tokens) - 1
            if token_idx >= 0:
                try:
                    # Encode the token sequence at the current position
                    context_vector = self.embedding_memory.encode_tokens(tokens, token_idx)
                    
                    # Update memory and get surprise
                    _, surprise = self.memory_system.update_memory(context_vector)
                    
                    # If training is enabled, train on this step
                    if self.train_memory and token_idx > 0:
                        prev_vector = self.embedding_memory.encode_tokens(tokens, token_idx - 1)
                        curr_vector = context_vector
                        loss = self.memory_system.train_step(prev_vector, curr_vector)
                        self.training_losses.append(loss)
                    
                    # Update statistics
                    self.tokens_processed += 1
                    self.surprise_values.append(surprise)
                    
                    # Emit memory state update
                    memory_state = self.memory_system.get_memory_state()
                    if isinstance(memory_state, list):
                        memory_state = np.array(memory_state)
                    self.memory_update.emit(memory_state)
                    
                    # Emit statistics update
                    avg_surprise = np.mean(self.surprise_values) if self.surprise_values else 0.0
                    avg_loss = np.mean(self.training_losses) if self.training_losses else 0.0
                    self.stats_update.emit(self.tokens_processed, avg_surprise, avg_loss)
                    
                except Exception as e:
                    print(f"Memory update error: {str(e)}")
        
        # Pass through the step update
        self.step_update.emit(step, tokens, masks, confidences)
    
    def handle_finished(self, result):
        """Handle completion of generation."""
        # In a full implementation, we might do some final memory updates here
        # For now, just pass through the result
        
        # Pass through the finished signal
        self.finished.emit(result)
    
    def handle_error(self, error_msg):
        """Handle errors from the original worker."""
        # Pass through the error signal
        self.error.emit(error_msg)
    
    def stop(self):
        """Stop the generation."""
        self.is_running = False
        self.original_worker.stop()


class CognitiveLLaDAGUI(LLaDAGUI):
    """Enhanced LLaDA GUI with cognitive memory capabilities."""
    
    def __init__(self):
        """Initialize the cognitive LLaDA GUI."""
        # Initialize memory systems first (before parent class)
        self.memory_system = None
        self.embedding_memory = None
        try:
            self.init_memory_systems()
        except Exception as e:
            print(f"Failed to initialize memory systems: {str(e)}")
        
        # Call parent constructor
        super().__init__()
        
        # Modify window title
        self.setWindowTitle(f"{WINDOW_TITLE} with Cognitive Memory")
        
        # Add memory visualization tab
        self.init_memory_ui()
    
    def init_memory_systems(self):
        """Initialize the memory and embedding systems."""
        # Create memory config
        memory_config = TitanMemoryConfig(
            input_dim=64,
            hidden_dim=32,
            memory_dim=64,
            learning_rate=0.001
        )
        
        # Create memory system
        self.memory_system = TitanMemorySystem(memory_config)
        
        # Estimate vocabulary size (in a real implementation, get this from the tokenizer)
        vocab_size = 50000
        
        # Create embedding memory system
        self.embedding_memory = EmbeddingAugmentedMemory(
            vocab_size=vocab_size,
            embedding_dim=64,
            memory_dim=64,
            context_size=3
        )
    
    def init_memory_ui(self):
        """Initialize memory-specific UI elements."""
        # Add memory tab to the output tab widget
        self.memory_viz = MemoryVisualizationWidget()
        self.tab_widget.addTab(self.memory_viz, "Memory")
        
        # Connect memory control buttons
        self.memory_viz.reset_btn.clicked.connect(self.reset_memory)
        self.memory_viz.save_btn.clicked.connect(self.save_memory)
        self.memory_viz.load_btn.clicked.connect(self.load_memory)
        
        # Add memory toggle to parameters group
        for child in self.findChildren(QGroupBox):
            if child.title() == "Generation Parameters":
                # Add to the parameters grid layout
                params_layout = child.layout()
                if params_layout and isinstance(params_layout, QGridLayout):
                    # Get the row count and add our memory controls
                    row = params_layout.rowCount()
                    
                    # Create memory checkbox
                    self.use_memory = QCheckBox("Use Memory Guidance")
                    self.use_memory.setChecked(True)
                    self.use_memory.setToolTip("Enable cognitive memory for guided diffusion")
                    
                    # Add to layout
                    params_layout.addWidget(QLabel("Memory:"), row, 0)
                    params_layout.addWidget(self.use_memory, row, 1, 1, 3)
    
    def reset_memory(self):
        """Reset the memory state."""
        if self.memory_system:
            self.memory_system.reset_memory()
            
            # Update visualization
            self.memory_viz.display_memory_state(
                self.memory_system.get_memory_state()
            )
            
            # Update statistics
            self.memory_viz.update_stats(0, 0.0, 0.0)
            
            QMessageBox.information(
                self,
                "Memory Reset",
                "Memory state has been reset to zeros."
            )
    
    def save_memory(self):
        """Save the current memory state."""
        if not self.memory_system:
            QMessageBox.warning(
                self,
                "Memory Not Available",
                "Memory system is not initialized."
            )
            return
            
        # Save memory model
        try:
            save_path = os.path.join(MEMORY_DATA_DIR, "memory_state.json")
            self.memory_system.save_model(save_path)
            
            QMessageBox.information(
                self,
                "Memory Saved",
                f"Memory state saved to:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save memory state: {str(e)}"
            )
    
    def load_memory(self):
        """Load a previously saved memory state."""
        if not self.memory_system:
            QMessageBox.warning(
                self,
                "Memory Not Available",
                "Memory system is not initialized."
            )
            return
            
        # Load memory model
        try:
            load_path = os.path.join(MEMORY_DATA_DIR, "memory_state.json")
            
            if not os.path.exists(load_path):
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"No saved memory state found at:\n{load_path}"
                )
                return
                
            self.memory_system.load_model(load_path)
            
            # Update visualization
            self.memory_viz.display_memory_state(
                self.memory_system.get_memory_state()
            )
            
            QMessageBox.information(
                self,
                "Memory Loaded",
                f"Memory state loaded from:\n{load_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load memory state: {str(e)}"
            )
    
    def get_generation_config(self):
        """Get generation configuration with memory settings."""
        # Get base configuration from parent class
        config = super().get_generation_config()
        
        # Add memory settings if available and enabled
        if hasattr(self, 'use_memory') and self.use_memory.isChecked():
            config['use_memory'] = True
            
            # Get memory weight from slider
            if hasattr(self, 'memory_viz'):
                config['memory_weight'] = self.memory_viz.get_memory_influence()
                config['train_memory'] = self.memory_viz.is_training_enabled()
        else:
            config['use_memory'] = False
        
        return config
    
    def start_generation(self):
        """Start the generation process with memory support."""
        prompt = self.input_text.toPlainText().strip()
        
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt before generating.")
            return
        
        # Get configuration from UI
        config = self.get_generation_config()
        
        # Check if using memory
        is_using_memory = config.get('use_memory', False) and self.memory_system is not None
        
        # Disable input controls during generation
        self.set_controls_enabled(False)
        
        # Setup progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        status_prefix = "Memory-guided " if is_using_memory else ""
        self.status_label.setText(f"{status_prefix}Initializing...")
        
        # Clear previous output
        self.output_text.clear()
        
        # Setup visualization for the diffusion process
        self.diffusion_viz.setup_process(config['gen_length'], config['steps'])
        
        # Create and start worker thread
        if is_using_memory:
            # Create memory-guided worker
            self.worker = MemoryGuidedDiffusionWorker(
                prompt, 
                config, 
                self.memory_system,
                self.embedding_memory,
                LLaDAWorker  # Original worker class
            )
            
            # Connect memory-specific signals
            self.worker.memory_update.connect(self.memory_viz.display_memory_state)
            self.worker.stats_update.connect(self.memory_viz.update_stats)
        else:
            # Use standard worker
            self.worker = LLaDAWorker(prompt, config)
        
        # Connect common signals
        self.worker.progress.connect(self.update_progress)
        self.worker.step_update.connect(self.update_visualization)
        self.worker.finished.connect(self.generation_finished)
        self.worker.error.connect(self.generation_error)
        
        # Start the worker
        self.worker.start()
        
        # Enable stop button
        self.stop_btn.setEnabled(True)
        
        # Switch to the appropriate tab
        if is_using_memory:
            # Show memory tab
            self.tab_widget.setCurrentIndex(2)  # Memory tab
        else:
            # Show visualization tab
            self.tab_widget.setCurrentIndex(1)  # Visualization tab


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = CognitiveLLaDAGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
