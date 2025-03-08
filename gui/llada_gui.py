# -*- coding: utf-8 -*-

"""
LLaDA GUI - A graphical interface for interacting with the LLaDA language model.
"""

import os
import sys

import torch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup,
    QSizePolicy
)

# Import our modules with updated paths
from core.config import UI_SETTINGS, DEFAULT_GENERATION_PARAMS as DEFAULT_PARAMS
from core.llada_worker import LLaDAWorker
from core.utils import format_memory_info
from gui.memory_monitor import MemoryMonitor
from gui.visualizations.diffusion_visualization import DiffusionProcessVisualizer

# Import memory adapter for integration
try:
    from core.memory.memory_adapter import handle_memory_integration, add_memory_visualization_tab

    HAS_MEMORY_INTEGRATION = True
except ImportError:
    HAS_MEMORY_INTEGRATION = False

# Import memory management if available
try:
    from core.memory_management.integration import create_memory_status_widget, get_memory_manager, \
        suggest_memory_optimizations

    HAS_ADVANCED_MEMORY = True
except ImportError:
    HAS_ADVANCED_MEMORY = False

# Import performance components if available
try:
    from core.performance.integration import enable_streaming
    from gui.visualizations.performance_monitor import add_performance_tab

    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False


class LLaDAGUI(QMainWindow):
    """Main GUI for the LLaDA application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_SETTINGS['window_title'])
        self.resize(UI_SETTINGS['window_width'], UI_SETTINGS['window_height'])

        # Set minimum size to ensure essential controls remain accessible
        self.setMinimumSize(800, 600)

        # Set size policy to allow proper resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set up memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.update.connect(self.update_memory_info)

        # Worker thread reference
        self.worker = None

        # Initialize UI
        self.init_ui()

        # Initialize memory integration if available
        if HAS_MEMORY_INTEGRATION:
            add_memory_visualization_tab(self)

            # Initialize memory server if "Enable Memory Integration" is checked by default
            QTimer.singleShot(1000, self.initialize_memory_if_enabled)

        # Initialize performance monitoring if available
        if HAS_PERFORMANCE_MONITOR:
            add_performance_tab(self)
            # Enable streaming output
            enable_streaming(self)

        # Display welcome message
        self.setup_welcome_message()

        # Start memory monitoring
        self.memory_monitor.start()

    def closeEvent(self, event):
        """Properly clean up when the window is closed."""
        # Stop memory monitoring
        self.memory_monitor.stop()

        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)  # Wait for thread to finish with timeout

            # If thread is still running, terminate it
            if self.worker.isRunning():
                self.worker.terminate()

        # Stop memory server if it was started
        if HAS_MEMORY_INTEGRATION:
            try:
                from core.memory.memory_integration import get_server_manager
                server_manager = get_server_manager()
                if server_manager and server_manager.is_running():
                    print("Stopping memory server...")
                    server_manager.stop()
            except Exception as e:
                print(f"Error stopping memory server: {e}")

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Accept the close event
        event.accept()

    def init_ui(self):
        """Initialize the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Create a splitter for flexible layout
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)  # Don't allow sections to collapse
        main_layout.addWidget(splitter)

        # Input area (top section) with scroll capability
        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)  # Key for proper scrolling
        input_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        input_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_scroll.setWidget(input_widget)

        # Memory usage display with enhanced information
        self.memory_group = QGroupBox("System Resources")
        memory_layout = QGridLayout(self.memory_group)

        # System memory with progress bar
        memory_layout.addWidget(QLabel("System RAM:"), 0, 0)

        # Add progress bar for system memory
        self.system_memory_progress = QProgressBar()
        self.system_memory_progress.setRange(0, 100)
        self.system_memory_progress.setValue(0)
        memory_layout.addWidget(self.system_memory_progress, 0, 1)

        self.system_memory_label = QLabel("- / - GB (-%)")
        memory_layout.addWidget(self.system_memory_label, 0, 2)

        # GPU memory with progress bar
        memory_layout.addWidget(QLabel("GPU Memory:"), 1, 0)

        # Add progress bar for GPU memory
        self.gpu_memory_progress = QProgressBar()
        self.gpu_memory_progress.setRange(0, 100)
        self.gpu_memory_progress.setValue(0)
        memory_layout.addWidget(self.gpu_memory_progress, 1, 1)

        self.gpu_memory_label = QLabel("- / - GB (-%)")
        memory_layout.addWidget(self.gpu_memory_label, 1, 2)

        # Auto-optimize button
        if HAS_ADVANCED_MEMORY:
            self.auto_optimize_btn = QPushButton("Auto-Optimize Parameters")
            self.auto_optimize_btn.setToolTip("Automatically optimize parameters based on available memory")
            self.auto_optimize_btn.clicked.connect(self.auto_optimize_parameters)
            memory_layout.addWidget(self.auto_optimize_btn, 2, 0, 1, 3)

        input_layout.addWidget(self.memory_group)

        # Input prompt area
        input_label = QLabel("Enter your prompt:")
        input_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        input_layout.addWidget(input_label)

        self.input_text = QTextEdit()
        self.input_text.setMinimumHeight(100)
        self.input_text.setPlaceholderText("Type your prompt here...")
        input_layout.addWidget(self.input_text)

        # Parameters area
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout(params_group)

        # Parameter controls
        self.gen_length_spin = QSpinBox()
        self.gen_length_spin.setRange(16, 512)
        self.gen_length_spin.setValue(DEFAULT_PARAMS['gen_length'])
        self.gen_length_spin.setSingleStep(16)
        self.gen_length_spin.valueChanged.connect(self.on_gen_length_changed)
        params_layout.addWidget(QLabel("Generation Length:"), 0, 0)
        params_layout.addWidget(self.gen_length_spin, 0, 1)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(16, 512)
        self.steps_spin.setValue(DEFAULT_PARAMS['steps'])
        self.steps_spin.setSingleStep(16)
        params_layout.addWidget(QLabel("Sampling Steps:"), 0, 2)
        params_layout.addWidget(self.steps_spin, 0, 3)

        self.block_length_spin = QSpinBox()
        # Block length must divide gen_length evenly - set limited options
        self.block_length_spin.setRange(16, 256)
        self.block_length_spin.setValue(DEFAULT_PARAMS['block_length'])
        self.block_length_spin.setSingleStep(16)
        self.block_length_spin.valueChanged.connect(self.on_block_length_changed)
        self.block_length_spin.setToolTip("Block length must divide generation length evenly")
        params_layout.addWidget(QLabel("Block Length:"), 1, 0)
        params_layout.addWidget(self.block_length_spin, 1, 1)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 2)
        self.temperature_spin.setValue(DEFAULT_PARAMS['temperature'])
        self.temperature_spin.setSingleStep(0.1)
        params_layout.addWidget(QLabel("Temperature:"), 1, 2)
        params_layout.addWidget(self.temperature_spin, 1, 3)

        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(0, 5)
        self.cfg_scale_spin.setValue(DEFAULT_PARAMS['cfg_scale'])
        self.cfg_scale_spin.setSingleStep(0.1)
        params_layout.addWidget(QLabel("CFG Scale:"), 2, 0)
        params_layout.addWidget(self.cfg_scale_spin, 2, 1)

        self.remasking_combo = QComboBox()
        self.remasking_combo.addItems(["low_confidence", "random"])
        self.remasking_combo.setCurrentText(DEFAULT_PARAMS['remasking'])
        params_layout.addWidget(QLabel("Remasking Strategy:"), 2, 2)
        params_layout.addWidget(self.remasking_combo, 2, 3)

        # Hardware options
        device_group = QGroupBox("Hardware & Memory Options")
        device_layout = QGridLayout(device_group)

        # Device selection
        device_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_group = QButtonGroup()

        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")

        # Set default based on availability
        if torch.cuda.is_available():
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(False)

        self.device_group.addButton(self.cpu_radio, 0)
        self.device_group.addButton(self.gpu_radio, 1)

        device_layout.addWidget(self.cpu_radio, 0, 1)
        device_layout.addWidget(self.gpu_radio, 0, 2)

        # Memory optimization options
        device_layout.addWidget(QLabel("Memory Optimization:"), 1, 0)

        self.use_normal = QRadioButton("Normal Precision")
        self.use_8bit = QRadioButton("8-bit Quantization")
        self.use_4bit = QRadioButton("4-bit Quantization")

        self.precision_group = QButtonGroup()
        self.precision_group.addButton(self.use_normal, 0)
        self.precision_group.addButton(self.use_8bit, 1)
        self.precision_group.addButton(self.use_4bit, 2)

        # Set default based on available memory
        self.use_8bit.setChecked(True)  # Default to 8-bit for safety

        device_layout.addWidget(self.use_normal, 1, 1)
        device_layout.addWidget(self.use_8bit, 1, 2)
        device_layout.addWidget(self.use_4bit, 1, 3)

        # Add extreme memory optimization option
        self.extreme_mode = QCheckBox("Extreme Memory Mode (for 8-12GB GPUs)")
        self.extreme_mode.setToolTip("Enable extreme memory optimizations for GPUs with limited VRAM")
        device_layout.addWidget(self.extreme_mode, 2, 0, 1, 4)

        # Add fast mode option for better performance
        self.fast_mode = QCheckBox("Fast Mode (faster generation, slightly lower quality)")
        self.fast_mode.setToolTip("Enable optimizations for faster generation at a slight quality cost")
        device_layout.addWidget(self.fast_mode, 3, 0, 1, 4)

        # Add memory integration option
        self.memory_integration = QCheckBox("Enable Memory Integration (context-aware generation)")
        self.memory_integration.setToolTip("Enable memory integration for context-aware generation")
        self.memory_integration.toggled.connect(self.toggle_memory_integration)
        device_layout.addWidget(self.memory_integration, 4, 0, 1, 4)

        # Configure based on GPU memory
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

                # For GPUs with limited memory, automatically set appropriate options
                if gpu_memory < 8:
                    # Very limited VRAM - enable all memory saving features
                    self.extreme_mode.setChecked(True)
                    self.use_4bit.setChecked(True)
                    self.status_label.setText(
                        f"Limited GPU memory ({gpu_memory:.1f}GB). Memory-saving options enabled.")
                    QTimer.singleShot(5000, lambda: self.status_label.setText(""))
                elif gpu_memory < 12:
                    # Limited VRAM - enable extreme mode and 8-bit
                    self.extreme_mode.setChecked(True)
                    self.use_8bit.setChecked(True)
                elif gpu_memory < 16:
                    # Moderate VRAM - enable 8-bit only
                    self.use_8bit.setChecked(True)
                else:
                    # Plenty of VRAM - can use normal precision
                    self.use_normal.setChecked(True)

                # Automatically optimize parameters if advanced memory management is available
                if HAS_ADVANCED_MEMORY:
                    QTimer.singleShot(1000, self.auto_optimize_parameters)

            except Exception as e:
                print(f"Error detecting GPU memory: {e}")

        # Connect device selection changes to update quantization options
        self.cpu_radio.toggled.connect(self.update_quantization_options)
        self.gpu_radio.toggled.connect(self.update_quantization_options)

        # Add memory info and parameter groups
        params_layout.addWidget(device_group, 3, 0, 1, 4)

        # Add parameters group to input layout
        input_layout.addWidget(params_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.generate_btn.clicked.connect(self.start_generation)
        button_layout.addWidget(self.generate_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        button_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_output)
        button_layout.addWidget(self.clear_btn)

        input_layout.addLayout(button_layout)

        # Add input scroll widget to splitter
        splitter.addWidget(input_scroll)

        # Output area (bottom section) with tabs
        output_scroll = QScrollArea()
        output_scroll.setWidgetResizable(True)  # Key for proper scrolling
        output_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        output_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_scroll.setWidget(output_widget)

        output_label = QLabel("Generated Output:")
        output_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        output_layout.addWidget(output_label)

        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        output_layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        output_layout.addWidget(self.status_label)

        # Create a tab widget for showing output and visualization
        self.tab_widget = QTabWidget()

        # Text output tab
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Generated text will appear here...")
        text_layout.addWidget(self.output_text)

        self.tab_widget.addTab(text_tab, "Text Output")

        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        self.diffusion_viz = DiffusionProcessVisualizer()
        viz_layout.addWidget(self.diffusion_viz)

        self.tab_widget.addTab(viz_tab, "Diffusion Visualization")

        # Add tabs to layout
        output_layout.addWidget(self.tab_widget)

        # Add output scroll widget to splitter
        splitter.addWidget(output_scroll)

        # Set the splitter sizes
        splitter.setSizes(UI_SETTINGS['splitter_ratio'])

        # Set the central widget
        self.setCentralWidget(main_widget)

    def start_generation(self):
        """Start the generation process."""
        prompt = self.input_text.toPlainText().strip()

        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Please enter a prompt before generating.")
            return

        # Get configuration from UI
        config = self.get_generation_config()

        # Check if advanced memory management is available and can predict memory usage
        if HAS_ADVANCED_MEMORY:
            try:
                from core.memory_management.memory_manager import get_memory_manager
                memory_manager = get_memory_manager()

                # Check if the current configuration can fit in memory
                if not memory_manager.can_fit_in_memory(config):
                    # Automatic optimization
                    optimized_config = memory_manager.optimize_parameters(config)

                    # Calculate the differences for display
                    changes = []
                    if optimized_config.get('gen_length') != config.get('gen_length'):
                        changes.append(f"Generation Length: {config['gen_length']} → {optimized_config['gen_length']}")

                    if optimized_config.get('steps') != config.get('steps'):
                        changes.append(f"Sampling Steps: {config['steps']} → {optimized_config['steps']}")

                    if optimized_config.get('use_4bit') and not config.get('use_4bit'):
                        changes.append("Enable 4-bit quantization")

                    if optimized_config.get('extreme_mode') and not config.get('extreme_mode'):
                        changes.append("Enable Extreme Memory Mode")

                    # Ask the user if they want to use the optimized configuration
                    if changes:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Icon.Warning)
                        msg.setWindowTitle("Memory Optimization Required")
                        msg.setText("Your current settings may cause out-of-memory errors.")
                        msg.setInformativeText(
                            "Would you like to automatically adjust parameters to fit available memory?")
                        msg.setDetailedText("• " + "\n• ".join(changes))
                        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        msg.setDefaultButton(QMessageBox.StandardButton.Yes)

                        if msg.exec() == QMessageBox.StandardButton.Yes:
                            # Apply optimized parameters
                            config = optimized_config

                            # Update UI controls
                            if 'gen_length' in optimized_config:
                                self.gen_length_spin.setValue(optimized_config['gen_length'])

                            if 'steps' in optimized_config:
                                self.steps_spin.setValue(optimized_config['steps'])

                            if optimized_config.get('use_4bit', False):
                                self.use_4bit.setChecked(True)

                            if optimized_config.get('extreme_mode', False):
                                self.extreme_mode.setChecked(True)
            except Exception as e:
                print(f"Error performing memory check: {e}")

        # Disable input controls during generation
        self.set_controls_enabled(False)

        # Setup progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Initializing...")

        # Clear previous output
        self.output_text.clear()

        # Setup visualization for the diffusion process
        self.diffusion_viz.setup_process(config['gen_length'], config['steps'])

        # Try memory-guided generation if enabled
        memory_handled = False
        if config['use_memory'] and HAS_MEMORY_INTEGRATION:
            # Add memory visualization tab if not already present
            if not hasattr(self, 'memory_viz'):
                add_memory_visualization_tab(self)

            # Try to handle memory integration
            memory_handled = handle_memory_integration(self, prompt, config)

        # Fall back to standard generation if memory integration not handled
        if not memory_handled:
            # Create and start standard worker thread
            self.worker = LLaDAWorker(prompt, config)
            self.worker.progress.connect(self.update_progress)
            self.worker.step_update.connect(self.update_visualization)
            self.worker.finished.connect(self.generation_finished)
            self.worker.error.connect(self.generation_error)
            self.worker.memory_warning.connect(self.display_memory_warning)
            self.worker.start()

        # Enable stop button
        self.stop_btn.setEnabled(True)

        # Switch to the visualization tab (or memory tab if available) only if not disabled
        if os.environ.get("LLADA_DISABLE_TAB_SWITCHING") != "1":
            if hasattr(self, 'memory_viz') and config['use_memory']:
                memory_tab_index = self.tab_widget.indexOf(self.memory_viz)
                if memory_tab_index >= 0:
                    self.tab_widget.setCurrentIndex(memory_tab_index)
            else:
                self.tab_widget.setCurrentIndex(1)

    def stop_generation(self):
        """Stop the current generation process."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping generation...")
            self.stop_btn.setEnabled(False)

    def display_memory_warning(self, warning_msg):
        """Display a memory warning to the user."""
        QMessageBox.warning(self, "Memory Warning", warning_msg)

    def initialize_memory_if_enabled(self):
        """Initialize memory system if memory integration is enabled."""
        if not HAS_MEMORY_INTEGRATION:
            return

        # Check if memory integration is enabled
        memory_enabled = False
        if hasattr(self, 'memory_integration'):
            memory_enabled = self.memory_integration.isChecked()

        if memory_enabled:
            try:
                from core.memory.memory_integration import initialize_memory
                self.status_label.setText("Initializing memory system...")
                if initialize_memory(start_server=True):
                    self.status_label.setText("Memory system initialized")
                    QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                else:
                    self.status_label.setText("Failed to initialize memory system")
            except Exception as e:
                self.status_label.setText(f"Error initializing memory: {str(e)}")
                print(f"Error initializing memory: {str(e)}")
        else:
            print("Memory integration not enabled, skipping initialization")

    def toggle_memory_integration(self, enabled):
        """Handle toggling of the memory integration checkbox."""
        if not HAS_MEMORY_INTEGRATION:
            return

        if enabled:
            # Start memory server
            try:
                from core.memory.memory_integration import initialize_memory
                self.status_label.setText("Starting memory server...")
                if initialize_memory(start_server=True):
                    self.status_label.setText("Memory server started")
                    QTimer.singleShot(3000, lambda: self.status_label.setText(""))

                    # Show memory tab if available if not disabled
                    if os.environ.get("LLADA_DISABLE_TAB_SWITCHING") != "1" and hasattr(self, 'memory_viz'):
                        memory_tab_index = self.tab_widget.indexOf(self.memory_viz)
                        if memory_tab_index >= 0:
                            self.tab_widget.setCurrentIndex(memory_tab_index)
                else:
                    self.status_label.setText("Failed to start memory server")
                    # Uncheck the checkbox
                    self.memory_integration.blockSignals(True)
                    self.memory_integration.setChecked(False)
                    self.memory_integration.blockSignals(False)
                    QMessageBox.warning(
                        self,
                        "Memory Server Error",
                        "Failed to start memory server. Memory integration has been disabled."
                    )
            except Exception as e:
                self.status_label.setText(f"Error starting memory server: {str(e)}")
                print(f"Error starting memory server: {str(e)}")
                # Uncheck the checkbox
                self.memory_integration.blockSignals(True)
                self.memory_integration.setChecked(False)
                self.memory_integration.blockSignals(False)
        else:
            # Stop memory server
            try:
                from core.memory.memory_integration import get_server_manager
                server_manager = get_server_manager()
                if server_manager and server_manager.is_running():
                    self.status_label.setText("Stopping memory server...")
                    if server_manager.stop():
                        self.status_label.setText("Memory server stopped")
                        QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                    else:
                        self.status_label.setText("Failed to stop memory server")
            except Exception as e:
                self.status_label.setText(f"Error stopping memory server: {str(e)}")
                print(f"Error stopping memory server: {str(e)}")

    def update_progress(self, progress, status, data):
        """Update the progress bar and status."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

        # Update partial output if available
        if 'partial_output' in data and data['partial_output']:
            self.output_text.setText(data['partial_output'])
            cursor = self.output_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.output_text.setTextCursor(cursor)

    def update_visualization(self, step, tokens, masks, confidences):
        """Update the diffusion visualization."""
        self.diffusion_viz.update_process(step, tokens, masks, confidences)

    def generation_finished(self, result):
        """Handle generation completion."""
        self.output_text.setText(result)
        self.progress_bar.setValue(100)
        self.status_label.setText("Generation complete")
        self.set_controls_enabled(True)

        # Switch to the text output tab if not disabled
        if os.environ.get("LLADA_DISABLE_TAB_SWITCHING") != "1":
            self.tab_widget.setCurrentIndex(0)

        # Hide progress bar after a delay
        QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))

    def generation_error(self, error_msg):
        """Handle generation errors."""
        self.output_text.setText(f"<span style='color:red'>Error during generation:</span><br><pre>{error_msg}</pre>")
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation failed")
        self.set_controls_enabled(True)

        if "CUDA out of memory" in error_msg:
            extreme_mode_suggestion = ""
            if not self.extreme_mode.isChecked():
                extreme_mode_suggestion = "5. Enabling Extreme Memory Mode\n"

            QMessageBox.critical(
                self,
                "Memory Error",
                "CUDA ran out of memory. Please try:\n\n"
                "1. Reducing generation length\n"
                "2. Reducing sampling steps\n"
                "3. Using 4-bit quantization\n"
                "4. Switching to CPU mode\n" +
                extreme_mode_suggestion
            )
        else:
            QMessageBox.critical(self, "Generation Error", f"An error occurred during generation:\n\n{error_msg}")

    def set_controls_enabled(self, enabled):
        """Enable or disable controls during generation."""
        # Enable/disable normal controls
        self.generate_btn.setEnabled(enabled)
        self.input_text.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.gen_length_spin.setEnabled(enabled)
        self.steps_spin.setEnabled(enabled)
        self.block_length_spin.setEnabled(enabled)
        self.temperature_spin.setEnabled(enabled)
        self.cfg_scale_spin.setEnabled(enabled)
        self.remasking_combo.setEnabled(enabled)
        self.cpu_radio.setEnabled(enabled)
        self.gpu_radio.setEnabled(enabled and torch.cuda.is_available())
        self.use_normal.setEnabled(enabled)
        self.use_8bit.setEnabled(enabled and self.gpu_radio.isChecked())
        self.use_4bit.setEnabled(enabled and self.gpu_radio.isChecked())
        self.extreme_mode.setEnabled(enabled and self.gpu_radio.isChecked())
        self.memory_integration.setEnabled(enabled)

        # Update stop button - safely check if worker is running
        is_worker_running = False
        if self.worker is not None:
            try:
                is_worker_running = self.worker.isRunning()
            except:
                is_worker_running = False

        # Only enable stop button when worker is running
        self.stop_btn.setEnabled(not enabled and is_worker_running)

        # Update button text
        self.generate_btn.setText("Generate" if enabled else "Generating...")

    def clear_output(self):
        """Clear the output text area."""
        self.output_text.clear()

    def on_gen_length_changed(self, new_value):
        """Ensure generation length is compatible with block length."""
        # Get current block length
        block_length = self.block_length_spin.value()

        # Calculate if gen_length is divisible by block_length
        if new_value % block_length != 0:
            # Find the closest multiple of block_length
            adjusted_value = ((new_value + block_length // 2) // block_length) * block_length

            # Make sure it's within range
            if adjusted_value < self.gen_length_spin.minimum():
                adjusted_value = block_length  # Set to the smallest valid value
            elif adjusted_value > self.gen_length_spin.maximum():
                adjusted_value = (self.gen_length_spin.maximum() // block_length) * block_length

            # Temporarily block signals to avoid recursion
            self.gen_length_spin.blockSignals(True)
            self.gen_length_spin.setValue(adjusted_value)
            self.gen_length_spin.blockSignals(False)

            # Show info message
            self.status_label.setText(
                f"Generation length adjusted to {adjusted_value} to be compatible with block length {block_length}")
            QTimer.singleShot(3000, lambda: self.status_label.setText(""))

    def on_block_length_changed(self, new_value):
        """Ensure block length is compatible with generation length."""
        # Get current generation length
        gen_length = self.gen_length_spin.value()

        # If block_length doesn't divide gen_length evenly
        if gen_length % new_value != 0:
            # Two approaches: adjust block length or adjust gen length
            # We'll adjust gen_length to be a multiple of the new block_length
            adjusted_gen_length = ((gen_length + new_value // 2) // new_value) * new_value

            # Keep within range
            if adjusted_gen_length > self.gen_length_spin.maximum():
                adjusted_gen_length = (self.gen_length_spin.maximum() // new_value) * new_value

            # Apply the change to gen_length
            self.gen_length_spin.blockSignals(True)
            self.gen_length_spin.setValue(adjusted_gen_length)
            self.gen_length_spin.blockSignals(False)

            # Show info message
            self.status_label.setText(
                f"Generation length adjusted to {adjusted_gen_length} to be compatible with block length {new_value}")
            QTimer.singleShot(3000, lambda: self.status_label.setText(""))


    def auto_optimize_parameters(self):
        """Automatically optimize generation parameters based on available memory."""
        if not HAS_ADVANCED_MEMORY:
            QMessageBox.information(
                self,
                "Auto-Optimization Unavailable",
                "Advanced memory management is not available.\n\nUse manual optimization instead."
            )
            return

        # Get current parameters
        current_config = self.get_generation_config()

        # Check if we have an RTX 3060 or similar (10-12 GB VRAM GPU)
        has_3060_class_gpu = False
        if torch.cuda.is_available():
            try:
                device_id = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device_id)
                gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

                # Check if we have a 10-12GB GPU like RTX 3060
                if 10 <= gpu_memory <= 12 or "3060" in gpu_name:
                    has_3060_class_gpu = True

            except Exception as e:
                print(f"Error detecting GPU: {e}")

        # Get optimized parameters
        try:
            # Start with memory manager optimization
            optimized_config = suggest_memory_optimizations(current_config)

            # For RTX 3060 class GPUs, we know we can handle higher values
            if has_3060_class_gpu:
                # Ensure we don't reduce unnecessarily if memory is available
                stats = get_memory_manager().get_memory_stats()
                gpu_free = stats.get('gpu_free', 0)

                # If plenty of free memory, keep higher values
                if gpu_free > 8.0:  # 8GB+ free memory
                    # If current settings are high, keep them
                    if current_config.get('gen_length', 64) >= 128:
                        optimized_config['gen_length'] = current_config.get('gen_length', 128)
                    # Keep steps in balance with gen_length
                    optimized_config['steps'] = optimized_config.get('gen_length', 128)
                elif gpu_free > 4.0:  # 4-8GB free memory
                    # Ensure reasonable values (128 is well-suited for RTX 3060)
                    if optimized_config.get('gen_length', 64) < 128 and current_config.get('gen_length', 64) >= 128:
                        optimized_config['gen_length'] = 128
                    if optimized_config.get('steps', 64) < 128 and current_config.get('steps', 64) >= 128:
                        optimized_config['steps'] = 128

            # Always ensure steps and gen_length stay in balance for optimal results
            # This is important for the diffusion process to work correctly
            # They should generally be similar values
            if 'gen_length' in optimized_config and 'steps' in optimized_config:
                if abs(optimized_config['gen_length'] - optimized_config['steps']) > 64:
                    # Balance gen_length and steps
                    # Use the larger value as the reference if not reducing due to memory constraints
                    if has_3060_class_gpu and gpu_free > 4.0:
                        target = max(optimized_config['gen_length'], optimized_config['steps'])
                    else:
                        target = min(optimized_config['gen_length'], optimized_config['steps'])

                    # Round to nearest multiple of 16
                    target = (target // 16) * 16

                    # Apply balanced values
                    optimized_config['gen_length'] = target
                    optimized_config['steps'] = target

            # Check if changes were made
            changes = []

            # Update generation length if needed
            if optimized_config.get('gen_length', 0) != current_config.get('gen_length', 0):
                changes.append(f"Generation Length: {current_config['gen_length']} → {optimized_config['gen_length']}")
                self.gen_length_spin.setValue(optimized_config['gen_length'])

            # Update steps in tandem with generation length
            if optimized_config.get('steps', 0) != current_config.get('steps', 0):
                changes.append(f"Sampling Steps: {current_config['steps']} → {optimized_config['steps']}")
                self.steps_spin.setValue(optimized_config['steps'])

            if optimized_config.get('block_length', 0) != current_config.get('block_length', 0):
                changes.append(f"Block Length: {current_config['block_length']} → {optimized_config['block_length']}")
                self.block_length_spin.setValue(optimized_config['block_length'])

            # Check quantization changes
            if optimized_config.get('use_8bit', False) and not current_config.get('use_8bit', False):
                changes.append("Enabled 8-bit quantization")
                self.use_8bit.setChecked(True)

            if optimized_config.get('use_4bit', False) and not current_config.get('use_4bit', False):
                changes.append("Enabled 4-bit quantization")
                self.use_4bit.setChecked(True)

            # Check extreme mode
            if optimized_config.get('extreme_mode', False) and not current_config.get('extreme_mode', False):
                changes.append("Enabled Extreme Memory Mode")
                self.extreme_mode.setChecked(True)

            # Show message with changes
            if changes:
                QMessageBox.information(
                    self,
                    "Parameters Optimized",
                    "Parameters have been optimized for your hardware:\n\n• " + "\n• ".join(changes) +
                    "\n\nNote: Generation Length and Sampling Steps are kept in balance for optimal results."
                )
            else:
                print(#QMessageBox.information(
                    #self,
                    #"Parameters Already Optimal",
                    "Your current parameters are already optimal for your hardware."
                )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Auto-Optimization Error",
                f"Error optimizing parameters: {str(e)}"
            )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = LLaDAGUI()

    # Always try to add memory visualization if available
    if HAS_MEMORY_INTEGRATION:
        from core.memory.memory_adapter import add_memory_visualization_tab
        add_memory_visualization_tab(window)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
