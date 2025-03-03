#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI for extreme memory optimizations for LLaDA.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llada_extreme_optimizer_gui")

# Check if PyQt6 is available
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QCheckBox, QGroupBox, QMessageBox,
        QProgressBar, QTextEdit, QSplitter
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
except ImportError:
    print("Error: PyQt6 is required for the GUI")
    print("Please install it with: pip install PyQt6")
    sys.exit(1)

class OptimizationWorker(QThread):
    """Worker thread for running optimizations."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, options=None, restore=False):
        super().__init__()
        self.options = options or {}
        self.restore = restore
    
    def run(self):
        """Run the optimization process."""
        try:
            # Get the path to the optimizer script
            script_dir = Path(__file__).parent
            optimizer_script = script_dir / "extreme_optimizer.py"
            
            # Build command
            cmd = [sys.executable, str(optimizer_script)]
            
            if self.restore:
                cmd.append("--restore")
            else:
                # Add options as command line arguments
                if self.options.get('apply_all', False):
                    cmd.append("--apply-all")
                if self.options.get('create_files', False):
                    cmd.append("--create-files")
                if self.options.get('modify_worker', False):
                    cmd.append("--modify-worker")
                if self.options.get('modify_gui', False):
                    cmd.append("--modify-gui")
            
            # Run the optimizer script
            self.progress.emit(f"Running command: {' '.join(cmd)}")
            
            # Start the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Read output line by line
            for line in process.stdout:
                self.progress.emit(line.strip())
            
            # Wait for process to finish
            process.wait()
            
            # Check result
            if process.returncode == 0:
                self.finished.emit(True, "Optimization completed successfully!")
            else:
                self.finished.emit(False, f"Optimization failed with return code: {process.returncode}")
        
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(False, f"Optimization failed: {str(e)}")

class ExtremeOptimizerGUI(QMainWindow):
    """GUI for extreme memory optimization."""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("LLaDA Extreme Memory Optimizer")
        self.setMinimumSize(800, 600)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Title
        title_label = QLabel("LLaDA Extreme Memory Optimizer")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title_label)
        
        # Description
        description = QLabel(
            "This tool applies extreme memory optimizations to allow LLaDA to run on GPUs with limited VRAM (8-12GB).\n"
            "These optimizations significantly reduce memory usage through aggressive techniques like model pruning, "
            "progressive loading, and memory-efficient generation."
        )
        description.setWordWrap(True)
        main_layout.addWidget(description)
        
        # GPU Memory Information
        memory_info = self.get_gpu_memory_info()
        memory_label = QLabel(memory_info)
        memory_label.setWordWrap(True)
        memory_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        main_layout.addWidget(memory_label)
        
        # Optimization options
        options_group = QGroupBox("Optimization Options")
        options_layout = QVBoxLayout(options_group)
        
        self.apply_all_cb = QCheckBox("Apply All Optimizations (Recommended)")
        self.apply_all_cb.setChecked(True)
        self.apply_all_cb.toggled.connect(self.toggle_individual_options)
        options_layout.addWidget(self.apply_all_cb)
        
        # Individual options
        self.create_files_cb = QCheckBox("Create Optimization Files")
        self.create_files_cb.setEnabled(False)
        options_layout.addWidget(self.create_files_cb)
        
        self.modify_worker_cb = QCheckBox("Modify Worker File")
        self.modify_worker_cb.setEnabled(False)
        options_layout.addWidget(self.modify_worker_cb)
        
        self.modify_gui_cb = QCheckBox("Modify GUI File")
        self.modify_gui_cb.setEnabled(False)
        options_layout.addWidget(self.modify_gui_cb)
        
        main_layout.addWidget(options_group)
        
        # Log area
        log_label = QLabel("Optimization Log:")
        main_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #f0f0f0;")
        main_layout.addWidget(self.log_text, 1)  # Give the log area extra space
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Optimizations")
        self.apply_btn.clicked.connect(self.apply_optimizations)
        button_layout.addWidget(self.apply_btn)
        
        self.restore_btn = QPushButton("Restore Original Files")
        self.restore_btn.clicked.connect(self.restore_original)
        button_layout.addWidget(self.restore_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(button_layout)
        
        # Set the central widget
        self.setCentralWidget(main_widget)
        
        # Add initial log message
        self.log_text.append("Welcome to LLaDA Extreme Memory Optimizer")
        self.log_text.append("This tool will help you run LLaDA on your GPU with limited VRAM")
        self.log_text.append("\nReady to apply optimizations. Click 'Apply Optimizations' to start.")
    
    def get_gpu_memory_info(self):
        """Get GPU memory information."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return "No CUDA-capable GPU detected. These optimizations are primarily for NVIDIA GPUs."
            
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                gpu_info.append(f"GPU {i}: {props.name} - {total_memory:.2f} GB VRAM")
            
            if gpu_count == 0:
                return "No CUDA-capable GPU detected. These optimizations are primarily for NVIDIA GPUs."
            
            # Check if optimizations are needed
            need_extreme = any(props.total_memory / (1024**3) < 16 for i in range(gpu_count) for props in [torch.cuda.get_device_properties(i)])
            
            if need_extreme:
                recommendation = "\n\nYour GPU has limited VRAM. Extreme optimizations are RECOMMENDED."
            else:
                recommendation = "\n\nYour GPU has sufficient VRAM. Extreme optimizations are optional but can still improve performance."
            
            return "GPU Information:\n" + "\n".join(gpu_info) + recommendation
        except:
            return "Could not detect GPU information. Please ensure you have a CUDA-capable GPU and the necessary drivers installed."
    
    def toggle_individual_options(self, checked):
        """Toggle individual option checkboxes based on 'Apply All' state."""
        self.create_files_cb.setEnabled(not checked)
        self.modify_worker_cb.setEnabled(not checked)
        self.modify_gui_cb.setEnabled(not checked)
        
        if not checked:
            # Enable individual options
            self.create_files_cb.setChecked(True)
            self.modify_worker_cb.setChecked(True)
            self.modify_gui_cb.setChecked(True)
    
    def apply_optimizations(self):
        """Apply the selected optimizations."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Optimization",
            "This will modify your LLaDA GUI files to enable extreme memory optimizations.\n\n"
            "Backups will be created, but it's still recommended to have your own backups.\n\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Get selected options
        options = {
            'apply_all': self.apply_all_cb.isChecked(),
            'create_files': self.create_files_cb.isChecked(),
            'modify_worker': self.modify_worker_cb.isChecked(),
            'modify_gui': self.modify_gui_cb.isChecked(),
        }
        
        # Clear log
        self.log_text.clear()
        self.log_text.append("Starting optimization process...")
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.toggle_buttons(False)
        
        # Start worker thread
        self.worker = OptimizationWorker(options)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.optimization_finished)
        self.worker.start()
    
    def restore_original(self):
        """Restore original files from backups."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Restore",
            "This will restore the original LLaDA GUI files from backups.\n\n"
            "Any optimization changes will be lost.\n\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear log
        self.log_text.clear()
        self.log_text.append("Restoring original files...")
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.toggle_buttons(False)
        
        # Start worker thread
        self.worker = OptimizationWorker(restore=True)
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.optimization_finished)
        self.worker.start()
    
    def update_log(self, message):
        """Update the log with a new message."""
        self.log_text.append(message)
        
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def optimization_finished(self, success, message):
        """Handle optimization process completion."""
        # Update UI
        self.progress_bar.setVisible(False)
        self.toggle_buttons(True)
        
        # Add final message to log
        self.update_log("\n" + message)
        
        # Show message box
        if success:
            QMessageBox.information(
                self,
                "Optimization Complete",
                message + "\n\nYou can now run LLaDA GUI with extreme memory optimizations."
            )
        else:
            QMessageBox.critical(
                self,
                "Optimization Failed",
                message + "\n\nPlease check the log for details."
            )
    
    def toggle_buttons(self, enabled):
        """Enable or disable buttons during processing."""
        self.apply_btn.setEnabled(enabled)
        self.restore_btn.setEnabled(enabled)
        self.close_btn.setEnabled(enabled)

def main():
    """Main function."""
    app = QApplication(sys.argv)
    window = ExtremeOptimizerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
