#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI interface for LLaDA optimization.

This script provides a simple GUI for applying performance optimizations
to the LLaDA application.
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QLabel, QCheckBox,
        QPushButton, QProgressBar, QMessageBox, QTextEdit
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
except ImportError:
    print("Error: PyQt6 is required. Please install it with:")
    print("pip install PyQt6")
    sys.exit(1)

class OptimizationWorker(QThread):
    """Worker thread for running optimization process."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, gpu_opt=True, config_patch=True, worker_patch=True):
        super().__init__()
        self.gpu_opt = gpu_opt
        self.config_patch = config_patch
        self.worker_patch = worker_patch
    
    def run(self):
        """Run the optimization process."""
        try:
            # Get the optimize.py path
            script_dir = Path(__file__).parent
            optimize_script = script_dir / "optimize.py"
            
            # Build command
            cmd = [sys.executable, str(optimize_script)]
            
            if not self.gpu_opt:
                cmd.append("--no-gpu-opt")
            
            if not self.config_patch:
                cmd.append("--no-config-patch")
            
            if not self.worker_patch:
                cmd.append("--no-worker-patch")
            
            # Start process
            self.progress.emit(f"Running: {' '.join(cmd)}")
            
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Read output line by line
            for line in process.stdout:
                self.progress.emit(line.strip())
            
            # Wait for process to complete
            process.wait()
            
            # Check result
            if process.returncode == 0:
                self.finished.emit(True, "Optimization completed successfully!")
            else:
                self.finished.emit(False, f"Optimization failed with return code: {process.returncode}")
        
        except Exception as e:
            self.finished.emit(False, f"Error during optimization: {str(e)}")


class OptimizationDialog(QDialog):
    """Dialog for applying LLaDA optimizations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLaDA Optimization")
        self.resize(500, 400)
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("LLaDA Performance Optimization")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title_label)
        
        description = QLabel(
            "Apply performance optimizations to improve LLaDA's speed "
            "and reduce memory usage."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Optimization options
        self.gpu_opt_cb = QCheckBox("GPU Memory Optimizations")
        self.gpu_opt_cb.setChecked(True)
        self.gpu_opt_cb.setToolTip("Apply optimizations for more efficient GPU memory usage")
        layout.addWidget(self.gpu_opt_cb)
        
        self.config_patch_cb = QCheckBox("Patch Configuration")
        self.config_patch_cb.setChecked(True)
        self.config_patch_cb.setToolTip("Add optimization settings to config.py")
        layout.addWidget(self.config_patch_cb)
        
        self.worker_patch_cb = QCheckBox("Patch Worker Code")
        self.worker_patch_cb.setChecked(True)
        self.worker_patch_cb.setToolTip("Add optimization code to llada_worker.py")
        layout.addWidget(self.worker_patch_cb)
        
        # Log output
        log_label = QLabel("Optimization Log:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.optimize_btn = QPushButton("Apply Optimizations")
        self.optimize_btn.clicked.connect(self.start_optimization)
        button_layout.addWidget(self.optimize_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def log(self, message):
        """Add a message to the log."""
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def start_optimization(self):
        """Start the optimization process."""
        # Check if any optimizations are selected
        if not any([
            self.gpu_opt_cb.isChecked(),
            self.config_patch_cb.isChecked(),
            self.worker_patch_cb.isChecked()
        ]):
            QMessageBox.warning(
                self,
                "No Optimizations Selected",
                "Please select at least one optimization to apply."
            )
            return
        
        # Update UI
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log("Starting optimization process...")
        
        # Create and start worker thread
        self.worker = OptimizationWorker(
            gpu_opt=self.gpu_opt_cb.isChecked(),
            config_patch=self.config_patch_cb.isChecked(),
            worker_patch=self.worker_patch_cb.isChecked()
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.optimization_finished)
        self.worker.start()
    
    def optimization_finished(self, success, message):
        """Handle completion of the optimization process."""
        # Update UI
        self.progress_bar.setVisible(False)
        self.optimize_btn.setEnabled(True)
        
        # Log result
        self.log(message)
        
        # Show message box
        if success:
            QMessageBox.information(
                self,
                "Optimization Complete",
                "Performance optimizations have been applied successfully!\n\n"
                "Please restart the LLaDA GUI application to apply the changes."
            )
        else:
            QMessageBox.critical(
                self,
                "Optimization Failed",
                f"Failed to apply some optimizations.\n\n{message}\n\n"
                "Check the log for details."
            )


def main():
    """Run the optimization GUI."""
    app = QApplication(sys.argv)
    dialog = OptimizationDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
