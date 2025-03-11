#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX management dialog for LLaDA GUI.

This module provides a dialog for managing ONNX models in the LLaDA GUI application.
"""

import logging
import os

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QGroupBox, QListWidget, QListWidgetItem,
    QProgressBar,
    QTabWidget, QWidget, QMessageBox, QFileDialog, QLineEdit, QFormLayout,
    QDialogButtonBox
)

# Import ONNX integration
from onnx_integration import ONNXModelManager

logger = logging.getLogger("llada_onnx_dialog")


class ConversionWorker(QThread):
    """Worker thread for ONNX model conversion."""

    progress = pyqtSignal(int, str, dict)
    finished = pyqtSignal(bool, str)

    def __init__(
            self,
            model_manager: ONNXModelManager,
            model_name: str,
            model_path: str,
            optimize: bool = True,
            quantize: bool = False,
            use_gpu: bool = True
    ):
        """
        Initialize the conversion worker.
        
        Args:
            model_manager: ONNXModelManager instance
            model_name: Name for the converted model
            model_path: Path to the model to convert
            optimize: Whether to optimize the model
            quantize: Whether to quantize the model
            use_gpu: Whether to use GPU for conversion
        """
        super().__init__()
        self.model_manager = model_manager
        self.model_name = model_name
        self.model_path = model_path
        self.optimize = optimize
        self.quantize = quantize
        self.use_gpu = use_gpu

        self.stopped = False

    def run(self):
        """Run the conversion process."""
        try:
            # Convert model
            success = self.model_manager.convert_model(
                model_name=self.model_name,
                model_path=self.model_path,
                optimize=self.optimize,
                quantize=self.quantize,
                use_gpu=self.use_gpu,
                progress_callback=self.progress.emit
            )

            # Emit finished signal
            if success:
                self.finished.emit(True, "Conversion completed successfully")
            else:
                self.finished.emit(False, "Conversion failed - see logs for details")

        except Exception as e:
            import traceback
            error_msg = f"Error during conversion: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.finished.emit(False, error_msg)

    def stop(self):
        """Stop the conversion process."""
        self.stopped = True


class ONNXModelDialog(QDialog):
    """Dialog for managing ONNX models."""

    model_selected = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("ONNX Model Management")
        self.resize(700, 500)

        # Create model manager
        self.model_manager = ONNXModelManager()

        # Conversion worker
        self.conversion_worker = None

        # Initialize UI
        self.init_ui()

        # Refresh model list
        self.refresh_model_list()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Model selection tab
        select_tab = QWidget()
        select_layout = QVBoxLayout(select_tab)

        # List of available models
        select_layout.addWidget(QLabel("Available ONNX Models:"))

        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(200)
        self.model_list.itemSelectionChanged.connect(self.update_model_details)
        select_layout.addWidget(self.model_list)

        # Model details
        self.model_details = QLabel("Select a model to view details")
        select_layout.addWidget(self.model_details)

        # Options for using the model
        options_group = QGroupBox("Model Usage Options")
        options_layout = QFormLayout(options_group)

        self.use_quantized_cb = QCheckBox("Use quantized version (if available)")
        self.use_gpu_cb = QCheckBox("Use GPU for inference")
        self.use_gpu_cb.setChecked(True)

        options_layout.addRow("", self.use_quantized_cb)
        options_layout.addRow("", self.use_gpu_cb)

        select_layout.addWidget(options_group)

        # Buttons for model selection
        btn_layout = QHBoxLayout()

        self.select_btn = QPushButton("Select Model")
        self.select_btn.setEnabled(False)
        self.select_btn.clicked.connect(self.select_model)
        btn_layout.addWidget(self.select_btn)

        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_model_list)
        btn_layout.addWidget(self.refresh_btn)

        select_layout.addLayout(btn_layout)

        # Add selection tab to tab widget
        self.tab_widget.addTab(select_tab, "Select Model")

        # Conversion tab
        convert_tab = QWidget()
        convert_layout = QVBoxLayout(convert_tab)

        # Form for conversion options
        form_layout = QFormLayout()

        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Name for the converted model")
        form_layout.addRow("Model Name:", self.model_name_edit)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to the model to convert (leave empty for default)")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(browse_btn)
        form_layout.addRow("Model Path:", path_layout)

        self.optimize_cb = QCheckBox("Optimize model for inference")
        self.optimize_cb.setChecked(True)
        form_layout.addRow("", self.optimize_cb)

        self.quantize_cb = QCheckBox("Quantize model to INT8 (reduces memory usage)")
        form_layout.addRow("", self.quantize_cb)

        self.use_gpu_convert_cb = QCheckBox("Use GPU for conversion")
        self.use_gpu_convert_cb.setChecked(True)
        form_layout.addRow("", self.use_gpu_convert_cb)

        convert_layout.addLayout(form_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        convert_layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        convert_layout.addWidget(self.status_label)

        # Conversion buttons
        convert_btn_layout = QHBoxLayout()

        self.convert_btn = QPushButton("Convert Model")
        self.convert_btn.clicked.connect(self.start_conversion)
        convert_btn_layout.addWidget(self.convert_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        convert_btn_layout.addWidget(self.cancel_btn)

        convert_layout.addLayout(convert_btn_layout)

        # Add conversion tab to tab widget
        self.tab_widget.addTab(convert_tab, "Convert Model")

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def refresh_model_list(self):
        """Refresh the list of available models."""
        self.model_list.clear()

        models = self.model_manager.list_available_models()

        if not models:
            item = QListWidgetItem("No ONNX models available")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(item)
            self.select_btn.setEnabled(False)
            return

        for model in models:
            item = QListWidgetItem(model["name"])
            item.setData(Qt.ItemDataRole.UserRole, model)
            self.model_list.addItem(item)

    def update_model_details(self):
        """Update the model details display."""
        selected_items = self.model_list.selectedItems()

        if not selected_items:
            self.model_details.setText("Select a model to view details")
            self.select_btn.setEnabled(False)
            return

        item = selected_items[0]
        model_data = item.data(Qt.ItemDataRole.UserRole)

        if not model_data:
            self.model_details.setText("No details available")
            self.select_btn.setEnabled(False)
            return

        # Format model information
        details = f"<b>Model:</b> {model_data['name']}<br>"
        details += f"<b>Path:</b> {model_data['path']}<br>"

        # Format available model variants
        details += "<b>Available variants:</b><br>"
        for key, path in model_data["model_paths"]:
            variant_name = {
                "onnx_model_path": "Base ONNX",
                "optimized_model_path": "Optimized",
                "quantized_model_path": "Quantized (INT8)"
            }.get(key, key)

            details += f"- {variant_name}<br>"

        self.model_details.setText(details)
        self.select_btn.setEnabled(True)

    def select_model(self):
        """Select the current model for use."""
        selected_items = self.model_list.selectedItems()

        if not selected_items:
            return

        item = selected_items[0]
        model_data = item.data(Qt.ItemDataRole.UserRole)

        if not model_data:
            return

        # Emit selected model
        self.model_selected.emit(
            model_data["name"],
            self.use_quantized_cb.isChecked()
        )

        # Close dialog
        self.accept()

    def browse_model_path(self):
        """Open a file dialog to select a model path."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            os.path.expanduser("~")
        )

        if path:
            self.model_path_edit.setText(path)

    def start_conversion(self):
        """Start the model conversion process."""
        # Get conversion parameters
        model_name = self.model_name_edit.text().strip()
        model_path = self.model_path_edit.text().strip() or None
        optimize = self.optimize_cb.isChecked()
        quantize = self.quantize_cb.isChecked()
        use_gpu = self.use_gpu_convert_cb.isChecked()

        # Validate model name
        if not model_name:
            QMessageBox.warning(
                self,
                "Missing Model Name",
                "Please enter a name for the converted model."
            )
            return

        # Check if model already exists
        existing_models = self.model_manager.list_available_models()
        if any(model["name"] == model_name for model in existing_models):
            result = QMessageBox.question(
                self,
                "Model Already Exists",
                f"A model named '{model_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if result != QMessageBox.StandardButton.Yes:
                return

        # Update UI
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Starting conversion...")

        # Disable controls during conversion
        self.set_conversion_controls(False)

        # Start conversion worker
        self.conversion_worker = ConversionWorker(
            model_manager=self.model_manager,
            model_name=model_name,
            model_path=model_path,
            optimize=optimize,
            quantize=quantize,
            use_gpu=use_gpu
        )

        self.conversion_worker.progress.connect(self.update_conversion_progress)
        self.conversion_worker.finished.connect(self.conversion_finished)
        self.conversion_worker.start()

    def cancel_conversion(self):
        """Cancel the current conversion process."""
        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.stop()
            self.status_label.setText("Cancelling conversion...")
            self.cancel_btn.setEnabled(False)

    def update_conversion_progress(self, progress: int, status: str, data: dict):
        """Update the conversion progress display."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def conversion_finished(self, success: bool, message: str):
        """Handle completion of the conversion process."""
        # Update UI
        self.progress_bar.setValue(100 if success else 0)
        self.status_label.setText(message)

        # Re-enable controls
        self.set_conversion_controls(True)

        # Show message
        if success:
            QMessageBox.information(
                self,
                "Conversion Complete",
                "Model conversion completed successfully."
            )

            # Refresh model list
            self.refresh_model_list()

            # Switch to select tab
            self.tab_widget.setCurrentIndex(0)
        else:
            QMessageBox.critical(
                self,
                "Conversion Failed",
                f"Model conversion failed: {message}"
            )

    def set_conversion_controls(self, enabled: bool):
        """Enable or disable controls during conversion."""
        self.model_name_edit.setEnabled(enabled)
        self.model_path_edit.setEnabled(enabled)
        self.optimize_cb.setEnabled(enabled)
        self.quantize_cb.setEnabled(enabled)
        self.use_gpu_convert_cb.setEnabled(enabled)
        self.convert_btn.setEnabled(enabled)

        # Update cancel button
        self.cancel_btn.setEnabled(not enabled and self.conversion_worker and self.conversion_worker.isRunning())
