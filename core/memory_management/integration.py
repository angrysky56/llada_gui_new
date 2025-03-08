#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration of memory management with LLaDA GUI components.

This module provides functions to integrate the memory management system with
the main LLaDA GUI components, adding dynamic memory optimization capabilities.
"""

import logging
import os
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Memory management imports
from .memory_manager import get_memory_manager
from .dynamic_worker import enhance_llada_worker


def integrate_memory_management():
    """
    Integrate memory management with the LLaDA GUI system.
    
    This function:
    1. Initializes the memory manager singleton
    2. Enhances the LLaDA worker with dynamic memory management
    3. Sets up memory monitoring integration
    
    Returns:
        bool: True if integration was successful
    """
    try:
        logger.info("Integrating memory management system")

        # Initialize memory manager
        memory_manager = get_memory_manager()

        # Enhance llada_worker.py
        from core.llada_worker import LLaDAWorker
        # Replace the original worker class with enhanced version
        enhanced_worker = enhance_llada_worker(LLaDAWorker)
        sys.modules['core.llada_worker'].LLaDAWorker = enhanced_worker
        logger.info("Enhanced LLaDAWorker with dynamic memory management")

        # Integrate with memory monitoring
        from gui.memory_monitor import MemoryMonitor

        # Store original init method
        original_init = MemoryMonitor.__init__

        # Create enhanced init that integrates with our memory manager
        def enhanced_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)

            # Connect to memory manager for more detailed stats
            memory_manager.add_listener(self._memory_callback)

        # Add callback method to MemoryMonitor
        def _memory_callback(self, stats):
            # Check if we have a more detailed update signal
            if hasattr(self, 'detailed_update'):
                # Emit detailed update
                self.detailed_update.emit(stats)
            elif hasattr(self, 'update'):
                # Map to standard update format
                mapped_stats = {
                    'system_total': stats.get('system_total', 0),
                    'system_used': stats.get('system_used', 0),
                    'system_percent': stats.get('system_percent', 0),
                    'gpu_available': stats.get('gpu_available', False),
                    'gpu_total': stats.get('gpu_total', 0),
                    'gpu_used': stats.get('gpu_used', 0),
                    'gpu_percent': stats.get('gpu_percent', 0)
                }
                self.update.emit(mapped_stats)

        # Add the new method to MemoryMonitor
        MemoryMonitor._memory_callback = _memory_callback

        # Replace the init method
        MemoryMonitor.__init__ = enhanced_init
        logger.info("Enhanced MemoryMonitor with detailed memory tracking")

        # Add additional monitoring for CPU and GPU processes
        _setup_process_monitoring()

        return True

    except Exception as e:
        logger.error(f"Error integrating memory management: {e}")
        return False


def _setup_process_monitoring():
    """Set up monitoring for CPU and GPU processes."""
    try:
        # Import process monitoring tools
        import psutil

        # Start process monitoring if available
        memory_manager = get_memory_manager()

        def monitor_processes():
            """Monitor CPU and GPU processes for memory usage."""
            try:
                # Get current process
                current_process = psutil.Process(os.getpid())

                # Get memory info
                memory_info = current_process.memory_info()
                cpu_percent = current_process.cpu_percent()

                # Log high usage
                if memory_info.rss > 4 * (1024 ** 3):  # More than 4GB
                    logger.warning(f"High memory usage detected: {memory_info.rss / (1024 ** 3):.2f}GB")

                if cpu_percent > 90:
                    logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")

            except Exception as e:
                logger.debug(f"Error monitoring processes: {e}")

        # Register process monitoring as a listener
        memory_manager.add_listener(lambda _: monitor_processes())

    except ImportError:
        logger.warning("psutil not available, process monitoring disabled")
    except Exception as e:
        logger.error(f"Error setting up process monitoring: {e}")


def create_memory_status_widget(parent=None):
    """
    Create a status widget for displaying memory usage.
    
    This widget can be added to the GUI to provide visual feedback on memory usage.
    
    Args:
        parent: Parent widget
        
    Returns:
        QWidget: Status widget
    """
    try:
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox
        )
        from PyQt6.QtCore import Qt, pyqtSlot

        class MemoryStatusWidget(QWidget):
            """Widget for displaying memory status."""

            def __init__(self, parent=None):
                super().__init__(parent)

                # Set up layout
                layout = QVBoxLayout(self)

                # Create group box
                group_box = QGroupBox("Memory Status")
                group_layout = QVBoxLayout(group_box)

                # System RAM
                system_layout = QHBoxLayout()
                system_layout.addWidget(QLabel("System RAM:"))
                self.system_progress = QProgressBar()
                self.system_progress.setRange(0, 100)
                self.system_progress.setValue(0)
                system_layout.addWidget(self.system_progress)
                self.system_label = QLabel("0 / 0 GB (0%)")
                system_layout.addWidget(self.system_label)
                group_layout.addLayout(system_layout)

                # GPU Memory
                gpu_layout = QHBoxLayout()
                gpu_layout.addWidget(QLabel("GPU Memory:"))
                self.gpu_progress = QProgressBar()
                self.gpu_progress.setRange(0, 100)
                self.gpu_progress.setValue(0)
                gpu_layout.addWidget(self.gpu_progress)
                self.gpu_label = QLabel("0 / 0 GB (0%)")
                gpu_layout.addWidget(self.gpu_label)
                group_layout.addLayout(gpu_layout)

                # Add group box to main layout
                layout.addWidget(group_box)

                # Connect to memory manager
                memory_manager = get_memory_manager()
                memory_manager.add_listener(self.update_memory_status)

            @pyqtSlot(dict)
            def update_memory_status(self, stats):
                """Update the memory status display."""
                # Update system RAM
                if 'system_total' in stats and 'system_used' in stats:
                    system_total = stats['system_total']
                    system_used = stats['system_used']
                    system_percent = stats.get('system_percent', 0)

                    self.system_progress.setValue(int(system_percent))
                    self.system_label.setText(f"{system_used:.1f} / {system_total:.1f} GB ({system_percent:.0f}%)")

                    # Set color based on usage
                    if system_percent > 90:
                        self.system_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
                    elif system_percent > 75:
                        self.system_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
                    else:
                        self.system_progress.setStyleSheet("")

                # Update GPU memory
                if stats.get('gpu_available', False):
                    gpu_total = stats.get('gpu_total', 0)
                    gpu_used = stats.get('gpu_used', 0)
                    gpu_percent = stats.get('gpu_percent', 0)

                    self.gpu_progress.setValue(int(gpu_percent))
                    self.gpu_label.setText(f"{gpu_used:.1f} / {gpu_total:.1f} GB ({gpu_percent:.0f}%)")

                    # Set color based on usage
                    if gpu_percent > 90:
                        self.gpu_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
                    elif gpu_percent > 75:
                        self.gpu_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
                    else:
                        self.gpu_progress.setStyleSheet("")

                    self.gpu_progress.setVisible(True)
                    self.gpu_label.setText(f"{gpu_used:.1f} / {gpu_total:.1f} GB ({gpu_percent:.0f}%)")
                else:
                    self.gpu_progress.setVisible(False)
                    self.gpu_label.setText("Not available")

        # Create and return widget
        return MemoryStatusWidget(parent)

    except Exception as e:
        logger.error(f"Error creating memory status widget: {e}")
        return None


def optimize_memory_for_startup():
    """
    Apply memory optimizations during application startup.
    
    This function should be called early in the application startup process.
    """
    try:
        import gc
        import torch

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set environment variables for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

        # Disable TF32 for precise computation at the cost of slightly more memory
        torch.backends.cuda.matmul.allow_tf32 = False

        logger.info("Applied startup memory optimizations")

    except Exception as e:
        logger.error(f"Error applying startup memory optimizations: {e}")


def suggest_memory_optimizations(current_config):
    """
    Suggest memory optimizations based on the current configuration.
    
    Args:
        current_config: Current generation configuration
        
    Returns:
        dict: Optimized configuration
    """
    memory_manager = get_memory_manager()
    return memory_manager.optimize_parameters(current_config)


def register_memory_optimization_api():
    """
    Register memory optimization functions in the system API.
    
    This function registers memory optimization functions in the core API
    to make them available to other components.
    
    Returns:
        bool: True if registration was successful
    """
    try:
        # Add to core utils to make functions available
        import core.utils as utils

        # Register memory optimization functions
        utils.optimize_memory_parameters = suggest_memory_optimizations
        utils.get_memory_manager = get_memory_manager
        utils.optimize_startup_memory = optimize_memory_for_startup

        logger.info("Registered memory optimization API")
        return True

    except Exception as e:
        logger.error(f"Error registering memory optimization API: {e}")
        return False
