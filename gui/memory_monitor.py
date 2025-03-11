#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory monitoring for the LLaDA GUI application.
Tracks both system RAM and GPU memory usage.
"""

import logging
import os
import time

import psutil
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.config import MEMORY_MONITORING_SETTINGS
MEMORY_WARNING_THRESHOLD = MEMORY_MONITORING_SETTINGS['memory_warning_threshold']
MEMORY_CHECK_INTERVAL = MEMORY_MONITORING_SETTINGS['memory_check_interval']

# Try to import advanced memory management
try:
    from core.memory_management.integration import get_memory_manager

    HAS_ADVANCED_MEMORY = True
except ImportError:
    HAS_ADVANCED_MEMORY = False

logger = logging.getLogger(__name__)


class MemoryMonitor(QObject):
    """Monitor system and GPU memory usage."""
    update = pyqtSignal(dict)
    warning = pyqtSignal(str)
    detailed_update = pyqtSignal(dict)  # More detailed stats signal

    def __init__(self, check_interval=MEMORY_CHECK_INTERVAL, parent=None):
        super().__init__(parent)

        # Ensure a QApplication exists before creating any QWidget
        if not QApplication.instance():
            self.app = QApplication([])
            logger.info("Created QApplication instance in MemoryMonitor")
        else:
            self.app = QApplication.instance()
            logger.info("Using existing QApplication instance in MemoryMonitor")

        # Ensure a QApplication exists before creating any QWidget
        if not QApplication.instance():
            self.app = QApplication([])
            logger.info("Created QApplication instance in MemoryMonitor")
        else:
            self.app = QApplication.instance()
            logger.info("Using existing QApplication instance in MemoryMonitor")
        self.check_interval = check_interval
        self.running = False
        self.timer = None
        self.last_warning_time = 0
        self.warning_cooldown = 30  # seconds between warnings

        # Connect to advanced memory management if available
        self.memory_manager = None
        if HAS_ADVANCED_MEMORY:
            try:
                self.memory_manager = get_memory_manager()
                logger.info("Connected to advanced memory management system")
            except Exception as e:
                logger.warning(f"Error connecting to memory manager: {e}")

    def start(self):
        """Start monitoring memory."""
        if self.running:
            return

        self.running = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_memory)
        self.timer.start(int(self.check_interval * 1000))
        logger.info("Memory monitor started")

    def stop(self):
        """Stop monitoring memory."""
        if not self.running:
            return

        self.running = False
        if self.timer:
            self.timer.stop()
        logger.info("Memory monitor stopped")

    def check_memory(self):
        """Check current memory usage and emit the update signal."""
        if not self.running:
            return

        # Check if we should use advanced memory management
        if self.memory_manager is not None:
            try:
                # Get detailed stats from memory manager
                detailed_stats = self.memory_manager.get_memory_stats()

                # Convert to our format
                stats = {
                    'system_total': detailed_stats.get('system_total', 0),
                    'system_used': detailed_stats.get('system_used', 0),
                    'system_percent': detailed_stats.get('system_percent', 0),
                    'cpu_percent': detailed_stats.get('cpu_percent', psutil.cpu_percent(interval=None)),
                    'gpu_available': detailed_stats.get('gpu_available', False)
                }

                # Add GPU stats if available
                if stats['gpu_available']:
                    stats.update({
                        'gpu_total': detailed_stats.get('gpu_total', 0),
                        'gpu_used': detailed_stats.get('gpu_used', 0),
                        'gpu_percent': detailed_stats.get('gpu_percent', 0),
                        'gpu_allocated': detailed_stats.get('gpu_allocated', 0),
                        'gpu_reserved': detailed_stats.get('gpu_reserved', 0)
                    })

                    # Check if we should issue a memory warning
                    current_time = time.time()
                    if (stats['gpu_percent'] > MEMORY_WARNING_THRESHOLD and
                            current_time - self.last_warning_time > self.warning_cooldown):
                        warning_msg = (
                            f"High GPU memory usage: {stats['gpu_percent']:.1f}% "
                            f"({stats['gpu_used']:.2f} GB used out of {stats['gpu_total']:.2f} GB)"
                        )
                        self.warning.emit(warning_msg)
                        self.last_warning_time = current_time
                        logger.warning(warning_msg)

                # Emit the update signals
                self.update.emit(stats)
                self.detailed_update.emit(detailed_stats)
                return

            except Exception as e:
                logger.error(f"Error getting memory stats from manager: {str(e)}")
                # Fall back to standard method

        # Standard memory checking (without advanced management)

        # System memory
        memory = psutil.virtual_memory()
        system_stats = {
            'system_total': memory.total / (1024 ** 3),  # GB
            'system_used': memory.used / (1024 ** 3),  # GB
            'system_percent': memory.percent
        }

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        system_stats['cpu_percent'] = cpu_percent

        # GPU memory if available
        gpu_stats = {'gpu_available': False}
        if torch.cuda.is_available():
            try:
                # Get total GPU memory
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

                # Get allocated and reserved memory
                allocated_memory = torch.cuda.memory_allocated(0)
                reserved_memory = torch.cuda.memory_reserved(0)

                # Calculate used memory and percentage
                used_gpu_memory = allocated_memory + reserved_memory
                gpu_percent = used_gpu_memory / total_gpu_memory * 100

                gpu_stats = {
                    'gpu_available': True,
                    'gpu_total': total_gpu_memory / (1024 ** 3),  # GB
                    'gpu_used': used_gpu_memory / (1024 ** 3),  # GB
                    'gpu_allocated': allocated_memory / (1024 ** 3),  # GB
                    'gpu_reserved': reserved_memory / (1024 ** 3),  # GB
                    'gpu_percent': gpu_percent
                }

                # Check if we should issue a memory warning
                current_time = os.times().elapsed
                if (gpu_percent > MEMORY_WARNING_THRESHOLD and
                        current_time - self.last_warning_time > self.warning_cooldown):
                    warning_msg = (
                        f"High GPU memory usage: {gpu_percent:.1f}% "
                        f"({used_gpu_memory / (1024 ** 3):.2f} GB used out of {total_gpu_memory / (1024 ** 3):.2f} GB)"
                    )
                    self.warning.emit(warning_msg)
                    self.last_warning_time = current_time
                    logger.warning(warning_msg)
            except Exception as e:
                logger.error(f"Error getting GPU memory: {str(e)}")

        # Combine stats and emit update
        stats = {**system_stats, **gpu_stats}
        self.update.emit(stats)
