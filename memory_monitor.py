#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory monitoring for the LLaDA GUI application.
Tracks both system RAM and GPU memory usage.
"""

import os
import psutil
import torch
import logging
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from config import MEMORY_CHECK_INTERVAL, MEMORY_WARNING_THRESHOLD

logger = logging.getLogger(__name__)

class MemoryMonitor(QObject):
    """Monitor system and GPU memory usage."""
    update = pyqtSignal(dict)
    warning = pyqtSignal(str)
    
    def __init__(self, check_interval=MEMORY_CHECK_INTERVAL, parent=None):
        super().__init__(parent)
        self.check_interval = check_interval
        self.running = False
        self.timer = None
        self.last_warning_time = 0
        self.warning_cooldown = 30  # seconds between warnings
    
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
        
        # System memory
        memory = psutil.virtual_memory()
        system_stats = {
            'system_total': memory.total / (1024**3),  # GB
            'system_used': memory.used / (1024**3),    # GB
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
                    'gpu_total': total_gpu_memory / (1024**3),  # GB
                    'gpu_used': used_gpu_memory / (1024**3),    # GB
                    'gpu_allocated': allocated_memory / (1024**3),  # GB
                    'gpu_reserved': reserved_memory / (1024**3),    # GB
                    'gpu_percent': gpu_percent
                }
                
                # Check if we should issue a memory warning
                current_time = os.times().elapsed
                if (gpu_percent > MEMORY_WARNING_THRESHOLD and 
                    current_time - self.last_warning_time > self.warning_cooldown):
                    warning_msg = (
                        f"High GPU memory usage: {gpu_percent:.1f}% "
                        f"({used_gpu_memory/(1024**3):.2f} GB used out of {total_gpu_memory/(1024**3):.2f} GB)"
                    )
                    self.warning.emit(warning_msg)
                    self.last_warning_time = current_time
                    logger.warning(warning_msg)
            except Exception as e:
                logger.error(f"Error getting GPU memory: {str(e)}")
        
        # Combine stats and emit update
        stats = {**system_stats, **gpu_stats}
        self.update.emit(stats)
