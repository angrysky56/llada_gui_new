#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory monitoring for the LLaDA GUI application.
"""

import psutil
import torch
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from config import MEMORY_CHECK_INTERVAL

class MemoryMonitor(QObject):
    """Monitor system and GPU memory usage."""
    update = pyqtSignal(dict)
    
    def __init__(self, check_interval=MEMORY_CHECK_INTERVAL, parent=None):
        super().__init__(parent)
        self.check_interval = check_interval
        self.running = False
        self.timer = None
    
    def start(self):
        """Start monitoring memory."""
        if self.running:
            return
        
        self.running = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_memory)
        self.timer.start(int(self.check_interval * 1000))
    
    def stop(self):
        """Stop monitoring memory."""
        if not self.running:
            return
        
        self.running = False
        if self.timer:
            self.timer.stop()
    
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
        
        # GPU memory if available
        gpu_stats = {'gpu_available': False}
        if torch.cuda.is_available():
            try:
                gpu_stats = {
                    'gpu_available': True,
                    'gpu_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                    'gpu_used': (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3), # GB
                    'gpu_percent': (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) 
                                  / torch.cuda.get_device_properties(0).total_memory * 100
                }
            except Exception:
                pass
        
        # Combine stats and emit update
        stats = {**system_stats, **gpu_stats}
        self.update.emit(stats)
