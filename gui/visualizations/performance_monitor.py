#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance monitoring visualization widget.

This module provides a visualization widget for monitoring and displaying
performance metrics related to model loading and generation times.
"""

import numpy as np
import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QGroupBox, QTableWidget, QTableWidgetItem, QPushButton
)


class PerformanceMonitor(QWidget):
    """Widget for visualizing performance metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Performance metrics
        self.metrics = {
            'model_load_times': [],
            'generation_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generations': 0
        }

        # Initialize UI
        self.init_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_stats)
        self.update_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Title and description
        title = QLabel("Performance")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        description = QLabel(
            "This panel displays performance metrics for model loading and generation, "
            "showing the benefits of optimizations like model caching and memory management."
        )
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)

        # System resources section
        system_group = QGroupBox("System")
        system_layout = QVBoxLayout(system_group)

        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_bar)
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_label)
        system_layout.addLayout(cpu_layout)

        # Memory usage
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("RAM:"))
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 100)
        mem_layout.addWidget(self.mem_bar)
        self.mem_label = QLabel("0 / 0 GB (0%)")
        mem_layout.addWidget(self.mem_label)
        system_layout.addLayout(mem_layout)

        layout.addWidget(system_group)

        # Generation metrics section
        metrics_group = QGroupBox("Generation Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        # Create a table for metrics
        self.metrics_table = QTableWidget(5, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setItem(0, 0, QTableWidgetItem("Average Model Load Time"))
        self.metrics_table.setItem(1, 0, QTableWidgetItem("Average Generation Time"))
        self.metrics_table.setItem(2, 0, QTableWidgetItem("Cache Hits"))
        self.metrics_table.setItem(3, 0, QTableWidgetItem("Cache Misses"))
        self.metrics_table.setItem(4, 0, QTableWidgetItem("Total Generations"))

        # Initialize values
        self.metrics_table.setItem(0, 1, QTableWidgetItem("0.0 sec"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem("0.0 sec"))
        self.metrics_table.setItem(2, 1, QTableWidgetItem("0"))
        self.metrics_table.setItem(3, 1, QTableWidgetItem("0"))
        self.metrics_table.setItem(4, 1, QTableWidgetItem("0"))

        # Adjust table size
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.resizeColumnsToContents()

        metrics_layout.addWidget(self.metrics_table)

        # Cache control
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("Model Cache:"))

        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_model_cache)
        cache_layout.addWidget(self.clear_cache_btn)

        metrics_layout.addLayout(cache_layout)

        layout.addWidget(metrics_group)

        # Performance tips section
        tips_group = QGroupBox("Performance Tips")
        tips_layout = QVBoxLayout(tips_group)

        tips_text = QLabel(
            "<ul>"
            "<li><b>Model Caching:</b> After first generation, subsequent generations will be much faster</li>"
            "<li><b>Memory Optimization:</b> 4-bit and 8-bit quantization significantly reduce memory usage</li>"
            "<li><b>Generation Length:</b> Shorter generations are faster and use less memory</li>"
            "<li><b>Sampling Steps:</b> Fewer steps are faster but may reduce quality</li>"
            "</ul>"
        )
        tips_text.setWordWrap(True)
        tips_layout.addWidget(tips_text)

        layout.addWidget(tips_group)

        # Initialize system stats
        self.update_system_stats()

    def update_system_stats(self):
        """Update system resource usage displays."""
        # Update CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_bar.setValue(int(cpu_percent))
        self.cpu_label.setText(f"{cpu_percent:.1f}%")

        # Update memory usage
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        mem_used = memory.used / (1024 ** 3)  # GB
        mem_total = memory.total / (1024 ** 3)  # GB

        self.mem_bar.setValue(int(mem_percent))
        self.mem_label.setText(f"{mem_used:.1f} / {mem_total:.1f} GB ({mem_percent:.1f}%)")

        # Set color based on usage
        self._set_bar_color(self.cpu_bar, cpu_percent)
        self._set_bar_color(self.mem_bar, mem_percent)

    @staticmethod
    def _set_bar_color(bar, value):
        """Set progress bar color based on value."""
        if value > 90:
            bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif value > 75:
            bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            bar.setStyleSheet("")

    def update_metrics(self, metrics):
        """
        Update performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        # Update stored metrics
        if 'load_time' in metrics and metrics['load_time'] > 0:
            self.metrics['model_load_times'].append(metrics['load_time'])

        if 'generation_time' in metrics and metrics['generation_time'] > 0:
            self.metrics['generation_times'].append(metrics['generation_time'])

        if 'cached_model' in metrics:
            if metrics['cached_model']:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1

        self.metrics['total_generations'] += 1

        # Update the table
        avg_load_time = np.mean(self.metrics['model_load_times']) if self.metrics['model_load_times'] else 0
        avg_gen_time = np.mean(self.metrics['generation_times']) if self.metrics['generation_times'] else 0

        self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{avg_load_time:.2f} sec"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{avg_gen_time:.2f} sec"))
        self.metrics_table.setItem(2, 1, QTableWidgetItem(str(self.metrics['cache_hits'])))
        self.metrics_table.setItem(3, 1, QTableWidgetItem(str(self.metrics['cache_misses'])))
        self.metrics_table.setItem(4, 1, QTableWidgetItem(str(self.metrics['total_generations'])))

    def clear_model_cache(self):
        """Clear the model cache."""
        try:
            # Import the model cache
            from core.performance.model_cache import get_model_cache

            # Clear the cache
            model_cache = get_model_cache()
            if model_cache.clear():
                # Update display
                self.metrics['cache_hits'] = 0
                self.metrics['cache_misses'] = 0
                self.update_metrics({})
        except Exception as e:
            print(f"Error clearing model cache: {e}")


def add_performance_tab(gui_instance):
    """
    Add a performance monitoring tab to the GUI.
    
    Args:
        gui_instance: LLaDA GUI instance
        
    Returns:
        bool: True if added successfully, False otherwise
    """
    try:
        # Create performance monitor
        performance_monitor = PerformanceMonitor()

        # Add to tab widget
        if hasattr(gui_instance, 'tab_widget'):
            gui_instance.tab_widget.addTab(performance_monitor, "Performance")

            # Save reference
            gui_instance.performance_monitor = performance_monitor

            # Connect to worker metrics
            old_generation_finished = gui_instance.generation_finished

            # Create enhanced generation_finished to track metrics
            def enhanced_generation_finished(result):
                # Call original method
                old_generation_finished(result)

                # Get metrics from worker if available
                if hasattr(gui_instance, 'worker') and gui_instance.worker:
                    try:
                        if hasattr(gui_instance.worker, 'metrics'):
                            gui_instance.performance_monitor.update_metrics(gui_instance.worker.metrics)
                        elif hasattr(gui_instance.worker, 'cached_model'):
                            metrics = {
                                'cached_model': gui_instance.worker.cached_model,
                                'load_time': getattr(gui_instance.worker, 'load_time', 0),
                                'generation_time': 0  # Can't measure this without instrumentation
                            }
                            gui_instance.performance_monitor.update_metrics(metrics)
                    except Exception as e:
                        print(f"Error updating performance metrics: {e}")

            # Replace the method
            gui_instance.generation_finished = enhanced_generation_finished

            return True

        return False

    except Exception as e:
        print(f"Error adding performance tab: {e}")
        return False
