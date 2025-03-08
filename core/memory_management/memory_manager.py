#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Manager for LLaDA GUI - Handles GPU and system memory optimizations.

This module provides advanced memory management features including:
- Dynamic parameter adjustment based on available resources
- Memory usage tracking and warnings
- Automatic optimization selection
- Scheduled memory cleanup
"""

import gc
import logging
import threading
import time

import psutil
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance
_memory_manager = None


def get_memory_manager():
    """
    Get or create the global memory manager instance (singleton).

    Returns:
        MemoryManager: The singleton memory manager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


class MemoryManager:
    """
    Advanced memory management for LLaDA operations.

    This class provides comprehensive memory management capabilities to prevent
    out-of-memory errors and optimize performance based on available resources.
    """

    def __init__(self):
        """Initialize the memory manager."""
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.memory_listeners = []
        self.last_memory_stats = {}
        self.history = []
        self.history_max_len = 100  # Keep last 100 snapshots
        self.monitoring_interval = 1.0  # 1 second by default

        # Thresholds for warnings and adjustments (percentages)
        self.warning_threshold = 90
        self.critical_threshold = 95
        self.adjustment_threshold = 85

        # Default adjustment parameters with memory impact multipliers
        self.default_parameters = {
            'gen_length': {
                'min': 16,
                'default': 64,
                'max': 512,
                'step': 16,
                'memory_multiplier': 0.5  # High memory impact
            },
            'steps': {
                'min': 16,
                'default': 64,
                'max': 512,
                'step': 16,
                'memory_multiplier': 0.3  # Medium memory impact
            },
            'block_length': {
                'min': 16,
                'default': 32,
                'max': 256,
                'step': 16,
                'memory_multiplier': 0.2  # Lower memory impact
            }
        }

        # Initialize monitoring thread
        self._start_monitoring_thread()

    def __del__(self):
        """Clean up resources when the instance is deleted."""
        self.stop_monitoring()

    def _start_monitoring_thread(self):
        """Start the memory monitoring thread if it's not already running."""
        with self.lock:
            if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
                self.stop_monitoring.clear()
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitoring_thread.start()
                logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        with self.lock:
            self.stop_monitoring.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(2)  # Wait up to 2 seconds
                logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Background thread for continuous memory monitoring."""
        while not self.stop_monitoring.is_set():
            try:
                stats = self.get_memory_stats()
                self._update_history(stats)
                self._check_thresholds(stats)
                self._notify_listeners(stats)
                self.last_memory_stats = stats

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            time.sleep(self.monitoring_interval)

    def _update_history(self, stats):
        """Add current memory stats to history and maintain history size."""
        self.history.append((time.time(), stats))
        if len(self.history) > self.history_max_len:
            self.history.pop(0)  # Remove oldest entry

    def _check_thresholds(self, stats):
        """Check memory usage against warning and critical thresholds."""
        if stats.get('gpu_available', False):
            gpu_percent = stats['gpu_percent']
            if gpu_percent > self.critical_threshold:
                logger.warning(f"CRITICAL: GPU memory usage at {gpu_percent:.1f}%")
                self.force_cleanup()  # Force cleanup on critical threshold
            elif gpu_percent > self.warning_threshold:
                logger.warning(f"WARNING: High GPU memory usage at {gpu_percent:.1f}%")

        system_percent = stats.get('system_percent', 0)
        if system_percent > self.critical_threshold:
            logger.warning(f"CRITICAL: System memory usage at {system_percent:.1f}%")
            gc.collect()  # System GC on critical system memory

    def _notify_listeners(self, stats):
        """Notify all registered listeners with current memory statistics."""
        for listener in self.memory_listeners:
            try:
                listener(stats)
            except Exception as e:
                logger.error(f"Error notifying memory listener: {e}")

    def add_listener(self, listener_func):
        """Register a listener function to receive memory statistics updates."""
        if listener_func not in self.memory_listeners:
            self.memory_listeners.append(listener_func)
            if self.last_memory_stats:
                try:
                    listener_func(self.last_memory_stats)  # Send initial stats
                except Exception as e:
                    logger.error(f"Error notifying new memory listener: {e}")

    def remove_listener(self, listener_func):
        """Unregister a listener function."""
        if listener_func in self.memory_listeners:
            self.memory_listeners.remove(listener_func)

    @staticmethod
    def get_memory_stats():
        """
        Get current memory statistics for system and GPU.

        Returns:
            dict: Memory statistics dictionary
        """
        stats = {
            'timestamp': time.time(),
            'system_available': True,
            'gpu_available': torch.cuda.is_available()
        }

        # System memory stats
        memory = psutil.virtual_memory()
        stats['system_total'] = memory.total / (1024 ** 3)  # GB
        stats['system_used'] = memory.used / (1024 ** 3)  # GB
        stats['system_free'] = memory.available / (1024 ** 3)  # GB
        stats['system_percent'] = memory.percent

        # GPU memory stats if available
        if stats['gpu_available']:
            try:
                device_id = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
                total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB

                used = allocated + (reserved - allocated)  # Account for reserved but not allocated
                free = total - used

                stats['gpu_device'] = device_id
                stats['gpu_name'] = torch.cuda.get_device_name(device_id)
                stats['gpu_total'] = total
                stats['gpu_used'] = used
                stats['gpu_free'] = free
                stats['gpu_allocated'] = allocated
                stats['gpu_reserved'] = reserved
                stats['gpu_percent'] = (used / total) * 100 if total > 0 else 0

            except Exception as e:
                logger.error(f"Error getting GPU memory stats: {e}")
                stats['gpu_available'] = False

        return stats

    @staticmethod
    def force_cleanup():
        """Force garbage collection and CUDA memory cleanup."""
        logger.info("Forcing memory cleanup: garbage collection and CUDA cache clear")
        for _ in range(3):  # Multiple GC passes for thorough cleanup
            gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                device_id = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                used = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
                logger.info(f"GPU memory after cleanup: {used:.2f}GB / {total:.2f}GB")
            except Exception as e:
                logger.error(f"Error cleaning GPU memory: {e}")

    def optimize_parameters(self, current_params, available_memory_gb=None):
        """
        Dynamically optimize generation parameters based on available GPU memory.

        Args:
            current_params (dict): Current generation parameters.
            available_memory_gb (float, optional): Available GPU memory in GB. If None, it will be detected.

        Returns:
            dict: Optimized parameters.
        """
        if not torch.cuda.is_available():
            return current_params  # No GPU, no optimization needed

        if available_memory_gb is None:
            stats = self.get_memory_stats()
            available_memory_gb = stats.get('gpu_free', 0) if stats.get('gpu_available', False) else 0

        logger.info(f"Optimizing parameters for {available_memory_gb:.2f}GB available memory")
        optimized = dict(current_params)  # Create a copy to avoid modifying original

        if available_memory_gb < 2:  # Extreme memory constraint
            logger.warning("Critically low GPU memory, applying aggressive optimizations")
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.75)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.75)
            self._adjust_parameter(optimized, 'block_length', decrease_factor=0.5)
            optimized['use_4bit'] = True
            optimized['extreme_mode'] = True

        elif available_memory_gb < 4:  # Very limited memory
            logger.warning("Very limited GPU memory, applying strong optimizations")
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.5)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.5)
            optimized['use_4bit'] = True
            optimized['extreme_mode'] = True

        elif available_memory_gb < 6:  # Limited memory
            logger.info("Limited GPU memory, applying moderate optimizations")
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.25)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.25)
            optimized['use_8bit'] = True

        elif available_memory_gb < 8:  # Somewhat limited memory
            logger.info("Somewhat limited GPU memory, applying light optimizations")
            if optimized.get('gen_length', 64) > 256 or optimized.get('steps', 64) > 256:
                self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.1)
                self._adjust_parameter(optimized, 'steps', decrease_factor=0.1)
                optimized['use_8bit'] = True

        elif available_memory_gb < 11:  # Approaching 12GB limit
            logger.info("Good GPU memory available (approaching limits for extreme settings)")
            if optimized.get('gen_length', 64) > 384 and optimized.get('steps', 64) > 384:
                self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.05)
                self._adjust_parameter(optimized, 'steps', decrease_factor=0.05)

        return optimized

    def _adjust_parameter(self, params, param_name, decrease_factor=None, target_value=None):
        """Adjust a single parameter within its defined constraints."""
        if param_name not in params or param_name not in self.default_parameters:
            return False

        constraints = self.default_parameters[param_name]
        current_value = params[param_name]

        if target_value is not None:
            new_value = target_value
        elif decrease_factor is not None:
            reduction = max(int(current_value * decrease_factor), constraints['step'])
            new_value = current_value - reduction
        else:
            return False

        new_value = max(constraints['min'], min(constraints['max'], new_value))
        if constraints['step'] > 1:
            new_value = round(new_value / constraints['step']) * constraints['step']

        if new_value != current_value:
            params[param_name] = new_value
            logger.info(f"Adjusted {param_name} from {current_value} to {new_value}")
            return True
        return False

    @staticmethod
    def estimate_memory_requirement(params):
        """Estimate memory required based on generation parameters."""
        base_memory_gb = 3.5  # Base memory for LLaDA-8B model (empirical value)

        if params.get('use_4bit', False):
            base_memory_gb *= 0.5
        elif params.get('use_8bit', False):
            base_memory_gb *= 0.75

        memory_gb = base_memory_gb
        gen_length = params.get('gen_length', 64)
        memory_gb += (gen_length / 64) * 0.4
        steps = params.get('steps', 64)
        memory_gb += (steps / 64) * 0.25
        block_length = params.get('block_length', 32)
        memory_gb += (block_length / 32) * 0.15

        if params.get('extreme_mode', False):
            memory_gb *= 0.7

        logger.info(f"Estimated memory requirement: {memory_gb:.2f}GB for generation parameters")
        return memory_gb

    def can_fit_in_memory(self, params):
        """Check if generation parameters can fit within available GPU memory."""
        stats = self.get_memory_stats()
        if not stats.get('gpu_available', False):
            return True  # No GPU, assume CPU can handle it

        available_memory_gb = stats.get('gpu_free', 0)
        estimated_memory_gb = self.estimate_memory_requirement(params)
        can_fit = available_memory_gb >= (estimated_memory_gb + 0.5)  # Add buffer

        if not can_fit:
            logger.warning(
                f"Insufficient memory: Need {estimated_memory_gb:.2f}GB, "
                f"but only {available_memory_gb:.2f}GB available"
            )
        return can_fit

    def get_memory_usage_history(self, duration_seconds=None):
        """Get historical memory usage data."""
        with self.lock:
            if not self.history:
                return {'timestamps': [], 'system_percent': [], 'gpu_percent': []}

            history = self._filter_history_by_duration(duration_seconds)
            timestamps, system_percent, gpu_percent = self._extract_history_data(history)

            return {
                'timestamps': timestamps,
                'system_percent': system_percent,
                'gpu_percent': gpu_percent
            }

    def _filter_history_by_duration(self, duration_seconds):
        """Filter memory history by a given duration."""
        if duration_seconds is None:
            return self.history
        cutoff_time = time.time() - duration_seconds
        return [(t, s) for t, s in self.history if t >= cutoff_time]

    @staticmethod
    def _extract_history_data(history):
        """Extract timestamps and memory percentages from history data."""
        timestamps = [t for t, _ in history]
        system_percent = [s.get('system_percent', 0) for _, s in history]
        gpu_percent = [s.get('gpu_percent', 0) for _, s in history if s.get('gpu_available', False)]

        if len(gpu_percent) < len(timestamps):  # Pad GPU data if shorter
            gpu_percent.extend([0] * (len(timestamps) - len(gpu_percent)))
        return timestamps, system_percent, gpu_percent

    def suggest_optimal_parameters(self, task_requirement=None):
        """Suggest optimal parameters based on memory and task requirements."""
        stats = self.get_memory_stats()
        params = {param: config['default'] for param, config in self.default_parameters.items()}

        params['device'] = 'cuda' if stats.get('gpu_available', False) else 'cpu'
        params['use_8bit'] = False
        params['use_4bit'] = False
        params['extreme_mode'] = False

        if task_requirement is not None:
            if task_requirement >= 5:  # Highest quality
                params['gen_length'] = 256
                params['steps'] = 128
            elif task_requirement >= 4:
                params['gen_length'] = 192
                params['steps'] = 96
            elif task_requirement >= 3:
                params['gen_length'] = 128
                params['steps'] = 80
            elif task_requirement >= 2:
                params['gen_length'] = 96
                params['steps'] = 64

        if params['device'] == 'cuda':
            available_memory_gb = stats.get('gpu_free', 0)
            params = self.optimize_parameters(params, available_memory_gb)

        return params

    def get_memory_warning(self, params):
        """Get a warning message if memory usage is likely to be problematic."""
        if not torch.cuda.is_available() or params.get('device', 'cuda') != 'cuda':
            return None  # No warning for CPU mode

        stats = self.get_memory_stats()
        if not stats.get('gpu_available', False):
            return None

        available_memory_gb = stats.get('gpu_free', 0)
        estimated_memory_gb = self.estimate_memory_requirement(params)

        if available_memory_gb < estimated_memory_gb:
            return (
                f"Warning: Current parameters require ~{estimated_memory_gb:.1f}GB GPU memory, "
                f"but only {available_memory_gb:.1f}GB is available. "
                f"May cause out-of-memory errors. Reduce gen_length/steps or enable optimizations."
            )
        elif available_memory_gb < (estimated_memory_gb * 1.2):
            return (
                f"Caution: Current parameters will use ~{estimated_memory_gb:.1f}GB GPU memory. "
                f"Available: {available_memory_gb:.1f}GB, which is close to the limit. "
                f"Consider enabling memory optimizations."
            )
        return None
