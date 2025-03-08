#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Background task management for responsive UI.

This module provides classes and utilities for executing long-running tasks
in background threads to prevent UI freezing.
"""

import logging
import traceback
from functools import wraps
from typing import Callable, TypeVar, Generic

from PyQt6.QtCore import QObject, QThread, pyqtSignal

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')
R = TypeVar('R')


class BackgroundTask(QThread, Generic[T, R]):
    """
    Background task that runs a function in a separate thread.
    
    This class provides a convenient way to run long operations in a background
    thread without freezing the UI. It provides signals for progress updates
    and completion handling.
    """

    # Signals
    started_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, str, dict)  # progress, status, data
    result_signal = pyqtSignal(object)  # result
    error_signal = pyqtSignal(str)  # error message
    finished_signal = pyqtSignal()

    def __init__(
            self,
            target: Callable[..., R],
            args: tuple = (),
            kwargs: dict = None,
            parent: QObject = None,
            on_result: Callable[[R], None] = None,
            on_error: Callable[[str], None] = None,
            on_progress: Callable[[int, str, dict], None] = None,
            name: str = None
    ):
        """
        Initialize the background task.
        
        Args:
            target: Function to run in the background
            args: Arguments to pass to the target function
            kwargs: Keyword arguments to pass to the target function
            parent: Parent QObject
            on_result: Callback for when the task completes successfully
            on_error: Callback for when the task fails
            on_progress: Callback for progress updates
            name: Name of the task (for logging)
        """
        super().__init__(parent)

        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.on_result = on_result
        self.on_error = on_error
        self.on_progress = on_progress
        self.name = name or target.__name__
        self.result = None
        self.error = None
        self.is_cancelled = False

        # Connect signals to callbacks
        if on_result:
            self.result_signal.connect(on_result)
        if on_error:
            self.error_signal.connect(on_error)
        if on_progress:
            self.progress_signal.connect(on_progress)

    def run(self):
        """Run the task in a background thread."""
        try:
            # Signal that we're starting
            self.started_signal.emit()
            logger.debug(f"Starting background task: {self.name}")

            # Create a progress callback for the target function
            def progress_callback(progress: int, status: str = "", data: dict = None):
                if self.is_cancelled:
                    return False  # Signal to stop the operation
                self.progress_signal.emit(progress, status, data or {})
                return True  # Continue the operation

            # Add progress callback to kwargs if not already provided
            if 'progress_callback' not in self.kwargs:
                self.kwargs['progress_callback'] = progress_callback

            # Run the target function
            self.result = self.target(*self.args, **self.kwargs)

            # Emit result signal if not cancelled
            if not self.is_cancelled:
                self.result_signal.emit(self.result)

        except Exception as e:
            # Handle and log error
            self.error = str(e)
            error_msg = f"Error in background task '{self.name}': {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)

            # Emit error signal if not cancelled
            if not self.is_cancelled:
                self.error_signal.emit(error_msg)

        finally:
            # Signal completion even if there was an error or cancellation
            self.finished_signal.emit()
            logger.debug(f"Finished background task: {self.name}")

    def cancel(self):
        """Cancel the task if it's running."""
        if self.isRunning():
            self.is_cancelled = True
            logger.debug(f"Cancelling background task: {self.name}")

            # Wait for a short time for the thread to finish gracefully
            if not self.wait(500):
                # Terminate the thread if it doesn't finish
                logger.warning(f"Terminating background task: {self.name}")
                self.terminate()


class BackgroundTaskManager(QObject):
    """
    Manager for background tasks.
    
    This class provides a centralized way to manage multiple background tasks
    and handle their lifecycle.
    """

    task_started = pyqtSignal(str)
    task_progress = pyqtSignal(str, int, str)
    task_completed = pyqtSignal(str)
    task_failed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        """Initialize the task manager."""
        super().__init__(parent)
        self.tasks = {}  # task_id -> BackgroundTask
        self.task_names = {}  # task_id -> task_name
        self._next_task_id = 0

    def run_task(
            self,
            target: Callable[..., R],
            args: tuple = (),
            kwargs: dict = None,
            on_result: Callable[[R], None] = None,
            on_error: Callable[[str], None] = None,
            on_progress: Callable[[int, str, dict], None] = None,
            name: str = None,
            auto_cleanup: bool = True
    ) -> str:
        """
        Run a function in a background thread.
        
        Args:
            target: Function to run in the background
            args: Arguments to pass to the target function
            kwargs: Keyword arguments to pass to the target function
            on_result: Callback for when the task completes successfully
            on_error: Callback for when the task fails
            on_progress: Callback for progress updates
            name: Name of the task (for logging)
            auto_cleanup: Whether to automatically clean up the task when it completes
            
        Returns:
            str: Task ID that can be used to cancel or check the task
        """
        # Create a unique ID for this task
        task_id = str(self._next_task_id)
        self._next_task_id += 1

        # Set default name if not provided
        name = name or f"Task-{task_id}"
        self.task_names[task_id] = name

        # Create the task
        task = BackgroundTask(
            target=target,
            args=args,
            kwargs=kwargs,
            parent=self,
            on_result=lambda result: self._on_task_result(task_id, result, on_result),
            on_error=lambda error: self._on_task_error(task_id, error, on_error),
            on_progress=lambda progress, status, data: self._on_task_progress(
                task_id, progress, status, data, on_progress
            ),
            name=name
        )

        # Store the task
        self.tasks[task_id] = task

        # Connect cleanup handler if auto_cleanup is enabled
        if auto_cleanup:
            task.finished_signal.connect(lambda: self._cleanup_task(task_id))

        # Start the task
        task.start()

        # Signal that the task has started
        self.task_started.emit(task_id)

        return task_id

    def _on_task_result(self, task_id, result, callback=None):
        """Handle task completion with result."""
        # Call the callback if provided
        if callback:
            callback(result)

        # Signal that the task has completed
        self.task_completed.emit(task_id)

    def _on_task_error(self, task_id, error, callback=None):
        """Handle task error."""
        # Call the callback if provided
        if callback:
            callback(error)

        # Signal that the task has failed
        self.task_failed.emit(task_id, error)

    def _on_task_progress(self, task_id, progress, status, data, callback=None):
        """Handle task progress update."""
        # Call the callback if provided
        if callback:
            callback(progress, status, data)

        # Signal the progress update
        self.task_progress.emit(task_id, progress, status)

    def _cleanup_task(self, task_id):
        """Clean up a completed task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]

            # Wait for the task to finish if it's still running
            if task.isRunning():
                task.wait()

            # Clean up signals
            try:
                task.result_signal.disconnect()
                task.error_signal.disconnect()
                task.progress_signal.disconnect()
                task.finished_signal.disconnect()
            except (TypeError, RuntimeError):
                # Ignore errors from disconnecting already disconnected signals
                pass

            # Remove the task
            del self.tasks[task_id]

            if task_id in self.task_names:
                del self.task_names[task_id]

    def cancel_task(self, task_id):
        """Cancel a running task."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            name = self.task_names.get(task_id, "Unknown")

            logger.info(f"Cancelling task {name} (ID: {task_id})")
            task.cancel()

            return True

        return False

    def is_task_running(self, task_id):
        """Check if a task is still running."""
        if task_id in self.tasks:
            return self.tasks[task_id].isRunning()

        return False

    def get_running_tasks(self):
        """Get a list of currently running tasks."""
        return [task_id for task_id in self.tasks if self.tasks[task_id].isRunning()]

    def cancel_all_tasks(self):
        """Cancel all running tasks."""
        for task_id in list(self.tasks.keys()):
            self.cancel_task(task_id)


# Singleton task manager
_task_manager = None


def get_task_manager():
    """Get or create the task manager singleton."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


def run_in_background(
        target=None,
        on_result=None,
        on_error=None,
        on_progress=None,
        name=None,
        auto_cleanup=True
):
    """
    Decorator to run a function in a background thread.
    
    This decorator can be used in two ways:
    1. With arguments: @run_in_background(on_result=handle_result)
    2. Without arguments: @run_in_background
    
    Args:
        target: Function to run in the background (for use as a decorator)
        on_result: Callback for when the task completes successfully
        on_error: Callback for when the task fails
        on_progress: Callback for progress updates
        name: Name of the task (for logging)
        auto_cleanup: Whether to automatically clean up the task when it completes
        
    Returns:
        Decorated function that runs in a background thread
    """
    # Check if we're being called as a decorator with arguments
    if target is None:
        # Return a decorator function
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                task_manager = get_task_manager()
                return task_manager.run_task(
                    target=func,
                    args=args,
                    kwargs=kwargs,
                    on_result=on_result,
                    on_error=on_error,
                    on_progress=on_progress,
                    name=name or func.__name__,
                    auto_cleanup=auto_cleanup
                )

            return wrapper

        return decorator

    # We're being called as a decorator without arguments
    @wraps(target)
    def wrapper(*args, **kwargs):
        task_manager = get_task_manager()
        return task_manager.run_task(
            target=target,
            args=args,
            kwargs=kwargs,
            on_result=on_result,
            on_error=on_error,
            on_progress=on_progress,
            name=name or target.__name__,
            auto_cleanup=auto_cleanup
        )

    return wrapper
