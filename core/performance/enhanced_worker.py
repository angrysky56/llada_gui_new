#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced worker for LLaDA with performance optimizations.

This module provides an enhanced worker for LLaDA that incorporates performance
optimizations including model caching and streaming output.
"""

import logging
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

# Import performance optimization components
from .model_cache import get_model_cache
from .stream_generator import StreamGenerator

# Try to import memory management if available
try:
    from core.memory_management.memory_manager import get_memory_manager

    HAS_MEMORY_MANAGER = True
except ImportError:
    HAS_MEMORY_MANAGER = False

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedLLaDAWorker(QThread):
    """
    Enhanced worker thread for LLaDA with performance optimizations.
    
    This worker incorporates model caching, streaming output, and memory optimizations
    to improve the performance of the LLaDA generation process.
    """

    # Progress signal - with an additional parameter for streaming visualization data
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    model_loaded = pyqtSignal()  # Signal when model is loaded

    def __init__(self, prompt, config, parent=None):
        """
        Initialize the worker.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            parent: Parent QObject
        """
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_running = True

        # Model cache for faster loading
        self.model_cache = get_model_cache()

        # Stream generator for progressive output
        self.stream_generator = StreamGenerator()

        # Memory manager if available
        self.memory_manager = get_memory_manager() if HAS_MEMORY_MANAGER else None

        # Performance metrics
        self.metrics = {
            'load_time': 0,
            'generation_time': 0,
            'total_time': 0,
            'cached_model': False
        }

    def run(self):
        """Run the generation process."""
        try:
            start_time = time.time()

            # Load or get model from cache
            load_start = time.time()
            self.progress.emit(0, "Loading model...", {})

            # Try to get from cache first
            self.model, self.tokenizer = self.model_cache.get(self.config)

            if self.model is None or self.tokenizer is None:
                # Not in cache, load the model
                self._load_model()
                # Put in cache for next time
                self.model_cache.put(self.config, self.model, self.tokenizer)
                self.metrics['cached_model'] = False
            else:
                self.metrics['cached_model'] = True

            self.model_loaded.emit()
            load_end = time.time()
            self.metrics['load_time'] = load_end - load_start

            # Check if we should continue
            if not self.is_running:
                return

            # Generate text with streaming
            self.progress.emit(10, "Starting generation...", {})

            # Set up streaming generator
            self.stream_generator.set_model(self.model, self.tokenizer)

            # Generate with streaming output
            generation_start = time.time()

            # Create a callback that emits both progress and visualization updates
            def streaming_callback(progress, status, data, tokens=None, masks=None, confidences=None):
                if not self.is_running:
                    return False

                # Emit progress update
                self.progress.emit(progress, status, data)

                # Emit visualization update if visualization data is provided
                if tokens is not None and masks is not None and confidences is not None:
                    step = int((progress - 10) / 90 * self.config.get('steps', 64))
                    self.step_update.emit(step, tokens, masks, confidences)

                return True

            # Generate text with streaming
            output = self.stream_generator.stream_generate(
                self.prompt,
                self.config,
                streaming_callback
            )

            generation_end = time.time()
            self.metrics['generation_time'] = generation_end - generation_start

            # Signal completion
            if self.is_running:
                self.progress.emit(100, "Generation complete", {})
                self.finished.emit(output)

            end_time = time.time()
            self.metrics['total_time'] = end_time - start_time
            logger.info(f"Generation metrics: {self.metrics}")

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            error_message = f"Error in generation: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_message)

    def _load_model(self):
        """Load the model and tokenizer."""
        # Import LLaDA model - this would be imported from the actual LLaDA module
        # For this example, just a placeholder
        from core.generate import initialize_model

        # Get model configuration parameters
        device = self.config.get('device', 'cuda')
        use_8bit = self.config.get('use_8bit', False)
        use_4bit = self.config.get('use_4bit', False)
        extreme_mode = self.config.get('extreme_mode', False)

        # Initialize with configuration
        self.model, self.tokenizer = initialize_model(
            device=device,
            use_8bit=use_8bit,
            use_4bit=use_4bit,
            extreme_mode=extreme_mode
        )

    def stop(self):
        """Stop the generation process."""
        self.is_running = False
        self.stream_generator.stop()
        logger.info("Generation stopped by user")


# Function to enhance the standard LLaDA worker
def enhance_llada_worker(original_worker_class):
    """
    Enhance the standard LLaDA worker with performance optimizations.
    
    Args:
        original_worker_class: The original worker class
        
    Returns:
        Enhanced worker class with performance optimizations
    """

    class OptimizedWorker(original_worker_class):
        """LLaDA worker with performance optimizations."""

        def __init__(self, prompt, config, parent=None):
            # Initialize the enhanced components
            self.model_cache = get_model_cache()
            self.stream_generator = StreamGenerator()
            self.cached_model = False
            self.load_time = 0

            # Memory manager if available
            if HAS_MEMORY_MANAGER:
                self.memory_manager = get_memory_manager()

            # Initialize the parent class
            super().__init__(prompt, config, parent)

        def run(self):
            """Run with performance optimizations."""
            try:
                start_time = time.time()

                # Try to get model from cache first
                self.progress.emit(0, "Loading model...", {})
                self.model, self.tokenizer = self.model_cache.get(self.config)

                # Model not in cache, load normally
                if self.model is None or self.tokenizer is None:
                    self.cached_model = False
                    load_start = time.time()
                    super().run()  # Call the original run method
                    load_end = time.time()
                    self.load_time = load_end - load_start

                    # Cache the model for next time if available
                    if hasattr(self, 'model') and hasattr(self, 'tokenizer') and self.model and self.tokenizer:
                        self.model_cache.put(self.config, self.model, self.tokenizer)
                else:
                    # Model was in cache, use stream generator for efficient generation
                    self.cached_model = True
                    self.stream_generator.set_model(self.model, self.tokenizer)

                    # Set up streaming callback
                    def streaming_callback(progress, status, data, tokens=None, masks=None, confidences=None):
                        if not self.is_running:
                            return False

                        # Emit progress update
                        self.progress.emit(progress, status, data)

                        # Emit visualization update if visualization data is provided
                        if tokens is not None and masks is not None and confidences is not None:
                            step = int((progress - 10) / 90 * self.config.get('steps', 64))
                            self.step_update.emit(step, tokens, masks, confidences)

                        return True

                    # Generate text with streaming
                    output = self.stream_generator.stream_generate(
                        self.prompt,
                        self.config,
                        streaming_callback
                    )

                    # Signal completion
                    if self.is_running:
                        self.progress.emit(100, "Generation complete", {})
                        self.finished.emit(output)

                end_time = time.time()
                logger.info(f"Total generation time: {end_time - start_time:.2f}s (cached model: {self.cached_model})")

            except Exception as e:
                logger.error(f"Error in enhanced worker: {str(e)}")
                error_message = f"Error in generation: {str(e)}\n\n{traceback.format_exc()}"
                self.error.emit(error_message)

        def stop(self):
            """Stop the generation process."""
            super().stop()
            if hasattr(self, 'stream_generator'):
                self.stream_generator.stop()

    return OptimizedWorker
