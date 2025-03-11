#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamic memory-aware worker for LLaDA model generation.

This module extends the base LLaDA worker with dynamic memory management capabilities,
providing optimized generation with automatic parameter adjustment based on available memory.
"""

import logging
import traceback

import torch
from PyQt6.QtCore import QThread, pyqtSignal

# Import memory management tools
from .memory_manager import get_memory_manager
from .memory_optimizer import (
    apply_model_optimizations,
    dynamic_memory_management,
    optimize_inference_params,
    CudaMemoryTracker
)

# Configure logging
logger = logging.getLogger(__name__)


class DynamicMemoryWorker(QThread):
    """
    Worker thread for LLaDA model generation with dynamic memory management.
    
    This class extends the base worker with memory optimization capabilities, providing
    more robust generation with automatic parameter adjustment based on available memory.
    """

    # Signals
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)

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
        self.original_config = config
        self.config = dict(config)  # Make a copy for modifications
        self.model = None
        self.tokenizer = None
        self.is_running = True

        # Get memory manager
        self.memory_manager = get_memory_manager()

        # Check memory compatibility and optimize parameters if needed
        self.check_memory_compatibility()

    def check_memory_compatibility(self):
        """
        Check if the current configuration is compatible with available memory.
        
        This will emit a memory warning signal if there are potential issues,
        and automatically adjust parameters for better memory usage.
        """
        # Only need to check for GPU compatibility
        if self.config.get('device', 'cuda') != 'cuda' or not torch.cuda.is_available():
            return

        # Get memory warning from manager
        warning = self.memory_manager.get_memory_warning(self.config)
        if warning:
            self.memory_warning.emit(warning)

        # Check if we need to optimize parameters based on memory constraints
        if not self.memory_manager.can_fit_in_memory(self.config):
            logger.warning("Configuration may not fit in memory, optimizing parameters")

            # Get optimized parameters
            original_config = dict(self.config)
            optimized_config = self.memory_manager.optimize_parameters(self.config)

            # Update configuration with optimized parameters
            self.config = optimized_config

            # Log changes
            self._log_parameter_changes(original_config, optimized_config)

            # Emit warning about parameter changes
            changes = []
            for param in ['gen_length', 'steps', 'block_length']:
                if param in original_config and param in optimized_config and original_config[param] != \
                        optimized_config[param]:
                    changes.append(f"{param}: {original_config[param]} → {optimized_config[param]}")

            if optimized_config.get('use_4bit', False) and not original_config.get('use_4bit', False):
                changes.append("Enabled 4-bit quantization")

            if optimized_config.get('use_8bit', False) and not original_config.get('use_8bit', False):
                changes.append("Enabled 8-bit quantization")

            if optimized_config.get('extreme_mode', False) and not original_config.get('extreme_mode', False):
                changes.append("Enabled extreme memory mode")

            if changes:
                self.memory_warning.emit(
                    "Parameters automatically adjusted to fit available memory:\n- " +
                    "\n- ".join(changes)
                )

    def _log_parameter_changes(self, original, optimized):
        """
        Log parameter changes between original and optimized configurations.
        
        Args:
            original: Original configuration
            optimized: Optimized configuration
        """
        changes = []
        for key in set(original.keys()).union(optimized.keys()):
            if key in original and key in optimized and original[key] != optimized[key]:
                changes.append(f"{key}: {original[key]} → {optimized[key]}")

        if changes:
            logger.info("Parameter optimizations applied: " + ", ".join(changes))

    def run(self):
        """Run the generation process."""
        try:
            # Force memory cleanup before starting
            self.memory_manager.force_cleanup()

            # Initialize model with memory tracking
            self.progress.emit(0, "Loading model...", {})
            with CudaMemoryTracker("Model initialization"):
                self._initialize_model()

            # Check for early termination
            if not self.is_running:
                return

            # Generate with dynamic memory management
            self.progress.emit(10, "Starting generation...", {})
            self._generate_with_memory_management()

        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            error_message = f"Error in generation: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_message)

    def _initialize_model(self):
        """Initialize the model and tokenizer with memory optimizations."""
        try:
            # Import LLaDA model - this would be imported from the actual LLaDA module
            # For now, just a placeholder as we're focusing on the memory management
            from core.generate import initialize_model

            # Initialize with optimized parameters
            self.model, self.tokenizer = initialize_model(
                device=self.config.get('device', 'cuda'),
                use_8bit=self.config.get('use_8bit', False),
                use_4bit=self.config.get('use_4bit', False),
                extreme_mode=self.config.get('extreme_mode', False)
            )

            # Apply memory optimizations
            self.model = apply_model_optimizations(self.model, self.config)

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    @dynamic_memory_management
    def _generate_with_memory_management(self):
        """Generate text with dynamic memory management."""
        try:
            # Import LLaDA generation function - this would be imported from the actual module
            # For now, just a placeholder as we're focusing on the memory management
            from core.generate import generate

            # Optimize parameters further if needed based on current memory
            runtime_config = optimize_inference_params(self.model, self.prompt, self.config)

            # Get generation parameters
            gen_length = runtime_config.get('gen_length', 64)
            steps = runtime_config.get('steps', 64)
            block_length = runtime_config.get('block_length', 32)
            temperature = runtime_config.get('temperature', 0.0)
            cfg_scale = runtime_config.get('cfg_scale', 0.0)
            remasking = runtime_config.get('remasking', 'low_confidence')

            # Set up progress reporting for steps
            def progress_callback(step, tokens=None, masks=None, confidences=None, partial_output=None):
                if not self.is_running:
                    return False  # Signal to stop generation

                # Calculate progress percentage
                progress = int((step + 1) / steps * 90) + 10  # Start from 10%

                # Update progress
                self.progress.emit(
                    progress,
                    f"Step {step + 1}/{steps}",
                    {'partial_output': partial_output or ""}
                )

                # Update visualization if data is available
                if tokens is not None and masks is not None and confidences is not None:
                    self.step_update.emit(step, tokens, masks, confidences)

                return True  # Continue generation

            # Start generation with the optimized parameters
            with CudaMemoryTracker("Text generation"):
                output = generate(
                    prompt=self.prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    gen_length=gen_length,
                    steps=steps,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    progress_callback=progress_callback
                )

            # Signal completion
            if self.is_running:
                self.progress.emit(100, "Generation complete", {})
                self.finished.emit(output)

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def stop(self):
        """Stop the generation process."""
        self.is_running = False
        logger.info("Generation stopped by user")


# Function to enhance the standard LLaDA worker with dynamic memory management
def enhance_llada_worker(original_worker_class):
    """
    Enhance the standard LLaDA worker with dynamic memory management.
    
    Args:
        original_worker_class: Original LLaDA worker class
        
    Returns:
        Enhanced worker class with dynamic memory management
    """

    class EnhancedLLaDAWorker(original_worker_class):
        """LLaDA worker enhanced with dynamic memory management."""

        def __init__(self, prompt, config, parent=None):
            # Initialize the memory manager first
            self.memory_manager = get_memory_manager()

            # Check memory compatibility and adjust parameters if needed
            optimized_config = self._check_memory_compatibility(config)

            # Initialize the parent class with optimized config
            super().__init__(prompt, optimized_config, parent)

        def _check_memory_compatibility(self, config):
            """
            Check if the configuration is compatible with available memory.
            
            Args:
                config: Generation configuration
                
            Returns:
                dict: Optimized configuration
            """
            # Only check for GPU compatibility
            if config.get('device', 'cuda') != 'cuda' or not torch.cuda.is_available():
                return config

            # Clone configuration
            optimized_config = dict(config)

            # Check if we need to optimize parameters
            if not self.memory_manager.can_fit_in_memory(config):
                logger.warning("Configuration may not fit in memory, optimizing parameters")
                optimized_config = self.memory_manager.optimize_parameters(config)

                # Emit warning about parameter changes if needed
                if hasattr(self, 'memory_warning'):
                    changes = []
                    for param in ['gen_length', 'steps', 'block_length']:
                        if param in config and param in optimized_config and config[param] != optimized_config[param]:
                            changes.append(f"{param}: {config[param]} → {optimized_config[param]}")

                    if optimized_config.get('use_4bit', False) and not config.get('use_4bit', False):
                        changes.append("Enabled 4-bit quantization")

                    if optimized_config.get('use_8bit', False) and not config.get('use_8bit', False):
                        changes.append("Enabled 8-bit quantization")

                    if changes:
                        self.memory_warning.emit(
                            "Parameters adjusted to fit available memory:\n- " +
                            "\n- ".join(changes)
                        )

            return optimized_config

        @dynamic_memory_management
        def run(self):
            """Run the enhanced generation process with memory management."""
            # Call the original run method with memory management
            super().run()

    return EnhancedLLaDAWorker
