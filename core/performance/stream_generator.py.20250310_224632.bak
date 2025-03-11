#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streaming generator for LLaDA model.

This module provides a streaming generator for the LLaDA model, enabling
progressive output generation with early display of partial results.
"""

import logging
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)


class StreamGenerator:
    """
    Stream generator for progressive output.
    
    This class wraps LLaDA's generation process to provide streaming output,
    where partial results are displayed as they become available.
    """

    def __init__(self, model=None, tokenizer=None, update_interval=0.1):
        """
        Initialize the stream generator.
        
        Args:
            model: LLaDA model (can be set later)
            tokenizer: LLaDA tokenizer (can be set later)
            update_interval: Interval in seconds for updates
        """
        self.model = model
        self.tokenizer = tokenizer
        self.update_interval = update_interval
        self.stop_flag = threading.Event()
        self.tokens_generated = []
        self.last_update_time = 0

    def set_model(self, model, tokenizer):
        """
        Set or update the model and tokenizer.
        
        Args:
            model: LLaDA model
            tokenizer: LLaDA tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def reset(self):
        """Reset the generator for a new generation."""
        self.stop_flag.clear()
        self.tokens_generated = []
        self.last_update_time = 0

    def stop(self):
        """Signal the generator to stop."""
        self.stop_flag.set()

    def stream_generate(self, prompt, config, callback=None):
        """
        Generate text with streaming output.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            callback: Callback function for updates (progress, status, data)
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be set before generation")

        # Reset for new generation
        self.reset()

        # Extract generation parameters
        gen_length = config.get('gen_length', 64)
        steps = config.get('steps', 64)
        block_length = config.get('block_length', 32)
        temperature = config.get('temperature', 0.0)
        cfg_scale = config.get('cfg_scale', 0.0)
        remasking = config.get('remasking', 'low_confidence')

        # We'll create a custom callback to intercept and forward updates
        def streaming_callback(step, tokens=None, masks=None, confidences=None, partial_output=None):
            # Check if we should stop
            if self.stop_flag.is_set():
                return False

            # Store tokens for incremental decoding
            if tokens is not None:
                self.tokens_generated = tokens

            # If we have decoded output, send it through the callback
            current_time = time.time()
            if partial_output is not None or (current_time - self.last_update_time >= self.update_interval):
                self.last_update_time = current_time

                # If no partial output provided, decode what we have
                if partial_output is None and self.tokens_generated:
                    try:
                        # This is a simplified example - actual implementation would depend on LLaDA-specific decoding
                        partial_output = prompt + "\n\n" + "Generating..."
                    except Exception as e:
                        logger.error(f"Error decoding partial output: {e}")

                # Calculate progress percentage
                progress = int((step + 1) / steps * 100)

                # Forward to the callback with streaming data
                if callback:
                    status = f"Step {step + 1}/{steps}"
                    data = {'partial_output': partial_output}
                    callback(progress, status, data)

                    # Also forward visualization data if provided
                    if tokens is not None and masks is not None and confidences is not None:
                        callback(progress, status, data, tokens, masks, confidences)

            # Continue generation
            return True

        # Call the LLaDA generation function with our streaming callback
        # For demonstration purposes, this is a simplified placeholder
        # In the actual implementation, you'd call the LLaDA specific generation function
        try:
            # Import the actual generation function
            from core.generate import generate

            # Call with the streaming callback
            final_output = generate(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                gen_length=gen_length,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                progress_callback=streaming_callback
            )

            # Final callback with complete output
            if callback and not self.stop_flag.is_set():
                callback(100, "Generation complete", {'partial_output': final_output})

            return final_output

        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            raise
