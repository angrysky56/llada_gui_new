#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diffusion adapter for integrating Titan Memory with LLaDA.

This module provides the integration between the LLaDA diffusion process
and the Titan Memory system.
"""

import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Import Titan Memory
from .memory_guidance import TitanMemoryGuidance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def integrate_memory_with_diffusion(generate_func, memory_guidance, memory_weight=0.3):
    """
    Create a memory-integrated version of the generate function.
    
    Args:
        generate_func: Original generate function
        memory_guidance: TitanMemoryGuidance instance
        memory_weight: Weight for memory guidance
        
    Returns:
        Memory-integrated generate function
    """
    from functools import wraps

    @wraps(generate_func)
    def memory_integrated_generate(*args, **kwargs):
        # Update memory weight
        memory_guidance.set_memory_weight(kwargs.pop('memory_weight', memory_weight))

        # Extract tokenizer if available
        tokenizer = kwargs.pop('tokenizer', None)
        if tokenizer:
            memory_guidance.tokenizer = tokenizer

            # Update mask ID if available
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                memory_guidance.mask_id = tokenizer.mask_token_id

        # Ensure mask_id is not None (otherwise torch.full will fail)
        if memory_guidance.mask_id is None:
            logger.warning("Memory guidance has no mask_id, using default 126336")
            memory_guidance.mask_id = 126336

        # Make sure mask_id gets passed to the generate function
        if 'mask_id' not in kwargs or kwargs['mask_id'] is None:
            kwargs['mask_id'] = memory_guidance.mask_id

        # Add memory_integration to kwargs
        kwargs['memory_integration'] = memory_guidance

        # Call original generate function
        return generate_func(*args, **kwargs)

    return memory_integrated_generate


class MemoryGuidedDiffusionWorker(QThread):
    """Worker thread for running memory-guided LLaDA generation."""

    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    memory_update = pyqtSignal(np.ndarray)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)

    def __init__(self,
                 prompt,
                 config,
                 memory_guidance: Optional[TitanMemoryGuidance] = None,
                 parent=None):
        """Initialize the worker.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            memory_guidance: Memory guidance system (or None to create)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.memory_guidance = memory_guidance
        self.memory_weight = config.get('memory_weight', 0.3)
        self.stopped = False
        self.current_step = 0
        self.total_steps = config.get('steps', 64)
        self.memory_update_interval = 5  # Update memory every 5 steps

    def stop(self):
        """Stop the generation process."""
        self.stopped = True

    def run(self):
        """Run the memory-guided generation."""
        try:
            # Import required modules
            import torch
            from transformers import AutoTokenizer, AutoModel
            from core.utils import cleanup_gpu_memory, get_model_path

            # Determine device
            device = 'cuda' if torch.cuda.is_available() and self.config['device'] == 'cuda' else 'cpu'

            # Report progress
            self.progress.emit(5, f"Starting memory-guided generation (device: {device})", {})

            # Clear CUDA cache if using GPU
            if device == 'cuda':
                cleanup_gpu_memory()

            # Get model path
            model_path = get_model_path()

            try:
                # Load tokenizer
                self.progress.emit(10, "Loading tokenizer...", {})
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )

                # Initialize or update memory guidance
                if self.memory_guidance is None:
                    # Create memory guidance
                    from ..titan_memory import TitanMemorySystem
                    memory_system = TitanMemorySystem()
                    self.memory_guidance = TitanMemoryGuidance(
                        memory_system=memory_system,
                        memory_weight=self.memory_weight,
                        tokenizer=tokenizer
                    )
                else:
                    # Update existing guidance
                    self.memory_guidance.tokenizer = tokenizer
                    self.memory_guidance.set_memory_weight(self.memory_weight)

                # Update mask token ID
                if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                    self.memory_guidance.mask_id = tokenizer.mask_token_id

                # Emit initial memory state
                self.memory_update.emit(self.memory_guidance.get_memory_state())

                # Load model
                self.progress.emit(15, f"Loading model (device: {device})...", {})

                # Determine appropriate dtype - avoid bfloat16 due to compatibility issues
                if device == 'cuda':
                    # Use float16 instead of bfloat16 for better compatibility
                    dtype = torch.float16
                else:
                    dtype = torch.float32

                # Load model with appropriate settings
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto" if device == 'cuda' else None
                )

                # Move model to CPU if specified
                if device == 'cpu':
                    model = model.to('cpu')

                # Set model to evaluation mode
                model = model.eval()

                self.progress.emit(25, "Model loaded successfully", {})

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise

            # Prepare input
            self.progress.emit(30, "Tokenizing input...", {})
            m = [{"role": "user", "content": self.prompt}]
            user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(user_input)['input_ids']

            # Convert to tensor and move to appropriate device
            input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

            # Get generation parameters
            gen_length = self.config.get('gen_length', 64)
            steps = self.config.get('steps', 64)
            block_length = self.config.get('block_length', 32)
            temperature = self.config.get('temperature', 0.0)
            cfg_scale = self.config.get('cfg_scale', 0.0)
            remasking = self.config.get('remasking', 'low_confidence')

            # Enable CPU offloading by default
            cpu_offload = device == 'cuda'

            # Configure additional optimizations
            adaptive_steps = True
            chunk_size = 512
            confidence_threshold = 0.9

            # Progress tracking function
            def progress_callback(progress_percentage, tokens=None):
                if self.stopped:
                    return

                step = int(progress_percentage * 100)
                if step != self.current_step:
                    self.current_step = step

                    # Update memory state periodically
                    if step % self.memory_update_interval == 0 and tokens is not None:
                        # Update memory with current token sequence
                        token_sequence = tokens[0].cpu().numpy().tolist()
                        new_memory, _ = self.memory_guidance.update_memory_from_tokens(token_sequence)

                        # Emit memory update
                        self.memory_update.emit(new_memory)

                    # Emit progress update
                    self.progress.emit(
                        step,
                        f"Generating with memory guidance: {step}% complete",
                        {'partial_progress': step}
                    )

                    # Emit step update for visualization if tokens are provided
                    if tokens is not None:
                        try:
                            # Extract token arrays for visualization
                            token_ids = tokens[0].cpu().tolist()
                            mask_id = self.memory_guidance.mask_id
                            mask_indices = [1 if t == mask_id else 0 for t in token_ids]

                            # Create visualization data
                            token_display = []
                            for t in token_ids:
                                if t == mask_id:
                                    token_display.append("[MASK]")
                                else:
                                    token_display.append(str(t))

                            # Generate confidence scores (1.0 for unmasked, 0.0 for masked)
                            confidence_scores = [0.0 if m else 1.0 for m in mask_indices]

                            # Format mask indices as booleans (easier for visualization)
                            mask_indices_bool = [bool(m) for m in mask_indices]

                            # Emit step update
                            self.step_update.emit(
                                self.current_step,  # Current step
                                token_display,  # Display tokens
                                mask_indices_bool,  # Mask indicators
                                confidence_scores  # Confidence scores
                            )
                        except Exception as e:
                            logger.error(f"Error in step update: {e}")

            # Start generation
            self.progress.emit(40, "Starting memory-guided generation...", {})

            # Import the generate function
            from core.generate import generate

            # Create memory-integrated version
            memory_generate = integrate_memory_with_diffusion(
                generate,
                self.memory_guidance,
                self.memory_weight
            )

            # Generate text with memory guidance
            # Use a default mask_id (126336) if the memory guidance doesn't have one
            mask_id_to_use = self.memory_guidance.mask_id if self.memory_guidance.mask_id is not None else 126336

            out = memory_generate(
                model=model,
                prompt=input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                progress_callback=progress_callback,
                cpu_offload=cpu_offload,
                mask_id=mask_id_to_use,
                adaptive_steps=adaptive_steps,
                chunk_size=chunk_size,
                confidence_threshold=confidence_threshold,
                memory_weight=self.memory_weight,
                tokenizer=tokenizer
            )

            # Check if generation was stopped
            if self.stopped:
                self.error.emit("Generation cancelled.")
                return

            # Decode the output
            self.progress.emit(95, "Decoding output...", {})
            answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            # Update memory with final result
            try:
                # Train memory on the prompt-generation pair
                try:
                    loss = self.memory_guidance.train_on_generation(self.prompt, answer)
                    logger.info(f"Memory training loss: {loss}")
                except Exception as e:
                    logger.error(f"Error in memory training: {e}")
                    # Try fallback method without gradients
                    try:
                        with torch.no_grad():
                            loss = self.memory_guidance.train_on_generation(self.prompt, answer)
                            logger.info(f"Memory training loss (fallback): {loss}")
                    except Exception as e2:
                        logger.error(f"Fallback memory training also failed: {e2}")
                logger.info(f"Memory training loss: {loss}")

                # Emit final memory state
                self.memory_update.emit(self.memory_guidance.get_memory_state())
            except Exception as e:
                logger.error(f"Error updating memory: {e}")

            # Complete
            self.progress.emit(100, "Memory-guided generation complete", {'output': answer})
            self.finished.emit(answer)

        except Exception as e:
            logger.error(f"Error in memory-guided generation: {e}")
            self.error.emit(f"Memory-guided generation error: {str(e)}")

            # Additional cleanup
            try:
                del model
            except:
                pass

            if torch.cuda.is_available():
                cleanup_gpu_memory()
