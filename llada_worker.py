#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker thread for running LLaDA model generation.
"""

import os
import sys
import gc
import torch
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import logging

from config import CRITICAL_GPU_MEMORY_THRESHOLD
from utils import cleanup_gpu_memory, get_model_path, format_error
from generate import generate  # Import from our optimized generate.py

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLaDAWorker(QThread):
    """Worker thread for handling LLaDA generation."""
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    
    def __init__(self, prompt, config, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.stopped = False
        self.current_step = 0
        self.total_steps = config.get('steps', 64)
        self.mask_id = 126336  # Default mask token ID
        
    def stop(self):
        """Stop the generation process."""
        self.stopped = True
    
    def update_progress(self, progress_percentage, tokens=None):
        """
        Update progress callback from the generator.
        
        Args:
            progress_percentage: Float between 0 and 1 indicating progress
            tokens: Current token tensor
        """
        if self.stopped:
            return
        
        step = int(progress_percentage * 100)
        if step != self.current_step:
            self.current_step = step
            
            # Emit progress update
            self.progress.emit(
                step, 
                f"Generating: {step}% complete", 
                {'partial_progress': step}
            )
            
            # Emit step update for visualization if tokens are provided
            if tokens is not None:
                try:
                    # Extract token arrays for visualization
                    token_ids = tokens[0].cpu().tolist()
                    mask_indices = [1 if t == self.mask_id else 0 for t in token_ids]
                    
                    # Create visualization data in the expected format
                    token_display = []
                    for t in token_ids:
                        if t == self.mask_id:
                            token_display.append("[MASK]")
                        else:
                            token_display.append(str(t))
                    
                    # Generate confidence scores (1.0 for unmasked, 0.0 for masked)
                    confidence_scores = [0.0 if m else 1.0 for m in mask_indices]
                    
                    # Format mask indices as booleans (easier for visualization)
                    mask_indices_bool = [bool(m) for m in mask_indices]
                    
                    # Emit step update
                    self.step_update.emit(
                        self.current_step,       # Current step
                        token_display,           # Display tokens
                        mask_indices_bool,       # Mask indicators
                        confidence_scores        # Confidence scores
                    )
                except Exception as e:
                    logger.error(f"Error in step update: {e}")
    
    def run(self):
        try:
            # Import required modules
            from transformers import AutoTokenizer, AutoModel
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() and self.config['device'] == 'cuda' else 'cpu'
            
            # Report progress
            self.progress.emit(5, f"Starting with device: {device}", {})
            
            # Clear CUDA cache if using GPU
            if device == 'cuda':
                cleanup_gpu_memory()
                
                # Check if there's enough GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used_memory = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
                free_memory = total_memory - used_memory
                
                if free_memory < 1.0:
                    self.memory_warning.emit(
                        f"Low GPU memory warning: Only {free_memory:.2f}GB available. "
                        f"CPU offloading will be enabled."
                    )
            
            # Get model path
            model_path = get_model_path()
            
            try:
                # Load tokenizer
                self.progress.emit(10, "Loading tokenizer...", {})
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                # Find the mask token ID from the tokenizer
                if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                    self.mask_id = tokenizer.mask_token_id
                
                # Load model
                self.progress.emit(15, f"Loading model (device: {device})...", {})
                
                # Determine appropriate dtype
                if device == 'cuda':
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    dtype = torch.float32
                
                # Load model with appropriate settings - use device_map for better memory management
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
            
            # Enable CPU offloading by default for GPU
            cpu_offload = device == 'cuda'
            
            # Start generation
            self.progress.emit(40, f"Starting generation (steps: {steps}, length: {gen_length})...", {
                'prompt_length': input_ids.shape[1],
                'params': {
                    'gen_length': gen_length,
                    'steps': steps,
                    'block_length': block_length,
                    'temperature': temperature,
                    'cfg_scale': cfg_scale,
                    'remasking': remasking,
                    'device': device,
                    'cpu_offload': cpu_offload
                }
            })
            
            # Generate text with progress updates
            out = generate(
                model=model,
                prompt=input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                progress_callback=self.update_progress,
                cpu_offload=cpu_offload,
                mask_id=self.mask_id
            )
            
            # Check if generation was stopped
            if self.stopped:
                self.error.emit("Generation cancelled.")
                return
            
            # Decode the output
            self.progress.emit(95, "Decoding output...", {})
            answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            # Complete
            self.progress.emit(100, "Generation complete", {'output': answer})
            self.finished.emit(answer)
            
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            self.error.emit(format_error(e))
            
            # Additional cleanup
            try:
                del model
            except:
                pass
            
            if torch.cuda.is_available():
                cleanup_gpu_memory()
