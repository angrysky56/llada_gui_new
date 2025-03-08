#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaDA Model Generation: worker thread
"""

import logging
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModel

from core.generate import generate  # Import from our optimized generate.py
from core.generation_mode import GenerationMode  # Import GenerationMode Enum
from core.utils import cleanup_gpu_memory, get_device_status, get_model_path, format_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MEMORY_WARNING_THRESHOLD_GB = 1.0
MEMORY_UNLOAD_THRESHOLD_GB = 2.0 # Threshold to trigger model unloading in GB

class LLaDAWorker(QThread):
    """Worker thread for handling LLaDA generation."""
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list, list, list) # Added step_confidences to signal
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    realtime_stats = pyqtSignal(dict) # Signal for realtime stats
    memory_influence_update = pyqtSignal(np.ndarray) # Signal for memory influence data
    cleanup_memory_signal = pyqtSignal() # Add this signal

    _model_cache = {} # Class-level model cache

    def __init__(self, prompt: str, config: Dict[str, Any], parent: Optional[Any] = None): # Type hints
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.stopped = False
        self.current_step = 0
        self.total_steps = config.get('steps', 64)
        self.mask_id = 126336  # Default mask token ID - Consider making configurable via config
        self.generation_mode = GenerationMode.STANDARD # Default to STANDARD mode
        self.last_step_time = None
        self.step_times = []
        self.tokenizer = None # Store tokenizer instance
        self.step_confidences = [] # To store step confidences


    def stop(self):
        """Stop the generation process."""
        self.stopped = True

    def update_progress(self, progress_percentage: float, tokens: Optional[torch.Tensor] = None):
        """
        Update progress callback from the generator.

        Args:
            progress_percentage: Float between 0 and 1 indicating progress
            tokens: Current token tensor
        """
        if self.stopped:
            return

        step = int(progress_percentage * 100)
        if step == self.current_step: # Prevent redundant updates
            return
        self.current_step = step

        # Emit progress update
        self.progress.emit(
            step,
            f"Generating: {step}% complete",
            {'partial_progress': step}
        )

        if tokens is None: # Early exit if no tokens for visualization
            return

        # Visualization data extraction - Moved visualization logic to a separate method
        self._emit_step_update_signal(tokens)
        self._emit_realtime_stats() # Emit realtime stats at each step
        self._emit_memory_influence_update() # Emit dummy memory influence data


    def _emit_step_update_signal(self, tokens: torch.Tensor):
        """Extract visualization data and emit step_update signal."""
        try:
            token_ids = tokens[0].cpu().tolist()
            mask_indices = [1 if t == self.mask_id else 0 for t in token_ids]
            #token_display = ["[MASK]" if t == self.mask_id else str(t) for t in token_ids] # List comprehension
            confidence_scores = [0.0 if m else 1.0 for m in mask_indices] # Initial confidence scores (can be refined later)
            mask_indices_bool = [bool(m) for m in mask_indices] # List comprehension

            # Decode token IDs to strings for better visualization
            token_display_strings = []
            for token_id in token_ids:
                if token_id == self.mask_id:
                    token_display_strings.append("[MASK]")
                else:
                    decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    # Handle empty decoded tokens (e.g., beginning of sequence tokens)
                    token_display_strings.append(decoded_token if decoded_token else "[UNK]")

            # Get current step's confidence scores from stored list
            current_step_confidences = []
            if self.step_confidences and len(self.step_confidences) > 0:
                current_step_confidences = self.step_confidences[-1].tolist()[0] # Get last step's confidences, convert to list

            self.step_update.emit(
                self.current_step,
                token_display_strings, # Use decoded token strings
                mask_indices_bool,
                confidence_scores, # Initial confidence scores
                token_ids, # Send token IDs as well
                current_step_confidences # Send per-step confidence scores
            )
        except Exception as e:
            logger.error(f"Error in step update: {e}")

    def _emit_realtime_stats(self):
        """Calculate and emit realtime statistics."""
        current_time = time.time()
        if self.last_step_time is not None:
            step_time = (current_time - self.last_step_time) * 1000 # in milliseconds
            self.step_times.append(step_time)
            if len(self.step_times) > 10: # Keep last 10 step times for moving average
                self.step_times.pop(0)
        self.last_step_time = current_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0


        tokens_per_sec = 0
        if avg_step_time > 0:
            tokens_per_sec = 1000 / avg_step_time # tokens per second, assuming 1 token per step

        memory_usage_str = "-"
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2) # MB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2) # MB
            memory_usage_str = f"{memory_allocated:.2f}MB allocated, {memory_reserved:.2f}MB reserved"


        stats = {
            "token_rate": f"{tokens_per_sec:.2f} tokens/s",
            "step_time": f"{avg_step_time:.2f}",
            "memory_usage": memory_usage_str
        }
        self.realtime_stats.emit(stats)

    def _emit_memory_influence_update(self):
        """Emit dummy memory influence data for visualization testing."""
        try:
            # Generate dummy 32x32 numpy array for memory influence - Structured data
            grid_resolution = 32
            x = np.linspace(-3, 3, grid_resolution)
            y = np.linspace(-3, 3, grid_resolution)
            X, Y = np.meshgrid(x, y)
            dummy_memory_influence_data = np.exp(-(X**2 + Y**2) / 2)  # 2D Gaussian as dummy data
            dummy_memory_influence_data = np.clip(dummy_memory_influence_data, 0, 1) # Ensure data in [0, 1]

            self.memory_influence_update.emit(dummy_memory_influence_data)

        except Exception as e:
            logger.error(f"Error generating dummy memory influence  {e}")


    def run(self):
        model = None # Initialize model and tokenizer outside try block for broader scope
        tokenizer = None
        try:
            # Determine device
            device = 'cuda' if torch.cuda.is_available() and self.config['device'] == 'cuda' else 'cpu'
            self.progress.emit(5, f"Starting with device: {device}", {})

            model_path = get_model_path()
            cache_key = (model_path, device)

            # Check if model is in cache
            if cache_key in LLaDAWorker._model_cache:
                self.progress.emit(15, "Using cached model...", {})
                model, tokenizer = LLaDAWorker._model_cache[cache_key]
                self.tokenizer = tokenizer # Store tokenizer instance in worker
            else:
                # Check GPU memory and unload if needed before loading new model
                if device == 'cuda':
                    available_memory_gb = get_device_status()['gpu_info'][0]['free_memory'] # Get free memory in GB
                    if available_memory_gb < MEMORY_UNLOAD_THRESHOLD_GB and LLaDAWorker._model_cache:
                        self._unload_cached_model()

                # Load tokenizer and model - Encapsulated loading into a separate method
                tokenizer, model = self._load_model_and_tokenizer(model_path, device)
                if model is None or tokenizer is None: # Handle loading failure
                    return # _load_model_and_tokenizer already emitted error
                self.tokenizer = tokenizer # Store tokenizer instance in worker
                # Store model in cache
                LLaDAWorker._model_cache[cache_key] = (model, tokenizer)


            # Prepare input - Input preparation into a separate method
            input_ids = self._prepare_input(tokenizer, device)
            if input_ids is None: # Handle input preparation failure
                return # _prepare_input already emitted error

            # Generation parameters - Extract parameters directly from config
            gen_length = self.config.get('gen_length', 64)
            steps = self.config.get('steps', 64)
            block_length = self.config.get('block_length', 32)
            temperature = self.config.get('temperature', 0.0)
            cfg_scale = self.config.get('cfg_scale', 0.0)
            remasking = self.config.get('remasking', 'low_confidence')
            fast_mode = self.config.get('fast_mode', True) # Still reading fast_mode for backward compatibility
            if fast_mode:
                self.generation_mode = GenerationMode.FAST
            else:
                self.generation_mode = GenerationMode.STANDARD

            cpu_offload = device == 'cuda' and self.generation_mode == GenerationMode.STANDARD # CPU offload only in STANDARD mode
            adaptive_steps = True
            chunk_size = 256 if self.generation_mode == GenerationMode.FAST else 512
            confidence_threshold = 0.8 if self.generation_mode == GenerationMode.FAST else 0.9


            self.progress.emit(40, f"Starting generation in {self.generation_mode.value} mode (steps: {steps}, length: {gen_length})...", {
                'prompt_length': input_ids.shape[1],
                'params': { # Params dict is cleaner
                    'gen_length': gen_length, 'steps': steps, 'block_length': block_length,
                    'temperature': temperature, 'cfg_scale': cfg_scale, 'remasking': remasking,
                    'device': device, 'cpu_offload': cpu_offload, 'generation_mode': self.generation_mode.value,
                    'adaptive_steps': adaptive_steps, 'chunk_size': chunk_size
                }
            })

            # Initialize step timing
            self.last_step_time = time.time()
            self.step_times = []
            self.step_confidences = [] # Initialize step_confidences list here

            # Generate text
            out, step_confidences_tensor = generate( # Capture step_confidences
                model=model, prompt=input_ids, steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=temperature, cfg_scale=cfg_scale,
                remasking=remasking, progress_callback=self.update_progress, cpu_offload=cpu_offload,
                mask_id=self.mask_id, adaptive_steps=adaptive_steps, chunk_size=chunk_size
            )
            self.step_confidences = [step_confidences_tensor[i, :, :] for i in range(step_confidences_tensor.shape[0])] # Store step confidences

            if self.stopped:
                self.error.emit("Generation cancelled.")
                return

            self.progress.emit(95, "Decoding output...", {})
            answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            self.progress.emit(100, "Generation complete", {'output': answer})
            self.finished.emit(answer)

            # Signal the main thread to perform cleanup (delayed, via timer)
            if not self.stopped:
                self.cleanup_memory_signal.emit()


        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            self.error.emit(format_error(e))


    def _unload_cached_model(self):
        """Unloads the cached model to free up GPU memory."""
        cache_key_to_unload = next(iter(LLaDAWorker._model_cache)) # Get the first key in cache (FIFO)
        if cache_key_to_unload:
            logger.info(f"Unloading cached model with key: {cache_key_to_unload} to free up memory.")
            model, tokenizer = LLaDAWorker._model_cache.pop(cache_key_to_unload) # Remove from cache
            del model
            del tokenizer
            cleanup_gpu_memory() # Clean up GPU memory
            logger.info("Cached model unloaded.")
        else:
            logger.warning("No model in cache to unload.")


    def _load_model_and_tokenizer(self, model_path: str, device: str) -> tuple[AutoTokenizer, AutoModel]:
        # dtype = torch.float16 if device == 'cuda' else torch.float32 # Consistent dtype setting
        #dtype = torch.float32
        dtype = torch.float16

        """Loads the tokenizer and model, emitting progress signals and handling errors."""
        try:
            self.progress.emit(10, "Loading tokenizer...", {})
            dmap = "auto" if device == 'cuda' else None
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                      torch_dtype=dtype,
                                                      device_map=dmap, trust_remote_code=True, use_fast=True, cache_dir="data")
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                self.mask_id = tokenizer.mask_token_id # Update mask_id from tokenizer if available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.progress.emit(15, f"Loading model (device: {device})...", {})

            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=dtype,
                device_map=dmap, cache_dir="data",
                low_cpu_mem_usage=True,

                #**( {'attn_implementation': 'flash_attention_2', #not supported yet
                #     'torch_compile': True} if device == 'cuda' else {} ) # Conditionally enable FlashAttention-2 and torch_compile
            )

            # Resize embeddings for special tokens if needed
            if len(tokenizer) > model.config.vocab_size:
                print("Resizing model embeddings to fit tokenizer vocabulary")
                model.resize_token_embeddings(len(tokenizer))
            # Try tying weights, but handle potential errors
            try:
                model.tie_weights()
            except AttributeError as e:
                print(f"Warning: Could not tie weights: {e}")

            if device == 'cpu':
                model = model.to('cpu')
            model.eval() # Ensure eval mode

            self.progress.emit(25, "Model loaded successfully", {})
            return tokenizer, model

        except Exception as e:
            error_msg = f"Error loading model: {format_error(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return None, None # Indicate loading failure


    def _prepare_input(self, tokenizer: AutoTokenizer, device: str) -> Optional[torch.Tensor]:
        """Tokenizes the input prompt and handles potential errors."""
        try:
            self.progress.emit(30, "Tokenizing input...", {})

            messages = [{"role": "user", "content": self.prompt}]

            user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(user_input)['input_ids'] # Consider pre-compiling tokenizer for speed
            input_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Create tensor once

            return input_ids_tensor

        except Exception as e:
            error_msg = f"Error preparing input: {format_error(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return None # Indicate input preparation failure

    def get_memory_usage(self): # Dummy implementation for now
        """Returns dummy memory usage statistics."""
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_percent = (gpu_used / gpu_total) * 100
            system_used = 10 # Dummy values
            system_total = 32 # Dummy values
            system_percent = (system_used / system_total) * 100
            return {
                'gpu_used': gpu_used,
                'gpu_reserved': gpu_reserved,
                'gpu_total': gpu_total,
                'gpu_percent': gpu_percent,
                'system_used': system_used,
                'system_total': system_total,
                'system_percent': system_percent,
                'gpu_available': True
            }
        else:
            return {
                'gpu_used': 0,
                'gpu_reserved': 0,
                'gpu_total': 0,
                'gpu_percent': 0,
                'system_used': 0,
                'system_total': 0,
                'system_percent': 0,
                'gpu_available': False
            }
