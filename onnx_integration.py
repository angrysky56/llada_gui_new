#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX integration for LLaDA GUI.

This module provides integration between the LLaDA GUI and the ONNX runtime
for faster text generation.
"""

import os
import sys
import json
import time
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# ONNX-related imports
import onnx
import onnxruntime as ort

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llada_onnx_integration")

# LLaDA-specific imports
from config import DEFAULT_MODEL_PATH, LOCAL_MODEL_PATH


class ONNXLLaDARunner:
    """Interface for running LLaDA using ONNX Runtime."""
    
    def __init__(
        self, 
        onnx_model_path: str,
        use_gpu: bool = True,
        mask_id: int = 126336,
        tokenizer = None
    ):
        """
        Initialize the ONNX LLaDA runner.
        
        Args:
            onnx_model_path: Path to the ONNX model file
            use_gpu: Whether to use GPU for inference
            mask_id: Token ID for [MASK]
            tokenizer: Pre-loaded tokenizer (if available)
        """
        self.onnx_model_path = onnx_model_path
        self.mask_id = mask_id
        self.tokenizer = tokenizer
        
        # Set providers based on availability
        self.providers = []
        if use_gpu and torch.cuda.is_available():
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = "cuda"
        else:
            self.providers = ['CPUExecutionProvider']
            self.device = "cpu"
        
        # Initialize session
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize the ONNX Runtime session."""
        logger.info(f"Initializing ONNX Runtime session with providers: {self.providers}")
        
        # Create session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable parallel execution
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Create session
        try:
            self.session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options=session_options,
                providers=self.providers
            )
            
            # Get input and output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX Runtime session initialized successfully")
            logger.info(f"Input names: {self.input_names}")
            logger.info(f"Output names: {self.output_names}")
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime session: {e}")
            raise
    
    def add_gumbel_noise(self, logits, temperature):
        """
        Add Gumbel noise to logits for sampling.
        
        Args:
            logits: Input logits
            temperature: Temperature for sampling
            
        Returns:
            Logits with Gumbel noise applied
        """
        # Convert to double precision
        logits = logits.astype(np.float64)
        
        # Generate uniform random noise
        noise = np.random.random(logits.shape).astype(np.float64)
        
        # Apply Gumbel noise
        gumbel_noise = (-np.log(noise)) ** temperature
        
        return np.exp(logits) / gumbel_noise
    
    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Compute the number of tokens to transfer at each step.
        
        Args:
            mask_index: Boolean mask indicating which tokens are masked
            steps: Number of diffusion steps
            
        Returns:
            Number of tokens to transfer at each step
        """
        # Sum along sequence dimension
        mask_num = np.sum(mask_index, axis=1, keepdims=True)
        
        # Compute base number of tokens per step
        base = mask_num // steps
        remainder = mask_num % steps
        
        # Allocate tokens evenly
        num_transfer_tokens = np.zeros((mask_num.shape[0], steps), dtype=np.int64)
        num_transfer_tokens += base
        
        # Distribute remainder
        for i in range(mask_num.shape[0]):
            num_transfer_tokens[i, :remainder[i, 0]] += 1
        
        return num_transfer_tokens
    
    def generate(
        self,
        prompt_ids: List[int],
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence',
        progress_callback = None,
        step_update_callback = None
    ):
        """
        Generate text using the ONNX model.
        
        Args:
            prompt_ids: Input token IDs
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for semi-autoregressive generation
            temperature: Temperature for sampling
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
            progress_callback: Callback for progress updates
            step_update_callback: Callback for step updates for visualization
            
        Returns:
            Generated token IDs
        """
        if progress_callback:
            progress_callback(0, "Starting ONNX generation...", {})
        
        # Convert to NumPy array
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        
        # Create the full sequence with masks
        x = np.full((1, prompt_ids.shape[1] + gen_length), self.mask_id, dtype=np.int64)
        x[:, :prompt_ids.shape[1]] = prompt_ids.copy()
        
        # Identify prompt positions
        prompt_index = (x != self.mask_id)
        
        # Calculate number of blocks and steps per block
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block = steps // num_blocks
        
        # Global step counter
        global_step = 0
        total_steps = steps_per_block * num_blocks
        
        # Process each block
        for num_block in range(num_blocks):
            if progress_callback:
                progress_callback(
                    int(40 + (50 * num_block / num_blocks)),
                    f"Processing block {num_block+1}/{num_blocks}",
                    {}
                )
            
            # Define block boundaries
            block_start = prompt_ids.shape[1] + num_block * block_length
            block_end = prompt_ids.shape[1] + (num_block + 1) * block_length
            
            # Identify masked tokens in the current block
            block_mask_index = (x[:, block_start:block_end] == self.mask_id)
            
            # Calculate token transfer schedule
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            # Process each step
            for i in range(steps_per_block):
                # Identify masked positions
                mask_index = (x == self.mask_id)
                
                # Handle classifier-free guidance if enabled
                if cfg_scale > 0.0:
                    # Create unconditional version with masked prompt
                    un_x = x.copy()
                    un_x[prompt_index] = self.mask_id
                    
                    # Combine conditional and unconditional inputs
                    x_combined = np.concatenate([x, un_x], axis=0)
                    
                    # Run inference
                    ort_inputs = {"input_ids": x_combined}
                    ort_outputs = self.session.run(None, ort_inputs)
                    
                    # Split outputs
                    logits, un_logits = np.split(ort_outputs[0], 2, axis=0)
                    
                    # Apply classifier-free guidance
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # Run inference with normal input
                    ort_inputs = {"input_ids": x}
                    ort_outputs = self.session.run(None, ort_inputs)
                    logits = ort_outputs[0]
                
                # Apply Gumbel noise for sampling
                logits_with_noise = self.add_gumbel_noise(logits, temperature)
                
                # Get most likely tokens
                x0 = np.argmax(logits_with_noise, axis=-1)
                
                # Calculate confidence for remasking strategy
                if remasking == 'low_confidence':
                    # Apply softmax to get probabilities
                    max_logits = np.max(logits, axis=-1, keepdims=True)
                    exp_logits = np.exp(logits - max_logits)
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    
                    # Get probability of the selected tokens
                    batch_indices = np.arange(x0.shape[0])[:, None]
                    seq_indices = np.arange(x0.shape[1])[None, :]
                    x0_p = probs[batch_indices, seq_indices, x0]
                elif remasking == 'random':
                    # Use random values for confidence
                    x0_p = np.random.random(x0.shape)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")
                
                # Don't remask tokens beyond the current block
                x0_p[:, block_end:] = -np.inf
                
                # Replace masked tokens with predictions
                x0 = np.where(mask_index, x0, x)
                confidence = np.where(mask_index, x0_p, -np.inf)
                
                # Select tokens to keep based on confidence
                transfer_index = np.zeros_like(x0, dtype=bool)
                for j in range(confidence.shape[0]):
                    # Get indices of top k confidence values
                    select_indices = np.argsort(confidence[j])[::-1][:num_transfer_tokens[j, i]]
                    transfer_index[j, select_indices] = True
                
                # Update tokens
                x[transfer_index] = x0[transfer_index]
                
                # Update step counter
                global_step += 1
                
                # Provide visualization update if callback is provided
                if step_update_callback and self.tokenizer:
                    try:
                        # Get tokens and mask status for generated part
                        output_ids = x[0, prompt_ids.shape[1]:].tolist()
                        mask_status = [id == self.mask_id for id in output_ids]
                        
                        # Decode tokens
                        token_texts = []
                        for token_id in output_ids:
                            if token_id == self.mask_id:
                                token_texts.append("[MASK]")
                            else:
                                token_texts.append(self.tokenizer.decode([token_id]))
                        
                        # Calculate token confidences
                        confs = []
                        for idx, (is_masked, token_id) in enumerate(zip(mask_status, output_ids)):
                            if is_masked:
                                confs.append(0.0)
                            else:
                                # Get confidence from x0_p if available
                                idx_in_full = prompt_ids.shape[1] + idx
                                if idx_in_full < x0_p.shape[1]:
                                    conf_val = x0_p[0, idx_in_full]
                                    if conf_val == -np.inf:
                                        confs.append(0.5)
                                    else:
                                        confs.append(float(conf_val))
                                else:
                                    confs.append(0.5)
                        
                        # Call the visualization update callback
                        step_update_callback(global_step, token_texts, mask_status, confs)
                        
                        # Update progress
                        if progress_callback:
                            progress_pct = 40 + int((global_step / total_steps) * 55)
                            
                            # Get the current partial output
                            partial_output = ""
                            if self.tokenizer:
                                # Only decode the unmasked tokens
                                output_array = x[0, prompt_ids.shape[1]:]
                                unmasked_tokens = output_array[output_array != self.mask_id]
                                if len(unmasked_tokens) > 0:
                                    partial_output = self.tokenizer.decode(unmasked_tokens, skip_special_tokens=True)
                            
                            progress_callback(
                                progress_pct,
                                f"Step {global_step}/{total_steps} - Block {num_block+1}/{num_blocks}",
                                {'partial_output': partial_output}
                            )
                    except Exception as viz_error:
                        logger.error(f"Visualization error: {viz_error}")
        
        # Return the generated sequence
        return x


class ONNXModelManager:
    """Manager for ONNX models in the LLaDA GUI application."""
    
    def __init__(self, models_dir: str = "onnx_models"):
        """
        Initialize the ONNX model manager.
        
        Args:
            models_dir: Directory for storing ONNX models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cached_models = {}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available ONNX models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Check for model directories
        for model_dir in self.models_dir.glob("*"):
            if not model_dir.is_dir():
                continue
                
            # Look for config file
            config_path = model_dir / "onnx_config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    # Get model paths
                    model_paths = []
                    for key in ["onnx_model_path", "optimized_model_path", "quantized_model_path"]:
                        if key in config and config[key]:
                            path = Path(config[key])
                            if path.exists():
                                model_paths.append((key, str(path)))
                    
                    if model_paths:
                        # Add model info
                        models.append({
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "config": config,
                            "model_paths": model_paths
                        })
                except Exception as e:
                    logger.error(f"Error loading model config from {config_path}: {e}")
        
        return models
    
    def get_runner(
        self, 
        model_name: str, 
        use_quantized: bool = False,
        use_gpu: bool = True,
        tokenizer = None
    ) -> Optional[ONNXLLaDARunner]:
        """
        Get a runner for the specified model.
        
        Args:
            model_name: Name of the model
            use_quantized: Whether to use the quantized version if available
            use_gpu: Whether to use GPU for inference
            tokenizer: Pre-loaded tokenizer (if available)
            
        Returns:
            ONNXLLaDARunner instance or None if model not found
        """
        # Check cache first
        cache_key = f"{model_name}_{use_quantized}_{use_gpu}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key]
        
        # Find model
        model_dir = self.models_dir / model_name
        config_path = model_dir / "onnx_config.json"
        
        if not config_path.exists():
            logger.error(f"Model config not found: {config_path}")
            return None
        
        try:
            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Determine which model file to use
            if use_quantized and config.get("quantized_model_path"):
                model_path = config["quantized_model_path"]
            elif config.get("optimized_model_path"):
                model_path = config["optimized_model_path"]
            elif config.get("onnx_model_path"):
                model_path = config["onnx_model_path"]
            else:
                logger.error(f"No model path found in config: {config_path}")
                return None
            
            # Check if model file exists
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Create runner
            runner = ONNXLLaDARunner(
                onnx_model_path=model_path,
                use_gpu=use_gpu,
                tokenizer=tokenizer
            )
            
            # Cache runner
            self.cached_models[cache_key] = runner
            
            return runner
            
        except Exception as e:
            logger.error(f"Error creating ONNX runner for {model_name}: {e}")
            return None
    
    def convert_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        optimize: bool = True,
        quantize: bool = False,
        use_gpu: bool = True,
        progress_callback = None
    ) -> bool:
        """
        Convert a model to ONNX format.
        
        Args:
            model_name: Name for the converted model
            model_path: Path to the model to convert (default: use LLaDA default)
            optimize: Whether to optimize the model
            quantize: Whether to quantize the model
            use_gpu: Whether to use GPU for conversion
            progress_callback: Callback for progress updates
            
        Returns:
            True if conversion was successful, False otherwise
        """
        # Use default model path if not provided
        if model_path is None:
            if Path(LOCAL_MODEL_PATH).exists():
                model_path = LOCAL_MODEL_PATH
            else:
                model_path = DEFAULT_MODEL_PATH
        
        # Import converter here to avoid circular imports
        from onnx_converter import LLaDAONNXConverter
        
        # Create output directory
        output_dir = self.models_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(0, f"Starting ONNX conversion of {model_path}", {})
        
        try:
            # Create converter
            converter = LLaDAONNXConverter(
                model_path=model_path,
                output_dir=str(output_dir),
                quantize=quantize,
                use_gpu=use_gpu
            )
            
            # Load model
            if progress_callback:
                progress_callback(5, "Loading model", {})
            converter.load_model()
            
            # Convert to ONNX
            if progress_callback:
                progress_callback(30, "Converting to ONNX format", {})
            converter.convert_to_onnx()
            
            # Optimize
            if optimize:
                if progress_callback:
                    progress_callback(60, "Optimizing ONNX model", {})
                converter.optimize_onnx_model()
            
            # Quantize
            if quantize:
                if progress_callback:
                    progress_callback(80, "Quantizing ONNX model", {})
                converter.quantize_onnx_model()
            
            if progress_callback:
                progress_callback(100, "Conversion completed", {})
            
            return True
            
        except Exception as e:
            logger.error(f"Error during ONNX conversion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if progress_callback:
                progress_callback(100, f"Error: {str(e)}", {})
            
            return False


# Integration with the worker thread
def create_onnx_generator(
    tokenizer,
    onnx_model_path,
    use_gpu=True,
    progress_callback=None,
    step_update_callback=None
):
    """
    Create a function that mimics the generate function but uses ONNX Runtime.
    
    Args:
        tokenizer: Tokenizer to use for encoding/decoding
        onnx_model_path: Path to the ONNX model file
        use_gpu: Whether to use GPU for inference
        progress_callback: Callback for progress updates
        step_update_callback: Callback for step updates for visualization
        
    Returns:
        Function that takes the same parameters as the generate function
    """
    try:
        # Create ONNX runner
        runner = ONNXLLaDARunner(
            onnx_model_path=onnx_model_path,
            use_gpu=use_gpu,
            tokenizer=tokenizer
        )
        
        # Create generator function
        def onnx_generate(
            model,  # Ignored - using ONNX instead
            prompt,
            steps=128,
            gen_length=128,
            block_length=128,
            temperature=0.,
            cfg_scale=0.,
            remasking='low_confidence'
        ):
            # Extract token IDs from prompt tensor
            prompt_ids = prompt[0].cpu().numpy().tolist()
            
            # Generate using ONNX runner
            result = runner.generate(
                prompt_ids=prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                progress_callback=progress_callback,
                step_update_callback=step_update_callback
            )
            
            # Convert back to torch tensor
            return torch.tensor(result, device=prompt.device)
        
        return onnx_generate
    
    except Exception as e:
        logger.error(f"Error creating ONNX generator: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
