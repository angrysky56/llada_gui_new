#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker thread for running LLaDA model generation.
"""

import os
import sys
import gc
import torch
import torch.nn.functional as F
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import CRITICAL_GPU_MEMORY_THRESHOLD
from utils import cleanup_gpu_memory, get_model_path, format_error

class LLaDAWorker(QThread):
    """Worker thread for handling LLaDA generation."""
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)  # step, tokens, masks, confidences
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    
    def __init__(self, prompt, config, parent=None):
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.stopped = False
        
    def stop(self):
        """Stop the generation process."""
        self.stopped = True
    
    def run(self):
        try:
            # Only import these here to avoid loading the model at startup
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Check CUDA memory before starting
            if torch.cuda.is_available() and self.config['device'] == 'cuda':
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used_memory = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
                free_memory = total_memory - used_memory
                
                if free_memory < 1.0:  # Less than 1GB free
                    self.memory_warning.emit(
                        f"Low GPU memory warning: Only {free_memory:.2f}GB available. "
                        f"Consider using CPU mode or reducing model parameters."
                    )
                    
                    # If critically low, switch to CPU automatically
                    if free_memory < CRITICAL_GPU_MEMORY_THRESHOLD:  # Less than 300MB
                        self.progress.emit(5, "Critically low GPU memory, switching to CPU", {})
                        self.config['device'] = 'cpu'
                        self.config['use_8bit'] = False
                        self.config['use_4bit'] = False
            
            # Find the generate.py module
            try:
                # Import the generate function
                from generate import generate
            except ImportError as e:
                self.error.emit(f"Could not import generation module: {str(e)}")
                return
            
            # Set device according to configuration
            device = self.config['device']
            
            # Report progress
            self.progress.emit(5, f"Loading tokenizer... (device: {device})", {})
            
            # Clear CUDA cache if using GPU
            if device == 'cuda' and torch.cuda.is_available():
                cleanup_gpu_memory()
            
            # Get model path
            model_path = get_model_path()
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                self.progress.emit(15, f"Loading model... (device: {device})", {})
                
                # Load model with memory optimizations if needed
                model_load_params = {"trust_remote_code": True}
                
                # Different quantization options for memory efficiency
                using_quantization = False
                if self.config.get('use_4bit', False) and device == 'cuda':
                    if not self.stopped:
                        self.progress.emit(18, "Applying 4-bit quantization...", {})
                    model_load_params.update({
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "device_map": "auto"  # Let the library handle device placement
                    })
                    using_quantization = True
                elif self.config.get('use_8bit', False) and device == 'cuda':
                    if not self.stopped:
                        self.progress.emit(18, "Applying 8-bit quantization...", {})
                    model_load_params.update({
                        "load_in_8bit": True,
                        "device_map": "auto"  # Let the library handle device placement
                    })
                    using_quantization = True
                else:
                    # Use lower precision for GPU but full precision for CPU
                    model_load_params["torch_dtype"] = torch.bfloat16 if device == 'cuda' else torch.float32
                
                # Check if thread should stop before loading the model (which is memory-intensive)
                if self.stopped:
                    self.error.emit("Generation cancelled.")
                    return
                
                # Load model
                model = AutoModel.from_pretrained(
                    model_path, 
                    **model_load_params
                )
                
                # Only move the model to device if not using quantization
                # For quantized models, the device placement is handled by the device_map parameter
                if not using_quantization:
                    model = model.to(device)
                
                # Set model to eval mode
                model = model.eval()
                
            except Exception as e:
                # If we hit an OOM error, try to recover by clearing memory and using CPU
                if "CUDA out of memory" in str(e) or "DeviceError" in str(e):
                    # Clear GPU memory
                    cleanup_gpu_memory()
                    
                    self.progress.emit(20, "GPU memory error, falling back to CPU", {})
                    
                    # Fall back to CPU - no quantization for CPU
                    model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float32  # Use full precision for CPU
                    ).to('cpu').eval()
                    
                    # Update device
                    device = 'cpu'
                elif "not supported for" in str(e) and ("4-bit" in str(e) or "8-bit" in str(e)):
                    # Handle error with quantized models
                    self.progress.emit(20, "Issue with quantization, trying without quantization", {})
                    
                    # Try again without quantization
                    if device == 'cuda':
                        model = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16
                        ).to(device).eval()
                    else:
                        model = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.float32
                        ).to(device).eval()
                else:
                    # Re-raise other errors
                    raise
            
            self.progress.emit(30, "Tokenizing input...", {})
            
            # Prepare input according to chat.py
            m = [{"role": "user", "content": self.prompt}]
            user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(user_input)['input_ids']
            
            # Put input tensor on the correct device
            try:
                # Try to get the model's device (for quantized models)
                model_device = model.device
                input_ids = torch.tensor(input_ids).to(model_device).unsqueeze(0)
            except:
                # Fall back to the configured device
                input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
            
            # Setup parameters from config
            gen_length = self.config['gen_length']
            steps = self.config['steps']
            block_length = self.config['block_length']
            temperature = self.config['temperature']
            cfg_scale = self.config['cfg_scale']
            remasking = self.config['remasking']
            
            # Safety check for out-of-memory issues (reduce parameters if necessary)
            if device == 'cuda' and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                used_memory = (torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024**3)
                free_memory = total_memory - used_memory
                
                # Estimated memory needed (rough approximation)
                token_factor = 128 / 32  # baseline ratio
                estimated_memory = (gen_length / 128) * (steps / 128) * token_factor
                
                if estimated_memory > free_memory * 0.8:
                    # Reduce parameters to fit in memory
                    original_length = gen_length
                    original_steps = steps
                    
                    reduction_factor = min(1.0, (free_memory * 0.8) / estimated_memory)
                    gen_length = max(32, int(gen_length * reduction_factor))
                    gen_length = gen_length - (gen_length % block_length)  # Ensure divisible by block length
                    steps = max(32, int(steps * reduction_factor))
                    
                    self.progress.emit(35, 
                        f"Reduced parameters for memory safety (length: {original_length}->{gen_length}, steps: {original_steps}->{steps})", 
                        {}
                    )
            
            self.progress.emit(40, f"Starting generation with {steps} steps...", {
                'prompt_length': input_ids.shape[1],
                'params': {
                    'gen_length': gen_length,
                    'steps': steps,
                    'block_length': block_length,
                    'device': device,
                    'temperature': temperature,
                    'cfg_scale': cfg_scale,
                    'remasking': remasking
                }
            })
            
            # Define a custom generation function with tracking
            @torch.no_grad()
            def monitored_generate(model, prompt, steps=128, gen_length=128, block_length=128, 
                                temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
                """Modified generation function that sends signals during the diffusion process."""
                # Determine the correct device - for quantized models, we need to use model.device
                try:
                    target_device = model.device
                except:
                    # Fall back to device from config
                    target_device = device
                
                try:
                    # Create full tensor with masks on the model's device
                    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(target_device)
                    # Set prompt portion
                    x[:, :prompt.shape[1]] = prompt.clone()
                except Exception as e:
                    # If device transfer fails (common with quantized models), create it on the prompt's device
                    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
                    x[:, :prompt.shape[1]] = prompt.clone()
                
                prompt_index = (x != mask_id)
                
                assert gen_length % block_length == 0
                num_blocks = gen_length // block_length
                
                assert steps % num_blocks == 0
                steps_per_block = steps // num_blocks
                
                # Function to get number of tokens to transfer at each step
                def get_num_transfer_tokens(mask_index, steps):
                    mask_num = mask_index.sum(dim=1, keepdim=True)
                    base = mask_num // steps
                    remainder = mask_num % steps
                    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
                    for i in range(mask_num.size(0)):
                        num_transfer_tokens[i, :remainder[i]] += 1
                    return num_transfer_tokens
                
                # Function to add Gumbel noise
                def add_gumbel_noise(logits, temperature):
                    logits = logits.to(torch.float64)
                    noise = torch.rand_like(logits, dtype=torch.float64)
                    gumbel_noise = (- torch.log(noise)) ** temperature
                    return logits.exp() / gumbel_noise
                
                # Global step counter for visualization
                global_step = 0
                total_steps = steps_per_block * num_blocks
                
                for num_block in range(num_blocks):
                    # Check if generation was stopped
                    if self.stopped:
                        break
                    
                    block_start = prompt.shape[1] + num_block * block_length
                    block_end = prompt.shape[1] + (num_block + 1) * block_length
                    
                    block_mask_index = (x[:, block_start:block_end] == mask_id)
                    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
                    
                    self.progress.emit(
                        40 + int(50 * num_block / num_blocks), 
                        f"Processing block {num_block+1}/{num_blocks}", 
                        {}
                    )
                    
                    for i in range(steps_per_block):
                        # Check if generation was stopped
                        if self.stopped:
                            break
                        
                        mask_index = (x == mask_id)
                        
                        # Handle classifier-free guidance
                        if cfg_scale > 0.:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            
                            # Clear unnecessary variables to save memory
                            if device == 'cuda':
                                torch.cuda.empty_cache()
                            
                            # For quantized models, make sure x_ is on the correct device
                            # as some quantized models don't support the .to() method for tensors
                            if x_.device != model.device:
                                try:
                                    x_ = x_.to(model.device)
                                except:
                                    # If to() method fails, the model is likely quantized
                                    # and the tensors need to match the device already
                                    pass
                            
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            
                            # Clear unnecessary variables
                            del un_x, un_logits
                            if device == 'cuda':
                                torch.cuda.empty_cache()
                        else:
                            # Clear unnecessary variables to save memory
                            if device == 'cuda':
                                torch.cuda.empty_cache()
                            
                            # For quantized models, make sure x is on the correct device
                            if x.device != model.device:
                                try:
                                    x = x.to(model.device)
                                except:
                                    # If to() method fails, the model is likely quantized
                                    # and the tensors need to match the device already
                                    pass
                                
                            logits = model(x).logits
                        
                        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                        
                        # Determine token confidences for remasking strategy
                        if remasking == 'low_confidence':
                            p = F.softmax(logits.to(torch.float64), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                            
                            # Clear unnecessary variables
                            del p
                        elif remasking == 'random':
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                        else:
                            raise NotImplementedError(remasking)
                        
                        # Don't remask tokens beyond the current block
                        x0_p[:, block_end:] = -np.inf
                        
                        # Replace masked tokens with predictions
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        
                        # Select tokens to keep (unmask)
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                            transfer_index[j, select_index] = True
                        
                        # Apply the new token values
                        x[transfer_index] = x0[transfer_index]
                        
                        # Clear unnecessary variables
                        del logits, logits_with_noise, confidence
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Emit signal for visualization update
                        global_step += 1
                        try:
                            # Get tokens and mask status for the generated part only
                            output_ids = x[0, prompt.shape[1]:].cpu().tolist()
                            mask_status = [id == mask_id for id in output_ids]
                            token_texts = [tokenizer.decode([id]) if id != mask_id else "[MASK]" for id in output_ids]
                            
                            # Calculate confidences for visualization
                            confs = []
                            for idx, (is_masked, token_id) in enumerate(zip(mask_status, output_ids)):
                                if is_masked:
                                    confs.append(0.0)
                                else:
                                    # Use x0_p if available, otherwise assign a default confidence
                                    idx_in_full = prompt.shape[1] + idx
                                    if idx_in_full < x0_p.shape[1]:
                                        conf_val = x0_p[0, idx_in_full].cpu().item()
                                        if conf_val == -np.inf:
                                            confs.append(0.5)
                                        else:
                                            confs.append(float(conf_val))
                                    else:
                                        confs.append(0.5)
                            
                            # Send update to visualization
                            self.step_update.emit(global_step, token_texts, mask_status, confs)
                            
                            # Also update progress information
                            progress_pct = 40 + int((global_step / total_steps) * 55)
                            current_output = tokenizer.decode(
                                x[0, prompt.shape[1]:][x[0, prompt.shape[1]:] != mask_id], 
                                skip_special_tokens=True
                            )
                            self.progress.emit(
                                progress_pct, 
                                f"Step {global_step}/{total_steps} - Block {num_block+1}/{num_blocks}",
                                {'partial_output': current_output}
                            )
                        except Exception as viz_error:
                            # Don't let visualization errors stop the generation
                            print(f"Visualization error: {viz_error}")
                
                return x
            
            # Use the monitored generate function
            self.progress.emit(40, "Starting diffusion process...", {})
            
            # If user cancelled during setup, don't start generation
            if self.stopped:
                self.error.emit("Generation cancelled.")
                return
            
            try:
                out = monitored_generate(
                    model, 
                    input_ids, 
                    steps=steps, 
                    gen_length=gen_length, 
                    block_length=block_length,
                    temperature=temperature, 
                    cfg_scale=cfg_scale, 
                    remasking=remasking
                )
                
                # Check if generation was stopped
                if self.stopped:
                    self.error.emit("Generation cancelled.")
                    return
                
                # Decode the output
                self.progress.emit(95, "Decoding output...", {})
                answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                
                self.progress.emit(100, "Generation complete", {'output': answer})
                self.finished.emit(answer)
            except Exception as gen_error:
                if "CUDA out of memory" in str(gen_error):
                    # Detailed recovery suggestion for OOM
                    self.error.emit(
                        f"GPU memory error during generation. Please try:\n"
                        f"1. Reducing generation length (currently {gen_length})\n"
                        f"2. Reducing number of steps (currently {steps})\n"
                        f"3. Using 8-bit or 4-bit quantization\n"
                        f"4. Switching to CPU mode\n\n"
                        f"Error details: {str(gen_error)}"
                    )
                else:
                    # Re-raise other errors
                    raise
            
            # Clean up memory
            del model
            cleanup_gpu_memory()
            
        except Exception as e:
            self.error.emit(format_error(e))
            
            # Additional cleanup
            cleanup_gpu_memory()
