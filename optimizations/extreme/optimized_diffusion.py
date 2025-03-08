#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized diffusion process for LLaDA GUI.
"""

import numpy as np
import torch


class OptimizedDiffusionGenerator:
    """Optimized implementation of the diffusion process for LLaDA."""

    def __init__(self, model, tokenizer, offloading_manager=None, device="cuda"):
        """
        Initialize the optimized diffusion generator.
        
        Args:
            model: The LLaDA model
            tokenizer: The tokenizer
            offloading_manager: OffloadingManager instance (optional)
            device: Device to use for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.offloading_manager = offloading_manager
        self.device = device
        self.mask_id = 126336  # LLaDA mask token ID

    def add_gumbel_noise(self, logits, temperature):
        """Add Gumbel noise to logits for sampling."""
        if isinstance(logits, torch.Tensor):
            logits = logits.to(torch.float64)
            noise = torch.rand_like(logits, dtype=torch.float64)
            gumbel_noise = (- torch.log(noise)) ** temperature
            return logits.exp() / gumbel_noise
        else:
            # Handle numpy arrays
            logits = logits.astype(np.float64)
            noise = np.random.random(logits.shape).astype(np.float64)
            gumbel_noise = (-np.log(noise)) ** temperature
            return np.exp(logits) / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        """Calculate number of tokens to transfer at each step."""
        # Sum along sequence dimension
        mask_num = mask_index.sum(dim=1, keepdim=True)

        # Compute base number of tokens per step
        base = mask_num // steps
        remainder = mask_num % steps

        # Allocate tokens evenly
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        num_transfer_tokens += base

        # Distribute remainder
        for i in range(mask_num.shape[0]):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    def generate(self, prompt, steps=64, gen_length=64, block_length=32,
                 temperature=0., cfg_scale=0., remasking='low_confidence',
                 progress_callback=None, step_update_callback=None,
                 memory_efficient=True):
        """
        Generate text using optimized diffusion process.
        
        Args:
            prompt: Input text or token IDs
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for semi-autoregressive generation
            temperature: Temperature for sampling
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy
            progress_callback: Callback for progress updates
            step_update_callback: Callback for visualization updates
            memory_efficient: Whether to use memory-efficient generation
            
        Returns:
            Generated text
        """
        # Tokenize input if it's text
        if isinstance(prompt, str):
            m = [{"role": "user", "content": prompt}]
            user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = self.tokenizer(user_input)['input_ids']
            prompt_tensor = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        else:
            # Handle case where prompt is already tokenized
            prompt_tensor = prompt.clone()
            if len(prompt_tensor.shape) == 1:
                prompt_tensor = prompt_tensor.unsqueeze(0)

        # Progress updates
        if progress_callback:
            progress_callback(5, "Starting generation...", {})

        # Create full sequence with masks
        x = torch.full((1, prompt_tensor.shape[1] + gen_length), self.mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_tensor.shape[1]] = prompt_tensor.clone()

        # Track prompt positions
        prompt_index = (x != self.mask_id)

        # Calculate blocks and steps
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block = steps // num_blocks

        # Track global steps for visualization
        global_step = 0
        total_steps = steps_per_block * num_blocks

        # Process each block sequentially
        for num_block in range(num_blocks):
            if progress_callback:
                progress_callback(
                    5 + int(90 * num_block / num_blocks),
                    f"Processing block {num_block + 1}/{num_blocks}",
                    {}
                )

            # Define block boundaries
            block_start = prompt_tensor.shape[1] + num_block * block_length
            block_end = prompt_tensor.shape[1] + (num_block + 1) * block_length

            # Only consider masks in current block
            block_mask_index = (x[:, block_start:block_end] == self.mask_id)

            # Calculate token transfer schedule
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

            # Process each step
            for i in range(steps_per_block):
                # Identify masked positions
                mask_index = (x == self.mask_id)

                # Clean up memory if needed
                if memory_efficient and self.device == "cuda":
                    torch.cuda.empty_cache()

                # Prepare model if using offloading
                if self.offloading_manager:
                    self.offloading_manager.prepare_for_inference()

                # Handle classifier-free guidance if enabled
                if cfg_scale > 0.:
                    # Create unconditional version with masked prompt
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id

                    # Combine conditional and unconditional inputs
                    x_combined = torch.cat([x, un_x], dim=0)

                    # Get logits from model
                    with torch.no_grad():
                        outputs = self.model(x_combined)

                    # Split outputs
                    logits, un_logits = torch.chunk(outputs.logits, 2, dim=0)

                    # Apply classifier-free guidance
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                    # Clean up
                    del un_x, un_logits, x_combined
                    if memory_efficient and self.device == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # Standard forward pass
                    with torch.no_grad():
                        outputs = self.model(x)
                        logits = outputs.logits

                # Add Gumbel noise for sampling
                logits_with_noise = self.add_gumbel_noise(logits, temperature)

                # Get most likely tokens
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Calculate confidence for remasking
                if remasking == 'low_confidence':
                    p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Don't remask tokens beyond current block
                x0_p[:, block_end:] = float('-inf')

                # Replace masked tokens with predictions
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(float('-inf'), device=x0.device))

                # Select tokens to keep based on confidence
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    if num_transfer_tokens[j, i] > 0:
                        # Get indices of top k confidence values
                        topk_values, topk_indices = torch.topk(
                            confidence[j], k=int(num_transfer_tokens[j, i]))
                        transfer_index[j, topk_indices] = True

                # Update tokens
                x[transfer_index] = x0[transfer_index]

                # Update step counter
                global_step += 1

                # Provide visualization update if callback is provided
                if step_update_callback:
                    try:
                        # Get tokens and mask status for the generated part
                        output_ids = x[0, prompt_tensor.shape[1]:].cpu().tolist()
                        mask_status = [id == self.mask_id for id in output_ids]

                        # Decode tokens
                        token_texts = []
                        for id in output_ids:
                            if id == self.mask_id:
                                token_texts.append("[MASK]")
                            else:
                                text = self.tokenizer.decode([id])
                                token_texts.append(text)

                        # Calculate confidences for visualization
                        confs = []
                        for idx, (is_masked, token_id) in enumerate(zip(mask_status, output_ids)):
                            if is_masked:
                                confs.append(0.0)
                            else:
                                idx_in_full = prompt_tensor.shape[1] + idx
                                if idx_in_full < x0_p.shape[1]:
                                    conf_val = x0_p[0, idx_in_full].cpu().item()
                                    if conf_val == float('-inf'):
                                        confs.append(0.5)
                                    else:
                                        confs.append(float(conf_val))
                                else:
                                    confs.append(0.5)

                        # Call visualization callback
                        step_update_callback(global_step, token_texts, mask_status, confs)

                        # Update progress if callback provided
                        if progress_callback:
                            progress_pct = 5 + int((num_block * steps_per_block + i + 1) / total_steps * 90)

                            # Get partial output
                            output_array = x[0, prompt_tensor.shape[1]:]
                            unmasked = output_array[output_array != self.mask_id]
                            partial_output = self.tokenizer.decode(unmasked, skip_special_tokens=True)

                            progress_callback(
                                progress_pct,
                                f"Step {global_step}/{total_steps} - Block {num_block + 1}/{num_blocks}",
                                {'partial_output': partial_output}
                            )
                    except Exception as viz_error:
                        print(f"Visualization error: {viz_error}")

                # Check memory and offload if needed
                if self.offloading_manager:
                    self.offloading_manager.check_memory()

        # Final cleanup
        if memory_efficient and self.device == "cuda":
            torch.cuda.empty_cache()

        # Return the completed sequence
        return x
