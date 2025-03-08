#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model pruning utilities for LLaDA.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def prune_model(model):
    """
    Prune model by removing unnecessary layers and components.
    
    Args:
        model: The model to prune
        
    Returns:
        Pruned model with reduced memory footprint
    """
    logger.info("Pruning model to reduce memory footprint")

    # Track original and pruned parameters
    original_params = sum(p.numel() for p in model.parameters())

    # Identify and prune attention heads
    attention_pruned = prune_attention_heads(model)

    # Prune intermediate layers
    ffn_pruned = prune_ffn_layers(model)

    # Count parameters after pruning
    pruned_params = sum(p.numel() for p in model.parameters())
    reduction = (original_params - pruned_params) / original_params * 100

    logger.info(f"Model pruned: {original_params:,} → {pruned_params:,} parameters ({reduction:.1f}% reduction)")
    logger.info(f"Pruned {attention_pruned} attention heads and {ffn_pruned} FFN components")

    return model


def prune_attention_heads(model, prune_ratio=0.3):
    """
    Prune attention heads to reduce memory usage.
    
    Args:
        model: The model to prune
        prune_ratio: Ratio of heads to prune (0.0-1.0)
        
    Returns:
        Number of heads pruned
    """
    pruned_count = 0

    # Find attention modules
    for name, module in model.named_modules():
        # Look for attention modules with multiple heads
        if ('attention' in name.lower() or 'attn' in name.lower()) and hasattr(module, 'num_heads'):
            num_heads = module.num_heads
            if num_heads <= 1:
                continue  # Skip if only one head

            # Calculate number of heads to prune
            heads_to_prune = max(1, int(num_heads * prune_ratio))

            # For safety, never prune more than half
            heads_to_prune = min(heads_to_prune, num_heads // 2)

            if heads_to_prune == 0:
                continue

            # Select heads to prune (assuming less important heads are at the beginning)
            indices = list(range(heads_to_prune))

            # Actually prune the heads
            try:
                head_size = module.head_size

                # Get the current weights
                if hasattr(module, 'q_proj'):
                    # Extract query, key, value weights
                    q_weight = module.q_proj.weight.data
                    k_weight = module.k_proj.weight.data
                    v_weight = module.v_proj.weight.data
                    o_weight = module.o_proj.weight.data

                    # Calculate sizes
                    hidden_size = q_weight.size(1)

                    # Create masks for heads to keep
                    keep_mask = torch.ones(num_heads, dtype=torch.bool, device=q_weight.device)
                    keep_mask[indices] = False

                    # Reshape to access heads
                    q_reshaped = q_weight.view(num_heads, head_size, hidden_size)
                    k_reshaped = k_weight.view(num_heads, head_size, hidden_size)
                    v_reshaped = v_weight.view(num_heads, head_size, hidden_size)
                    o_reshaped = o_weight.view(hidden_size, num_heads, head_size)

                    # Filter out pruned heads
                    q_pruned = q_reshaped[keep_mask]
                    k_pruned = k_reshaped[keep_mask]
                    v_pruned = v_reshaped[keep_mask]
                    o_pruned = o_reshaped[:, keep_mask, :]

                    # Update module parameters
                    new_num_heads = num_heads - heads_to_prune
                    module.num_heads = new_num_heads

                    # Update weights
                    module.q_proj.weight.data = q_pruned.reshape(-1, hidden_size)
                    module.k_proj.weight.data = k_pruned.reshape(-1, hidden_size)
                    module.v_proj.weight.data = v_pruned.reshape(-1, hidden_size)
                    module.o_proj.weight.data = o_pruned.reshape(hidden_size, -1)

                    # Update bias if present
                    if hasattr(module, 'q_proj.bias') and module.q_proj.bias is not None:
                        q_bias = module.q_proj.bias.data
                        k_bias = module.k_proj.bias.data
                        v_bias = module.v_proj.bias.data

                        # Reshape to access heads
                        q_bias_reshaped = q_bias.view(num_heads, head_size)
                        k_bias_reshaped = k_bias.view(num_heads, head_size)
                        v_bias_reshaped = v_bias.view(num_heads, head_size)

                        # Filter out pruned heads
                        q_bias_pruned = q_bias_reshaped[keep_mask]
                        k_bias_pruned = k_bias_reshaped[keep_mask]
                        v_bias_pruned = v_bias_reshaped[keep_mask]

                        # Update biases
                        module.q_proj.bias.data = q_bias_pruned.reshape(-1)
                        module.k_proj.bias.data = k_bias_pruned.reshape(-1)
                        module.v_proj.bias.data = v_bias_pruned.reshape(-1)

                pruned_count += heads_to_prune
                logger.info(f"Pruned {heads_to_prune} heads from {name}")

            except Exception as e:
                logger.warning(f"Failed to prune heads in {name}: {e}")

    return pruned_count


def prune_ffn_layers(model, prune_ratio=0.2):
    """
    Prune feed-forward network layers to reduce memory usage.
    
    Args:
        model: The model to prune
        prune_ratio: Ratio of FFN neurons to prune (0.0-1.0)
        
    Returns:
        Number of FFN components pruned
    """
    pruned_count = 0

    # Find FFN modules
    for name, module in model.named_modules():
        # Look for MLP or FFN modules
        if ('mlp' in name.lower() or 'ffn' in name.lower() or 'feed_forward' in name.lower()):
            # Find the up and down projection layers (expansion and contraction)
            up_proj = None
            down_proj = None

            # Look for the projection layers
            for child_name, child in module.named_children():
                if any(x in child_name.lower() for x in ['up_proj', 'fc1', 'w1', 'gate']):
                    up_proj = child
                elif any(x in child_name.lower() for x in ['down_proj', 'fc2', 'w2', 'output']):
                    down_proj = child

            if up_proj is None or down_proj is None:
                continue

            # Get layer dimensions
            if not hasattr(up_proj, 'weight') or not hasattr(down_proj, 'weight'):
                continue

            up_weight = up_proj.weight.data
            down_weight = down_proj.weight.data

            hidden_dim = up_weight.size(0)  # Output dimension of up projection
            input_dim = up_weight.size(1)  # Input dimension of up projection

            # Calculate number of neurons to prune
            neurons_to_prune = max(1, int(hidden_dim * prune_ratio))

            # For safety, never prune more than 30% of neurons
            neurons_to_prune = min(neurons_to_prune, int(hidden_dim * 0.3))

            if neurons_to_prune == 0:
                continue

            # Find least important neurons
            # Use L2 norm of weights as importance metric
            importance = torch.norm(up_weight, dim=1) * torch.norm(down_weight, dim=0)
            _, indices = torch.topk(importance, hidden_dim - neurons_to_prune, largest=True)

            # Create mask for neurons to keep
            keep_mask = torch.zeros(hidden_dim, dtype=torch.bool, device=up_weight.device)
            keep_mask[indices] = True

            # Prune up and down projections
            try:
                up_proj.weight.data = up_weight[keep_mask]
                down_proj.weight.data = down_weight[:, keep_mask]

                # Update biases if present
                if hasattr(up_proj, 'bias') and up_proj.bias is not None:
                    up_proj.bias.data = up_proj.bias.data[keep_mask]

                # Update output dimension of up projection
                up_proj.out_features = len(indices)

                # Update input dimension of down projection
                down_proj.in_features = len(indices)

                pruned_count += 1
                logger.info(f"Pruned {neurons_to_prune} neurons from {name}")

            except Exception as e:
                logger.warning(f"Failed to prune FFN in {name}: {e}")

    return pruned_count


def quantize_model(model, bits=8):
    """
    Apply quantization to model weights.
    
    Args:
        model: The model to quantize
        bits: Bit precision (4, 8, or 16)
        
    Returns:
        Quantized model
    """
    logger.info(f"Quantizing model to {bits}-bit precision")

    if bits == 16:
        # Half precision
        model = model.half()
        return model

    if bits not in [4, 8]:
        logger.warning(f"Unsupported bit precision: {bits}. Using 8 bits.")
        bits = 8

    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear8bitLt, Linear4bit

        # Replace linear layers with quantized versions
        modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))

        # Replace the layers
        for name, module in modules_to_replace:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            # Get parent module
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            # Create quantized replacement
            if bits == 8:
                # 8-bit Linear
                quantized = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
            else:
                # 4-bit Linear with double quantization
                quantized = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4"
                )

            # Copy weights
            with torch.no_grad():
                if bits == 8:
                    quantized.weight.data = module.weight.data.to(quantized.weight.dtype)
                else:
                    # For 4-bit, the bnb module handles conversion
                    quantized.weight.data = module.weight.data

                if module.bias is not None:
                    quantized.bias.data = module.bias.data

            # Replace the module
            setattr(parent, child_name, quantized)

        logger.info(f"Quantized {len(modules_to_replace)} linear layers to {bits} bits")
        return model

    except ImportError:
        logger.warning("bitsandbytes not available. Skipping quantization.")
        return model
