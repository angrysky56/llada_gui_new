#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Progressive model loading for LLaDA GUI.
"""

import gc
import logging

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


def progressive_loading(model_path, device="cuda", block_size=4, precision="int8"):
    """
    Load model progressively in blocks to reduce peak memory.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        block_size: Number of layers to load at once
        precision: Precision to use ("float16", "bfloat16", "int8", or "int4")
        
    Returns:
        Loaded model and tokenizer
    """
    logger.info(f"Loading model {model_path} progressively with block size {block_size}")

    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Get model configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Determine number of layers
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(config, "n_layer", None)
        if num_layers is None:
            logger.warning("Could not determine number of layers, falling back to standard loading")
            return load_model_standard(model_path, device, precision)

        block_size = min(block_size, num_layers)
        logger.info(f"Model has {num_layers} layers, loading in blocks of {block_size}")

        # Prepare loading parameters based on precision
        load_params = {
            "trust_remote_code": True,
            "config": config,
        }

        # Set precision
        if precision == "float16":
            load_params["torch_dtype"] = torch.float16
        elif precision == "bfloat16":
            load_params["torch_dtype"] = torch.bfloat16
        elif precision == "int8":
            load_params["load_in_8bit"] = True
        elif precision == "int4":
            load_params["load_in_4bit"] = True
            load_params["bnb_4bit_compute_dtype"] = torch.float16
            load_params["bnb_4bit_quant_type"] = "nf4"
            load_params["bnb_4bit_use_double_quant"] = True

        # Add device map
        if device == "cuda" and torch.cuda.is_available():
            # For quantized models, let the library handle device placement
            if precision in ["int8", "int4"]:
                load_params["device_map"] = "auto"
            else:
                # For progressive loading, we'll manually handle device placement
                load_params["device_map"] = {"": "cpu"}  # Start on CPU and move as needed
        else:
            # For CPU, just use CPU
            load_params["device_map"] = {"": "cpu"}

        # Initialize model
        model = AutoModel.from_pretrained(model_path, **load_params)

        # If using quantization, the model is already loaded completely
        if precision in ["int8", "int4"]:
            logger.info(f"Model loaded with {precision} quantization")
            return model, tokenizer

        # For non-quantized models, we need to move layers to the target device progressively
        if device == "cuda" and torch.cuda.is_available():
            logger.info("Moving model layers to GPU progressively")

            # Find model layers
            layers = None
            for attribute in ["layers", "encoder.layer", "transformer.h", "model.layers"]:
                try:
                    current = model
                    for part in attribute.split('.'):
                        current = getattr(current, part)
                    if isinstance(current, (list, torch.nn.ModuleList)) and len(current) == num_layers:
                        layers = current
                        break
                except (AttributeError, TypeError):
                    continue

            if layers is None:
                logger.warning("Could not find model layers, moving whole model to GPU")
                model = model.to(device)
            else:
                # Move embeddings and other parts to GPU first
                for name, module in model.named_children():
                    if name not in ["layers", "encoder.layer", "transformer.h", "model.layers"]:
                        try:
                            module.to(device)
                            logger.info(f"Moved {name} to GPU")
                        except Exception as e:
                            logger.warning(f"Failed to move {name} to GPU: {e}")

                # Move layers to GPU in blocks
                for i in range(0, num_layers, block_size):
                    end = min(i + block_size, num_layers)
                    logger.info(f"Moving layers {i}-{end - 1} to GPU")

                    # Move current block to GPU
                    for j in range(i, end):
                        try:
                            layers[j] = layers[j].to(device)

                            # Clear cache after each layer
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.warning(f"Failed to move layer {j} to GPU: {e}")

                    # Run garbage collection
                    gc.collect()

        logger.info("Model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error during progressive loading: {e}")
        logger.info("Falling back to standard loading")
        return load_model_standard(model_path, device, precision)


def load_model_standard(model_path, device="cuda", precision="int8"):
    """
    Load model using standard method.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
        precision: Precision to use
        
    Returns:
        Loaded model and tokenizer
    """
    logger.info(f"Loading model using standard method with {precision} precision")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare loading parameters
    load_params = {
        "trust_remote_code": True,
    }

    # Set precision
    if precision == "float16":
        load_params["torch_dtype"] = torch.float16
    elif precision == "bfloat16":
        load_params["torch_dtype"] = torch.bfloat16
    elif precision == "int8":
        load_params["load_in_8bit"] = True
    elif precision == "int4":
        load_params["load_in_4bit"] = True
        load_params["bnb_4bit_compute_dtype"] = torch.float16
        load_params["bnb_4bit_quant_type"] = "nf4"
        load_params["bnb_4bit_use_double_quant"] = True

    # Add device map
    if device == "cuda" and torch.cuda.is_available():
        if precision in ["int8", "int4"]:
            load_params["device_map"] = "auto"
        else:
            load_params["device_map"] = {"": device}
    else:
        load_params["device_map"] = {"": "cpu"}

    # Load model
    model = AutoModel.from_pretrained(model_path, **load_params)

    logger.info("Model loaded successfully")
    return model, tokenizer
