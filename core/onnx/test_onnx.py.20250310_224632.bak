#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for LLaDA ONNX conversion and inference.

This script tests the ONNX conversion and inference process on a small test case.
It measures performance compared to the original PyTorch model.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llada_onnx_test")


def test_pytorch_inference(model_path, prompt, device="cuda"):
    """
    Test PyTorch model inference and measure performance.
    
    Args:
        model_path: Path to the model
        prompt: Test prompt
        device: Device to run inference on
    
    Returns:
        Inference time in seconds
    """
    try:
        from transformers import AutoTokenizer, AutoModel

        # Load tokenizer and model
        logger.info(f"Loading PyTorch model from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set up loading parameters
        load_params = {
            "trust_remote_code": True,
        }

        # Use lower precision for GPU
        if device == "cuda":
            load_params["torch_dtype"] = torch.bfloat16
        else:
            load_params["torch_dtype"] = torch.float32

        # Load and prepare model
        model = AutoModel.from_pretrained(model_path, **load_params)
        model = model.to(device).eval()

        # Prepare input
        m = [{"role": "user", "content": prompt}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']

        # Convert to tensor and put on device
        input_tensor = torch.tensor([input_ids]).to(device)

        # Warmup
        with torch.no_grad():
            _ = model(input_tensor)

        # Measure performance
        logger.info("Running PyTorch inference")
        start_time = time.time()

        with torch.no_grad():
            output = model(input_tensor)

        end_time = time.time()
        inference_time = end_time - start_time

        logger.info(f"PyTorch inference completed in {inference_time:.3f} seconds")
        logger.info(f"Output shape: {output.logits.shape}")

        # Clean up
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

        return inference_time

    except Exception as e:
        logger.error(f"Error during PyTorch inference: {e}")
        return None


def test_onnx_inference(onnx_path, model_path, prompt, device="cuda"):
    """
    Test ONNX model inference and measure performance.
    
    Args:
        onnx_path: Path to the ONNX model
        model_path: Path to the original model (for tokenizer)
        prompt: Test prompt
        device: Device to run inference on
    
    Returns:
        Inference time in seconds
    """
    try:
        import onnxruntime as ort
        import numpy as np
        from transformers import AutoTokenizer

        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Prepare input
        m = [{"role": "user", "content": prompt}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_tensor = np.array([input_ids], dtype=np.int64)

        # Set up ONNX Runtime session
        providers = []
        if device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        logger.info(f"Creating ONNX Runtime session with providers: {providers}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers
        )

        # Warmup
        _ = session.run(None, {"input_ids": input_tensor})

        # Measure performance
        logger.info("Running ONNX inference")
        start_time = time.time()

        ort_outputs = session.run(None, {"input_ids": input_tensor})

        end_time = time.time()
        inference_time = end_time - start_time

        logger.info(f"ONNX inference completed in {inference_time:.3f} seconds")
        logger.info(f"Output shape: {ort_outputs[0].shape}")

        return inference_time

    except Exception as e:
        logger.error(f"Error during ONNX inference: {e}")
        return None


def main():
    """Run the tests and compare performance."""
    parser = argparse.ArgumentParser(description="Test LLaDA ONNX conversion and inference")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Path to the original model")
    parser.add_argument("--onnx_dir", type=str, default="onnx_models",
                        help="Directory containing ONNX models")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?",
                        help="Test prompt")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Set device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    logger.info(f"Using device: {device}")

    # Check if ONNX model exists
    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        logger.error(f"ONNX directory {onnx_dir} not found")
        logger.info("Please run the conversion script first:")
        logger.info("python convert_to_onnx.py")
        return 1

    # Find available ONNX models
    onnx_models = {}

    if (onnx_dir / "llada_model_quantized.onnx").exists():
        onnx_models["quantized"] = str(onnx_dir / "llada_model_quantized.onnx")

    if (onnx_dir / "llada_model_optimized.onnx").exists():
        onnx_models["optimized"] = str(onnx_dir / "llada_model_optimized.onnx")

    if (onnx_dir / "llada_model.onnx").exists():
        onnx_models["base"] = str(onnx_dir / "llada_model.onnx")

    if not onnx_models:
        logger.error(f"No ONNX models found in {onnx_dir}")
        logger.info("Please run the conversion script first:")
        logger.info("python convert_to_onnx.py")
        return 1

    # Test PyTorch model
    pt_time = test_pytorch_inference(args.model_path, args.prompt, device)

    # Test ONNX models
    results = {}

    for model_type, onnx_path in onnx_models.items():
        logger.info(f"Testing {model_type} ONNX model")
        onnx_time = test_onnx_inference(onnx_path, args.model_path, args.prompt, device)

        if onnx_time and pt_time:
            speedup = pt_time / onnx_time
            results[model_type] = (onnx_time, speedup)

    # Print results
    logger.info("\n===== Performance Comparison =====")
    logger.info(f"PyTorch inference time: {pt_time:.3f} seconds")

    for model_type, (time, speedup) in results.items():
        logger.info(f"{model_type.capitalize()} ONNX inference time: {time:.3f} seconds (speedup: {speedup:.2f}x)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
