#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX Converter for LLaDA model.

This module provides a command-line tool to convert the LLaDA model to ONNX format,
and optimize it for inference with ONNX Runtime.

Usage:
  python -m onnx.converter --model_path <path_to_model> --output_dir <output_directory> [--quantize] [--cpu]
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llada_onnx")

# Configuration constants
DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_OUTPUT_DIR = "onnx_models"
DEFAULT_OPSET = 17  # Latest stable opset as of early 2025


class LLaDAONNXConverter:
    """Handles conversion of LLaDA model to ONNX format."""

    def __init__(
            self,
            model_path: str = DEFAULT_MODEL_PATH,
            output_dir: str = DEFAULT_OUTPUT_DIR,
            opset: int = DEFAULT_OPSET,
            quantize: bool = False,
            use_gpu: bool = True,
    ):
        """
        Initialize the LLaDA ONNX converter.

        Args:
            model_path: Path to the original model
            output_dir: Directory to save the ONNX model
            opset: ONNX opset version
            quantize: Whether to quantize the model to INT8
            use_gpu: Whether to use GPU for conversion
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.opset = opset
        self.quantize = quantize
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Original PyTorch model and tokenizer
        self.tokenizer = None
        self.pt_model = None

        # ONNX model paths
        self.onnx_model_path = self.output_dir / "llada_model.onnx"
        self.optimized_model_path = self.output_dir / "llada_model_optimized.onnx"
        self.quantized_model_path = self.output_dir / "llada_model_quantized.onnx"

        # Config file path
        self.config_path = self.output_dir / "config.json"

    def load_model(self):
        """Load the original PyTorch model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        try:
            # Import here to avoid requiring these at module level
            from transformers import AutoTokenizer, AutoModel

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Prepare loading parameters based on device
            load_params = {
                "trust_remote_code": True,
            }

            # Use lower precision for GPU to save memory
            if self.device == "cuda":
                load_params["torch_dtype"] = torch.bfloat16
            else:
                load_params["torch_dtype"] = torch.float32

            # Load model
            self.pt_model = AutoModel.from_pretrained(
                self.model_path, **load_params
            )

            # Move model to device and set to eval mode
            self.pt_model = self.pt_model.to(self.device).eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def convert_to_onnx(self):
        """Convert the model to ONNX format."""
        if self.pt_model is None:
            self.load_model()

        logger.info(f"Converting model to ONNX (opset {self.opset})")

        # Create dummy input
        batch_size = 1
        seq_length = 32  # Small sequence length for conversion

        # Create dummy input tensor on the correct device
        dummy_input = torch.ones(batch_size, seq_length, dtype=torch.long).to(self.device)

        # Set up export parameters
        input_names = ["input_ids"]
        output_names = ["logits"]

        # Dynamic axes for variable sequence length
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }

        # Export the model to ONNX with external data format to handle models > 2GB
        with torch.no_grad():
            torch.onnx.export(
                self.pt_model,  # model being run
                dummy_input,  # model input
                str(self.onnx_model_path),  # where to save the model (must be string for external data)
                export_params=True,  # store the trained parameters
                opset_version=self.opset,  # the ONNX version to export to
                do_constant_folding=True,  # constant folding optimization
                input_names=input_names,  # model's input names
                output_names=output_names,  # model's output names
                dynamic_axes=dynamic_axes,  # variable length axes
                verbose=False,
                # Enable external data format for large models
                use_external_data_format=True  # Use external data format for models > 2GB
            )

        logger.info(f"ONNX model saved to {self.onnx_model_path}")

        # Save configuration
        self._save_config()

    def optimize_onnx_model(self):
        """Optimize the ONNX model for inference."""
        if not self.onnx_model_path.exists():
            logger.error("ONNX model not found. Run convert_to_onnx() first.")
            return

        logger.info("Optimizing ONNX model")

        try:
            # Import here to avoid requiring these at module level
            import onnx
            from onnxruntime.transformers import optimizer

            # Load the model
            onnx_model = onnx.load(str(self.onnx_model_path))

            # Configure optimization options
            opt_options = optimizer.OptimizationOptions()

            # Enhanced optimizations for GPU
            if self.device == "cuda":
                opt_options.enable_gelu_approximation = True
                opt_options.enable_layer_norm = True
                opt_options.enable_attention = True

            # Use transformer model optimizer
            # For large models, make sure to use external data format
            model_optimizer = optimizer.optimize_model(
                str(self.onnx_model_path),
                'llada',
                num_heads=32,  # This should match model's configuration
                hidden_size=4096,  # This should match model's configuration
                optimization_options=opt_options,
                use_external_data_format=True  # Use external data format for large models
            )

            # Save the optimized model with external data format
            model_optimizer.save_model_to_file(str(self.optimized_model_path), use_external_data_format=True)

            logger.info(f"Optimized ONNX model saved to {self.optimized_model_path}")

            # Update configuration
            self._save_config()

        except Exception as e:
            logger.error(f"Error optimizing ONNX model: {e}")
            logger.warning("Skipping optimization step")

    def quantize_onnx_model(self):
        """Quantize the ONNX model to INT8 precision."""
        if not self.optimized_model_path.exists():
            if not self.onnx_model_path.exists():
                logger.error("ONNX model not found. Run convert_to_onnx() first.")
                return
            logger.warning("Optimized ONNX model not found. Using base model for quantization.")
            source_model = self.onnx_model_path
        else:
            source_model = self.optimized_model_path

        if not self.quantize:
            logger.info("Quantization not requested. Skipping.")
            return

        logger.info("Quantizing ONNX model to INT8")

        try:
            # Import here to avoid requiring these at module level
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Quantize the model to INT8
            quantize_dynamic(
                model_input=str(source_model),
                model_output=str(self.quantized_model_path),
                per_channel=False,
                weight_type=QuantType.QInt8
            )

            logger.info(f"Quantized ONNX model saved to {self.quantized_model_path}")

            # Update configuration
            self._save_config()

        except Exception as e:
            logger.error(f"Error quantizing ONNX model: {e}")
            logger.warning("Skipping quantization step")

    def test_onnx_model(self, prompt: str, steps: int = 4, gen_length: int = 16):
        """
        Test the ONNX model with a simple prompt.
        
        Args:
            prompt: The prompt text
            steps: Number of diffusion steps (reduced for testing)
            gen_length: Length of text to generate (reduced for testing)
            
        Returns:
            Test result message
        """
        if not self.onnx_model_path.exists() and not self.optimized_model_path.exists():
            logger.error("ONNX model not found. Run convert_to_onnx() first.")
            return "No ONNX model found"

        # Determine which model to use
        if self.quantized_model_path.exists() and self.quantize:
            model_path = str(self.quantized_model_path)
            model_type = "quantized"
        elif self.optimized_model_path.exists():
            model_path = str(self.optimized_model_path)
            model_type = "optimized"
        else:
            model_path = str(self.onnx_model_path)
            model_type = "base"

        logger.info(f"Testing {model_type} ONNX model")

        try:
            # Import here to avoid requiring these at module level
            import onnxruntime as ort

            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )

            # Prepare input
            m = [{"role": "user", "content": prompt}]
            user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = self.tokenizer(user_input)['input_ids']
            input_tensor = np.array([input_ids], dtype=np.int64)

            # Set up ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else [
                'CPUExecutionProvider']

            logger.info(f"Creating ONNX Runtime session with providers: {providers}")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )

            # Run single forward pass for testing
            logger.info("Running inference (single forward pass)")
            ort_inputs = {
                "input_ids": input_tensor
            }

            # Run inference
            start_time = datetime.datetime.now()
            ort_outputs = session.run(None, ort_inputs)
            end_time = datetime.datetime.now()

            inference_time = (end_time - start_time).total_seconds()

            # Log results
            logger.info(f"ONNX Runtime inference completed in {inference_time:.3f} seconds")
            logger.info(f"Output shape: {ort_outputs[0].shape}")

            return f"ONNX model inference successful (single forward pass only). Time: {inference_time:.3f}s"

        except Exception as e:
            logger.error(f"Error testing ONNX model: {e}")
            return f"Error testing ONNX model: {str(e)}"

    def _save_config(self):
        """Save the model configuration."""
        config = {
            "model_path": str(self.model_path),
            "onnx_model_path": str(self.onnx_model_path),
            "optimized_model_path": str(self.optimized_model_path),
            "quantized_model_path": str(self.quantized_model_path) if self.quantize else None,
            "opset": self.opset,
            "quantize": self.quantize,
            "device": self.device,
            "conversion_timestamp": str(datetime.datetime.now())
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {self.config_path}")


def main():
    """Run the ONNX conversion process from command line."""
    parser = argparse.ArgumentParser(description="Convert LLaDA model to ONNX format")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the original model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the ONNX model (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                        help=f"ONNX opset version (default: {DEFAULT_OPSET})")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize the model to INT8 (default: False)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU (default: use GPU if available)")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after conversion (default: False)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Prompt to use for testing (default: 'Hello, how are you?')")

    args = parser.parse_args()

    # Create converter
    converter = LLaDAONNXConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        opset=args.opset,
        quantize=args.quantize,
        use_gpu=not args.cpu
    )

    # Convert model
    try:
        # Steps in the conversion process
        converter.load_model()
        converter.convert_to_onnx()
        converter.optimize_onnx_model()

        if args.quantize:
            converter.quantize_onnx_model()

        if args.test:
            result = converter.test_onnx_model(args.prompt)
            print(f"Test result: {result}")

        logger.info("Conversion completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
