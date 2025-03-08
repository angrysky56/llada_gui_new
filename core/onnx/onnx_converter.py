#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX Converter for LLaDA model.

This module provides functionality to convert the LLaDA model to ONNX format,
and optimize it for inference with ONNX Runtime.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llada_onnx")

# Configuration constants
DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_OUTPUT_DIR = "onnx_model"
DEFAULT_OPSET = 17  # Latest stable opset version as of 2025


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
            model_path (str): Path to the original model
            output_dir (str): Directory to save the ONNX model
            opset (int): ONNX opset version
            quantize (bool): Whether to quantize the model to int8
            use_gpu (bool): Whether to use GPU for conversion and inference
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.opset = opset
        self.quantize = quantize
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set ONNX Runtime providers based on device
        self.providers = []
        if self.device == "cuda":
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

        # Original PyTorch model and tokenizer
        self.tokenizer = None
        self.pt_model = None

        # ONNX model
        self.onnx_model_path = self.output_dir / "llada_model.onnx"
        self.optimized_model_path = self.output_dir / "llada_model_optimized.onnx"
        self.quantized_model_path = self.output_dir / "llada_model_quantized.onnx"

        # Config file path
        self.config_path = self.output_dir / "onnx_config.json"

    def load_model(self) -> None:
        """Load the original PyTorch model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

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

        # Move model to device
        self.pt_model = self.pt_model.to(self.device)

        # Set model to evaluation mode
        self.pt_model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def convert_to_onnx(self) -> None:
        """Convert the model to ONNX format."""
        if self.pt_model is None:
            self.load_model()

        logger.info(f"Converting model to ONNX (opset {self.opset})")

        # Create dummy input
        # For LLaDA, the input is just token IDs
        batch_size = 1
        seq_length = 32  # Small sequence length for conversion

        # Create dummy input tensor on the correct device
        dummy_input = torch.ones(batch_size, seq_length, dtype=torch.long).to(self.device)

        # Get the names of inputs and outputs
        input_names = ["input_ids"]
        output_names = ["logits"]

        # Dynamic axes definition for variable sequence length
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }

        # Export the model to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.pt_model,  # model being run
                dummy_input,  # model input
                self.onnx_model_path,  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=self.opset,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=input_names,  # the model's input names
                output_names=output_names,  # the model's output names
                dynamic_axes=dynamic_axes,  # variable length axes
                verbose=False
            )

        logger.info(f"ONNX model saved to {self.onnx_model_path}")

        # Save configuration
        self._save_config()

    def optimize_onnx_model(self) -> None:
        """Optimize the ONNX model for inference."""
        if not self.onnx_model_path.exists():
            logger.error("ONNX model not found. Run convert_to_onnx() first.")
            return

        logger.info("Optimizing ONNX model")

        # Load the model
        onnx_model = onnx.load(str(self.onnx_model_path))

        # Basic model optimization with ONNX Runtime
        # This includes constant folding, redundant node elimination, etc.
        from onnxruntime.transformers import optimizer
        opt_options = optimizer.OptimizationOptions()

        # Configure the optimizer based on the device
        if self.device == "cuda":
            opt_options.enable_gelu_approximation = True
            opt_options.enable_layer_norm = True
            opt_options.enable_attention = True

        # Use transformer model optimizer
        model_optimizer = optimizer.optimize_model(
            str(self.onnx_model_path),
            'llada',
            num_heads=32,  # This should match the model's configuration
            hidden_size=4096,  # This should match the model's configuration
            optimization_options=opt_options
        )

        # Save the optimized model
        model_optimizer.save_model_to_file(str(self.optimized_model_path))

        logger.info(f"Optimized ONNX model saved to {self.optimized_model_path}")

        # Update configuration
        self._save_config()

    def quantize_onnx_model(self) -> None:
        """Quantize the ONNX model to INT8 precision."""
        if not self.optimized_model_path.exists():
            logger.error("Optimized ONNX model not found. Run optimize_onnx_model() first.")
            return

        if not self.quantize:
            logger.info("Quantization not requested. Skipping.")
            return

        logger.info("Quantizing ONNX model to INT8")

        from onnxruntime.quantization import quantize_dynamic, QuantType

        # Quantize the model to INT8
        quantize_dynamic(
            model_input=str(self.optimized_model_path),
            model_output=str(self.quantized_model_path),
            per_channel=False,
            weight_type=QuantType.QInt8
        )

        logger.info(f"Quantized ONNX model saved to {self.quantized_model_path}")

        # Update configuration
        self._save_config()

    def test_onnx_model(self, prompt: str, steps: int = 64, gen_length: int = 64) -> str:
        """
        Test the ONNX model with a simple prompt.
        
        Args:
            prompt (str): The prompt text
            steps (int): Number of diffusion steps
            gen_length (int): Length of text to generate
            
        Returns:
            str: The generated text
        """
        if not self.onnx_model_path.exists() and not self.optimized_model_path.exists():
            logger.error("ONNX model not found. Run convert_to_onnx() first.")
            return ""

        # Determine which model to use
        if self.quantized_model_path.exists() and self.quantize:
            model_path = str(self.quantized_model_path)
            logger.info("Using quantized model for inference")
        elif self.optimized_model_path.exists():
            model_path = str(self.optimized_model_path)
            logger.info("Using optimized model for inference")
        else:
            model_path = str(self.onnx_model_path)
            logger.info("Using base ONNX model for inference")

        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        # Prepare input
        m = [{"role": "user", "content": prompt}]
        user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(user_input)['input_ids']
        input_tensor = np.array([input_ids], dtype=np.int64)

        # Create ONNX Runtime session
        logger.info(f"Creating ONNX Runtime session with providers: {self.providers}")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=self.providers
        )

        # At this point, we would normally run the generate function
        # However, since LLaDA uses a custom generation process that's not
        # part of the model's forward pass, we would need to reimplement it using ONNX Runtime
        # For this example, we'll just run a single forward pass to confirm the model works

        logger.info("Running inference (single forward pass)")
        ort_inputs = {
            "input_ids": input_tensor
        }

        # Run inference
        ort_outputs = session.run(None, ort_inputs)

        # Log results
        logger.info(f"ONNX Runtime inference completed. Output shape: {ort_outputs[0].shape}")

        # For complete generation, we would need to reimplement the generate function
        # using the ONNX Runtime session's run method instead of the PyTorch model's forward pass
        # This would be a more complex effort beyond this example

        return "ONNX model inference successful (single forward pass only)."

    def _save_config(self) -> None:
        """Save the model configuration."""
        config = {
            "model_path": str(self.model_path),
            "onnx_model_path": str(self.onnx_model_path),
            "optimized_model_path": str(self.optimized_model_path),
            "quantized_model_path": str(self.quantized_model_path) if self.quantize else None,
            "opset": self.opset,
            "quantize": self.quantize,
            "device": self.device,
            "providers": self.providers,
            "conversion_timestamp": str(datetime.datetime.now())
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {self.config_path}")


def main():
    """Run the ONNX conversion process."""
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
        converter.load_model()
        converter.convert_to_onnx()
        converter.optimize_onnx_model()

        if args.quantize:
            converter.quantize_onnx_model()

        if args.test:
            result = converter.test_onnx_model(args.prompt)
            print(f"Test result: {result}")

        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    import datetime

    sys.exit(main())
