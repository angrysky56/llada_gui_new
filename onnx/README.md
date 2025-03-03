# LLaDA ONNX Conversion Tools

This directory contains tools for converting LLaDA models to ONNX format for faster inference.

## Overview

ONNX (Open Neural Network Exchange) is an open standard for machine learning models that allows models to be transferred between different frameworks and optimized for inference. Converting the LLaDA model to ONNX format can provide significant performance improvements:

- **Faster inference**: Up to 2-4x speedup for text generation
- **Reduced memory usage**: Especially with quantized models
- **Better hardware compatibility**: Works with optimized inference engines

## Requirements

Before using these tools, install the required packages:

```bash
pip install onnx onnxruntime-gpu onnxruntime
```

Or update all requirements:

```bash
pip install -r requirements.txt
```

## How to Use

### Simple Conversion

To convert the LLaDA model to ONNX format with default settings:

```bash
python convert_to_onnx.py
```

This will:
1. Load the default LLaDA model
2. Convert it to ONNX format
3. Apply optimizations
4. Save the model to `onnx_models/` directory

### Advanced Options

The converter supports several options:

```bash
python convert_to_onnx.py --model_path <path_to_model> --output_dir <output_directory> --quantize --test
```

Options:
- `--model_path`: Path to the original model (default: GSAI-ML/LLaDA-8B-Instruct)
- `--output_dir`: Directory to save the ONNX model (default: onnx_models)
- `--quantize`: Quantize the model to INT8 to reduce size and memory usage
- `--cpu`: Use CPU instead of GPU for conversion (slower but works on machines without GPU)
- `--test`: Test the model after conversion with a simple prompt
- `--prompt`: Specify a custom prompt for testing (default: "Hello, how are you?")
- `--opset`: ONNX opset version to use (default: 17)

### Testing Performance

To test and compare the performance of PyTorch vs. ONNX models:

```bash
python test_onnx.py
```

This script will:
1. Run inference on the original PyTorch model
2. Run inference on all available ONNX models (base, optimized, quantized)
3. Compare inference times and report speedup

Options:
- `--model_path`: Path to the original model
- `--onnx_dir`: Directory containing ONNX models
- `--prompt`: Custom prompt for testing
- `--cpu`: Use CPU instead of GPU for testing

## Model Variants

The conversion process creates up to three ONNX model variants:

1. **Base ONNX** (`llada_model.onnx`): Direct conversion of the PyTorch model
2. **Optimized** (`llada_model_optimized.onnx`): Optimized version with graph transformations for better performance
3. **Quantized** (`llada_model_quantized.onnx`): INT8 quantized version for reduced memory usage (only if --quantize is specified)

**Note**: Due to the large size of the LLaDA model, the ONNX files use external data format. This means each `.onnx` file will have an accompanying `.onnx.data` file that contains the model weights. Both files must be kept together.

## Performance Expectations

Performance improvement depends on hardware and model size:

- **NVIDIA GPUs**: 1.5x to 3x speedup with optimized models
- **CPU**: 1.2x to 2x speedup with optimized models
- **Quantized models**: Up to 4x memory reduction with modest performance impact

## Troubleshooting

### Common Issues

1. **Out of memory during conversion**:
   - Try using CPU mode: `python convert_to_onnx.py --cpu`
   - Convert on a machine with more memory

2. **Slow conversion**:
   - Conversion is a one-time process that can be slow, especially on CPU
   - The resulting ONNX model will be faster for inference

3. **Missing optimizations**:
   - Make sure you have the latest onnxruntime-gpu installed
   - Some optimizations are only available for specific hardware

4. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

## Future Work

Potential improvements to the ONNX conversion tools:

- Integration with the LLaDA GUI for one-click conversion
- Support for more quantization options (e.g., FP16, INT4)
- Kernel fusion optimizations for specific hardware
- Streaming generation support
- Support for more diffusion-specific optimizations

## References

- [ONNX Format](https://github.com/onnx/onnx)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [LLaDA Paper](https://arxiv.org/abs/2502.09992)
