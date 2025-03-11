#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaDA ONNX Conversion Launcher

This script provides a simple command-line interface to convert
the LLaDA model to ONNX format for faster inference.

Examples:
    # Convert with default settings:
    python convert_to_onnx.py
    
    # Convert with quantization:
    python convert_to_onnx.py --quantize
    
    # Convert using CPU:
    python convert_to_onnx.py --cpu
    
    # Convert and test:
    python convert_to_onnx.py --test
"""

import sys
from pathlib import Path

# Add the current directory to the path to find the onnx module
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import the converter
try:
    from onnx.converter import main as converter_main
except ImportError:
    print("Error: Could not import ONNX converter module")
    print("Make sure the onnx directory exists and contains converter.py")
    sys.exit(1)

if __name__ == "__main__":
    # Pass all arguments to the converter's main function
    sys.exit(converter_main())
