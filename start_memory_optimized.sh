#!/bin/bash

# Start LLaDA with memory integration and GPU optimization fixes
# This script applies GPU memory optimization patches and launches the memory-enhanced GUI

echo "Starting LLaDA with memory integration and GPU optimization..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Apply memory fixes and launch the application
python3 "$SCRIPT_DIR/fix_gpu_memory.py" --launch
