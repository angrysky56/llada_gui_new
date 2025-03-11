
# LLaDA GUI new version

# 3/8/25 Much thanks to automenta for the advanced work.
[GUI Remix, Refactoring, Optimization](https://github.com/angrysky56/llada_gui_new/pull/2)
![alt text](image.png)
# Currently you can run inference but the memory system is only partly functional ie it will make some vectors, but the rest of the memory model system is not properly integrated and only uses dummy output. Sorry not a coder.

A graphical user interface for interacting with the LLaDA (Large Language Diffusion with mAsking) model with integrated cognitive memory.

![LLaDA GUI Screenshot](https://github.com/user-attachments/assets/3c189491-0a68-4fbb-998d-69a4865a02d7)

This enhanced version includes a cognitive memory system that allows the model to maintain context across generations.

## Overview

This is a GUI wrapper for the [LLaDA model](https://github.com/ML-GSAI/LLaDA), an 8B scale diffusion model trained entirely from scratch that rivals LLaMA3 8B in performance. Unlike conventional autoregressive language models, LLaDA uses a diffusion approach with masking to generate text.

**Important:** This GUI is a third-party tool and not officially affiliated with the original LLaDA project. All credit for the underlying LLaDA model goes to the original authors at the Gaoling School of Artificial Intelligence, Renmin University of China. Please visit their [official repository](https://github.com/ML-GSAI/LLaDA) for more information about the model.

## 🚀 Features

### Unified Launcher
- Single launcher with multiple modes:
  - **Standard Mode**: Basic LLaDA generation
  - **Memory-Enhanced Mode**: LLaDA with cognitive memory integration
  - **Optimized Mode**: Memory-efficient operation
  - **Extreme Optimization Mode**: For GPUs with limited VRAM (8-12GB)

### Memory Efficiency
- **Smart CPU-GPU Offloading**: Intelligently moves tensors between CPU and GPU
- **Token Buffer Management**: Efficiently manages token data 
- **Adaptive Step Scheduling**: Uses fewer steps for easier tokens, more for difficult ones
- **4-bit and 8-bit Quantization**: Reduced precision options for lower memory usage

### Memory Integration
- **Persistent Memory**: Remembers context across sessions using a vector database
- **Memory Guidance**: Leverages past interactions to guide new generations
- **Trainable Memory**: Learns from your generations to improve over time
- **Memory Visualization**: See the memory state and influence generation

### User Interface
- **Intuitive Controls**: Easy-to-use interface for all features
- **Diffusion Visualization**: Watch the diffusion process unfold in real-time
- **Token Evolution**: See how masked tokens evolve into predicted text
- **Real-time Memory Monitoring**: Track system and GPU memory usage

## Requirements

- Python 3.10 or later
- PyQt6
- PyTorch 2.0 or later
- Transformers 4.38.2
- CUDA-capable GPU with at least 10GB memory (for optimal performance)
- CPU-only mode is also supported (slower but works on any machine)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/angrysky56/llada-gui.git
   cd llada_gui_new
   ```

2. Use the provided installation script:
   ```
   # Make executable
   chmod +x install.sh
   
   # Install everything
   ./install.sh
   ```
   
   The script will:
   - Create a virtual environment
   - Install all required packages
   - Apply memory integration fixes
   - Create a desktop icon for easy access
   - Check for the LLaDA model on first generate (downloading to /data if needed)
   - Create a desktop icon

## Usage

### Starting the Application

You can start the application in several ways:

1. **Using the desktop icon (recommended)**:
   Simply double-click the LLaDA GUI icon on your desktop.
   This launches the application with memory integration enabled.

2. **Using the memory integration script directly**:
   ```
   ./run_memory.sh
   ```
   This script provides enhanced memory server management with automatic startup and cleanup.

   If you have issues run:
    ```
    ./fix_all.py
    ```

3. **Command-line options** (because Claude kept messing up and insisted they weren't mistakes "For advanced users" lol):
   ```
   # Standard mode
   ./run.py --standard
   
   # Memory-enhanced mode (basic)
   ./run.py --memory
   
   # Memory-enhanced mode (improved)
   python run_with_memory_improved.py
   
   # Optimized mode
   ./run.py --optimize
   
   # Extreme optimization mode (for 8-12GB GPUs)
   ./run.py --extreme
   ```

### Using the Interface

1. **Enter your prompt in the text input area**
2. **Adjust generation parameters as needed**:
   - Generation Length: Number of tokens to generate
   - Sampling Steps: Number of diffusion steps (higher = better quality but slower)
   - Block Length: Size of blocks for semi-autoregressive generation
   - Temperature: Controls randomness (0 = deterministic, higher = more random)
   - CFG Scale: Classifier-free guidance strength
   - Remasking Strategy: Method to select which tokens remain masked

3. **Select hardware options**:
   - Choose between CPU or GPU
   - Select memory optimization (normal precision, 8-bit, or 4-bit quantization)
   - Enable Memory Integration for context-aware generation

4. **Click "Generate" to start the process**
5. **Watch the generation process** in the visualization tab
6. **View the final output** in the text output tab

### Memory Integration

This version includes robust memory integration with several important fixes:

- **Fixed Memory Server**: Reliable memory server startup and communication
- **Tensor Handling**: Improved tensor detachment and handling
- **Error Recovery**: Graceful recovery from common error conditions
- **Simplified Workflow**: Memory integration is enabled by default

When using the application:

1. **Memory is enabled by default** - you'll see the Memory tab in the interface
2. **Adjust Memory Influence** using the slider in the Memory tab (30% is default)
3. **Train the Memory** on your generations to improve over time
4. **Save and Load Memory Models** to preserve your training

## Memory Integration Fixes

This version includes comprehensive fixes for the memory integration system. For detailed information about these fixes, please see the [MEMORY_FIXES.md](MEMORY_FIXES.md) file.

## Understanding Diffusion in LLaDA

Unlike autoregressive models that generate one token at a time, LLaDA works by:

1. Starting with a completely masked sequence of the desired length
2. At each step, predicting values for all masked tokens simultaneously
3. Based on prediction confidence, keeping some tokens and remasking others
4. Repeating until all tokens are predicted

The visualization tab shows this process in action, with:
- Gray boxes for masked tokens
- Colored boxes for predicted tokens (color intensity indicates confidence)

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: 
   - Try running in extreme optimization mode with 4-bit quantization
   - Reduce generation length and sampling steps
   - Switch to CPU mode if necessary

2. **Memory Integration Problems**:
   - The memory server should start automatically
   - If the status in the Memory tab shows "Not Connected", click the "Connect" button
   - If the memory server seems stuck, you can try:
     1. Restart the application: `./run_memory.sh`
     2. Run `./stop_memory_server.py` to force stop any existing servers
     3. Delete the memory model to start fresh: `rm -f core/memory/memory_data/titan_memory_model.json`

3. **Port Already in Use Errors**:
   - If you see "Port 3000 is already in use" errors:
     1. Run `python stop_memory_server.py` to stop any running servers
     2. If that doesn't work, find the process using the port: `lsof -i :3000`
     3. Stop the process: `kill -9 [PID]` (replace [PID] with the process ID)
     4. Then start the application again

4. **Installation Issues**:
   - Make sure you have Python 3.10+ installed
   - Check that PyTorch is correctly installed with CUDA support if using GPU

## Acknowledgements

This GUI is built on top of the [LLaDA model](https://github.com/ML-GSAI/LLaDA) developed by researchers at the Gaoling School of Artificial Intelligence, Renmin University of China. Please cite their work when using this application:

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```
Cognitive memory developed from:
https://github.com/synthience/mcp-titan-cognitive-memory

## License

This application is provided as-is under the MIT License. See the LICENSE file for details.

The LLaDA model has its own license from the original developers. Please refer to the [original repository](https://github.com/ML-GSAI/LLaDA) for more information.
