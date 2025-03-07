# LLaDA GUI

# New version available that is slightly more functional-

https://github.com/angrysky56/llada_gui_new/tree/main

A graphical user interface for interacting with the LLaDA (Large Language Diffusion with mAsking) model.

![image](https://github.com/user-attachments/assets/3c189491-0a68-4fbb-998d-69a4865a02d7)


Currently maxes out my 12gb VRAM using 4 bit (reads around 20gb) but the new optimizations are working well and much faster- maybe 10x.

![image](https://github.com/user-attachments/assets/062e15e0-98f9-4898-82d2-c297533817d1)

![image](https://github.com/user-attachments/assets/4051dee9-b6d3-4f90-9885-c461e22d5236)


![image](https://github.com/user-attachments/assets/c6f6e8c5-f163-4ab9-b929-6cdbba89c2fa)

```
./start_memory_optimized.sh
```

Prototype memory system now available, slower and VRAM intensive. 
Derived from:

https://github.com/synthience/mcp-titan-cognitive-memory

Generally uses around 40gb RAM in CPU mode.


## Overview

This is a GUI wrapper for the [LLaDA model](https://github.com/ML-GSAI/LLaDA), an 8B scale diffusion model trained entirely from scratch that rivals LLaMA3 8B in performance. Unlike conventional autoregressive language models, LLaDA uses a diffusion approach with masking to generate text.

**Important:** This GUI is a third-party tool and not officially affiliated with the original LLaDA project. All credit for the underlying LLaDA model goes to the original authors at the Gaoling School of Artificial Intelligence, Renmin University of China. Please visit their [official repository](https://github.com/ML-GSAI/LLaDA) for more information about the model.

## 🚀 Performance Optimizations

This GUI includes several optimizations to make the model run efficiently on consumer hardware:

### Memory Efficiency
- **Smart CPU-GPU Offloading**: Intelligently moves tensors between CPU and GPU to minimize memory usage
- **Token Buffer Management**: Manages token data efficiently to reduce peak memory requirements
- **Adaptive Step Scheduling**: Uses fewer steps for easier tokens, more for difficult ones

### Generation Speed
- **Block-Level Processing**: Processes tokens in blocks for better GPU utilization
- **Progressive Generation**: High-confidence tokens are revealed early in the process
- **Chunked Operations**: Large operations are broken into manageable chunks

These optimizations allow the model to run on GPUs with 8-12GB VRAM while providing faster generation than the original implementation.

## Features

- **Text Generation**: Generate text responses to your prompts
- **Intuitive Interface**: Easy-to-use controls for interacting with the model
- **Configurable Parameters**: Adjust generation length, sampling steps, and more
- **Diffusion Visualization**: Watch the diffusion process unfold in real-time
- **Token Evolution**: See how masked tokens evolve into predicted text
- **Memory Management**: Options to optimize memory usage, including:
  - Real-time memory monitoring
  - 4-bit and 8-bit quantization options
  - CPU fallback for low-memory situations
  - Automatic parameter adjustment based on available memory
- **Performance Optimizations**: Built-in tools to improve performance:
  - Memory-efficient settings for lower GPU usage
  - Attention slicing for handling larger prompts
  - Precision control for speed/memory tradeoffs

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
   cd llada-gui
   ```

2. Use the provided installation script:
   ```
   chmod +x install.sh
   ./install.sh
   ```
   
   The script will:
   - Create a virtual environment
   - Install all required packages
   - Set up desktop integration if applicable

3. Alternatively, you can manually set up the environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

There are several ways to start the application:

1. **Using the start script**:
   ```
   ./start_gui.sh
   ```

2. **Direct Python execution**:
   ```
   ./venv/bin/python run.py
   ```

3. **Using the desktop file** (if installed):
   Double-click the `LLaDA_GUI.desktop` file in your applications menu or desktop.

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

4. **Click "Generate" to start the process**
5. **Watch the diffusion process** in the visualization tab
6. **View the final output** in the text output tab

### Memory Optimization

If you encounter out-of-memory errors:

1. Reduce Generation Length and Sampling Steps
2. Try 8-bit or 4-bit quantization options
3. Switch to CPU mode if necessary (will be slower but more reliable)
4. Use the built-in performance optimizer (described below)

## Performance Optimization

This application includes built-in performance optimization tools that can significantly reduce memory usage and improve generation speed.

### Using the Optimizer

1. **Launch the optimizer**:
   ```
   python optimize_launcher.py
   ```
   
   Or use the desktop shortcut:
   Double-click the `LLaDA_Optimizer.desktop` file.

2. **Select optimizations** in the GUI:
   - GPU Memory Optimizations
   - Config File Patches
   - Worker Code Optimizations

3. **Apply optimizations** by clicking "Apply Optimizations"

4. **Restart the application** to use the optimized version

## Understanding Diffusion in LLaDA

Unlike autoregressive models that generate one token at a time, LLaDA works by:

1. Starting with a completely masked sequence of the desired length
2. At each step, predicting values for all masked tokens simultaneously
3. Based on prediction confidence, keeping some tokens and remasking others
4. Repeating until all tokens are predicted

The visualization tab shows this process in action, with:
- Gray boxes for masked tokens
- Colored boxes for predicted tokens (color intensity indicates confidence)

## Project Structure

The application is organized into the following components:

- `llada_gui.py`: Main GUI application code
- `llada_worker.py`: Worker thread for asynchronous model execution
- `diffusion_visualization.py`: Visualization of the diffusion process
- `memory_monitor.py`: Real-time memory usage tracking
- `config.py`: Application configuration and constants
- `utils.py`: Utility functions
- `run.py`: Entry point script
- `optimizations/`: Performance optimization tools
- `onnx/`: Experimental ONNX conversion utilities

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

## License

This application is provided as-is under the MIT License. See the LICENSE file for details.

The LLaDA model has its own license from the original developers. Please refer to the [original repository](https://github.com/ML-GSAI/LLaDA) for more information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
