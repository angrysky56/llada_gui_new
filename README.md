# LLaDA GUI

A graphical user interface for interacting with the LLaDA (Large Language Diffusion with mAsking) model.

![image](https://github.com/user-attachments/assets/ace0cbfe-d5c4-4a37-bd49-e615fc75e791)


## Overview

This is a GUI wrapper for the [LLaDA model](https://github.com/ML-GSAI/LLaDA), an 8B scale diffusion model trained entirely from scratch that rivals LLaMA3 8B in performance. Unlike conventional autoregressive language models, LLaDA uses a diffusion approach with masking to generate text.

**Important:** This GUI is a third-party tool and not officially affiliated with the original LLaDA project. All credit for the underlying LLaDA model goes to the original authors at the Gaoling School of Artificial Intelligence, Renmin University of China. Please visit their [official repository](https://github.com/ML-GSAI/LLaDA) for more information about the model.

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
