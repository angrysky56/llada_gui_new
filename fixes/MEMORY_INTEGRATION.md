# Memory Integration for LLaDA GUI

This document provides information about the memory integration feature in LLaDA GUI, which allows the model to maintain context across generations for more coherent outputs.

## How Memory Integration Works

The memory integration feature works by:

1. Running a memory server that maintains a neural memory model
2. Integrating the memory system with the diffusion process during text generation
3. Providing a visualization of the memory state and controls for memory influence

## Getting Started

### Running with Memory Integration

You can run LLaDA GUI with memory integration in several ways:

1. **Using the improved memory script (recommended)**: Run `./start_llada_with_memory.sh` or `python run_with_memory_improved.py`
2. **Using the desktop launcher**: Click on "LLaDA GUI with Memory" in your applications menu
3. **From the command line**: Run `python run_with_memory.py`
4. **From standard GUI**: Run `python run.py --memory` or check the "Enable Memory Integration" checkbox in the GUI

We recommend using the improved memory script as it provides better server management with automatic startup and cleanup.

### Memory Tab

When memory integration is enabled, a "Memory" tab will appear in the GUI interface. This tab provides:

- Visualization of the current memory state
- Controls for memory influence
- Training options for the memory model
- Buttons to save/load memory models

## Troubleshooting

If you encounter issues with memory integration:

1. **Memory tab is missing**: Make sure memory integration is enabled in the GUI
2. **Server connection errors**: Run the `fix_memory_server.py` script to repair the memory server installation
3. **Generation doesn't use memory**: Check that "Use Memory Guidance" is selected in the Memory tab

If problems persist, try these steps:

1. Stop any running memory servers with `python stop_memory_server.py`
2. Run `python fix_memory_server.py` to repair the memory server installation
3. Run `python fix_memory_dependencies.py` to install required dependencies
4. Start the application using our improved script: `./start_llada_with_memory.sh`
5. If the issue persists, check the logs in `llada_memory.log` for more details

### Port Already in Use Errors

If you see "Port 3000 is already in use" errors:

1. First try our improved stop script: `python stop_memory_server.py`
2. If that doesn't work, find the process using the port: `lsof -i :3000`
3. Stop the process: `kill -9 [PID]` (replace [PID] with the process ID)
4. Then start the application again

## Using the Memory System

### Memory Influence

You can adjust the memory influence using the slider in the Memory tab. Higher values give more weight to the memory system's predictions during generation.

### Training the Memory System

After generating text, you can train the memory system on your generations:

1. Generate text with the model
2. Go to the Memory tab
3. Click "Train on Last Generation"

This improves memory performance over time as it learns from your usage patterns.

### Saving and Loading Memory Models

You can save and load memory models to preserve your training:

1. Click "Save Memory Model" to save the current memory state
2. Click "Load Memory Model" to load a previously saved memory state

## Technical Details

The memory system consists of:

- A neural memory model implemented in either JavaScript (Node.js) or Python
- A server that provides memory services via HTTP API
- Integration with the LLaDA diffusion process for guided generation

The memory model maintains a state vector that evolves during generation, helping to guide the model toward more coherent and contextually appropriate text.
