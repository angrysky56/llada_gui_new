# LLaDA GUI with Cognitive Memory

This extension to the LLaDA GUI adds cognitive memory capabilities for enhanced text generation.

## Overview

The cognitive memory system enables the LLaDA diffusion model to maintain coherence and consistency over longer outputs by using a neural memory model that learns patterns during generation.

## Features

- **Neural Memory Model**: Built-in memory system that maintains state during generation
- **Memory Visualization**: Visual representation of the memory state during generation
- **Memory Influence Control**: Adjustable memory influence level via UI slider
- **Integrated Server**: Self-contained memory server implementation with no external dependencies
- **Automatic Initialization**: Memory server automatically starts and initializes

## Installation

First, make sure you have the base LLaDA GUI installed. Then install the additional dependencies for the memory system:

```bash
pip install -r requirements_memory.txt
```

## Usage

1. Launch the memory-enhanced GUI:
   ```bash
   python memory_integration_auto.py
   ```

2. The interface will appear with a new "Memory Visualization" tab and a "Use Memory Guidance" checkbox in the generation parameters.

3. The memory system should automatically connect during startup (indicated by a green status indicator).

4. Check the "Use Memory Guidance" box to enable memory-guided generation.

5. Adjust the memory influence slider to control how strongly the memory affects generation.

## How It Works

The cognitive memory system:

1. Maintains a hidden state vector during generation
2. Updates this state based on generated tokens
3. Influences subsequent token probabilities
4. Helps maintain coherence and consistency
5. Adapts to the generated text

## Technical Details

The memory system includes:

- **Memory Server**: A Flask-based server implementing the memory model
- **Memory Interface**: Communicates with the server via HTTP API
- **Memory Visualization**: Visualizes the memory state during generation
- **Memory Worker**: Extends the LLaDA worker with memory capabilities

## Troubleshooting

If you experience any issues:

1. Check the memory status indicator (should be green when connected)
2. Look for error messages in the console output
3. Try clicking the "Connect" button in the Memory Visualization tab
4. Restart the application if needed

## Advanced Usage

- **Memory Reset**: Click the "Reset Memory" button to clear the memory state
- **Memory Disconnect**: Click the "Connect/Disconnect" button to toggle memory system
- **Memory Visualization**: View the memory state as a color-coded heatmap

Using the Memory-Enhanced LLaDA GUI
The memory feature you've implemented adds cognitive memory capabilities to the LLaDA diffusion model, which helps improve coherence and consistency in longer text generations. Here's how to best use this feature:

Memory Connection: The memory system should automatically connect during startup. You'll see a green status indicator in the Memory Visualization tab if it's connected properly. If not, you can click the "Connect" button to manually establish the connection.
Enable Memory Guidance: In the Generation Parameters section, check the "Use Memory Guidance" checkbox to activate memory-guided generation.
Adjust Memory Influence: Use the slider in the Memory Visualization tab to control how strongly the memory affects generation:

Higher values (closer to 100%) will make the system rely more heavily on memory patterns
Lower values (closer to 0%) will reduce memory influence and allow more variation
The default setting of 30% provides a good balance for most use cases

Best Practices

For Long-Form Content: Memory guidance is most beneficial when generating longer texts where consistency is important, such as stories, articles, or technical explanations.
Prompt Engineering: Start with a clear, detailed prompt that establishes the context and style you want. The memory system will learn from this initial content.
Iterative Generation: For very long content, you might want to generate in sections:

Generate the first section
Review and potentially edit the output
Use the output as part of the prompt for the next section
The memory system will maintain continuity between generations


Memory Reset: If you're starting a completely new topic or want fresh generation without influence from previous content, use the "Reset Memory" button to clear the memory state.
Topic Consistency: The memory feature excels at maintaining topical consistency, keeping characters, concepts, and terminology consistent throughout a generation.

Advanced Tips

Memory Visualization: The colored blocks in the Memory Visualization tab represent the current memory state. Brighter red values indicate stronger activation in those memory dimensions.
Calibrating Memory Influence:

For creative writing: Use around 20-40% memory influence
For technical content: Use 40-60% for better terminology consistency
For very structured content: Use 60-80% for maximum coherence


When to Disconnect Memory: If you notice the system becoming too repetitive or fixated on certain patterns, you can try:

Reducing the memory influence
Resetting the memory state
Or completely disconnecting the memory system for that particular generation


GPU Performance: The memory system adds some computational overhead, particularly during the first generation after connecting. If you're experiencing performance issues on lower-end hardware, you might want to disable the memory feature for shorter generations.

Troubleshooting
If you encounter any issues:

Check if the memory system is properly connected (green indicator)
Try manually connecting via the "Connect" button
If problems persist, restart the application
Check the console for any error logs related to the memory server

The memory feature works best when generating text with a clear structure and context, helping the model maintain consistency that standard diffusion models sometimes struggle with in longer outputs.

## License

MIT: Same as the main LLaDA GUI project. See LICENSE file.
