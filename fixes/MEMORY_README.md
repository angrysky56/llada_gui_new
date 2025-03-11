# LLaDA Memory Integration Guide

This document provides instructions for using the memory integration feature in LLaDA GUI.

## Overview

The memory component allows LLaDA to maintain context and improve generation quality by learning from previous interactions. It can be run in two modes:

1. **GUI Mode**: Full GUI with memory integration
2. **Headless Mode**: Memory server only, without GUI (useful for servers or API integrations)

## Quick Start

### GUI Mode (with Display)

To run the full LLaDA GUI with memory integration:

```bash
./run_memory.sh
```

### Headless Mode (without Display)

To run just the memory server component without requiring a display:

```bash
./start_memory_server.sh
```

This starts the memory server on localhost:3000 by default. You can customize the host and port:

```bash
./start_memory_server.sh --host 0.0.0.0 --port 3001
```

## Troubleshooting

If you encounter any issues with the memory integration, try these steps:

1. **Clean Start**: Run the consolidated fix script to repair any issues:
   ```bash
   python fix_all.py
   ```

2. **Check Dependencies**: Ensure all required packages are installed:
   ```bash
   pip install -r requirements_memory.txt
   ```

3. **Server Connection**: If the GUI cannot connect to the memory server, check if it's running:
   ```bash
   curl http://localhost:3000/status
   ```

4. **Memory Database**: Reset the memory database if it becomes corrupted:
   ```bash
   python fix_memory_db.py
   ```

5. **Log Files**: Check the log files for error messages:
   - `memory_server.log`: Memory server logs
   - `llada_gui.log`: GUI logs

## Technical Details

The memory integration consists of several components:

1. **Memory Server**: Flask-based server that handles memory state and processing
2. **Memory Database**: Stores and manages memory records
3. **Memory Integration**: Connects the GUI to the memory system
4. **Visualization**: Provides a visual representation of memory state

The server communicates over HTTP on port 3000 (by default) and provides endpoints for:
- Initialization
- Forward pass (prediction)
- Training
- Memory state management

## API Endpoints

The memory server provides these main endpoints:

- `GET /status`: Check server status
- `POST /init`: Initialize memory model
- `POST /forward`: Get predictions from memory
- `POST /trainStep`: Train on new data
- `GET /api/memory_state`: Get current memory state
- `POST /api/reset_memory`: Reset memory to initial state

## Architecture

```
core/
  └─ memory/
     ├─ memory_data/         # Memory models and databases
     ├─ memory_server/       # Memory server implementation
     ├─ memory_integration.py # Integration with GUI
     └─ memory_adapter.py    # Adapter for memory visualization
```

The memory system uses a simplified memory model to enhance generation quality by providing contextual guidance based on learned patterns.
