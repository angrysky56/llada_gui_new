#!/bin/bash

# Start the memory server in headless mode
# This script starts just the memory server component without requiring a GUI

# Get script directory  
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Find Python interpreter
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo "Using virtual environment Python"
else
    PYTHON="python3"
    echo "Using system Python (virtual environment recommended)"
fi

# Kill any existing memory server processes
echo "Cleaning up any existing memory server processes..."
pkill -f "server.py" 2>/dev/null || true
pkill -f "memory_server" 2>/dev/null || true

# Wait for processes to terminate
sleep 1

# Run the headless server
echo "Starting memory server in headless mode..."
"$PYTHON" run_headless.py "$@"
