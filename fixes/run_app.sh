#!/bin/bash

# Clean launcher for LLaDA GUI
# This script cleans up any existing memory server and launches the GUI

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python virtual environment
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python"
fi

echo "=== LLaDA GUI Launcher ==="
echo "Checking for existing memory server..."

# Kill any memory server processes
PIDS=$(lsof -i :3000 -t 2>/dev/null)
if [ ! -z "$PIDS" ]; then
    echo "Found memory server processes: $PIDS"
    for PID in $PIDS; do
        echo "Killing process $PID..."
        kill -9 $PID 2>/dev/null
    done
    sleep 1
    
    # Verify they are killed
    PIDS=$(lsof -i :3000 -t 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo "Warning: Some processes still using port 3000: $PIDS"
    else
        echo "All memory server processes terminated."
    fi
else
    echo "No memory server processes found."
fi

# Run the actual app
echo "Launching LLaDA GUI..."
"$PYTHON" run.py "$@"

echo "LLaDA GUI session ended."
