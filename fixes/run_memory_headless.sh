#!/bin/bash

# Headless memory server launcher for LLaDA GUI
# This script starts only the memory server component without requiring a display

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

# Check for port 3000 being in use
PORT_IN_USE=$(lsof -ti:3000 2>/dev/null)
if [ -n "$PORT_IN_USE" ]; then
    echo "Port 3000 is still in use by process $PORT_IN_USE. Attempting to kill..."
    kill -9 $PORT_IN_USE 2>/dev/null || true
    sleep 1
fi

echo "Starting headless memory server..."
"$PYTHON" run_memory_headless.py

# The Python script handles Ctrl+C and proper cleanup
