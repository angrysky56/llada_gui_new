#!/bin/bash

# Script to start the memory server first, then launch the LLaDA GUI
echo "Starting Memory Server and LLaDA GUI..."

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Python virtual environment if it exists
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo "Activating virtual environment..."
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Fix memory server dependencies
echo "Checking memory server dependencies..."
python "${SCRIPT_DIR}/fix_memory_dependencies.py" || {
    echo "Warning: Failed to fix memory server dependencies"
}

# Start memory server
echo "Starting memory server..."
python "${SCRIPT_DIR}/start_memory_server.py" &
MEMORY_SERVER_PID=$!

# Wait for server to start
echo "Waiting for memory server to start..."
sleep 5

# Run the application
echo "Launching application with memory integration..."
python "${SCRIPT_DIR}/run.py" --memory

# When application terminates, kill memory server
echo "Shutting down memory server..."
kill -9 $MEMORY_SERVER_PID 2>/dev/null

# Deactivate virtual environment if activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
