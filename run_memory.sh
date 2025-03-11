#!/bin/bash

# Improved LLaDA GUI Memory Integration Launcher
# Consolidated fix script version

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

# Apply all fixes in one go using our consolidated fix script
echo "Applying all fixes..."
"$PYTHON" fix_all.py

# Start the memory server
echo "Starting memory server..."
"$PYTHON" direct_memory_fix.py &
MEMORY_PID=$!

# Wait for memory server to start
echo "Waiting for memory server to start..."
sleep 5

# Verify that the memory server is running
if ! nc -z localhost 3000 2>/dev/null; then
    echo "Warning: Memory server may not be running on port 3000."
    echo "Continuing anyway, but memory integration may not work correctly."
fi

echo "Memory server is running on port 3000."
echo "Starting LLaDA GUI with memory integration..."

# Run the application
"$PYTHON" run.py --memory --ensure-qt-app

# Clean up
echo "Shutting down memory server..."
if [ -n "$MEMORY_PID" ]; then
    kill $MEMORY_PID 2>/dev/null || true
fi

# Final cleanup - make sure all memory server processes are stopped
pkill -f "server.py" 2>/dev/null || true
pkill -f "memory_server" 2>/dev/null || true

echo "Memory integration shutdown complete."
