#!/bin/bash

# Fix Memory Connection Issues Script
# This script addresses all potential memory connection issues in LLaDA GUI

echo "Starting memory connection fix..."

# Kill any existing memory server processes
echo "Killing any existing memory server processes..."
pkill -f "server.py"
pkill -f "server.js"
pkill -f "memory_server"

# Try to kill any process using port 3000
PORT_PROCESSES=$(lsof -ti:3000 2>/dev/null)
if [ -n "$PORT_PROCESSES" ]; then
    echo "Killing processes using port 3000: $PORT_PROCESSES"
    kill -9 $PORT_PROCESSES 2>/dev/null
fi

# Wait for processes to terminate
sleep 2

# Apply the memory widget patch
echo "Patching memory visualization widget..."
./venv/bin/python patch_memory_widget.py

# Fix tensor detach issues
echo "Fixing tensor detach issues..."
./venv/bin/python fix_titan_memory.py

# Delete existing memory model to avoid loading issues
echo "Deleting existing memory model..."
rm -f core/memory/memory_data/titan_memory_model.json

# Run the memory database fix script
echo "Running memory database fix..."
./venv/bin/python fix_memory_db.py

# Test Titan memory fixes
echo "Testing Titan memory fixes..."
./venv/bin/python test_titan_memory.py

# Start the direct memory server in the background 
echo "Starting memory server..."
./venv/bin/python direct_memory_fix.py &
MEMORY_PID=$!

# Wait for server to start
echo "Waiting for memory server to start..."
sleep 5

# Verify memory server is running
if nc -z localhost 3000; then
    echo "Memory server is running on port 3000."
else
    echo "Warning: Memory server does not appear to be running."
fi

# Run the application
echo "Starting LLaDA GUI with memory integration..."
./venv/bin/python run.py --memory

# Clean up memory server process
if [ -n "$MEMORY_PID" ]; then
    echo "Stopping memory server..."
    kill $MEMORY_PID
fi

echo "Memory fix complete."
