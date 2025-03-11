#!/bin/bash

# Make sure we're running from the right directory
cd "$(dirname "$0")"

# Kill any existing memory server processes
pkill -f "server.py" || true
pkill -f "server.js" || true
pkill -f "memory_server" || true

# Wait for processes to terminate
sleep 1

echo "Starting memory server..."
python3 direct_memory_fix.py &
MEMORY_PID=$!

# Wait for memory server to start
sleep 5

echo "Memory server is running on port 3000."
echo "Starting LLaDA GUI with memory integration..."

# Start LLaDA GUI with memory integration
if [ -d "venv" ]; then
    # Use virtual environment if available
    source venv/bin/activate
    python run_with_memory_fixed.py
else
    # Use system Python otherwise
    python3 run_with_memory_fixed.py
fi

# Clean up memory server
kill $MEMORY_PID || true
