#!/bin/bash

# Kill any existing processes using port 3000
pkill -f "server.py"
pkill -f "server.js"
pkill -f "memory_server"
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Wait for processes to terminate
sleep 1

# Start the memory server in the background
./venv/bin/python direct_memory_fix.py &

# Wait for server to start
echo "Waiting for memory server to start..."
sleep 5
echo "Memory server should be running now."

# Now run the GUI with memory enabled
./venv/bin/python run.py --memory
