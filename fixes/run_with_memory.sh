#!/bin/bash

# Kill any existing memory server processes
pkill -f "server.py"
pkill -f "server.js"
pkill -f "memory_server"

# Wait for processes to terminate
sleep 1

# Fix memory database
./venv/bin/python fix_memory_db.py

# Run the application with memory
./venv/bin/python run_with_memory_fixed.py
