#!/bin/bash

# Start the MCP Titan Memory server in the background
echo "Starting MCP Titan Memory server..."
cd /home/ty/Repositories/mcp-titan-cognitive-memory
npm start &
MEMORY_PID=$!

# Wait for the server to initialize
echo "Waiting for memory server to initialize..."
sleep 5

# Start the LLaDA GUI with memory integration
echo "Starting LLaDA GUI with memory integration..."
cd /home/ty/Repositories/ai_workspace/llada_gui
python memory_integration.py

# When the main application exits, kill the memory server
kill $MEMORY_PID
