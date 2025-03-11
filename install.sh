#!/bin/bash

# LLaDA GUI Installation Script - Updated to use fixes directory
echo "Installing LLaDA GUI..."

# Get the project root directory
PROJECT_ROOT=$(pwd)
FIXES_DIR="$PROJECT_ROOT/fixes"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install all requirements
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements_memory.txt
else
    # Activate existing virtual environment
    source venv/bin/activate
    echo "Using existing virtual environment"
fi

# Install additional dependencies for memory integration
pip install flask flask-cors numpy torch
# Create necessary directories
mkdir -p data/memory/vector_db
mkdir -p core/memory/memory_server/models

# Make scripts executable
chmod +x run.sh
if [ -f "$FIXES_DIR/run_with_memory.sh" ]; then
    cp "$FIXES_DIR/run_with_memory.sh" run_with_memory.sh
    chmod +x run_with_memory.sh
fi

if [ -f "$FIXES_DIR/fix_run_memory.sh" ]; then
    cp "$FIXES_DIR/fix_run_memory.sh" fix_run_memory.sh
    chmod +x fix_run_memory.sh
fi

# Make server scripts executable if they exist
if [ -f "core/memory/memory_server/server.py" ]; then
    chmod +x core/memory/memory_server/server.py
fi
if [ -f "core/memory/memory_server/server_manager.py" ]; then
    chmod +x core/memory/memory_server/server_manager.py
fi

# Define Python path
PYTHON="venv/bin/python"

# Apply memory fixes from the fixes directory
echo "Applying memory fixes..."
if [ -f "$FIXES_DIR/fix_memory_db.py" ]; then
    "$PYTHON" "$FIXES_DIR/fix_memory_db.py"
else
    echo "Warning: fix_memory_db.py not found in fixes directory, skipping"
fi

if [ -f "$FIXES_DIR/fix_titan_memory.py" ]; then
    "$PYTHON" "$FIXES_DIR/fix_titan_memory.py"
else
    echo "Warning: fix_titan_memory.py not found in fixes directory, skipping"
fi

if [ -f "$FIXES_DIR/direct_memory_fix.py" ]; then
    cp "$FIXES_DIR/direct_memory_fix.py" direct_memory_fix.py
    chmod +x direct_memory_fix.py
    "$PYTHON" direct_memory_fix.py --prepare-only
else
    echo "Warning: direct_memory_fix.py not found in fixes directory, skipping"
fi

# Create desktop icons (preferred method)
echo "Creating desktop icons..."
if [ -f "$FIXES_DIR/create_desktop_icons.sh" ]; then
    bash "$FIXES_DIR/create_desktop_icons.sh"
    echo "Desktop icons created successfully"
else
    echo "Warning: create_desktop_icons.sh not found in fixes directory, skipping"
fi

echo "Installation complete! You can now run LLaDA GUI from your desktop icons."
echo "Or use ./run.sh for regular mode and ./run_with_memory.sh for memory mode."
