#!/bin/bash

# LLaDA GUI Installation Script

echo "============================================"
echo "LLaDA GUI Installation Script"
echo "============================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists and create it if necessary
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Please make sure python3-venv is installed."
        exit 1
    fi
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing required packages from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Some packages may not have installed correctly."
    fi
else
    echo "requirements.txt not found. Installing essential packages..."
    pip install PyQt6 torch transformers==4.38.2 numpy bitsandbytes psutil
fi

# Make sure PyQt6 is installed
echo "Checking for PyQt6..."
pip install PyQt6

# Make scripts executable
chmod +x run.py
chmod +x llada_gui.py
chmod +x start_gui.sh

# Copy the desktop file to the user's desktop if it exists
if [ -d "$HOME/Desktop" ]; then
    echo "Creating desktop shortcut..."
    cp LLaDA_GUI.desktop $HOME/Desktop/
    chmod +x $HOME/Desktop/LLaDA_GUI.desktop
fi

echo "============================================"
echo "Installation completed!"
echo "You can now run the application with:"
echo "  ./start_gui.sh"
echo "or"
echo "  ./run.py"
echo "============================================"
