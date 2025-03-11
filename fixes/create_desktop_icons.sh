#!/bin/bash

# Script to create desktop icons for LLaDA GUI

# Get the absolute path to the project directory (parent of the fixes directory)
FIXES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$FIXES_DIR")"

echo "Creating desktop icons with project directory: $PROJECT_DIR"

# Create standard desktop icon
echo "Creating standard LLaDA GUI desktop icon..."
cat > ~/Desktop/LLaDA_GUI.desktop << EOF
[Desktop Entry]
Type=Application
Name=LLaDA GUI
Comment=LLaDA Text Generation
Exec=bash -c "cd ${PROJECT_DIR} && ./run.sh"
Icon=applications-ai
Terminal=false
Categories=AI;MachineLearning;
Keywords=AI;Text;Generation;LLM;
StartupNotify=true
EOF

# Create memory-enabled desktop icon
echo "Creating LLaDA GUI with Memory desktop icon..."
cat > ~/Desktop/LLaDA_GUI_Memory.desktop << EOF
[Desktop Entry]
Type=Application
Name=LLaDA GUI with Memory
Comment=LLaDA Text Generation with Memory Integration
Exec=bash -c "cd ${PROJECT_DIR} && ./run_memory.sh"
Icon=applications-science
Terminal=false
Categories=AI;MachineLearning;
Keywords=AI;Text;Generation;LLM;Memory;
StartupNotify=true
EOF

# Make them executable
chmod +x ~/Desktop/LLaDA_GUI.desktop
chmod +x ~/Desktop/LLaDA_GUI_Memory.desktop

echo "Desktop icons created successfully!"
