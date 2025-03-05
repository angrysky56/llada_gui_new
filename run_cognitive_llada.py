#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launcher for Cognitive LLaDA GUI.

This script sets up the environment and launches the Cognitive LLaDA GUI,
which integrates the LLaDA diffusion model with the Titan Memory system.
"""

import os
import sys
from PyQt6.QtWidgets import QApplication

# Create the memory data directory if it doesn't exist
MEMORY_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_data")
os.makedirs(MEMORY_DATA_DIR, exist_ok=True)

def main():
    """Main function."""
    # Import the GUI class
    try:
        from cognitive_llada import CognitiveLLaDAGUI
    except ImportError as e:
        print(f"Error importing Cognitive LLaDA GUI: {e}")
        print("Make sure all required modules are installed.")
        return 1
    
    # Create and run the application
    app = QApplication(sys.argv)
    
    # Create the main window
    window = CognitiveLLaDAGUI()
    
    # Show the window
    window.show()
    
    # Run the application event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
