#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced LLaDA GUI with MCP Titan Memory integration.

This extends the existing LLaDA GUI to incorporate cognitive memory capabilities
using the MCP Titan Memory system.

This is the main integration module that ties together all memory components.
"""

import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_server.log'
)
logger = logging.getLogger("memory_integration")

# Import memory initialization functions
from .memory_init import (
    initialize_memory,
    get_memory_interface,
    reset_memory,
    initialize_server_manager,
    get_server_manager
)

# Import the memory interface
try:
    from .titan_integration.memory_interface import MCPTitanMemoryInterface
except ImportError:
    logger.error("Memory interface not available")
    MCPTitanMemoryInterface = None

# Import the server fallback
try:
    from .memory_server.fallback_server import try_python_server_fallback
except ImportError:
    logger.warning("Fallback server not available")
    try_python_server_fallback = None

# Import the UI components
try:
    from .memory_ui import (
        MemoryVisualizationWidget,
        TrainingThread,
        MemoryGuidanceDiffusionWorker,
        enhance_llada_gui
    )
except ImportError:
    logger.error("Memory UI components not available")
    MemoryVisualizationWidget = None
    TrainingThread = None
    MemoryGuidanceDiffusionWorker = None
    enhance_llada_gui = None

# Re-export important components
__all__ = [
    # Main functions
    'initialize_memory',
    'get_memory_interface',
    'reset_memory',
    'enhance_llada_gui',
    'main',
    
    # Classes
    'MCPTitanMemoryInterface',
    'MemoryVisualizationWidget',
    'TrainingThread',
    'MemoryGuidanceDiffusionWorker',
    
    # Server management
    'initialize_server_manager',
    'get_server_manager',
    'try_python_server_fallback'
]

def main():
    """Main function to launch the enhanced GUI."""
    # Import the original LLaDAGUI
    try:
        from llada_gui import LLaDAGUI
    except ImportError:
        logger.error("Could not import LLaDAGUI. Make sure it's in the Python path.")
        print("Error: Could not import LLaDAGUI. Make sure it's in the Python path.")
        return 1

    # Make sure we have the enhancer
    if enhance_llada_gui is None:
        logger.error("Memory UI enhancement function not available")
        print("Error: Memory UI enhancement function not available")
        return 1

    # Create enhanced version
    EnhancedLLaDAGUI = enhance_llada_gui(LLaDAGUI)

    # Initialize server manager before application starts
    initialize_server_manager(auto_start=False)

    # Launch the application
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = EnhancedLLaDAGUI()
    window.show()

    # Start the Qt event loop
    exit_code = app.exec()

    # Cleanup on exit - stop server if we started it
    server_manager = get_server_manager()
    if server_manager is not None:
        logger.info("Stopping memory server...")
        server_manager.stop()

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
