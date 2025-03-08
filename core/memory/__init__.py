#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory integration module for LLaDA GUI.

This module provides long-term memory capabilities for the LLaDA model,
allowing it to maintain context across generations.
"""

import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Check if all required memory modules are available
MEMORY_AVAILABLE = True

try:
    # Check memory integration
    from .memory_integration import (
        initialize_memory,
        get_memory_interface,
        reset_memory,
        MCPTitanMemoryInterface,
        MemoryVisualizationWidget
    )

    # Check for server manager
    try:
        from .memory_server.server_manager import (
            initialize_server_manager,
            get_server_manager
        )

        SERVER_MANAGER_AVAILABLE = True
    except ImportError:
        SERVER_MANAGER_AVAILABLE = False
        logger.warning("Memory server manager not available")

    # Try to create memory directories
    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), "memory_server", "models"), exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create memory models directory: {e}")

except ImportError as e:
    MEMORY_AVAILABLE = False
    logger.warning(f"Memory integration not available: {e}")


def is_memory_available():
    """Check if memory integration is available."""
    return MEMORY_AVAILABLE
