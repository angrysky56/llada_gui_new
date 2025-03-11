#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory system initialization.

This module provides functions for initializing and managing the memory system.
"""

import time
import logging
import numpy as np
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_server.log'
)
logger = logging.getLogger("memory_init")

# Import the server manager
try:
    from .memory_server.server_manager import MemoryServerManager
    SERVER_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Memory server manager not available, will run without server control")
    SERVER_MANAGER_AVAILABLE = False

# Import the fallback server
try:
    from .memory_server.fallback_server import try_python_server_fallback
    FALLBACK_SERVER_AVAILABLE = True
except ImportError:
    logger.warning("Fallback server not available")
    FALLBACK_SERVER_AVAILABLE = False
    
# Import the memory interface
try:
    from .titan_integration.memory_interface import MCPTitanMemoryInterface
    MEMORY_INTERFACE_AVAILABLE = True
except ImportError:
    logger.warning("Memory interface not available")
    MEMORY_INTERFACE_AVAILABLE = False

# Global variables
_server_manager = None
_memory_interface = None
_server_started = False

def initialize_server_manager(auto_start=False):
    """Initialize the server manager.
    
    Args:
        auto_start: Whether to start the server automatically
        
    Returns:
        MemoryServerManager instance or None if not available
    """
    global _server_manager

    if _server_manager is not None:
        return _server_manager

    if not SERVER_MANAGER_AVAILABLE:
        logger.warning("Server manager not available, will try Python fallback if needed")
        return None

    try:
        # Create server manager
        _server_manager = MemoryServerManager()

        # Auto-start if requested
        if auto_start:
            # First check if server is already running
            if not _server_manager.is_server_running():
                logger.info("Auto-starting memory server...")

                # Ensure port is free first
                if _server_manager.is_port_in_use():
                    logger.info("Port in use but server not responding, cleaning up...")
                    _server_manager.stop()

                # Now start the server
                for attempt in range(2):  # Try twice
                    logger.info(f"Starting memory server (attempt {attempt + 1}/2)")
                    if _server_manager.start(background=True, wait=True):
                        logger.info("Memory server started successfully")
                        return _server_manager
                    time.sleep(1)

                logger.error("Failed to start memory server with server manager")
                # If we can't start the server with the manager, try Python fallback
                if FALLBACK_SERVER_AVAILABLE and try_python_server_fallback():
                    logger.info("Python fallback server started successfully")
                else:
                    logger.error("All server start attempts failed")
            else:
                logger.info("Memory server is already running")

        return _server_manager
    except Exception as e:
        logger.error(f"Error initializing server manager: {e}")
        # Try Python fallback as last resort
        if auto_start and FALLBACK_SERVER_AVAILABLE and try_python_server_fallback():
            logger.info("Python fallback server started after server manager error")
        return None


def get_server_manager():
    """Get the server manager instance.
    
    Returns:
        MemoryServerManager instance or None if not available
    """
    global _server_manager
    return _server_manager


def initialize_memory(start_server=True, max_retries=5):
    """Initialize the memory system.
    
    Args:
        start_server: Whether to start the memory server if not running
        max_retries: Maximum number of retries for server start
    
    Returns:
        True if successful, False otherwise
    """
    global _memory_interface, _server_started

    # Check if memory interface is available
    if not MEMORY_INTERFACE_AVAILABLE:
        logger.error("Memory interface not available")
        return False

    # If already initialized and working, return success
    if _memory_interface is not None and _memory_interface.initialized:
        # Verify connectivity
        try:
            response = requests.get("http://localhost:3000/status", timeout=1)
            if response.status_code == 200:
                # Already initialized and server is responsive
                logger.info("Memory server is already running and responding")
                return True
            # Server isn't responding properly even though interface is initialized
            logger.warning("Memory interface initialized but server not responding, will restart")
            _memory_interface.initialized = False
        except Exception as e:
            # Server isn't accessible, interface needs reinitialization
            logger.warning(f"Memory interface initialized but server not accessible, will restart: {e}")
            _memory_interface.initialized = False

    # Initialize server manager first
    if start_server and SERVER_MANAGER_AVAILABLE:
        # Get or create the server manager
        server_manager = get_server_manager() or initialize_server_manager()

        # Start the server if not running
        try:
            if server_manager:
                if not server_manager.is_server_running():
                    logger.info("Starting memory server...")

                    # Stop any existing misbehaving server
                    if server_manager.is_port_in_use():
                        logger.info("Port in use but server not responding, cleaning up...")
                        server_manager.stop()

                    # Now start the server with retries
                    for attempt in range(max_retries):
                        logger.info(f"Starting memory server (attempt {attempt + 1}/{max_retries})")
                        if server_manager.start(background=True, wait=True):
                            _server_started = True
                            logger.info("Memory server started successfully")
                            break
                        else:
                            logger.info(f"Failed on attempt {attempt + 1}")
                            # Wait briefly before retry
                            time.sleep(1)

                    if not _server_started:
                        logger.warning("Failed to start memory server after all attempts, continuing in standalone mode")
                        # Try to use Python server as fallback
                        if FALLBACK_SERVER_AVAILABLE:
                            try_python_server_fallback()
                else:
                    logger.info("Memory server is already running")
                    _server_started = True
        except Exception as e:
            logger.error(f"Error managing memory server: {e}")
            # Try Python server fallback
            if FALLBACK_SERVER_AVAILABLE:
                try_python_server_fallback()

    # Create memory interface if not exists
    if _memory_interface is None:
        _memory_interface = MCPTitanMemoryInterface()

    # Try to initialize
    return _memory_interface.initialize()


def get_memory_interface():
    """Get the memory interface instance.
    
    Returns:
        MCPTitanMemoryInterface instance, or None if not initialized
    """
    global _memory_interface
    return _memory_interface


def reset_memory():
    """Reset the memory state.
    
    Returns:
        True if successful, False otherwise
    """
    global _memory_interface

    if _memory_interface is None or not _memory_interface.initialized:
        return False

    _memory_interface.reset()
    return True
