#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to stop the memory server.

This script provides a reliable way to stop the memory server
using the improved server manager.
"""

import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_venv_python():
    """Find and return the path to the virtual environment Python if it exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def main():
    """Main function to stop the memory server."""
    # Add repository root to path to find modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        from core.memory.memory_server.server_manager import MemoryServerManager

        # Create server manager
        logger.info("Stopping memory server...")
        manager = MemoryServerManager()

        # Stop the server
        if manager.stop():
            logger.info("Memory server stopped successfully")
            return True
        else:
            logger.error("Failed to stop memory server")
            return False
    except ImportError:
        logger.error("Could not import server manager. Make sure your installation is correct.")
        return False
    except Exception as e:
        logger.error(f"Error stopping memory server: {e}")
        return False


if __name__ == "__main__":
    # Check if we're already running in the right Python
    if 'VENV_ACTIVATED' not in os.environ:
        venv_python = find_venv_python()
        if venv_python != sys.executable:
            logger.info(f"Restarting with Python from virtual environment: {venv_python}")
            os.environ['VENV_ACTIVATED'] = '1'
            os.execl(venv_python, venv_python, *sys.argv)

    # Run main function
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
