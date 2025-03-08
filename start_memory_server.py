#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to start the memory server.

This script provides a reliable way to start the memory server
using the improved server manager.
"""

import argparse
import logging
import os
import sys
import time

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
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory Server Starter")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground (don't keep script running)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Add repository root to path to find modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        from core.memory.memory_server.server_manager import MemoryServerManager

        # Create and start server manager
        logger.info("Starting memory server...")
        manager = MemoryServerManager()

        # Clean up any existing server processes
        if manager.is_port_in_use() and not manager.is_server_running():
            logger.warning("Port 3000 is in use but server is not responding properly")
            manager.stop()

        # Start the server
        if manager.start(background=True, wait=True):
            logger.info("Memory server started successfully")

            # If in foreground mode, exit here
            if args.foreground:
                return True

            # Wait for keyboard interrupt
            logger.info("Server is running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping memory server...")
                manager.stop()
                return True
        else:
            logger.error("Failed to start memory server")
            return False
    except ImportError:
        logger.error("Could not import server manager. Make sure your installation is correct.")
        return False
    except Exception as e:
        logger.error(f"Error starting memory server: {e}")
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
