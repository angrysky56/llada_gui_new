#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified LLaDA GUI Launcher

This script provides a single entry point to launch the LLaDA GUI
with optional memory integration. It handles all the memory server
management automatically when memory integration is enabled.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
memory_server_process = None


def find_venv_python():
    """Find and return the path to the virtual environment Python if it exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def cleanup_memory_server():
    """Clean up the memory server when exiting."""
    global memory_server_process

    # If memory server process exists, terminate it
    if memory_server_process:
        logger.info("Cleaning up memory server...")
        try:
            # Try graceful termination
            memory_server_process.terminate()
            try:
                memory_server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                memory_server_process.kill()
        except Exception as e:
            logger.error(f"Error terminating memory server: {e}")

    # Run stop script as a final safety measure
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stop_script = os.path.join(script_dir, 'stop_memory_server.py')
        if os.path.exists(stop_script):
            subprocess.run([sys.executable, stop_script], check=False)
    except Exception as e:
        logger.error(f"Error running stop script: {e}")


def signal_handler(sig, frame):
    """Handle signals like Ctrl+C."""
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_memory_server()
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified LLaDA GUI Launcher")
    parser.add_argument("--memory", action="store_true", help="Enable memory integration")
    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register cleanup function
    import atexit
    atexit.register(cleanup_memory_server)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Command to run the GUI
    run_cmd = [sys.executable, os.path.join(script_dir, 'run.py')]

    # Start memory server if requested
    if args.memory:
        global memory_server_process

        logger.info("Starting memory server...")

        # First, ensure any existing memory server is stopped
        try:
            stop_script = os.path.join(script_dir, 'stop_memory_server.py')
            if os.path.exists(stop_script):
                subprocess.run([sys.executable, stop_script], check=False)
        except Exception as e:
            logger.error(f"Error stopping existing memory server: {e}")

        # Start the memory server
        try:
            start_script = os.path.join(script_dir, 'start_memory_server.py')
            if os.path.exists(start_script):
                memory_server_process = subprocess.Popen([sys.executable, start_script, '--foreground'])

                # Wait a bit for server to start
                logger.info("Waiting for memory server to initialize...")
                time.sleep(3)

                # Add memory flag to run command
                run_cmd.append('--memory')
            else:
                logger.error("Memory server start script not found")
                return 1
        except Exception as e:
            logger.error(f"Error starting memory server: {e}")
            return 1

    # Run the GUI
    try:
        logger.info(f"Starting LLaDA GUI with command: {' '.join(run_cmd)}")
        subprocess.run(run_cmd, check=False)
    except Exception as e:
        logger.error(f"Error starting LLaDA GUI: {e}")
        return 1

    # Clean up
    if args.memory:
        cleanup_memory_server()

    return 0


if __name__ == "__main__":
    # Check if we're already running in the right Python
    if 'VENV_ACTIVATED' not in os.environ:
        venv_python = find_venv_python()
        if venv_python != sys.executable:
            logger.info(f"Restarting with Python from virtual environment: {venv_python}")
            os.environ['VENV_ACTIVATED'] = '1'
            os.execl(venv_python, venv_python, *sys.argv)

    # Run main function
    sys.exit(main())
