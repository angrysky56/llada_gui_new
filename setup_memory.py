#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the memory server.
"""

import logging
import os
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the setup."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the memory server directory
    memory_server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')

    # Check if the directory exists
    if not os.path.isdir(memory_server_dir):
        logger.error(f"Memory server directory not found: {memory_server_dir}")
        return False

    # Change to the memory server directory
    os.chdir(memory_server_dir)
    logger.info(f"Changed to directory: {memory_server_dir}")

    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
        logger.info("Node.js is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Node.js is not installed. Please install Node.js and try again.")
        return False

    # Install Node.js dependencies
    logger.info("Installing Node.js dependencies...")
    try:
        subprocess.run(['npm', 'install'], check=True)
        logger.info("Node.js dependencies installed successfully")
    except subprocess.SubprocessError as e:
        logger.error(f"Error installing Node.js dependencies: {e}")
        return False

    # Install Python dependencies
    logger.info("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        logger.info("Python dependencies installed successfully")
    except subprocess.SubprocessError as e:
        logger.error(f"Error installing Python dependencies: {e}")
        return False

    # Create models directory if it doesn't exist
    models_dir = os.path.join(memory_server_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Created models directory: {models_dir}")

    # Make start script executable
    start_script = os.path.join(memory_server_dir, 'start_memory_server.sh')
    try:
        os.chmod(start_script, 0o755)  # rwxr-xr-x
        logger.info(f"Made start script executable: {start_script}")
    except OSError as e:
        logger.error(f"Error making start script executable: {e}")
        return False

    logger.info("Setup complete. You can now start the memory server using ./start_memory_server.sh")
    return True


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
