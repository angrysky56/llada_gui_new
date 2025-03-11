#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix memory server dependencies.
This script checks and installs required dependencies for the memory server.
"""

import importlib.util
import logging
import os
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required packages
REQUIRED_PACKAGES = ['numpy', 'flask', 'requests', 'torch']


def check_package(package_name):
    """Check if a package is installed."""
    if importlib.util.find_spec(package_name) is not None:
        return True
    return False


def install_package(package_name):
    """Install a package using pip."""
    logger.info(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False


def setup_memory_server():
    """Set up the memory server environment."""
    logger.info("Setting up memory server dependencies...")

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')

    # Check if server directory exists
    if not os.path.isdir(server_dir):
        logger.error(f"Memory server directory not found: {server_dir}")
        return False

    # Check for requirements.txt
    req_file = os.path.join(server_dir, 'requirements.txt')
    if os.path.isfile(req_file):
        logger.info(f"Installing requirements from {req_file}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
            logger.info("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {e}")

            # Try installing individual packages
            logger.info("Trying to install individual packages...")
            for package in REQUIRED_PACKAGES:
                install_package(package)
    else:
        # No requirements file, install packages individually
        logger.info("No requirements.txt found, installing individual packages...")
        for package in REQUIRED_PACKAGES:
            if not check_package(package):
                install_package(package)

    # Check if packages are installed
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False

    logger.info("All required packages are installed")
    return True


if __name__ == "__main__":
    if setup_memory_server():
        logger.info("Memory server dependencies installed successfully")
        sys.exit(0)
    else:
        logger.error("Failed to set up memory server dependencies")
        sys.exit(1)
