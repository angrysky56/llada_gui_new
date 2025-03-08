#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for memory server connection.

This script tests the connection to the memory server and verifies that
the memory integration is working properly.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_test")


def test_server_connection(host="localhost", port=3000):
    """Test the connection to the memory server."""
    server_urls = [
        f"http://{host}:{port}/status",
        f"http://{host}:{port}/api/status"
    ]

    for url in server_urls:
        try:
            logger.info(f"Testing connection to {url}...")
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info(f"Connection successful: {response.json()}")
                return True
        except Exception as e:
            logger.warning(f"Error connecting to {url}: {e}")

    logger.error("All connection attempts failed")
    return False


def test_server_api(host="localhost", port=3000):
    """Test the memory server API functionality."""
    logger.info("Testing memory server API...")

    # Base URL
    base_url = f"http://{host}:{port}"

    # Test endpoints
    endpoints = [
        # (method, endpoint, payload, expected_status)
        ("POST", "/init", {"inputDim": 64, "outputDim": 64}, 200),
        ("POST", "/api/init_model", {"inputDim": 64, "outputDim": 64}, 200),
        ("POST", "/forward", {"x": [0.0] * 64, "memoryState": [0.0] * 64}, 200),
        ("POST", "/api/forward_pass", {"x": [0.0] * 64, "memoryState": [0.0] * 64}, 200),
        ("POST", "/trainStep", {"x_t": [0.0] * 64, "x_next": [0.0] * 64}, 200),
        ("POST", "/api/train_step", {"x_t": [0.0] * 64, "x_next": [0.0] * 64}, 200),
        ("GET", "/api/memory_state", None, 200)
    ]

    results = []

    for method, endpoint, payload, expected_status in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            logger.info(f"Testing {method} {url}...")

            if method == "GET":
                response = requests.get(url, timeout=2)
            else:
                response = requests.post(url, json=payload, timeout=2)

            if response.status_code == expected_status:
                logger.info(f"Success: {method} {endpoint} returned status {response.status_code}")
                results.append(True)
            else:
                logger.warning(
                    f"Unexpected status: {method} {endpoint} returned {response.status_code} instead of {expected_status}")
                results.append(False)
        except Exception as e:
            logger.error(f"Error testing {method} {endpoint}: {e}")
            results.append(False)

    # Calculate success rate
    success_rate = sum(results) / len(results) if results else 0
    logger.info(f"API test completed - Success rate: {success_rate * 100:.1f}%")

    return success_rate >= 0.5  # At least 50% success


def start_test_server():
    """Start the memory server for testing."""
    logger.info("Starting memory server for testing...")

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_server_dir = os.path.join(script_dir, "core", "memory", "memory_server")

    # Try to use the server_manager first
    try:
        # Import the server manager from the project
        sys.path.insert(0, os.path.join(script_dir, "core", "memory"))
        from memory_server.server_manager import MemoryServerManager

        manager = MemoryServerManager()

        # Check if server is already running
        if manager.is_server_running():
            logger.info("Memory server is already running")
            return True

        # Try to start the server
        if manager.start(background=True, wait=True):
            logger.info("Memory server started successfully via server manager")
            return True

        logger.warning("Failed to start memory server via server manager")
    except Exception as e:
        logger.warning(f"Error using server manager: {e}")

    # Try to use the run_with_memory.py script
    try:
        logger.info("Trying to use run_with_memory.py to start server...")
        run_script = os.path.join(script_dir, "start_memory_server.py")

        if os.path.exists(run_script):
            # Run the script in a subprocess
            process = subprocess.Popen(
                [sys.executable, run_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for server to start
            time.sleep(2)

            # Check if server is running
            if test_server_connection():
                logger.info("Memory server started successfully via start_memory_server.py")
                return True

            logger.warning("Failed to start server via start_memory_server.py")
        else:
            logger.warning(f"Script not found: {run_script}")
    except Exception as e:
        logger.warning(f"Error starting server via script: {e}")

    # Try to directly start the server
    try:
        logger.info("Trying direct server start...")

        # Try Node.js server first
        node_server = os.path.join(memory_server_dir, "server.js")
        if os.path.exists(node_server):
            # Check if Node.js is available
            try:
                subprocess.run(["node", "--version"], check=True, capture_output=True)

                # Start the server
                process = subprocess.Popen(
                    ["node", node_server],
                    cwd=memory_server_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Wait for server to start
                time.sleep(2)

                # Check if server is running
                if test_server_connection():
                    logger.info("Memory server started successfully via direct Node.js start")
                    return True

                logger.warning("Failed to start Node.js server directly")
            except Exception as e:
                logger.warning(f"Error starting Node.js server: {e}")

        # Try Python server
        python_server = os.path.join(memory_server_dir, "server.py")
        if os.path.exists(python_server):
            # Start the server
            process = subprocess.Popen(
                [sys.executable, python_server],
                cwd=memory_server_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for server to start
            time.sleep(2)

            # Check if server is running
            if test_server_connection():
                logger.info("Memory server started successfully via direct Python start")
                return True

            logger.warning("Failed to start Python server directly")
    except Exception as e:
        logger.warning(f"Error starting server directly: {e}")

    logger.error("All server start attempts failed")
    return False


def test_memory_integration():
    """Test the memory integration module."""
    logger.info("Testing memory integration module...")

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add necessary paths
    sys.path.insert(0, script_dir)
    sys.path.insert(0, os.path.join(script_dir, "core"))

    # Try to import memory modules
    try:
        from core.memory.memory_integration import initialize_memory, get_memory_interface

        # Try to initialize memory
        if initialize_memory(start_server=True):
            logger.info("Memory initialization successful")

            # Get memory interface
            memory_interface = get_memory_interface()
            if memory_interface and memory_interface.initialized:
                logger.info("Memory interface is available and initialized")

                # Test forward pass
                try:
                    import numpy as np
                    result = memory_interface.forward_pass(np.random.rand(64))
                    logger.info(f"Forward pass result: {result}")

                    if "predicted" in result and "newMemory" in result:
                        logger.info("Forward pass test successful")
                        return True
                    else:
                        logger.warning("Forward pass did not return expected data")
                except Exception as e:
                    logger.error(f"Error in forward pass test: {e}")
            else:
                logger.warning("Memory interface not available or not initialized")
        else:
            logger.warning("Failed to initialize memory")
    except Exception as e:
        logger.error(f"Error importing memory modules: {e}")

    return False


def main():
    """Main function for memory testing."""
    parser = argparse.ArgumentParser(description="Test memory server and integration")
    parser.add_argument("--skip-start", action="store_true", help="Skip starting the server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=3000, help="Server port")

    args = parser.parse_args()

    # Start server if needed
    if not args.skip_start:
        if not start_test_server():
            logger.error("Failed to start memory server")
            return 1

    # Test server connection
    if not test_server_connection(args.host, args.port):
        logger.error("Server connection test failed")
        return 1

    # Test server API
    if not test_server_api(args.host, args.port):
        logger.error("Server API test failed")
        return 1

    # Test memory integration
    if not test_memory_integration():
        logger.error("Memory integration test failed")
        return 1

    logger.info("All tests passed! Memory system is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
