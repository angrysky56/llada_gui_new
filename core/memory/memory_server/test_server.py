#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Titan Memory Server.

This script provides a simple test to verify that the memory server
is working correctly. It starts the server, initializes a model,
and performs some basic operations.
"""

import argparse
import os
import sys
import time

import numpy as np
import requests

from server_manager import MemoryServerManager


def test_memory_server(host='127.0.0.1', port=3000):
    """Run a basic test of the memory server.
    
    Args:
        host: Host to connect to
        port: Port to use
    """
    print(f"Testing memory server at {host}:{port}")
    base_url = f"http://{host}:{port}"

    # Start the server
    print("\n1. Starting server...")
    server_manager = MemoryServerManager(host, port)
    if not server_manager.start():
        print("Failed to start memory server!")
        return False

    # Wait for server to start
    time.sleep(2)

    # Check status
    print("\n2. Checking server status...")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        response.raise_for_status()
        print(f"Server status: {response.json()}")
    except Exception as e:
        print(f"Failed to get server status: {str(e)}")
        server_manager.stop()
        return False

    # Initialize model
    print("\n3. Initializing model...")
    try:
        response = requests.post(
            f"{base_url}/init",
            json={"inputDim": 32, "outputDim": 32},
            timeout=5
        )
        response.raise_for_status()
        print(f"Model initialization response: {response.json()}")
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        server_manager.stop()
        return False

    # Run forward pass
    print("\n4. Running forward pass...")
    try:
        # Create random input
        x = np.random.randn(32).tolist()

        response = requests.post(
            f"{base_url}/forward",
            json={"x": x},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        print(f"Forward pass successful")
        print(f"Surprise value: {result.get('surprise', 'N/A')}")
    except Exception as e:
        print(f"Failed to run forward pass: {str(e)}")
        server_manager.stop()
        return False

    # Run training step
    print("\n5. Running training step...")
    try:
        # Create random input and target
        x_t = np.random.randn(32).tolist()
        x_next = np.random.randn(32).tolist()

        response = requests.post(
            f"{base_url}/trainStep",
            json={"x_t": x_t, "x_next": x_next},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        print(f"Training step successful")
        print(f"Cost value: {result.get('cost', 'N/A')}")
    except Exception as e:
        print(f"Failed to run training step: {str(e)}")
        server_manager.stop()
        return False

    # Test model save/load
    print("\n6. Testing model save/load...")
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model.json")

    try:
        # Save model
        response = requests.post(
            f"{base_url}/save",
            json={"path": save_path},
            timeout=5
        )
        response.raise_for_status()
        print(f"Model saved to {save_path}")

        # Load model
        response = requests.post(
            f"{base_url}/load",
            json={"path": save_path},
            timeout=5
        )
        response.raise_for_status()
        print(f"Model loaded from {save_path}")
    except Exception as e:
        print(f"Failed to save/load model: {str(e)}")
        server_manager.stop()
        return False

    # Clean up
    print("\n7. Stopping server...")
    server_manager.stop()

    print("\nAll tests passed! Memory server is working correctly.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the memory server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to connect to")
    parser.add_argument("--port", type=int, default=3000, help="Port to use")

    args = parser.parse_args()

    success = test_memory_server(args.host, args.port)
    sys.exit(0 if success else 1)
