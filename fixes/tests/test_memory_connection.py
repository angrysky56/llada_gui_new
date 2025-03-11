#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test memory server connection without Qt dependencies.

This script verifies that the memory server is running and accessible.
It doesn't require any GUI components.
"""

import requests
import time
import sys
import subprocess
import os
import signal

def main():
    """Check if memory server is running at the specified port."""
    port = 3000
    url = f"http://localhost:{port}/status"
    
    # First try to connect to an existing server
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print("Memory server is already running!")
            return 0
    except Exception:
        print("No memory server detected, will start one.")
    
    # Start the memory server in the background
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_process = subprocess.Popen(
        [sys.executable, os.path.join(current_dir, "run_memory_headless.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    
    # Wait for the server to start
    print("Waiting for server to start...")
    for i in range(10):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("Memory server started successfully!")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("Failed to start memory server.")
        return 1
    
    # Initialize memory model
    try:
        response = requests.post(
            f"http://localhost:{port}/init",
            json={"inputDim": 64, "outputDim": 64},
            timeout=2
        )
        if response.status_code == 200:
            print("Memory model initialized successfully.")
        else:
            print(f"Error initializing memory model: {response.status_code}")
    except Exception as e:
        print(f"Error initializing memory model: {e}")
    
    # Get memory state
    try:
        response = requests.get(f"http://localhost:{port}/api/memory_state", timeout=2)
        if response.status_code == 200:
            state = response.json()
            print(f"Memory state retrieved: {len(state['memoryState'])} elements")
        else:
            print(f"Error getting memory state: {response.status_code}")
    except Exception as e:
        print(f"Error getting memory state: {e}")
    
    # Simulate storing a memory
    try:
        import numpy as np
        
        # Fake input and next vectors
        input_vec = np.random.rand(64).tolist()
        next_vec = np.random.rand(64).tolist()
        
        response = requests.post(
            f"http://localhost:{port}/trainStep",
            json={"x_t": input_vec, "x_next": next_vec},
            timeout=2
        )
        if response.status_code == 200:
            print("Memory trained successfully.")
        else:
            print(f"Error training memory: {response.status_code}")
    except Exception as e:
        print(f"Error training memory: {e}")
    
    # Wait a moment to see output
    time.sleep(3)
    
    # Clean up
    print("Test completed. Stopping server...")
    try:
        # Try to kill the process group
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    except Exception:
        try:
            # Fallback to terminating just the process
            server_process.terminate()
        except Exception as e:
            print(f"Error stopping server: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
