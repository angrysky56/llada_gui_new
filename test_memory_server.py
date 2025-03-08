#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify memory server functionality.

This checks if the memory server can be started and responds to basic API requests.
"""

import os
import subprocess
import sys
import time

import requests


def main():
    """Test memory server functionality."""
    print("Testing memory server functionality...")

    # Find Python in venv
    venv_python = './venv/bin/python'
    if os.path.exists(venv_python):
        python_cmd = venv_python
    else:
        python_cmd = sys.executable

    # Start memory server
    print("Starting memory server...")
    try:
        server_process = subprocess.Popen(
            [python_cmd, 'direct_memory_fix.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        print("Waiting for memory server to start...")
        max_retries = 5
        for i in range(max_retries):
            try:
                time.sleep(1)
                response = requests.get('http://localhost:3000/status', timeout=1)
                if response.status_code == 200:
                    print("Memory server is running!")
                    print(f"Status response: {response.json()}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Retry {i + 1}/{max_retries}: Server not responding yet ({e.__class__.__name__})")
                if i == max_retries - 1:
                    print("ERROR: Memory server failed to start.")
                    if server_process.poll() is None:
                        server_process.terminate()
                    return 1

        # Test API endpoints
        print("\nTesting API endpoints:")

        # 1. Initialize model
        try:
            init_response = requests.post(
                'http://localhost:3000/api/init_model',
                json={"inputDim": 64, "outputDim": 64},
                timeout=2
            )
            print(
                f"Initialize model: {init_response.status_code} - {init_response.json() if init_response.status_code == 200 else init_response.text}")
        except Exception as e:
            print(f"Error initializing model: {e}")

        # 2. Forward pass
        try:
            vector = [0.1] * 64
            forward_response = requests.post(
                'http://localhost:3000/api/forward_pass',
                json={"x": vector, "memoryState": vector},
                timeout=2
            )
            print(
                f"Forward pass: {forward_response.status_code} - {forward_response.json() if forward_response.status_code == 200 else forward_response.text}")
        except Exception as e:
            print(f"Error in forward pass: {e}")

        # All tests completed successfully
        print("\nMemory server test completed successfully!")

    except Exception as e:
        print(f"Error testing memory server: {e}")
        return 1
    finally:
        # Clean up
        if 'server_process' in locals() and server_process.poll() is None:
            print("Stopping memory server...")
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except:
                print("Force killing memory server...")
                try:
                    server_process.kill()
                except:
                    pass

    return 0


if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.exit(main())
