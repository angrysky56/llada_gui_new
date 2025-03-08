#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the direct_memory_fix.py works correctly.
"""

import subprocess
import sys
import time

import requests


def main():
    """Test if the memory server can be started using direct_memory_fix.py."""
    print("Starting memory server using direct_memory_fix.py...")

    # Start the memory server in a subprocess
    try:
        process = subprocess.Popen(
            [sys.executable, "direct_memory_fix.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait a moment for server to start
        print("Waiting for memory server to start...")
        time.sleep(5)

        # Check if server is responding
        try:
            response = requests.get("http://localhost:3000/status", timeout=2)
            if response.status_code == 200:
                print("Memory server is running and responding.")
                print("Response:", response.json())

                # Test initialization
                init_response = requests.post(
                    "http://localhost:3000/api/init_model",
                    json={"inputDim": 64, "outputDim": 64},
                    timeout=2
                )

                if init_response.status_code == 200:
                    print("Memory model initialized successfully!")
                    return 0
                else:
                    print("Error initializing memory model:", init_response.text)
                    return 1
            else:
                print("Memory server responded with unexpected status code:", response.status_code)
                return 1
        except Exception as e:
            print("Error connecting to memory server:", e)
            return 1
        finally:
            # Try to terminate the server process
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                # If it doesn't terminate, kill it
                try:
                    process.kill()
                except:
                    pass
    except Exception as e:
        print("Error starting memory server:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
