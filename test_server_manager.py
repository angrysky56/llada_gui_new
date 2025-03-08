#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the memory server manager.
"""

import os
import sys
import time

# Add repository root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main function."""
    print("Testing memory server manager...")

    try:
        from core.memory.memory_server.server_manager import MemoryServerManager
        print("Successfully imported server manager")

        manager = MemoryServerManager()
        print("Created server manager instance")

        # Check if server is running
        if manager.is_server_running():
            print("Server is already running. Stopping it now...")
            manager.stop()
            print("Stopped server.")
            time.sleep(2)

        # Start server
        print("Starting server...")
        if manager.start(background=True, wait=True):
            print("Server started successfully")

            # Wait a bit
            print("Server is running. Waiting 5 seconds...")
            time.sleep(5)

            # Stop server
            print("Stopping server...")
            if manager.stop():
                print("Server stopped successfully")
            else:
                print("Failed to stop server")
        else:
            print("Failed to start server")

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
