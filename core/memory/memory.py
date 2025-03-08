#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory system utility script for LLaDA GUI.

This script provides command-line utilities for controlling the memory system.
"""

import argparse
import sys

from memory_server.server_manager import MemoryServerManager


def main():
    """Main function for the memory utility script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="LLaDA GUI Memory System Utility")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 'start' command
    start_parser = subparsers.add_parser("start", help="Start the memory server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    start_parser.add_argument("--port", type=int, default=3000, help="Port number to use")

    # 'stop' command
    stop_parser = subparsers.add_parser("stop", help="Stop the memory server")

    # 'status' command
    status_parser = subparsers.add_parser("status", help="Check memory server status")

    # 'test' command
    test_parser = subparsers.add_parser("test", help="Run a test of the memory system")
    test_parser.add_argument("--skip-server", action="store_true", help="Skip server start/stop during test")

    # Parse arguments
    args = parser.parse_args()

    # Handle command
    if args.command == "start":
        # Create and start the server manager
        manager = MemoryServerManager(args.host, args.port)
        if manager.is_running():
            print(f"Memory server is already running on {args.host}:{args.port}")
            return 0

        if manager.start():
            print(f"Memory server started on {args.host}:{args.port}")
            # Keep running until interrupted
            try:
                print("Press Ctrl+C to stop the server...")
                while manager.is_running():
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server due to keyboard interrupt...")
                manager.stop()
                print("Memory server stopped")
            return 0
        else:
            print("Failed to start memory server")
            return 1

    elif args.command == "stop":
        # Create manager and stop the server
        manager = MemoryServerManager()
        if not manager.is_running():
            print("Memory server is not running")
            return 0

        if manager.stop():
            print("Memory server stopped")
            return 0
        else:
            print("Failed to stop memory server")
            return 1

    elif args.command == "status":
        # Check status
        manager = MemoryServerManager()
        running, status = manager.get_status()
        if running:
            print(f"Memory server is running: {status}")
            return 0
        else:
            print("Memory server is not running")
            return 1

    elif args.command == "test":
        # Import test module
        try:
            from memory_server.test_server import test_memory_server
            success = test_memory_server()
            if success:
                return 0
            else:
                return 1
        except ImportError:
            print("Error: Test module not found")
            return 1

    else:
        # No command specified, show help
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
