#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified memory server starter script.

This script provides a simple and reliable way to start the memory server
without depending on the server manager.
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='memory_server_simple.log'
)
logger = logging.getLogger("memory_server_simple")


def find_venv_python():
    """Find the Python executable in the virtual environment."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def is_port_in_use(port=3000):
    """Check if the port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_processes_using_port(port=3000):
    """Find processes using the specified port."""
    try:
        # Use lsof to find processes using the port
        output = subprocess.check_output(['lsof', '-i', f':{port}', '-t'], text=True)
        return [int(pid) for pid in output.strip().split('\n') if pid]
    except subprocess.SubprocessError:
        return []


def kill_process(pid, force=False):
    """Kill a process by PID."""
    try:
        if force:
            os.kill(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except:
        return False


def start_python_server():
    """Start the Python server."""
    # Find Python executable
    python_exe = find_venv_python()

    # Get server script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.py')

    # If server script doesn't exist, use the fallback fixer
    if not os.path.isfile(server_script):
        logger.info("Server script not found, using fallback script")
        # Use the fix script
        fallback_script = os.path.join(script_dir, 'fix_memory_server_fallback.py')
        if os.path.isfile(fallback_script):
            subprocess.run([python_exe, fallback_script], check=False)
            return True
        else:
            logger.error("Fallback script not found")
            return False

    # Start server process
    logger.info(f"Starting Python memory server: {server_script}")
    log_file = open('memory_server_python.log', 'w')
    process = subprocess.Popen(
        [python_exe, server_script, '--host', '127.0.0.1', '--port', '3000'],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True
    )

    # Store PID in file for future reference
    pid_file = os.path.join(script_dir, 'memory_server.pid')
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))

    # Wait for server to start
    logger.info("Waiting for server to start...")
    max_wait = 10  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if is_port_in_use(3000):
            # Try to connect to server
            try:
                import requests
                response = requests.get('http://localhost:3000/status', timeout=1)
                if response.status_code == 200:
                    logger.info("Python memory server started successfully")
                    return True
            except Exception as e:
                logger.debug(f"Connection test failed: {e}")

        time.sleep(0.5)

    logger.error("Failed to start Python memory server")
    return False


def start_node_server():
    """Start the Node.js server."""
    # Get server script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.js')

    # Check if the script exists
    if not os.path.isfile(server_script):
        logger.error(f"Node.js server script not found: {server_script}")
        return False

    # Check if Node.js is available
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Node.js not available")
        return False

    # Start server process
    logger.info(f"Starting Node.js memory server: {server_script}")
    log_file = open('memory_server_node.log', 'w')
    process = subprocess.Popen(
        ['node', server_script],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        cwd=os.path.dirname(server_script)
    )

    # Store PID in file for future reference
    pid_file = os.path.join(script_dir, 'memory_server.pid')
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))

    # Wait for server to start
    logger.info("Waiting for server to start...")
    max_wait = 10  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if is_port_in_use(3000):
            # Try to connect to server
            try:
                import requests
                response = requests.get('http://localhost:3000/status', timeout=1)
                if response.status_code == 200:
                    logger.info("Node.js memory server started successfully")
                    return True
            except Exception as e:
                logger.debug(f"Connection test failed: {e}")

        time.sleep(0.5)

    logger.error("Failed to start Node.js memory server")
    return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simplified Memory Server Starter")
    parser.add_argument("--node", action="store_true", help="Try Node.js server first")
    args = parser.parse_args()

    # Check if port is already in use
    if is_port_in_use(3000):
        logger.info("Port 3000 is already in use, checking if it's a memory server")
        try:
            import requests
            response = requests.get('http://localhost:3000/status', timeout=1)
            if response.status_code == 200:
                logger.info("Memory server is already running")
                return True
        except:
            logger.info("Port is in use but not by a memory server, killing processes")
            pids = find_processes_using_port(3000)
            for pid in pids:
                logger.info(f"Killing process {pid} using port 3000")
                kill_process(pid, force=True)

    # Try to start the server
    if args.node:
        # Try Node.js first, then Python
        if start_node_server():
            return True
        logger.info("Falling back to Python server")
        return start_python_server()
    else:
        # Try Python first, then Node.js
        if start_python_server():
            return True
        logger.info("Falling back to Node.js server")
        return start_node_server()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
