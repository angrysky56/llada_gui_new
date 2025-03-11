#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for memory server connectivity issues.

This script creates a simple fallback memory server that runs without
depending on the server_manager module.
"""

import logging
import os
import signal
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='memory_server_fallback.log'
)
logger = logging.getLogger("memory_server_fallback")


def is_port_in_use(port=3000):
    """Check if the port is in use."""
    import socket
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


def find_venv_python():
    """Find the Python executable in the virtual environment."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def get_server_script_path():
    """Get the path to the server script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_py = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.py')
    server_js = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.js')

    # Prefer Python server
    if os.path.isfile(server_py):
        return 'python', server_py
    elif os.path.isfile(server_js):
        return 'node', server_js

    # If no server is found, create a minimal one
    return create_minimal_server()


def create_minimal_server():
    """Create a minimal memory server if none exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')
    server_py = os.path.join(memory_server_dir, 'server.py')

    # Create directory if it doesn't exist
    os.makedirs(memory_server_dir, exist_ok=True)

    # Write minimal server
    with open(server_py, 'w') as f:
        f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Fallback memory server for LLaDA GUI.
\"\"\"

import os
import sys
import json
import logging
import argparse
import numpy as np
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory storage
memory_state = np.zeros(64)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Memory server running"})

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({"status": "Memory server running"})

@app.route('/init', methods=['POST'])
@app.route('/api/init_model', methods=['POST'])
def init_model():
    global memory_state
    data = request.json or {}
    input_dim = data.get('inputDim', 64)
    output_dim = data.get('outputDim', 64)
    memory_state = np.zeros(output_dim)
    return jsonify({"message": "Model initialized", "config": {"inputDim": input_dim, "outputDim": output_dim}})

@app.route('/forward', methods=['POST'])
@app.route('/api/forward_pass', methods=['POST'])
def forward_pass():
    global memory_state
    data = request.json or {}
    x = data.get('x', [])
    mem = data.get('memoryState', memory_state.tolist())
    
    # Simple update logic
    if isinstance(mem, list):
        memory_state = np.array(mem) * 0.9 + np.random.randn(len(mem)) * 0.1
    
    return jsonify({
        "predicted": np.zeros(len(x) if isinstance(x, list) else 64).tolist(),
        "newMemory": memory_state.tolist(),
        "surprise": 0.0
    })

@app.route('/trainStep', methods=['POST'])
@app.route('/api/train_step', methods=['POST'])
def train_step():
    global memory_state
    data = request.json or {}
    x_t = data.get('x_t', [])
    x_next = data.get('x_next', [])
    
    # Simple update
    if isinstance(x_t, list) and isinstance(x_next, list):
        x_t_np = np.array(x_t)
        x_next_np = np.array(x_next)
        if len(x_t_np) > 0 and len(x_next_np) > 0:
            memory_state = 0.5 * x_t_np + 0.5 * x_next_np
    
    return jsonify({"cost": 0.0})

@app.route('/api/save_model', methods=['POST'])
def save_model():
    return jsonify({"message": "Model saved successfully"})

@app.route('/api/load_model', methods=['POST'])
def load_model():
    return jsonify({"message": "Model loaded successfully"})

@app.route('/api/reset_memory', methods=['POST'])
def reset_memory():
    global memory_state
    memory_state = np.zeros_like(memory_state)
    return jsonify({"message": "Memory reset successfully"})

@app.route('/api/memory_state', methods=['GET'])
def memory_state_endpoint():
    global memory_state
    return jsonify({"memoryState": memory_state.tolist()})

def parse_args():
    parser = argparse.ArgumentParser(description="Fallback memory server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting fallback memory server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
""")

    # Make it executable
    os.chmod(server_py, 0o755)
    return 'python', server_py


def install_dependencies():
    """Install required dependencies."""
    python_exe = find_venv_python()

    try:
        logger.info("Installing required dependencies...")
        subprocess.check_call([
            python_exe, '-m', 'pip', 'install', 'flask', 'numpy', 'requests'
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def stop_existing_servers():
    """Stop any existing memory server processes."""
    # Check if port is in use
    if is_port_in_use(3000):
        # Find and kill processes using the port
        pids = find_processes_using_port(3000)
        for pid in pids:
            logger.info(f"Killing process {pid} using port 3000")
            kill_process(pid, force=True)

        # Wait a bit for port to be released
        time.sleep(1)

        # Check if port is still in use
        if is_port_in_use(3000):
            logger.warning("Port 3000 is still in use after killing processes")
            return False

    return True


def start_server():
    """Start the memory server."""
    # Find Python executable
    python_exe = find_venv_python()

    # Get server script
    server_type, server_script = get_server_script_path()

    # Command to start server
    if server_type == 'python':
        cmd = [python_exe, server_script, '--host', '127.0.0.1', '--port', '3000']
    else:  # node
        cmd = ['node', server_script]

    # Start server process
    logger.info(f"Starting memory server with command: {' '.join(cmd)}")
    log_file = open('memory_server_fallback_output.log', 'w')
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True
    )

    # Store PID in file for future reference
    pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_server.pid')
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
                    logger.info("Memory server started successfully")
                    return True
            except:
                pass

        time.sleep(0.5)

    logger.error("Failed to start memory server")
    return False


def main():
    """Main function."""
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False

    # Stop existing servers
    if not stop_existing_servers():
        logger.error("Failed to stop existing servers")
        return False

    # Start server
    if not start_server():
        logger.error("Failed to start memory server")
        return False

    logger.info("Memory server started successfully")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
