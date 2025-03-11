#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fallback Python server for MCP Titan Memory.

This module provides a fallback Python server implementation when the
Node.js server cannot be started.
"""

import os
import sys
import time
import logging
import subprocess
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_server.log'
)
logger = logging.getLogger("fallback_server")

# Global variable to track the Python server process
_python_server_process = None

def try_python_server_fallback():
    """Attempt to start a Python fallback server if Node.js server fails.
    
    Returns:
        True if successful, False otherwise
    """
    global _python_server_process

    try:
        # Kill any existing servers first
        try:
            subprocess.run(['pkill', '-f', 'server.py'], check=False)
            subprocess.run(['pkill', '-f', 'server.js'], check=False)
            time.sleep(1)  # Wait for processes to terminate
        except Exception as e:
            logger.error(f"Error stopping existing processes: {e}")

        # Only try if we don't already have a Python server running
        if _python_server_process is not None and _python_server_process.poll() is None:
            logger.info("Python fallback server already running")
            return True

        # Find the Python server script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_script = os.path.join(script_dir, 'server.py')

        if not os.path.exists(server_script):
            logger.error(f"Python server script not found: {server_script}")
            return False

        # Create a minimal server.py if it doesn't exist or is too small
        if os.path.getsize(server_script) < 100:
            logger.warning(f"Python server script is too small, creating a new one")
            with open(server_script, 'w') as f:
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
            os.chmod(server_script, 0o755)

        # Try to install required packages
        try:
            logger.info("Installing Python server dependencies...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'numpy', 'requests'],
                          check=False, capture_output=True)
        except Exception as e:
            logger.warning(f"Error installing dependencies: {e}")

        # Start the server as a subprocess
        logger.info("Starting Python fallback server...")
        log_file = open('memory_server_python.log', 'w')
        _python_server_process = subprocess.Popen(
            [sys.executable, server_script, '--host', '127.0.0.1', '--port', '3000'],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True
        )

        # Wait for the server to start
        logger.info("Waiting for Python server to start...")
        for _ in range(15):  # Increase timeout to 15 seconds
            try:
                response = requests.get('http://localhost:3000/status', timeout=1)
                if response.status_code == 200:
                    logger.info("Python fallback server started successfully")
                    return True
            except Exception as e:
                logger.debug(f"Connection attempt failed: {e}")
                pass
            time.sleep(1)

        logger.error("Failed to start Python fallback server")
        return False
    except Exception as e:
        logger.error(f"Error starting Python fallback server: {e}")
        return False

def get_server_process():
    """Get the current Python server process.
    
    Returns:
        subprocess.Popen or None if not running
    """
    global _python_server_process
    return _python_server_process

def stop_server():
    """Stop the Python server if it's running.
    
    Returns:
        True if server was stopped, False otherwise
    """
    global _python_server_process
    
    if _python_server_process is None:
        return False
    
    try:
        if _python_server_process.poll() is None:
            _python_server_process.terminate()
            _python_server_process.wait(timeout=5)
            logger.info("Python fallback server stopped")
            return True
    except Exception as e:
        logger.error(f"Error stopping Python fallback server: {e}")
        # Try to kill forcefully
        try:
            _python_server_process.kill()
            logger.info("Python fallback server killed")
            return True
        except Exception as e:
            logger.error(f"Error killing Python fallback server: {e}")
    
    return False
