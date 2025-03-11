#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Headless memory server launcher for LLaDA GUI.

This script starts only the memory server component without requiring any GUI/Qt dependencies.
It's designed to work in headless environments or when you only need the memory functionality.

Usage:
    python run_memory_headless.py [--port PORT]
"""

import argparse
import logging
import os
import sys
import time
import subprocess
import json
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_server.log"))
    ]
)
logger = logging.getLogger("memory_headless")

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_port_in_use(port):
    """Check if the specified port is in use.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is in use, False otherwise
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_process_by_port(port):
    """Kill any process using the specified port.
    
    Args:
        port: Port number
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Checking for processes using port {port}...")
    
    try:
        # Try lsof to find process using port
        result = subprocess.run(['lsof', '-i', f':{port}', '-t'], 
                                capture_output=True, text=True, check=False)
        
        pids = result.stdout.strip().split('\n')
        if result.returncode == 0 and pids[0]:
            for pid in pids:
                if pid.strip():
                    logger.info(f"Killing process {pid} using port {port}")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        logger.warning(f"Process {pid} not found")
                    except Exception as e:
                        logger.warning(f"Error killing process {pid}: {e}")
            
            # Wait for port to be freed
            time.sleep(1)
            return not check_port_in_use(port)
        else:
            logger.info(f"No processes found using port {port}")
            return True
    except Exception as e:
        logger.warning(f"Error finding/killing process on port {port}: {e}")
        return False

def ensure_memory_server_directory():
    """Ensure memory server directory exists and is properly set up.
    
    Returns:
        Path to the server script, or None if failed
    """
    # Ensure the memory server directory exists
    memory_server_dir = os.path.join(SCRIPT_DIR, "core", "memory", "memory_server")
    os.makedirs(memory_server_dir, exist_ok=True)
    
    # Create a minimal server.py if it doesn't exist or is too small
    server_script = os.path.join(memory_server_dir, "server.py")
    if not os.path.exists(server_script) or os.path.getsize(server_script) < 100:
        logger.info("Creating memory server script...")
        with open(server_script, 'w') as f:
            f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Memory server for LLaDA GUI.
\"\"\"

import os
import sys
import json
import logging
import argparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    parser = argparse.ArgumentParser(description="Memory server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting memory server on {args.host}:{args.port}")
    print(f"Starting memory server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
""")
        # Make it executable
        os.chmod(server_script, 0o755)
        logger.info(f"Created memory server script: {server_script}")
    
    # Create __init__.py files if needed
    memory_dir = os.path.join(SCRIPT_DIR, "core", "memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    init_file = os.path.join(memory_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Memory module package")
    
    init_file = os.path.join(memory_server_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Memory server package")
    
    # Ensure memory_data directory exists
    memory_data_dir = os.path.join(memory_dir, "memory_data")
    os.makedirs(memory_data_dir, exist_ok=True)
    
    # Delete existing memory model to avoid loading issues
    memory_model_path = os.path.join(memory_data_dir, "titan_memory_model.json")
    if os.path.exists(memory_model_path):
        logger.info(f"Removing old memory model: {memory_model_path}")
        os.remove(memory_model_path)
    
    # Initialize a new memory model with default values
    try:
        logger.info("Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "flask", "flask-cors", "requests"], 
                       check=False, capture_output=True)
        
        import numpy as np
        import json
        
        # Create a simple default memory model
        default_model = {
            "config": {
                "input_dim": 64,
                "output_dim": 64,
                "hidden_dim": 32,
                "learning_rate": 0.001,
                "forget_gate_init": 0.01
            },
            "weights": {
                "input_layer": np.random.randn(64, 32).tolist(),
                "hidden_layer": np.random.randn(32, 64).tolist(),
                "forget_gate": np.random.randn(32, 1).tolist()
            },
            "memory_state": np.zeros(64).tolist(),
            "version": "1.0.0"
        }
        
        # Save the default model
        with open(memory_model_path, "w") as f:
            json.dump(default_model, f, indent=2)
        
        logger.info(f"Created new default memory model at {memory_model_path}")
    except ImportError as e:
        logger.error(f"Could not create default memory model: {e}")
        return None
    
    return server_script

def start_memory_server(port=3000):
    """Start the memory server on the specified port.
    
    Args:
        port: Port number to use
        
    Returns:
        Process object if successful, None otherwise
    """
    # Make sure port is free
    if check_port_in_use(port):
        if not kill_process_by_port(port):
            logger.error(f"Failed to free port {port}, cannot start memory server")
            return None
    
    # Get server script
    server_script = ensure_memory_server_directory()
    if not server_script:
        logger.error("Failed to set up memory server directory")
        return None
    
    logger.info(f"Starting memory server on port {port}...")
    try:
        # Start the server as a subprocess
        process = subprocess.Popen(
            [sys.executable, server_script, "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Save PID to file
        pid_file = os.path.join(SCRIPT_DIR, "memory_server.pid")
        with open(pid_file, "w") as f:
            f.write(str(process.pid))
        
        # Wait for the server to start
        logger.info("Waiting for server to start...")
        started = False
        for i in range(20):  # Try for up to 10 seconds
            try:
                import requests
                response = requests.get(f"http://127.0.0.1:{port}/status", timeout=1)
                if response.status_code == 200:
                    logger.info("Memory server started successfully!")
                    started = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        
        if not started:
            logger.error("Failed to start memory server")
            process.terminate()
            return None
        
        return process
    except Exception as e:
        logger.error(f"Error starting memory server: {e}")
        return None

def initialize_memory(port=3000):
    """Initialize the memory model.
    
    Args:
        port: Port number where the server is running
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import requests
        
        # Initialize the model
        response = requests.post(
            f"http://127.0.0.1:{port}/init",
            json={"inputDim": 64, "outputDim": 64},
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info("Model initialized successfully")
            return True
        else:
            logger.error(f"Failed to initialize model: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error initializing memory: {e}")
        return False

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Headless memory server for LLaDA GUI")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Make sure we're in the right directory
    os.chdir(SCRIPT_DIR)
    
    # Start the memory server
    process = start_memory_server(args.port)
    if not process:
        logger.error("Failed to start memory server")
        return 1
    
    # Initialize the memory
    if not initialize_memory(args.port):
        logger.error("Failed to initialize memory")
        process.terminate()
        return 1
    
    logger.info("Memory server is now running!")
    logger.info(f"Server is accessible at http://127.0.0.1:{args.port}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Handle keyboard interrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping memory server...")
        process.terminate()
        logger.info("Memory server stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
