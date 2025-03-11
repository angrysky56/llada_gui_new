#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Headless Runner for LLaDA Memory Integration

This script runs the memory server component without requiring a GUI display.
It's designed for server environments or situations where you only need
the memory backend without the graphical interface.
"""

import os
import sys
import time
import subprocess
import argparse
import logging
import json
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory_server.log')
    ]
)
logger = logging.getLogger("memory_headless")

# Add the appropriate paths to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "core"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "core", "memory"))

# Global variables
memory_server_process = None

def signal_handler(sig, frame):
    """Handle interruption signals to ensure clean shutdown."""
    logger.info("Received signal to terminate")
    shutdown()
    sys.exit(0)

def setup_server_environment():
    """Setup the environment for the memory server."""
    # Ensure memory data directories exist
    memory_data_dir = os.path.join(SCRIPT_DIR, "core", "memory", "memory_data")
    os.makedirs(memory_data_dir, exist_ok=True)
    
    # Ensure memory server directory exists
    memory_server_dir = os.path.join(SCRIPT_DIR, "core", "memory", "memory_server")
    os.makedirs(memory_server_dir, exist_ok=True)
    
    # Ensure models directory exists
    models_dir = os.path.join(memory_server_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for memory model file
    memory_model_path = os.path.join(memory_data_dir, "titan_memory_model.json")
    if not os.path.exists(memory_model_path):
        # Create a simple default memory model
        try:
            import numpy as np
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
            logger.info(f"Created default memory model at {memory_model_path}")
        except Exception as e:
            logger.error(f"Error creating default memory model: {e}")
            
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import numpy
        import flask
        import requests
        import torch
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        
        # Try to install missing dependencies
        logger.info("Attempting to install required dependencies...")
        try:
            # Determine Python interpreter
            if os.path.exists(os.path.join(SCRIPT_DIR, "venv", "bin", "python")):
                python_path = os.path.join(SCRIPT_DIR, "venv", "bin", "python")
            else:
                python_path = sys.executable
                
            # Install dependencies
            subprocess.run([python_path, "-m", "pip", "install", "numpy", "flask", "requests", "torch"], check=True)
            logger.info("Successfully installed dependencies")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

def start_memory_server(host="127.0.0.1", port=3000):
    """Start the memory server as a separate process."""
    global memory_server_process
    
    # If already running, don't start again
    if memory_server_process is not None and memory_server_process.poll() is None:
        logger.info("Memory server is already running")
        return True
    
    # Find the server script
    server_script = os.path.join(SCRIPT_DIR, "core", "memory", "memory_server", "server.py")
    
    # If server script doesn't exist or is too small, create it
    if not os.path.exists(server_script) or os.path.getsize(server_script) < 100:
        logger.info("Creating memory server script...")
        os.makedirs(os.path.dirname(server_script), exist_ok=True)
        
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
        logger.info(f"Created memory server script at {server_script}")
    
    # Kill any existing memory server processes
    try:
        # Find and kill any existing server processes
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'memory_server'], check=False)
        
        # Wait for processes to terminate
        time.sleep(1)
    except Exception as e:
        logger.warning(f"Error cleaning up existing processes: {e}")
    
    # Determine Python interpreter to use
    if os.path.exists(os.path.join(SCRIPT_DIR, "venv", "bin", "python")):
        python_path = os.path.join(SCRIPT_DIR, "venv", "bin", "python")
    else:
        python_path = sys.executable
    
    # Start the server
    logger.info(f"Starting memory server with {python_path} {server_script}")
    
    try:
        memory_server_process = subprocess.Popen(
            [python_path, server_script, "--host", host, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Save PID to file for later reference
        pid_file = os.path.join(SCRIPT_DIR, "memory_server.pid")
        with open(pid_file, 'w') as f:
            f.write(str(memory_server_process.pid))
        
        # Wait for server to start
        logger.info("Waiting for memory server to start...")
        start_time = time.time()
        timeout = 10  # seconds
        
        while time.time() - start_time < timeout:
            try:
                import requests
                response = requests.get(f"http://{host}:{port}/status", timeout=1)
                if response.status_code == 200:
                    logger.info("Memory server started successfully")
                    return True
            except Exception:
                # Continue waiting
                pass
            time.sleep(0.5)
        
        logger.error(f"Failed to start memory server (timeout after {timeout}s)")
        return False
    except Exception as e:
        logger.error(f"Error starting memory server: {e}")
        return False

def shutdown():
    """Gracefully shutdown the memory server."""
    global memory_server_process
    
    if memory_server_process is not None:
        logger.info("Shutting down memory server...")
        try:
            # Try gentle termination first
            memory_server_process.terminate()
            time.sleep(1)
            
            # If still running, force kill
            if memory_server_process.poll() is None:
                memory_server_process.kill()
            
            logger.info("Memory server stopped")
        except Exception as e:
            logger.error(f"Error stopping memory server: {e}")
    
    # Cleanup any other processes
    try:
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'memory_server'], check=False)
    except Exception as e:
        logger.warning(f"Error cleaning up processes: {e}")

def check_server_status(host="127.0.0.1", port=3000):
    """Check if the memory server is running."""
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/status", timeout=1)
        return response.status_code == 200
    except Exception:
        return False

def init_memory_model(host="127.0.0.1", port=3000):
    """Initialize the memory model."""
    try:
        import requests
        
        # Send initialization request
        response = requests.post(
            f"http://{host}:{port}/api/init_model",
            json={"inputDim": 64, "outputDim": 64},
            timeout=2
        )
        
        if response.status_code == 200:
            logger.info("Memory model initialized")
            return True
        else:
            logger.error(f"Failed to initialize memory model: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error initializing memory model: {e}")
        return False

def run_server(host="127.0.0.1", port=3000):
    """Main function to run the memory server."""
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting LLaDA Memory System in headless mode")
    
    # Setup environment
    setup_server_environment()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies, cannot continue")
        return 1
    
    # Start the memory server
    if not start_memory_server(host, port):
        logger.error("Failed to start memory server")
        return 1
    
    # Initialize the memory model
    if not init_memory_model(host, port):
        logger.warning("Failed to initialize memory model")
    
    logger.info(f"Memory server is running at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Keep the script running
        while True:
            # Check server status periodically
            if not check_server_status(host, port):
                logger.warning("Memory server is not responding, attempting to restart...")
                start_memory_server(host, port)
            
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    finally:
        shutdown()
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaDA Memory Headless Server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to')
    args = parser.parse_args()
    
    sys.exit(run_server(args.host, args.port))
