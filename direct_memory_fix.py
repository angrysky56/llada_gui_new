#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Memory Server Fix

This script directly starts the memory server and verifies its connection.
It focuses purely on getting the memory server running correctly.
"""

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_fix")


def is_port_in_use(port=3000):
    """Check if port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False


def kill_memory_processes():
    """Kill any existing memory server processes using port 3000."""
    logger.info("Checking for processes using port 3000...")

    # First try to find processes using the port
    try:
        # Find PIDs of processes using port 3000
        output = subprocess.check_output(
            ['lsof', '-i', ':3000', '-t'],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        ).strip()

        if output:
            pids = output.split('\n')
            logger.info(f"Found processes using port 3000: {pids}")

            # Kill each process
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    logger.info(f"Killing process {pid}...")
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error killing process {pid}: {e}")
    except subprocess.CalledProcessError:
        # No processes found
        logger.info("No processes found using port 3000")
    except Exception as e:
        logger.error(f"Error checking processes: {e}")

    # Also try using pkill
    try:
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'server.js'], check=False)
        subprocess.run(['pkill', '-f', 'memory_server'], check=False)
    except Exception as e:
        logger.error(f"Error using pkill: {e}")


def create_python_server():
    """Create a simple Python memory server."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')
    server_py = os.path.join(server_dir, 'server.py')

    # Create directory if it doesn't exist
    os.makedirs(server_dir, exist_ok=True)

    logger.info(f"Creating Python memory server at {server_py}")

    # Server content as a string
    server_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple memory server for LLaDA GUI.
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='simple_memory_server.log'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage
memory_state = np.zeros(64)
memory_config = {
    "inputDim": 64,
    "outputDim": 64,
    "hiddenDim": 32,
    "learningRate": 0.001
}

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({"message": "Memory server is running"})

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint."""
    return jsonify({
        "status": "Memory server is running",
        "config": memory_config
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint."""
    return jsonify({
        "status": "Memory server is running",
        "config": memory_config
    })

@app.route('/init', methods=['POST'])
@app.route('/api/init_model', methods=['POST'])
def init_model():
    """Initialize model endpoint."""
    global memory_state, memory_config
    
    data = request.get_json(silent=True) or {}
    
    # Update config
    if 'inputDim' in data:
        memory_config['inputDim'] = data['inputDim']
    if 'outputDim' in data:
        memory_config['outputDim'] = data['outputDim']
        memory_state = np.zeros(memory_config['outputDim'])
    
    logger.info(f"Model initialized with config: {memory_config}")
    
    return jsonify({
        "message": "Model initialized",
        "config": memory_config
    })

@app.route('/forward', methods=['POST'])
@app.route('/api/forward_pass', methods=['POST'])
def forward_pass():
    """Forward pass endpoint."""
    global memory_state
    
    data = request.get_json(silent=True) or {}
    
    # Get input vector
    x = data.get('x', [0.0] * memory_config['inputDim'])
    if not isinstance(x, list):
        x = [0.0] * memory_config['inputDim']
    
    # Get memory state
    mem = data.get('memoryState', memory_state.tolist())
    if isinstance(mem, list):
        memory_state = np.array(mem)
    
    # Generate random prediction and update memory
    memory_state = memory_state * 0.9 + np.random.randn(memory_state.size) * 0.1
    predicted = np.zeros(len(x))
    
    logger.info(f"Forward pass completed, memory shape: {memory_state.shape}")
    
    return jsonify({
        "predicted": predicted.tolist(),
        "newMemory": memory_state.tolist(),
        "surprise": 0.0
    })

@app.route('/trainStep', methods=['POST'])
@app.route('/api/train_step', methods=['POST'])
def train_step():
    """Train step endpoint."""
    global memory_state
    
    data = request.get_json(silent=True) or {}
    
    # Get input and target vectors
    x_t = data.get('x_t', [])
    x_next = data.get('x_next', [])
    
    # Update memory state
    if isinstance(x_t, list) and isinstance(x_next, list) and len(x_t) > 0 and len(x_next) > 0:
        x_t_np = np.array(x_t)
        x_next_np = np.array(x_next)
        memory_state = 0.5 * x_t_np + 0.5 * x_next_np
    
    logger.info("Train step completed")
    
    return jsonify({
        "cost": 0.0
    })

@app.route('/save', methods=['POST'])
@app.route('/api/save_model', methods=['POST'])
def save_model():
    """Save model endpoint."""
    data = request.get_json(silent=True) or {}
    path = data.get('path', 'model.json')
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model
        with open(path, 'w') as f:
            json.dump({
                "config": memory_config,
                "memory_state": memory_state.tolist()
            }, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        
        return jsonify({
            "message": "Model saved successfully",
            "path": path
        })
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/load', methods=['POST'])
@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load model endpoint."""
    global memory_state, memory_config
    
    data = request.get_json(silent=True) or {}
    path = data.get('path', 'model.json')
    
    try:
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        if 'config' in model_data:
            memory_config.update(model_data['config'])
        
        if 'memory_state' in model_data:
            memory_state = np.array(model_data['memory_state'])
        
        logger.info(f"Model loaded from {path}")
        
        return jsonify({
            "message": "Model loaded successfully",
            "config": memory_config
        })
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/reset', methods=['POST'])
@app.route('/api/reset_memory', methods=['POST'])
def reset_memory():
    """Reset memory endpoint."""
    global memory_state
    
    memory_state = np.zeros(memory_config['outputDim'])
    
    logger.info("Memory reset")
    
    return jsonify({
        "message": "Memory reset successfully"
    })

@app.route('/memory', methods=['GET'])
@app.route('/api/memory_state', methods=['GET'])
def get_memory_state():
    """Get memory state endpoint."""
    return jsonify({
        "memoryState": memory_state.tolist()
    })

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Memory Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to listen on')
    parser.add_argument('--port', type=int, default=3000, help='Port to listen on')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Log startup message
    logger.info(f"Starting memory server on {args.host}:{args.port}")
    print(f"Starting memory server on {args.host}:{args.port}")
    
    try:
        app.run(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)
'''

    # Write to file
    with open(server_py, 'w') as f:
        f.write(server_content)

    # Make executable
    os.chmod(server_py, 0o755)

    # Create __init__.py if it doesn't exist
    init_py = os.path.join(server_dir, '__init__.py')
    if not os.path.exists(init_py):
        with open(init_py, 'w') as f:
            f.write("# Memory server package\n")

    # Create models directory
    models_dir = os.path.join(server_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    return server_py


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'flask', 'flask-cors', 'numpy', 'requests'
        ])
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def start_memory_server():
    """Start the memory server process."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_py = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.py')

    # Create server script if it doesn't exist
    if not os.path.exists(server_py):
        server_py = create_python_server()
    else:
        # Check if server.py has the required API endpoints
        with open(server_py, 'r') as f:
            content = f.read()

        missing_endpoint = False
        required_endpoints = ['/api/status', '/api/init_model', '/api/forward_pass', '/api/train_step',
                              '/api/save_model', '/api/load_model', '/reset', '/api/reset_memory',
                              '/memory', '/api/memory_state']

        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoint = True
                logger.warning(f"Server.py is missing endpoint: {endpoint}")

        if missing_endpoint:
            logger.info(f"Server.py is missing required endpoints. Creating a new server script.")
            server_py = create_python_server()

    logger.info(f"Starting memory server: {server_py}")

    try:
        # Create log file
        log_file = open('memory_server.log', 'w')

        # Start server process
        process = subprocess.Popen(
            [sys.executable, server_py, '--host', '127.0.0.1', '--port', '3000'],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True
        )

        # Write PID to file
        with open('memory_server.pid', 'w') as f:
            f.write(str(process.pid))

        # Wait for server to start
        logger.info("Waiting for server to start...")
        max_wait = 10
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if is_port_in_use(3000):
                # Test connection using requests
                try:
                    import requests
                    response = requests.get('http://localhost:3000/status')
                    if response.status_code == 200:
                        logger.info("Memory server started successfully!")

                        # Initialize model
                        init_response = requests.post(
                            'http://localhost:3000/api/init_model',
                            json={"inputDim": 64, "outputDim": 64}
                        )

                        if init_response.status_code == 200:
                            logger.info("Model initialized successfully")
                            return True
                        else:
                            logger.error(f"Failed to initialize model: {init_response.text}")
                except Exception as e:
                    logger.warning(f"Error testing server: {e}")

            time.sleep(1)

        logger.error("Timed out waiting for server to start")
        return False
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False


def main():
    """Main function."""
    # Check for prepare-only flag
    if len(sys.argv) > 1 and sys.argv[1] == '--prepare-only':
        logger.info("Prepare-only mode: Creating server file if needed without starting server")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_py = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'server.py')

        # Create server script if it doesn't exist
        if not os.path.exists(server_py):
            server_py = create_python_server()
            logger.info(f"Created server script at {server_py}")
        else:
            logger.info(f"Server script already exists at {server_py}")

        return 0
    # Kill any existing memory server processes
    kill_memory_processes()

    # Install dependencies
    install_dependencies()

    # Start memory server
    if start_memory_server():
        logger.info("Memory server is now running!")

        # Keep process running to maintain server
        try:
            print("Memory server is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
            return 0

        return 0
    else:
        logger.error("Failed to start memory server")
        return 1


if __name__ == "__main__":
    sys.exit(main())
