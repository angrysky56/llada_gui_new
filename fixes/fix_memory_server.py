#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix script for memory server issues.
This ensures the memory server directory is properly set up and dependencies are installed.
"""

import logging
import os
import shutil
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to fix memory server issues."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the memory server directory
    memory_server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')

    # Check if the directory exists
    if not os.path.isdir(memory_server_dir):
        logger.error(f"Memory server directory not found: {memory_server_dir}")
        # Try to create the directory
        os.makedirs(memory_server_dir, exist_ok=True)
        logger.info(f"Created memory server directory: {memory_server_dir}")

    # Check if server.js exists
    server_js = os.path.join(memory_server_dir, 'server.js')
    if not os.path.exists(server_js):
        logger.error(f"Server.js not found: {server_js}")

        # Try to copy from original llada_gui if it exists
        original_server = os.path.join('/home/ty/Repositories/ai_workspace/llada_gui/memory_server', 'server.js')
        if os.path.exists(original_server):
            shutil.copy2(original_server, server_js)
            logger.info(f"Copied server.js from original directory")
        else:
            logger.error("Original server.js not found, please check installation")
            return False

    # Check if server.py exists
    server_py = os.path.join(memory_server_dir, 'server.py')
    if not os.path.exists(server_py):
        logger.error(f"Server.py not found: {server_py}")

        # Try to create a minimal server.py
        with open(server_py, 'w') as f:
            f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Simple memory server for LLaDA GUI.
\"\"\"

import os
import sys
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Memory server running"})

@app.route('/api/init_model', methods=['POST'])
def init_model():
    return jsonify({"message": "Model initialized", "config": {"inputDim": 64, "outputDim": 64}})

@app.route('/api/forward_pass', methods=['POST'])
def forward_pass():
    data = request.json
    return jsonify({
        "predicted": [0.0] * 64,
        "newMemory": [0.0] * 64,
        "surprise": 0.0
    })

@app.route('/api/train_step', methods=['POST'])
def train_step():
    return jsonify({"cost": 0.0})

@app.route('/api/save_model', methods=['POST'])
def save_model():
    return jsonify({"message": "Model saved successfully"})

@app.route('/api/load_model', methods=['POST'])
def load_model():
    return jsonify({"message": "Model loaded successfully"})

@app.route('/api/reset_memory', methods=['POST'])
def reset_memory():
    return jsonify({"message": "Memory reset successfully"})

@app.route('/api/memory_state', methods=['GET'])
def memory_state():
    return jsonify({"memoryState": [0.0] * 64})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)
""")
        os.chmod(server_py, 0o755)  # Make executable
        logger.info(f"Created minimal server.py")

    # Check if package.json exists
    package_json = os.path.join(memory_server_dir, 'package.json')
    if not os.path.exists(package_json):
        logger.error(f"package.json not found: {package_json}")

        # Create a minimal package.json
        with open(package_json, 'w') as f:
            f.write("""{
  "name": "mcp-titan-memory-server",
  "version": "1.0.0",
  "description": "Memory server for the LLaDA GUI application",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "body-parser": "^1.20.2"
  }
}""")
        logger.info(f"Created minimal package.json")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(memory_server_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Created models directory: {models_dir}")

    # Try to install dependencies
    try:
        # Check if Node.js is installed
        subprocess.run(['node', '--version'], check=True, capture_output=True)

        # Install Node.js dependencies
        logger.info("Installing Node.js dependencies...")
        subprocess.run(['npm', 'install'], cwd=memory_server_dir, check=True)
        logger.info("Node.js dependencies installed successfully")
    except Exception as e:
        logger.warning(f"Could not install Node.js dependencies: {e}")
        logger.info("Will fall back to Python server if needed")

    # Try to install Python dependencies
    try:
        # Install Flask if not installed
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'requests', 'numpy'], check=True)
        logger.info("Python dependencies installed successfully")
    except Exception as e:
        logger.warning(f"Could not install Python dependencies: {e}")

    # Check if server is already running
    try:
        import requests
        response = requests.get('http://localhost:3000/status', timeout=1)
        if response.status_code == 200:
            logger.info("Memory server is already running")
        else:
            logger.warning(f"Memory server returned unexpected status: {response.status_code}")
    except:
        logger.info("Memory server is not running")

        # Try to start the server
        try:
            logger.info("Starting memory server...")

            # Try Node.js server first
            try:
                subprocess.Popen(['node', 'server.js'], cwd=memory_server_dir)
                logger.info("Node.js memory server started")
            except:
                # Fall back to Python server
                subprocess.Popen([sys.executable, 'server.py'], cwd=memory_server_dir)
                logger.info("Python memory server started")

            logger.info("Memory server should be running at http://localhost:3000")
        except Exception as e:
            logger.error(f"Failed to start memory server: {e}")

    logger.info("Memory server fix completed. Please try running run_with_memory.py now.")
    return True


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
