#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix memory database issues.

This script fixes issues with the memory database by ensuring the directories exist
and initializing the vector database.
"""

import json
import logging
import os
import subprocess
import sys
import time

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_db_fix")


def fix_vector_db():
    """Fix vector database issues."""
    # Ensure data directories exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_db_path = os.path.join(script_dir, "data", "memory", "vector_db")
    os.makedirs(vector_db_path, exist_ok=True)

    # Create config file if it doesn't exist
    config_path = os.path.join(script_dir, "core", "memory", "vector_db_config.json")
    if not os.path.exists(config_path):
        config = {
            "vector_db_path": vector_db_path,
            "dimension": 64,
            "use_vector_db": True,
            "similarity_threshold": 0.7,
            "max_vectors": 1000,
            "pruning_strategy": "lru"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created vector DB config: {config_path}")

    # Initialize empty vector database if it doesn't exist
    vectors_file = os.path.join(vector_db_path, "vectors.npy")
    metadata_file = os.path.join(vector_db_path, "metadata.json")

    if not os.path.exists(vectors_file):
        # Create empty vectors (0 x 64)
        empty_vectors = np.zeros((0, 64))
        np.save(vectors_file, empty_vectors)
        logger.info(f"Initialized empty vectors: {vectors_file}")

    if not os.path.exists(metadata_file):
        # Create empty metadata
        metadata = {
            "metadata": [],
            "usage_info": []
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Initialized empty metadata: {metadata_file}")

    # Create a special file to indicate the vector DB has been initialized
    initialized_file = os.path.join(vector_db_path, "DB_INITIALIZED")
    with open(initialized_file, 'w') as f:
        f.write(str(time.time()))
    logger.info(f"Created initialization marker: {initialized_file}")


def fix_memory_dependencies():
    """Install required dependencies."""
    try:
        logger.info("Installing required dependencies")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask', 'numpy', 'requests'])
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def kill_memory_processes():
    """Kill any existing memory server processes."""
    try:
        import subprocess
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'server.js'], check=False)
        subprocess.run(['pkill', '-f', 'memory_server'], check=False)
        import time
        time.sleep(1)  # Wait for processes to terminate
        logger.info("Killed existing memory server processes")
    except Exception as e:
        logger.error(f"Error killing processes: {e}")


def fix_memory_server():
    """Fix memory server and ensure it's correctly installed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_server_dir = os.path.join(script_dir, "core", "memory", "memory_server")
    server_py = os.path.join(memory_server_dir, "server.py")

    # Check if server.py exists and is executable
    if os.path.exists(server_py):
        os.chmod(server_py, 0o755)
        logger.info(f"Made server.py executable: {server_py}")

    # Ensure memory_server/__init__.py exists
    init_py = os.path.join(memory_server_dir, "__init__.py")
    if not os.path.exists(init_py):
        with open(init_py, 'w') as f:
            f.write("# Memory server initialization\n")
        logger.info(f"Created __init__.py: {init_py}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(memory_server_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Ensured models directory exists: {models_dir}")

    # Create a simple test model
    test_model_path = os.path.join(models_dir, "memory_model.json")
    if not os.path.exists(test_model_path):
        test_model = {
            "config": {
                "inputDim": 64,
                "outputDim": 64,
                "hiddenDim": 32,
                "learningRate": 0.001,
                "forgetGateInit": 0.01
            },
            "weights": {
                "forgetGate": 0.01
            },
            "memoryState": [0.0] * 64
        }
        with open(test_model_path, 'w') as f:
            json.dump(test_model, f, indent=4)
        logger.info(f"Created test memory model: {test_model_path}")


def start_server():
    """Start the memory server."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_py = os.path.join(script_dir, "core", "memory", "memory_server", "server.py")

        if os.path.exists(server_py):
            logger.info(f"Starting memory server: {server_py}")
            # Start server in background
            log_file = open("memory_server_python.log", "w")
            process = subprocess.Popen(
                [sys.executable, server_py],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True
            )
            # Store PID in a file
            pid_file = os.path.join(script_dir, "memory_server.pid")
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            # Wait for server to start
            time.sleep(2)
            logger.info("Memory server started")
            return True
        else:
            logger.error(f"Server script not found: {server_py}")
            return False
    except Exception as e:
        logger.error(f"Error starting memory server: {e}")
        return False


def check_server():
    """Check if memory server is running."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 3000)) == 0
    except:
        return False


def main():
    """Main function."""
    # Make sure we have the current directory in the path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Kill any existing memory processes
    kill_memory_processes()

    # Fix memory dependencies
    fix_memory_dependencies()

    # Fix vector database
    fix_vector_db()

    # Fix memory server
    fix_memory_server()

    # Start the server if not already running
    if not check_server():
        start_server()

    logger.info("Memory database fix completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
