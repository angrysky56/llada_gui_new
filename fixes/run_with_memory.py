#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for the LLaDA GUI application with memory integration.
This script ensures the memory system is properly initialized and connected.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llada_memory.log")
    ]
)
logger = logging.getLogger("llada_memory")

# Make sure we're running from the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
logger.info(f"Working directory: {script_dir}")

# Make sure we have the correct Python environment
if os.path.exists('./venv/bin/python'):
    python_path = os.path.abspath('./venv/bin/python')
    if sys.executable != python_path:
        logger.info(f"Restarting with the virtual environment Python: {python_path}")
        os.execl(python_path, python_path, *sys.argv)

# Add the appropriate paths to Python path
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "core"))
sys.path.insert(0, os.path.join(script_dir, "gui"))
sys.path.insert(0, os.path.join(script_dir, "optimizations"))


# Check if model exists and download if needed
def check_model():
    """Check if the model exists and download if needed."""
    model_dir = os.path.join(script_dir, "GSAI-ML_LLaDA-8B-Instruct")
    if not os.path.exists(model_dir):
        try:
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)

            # Check if original model exists to copy
            original_model = os.path.join(os.path.dirname(script_dir), "llada_gui/GSAI-ML_LLaDA-8B-Instruct")
            if os.path.exists(original_model):
                logger.info(f"Copying model from original directory: {original_model}")
                import shutil
                shutil.copytree(original_model, model_dir, dirs_exist_ok=True)
                return True

            # If model doesn't exist, download from Hugging Face
            logger.info("Model not found. Downloading from Hugging Face...")

            # Offer to download model
            try:
                from PyQt6.QtWidgets import QApplication, QMessageBox
                app = QApplication([])
                response = QMessageBox.question(
                    None,
                    "Download Model",
                    "The LLaDA model is not found. Would you like to download it from Hugging Face? \n\n" +
                    "This will download the GSAI-ML/LLaDA-8B-Instruct model (about 16GB).\n\n" +
                    "Note: You need a Hugging Face account and must accept the model license.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if response == QMessageBox.StandardButton.Yes:
                    # Use huggingface_hub to download the model
                    try:
                        from huggingface_hub import snapshot_download
                        logger.info("Downloading model from Hugging Face...")
                        QMessageBox.information(
                            None,
                            "Downloading Model",
                            "The model will now be downloaded. This may take some time depending on your internet connection.\n\n" +
                            "The download will continue in the background. The application will start once the download is complete."
                        )

                        # Download the model
                        snapshot_download(
                            repo_id="GSAI-ML/LLaDA-8B-Instruct",
                            local_dir=model_dir,
                            token=None  # User will be prompted for login if needed
                        )

                        logger.info("Model downloaded successfully")
                        return True
                    except Exception as e:
                        logger.error(f"Error downloading model: {e}")
                        QMessageBox.critical(
                            None,
                            "Download Error",
                            f"Failed to download the model: {str(e)}\n\nPlease download it manually from https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct"
                        )
                        return False
                else:
                    logger.warning("User declined to download the model")
                    return False
            except Exception as e:
                logger.error(f"Error showing download dialog: {e}")
                return False
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return False
    return True


# Ensure data directory exists for vector database
def setup_vector_db():
    """Set up vector database for persistent memory."""
    try:
        # Create data directory
        data_dir = os.path.join(script_dir, "data", "memory", "vector_db")
        os.makedirs(data_dir, exist_ok=True)

        # Create a config file for vector database
        config_file = os.path.join(script_dir, "core", "memory", "vector_db_config.json")
        if not os.path.exists(config_file):
            config = {
                "vector_db_path": data_dir,
                "dimension": 64,
                "use_vector_db": True,
                "similarity_threshold": 0.7
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return False


# Function to start the memory server
def start_memory_server():
    """Start the memory server and return the process."""
    try:
        # Path to memory server
        memory_server_dir = os.path.join(script_dir, "core", "memory", "memory_server")
        if not os.path.exists(memory_server_dir):
            logger.error(f"Memory server directory not found: {memory_server_dir}")
            return None

        # Check if server is already running - try both status endpoints
        server_running = False
        try:
            import requests
            for endpoint in ["/status", "/api/status"]:
                try:
                    response = requests.get(f"http://localhost:3000{endpoint}", timeout=1)
                    if response.status_code == 200:
                        logger.info(f"Memory server is already running (detected via {endpoint})")
                        server_running = True
                        break
                except Exception as e:
                    logger.debug(f"Endpoint {endpoint} check failed: {e}")
            if server_running:
                return None
        except Exception as e:
            logger.debug(f"Server connectivity check failed: {e}")

        # If we get here, no server is running or responding - try to start one

        # First check if we actually need to install dependencies
        try:
            # Install or update dependencies via the fix script
            fix_script = os.path.join(script_dir, "fix_memory_dependencies.py")
            if os.path.exists(fix_script):
                logger.info("Running dependency fix script...")
                subprocess.run([sys.executable, fix_script], check=False)
        except Exception as e:
            logger.warning(f"Error running dependency fix script: {e}")

        # Try to start Node.js server first
        try:
            logger.info("Starting Node.js memory server...")
            node_server = os.path.join(memory_server_dir, "server.js")
            if os.path.exists(node_server):
                # First make sure we have the required node modules
                try:
                    if not os.path.exists(os.path.join(memory_server_dir, "node_modules", "express")):
                        logger.info("Installing Node.js dependencies...")
                        # Create log file to capture npm output
                        npm_log = open(os.path.join(script_dir, "npm_install.log"), "w")
                        subprocess.run(
                            ["npm", "install"],
                            cwd=memory_server_dir,
                            stdout=npm_log,
                            stderr=npm_log,
                            check=False
                        )
                        npm_log.close()
                except Exception as e:
                    logger.warning(f"Error installing Node.js dependencies: {e}")

                # Create log file for server output
                server_log = open(os.path.join(script_dir, "memory_server.log"), "w")

                # Start Node.js server
                server_process = subprocess.Popen(
                    ["node", "server.js"],
                    cwd=memory_server_dir,
                    stdout=server_log,
                    stderr=server_log
                )

                # Wait a bit for server to start
                time.sleep(2)

                # Check if server started successfully
                if server_process.poll() is None:
                    # Also check if it's responding to requests
                    try:
                        import requests
                        for endpoint in ["/status", "/api/status"]:
                            try:
                                response = requests.get(f"http://localhost:3000{endpoint}", timeout=1)
                                if response.status_code == 200:
                                    logger.info(f"Node.js memory server started and responding via {endpoint}")
                                    return server_process
                            except:
                                pass

                        # If we get here, server is running but not responding
                        logger.warning("Node.js server process is running but not responding to requests")
                        # Try to kill it so we can try Python server
                        server_process.terminate()
                        server_process.wait(timeout=2)
                    except Exception as e:
                        logger.warning(f"Error checking Node.js server response: {e}")
                else:
                    logger.warning("Node.js server process terminated immediately")
                    try:
                        server_log.close()
                        with open(os.path.join(script_dir, "memory_server.log"), "r") as f:
                            log_contents = f.read()
                            logger.warning(f"Node.js server log: {log_contents[-1000:]}")
                    except Exception as e:
                        logger.warning(f"Could not read server log: {e}")
            else:
                logger.warning(f"Node.js server not found: {node_server}")
        except Exception as e:
            logger.error(f"Error starting Node.js server: {e}")

        # Fall back to Python server
        try:
            logger.info("Starting Python memory server...")
            python_server = os.path.join(memory_server_dir, "server.py")

            # Check if the Python server exists, if not create a minimal version
            if not os.path.exists(python_server) or os.path.getsize(python_server) < 100:
                logger.warning("Python server not found or too small, creating a minimal version")
                with open(python_server, "w") as f:
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
                os.chmod(python_server, 0o755)

            # Ensure requirements are installed
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "flask", "numpy", "requests"],
                    check=False
                )
            except Exception as e:
                logger.warning(f"Error installing Python server dependencies: {e}")

            # Create log file for server output
            server_log = open(os.path.join(script_dir, "memory_server_python.log"), "w")

            # Start Python server
            server_process = subprocess.Popen(
                [sys.executable, "server.py", "--host", "127.0.0.1", "--port", "3000"],
                cwd=memory_server_dir,
                stdout=server_log,
                stderr=server_log
            )

            # Wait a bit for server to start
            time.sleep(2)

            # Check if server started successfully
            if server_process.poll() is None:
                # Also check if it's responding to requests
                try:
                    import requests
                    for endpoint in ["/status", "/api/status"]:
                        try:
                            response = requests.get(f"http://localhost:3000{endpoint}", timeout=1)
                            if response.status_code == 200:
                                logger.info(f"Python memory server started and responding via {endpoint}")
                                return server_process
                        except:
                            pass

                    # If we get here, server is running but not responding
                    logger.warning("Python server process is running but not responding to requests")
                except Exception as e:
                    logger.warning(f"Error checking Python server response: {e}")
            else:
                logger.warning("Python server process terminated immediately")
                try:
                    server_log.close()
                    with open(os.path.join(script_dir, "memory_server_python.log"), "r") as f:
                        log_contents = f.read()
                        logger.warning(f"Python server log: {log_contents[-1000:]}")
                except Exception as e:
                    logger.warning(f"Could not read server log: {e}")
        except Exception as e:
            logger.error(f"Error starting Python server: {e}")

        # All attempts failed
        logger.error("Failed to start memory server")
        return None

    except Exception as e:
        logger.error(f"Error in start_memory_server: {e}")
        return None


# Function to stop the memory server
def stop_memory_server(server_process):
    """Stop the memory server."""
    if server_process is None:
        return

    try:
        logger.info("Stopping memory server...")
        server_process.terminate()
        server_process.wait(timeout=5)
        logger.info("Memory server stopped")
    except Exception as e:
        logger.error(f"Error stopping memory server: {e}")
        try:
            server_process.kill()
        except:
            pass


# Check model first
if not check_model():
    # Display error message
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication([])
        error_msg = "LLaDA model not found. Please download the model before running the application."
        QMessageBox.critical(None, "Model Error", error_msg)
    except:
        print("LLaDA model not found. Please download the model before running the application.")
    sys.exit(1)


# Ensure data directory exists for vector database
def setup_vector_db():
    """Set up vector database for persistent memory."""
    try:
        # Create data directory
        data_dir = os.path.join(script_dir, "data", "memory", "vector_db")
        os.makedirs(data_dir, exist_ok=True)

        # Create a config file for vector database
        config_file = os.path.join(script_dir, "core", "memory", "vector_db_config.json")
        if not os.path.exists(config_file):
            config = {
                "vector_db_path": data_dir,
                "dimension": 64,
                "use_vector_db": True,
                "similarity_threshold": 0.7
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

        return True
    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return False


# Setup vector DB
setup_vector_db()

# Start the memory server
server_process = start_memory_server()

# Register cleanup function to stop server on exit
if server_process is not None:
    atexit.register(lambda: stop_memory_server(server_process))


    # Also handle SIGINT and SIGTERM
    def signal_handler(sig, frame):
        stop_memory_server(server_process)
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Import and run the main application with memory
try:
    # Run with memory flag
    os.environ["LLADA_MEMORY_ENABLED"] = "1"


    # Create custom run function to position window properly
    def custom_run():
        # Import UI components
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtCore import QRect

        # Import from run.py
        from gui.llada_gui import LLaDAGUI
        from core.memory.memory_ui import enhance_llada_gui

        # Initialize app
        app = QApplication(sys.argv)

        # Create enhanced GUI with memory and let it position itself naturally
        EnhancedLLaDAGUI = enhance_llada_gui(LLaDAGUI)
        window = EnhancedLLaDAGUI()

        # Ensure memory is enabled
        if hasattr(window, 'memory_integration'):
            window.memory_integration.setChecked(True)

        # Register shutdown handler to properly terminate server
        def shutdown_handler():
            logger.info("Application shutting down, stopping memory server...")
            if server_process is not None:
                try:
                    # Try to kill server process directly
                    logger.info(f"Killing server process (PID: {server_process.pid})")
                    # First try SIGTERM
                    os.kill(server_process.pid, signal.SIGTERM)
                    time.sleep(1)  # Give it a moment to shut down

                    # If still running, use SIGKILL
                    if server_process.poll() is None:
                        os.kill(server_process.pid, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error killing server process: {e}")

            # Also try to use the server manager if available
            try:
                from core.memory.memory_integration import get_server_manager
                server_manager = get_server_manager()
                if server_manager and server_manager.is_running():
                    logger.info("Stopping server via server manager")
                    server_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping server via manager: {e}")

            # Last resort: kill any Node.js server process
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] == 'node' and any(
                                'server.js' in arg for arg in (proc.info['cmdline'] or [])):
                            logger.info(f"Killing Node.js server process: {proc.info['pid']}")
                            os.kill(proc.info['pid'], signal.SIGKILL)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception as e:
                logger.error(f"Error finding and killing Node.js processes: {e}")

        # Register shutdown handler
        app.aboutToQuit.connect(shutdown_handler)

        # Show the window
        window.show()

        # Run the app
        return app.exec()


    # Run our custom function
    sys.exit(custom_run())
except Exception as e:
    logger.error(f"Error running LLaDA GUI: {e}")
    logger.error(traceback.format_exc())

    # Display error message
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication([])
        error_msg = f"Error starting LLaDA GUI with memory: {str(e)}\n\n{traceback.format_exc()}"
        QMessageBox.critical(None, "LLaDA GUI Error", error_msg)
    except:
        print(f"Error starting LLaDA GUI with memory: {e}")

    # Stop memory server before exiting
    if server_process is not None:
        stop_memory_server(server_process)

    sys.exit(1)
