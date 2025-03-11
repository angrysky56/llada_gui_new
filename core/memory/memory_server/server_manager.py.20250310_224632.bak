#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Server manager for the LLaDA GUI memory server.

This provides utilities for starting, stopping, and checking the status
of the memory server in a reliable way.
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_server_manager")

# Default configuration
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 3000
SERVER_CHECK_TIMEOUT = 0.5  # Seconds
START_TIMEOUT = 10  # Seconds
STOP_TIMEOUT = 5  # Seconds


class MemoryServerManager:
    """Manages the Titan Memory Server."""

    def __init__(self, host=None, port=None, server_dir=None):
        """Initialize the server manager.
        
        Args:
            host: Host interface for server (default: 127.0.0.1)
            port: Port number for server (default: 3000)
            server_dir: Directory containing server files (default: same as this script)
        """
        self.host = host or DEFAULT_HOST
        self.port = port or DEFAULT_PORT

        # Determine server directory
        if server_dir is None:
            self.server_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.server_dir = os.path.abspath(server_dir)

        # Track server process
        self.process = None
        self.pid_file = os.path.join(self.server_dir, f"memory_server_pid_{self.port}.pid")
        self.log_file = os.path.join(self.server_dir, f"memory_server_{self.port}.log")

    def is_port_in_use(self):
        """Check if the port is in use.
        
        Returns:
            True if port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(SERVER_CHECK_TIMEOUT)
            try:
                s.connect((self.host, self.port))
                return True
            except:
                return False

    def is_server_running(self):
        """Check if the server is running.
        
        This checks both the port and verifies that the server responds properly.
        
        Returns:
            True if server is running, False otherwise
        """
        # Check if port is in use
        if not self.is_port_in_use():
            return False

        # Try to connect to server - first check status endpoint
        try:
            import requests
            # First try /status endpoint
            response = requests.get(
                f"http://{self.host}:{self.port}/status",
                timeout=SERVER_CHECK_TIMEOUT
            )
            # Check if response is valid JSON
            response.json()
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Error checking status endpoint: {e}")
            # Try API status endpoint as fallback
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/api/status",
                    timeout=SERVER_CHECK_TIMEOUT
                )
                response.json()  # Check if valid JSON
                return response.status_code == 200
            except Exception as e:
                logger.debug(f"Error checking API status endpoint: {e}")
                return False

    def get_stored_pid(self):
        """Get the stored PID from file.
        
        Returns:
            PID as integer, or None if not found
        """
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return None

    def store_pid(self, pid):
        """Store PID in file.
        
        Args:
            pid: Process ID to store
        """
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
        except Exception as e:
            logger.warning(f"Failed to store PID: {e}")

    @staticmethod
    def is_process_running(pid):
        """Check if a process with given PID is running.
        
        Args:
            pid: Process ID to check
            
        Returns:
            True if process is running, False otherwise
        """
        if pid is None:
            return False

        try:
            # Send signal 0 to check process existence
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except:
            return False

    def kill_process(self, pid, force=False):
        """Attempt to kill a process with given PID.
        
        Args:
            pid: Process ID to kill
            force: Use SIGKILL immediately if True
            
        Returns:
            True if process was killed, False otherwise
        """
        if pid is None:
            return False

        try:
            if not force:
                # Try to terminate gracefully first
                os.kill(pid, signal.SIGTERM)

                # Wait for process to terminate
                for _ in range(STOP_TIMEOUT):
                    if not self.is_process_running(pid):
                        return True
                    time.sleep(1)

            # Force kill if still running or force=True
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)  # Small wait to ensure kill is processed
            return not self.is_process_running(pid)
        except:
            return False

    def find_memory_server_pid(self):
        """Find the PID of the memory server process using lsof.
        
        Returns:
            PID as integer, or None if not found
        """
        try:
            # Try to find process listening on the specified port
            result = subprocess.run(
                ['lsof', '-i', f':{self.port}', '-t'],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return None

    def find_python_path(self):
        """Find the Python executable path.
        
        Returns:
            Path to Python executable
        """
        # Use current Python by default
        python_path = sys.executable

        # Try to find venv Python
        venv_dirs = [
            os.path.join(self.server_dir, '..', '..', '..', 'venv'),
            os.path.join(self.server_dir, '..', '..', 'venv'),
            os.path.join(self.server_dir, '..', 'venv'),
            os.path.join(self.server_dir, 'venv')
        ]

        for venv_dir in venv_dirs:
            venv_dir = os.path.abspath(venv_dir)
            venv_python = os.path.join(venv_dir, 'bin', 'python')
            if os.path.exists(venv_python):
                python_path = venv_python
                break

        return python_path

    def find_server_script(self):
        """Find the server script path.
        
        Returns:
            Path to server script
        """
        # Try Node.js server first
        node_server = os.path.join(self.server_dir, 'server.js')
        if os.path.exists(node_server):
            return node_server

        # Try Python server as fallback
        python_server = os.path.join(self.server_dir, 'server.py')
        if os.path.exists(python_server):
            return python_server

        # No server found
        return None

    def start(self, background=True, wait=True):
        """Start the memory server.
        
        Args:
            background: Run server in background
            wait: Wait for server to start
            
        Returns:
            True if server started successfully, False otherwise
        """
        # Check if server is already running
        if self.is_server_running():
            logger.info("Server is already running")
            return True

        # Check if port is in use by something else
        if self.is_port_in_use():
            logger.warning(f"Port {self.port} is in use but server is not responding properly")

            # Try to kill any process using the port
            pid = self.find_memory_server_pid()
            if pid:
                logger.info(f"Found process {pid} using port {self.port}, attempting to kill it")
                if self.kill_process(pid, force=True):
                    logger.info(f"Successfully killed process {pid}")
                    # Small wait to ensure port is released
                    time.sleep(1)
                else:
                    logger.error(f"Failed to kill process {pid}")
                    return False

            # Check if port is still in use
            if self.is_port_in_use():
                logger.error(f"Port {self.port} is still in use after cleanup attempt")
                return False

        # Check for required dependencies
        self._check_dependencies()

        # Try to find appropriate server script - try both Node.js and Python options
        server_scripts = []
        js_server = os.path.join(self.server_dir, 'server.js')
        py_server = os.path.join(self.server_dir, 'server.py')

        # Check Node.js server first if available
        if os.path.exists(js_server):
            try:
                # Check if Node.js is available
                subprocess.run(['node', '--version'], capture_output=True, check=True)
                server_scripts.append(('node', js_server))
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("Node.js not available, will skip server.js")

        # Check Python server as fallback or alternative
        if os.path.exists(py_server):
            python_path = self.find_python_path()
            server_scripts.append((python_path, py_server))

        if not server_scripts:
            logger.error("No valid server scripts found")
            return False

        # Try to start the servers in order, with appropriate error handling
        success = False
        for interpreter, script in server_scripts:
            cmd = [interpreter, script]

            # Add host and port if supported
            try:
                help_output = subprocess.run([cmd[0], script, '--help'],
                                             capture_output=True, text=True).stdout
                if '--host' in help_output or '--port' in help_output:
                    cmd.extend(['--host', self.host, '--port', str(self.port)])
            except Exception as e:
                logger.warning(f"Error checking command options: {e}")

            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            os.makedirs(log_dir, exist_ok=True)

            # Start server process
            try:
                if background:
                    # Open log file
                    log_file = open(self.log_file, 'w')

                    # Start process
                    logger.info(f"Starting server: {' '.join(cmd)}")
                    self.process = subprocess.Popen(
                        cmd,
                        cwd=self.server_dir,
                        stdout=log_file,
                        stderr=log_file,
                        start_new_session=True
                    )
                else:
                    # Start process in foreground
                    logger.info(f"Starting server: {' '.join(cmd)}")
                    self.process = subprocess.Popen(
                        cmd,
                        cwd=self.server_dir
                    )

                # Store PID
                self.store_pid(self.process.pid)

                # Wait for server to start
                if wait:
                    logger.info("Waiting for server to start...")
                    start_time = time.time()
                    while time.time() - start_time < START_TIMEOUT:
                        if self.is_server_running():
                            logger.info(f"Server started successfully: {interpreter} {script}")
                            success = True
                            return True
                        time.sleep(0.5)

                    logger.warning(f"Server {interpreter} {script} failed to start within timeout")

                    # Check if process is still running but not responding
                    if self.process and self.process.poll() is None:
                        # Process is running but not responding correctly
                        # Let's try to examine the log file
                        try:
                            with open(self.log_file, 'r') as f:
                                log_tail = f.read()[-1000:]  # Read last 1000 chars
                                logger.warning(f"Server log tail: {log_tail}")
                        except Exception as e:
                            logger.warning(f"Could not read log file: {e}")

                        # Try to terminate the process and try next option
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                else:
                    # Not waiting, assume success
                    logger.info(f"Server process started: {interpreter} {script}")
                    return True
            except Exception as e:
                logger.error(f"Error starting server with {interpreter} {script}: {e}")
                # Continue to next option

        if not success:
            logger.error("All server start attempts failed")
            return False

        return True

    def _check_dependencies(self):
        """Check for required dependencies and try to install them if missing."""
        try:
            # Check for basic Node.js and Python dependencies
            node_dependencies_ok = False
            python_dependencies_ok = False

            # Check Node.js deps first
            try:
                # Check if Node.js is available
                node_version = subprocess.run(['node', '--version'],
                                              capture_output=True, text=True, check=True).stdout.strip()
                logger.info(f"Node.js version: {node_version}")

                # Check npm and package.json
                if os.path.exists(os.path.join(self.server_dir, 'package.json')):
                    # Check if node_modules exists and has required modules
                    node_modules = os.path.join(self.server_dir, 'node_modules')
                    if os.path.isdir(node_modules) and os.path.isdir(os.path.join(node_modules, 'express')):
                        logger.info("Node.js dependencies appear to be installed")
                        node_dependencies_ok = True
                    else:
                        # Try to install
                        logger.info("Installing Node.js dependencies...")
                        try:
                            subprocess.run(['npm', 'install'], cwd=self.server_dir, check=True)
                            logger.info("Node.js dependencies installed successfully")
                            node_dependencies_ok = True
                        except subprocess.SubprocessError as e:
                            logger.warning(f"Error installing Node.js dependencies: {e}")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.warning(f"Node.js not available: {e}")

            # Check Python dependencies
            try:
                # Check for fix_memory_dependencies.py in project root
                project_root = os.path.abspath(os.path.join(self.server_dir, '..', '..', '..'))
                fix_script = os.path.join(project_root, 'fix_memory_dependencies.py')

                if os.path.isfile(fix_script):
                    logger.info("Running dependency fix script...")
                    try:
                        python_path = self.find_python_path()
                        subprocess.check_call([python_path, fix_script])
                        logger.info("Python dependencies installed successfully")
                        python_dependencies_ok = True
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to install Python dependencies: {e}")
                else:
                    # Try to check requirements file
                    requirements_file = os.path.join(self.server_dir, 'requirements.txt')
                    if os.path.isfile(requirements_file):
                        logger.info("Installing requirements from requirements.txt")
                        try:
                            python_path = self.find_python_path()
                            subprocess.check_call([python_path, '-m', 'pip', 'install', '-r', requirements_file])
                            logger.info("Python requirements installed successfully")
                            python_dependencies_ok = True
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Failed to install Python requirements: {e}")
            except Exception as e:
                logger.warning(f"Error checking Python dependencies: {e}")

            # Log dependency status
            logger.info(f"Dependency check summary - Node.js: {node_dependencies_ok}, Python: {python_dependencies_ok}")

            # If no dependencies could be installed, still continue and try to use what we have
            if not (node_dependencies_ok or python_dependencies_ok):
                logger.warning("Could not verify any dependencies, will attempt to continue anyway")

        except Exception as e:
            logger.warning(f"Error checking dependencies: {e}")

    def stop(self):
        """Stop the memory server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        # Check if server is running
        if not self.is_port_in_use():
            logger.info("Server is not running")

            # Clean up PID file
            if os.path.exists(self.pid_file):
                try:
                    os.remove(self.pid_file)
                except:
                    pass

            return True

        # Try to terminate process
        result = False

        # First try stored PID
        stored_pid = self.get_stored_pid()
        if stored_pid and self.is_process_running(stored_pid):
            logger.info(f"Terminating process with stored PID: {stored_pid}")
            result = self.kill_process(stored_pid)

            # Clean up PID file
            try:
                os.remove(self.pid_file)
            except:
                pass

        # If that failed, try process attribute
        if not result and self.process and self.process.poll() is None:
            logger.info(f"Terminating process with PID: {self.process.pid}")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=STOP_TIMEOUT)
                    result = True
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {self.process.pid} did not terminate gracefully, using force")
                    self.process.kill()
                    result = True
            except:
                pass

        # If all else fails, try to find and kill the process via port
        if not result or self.is_port_in_use():
            logger.info(f"Attempting to find process using port {self.port}...")

            pid = self.find_memory_server_pid()
            if pid:
                logger.info(f"Found process using port {self.port}: {pid}")
                result = self.kill_process(pid, force=True)

        # Check if server is still running
        if self.is_port_in_use():
            logger.error("Failed to stop server")
            return False

        logger.info("Server stopped successfully")
        return True

    def restart(self, background=True, wait=True):
        """Restart the memory server.
        
        Args:
            background: Run server in background
            wait: Wait for server to start
            
        Returns:
            True if server restarted successfully, False otherwise
        """
        self.stop()
        return self.start(background, wait)

    def status(self):
        """Get server status.
        
        Returns:
            Status dictionary
        """
        running = self.is_server_running()
        port_in_use = self.is_port_in_use()

        status = {
            "running": running,
            "port_in_use": port_in_use,
            "host": self.host,
            "port": self.port,
            "pid_file": self.pid_file
        }

        if running:
            # Get server info
            try:
                import requests
                response = requests.get(
                    f"http://{self.host}:{self.port}/status",
                    timeout=SERVER_CHECK_TIMEOUT
                )
                server_info = response.json()
                status["server_info"] = server_info
            except:
                status["server_info"] = "Error getting server info"

        # Add stored PID
        stored_pid = self.get_stored_pid()
        if stored_pid:
            status["stored_pid"] = stored_pid
            status["process_running"] = self.is_process_running(stored_pid)

        # Add port checker
        if port_in_use and not running:
            pid = self.find_memory_server_pid()
            if pid:
                status["port_used_by"] = pid

        return status


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Memory Server Manager")

    # Commands
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'status', 'clean'],
                        help='Command to execute')

    # Options
    parser.add_argument('--host', default=DEFAULT_HOST,
                        help=f'Host interface (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'Port number (default: {DEFAULT_PORT})')
    parser.add_argument('--server-dir', type=str, default=None,
                        help='Server directory (default: directory of this script)')
    parser.add_argument('--background', action='store_true',
                        help='Run server in background')
    parser.add_argument('--no-wait', action='store_true',
                        help="Don't wait for server to start")
    parser.add_argument('--force', action='store_true',
                        help='Force operation (e.g., force kill for stop)')

    args = parser.parse_args()

    # Create server manager
    manager = MemoryServerManager(args.host, args.port, args.server_dir)

    # Execute command
    if args.command == 'start':
        success = manager.start(background=args.background, wait=not args.no_wait)
        sys.exit(0 if success else 1)
    elif args.command == 'stop':
        success = manager.stop()
        sys.exit(0 if success else 1)
    elif args.command == 'restart':
        success = manager.restart(background=args.background, wait=not args.no_wait)
        sys.exit(0 if success else 1)
    elif args.command == 'status':
        status = manager.status()
        print(json.dumps(status, indent=2))
        sys.exit(0 if status.get('running', False) else 1)
    elif args.command == 'clean':
        # Kill any process using the port
        pid = manager.find_memory_server_pid()
        if pid:
            print(f"Found process {pid} using port {args.port}, killing it")
            manager.kill_process(pid, force=True)
        # Remove PID file
        if os.path.exists(manager.pid_file):
            os.remove(manager.pid_file)
        print("Cleanup completed")
        sys.exit(0)


if __name__ == '__main__':
    main()
