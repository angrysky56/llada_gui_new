#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive memory server status checker.
This script performs diagnostics on the memory server setup.
"""

import json
import logging
import os
import socket
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_port_in_use(port=3000):
    """Check if the port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_memory_server_pid():
    """Find the PID of the memory server process."""
    try:
        # Try to find process listening on port 3000
        result = subprocess.run(['lsof', '-i', ':3000', '-t'],
                                capture_output=True, text=True, check=False)
        if result.stdout:
            return result.stdout.strip().split('\n')[0]
        return None
    except Exception as e:
        logger.error(f"Error finding memory server PID: {e}")
        return None


def check_memory_server_files():
    """Check if memory server files are present."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_server_dir = os.path.join(script_dir, 'core', 'memory', 'memory_server')

    files_to_check = [
        os.path.join(memory_server_dir, 'server.js'),
        os.path.join(memory_server_dir, 'server.py'),
        os.path.join(memory_server_dir, 'requirements.txt'),
        os.path.join(memory_server_dir, 'package.json'),
    ]

    missing_files = []
    for file_path in files_to_check:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)

    return {
        'memory_server_dir': memory_server_dir,
        'dir_exists': os.path.isdir(memory_server_dir),
        'missing_files': missing_files,
        'all_files_present': len(missing_files) == 0
    }


def check_node_js_installed():
    """Check if Node.js is installed."""
    try:
        node_version = subprocess.run(['node', '--version'],
                                      capture_output=True, text=True, check=False)
        if node_version.returncode == 0:
            return True, node_version.stdout.strip()
        return False, None
    except Exception:
        return False, None


def check_pip_dependencies():
    """Check if Python dependencies are installed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, 'core', 'memory', 'memory_server', 'requirements.txt')

    if os.path.isfile(requirements_file):
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            missing_packages = []
            for package in requirements:
                package_name = package.split('==')[0].split('>=')[0].strip()
                try:
                    __import__(package_name)
                except ImportError:
                    missing_packages.append(package_name)

            return {
                'requirements_file': requirements_file,
                'total_packages': len(requirements),
                'missing_packages': missing_packages,
                'all_packages_installed': len(missing_packages) == 0
            }
        except Exception as e:
            return {
                'requirements_file': requirements_file,
                'error': str(e),
                'all_packages_installed': False
            }
    else:
        return {
            'requirements_file': requirements_file,
            'error': 'Requirements file not found',
            'all_packages_installed': False
        }


def test_memory_server_connection():
    """Test connection to memory server."""
    if not check_port_in_use(3000):
        return {
            'connected': False,
            'error': 'Memory server not running (port 3000 not in use)'
        }

    try:
        import requests
        response = requests.get('http://localhost:3000/status', timeout=2)
        if response.status_code == 200:
            try:
                data = response.json()
                return {
                    'connected': True,
                    'status': data,
                    'response_code': response.status_code
                }
            except Exception:
                return {
                    'connected': True,
                    'raw_response': response.text,
                    'response_code': response.status_code
                }
        else:
            return {
                'connected': False,
                'error': f'Unexpected status code: {response.status_code}',
                'raw_response': response.text,
                'response_code': response.status_code
            }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e)
        }


def check_log_files():
    """Check log files for errors."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_files = {
        'llada_memory.log': os.path.join(script_dir, 'llada_memory.log'),
        'memory_server.log': os.path.join(script_dir, 'memory_server.log'),
        'memory_server_py.log': os.path.join(script_dir, 'memory_server_py.log')
    }

    log_status = {}
    for name, path in log_files.items():
        if os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    # Get last 10 lines containing errors or warnings
                    error_lines = [line for line in lines if 'ERROR' in line or 'WARNING' in line][-10:]
                    log_status[name] = {
                        'exists': True,
                        'size': os.path.getsize(path),
                        'error_count': len(error_lines),
                        'recent_errors': error_lines
                    }
            except Exception as e:
                log_status[name] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            log_status[name] = {
                'exists': False
            }

    return log_status


def main():
    """Run comprehensive checks and display results."""
    print("Running LLaDA GUI Memory Server Status Check...")
    print("-" * 60)

    # Check if memory server is running
    port_in_use = check_port_in_use()
    pid = find_memory_server_pid() if port_in_use else None

    print(f"Memory Server Running: {port_in_use}")
    if pid:
        print(f"Process ID: {pid}")

    print("-" * 60)
    print("File System Checks:")
    files_check = check_memory_server_files()
    print(f"Memory Server Directory: {files_check['memory_server_dir']}")
    print(f"Directory Exists: {files_check['dir_exists']}")
    print(f"All Required Files Present: {files_check['all_files_present']}")
    if not files_check['all_files_present']:
        print("Missing Files:")
        for file in files_check['missing_files']:
            print(f"  - {file}")

    print("-" * 60)
    print("Dependency Checks:")

    node_installed, node_version = check_node_js_installed()
    print(f"Node.js Installed: {node_installed}")
    if node_installed:
        print(f"Node.js Version: {node_version}")

    pip_check = check_pip_dependencies()
    print(f"All Python Packages Installed: {pip_check.get('all_packages_installed', False)}")
    if not pip_check.get('all_packages_installed', False) and 'missing_packages' in pip_check:
        if pip_check['missing_packages']:
            print("Missing Python Packages:")
            for package in pip_check['missing_packages']:
                print(f"  - {package}")

    print("-" * 60)
    print("Connection Test:")
    connection_test = test_memory_server_connection()
    print(f"Connected to Memory Server: {connection_test['connected']}")
    if not connection_test['connected'] and 'error' in connection_test:
        print(f"Connection Error: {connection_test['error']}")
    elif connection_test['connected'] and 'status' in connection_test:
        print(f"Server Status: {json.dumps(connection_test['status'], indent=2)}")

    print("-" * 60)
    print("Log Files:")
    log_check = check_log_files()
    for name, status in log_check.items():
        print(f"{name}: {'Exists' if status.get('exists', False) else 'Not Found'}")
        if status.get('exists', False) and 'error_count' in status:
            print(f"  Error Count: {status['error_count']}")
            if status['error_count'] > 0:
                print("  Recent Errors:")
                for line in status['recent_errors'][-5:]:  # Show only the last 5 errors
                    print(f"    {line.strip()}")

    print("-" * 60)
    print("Recommended Actions:")

    if not files_check['all_files_present']:
        print("- Install/reinstall memory server files")

    if not node_installed:
        print("- Install Node.js (required for optimal performance)")

    if not pip_check.get('all_packages_installed', False):
        print("- Run 'python fix_memory_dependencies.py' to install missing Python packages")

    if port_in_use and not connection_test['connected']:
        print("- Kill the current memory server process and restart it")
        print(f"  kill -9 {pid}" if pid else "- Use 'lsof -i :3000' to find the process and kill it")

    if not port_in_use:
        print("- Start the memory server using './run_simple_memory.sh'")

    # Check if the memory GUI script exists and is executable
    simple_memory_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_simple_memory.sh')
    if os.path.isfile(simple_memory_script) and os.access(simple_memory_script, os.X_OK):
        print("- Use the simplified memory script: './run_simple_memory.sh'")
    else:
        print("- Create a simple script to launch the memory server and GUI together")


if __name__ == "__main__":
    main()
