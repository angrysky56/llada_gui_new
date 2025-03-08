#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation script for LLaDA GUI.
This creates a virtual environment and installs all required dependencies.
"""

import os
import subprocess
import sys

import venv


def main():
    """Main installation function."""
    # Get repository root directory
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)

    print(f"Installing LLaDA GUI in {repo_dir}...")

    # Create virtual environment if it doesn't exist
    venv_dir = os.path.join(repo_dir, "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)

    # Get the path to the Python executable in the venv
    if os.name == 'nt':  # Windows
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix/Linux/Mac
        python_exe = os.path.join(venv_dir, "bin", "python")

    # Check if the virtual environment was created successfully
    if not os.path.exists(python_exe):
        print("Error: Failed to create virtual environment")
        return 1

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install main requirements
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing main requirements...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", req_file], check=True)

    # Install memory requirements
    req_memory_file = os.path.join(repo_dir, "requirements_memory.txt")
    if os.path.exists(req_memory_file):
        print("Installing memory-related requirements...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", req_memory_file], check=True)

    # Make scripts executable
    scripts_dir = os.path.join(repo_dir, "scripts")
    if os.path.exists(scripts_dir):
        print("Making scripts executable...")
        for script in os.listdir(scripts_dir):
            if script.endswith(".sh"):
                script_path = os.path.join(scripts_dir, script)
                os.chmod(script_path, 0o755)

    # Make main script executable
    run_py = os.path.join(repo_dir, "run.py")
    if os.path.exists(run_py):
        os.chmod(run_py, 0o755)

    # Create data directories
    data_dir = os.path.join(repo_dir, "data")
    memory_dir = os.path.join(data_dir, "memory")
    models_dir = os.path.join(data_dir, "models")

    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Installation complete!")
    print(f"You can now run LLaDA GUI using: {python_exe} {run_py}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
