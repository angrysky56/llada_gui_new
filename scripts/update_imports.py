#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to update import statements in the codebase to match the new directory structure.
"""

import os
import re


def update_imports_in_file(filepath):
    """Update import statements in a single file."""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Update direct imports
    replacements = [
        (r'from config import', 'from core.config import'),
        (r'from utils import', 'from core.utils import'),
        (r'from llada_worker import', 'from core.llada_worker import'),
        (r'from memory_monitor import', 'from gui.memory_monitor import'),
        (r'from diffusion_visualization import', 'from gui.visualizations.diffusion_visualization import'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write updated content back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"Updated {filepath}")


def process_directory(directory):
    """Process all Python files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                update_imports_in_file(filepath)


if __name__ == "__main__":
    # Get the repository root directory
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Process main directories
    for directory in ['core', 'gui', 'optimizations']:
        dir_path = os.path.join(repo_dir, directory)
        if os.path.exists(dir_path):
            process_directory(dir_path)

    print("Import statements updated successfully")
