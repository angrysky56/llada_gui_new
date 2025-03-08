#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to verify the directory structure of the LLaDA GUI project.
"""

import os
import sys


def verify_directory(path, required_items=None, optional_items=None):
    """Verify that a directory exists and contains required items."""
    if not os.path.isdir(path):
        print(f"❌ Missing directory: {path}")
        return False

    print(f"✓ Found directory: {path}")

    if required_items:
        missing = []
        for item in required_items:
            if not os.path.exists(os.path.join(path, item)):
                missing.append(item)

        if missing:
            print(f"❌ Missing required items in {path}:")
            for item in missing:
                print(f"  - {item}")
            return False
        else:
            print(f"✓ All required items found in {path}")

    if optional_items:
        missing = []
        for item in optional_items:
            if not os.path.exists(os.path.join(path, item)):
                missing.append(item)

        if missing:
            print(f"ℹ️ Missing optional items in {path}:")
            for item in missing:
                print(f"  - {item}")

    return True


def main():
    """Main function to verify directory structure."""
    # Get repository root
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_dir)

    print(f"Verifying directory structure for: {repo_dir}")
    print("=" * 80)

    # Verify top-level directories
    top_dirs = ["core", "gui", "optimizations", "scripts", "docs", "resources", "archive", "data"]
    top_files = ["run.py", "requirements.txt", "requirements_memory.txt", "README.md"]

    all_good = True

    # Check top-level directories
    for directory in top_dirs:
        if not verify_directory(os.path.join(repo_dir, directory)):
            all_good = False

    # Check top-level files
    for file in top_files:
        if not os.path.isfile(os.path.join(repo_dir, file)):
            print(f"❌ Missing file: {file}")
            all_good = False
        else:
            print(f"✓ Found file: {file}")

    # Check subdirectories
    subdirs = [
        ("core", ["config.py", "generate.py", "llada_worker.py", "utils.py", "memory", "onnx"]),
        ("core/onnx", ["__init__.py", "converter.py", "onnx_integration.py"]),
        ("core/memory", ["__init__.py", "memory_server"]),
        ("gui", ["llada_gui.py", "memory_monitor.py", "visualizations"]),
        ("gui/visualizations", ["__init__.py", "diffusion_visualization.py"]),
        ("optimizations", ["standard", "extreme", "__init__.py"]),
        ("optimizations/standard", ["optimize.py", "__init__.py"]),
        ("optimizations/extreme", ["apply_optimizations.py", "memory_patches.py", "__init__.py"]),
        ("scripts", ["install.sh", "start_gui.sh", "start_memory.sh"]),
        ("resources", ["desktop"]),
        ("data", ["models", "memory"]),
        ("data/models", ["GSAI-ML_LLaDA-8B-Instruct", "onnx_models"])
    ]

    for subdir, required_items in subdirs:
        if not verify_directory(os.path.join(repo_dir, subdir), required_items=required_items):
            all_good = False

    print("=" * 80)
    if all_good:
        print("✅ Directory structure verification passed!")
        return 0
    else:
        print("❌ Directory structure verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
