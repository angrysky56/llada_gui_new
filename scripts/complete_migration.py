#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to complete the migration from the old repository structure to the new one.
This script updates imports and creates any missing files.
"""

import os
import shutil
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
os.chdir(repo_dir)

# Original repository path
ORIGINAL_REPO = "/home/ty/Repositories/ai_workspace/llada_gui"


def ensure_directory(dir_path):
    """Ensure a directory exists."""
    os.makedirs(dir_path, exist_ok=True)


def create_init_file(dir_path, description=""):
    """Create an __init__.py file in a directory."""
    init_path = os.path.join(dir_path, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write(f'"""\n{description}\n"""\n')
        print(f"Created {init_path}")


def check_and_copy_file(src, dest):
    """Check if a file exists and copy it if not."""
    if not os.path.exists(dest) and os.path.exists(src):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"Copied {src} to {dest}")
    elif not os.path.exists(src):
        print(f"Warning: Source file {src} does not exist")


def main():
    """Main function to complete the migration."""
    # Ensure all required directories exist
    for dir_name in ["core/memory", "gui/visualizations", "optimizations/standard",
                     "optimizations/extreme", "scripts", "docs", "resources"]:
        dir_path = os.path.join(repo_dir, dir_name)
        ensure_directory(dir_path)

    # Create __init__.py files
    create_init_file(os.path.join(repo_dir, "core"), "Core functionality for the LLaDA application.")
    create_init_file(os.path.join(repo_dir, "core/memory"), "Memory management components.")
    create_init_file(os.path.join(repo_dir, "gui"), "GUI components for the LLaDA application.")
    create_init_file(os.path.join(repo_dir, "gui/visualizations"), "Visualization components.")
    create_init_file(os.path.join(repo_dir, "optimizations"), "Memory optimization modules.")
    create_init_file(os.path.join(repo_dir, "optimizations/standard"), "Standard memory optimizations.")
    create_init_file(os.path.join(repo_dir, "optimizations/extreme"), "Extreme memory optimizations.")
    create_init_file(os.path.join(repo_dir, "resources"), "Resource files (images, icons, etc.)")

    # Check and copy critical files if they're missing
    essential_files = [
        (os.path.join(ORIGINAL_REPO, "llada_gui.py"), os.path.join(repo_dir, "gui/llada_gui.py")),
        (os.path.join(ORIGINAL_REPO, "llada_worker.py"), os.path.join(repo_dir, "core/llada_worker.py")),
        (os.path.join(ORIGINAL_REPO, "diffusion_visualization.py"),
         os.path.join(repo_dir, "gui/visualizations/diffusion_visualization.py")),
        (os.path.join(ORIGINAL_REPO, "memory_monitor.py"), os.path.join(repo_dir, "gui/memory_monitor.py")),
        (os.path.join(ORIGINAL_REPO, "utils.py"), os.path.join(repo_dir, "core/utils.py")),
        (os.path.join(ORIGINAL_REPO, "config.py"), os.path.join(repo_dir, "core/config.py")),
        (os.path.join(ORIGINAL_REPO, "generate.py"), os.path.join(repo_dir, "core/generate.py")),
        (os.path.join(ORIGINAL_REPO, "requirements.txt"), os.path.join(repo_dir, "requirements.txt")),
        (os.path.join(ORIGINAL_REPO, "start_gui.sh"), os.path.join(repo_dir, "scripts/start_gui.sh")),
        (os.path.join(ORIGINAL_REPO, "install.sh"), os.path.join(repo_dir, "scripts/install.sh")),
        (os.path.join(ORIGINAL_REPO, "start_with_memory.sh"), os.path.join(repo_dir, "scripts/start_memory.sh")),
    ]

    for src, dest in essential_files:
        check_and_copy_file(src, dest)

    # Update imports in all Python files
    update_imports_script = os.path.join(repo_dir, "scripts/update_imports.py")
    if os.path.exists(update_imports_script):
        print("Running import update script...")
        import subprocess
        try:
            subprocess.run([sys.executable, update_imports_script], check=True)
            print("Import update completed successfully")
        except subprocess.CalledProcessError:
            print("Error running import update script")

    # Make scripts executable
    script_files = [
        os.path.join(repo_dir, "scripts/install.sh"),
        os.path.join(repo_dir, "scripts/start_gui.sh"),
        os.path.join(repo_dir, "scripts/start_memory.sh"),
    ]

    for script in script_files:
        if os.path.exists(script):
            try:
                os.chmod(script, 0o755)  # Make executable
                print(f"Made {script} executable")
            except Exception as e:
                print(f"Error making {script} executable: {e}")

    print("\nMigration completed successfully!")
    print("\nNext steps:")
    print("1. Test the application by running: python run.py")
    print("2. Check the imports in key files to ensure they're working correctly")
    print("3. Review the new structure in docs/STRUCTURE.md")


if __name__ == "__main__":
    main()
