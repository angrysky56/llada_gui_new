#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create desktop entries for LLaDA GUI.
Creates a main entry for the standard application and a memory-enabled entry.
"""

import os


def main():
    """Main function."""
    # Get repository path
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get user's desktop files directory
    desktop_dir = os.path.expanduser("~/.local/share/applications")
    os.makedirs(desktop_dir, exist_ok=True)

    # Remove any existing LLaDA desktop files
    for filename in os.listdir(desktop_dir):
        if filename.startswith("LLaDA_") and filename.endswith(".desktop"):
            os.remove(os.path.join(desktop_dir, filename))
            print(f"Removed old entry: {filename}")

    # Python interpreter path
    python_path = os.path.join(repo_dir, "venv", "bin", "python")

    # Create the main desktop file
    main_desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=LLaDA GUI
Comment=Graphical interface for the LLaDA language model with integrated optimizations
Exec={python_path} {os.path.join(repo_dir, 'run.py')}
Icon=utilities-terminal
Terminal=false
Categories=Development;Science;AI;
Keywords=AI;LLM;Diffusion;
"""

    # Create the memory-enabled desktop file
    memory_desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=LLaDA GUI (with Memory)
Comment=LLaDA GUI with integrated cognitive memory
Exec={python_path} {os.path.join(repo_dir, 'run.py')} --memory
Icon=text-editor
Terminal=false
Categories=Development;Science;AI;
Keywords=AI;LLM;Diffusion;Memory;
"""

    # Write the desktop files
    main_desktop_path = os.path.join(desktop_dir, "LLaDA_GUI.desktop")
    with open(main_desktop_path, "w") as f:
        f.write(main_desktop_content)

    memory_desktop_path = os.path.join(desktop_dir, "LLaDA_Memory.desktop")
    with open(memory_desktop_path, "w") as f:
        f.write(memory_desktop_content)

    print(f"Created desktop entries:")
    print(f"1. LLaDA_GUI.desktop - Standard application with all optimization options")
    print(f"2. LLaDA_Memory.desktop - Application with memory integration")

    print("\nDesktop entries have been created. You should now have two options in your launcher:")
    print("- LLaDA GUI: The standard application with all optimization options")
    print("- LLaDA GUI (with Memory): The application with memory integration")
    print("\nYou may need to log out and log back in for the changes to take effect in some desktop environments.")


if __name__ == "__main__":
    main()
