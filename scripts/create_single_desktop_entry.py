#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a single desktop entry for LLaDA GUI with all optimizations integrated.
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

    # Create a single desktop file
    desktop_content = f"""[Desktop Entry]
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

    # Write the desktop file
    desktop_path = os.path.join(desktop_dir, "LLaDA_GUI.desktop")
    with open(desktop_path, "w") as f:
        f.write(desktop_content)

    print(f"Created single desktop entry: LLaDA_GUI.desktop")
    print(
        "\nA single integrated desktop entry has been created. You should be able to find the LLaDA GUI application in your launcher.")
    print("All optimizations are now accessible through the GUI interface.")
    print("You may need to log out and log back in for the changes to take effect in some desktop environments.")


if __name__ == "__main__":
    main()
