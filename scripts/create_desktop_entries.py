#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create desktop entries for LLaDA GUI.
"""

import os


def create_desktop_file(name, comment, exec_command, icon="utilities-terminal", keywords=None):
    """Create a desktop file with the given parameters."""
    if keywords is None:
        keywords = ["AI", "LLM", "Diffusion"]

    content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={name}
Comment={comment}
Exec={exec_command}
Icon={icon}
Terminal=false
Categories=Development;Science;AI;
Keywords={';'.join(keywords)};
"""
    return content


def main():
    """Main function."""
    # Get repository path
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get user's desktop files directory
    desktop_dir = os.path.expanduser("~/.local/share/applications")
    os.makedirs(desktop_dir, exist_ok=True)

    # Python interpreter path
    python_path = os.path.join(repo_dir, "venv", "bin", "python")

    # Create desktop files
    desktop_files = [
        {
            "filename": "LLaDA_GUI.desktop",
            "name": "LLaDA GUI (Standard)",
            "comment": "Graphical interface for the LLaDA language model",
            "exec": f"{python_path} {os.path.join(repo_dir, 'run.py')}",
            "keywords": ["AI", "LLM", "Diffusion"]
        },
        {
            "filename": "LLaDA_Optimized.desktop",
            "name": "LLaDA GUI (Optimized)",
            "comment": "Memory-optimized LLaDA GUI",
            "exec": f"{python_path} {os.path.join(repo_dir, 'run.py')} --optimize",
            "keywords": ["AI", "LLM", "Diffusion", "Optimized"]
        },
        {
            "filename": "LLaDA_Extreme.desktop",
            "name": "LLaDA GUI (Extreme)",
            "comment": "Extreme memory optimization for GPUs with limited VRAM",
            "exec": f"{python_path} {os.path.join(repo_dir, 'run.py')} --extreme",
            "keywords": ["AI", "LLM", "Diffusion", "Extreme", "Optimization"]
        },
        {
            "filename": "LLaDA_Memory.desktop",
            "name": "LLaDA GUI with Memory",
            "comment": "LLaDA GUI with integrated cognitive memory",
            "exec": f"{python_path} {os.path.join(repo_dir, 'run.py')} --memory",
            "icon": "text-editor",
            "keywords": ["AI", "LLM", "Diffusion", "Memory"]
        }
    ]

    for file_info in desktop_files:
        filename = file_info["filename"]
        name = file_info["name"]
        comment = file_info["comment"]
        exec_command = file_info["exec"]
        icon = file_info.get("icon", "utilities-terminal")
        keywords = file_info.get("keywords", ["AI", "LLM", "Diffusion"])

        # Create desktop file content
        content = create_desktop_file(name, comment, exec_command, icon, keywords)

        # Write desktop file
        desktop_path = os.path.join(desktop_dir, filename)
        with open(desktop_path, "w") as f:
            f.write(content)

        print(f"Created desktop entry: {filename}")

    print(
        "\nDesktop entries have been created. You should be able to find the LLaDA GUI applications in your launcher.")
    print("You may need to log out and log back in for the changes to take effect in some desktop environments.")


if __name__ == "__main__":
    main()
