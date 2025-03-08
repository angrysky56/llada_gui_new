#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix desktop entries for LLaDA GUI.
This script removes duplicate desktop entries and updates existing ones.
"""

import os
import shutil


def main():
    """Main function."""
    # Get user's desktop files directory
    desktop_dir = os.path.expanduser("~/.local/share/applications")

    # Repository paths
    old_repo = "/home/ty/Repositories/ai_workspace/llada_gui"
    new_repo = "/home/ty/Repositories/ai_workspace/llada_gui_new"

    # Get all LLaDA desktop files
    desktop_files = []
    for filename in os.listdir(desktop_dir):
        if filename.startswith("LLaDA_") and filename.endswith(".desktop"):
            desktop_files.append(os.path.join(desktop_dir, filename))

    print(f"Found {len(desktop_files)} LLaDA desktop files:")
    for i, file in enumerate(desktop_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    # Check content of each file
    old_files = []
    new_files = []

    for file in desktop_files:
        with open(file, "r") as f:
            content = f.read()

        if old_repo in content:
            old_files.append(file)
        elif new_repo in content:
            new_files.append(file)

    print(f"\nFiles referencing the old repository ({len(old_files)}):")
    for file in old_files:
        print(f"- {os.path.basename(file)}")

    print(f"\nFiles referencing the new repository ({len(new_files)}):")
    for file in new_files:
        print(f"- {os.path.basename(file)}")

    # Create backups of old files
    if old_files:
        backup_dir = os.path.join(os.path.expanduser("~"), "llada_desktop_backups")
        os.makedirs(backup_dir, exist_ok=True)

        print(f"\nCreating backups in {backup_dir}")
        for file in old_files:
            backup_file = os.path.join(backup_dir, os.path.basename(file))
            shutil.copy2(file, backup_file)
            print(f"Backed up {os.path.basename(file)}")

    # Remove old files
    if old_files:
        print("\nRemoving old desktop files...")
        for file in old_files:
            os.remove(file)
            print(f"Removed {os.path.basename(file)}")

    # Add descriptions to new files
    if new_files:
        print("\nUpdating descriptions for clarity...")
        for file in new_files:
            with open(file, "r") as f:
                content = f.read()

            filename = os.path.basename(file)
            if filename == "LLaDA_GUI.desktop":
                content = content.replace("Name=LLaDA GUI", "Name=LLaDA GUI (Standard)")

            with open(file, "w") as f:
                f.write(content)
                print(f"Updated {filename}")

    print(
        "\nDesktop files have been cleaned up. You should now see only the new version's entries in your application launcher.")
    print("You may need to log out and log back in for changes to take effect in some desktop environments.")


if __name__ == "__main__":
    main()
