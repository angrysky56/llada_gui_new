#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidated Fix Script for LLaDA GUI Memory Integration

This script serves as a launcher for the fix scripts in the "fixes" directory.
Instead of duplicating code, it simply finds and executes the relevant scripts.

Usage:
    python fix_all.py
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llada_fixes")

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXES_DIR = os.path.join(SCRIPT_DIR, "fixes")


def run_script(script_name, required=False):
    """Run a script from the fixes directory."""
    script_path = os.path.join(FIXES_DIR, script_name)
    if os.path.exists(script_path):
        logger.info(f"Running {script_name}...")
        try:
            if script_path.endswith(".py"):
                subprocess.run([sys.executable, script_path], check=True)
            elif script_path.endswith(".sh"):
                subprocess.run(["bash", script_path], check=True)
            else:
                logger.warning(f"Unknown script type: {script_path}")
                return False
            logger.info(f"Successfully ran {script_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {script_name}: {e}")
            if required:
                logger.error(f"Required script {script_name} failed, aborting.")
                sys.exit(1)
            return False
    else:
        msg = f"{script_name} not found in {FIXES_DIR}"
        if required:
            logger.error(f"Required script {msg}, aborting.")
            sys.exit(1)
        else:
            logger.warning(msg)
        return False


def copy_script_to_main(script_name, destination_name=None):
    """Copy a script from fixes to the main directory and make it executable."""
    if destination_name is None:
        destination_name = script_name
        
    source_path = os.path.join(FIXES_DIR, script_name)
    dest_path = os.path.join(SCRIPT_DIR, destination_name)
    
    if os.path.exists(source_path):
        logger.info(f"Copying {script_name} to main directory as {destination_name}...")
        try:
            with open(source_path, 'rb') as src_file:
                with open(dest_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            
            # Make executable
            os.chmod(dest_path, 0o755)
            logger.info(f"Successfully copied and made executable: {destination_name}")
            return True
        except Exception as e:
            logger.error(f"Error copying {script_name}: {e}")
            return False
    else:
        logger.warning(f"{script_name} not found in {FIXES_DIR}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("LLaDA GUI Memory Integration Fix Launcher")
    print("=" * 60)
    
    if not os.path.exists(FIXES_DIR):
        logger.error(f"Fixes directory not found: {FIXES_DIR}")
        sys.exit(1)
    
    # 1. Essential scripts that should be copied to main directory
    essential_scripts = {
        "run_with_memory.sh": "run_with_memory.sh",
        "fix_run_memory.sh": "fix_run_memory.sh",
        "direct_memory_fix.py": "direct_memory_fix.py"
    }
    
    print("\n[1/4] Copying essential scripts to main directory...")
    for source, dest in essential_scripts.items():
        copy_script_to_main(source, dest)
    
    # 2. Run memory database and server fixes
    print("\n[2/4] Running memory system fixes...")
    memory_fixes = [
        "fix_memory_db.py",
        "fix_titan_memory.py",
        "fix_memory_server.py",
        "fix_memory_dependencies.py",
        "fix_memory_integration.py"
    ]
    
    for fix in memory_fixes:
        run_script(fix)
    
    # 3. Run GUI integration fixes and create desktop icons
    print("\n[3/4] Running GUI integration fixes...")
    gui_fixes = [
        "fix_memory_gui_instance.py", 
        "patch_memory_widget.py",
        "fix_qobject_import.py",
        "fix_qt_application.py",
        "create_desktop_icons.sh"
    ]
    
    for fix in gui_fixes:
        run_script(fix)
    
    # 4. Run training module fixes
    print("\n[4/4] Running training module fixes...")
    training_fixes = [
        "fix_train2.py"
    ]
    
    for fix in training_fixes:
        run_script(fix)
    
    print("\n" + "=" * 60)
    print("Fix process completed!")
    print("=" * 60)
    print("\nYou can now run the LLaDA GUI with memory integration:")
    print("  ./run_with_memory.sh")
    print("\nIf you encounter any issues, you can run individual fixes from the fixes directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
