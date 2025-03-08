#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct launcher for LLaDA GUI with memory integration.

This script directly runs the LLaDA GUI from the correct module path.
"""

import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llada_launcher")


def find_venv_python():
    """Find the Python executable in the virtual environment."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def kill_processes():
    """Kill any existing memory server processes."""
    try:
        import subprocess
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'server.js'], check=False)
        import time
        time.sleep(1)  # Wait for processes to terminate
        return True
    except Exception as e:
        logger.error(f"Error killing processes: {e}")
        return False


def run_fix_script():
    """Run the memory database fix script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fix_script = os.path.join(script_dir, 'fix_memory_db.py')

    if os.path.exists(fix_script):
        try:
            subprocess.run([find_venv_python(), fix_script], check=True)
            return True
        except Exception as e:
            logger.error(f"Error running fix script: {e}")
            return False
    else:
        logger.error(f"Fix script not found: {fix_script}")
        return False


def main():
    """Main function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # First, kill any existing memory server processes
    kill_processes()

    # Run the fix script
    run_fix_script()

    # Now run the main GUI in this process directly
    # Execute the GUI module directly
    sys.path.insert(0, script_dir)  # Add script dir to path

    # Import the module from the correct location
    try:
        if os.path.exists(os.path.join(script_dir, 'gui', 'llada_gui.py')):
            sys.path.insert(0, os.path.join(script_dir, 'gui'))
            from gui.llada_gui import main
        else:
            # Fall back to import from the main directory
            import llada_gui
            main = llada_gui.main

        # Run the main function
        main()
        return 0
    except ImportError as e:
        logger.error(f"Error importing GUI module: {e}")

        # As a last resort, try to run the original script using subprocess
        run_script = os.path.join(script_dir, 'run.py')
        if os.path.exists(run_script):
            try:
                subprocess.run([find_venv_python(), run_script], check=True)
                return 0
            except Exception as e:
                logger.error(f"Error running original script: {e}")
                return 1
        else:
            return 1


if __name__ == "__main__":
    # Check if we're already running in the right Python
    if 'VENV_ACTIVATED' not in os.environ:
        venv_python = find_venv_python()
        if venv_python != sys.executable:
            # Re-launch with the Python from the virtual environment
            os.environ['VENV_ACTIVATED'] = '1'
            os.execl(venv_python, venv_python, *sys.argv)

    sys.exit(main())
