#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fixed version of run_with_memory.py that ensures memory server connects reliably.
"""

import logging
import os
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='llada_memory.log'
)
logger = logging.getLogger("llada_main")


def find_venv_python():
    """Find the Python executable in the virtual environment."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, 'venv')

    if os.path.isdir(venv_dir):
        venv_python = os.path.join(venv_dir, 'bin', 'python')
        if os.path.isfile(venv_python):
            return venv_python

    return sys.executable


def kill_memory_processes():
    """Kill any existing memory server processes."""
    logger.info("Killing any existing memory server processes...")
    try:
        # Use pkill to find and kill server processes
        subprocess.run(['pkill', '-f', 'server.py'], check=False)
        subprocess.run(['pkill', '-f', 'server.js'], check=False)
        subprocess.run(['pkill', '-f', 'memory_server'], check=False)
        time.sleep(1)  # Wait for processes to terminate
    except Exception as e:
        logger.error(f"Error killing memory processes: {e}")


def install_requirements():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    try:
        python_exe = find_venv_python()
        subprocess.check_call([python_exe, '-m', 'pip', 'install', 'flask', 'numpy', 'requests'])
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def main():
    """Main function."""
    # Make sure the system path is set up correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # First kill any existing memory server processes
    kill_memory_processes()

    # Make sure dependencies are installed
    install_requirements()

    # Fix memory database
    try:
        fix_script = os.path.join(script_dir, 'fix_memory_db.py')
        if os.path.exists(fix_script):
            logger.info("Running memory database fix script")
            subprocess.run([find_venv_python(), fix_script], check=False)
    except Exception as e:
        logger.error(f"Error running memory database fix: {e}")

    # Import and run the main application
    try:
        # Adjust the path to try different module locations
        if os.path.exists(os.path.join(script_dir, 'gui', 'llada_gui.py')):
            sys.path.insert(0, os.path.join(script_dir, 'gui'))
            from llada_gui import main as gui_main
        elif os.path.exists(os.path.join(script_dir, 'llada_gui.py')):
            sys.path.insert(0, script_dir)
            from llada_gui import main as gui_main
        else:
            # Try to find the module
            possible_locations = [
                os.path.join(script_dir, 'gui'),
                script_dir,
                os.path.join(script_dir, 'core')
            ]

            for location in possible_locations:
                if location not in sys.path:
                    sys.path.insert(0, location)

            try:
                from gui.llada_gui import main as gui_main
            except ImportError:
                try:
                    # Try to import directly
                    sys.path.insert(0, os.path.join(script_dir, 'gui'))
                    import llada_gui
                    gui_main = llada_gui.main
                except ImportError:
                    raise ImportError("Could not find llada_gui module")

        gui_main()
    except Exception as e:
        logger.error(f"Failed to run GUI: {e}")
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Check if we're already running in the right Python
    if 'VENV_ACTIVATED' not in os.environ:
        venv_python = find_venv_python()
        if venv_python != sys.executable:
            # Re-launch with the Python from the virtual environment
            os.environ['VENV_ACTIVATED'] = '1'
            os.execl(venv_python, venv_python, *sys.argv)

    sys.exit(main())
