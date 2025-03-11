#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix QWidget: Must construct a QApplication before a QWidget issue in the LLaDA GUI.

This script patches the necessary files to ensure QApplication is created before any QWidget instances.
"""

import logging
import os
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qt_application_fix")


def fix_memory_monitor():
    """Fix the MemoryMonitor class to ensure QApplication exists before QWidget creation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_monitor_path = os.path.join(script_dir, "gui", "memory_monitor.py")

    if not os.path.exists(memory_monitor_path):
        logger.error(f"Memory monitor file not found: {memory_monitor_path}")
        return False

    # Read the file content
    with open(memory_monitor_path, "r") as f:
        content = f.read()

    # Update imports to include QApplication
    import_section_end = content.find("logger = logging.getLogger(__name__)")
    if import_section_end > 0:
        # Check if QApplication is already imported
        if "QApplication" not in content[:import_section_end]:
            # Find the PyQt6 import line
            pyqt_import_line = re.search(r"from PyQt6\.QtCore import.*", content[:import_section_end])
            
            if pyqt_import_line:
                # Add QApplication to the import list if not already there
                modified_content = content[:pyqt_import_line.start()] + \
                                "from PyQt6.QtWidgets import QApplication, QObject\n" + \
                                content[pyqt_import_line.start():]
                
                # Update content with new import
                content = modified_content
                logger.info("Added QApplication import to memory_monitor.py")

    # Add a check for QApplication instance in the MemoryMonitor constructor
    constructor_pattern = r"def __init__\(self, .*?\):"
    constructor_match = re.search(constructor_pattern, content)
    
    if constructor_match:
        constructor_end = content.find("super().__init__", constructor_match.end())
        if constructor_end > 0:
            # Add code to ensure QApplication exists
            app_check_code = """
        # Ensure a QApplication exists before creating any QWidget
        if not QApplication.instance():
            self.app = QApplication([])
            logger.info("Created QApplication instance in MemoryMonitor")
        else:
            self.app = QApplication.instance()
            logger.info("Using existing QApplication instance in MemoryMonitor")
"""
            # Insert the app check code after super().__init__ line
            next_line_pos = content.find("\n", constructor_end)
            if next_line_pos > 0:
                modified_content = content[:next_line_pos+1] + app_check_code + content[next_line_pos+1:]
                content = modified_content
                logger.info("Added QApplication instance check to MemoryMonitor constructor")

    # Write back the modified content
    with open(memory_monitor_path, "w") as f:
        f.write(content)

    logger.info("Successfully patched memory_monitor.py")
    return True


def fix_run_memory_script():
    """Fix the run_memory.sh script to ensure QApplication is created."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_memory_path = os.path.join(script_dir, "run_memory.sh")

    if not os.path.exists(run_memory_path):
        logger.error(f"run_memory.sh file not found: {run_memory_path}")
        return False

    # Read the file content
    with open(run_memory_path, "r") as f:
        content = f.read()

    # Find the line that runs the Python script
    run_line_pattern = r'"\$PYTHON" run\.py --memory'
    run_line_match = re.search(run_line_pattern, content)
    
    if not run_line_match:
        logger.error("Could not find the line that runs Python in run_memory.sh")
        return False
    
    # Check if --ensure-qt-app flag is already added
    if "--ensure-qt-app" not in content:
        # Add the flag to ensure QApplication is created
        modified_content = content[:run_line_match.end()] + " --ensure-qt-app" + content[run_line_match.end():]
        
        # Write back the modified content
        with open(run_memory_path, "w") as f:
            f.write(modified_content)
        
        logger.info("Added --ensure-qt-app flag to run_memory.sh")
    else:
        logger.info("--ensure-qt-app flag already present in run_memory.sh")

    return True


def fix_run_py():
    """Add handling for the --ensure-qt-app flag in run.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_py_path = os.path.join(script_dir, "run.py")

    if not os.path.exists(run_py_path):
        logger.error(f"run.py file not found: {run_py_path}")
        return False

    # Read the file content
    with open(run_py_path, "r") as f:
        content = f.read()

    # Add the new command-line flag if not present
    args_parser_pattern = r"parser = argparse\.ArgumentParser\("
    args_parser_match = re.search(args_parser_pattern, content)
    
    if args_parser_match:
        # Find where argument definitions end
        args_end = content.find("args = parser.parse_args()", args_parser_match.end())
        if args_end > 0:
            # Check if --ensure-qt-app is already added
            if "--ensure-qt-app" not in content[:args_end]:
                args_lines = content[:args_end].split("\n")
                
                # Find the last argument definition
                last_arg_line = next((i for i in range(len(args_lines)-1, -1, -1) 
                                    if "parser.add_argument" in args_lines[i]), -1)
                
                if last_arg_line >= 0:
                    # Add the new argument after the last one
                    new_arg = 'parser.add_argument("--ensure-qt-app", action="store_true", ' \
                            'help="Ensure QApplication is created before widgets")'
                    args_lines.insert(last_arg_line + 1, new_arg)
                    
                    # Reconstruct content up to args_end
                    modified_content = "\n".join(args_lines) + "\n" + content[args_end:]
                    content = modified_content
                    logger.info("Added --ensure-qt-app argument to run.py")

    # Check if QApplication is imported at the top
    if "from PyQt6.QtWidgets import QApplication" not in content:
        # Add import after other PyQt imports
        imports_section_end = re.search(r"import torch", content).start()
        if imports_section_end > 0:
            modified_content = content[:imports_section_end] + \
                            "from PyQt6.QtWidgets import QApplication\n" + \
                            content[imports_section_end:]
            content = modified_content
            logger.info("Added QApplication import to run.py")

    # Add code to create QApplication if the flag is set
    main_function_pattern = r"def main\(argv=None\):"
    main_function_match = re.search(main_function_pattern, content)
    
    if main_function_match:
        # Find a good place to add the QApplication creation
        # Look for the first code after arguments are parsed
        args_parsed_pos = content.find("if argv is not None:", main_function_match.end())
        if args_parsed_pos > 0:
            # Find the end of the argument parsing block
            after_args_parsed = content.find("# Setup advanced memory management", args_parsed_pos)
            if after_args_parsed > 0:
                # Add code to ensure QApplication exists
                app_check_code = """
    # Create a QApplication instance if requested
    if args.ensure_qt_app and 'QApplication' in globals():
        app_instance = QApplication.instance()
        if not app_instance:
            logger.info("Creating QApplication instance early")
            app = QApplication([])
        else:
            logger.info("QApplication instance already exists")

"""
                modified_content = content[:after_args_parsed] + app_check_code + content[after_args_parsed:]
                content = modified_content
                logger.info("Added QApplication creation code to run.py main function")

    # Write back the modified content
    with open(run_py_path, "w") as f:
        f.write(content)

    logger.info("Successfully patched run.py")
    return True


def main():
    """Main function."""
    print("Starting QWidget/QApplication fix...")
    
    # Fix the memory monitor
    if fix_memory_monitor():
        print("✅ Fixed memory_monitor.py")
    else:
        print("❌ Failed to fix memory_monitor.py")
    
    # Fix the run_memory.sh script
    if fix_run_memory_script():
        print("✅ Fixed run_memory.sh")
    else:
        print("❌ Failed to fix run_memory.sh")
    
    # Fix the run.py script
    if fix_run_py():
        print("✅ Fixed run.py")
    else:
        print("❌ Failed to fix run.py")
    
    print("\nQWidget/QApplication fix completed. Please run ./run_memory.sh again.")
    print("This should prevent the 'QWidget: Must construct a QApplication before a QWidget' error.")


if __name__ == "__main__":
    main()
