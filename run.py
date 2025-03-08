#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for the LLaDA GUI application with optimized generation.

Command-line options:
  --optimize        Apply standard memory optimizations
  --extreme         Apply extreme memory optimizations (for 8-12GB GPUs)
  --memory          Enable memory integration
  --help            Show this help message
"""

import argparse
import gc
import logging
import os
import sys
import traceback

import torch

venv_path = './venv/bin/python'

# Parse command-line arguments
parser = argparse.ArgumentParser(description="LLaDA GUI - Large Language Diffusion with mAsking")
parser.add_argument("--optimize", action="store_true", help="Apply standard memory optimizations")
parser.add_argument("--extreme", action="store_true", help="Apply extreme memory optimizations")
parser.add_argument("--memory", action="store_true", help="Enable memory integration")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--advanced-memory", action="store_true", help="Enable advanced memory management")
parser.add_argument("--gpu-monitor", action="store_true", help="Enable detailed GPU monitoring")
args = parser.parse_args()

# Make sure we're running from the right directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Make sure we have the correct Python environment
if os.path.exists(venv_path):
    python_path = os.path.abspath(venv_path)
    if sys.executable != python_path:
        print(f"Restarting with the virtual environment Python: {python_path}")
        os.execl(python_path, python_path, *sys.argv)

# Add the appropriate paths to Python path
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "core"))
sys.path.insert(0, os.path.join(current_dir, "gui"))
sys.path.insert(0, os.path.join(current_dir, "optimizations"))

# Configure logging
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(current_dir, "llada_gui.log"))
    ]
)
logger = logging.getLogger("llada_main")

# Check for memory environment variable
if os.environ.get("LLADA_MEMORY_ENABLED") == "1" and not args.memory:
    args.memory = True
    print("Memory integration enabled from environment variable")


# Main entry point
def main(argv=None):
    """Main entry point for the application."""

    if argv is not None:
        # Use provided arguments
        global args
        args = parser.parse_args(argv)

    # Setup advanced memory management by default
    try:
        print("Initializing advanced memory management...")
        # Import and initialize memory management
        from core.memory_management.integration import integrate_memory_management, optimize_memory_for_startup

        # Apply startup optimizations
        optimize_memory_for_startup()

        # Integrate with the application
        if integrate_memory_management():
            print("Advanced memory management initialized successfully")
        else:
            print("Warning: Failed to initialize advanced memory management, falling back to standard memory handling")

    except ImportError:
        print("Warning: Advanced memory management module not found, using standard memory handling")
    except Exception as e:
        print(f"Warning: Error initializing advanced memory management: {str(e)}")

    # Setup performance optimizations by default
    try:
        print("Initializing performance optimizations...")
        # Import and initialize performance optimizations
        from core.performance.integration import integrate_performance_optimizations

        # Integrate with the application
        if integrate_performance_optimizations():
            print("Performance optimizations initialized successfully")
        else:
            print("Warning: Failed to initialize performance optimizations, falling back to standard performance")

    except ImportError:
        print("Warning: Performance optimization module not found, using standard performance")
    except Exception as e:
        print(f"Warning: Error initializing performance optimizations: {str(e)}")

    # Apply optimizations if requested
    if args.extreme:
        logger.info("Applying extreme memory optimizations (for 8-12GB GPUs)")
        try:
            # Use the streamlined extreme optimizer
            from optimizations.extreme.apply_optimizations import apply_optimizations
            apply_optimizations()
        except Exception as e:
            logger.error(f"Error applying extreme optimizations: {str(e)}")
            print(f"Warning: Error applying extreme optimizations: {str(e)}")
    elif args.optimize:
        logger.info("Applying standard memory optimizations")
        try:
            # Use the standard optimizer
            from optimizations.standard.optimize import apply_optimizations
            apply_optimizations()
        except Exception as e:
            logger.error(f"Error applying standard optimizations: {str(e)}")
            print(f"Warning: Error applying standard optimizations: {str(e)}")

    # Apply memory integration if requested
    if args.memory:
        logger.info("Enabling memory integration")
        try:
            # Run memory database fix
            fix_path = os.path.join(current_dir, 'fix_memory_db.py')
            if os.path.exists(fix_path):
                logger.info("Running memory database fix")
                import subprocess
                subprocess.run([sys.executable, fix_path], check=False)

            # Initialize memory system
            from core.memory.memory_integration import initialize_memory
            if initialize_memory(start_server=True, max_retries=5):
                logger.info("Memory integration initialized successfully")
            else:
                logger.warning("Memory initialization failed, but continuing without it")
        except Exception as e:
            logger.error(f"Error initializing memory integration: {str(e)}")
            print(f"Warning: Error initializing memory integration: {str(e)}")

    # Check if required packages are installed
    try:
        import PyQt6
        import transformers
        import psutil
        import numpy
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages using:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)

    # Clean up any leftover GPU memory
    if torch.cuda.is_available():
        try:
            # More aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()
            # Set environment variables for better memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory / (1024 ** 3)
                print(f"  Device {i}: {device_props.name} with {total_memory:.2f}GB memory")
        except Exception as e:
            print(f"Warning: Error accessing CUDA: {str(e)}")

    # Import core utilities to create required directories
    try:
        from core.utils import create_data_directories
        create_data_directories()
    except Exception as e:
        print(f"Warning: Unable to create data directories: {str(e)}")

    # Import and run the main application
    try:
        # Always import memory adapter if available
        try:
            from core.memory.memory_adapter import add_memory_visualization_tab
            HAS_MEMORY_ADAPTER = True
        except ImportError:
            HAS_MEMORY_ADAPTER = False
            logger.warning("Memory adapter not available")

        if args.memory:
            # Use enhanced GUI with memory integration
            from core.memory.memory_integration import enhance_llada_gui
            from gui.llada_gui import LLaDAGUI
            EnhancedLLaDAGUI = enhance_llada_gui(LLaDAGUI)

            # Manually initialize the app
            from PyQt6.QtWidgets import QApplication
            app = QApplication(sys.argv)
            window = EnhancedLLaDAGUI()

            # Explicitly set memory integration checkbox to checked
            if hasattr(window, 'memory_integration'):
                window.memory_integration.setChecked(True)

            window.show()
            return app.exec()
        else:
            # Use standard GUI without memory
            from gui.llada_gui import main as gui_main
            return gui_main()
    except Exception as e:
        # Display error in a GUI dialog if possible, otherwise print to console
        error_msg = f"Error starting LLaDA GUI: {str(e)}\n\n{traceback.format_exc()}"
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication([])
            QMessageBox.critical(None, "LLaDA GUI Error", error_msg)
        except:
            print(error_msg)
        return 1


# Run main if this is the main script
if __name__ == "__main__":
    sys.exit(main())
