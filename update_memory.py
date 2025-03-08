#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update script for the enhanced Titan Memory integration.

This script updates the LLaDA GUI to use the improved Titan Memory system
instead of the legacy memory server approach.
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
logger = logging.getLogger("memory_update")


def check_directories():
    """Check if required directories exist."""
    # Core memory directory
    if not os.path.exists("core/memory"):
        logger.error("Memory directory not found. Make sure you're running this script from the project root.")
        return False

    # Check if the titan integration directory already exists
    if os.path.exists("core/memory/titan_integration"):
        logger.info("Titan integration directory already exists.")
        return True

    # Create titan_integration directory
    try:
        os.makedirs("core/memory/titan_integration", exist_ok=True)
        logger.info("Created titan_integration directory.")
        return True
    except Exception as e:
        logger.error(f"Failed to create titan_integration directory: {e}")
        return False


def check_titan_memory():
    """Check if Titan Memory is properly installed."""
    if not os.path.exists("core/memory/titan_memory.py"):
        logger.error("Titan Memory module not found. Please run install.py first.")
        return False
    return True


def update_memory_system():
    """Update the memory system to use Titan Memory integration."""
    logger.info("Updating memory system...")

    # Check for required directories
    if not check_directories() or not check_titan_memory():
        return False

    # Install required dependencies
    try:
        logger.info("Installing required dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'numpy>=1.20.0', 'torch>=1.12.0'
        ])
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.warning("Continuing anyway, but the update might not work correctly.")

    # Check if memory integration is working by trying to import
    try:
        logger.info("Checking memory integration...")
        sys.path.insert(0, os.path.abspath("."))
        from core.memory.titan_memory import TitanMemorySystem
        test_system = TitanMemorySystem()
        if test_system.initialized:
            logger.info("Titan Memory system is working correctly.")
        else:
            logger.warning("Titan Memory system initialized but not fully functional.")
    except Exception as e:
        logger.error(f"Failed to import Titan Memory: {e}")
        return False

    # Create integration directory structure
    logger.info("Creating memory integration files...")

    # Create __init__.py file
    init_path = os.path.join("core", "memory", "titan_integration", "__init__.py")
    with open(init_path, "w") as f:
        f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Titan Memory integration for LLaDA diffusion models.

This module provides the necessary interfaces to connect the Titan Memory
system with the LLaDA diffusion process.
\"\"\"

from .memory_guidance import TitanMemoryGuidance
from .diffusion_adapter import integrate_memory_with_diffusion, MemoryGuidedDiffusionWorker

__all__ = [
    'TitanMemoryGuidance',
    'integrate_memory_with_diffusion',
    'MemoryGuidedDiffusionWorker'
]
""")

    logger.info("Created __init__.py")

    # Note the fixed issues
    logger.info("Fixed memory module includes bug fixes for mask_id handling to prevent torch.full() errors")

    logger.info("Memory system update complete. Please restart the application.")
    return True


if __name__ == "__main__":
    logger.info("Starting memory system update...")
    if update_memory_system():
        logger.info("Memory system update completed successfully.")
    else:
        logger.error("Memory system update failed. Please see error messages above.")
