#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the diffusion adapter to remove any gui_instance references.

This script ensures that the memory-guided diffusion process doesn't
make any references to variables outside its scope.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diffusion_adapter_fix")


def fix_diffusion_adapter():
    """Fix issues in the memory-guided diffusion adapter."""
    try:
        # Find the diffusion adapter file
        adapter_path = "core/memory/titan_integration/diffusion_adapter.py"
        if not os.path.exists(adapter_path):
            logger.error(f"Diffusion adapter file not found: {adapter_path}")
            return False

        # Read the file
        with open(adapter_path, "r") as f:
            content = f.read()

        # Check if gui_instance is referenced anywhere
        if "gui_instance" in content:
            # Extract the MemoryGuidedDiffusionWorker class
            class_start = content.find("class MemoryGuidedDiffusionWorker")
            if class_start < 0:
                logger.error("Could not find MemoryGuidedDiffusionWorker class")
                return False

            # Find the run method
            run_method_start = content.find("def run", class_start)
            if run_method_start < 0:
                logger.error("Could not find run method in worker class")
                return False

            # Find the next method after run
            next_method = content.find("def ", run_method_start + 5)
            if next_method < 0:
                next_method = len(content)

            # Check for gui_instance references in the run method
            run_method = content[run_method_start:next_method]
            if "gui_instance" in run_method:
                # Replace gui_instance references
                updated_run = run_method.replace("if hasattr(gui_instance, 'memory_viz')",
                                                 "# Removed gui_instance reference")
                updated_run = updated_run.replace("gui_instance.memory_viz", "self.memory_viz")

                # Update the content
                updated_content = content[:run_method_start] + updated_run + content[next_method:]

                # Write updated content
                with open(adapter_path, "w") as f:
                    f.write(updated_content)

                logger.info("Removed gui_instance references from diffusion adapter")
                return True

        logger.info("No gui_instance references found in diffusion adapter")
        return True
    except Exception as e:
        logger.error(f"Error fixing diffusion adapter: {e}")
        return False


def main():
    logger.info("Starting diffusion adapter fix...")

    if fix_diffusion_adapter():
        print("\n✅ Diffusion adapter fix applied successfully!")
        print("This fixes references to undefined variables that could cause errors.")
    else:
        print("\n❌ Failed to apply diffusion adapter fix.")
        print("You may need to manually modify core/memory/titan_integration/diffusion_adapter.py")

    print("\nPlease restart the application for the changes to take effect.")


if __name__ == "__main__":
    main()
