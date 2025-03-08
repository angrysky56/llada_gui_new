#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix the gui_instance reference error in memory-guided generation.

This script patches the LLaDA GUI code to fix the 'gui_instance' not defined error
when using memory-guided generation.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_fix")


def fix_diffusion_adapter():
    """Fix the diffusion adapter to properly handle the worker thread."""
    try:
        # Find the diffusion adapter file
        adapter_path = "core/memory/titan_integration/diffusion_adapter.py"
        if not os.path.exists(adapter_path):
            logger.error(f"Diffusion adapter file not found: {adapter_path}")
            return False

        # Read the file
        with open(adapter_path, "r") as f:
            content = f.read()

        # Find the MemoryGuidedDiffusionWorker class
        worker_class_pos = content.find("class MemoryGuidedDiffusionWorker")
        if worker_class_pos < 0:
            logger.error("Could not find MemoryGuidedDiffusionWorker class")
            return False

        # Check if there's a reference to gui_instance in the worker
        if "gui_instance.memory_viz" in content[worker_class_pos:]:
            # Find the constructor
            init_pos = content.find("def __init__", worker_class_pos)
            if init_pos > 0:
                # Update the constructor to pass gui_instance
                if "gui_instance" not in content[init_pos:init_pos + 200]:
                    # Find the parameter list
                    params_start = content.find("(", init_pos)
                    params_end = content.find(")", params_start)

                    if params_start > 0 and params_end > params_start:
                        # Add gui_instance parameter
                        new_params = content[params_start + 1:params_end].strip()
                        if new_params.endswith(","):
                            new_params += " gui_instance=None"
                        else:
                            new_params += ", gui_instance=None"

                        updated_content = content[:params_start + 1] + new_params + content[params_end:]

                        # Find where to store gui_instance
                        method_body_start = content.find(":", params_end)
                        if method_body_start > 0:
                            next_line = content.find("\n", method_body_start)
                            if next_line > 0:
                                # Add gui_instance storage
                                gui_instance_line = "\n        self.gui_instance = gui_instance"
                                updated_content = updated_content[:next_line] + gui_instance_line + updated_content[
                                                                                                    next_line:]

                                # Modify any references to gui_instance to use self.gui_instance
                                updated_content = updated_content.replace("gui_instance.memory_viz",
                                                                          "self.gui_instance.memory_viz")

                                # Write updated content
                                with open(adapter_path, "w") as f:
                                    f.write(updated_content)

                                logger.info("Added gui_instance parameter to worker constructor")

                                # Now update the memory handler to pass gui_instance
                                return fix_memory_handler()

        logger.info("No gui_instance reference found in worker, no fix needed")
        return True
    except Exception as e:
        logger.error(f"Error fixing diffusion adapter: {e}")
        return False


def fix_memory_handler():
    """Fix the memory handler to pass gui_instance to the worker."""
    try:
        # Find the memory handler file
        handler_path = "core/memory/titan_integration/memory_handler.py"
        if not os.path.exists(handler_path):
            logger.error(f"Memory handler file not found: {handler_path}")
            return False

        # Read the file
        with open(handler_path, "r") as f:
            content = f.read()

        # Find the worker creation
        worker_creation_pos = content.find("worker = MemoryGuidedDiffusionWorker(")
        if worker_creation_pos < 0:
            logger.error("Could not find worker creation in memory handler")
            return False

        # Check if gui_instance is already being passed
        if "gui_instance=" not in content[worker_creation_pos:worker_creation_pos + 100]:
            # Find the end of the parameter list
            params_end = content.find(")", worker_creation_pos)
            if params_end > 0:
                # Add gui_instance parameter
                updated_content = content[:params_end] + ", gui_instance=gui_instance" + content[params_end:]

                # Write updated content
                with open(handler_path, "w") as f:
                    f.write(updated_content)

                logger.info("Added gui_instance parameter to worker creation")
                return True

        logger.info("gui_instance already being passed to worker, no fix needed")
        return True
    except Exception as e:
        logger.error(f"Error fixing memory handler: {e}")
        return False


def main():
    logger.info("Starting gui_instance reference fix...")

    if fix_diffusion_adapter():
        print("\n✅ gui_instance reference fix applied successfully!")
        print("This should resolve the 'name 'gui_instance' is not defined' error.")
    else:
        print("\n❌ Failed to apply gui_instance reference fix.")
        print("You may need to manually modify core/memory/titan_integration/ files.")

    print("\nPlease restart the application for the changes to take effect.")


if __name__ == "__main__":
    main()
