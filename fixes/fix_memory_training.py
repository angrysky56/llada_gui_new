#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick fix for memory training button in LLaDA GUI.

This script ensures that the memory training button is enabled
after each generation, allowing for immediate training.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_fix")


def fix_memory_training_button():
    """Fix the memory training button activation."""
    try:
        # Find the memory integration file
        memory_integration_path = "core/memory/memory_integration.py"
        if not os.path.exists(memory_integration_path):
            logger.error(f"Memory integration file not found: {memory_integration_path}")
            return False

        # Read the file
        with open(memory_integration_path, "r") as f:
            content = f.read()

        # Find the MemoryVisualizationWidget class
        widget_class_pos = content.find("class MemoryVisualizationWidget")
        if widget_class_pos < 0:
            logger.error("Could not find MemoryVisualizationWidget class")
            return False

        # Find if set_generation_data method exists
        if "def set_generation_data" in content:
            # Find the method
            method_pos = content.find("def set_generation_data", widget_class_pos)
            if method_pos > 0:
                # Find where the method updates the training button
                enable_pos = content.find("self.train_btn.setEnabled(", method_pos)
                if enable_pos > 0:
                    # Get the line
                    line_end = content.find("\n", enable_pos)
                    line = content[enable_pos:line_end]

                    # Check if the button is conditionally enabled
                    if "self.memory_interface.initialized" in line:
                        # Update to always enable the button
                        new_line = "        self.train_btn.setEnabled(True)"
                        updated_content = content[:enable_pos] + new_line + content[line_end:]

                        # Write updated content
                        with open(memory_integration_path, "w") as f:
                            f.write(updated_content)

                        logger.info("Updated train button enabling logic")
                        return True
                    else:
                        logger.info("Train button already properly enabled")
                        return True

        logger.warning("Could not find set_generation_data method")
        return False
    except Exception as e:
        logger.error(f"Error fixing memory training button: {e}")
        return False


def main():
    logger.info("Starting memory training button fix...")

    if fix_memory_training_button():
        print("\n✅ Memory training button fix applied successfully!")
        print("Now the button will be enabled after each generation.")
    else:
        print("\n❌ Failed to apply memory training button fix.")
        print("You may need to manually modify core/memory/memory_integration.py")

    print("\nPlease restart the application for the changes to take effect.")


if __name__ == "__main__":
    main()
