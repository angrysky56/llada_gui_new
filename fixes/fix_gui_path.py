#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix LLaDA GUI generation_finished method with correct file path.

This script modifies the generation_finished method in the correct file location
to enable memory training features.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gui_path_fix")


def fix_generation_finished_method():
    """Fix the generation_finished method to enable memory training."""
    try:
        # Find the LLaDA GUI file - check both locations
        gui_paths = [
            "gui/llada_gui.py",  # Subdirectory location
            "llada_gui.py"  # Root directory location
        ]

        llada_gui_path = None
        for path in gui_paths:
            if os.path.exists(path):
                llada_gui_path = path
                logger.info(f"Found LLaDA GUI file at: {path}")
                break

        if not llada_gui_path:
            logger.error("LLaDA GUI file not found in either location")
            return False

        # Read the file
        with open(llada_gui_path, "r") as f:
            content = f.read()

        # Find the generation_finished method
        method_pos = content.find("def generation_finished")
        if method_pos < 0:
            logger.error("Could not find generation_finished method")
            return False

        # Find where to add memory training code
        completion_pos = content.find("self.status_label.setText(\"Generation complete\")", method_pos)
        if completion_pos < 0:
            # Try alternative text
            completion_pos = content.find("Generation complete", method_pos)
            if completion_pos < 0:
                logger.error("Could not find completion message in generation_finished method")
                return False

        # Find the end of the line
        line_end = content.find("\n", completion_pos)
        if line_end < 0:
            logger.error("Could not find end of completion line")
            return False

        # Check if memory training code is already present
        memory_check = "# Pass generated text to memory system for training"
        if memory_check in content[line_end:line_end + 500]:
            logger.info("Memory training code already present in generation_finished method")
            return True

        # Add memory training code
        memory_code = """
        
        # Pass generated text to memory system for training
        try:
            # Check if memory tab exists
            if hasattr(self, 'memory_viz'):
                # Store generation data
                if hasattr(self.memory_viz, 'set_generation_data'):
                    prompt = self.input_text.toPlainText().strip()
                    self.memory_viz.set_generation_data(prompt, result)
                    print("Stored generation data for memory training")
                
                # Enable train button
                if hasattr(self.memory_viz, 'train_btn'):
                    self.memory_viz.train_btn.setEnabled(True)
                
                # Check for auto-train
                if hasattr(self.memory_viz, 'auto_train'):
                    if hasattr(self.memory_viz.auto_train, 'isChecked') and self.memory_viz.auto_train.isChecked():
                        # Auto-train
                        if hasattr(self.memory_viz, 'train_memory'):
                            self.memory_viz.train_memory()
                            print("Auto-training memory system")
        except Exception as e:
            print(f"Error updating memory system: {e}")
"""

        # Insert memory code
        updated_content = content[:line_end] + memory_code + content[line_end:]

        # Write updated content
        with open(llada_gui_path, "w") as f:
            f.write(updated_content)

        logger.info("Updated generation_finished method with memory training code")
        return True
    except Exception as e:
        logger.error(f"Error fixing generation_finished method: {e}")
        return False


def fix_memory_training_feature():
    """Apply all memory training fixes."""
    # Fix generation_finished method
    if fix_generation_finished_method():
        print("\n✅ Generation finished method fix applied successfully!")
        print("Now the memory system will be updated after each generation.")

        # Update memory update script path references
        try:
            memory_update_path = "core/memory/memory_update.py"
            if os.path.exists(memory_update_path):
                # Update logging in memory_update.py
                with open(memory_update_path, "r") as f:
                    content = f.read()

                # Replace logger.info with print for better visibility
                updated_content = content.replace("logger.info", "print")

                with open(memory_update_path, "w") as f:
                    f.write(updated_content)

                print("✅ Updated memory update module for better logging")
        except Exception as e:
            logger.error(f"Error updating memory update module: {e}")

        print("\nPlease restart the application for the changes to take effect.")
    else:
        print("\n❌ Failed to apply generation_finished method fix.")
        print("You may need to manually modify gui/llada_gui.py")


if __name__ == "__main__":
    logger.info("Starting generation_finished method fix with corrected path...")
    fix_memory_training_feature()
