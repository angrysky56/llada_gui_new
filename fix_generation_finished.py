#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix LLaDA GUI generation_finished method to enable memory training.

This script modifies the generation_finished method to:
1. Store the generated text for memory training
2. Enable the training button
3. Perform automatic training if enabled
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generation_fix")


def fix_generation_finished_method():
    """Fix the generation_finished method to enable memory training."""
    try:
        # Find the LLaDA GUI file
        llada_gui_path = "llada_gui.py"
        if not os.path.exists(llada_gui_path):
            logger.error(f"LLaDA GUI file not found: {llada_gui_path}")
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
            logger.error("Could not find completion message in generation_finished method")
            return False

        # Find the end of the line
        line_end = content.find("\n", completion_pos)
        if line_end < 0:
            logger.error("Could not find end of completion line")
            return False

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
                    logger.info("Stored generation data for memory training")
                
                # Enable train button
                if hasattr(self.memory_viz, 'train_btn'):
                    self.memory_viz.train_btn.setEnabled(True)
                
                # Check for auto-train
                if hasattr(self.memory_viz, 'auto_train'):
                    if hasattr(self.memory_viz.auto_train, 'isChecked') and self.memory_viz.auto_train.isChecked():
                        # Auto-train
                        if hasattr(self.memory_viz, 'train_memory'):
                            self.memory_viz.train_memory()
                            logger.info("Auto-training memory system")
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


def main():
    logger.info("Starting generation_finished method fix...")

    if fix_generation_finished_method():
        print("\n✅ Generation finished method fix applied successfully!")
        print("Now the memory system will be updated after each generation.")
        print("This enables immediate training and auto-training.")
    else:
        print("\n❌ Failed to apply generation_finished method fix.")
        print("You may need to manually modify llada_gui.py")

    print("\nPlease restart the application for the changes to take effect.")


if __name__ == "__main__":
    main()
