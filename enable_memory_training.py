#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enable memory training for LLaDA GUI.

This script adds the auto-train option to the memory visualization
and ensures training data flow works correctly.
"""

import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_patch")


def patch_memory_adapter():
    """Patch the memory adapter module to include auto-training feature."""
    try:
        # Find the memory adapter file
        memory_adapter_path = "core/memory/memory_adapter.py"
        if not os.path.exists(memory_adapter_path):
            logger.error(f"Memory adapter file not found: {memory_adapter_path}")
            return False

        logger.info(f"Memory adapter file: {memory_adapter_path}")

        # Check if update file exists and is newer
        memory_update_path = os.path.join(os.path.dirname(memory_adapter_path), "memory_update.py")
        if not os.path.exists(memory_update_path):
            # Create memory update module
            logger.info("Creating memory update module")
            with open(memory_update_path, "w") as f:
                f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Memory update helper for LLaDA GUI.

This module provides functionality to update the memory system after generation,
enabling automatic training and data flow.
\"\"\"

import logging
from PyQt6.QtWidgets import QMessageBox

# Configure logging
logger = logging.getLogger(__name__)

def update_memory_after_generation(gui_instance, prompt, generated_text):
    \"\"\"Update memory system after generation.
    
    This function should be called after generation completes to:
    1. Store the generation data for future training
    2. Enable the training button
    3. Perform automatic training if enabled
    
    Args:
        gui_instance: LLaDAGUI instance
        prompt: Input prompt
        generated_text: Generated text
    \"\"\"
    if not hasattr(gui_instance, 'memory_viz'):
        logger.warning("GUI has no memory_viz attribute, cannot update memory")
        return
    
    memory_viz = gui_instance.memory_viz
    
    # Store data
    if hasattr(memory_viz, 'set_generation_data'):
        logger.info("Storing generation data in memory system")
        memory_viz.set_generation_data(prompt, generated_text)
    
    # Enable training button
    if hasattr(memory_viz, 'train_btn'):
        memory_viz.train_btn.setEnabled(True)
        memory_viz.training_status.setText("Training data available")
    
    # Check if automatic training is enabled (only using Titan memory)
    if hasattr(gui_instance, 'worker') and hasattr(gui_instance.worker, 'memory_guidance'):
        memory_guidance = gui_instance.worker.memory_guidance
        if hasattr(memory_guidance, 'auto_train') and memory_guidance.auto_train:
            logger.info("Auto-training enabled, training memory on generation")
            if hasattr(memory_viz, 'train_memory'):
                try:
                    memory_viz.train_memory()
                except Exception as e:
                    logger.error(f"Error in auto-training: {e}")
""")

        # Update the memory_adapter.py file to import and use the new module
        with open(memory_adapter_path, "r") as f:
            content = f.read()

        # Add import if it doesn't exist
        if "from .memory_update import update_memory_after_generation" not in content:
            import_line = "# Import memory update helper\ntry:\n    from .memory_update import update_memory_after_generation\n    HAS_MEMORY_UPDATE = True\nexcept ImportError:\n    HAS_MEMORY_UPDATE = False\n    print(\"Warning: Memory update module not available\")\n"

            # Find a good place to add the import
            import_section_end = content.find("# Configure logging")
            if import_section_end > 0:
                content = content[:import_section_end] + import_line + content[import_section_end:]
            else:
                import_section_end = content.find("import logging")
                if import_section_end > 0:
                    line_end = content.find("\n", import_section_end)
                    content = content[:line_end + 1] + "\n" + import_line + content[line_end + 1:]
                else:
                    content = import_line + content

            # Add HAS_MEMORY_UPDATE global if needed
            memory_interfaces_pos = content.find("# Memory interfaces")
            if memory_interfaces_pos > 0:
                if "HAS_MEMORY_UPDATE = " not in content:
                    # Find the end of memory interface declarations
                    end_pos = content.find("\n\n", memory_interfaces_pos)
                    if end_pos > 0:
                        content = content[:end_pos] + "\nHAS_MEMORY_UPDATE = False" + content[end_pos:]

            # Add hook for generation_finished
            if "def handle_memory_integration" in content:
                handle_pos = content.find("def handle_memory_integration")

                # Find where to insert the memory update hook
                func_start = content.find("{", handle_pos)
                if func_start > 0:
                    # Get the next content line after function start
                    next_line = content.find("\n", func_start)
                    if next_line > 0:
                        memory_hook = """
    # Set up the generation_finished handler to update memory
    if HAS_MEMORY_UPDATE:
        # Store the original generation_finished function
        if hasattr(gui_instance, 'generation_finished'):
            original_generation_finished = gui_instance.generation_finished
            
            # Create a wrapper function that also updates memory
            def memory_aware_generation_finished(result):
                # Call the original function first
                original_generation_finished(result)
                
                # Then update memory with the generation result
                if result:
                    update_memory_after_generation(gui_instance, prompt, result)
            
            # Replace the function with our wrapper
            gui_instance.generation_finished = memory_aware_generation_finished
"""
                        content = content[:next_line + 1] + memory_hook + content[next_line + 1:]

        # Write updated content back to file
        with open(memory_adapter_path, "w") as f:
            f.write(content)

        logger.info("Updated memory adapter to include memory update module")
        return True
    except Exception as e:
        logger.error(f"Error patching memory adapter: {e}")
        return False


def patch_memory_visualization():
    """Patch the memory visualization widget to add auto-train checkbox."""
    try:
        # Find the memory integration file
        memory_integration_path = "core/memory/memory_integration.py"
        if not os.path.exists(memory_integration_path):
            logger.error(f"Memory integration file not found: {memory_integration_path}")
            return False

        # Read the file
        with open(memory_integration_path, "r") as f:
            content = f.read()

        # Check if auto-train already exists
        if "auto_train = QCheckBox" in content:
            logger.info("Auto-train checkbox already exists")
            return True

        # Find the memory influence section where we want to add the checkbox
        influence_section = "# Memory influence settings"
        influence_pos = content.find(influence_section)
        if influence_pos < 0:
            logger.error("Could not find memory influence section")
            return False

        # Find where to insert the auto-train checkbox
        memory_layout_pos = content.find("memory_layout.addLayout(influence_layout)", influence_pos)
        if memory_layout_pos > 0:
            # Insert auto-train checkbox
            auto_train_code = """
        # Auto-training option
        self.auto_train = QCheckBox("Auto-train after generation")
        self.auto_train.setChecked(True)  # Enable by default
        self.auto_train.setToolTip("Automatically train the memory system on each generated output")
        memory_layout.addWidget(self.auto_train)
        """
            updated_content = content[:memory_layout_pos + len(
                "memory_layout.addLayout(influence_layout)")] + "\n" + auto_train_code + content[
                                                                                         memory_layout_pos + len(
                                                                                             "memory_layout.addLayout(influence_layout)"):]

            # Write updated content
            with open(memory_integration_path, "w") as f:
                f.write(updated_content)

            logger.info(f"Added auto-train checkbox to {memory_integration_path}")
            return True

        logger.warning("Could not find right location to add auto-train checkbox")
        return False
    except Exception as e:
        logger.error(f"Error patching memory visualization: {e}")
        return False


def update_memory_guidance():
    """Update the memory guidance class to support auto-training."""
    try:
        # Find the memory guidance file
        memory_guidance_path = "core/memory/titan_integration/memory_guidance.py"
        if not os.path.exists(memory_guidance_path):
            logger.error("Could not find memory guidance file")
            return False

        # Read the file
        with open(memory_guidance_path, "r") as f:
            content = f.read()

        # Check if auto_train already exists
        if "self.auto_train = " in content:
            logger.info("Auto-train flag already exists in memory guidance")
            return True

        # Find the initialization section
        init_pos = content.find("self.confidence_threshold = 0.7")
        if init_pos > 0:
            # Add auto_train property
            auto_train_code = "\n        self.auto_train = True  # Enable automatic training by default"
            updated_content = content[:init_pos + len("self.confidence_threshold = 0.7")] + auto_train_code + content[
                                                                                                              init_pos + len(
                                                                                                                  "self.confidence_threshold = 0.7"):]

            # Write updated content
            with open(memory_guidance_path, "w") as f:
                f.write(updated_content)

            logger.info("Added auto-train flag to memory guidance")
            return True

        logger.warning("Could not find right location to add auto-train flag")
        return False
    except Exception as e:
        logger.error(f"Error updating memory guidance: {e}")
        return False


def update_memory_handler():
    """Update the memory handler to connect the auto-train option."""
    try:
        # Find the memory handler file
        memory_handler_path = "core/memory/titan_integration/memory_handler.py"
        if not os.path.exists(memory_handler_path):
            logger.error("Could not find memory handler file")
            return False

        # Read the file
        with open(memory_handler_path, "r") as f:
            content = f.read()

        # Check if auto-train checking already exists
        if "memory_guidance.auto_train = " in content:
            logger.info("Auto-train flag setting already exists in memory handler")
            return True

        # Find where to insert the auto-train code
        memory_guidance_pos = content.find("memory_guidance = TitanMemoryGuidance(")
        if memory_guidance_pos > 0:
            # Find the end of initialization
            end_pos = content.find(")", memory_guidance_pos)
            if end_pos > 0:
                # Find the next line
                next_line = content.find("\n", end_pos)
                if next_line > 0:
                    # Add auto-train setting code
                    auto_train_code = """
        
        # Set auto-train based on UI if available
        if hasattr(gui_instance, 'memory_viz') and hasattr(gui_instance.memory_viz, 'auto_train'):
            memory_guidance.auto_train = gui_instance.memory_viz.auto_train.isChecked()
"""
                    updated_content = content[:next_line] + auto_train_code + content[next_line:]

                    # Write updated content
                    with open(memory_handler_path, "w") as f:
                        f.write(updated_content)

                    logger.info("Added auto-train setting to memory handler")
                    return True

        logger.warning("Could not find right location to add auto-train setting")
        return False
    except Exception as e:
        logger.error(f"Error updating memory handler: {e}")
        return False


def add_automatic_training_generation_finished():
    """Add automatic training to the generation_finished method in LLaDA GUI."""
    try:
        # Find the llada_gui.py file
        gui_file = "llada_gui.py"
        if not os.path.exists(gui_file):
            logger.error(f"GUI file not found: {gui_file}")
            return False

        # Read the file
        with open(gui_file, "r") as f:
            content = f.read()

        # Check if memory update is already being called
        if "update_memory_after_generation" in content:
            logger.info("Memory update already being called in generation_finished")
            return True

        # Find the generation_finished method
        gen_finished_pos = content.find("def generation_finished")
        if gen_finished_pos < 0:
            logger.error("Could not find generation_finished method")
            return False

        # Find where to insert the memory update code
        method_end = content.find("def", gen_finished_pos + 10)
        if method_end > 0:
            # Prepare the import statement
            import_code = """
    # Import memory update if available
    try:
        from core.memory.memory_update import update_memory_after_generation
        HAS_MEMORY_UPDATE = True
    except ImportError:
        HAS_MEMORY_UPDATE = False
"""
            # Add import to class initialization
            class_init_pos = content.find("def __init__", 0, gen_finished_pos)
            if class_init_pos > 0:
                init_end = content.find("\n\n", class_init_pos)
                if init_end > 0:
                    content = content[:init_end] + import_code + content[init_end:]

            # Find where to add the memory update in generation_finished
            completion_pos = content.find("self.status_label.setText(\"Generation complete\")", gen_finished_pos)
            if completion_pos > 0:
                line_end = content.find("\n", completion_pos)
                if line_end > 0:
                    # Add memory update code
                    memory_update_code = """
        
        # Update memory system with the generated text
        if hasattr(self, 'HAS_MEMORY_UPDATE') and self.HAS_MEMORY_UPDATE:
            try:
                # Get the prompt and generated text
                prompt = self.input_text.toPlainText().strip()
                generated_text = result
                
                # Update memory
                update_memory_after_generation(self, prompt, generated_text)
            except Exception as e:
                print(f"Error updating memory: {e}")
"""
                    content = content[:line_end] + memory_update_code + content[line_end:]

                    # Write updated content
                    with open(gui_file, "w") as f:
                        f.write(content)

                    logger.info("Added memory update to generation_finished")
                    return True

        logger.warning("Could not find right location to add memory update")
        return False
    except Exception as e:
        logger.error(f"Error adding automatic training: {e}")
        return False


def main():
    logger.info("Starting memory training patch...")

    success_count = 0
    total_count = 5

    # Patch memory adapter
    if patch_memory_adapter():
        logger.info("Memory adapter patched successfully")
        success_count += 1
    else:
        logger.error("Failed to patch memory adapter")

    # Patch memory visualization widget
    if patch_memory_visualization():
        logger.info("Memory visualization patched successfully")
        success_count += 1
    else:
        logger.error("Failed to patch memory visualization")

    # Update memory guidance class
    if update_memory_guidance():
        logger.info("Memory guidance updated successfully")
        success_count += 1
    else:
        logger.error("Failed to update memory guidance")

    # Update memory handler
    if update_memory_handler():
        logger.info("Memory handler updated successfully")
        success_count += 1
    else:
        logger.error("Failed to update memory handler")

    # Add automatic training to generation_finished
    if add_automatic_training_generation_finished():
        logger.info("Automatic training added to generation_finished")
        success_count += 1
    else:
        logger.error("Failed to add automatic training to generation_finished")

    logger.info(f"Memory training patch complete: {success_count}/{total_count} steps successful")

    print("\n" + "=" * 60)
    print(" Memory Training Feature Enablement")
    print("=" * 60)
    print(f"\nCompleteness: {success_count}/{total_count} steps successful")

    if success_count == total_count:
        print("\n✅ Memory training has been fully enabled!")
    elif success_count > 0:
        print("\n⚠️ Memory training has been partially enabled.")
    else:
        print("\n❌ Failed to enable memory training.")

    print("\nPlease restart the application for the changes to take effect.")
    print("After restart, you should see an 'Auto-train after generation' checkbox in the Memory tab.")
    print("This will allow automatic training of the memory system after each generation.")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
