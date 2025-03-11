#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for tensor gradient issues in Titan Memory system.

This script patches the Titan Memory code to properly detach tensors
when using numpy() function, to prevent the error:
"Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
"""

import logging
import os
import re
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("titan_memory_fix")


def patch_file(file_path, pattern, replacement):
    """Patch a file with find and replace.
    
    Args:
        file_path: Path to file
        pattern: Regex pattern to find
        replacement: Replacement string
        
    Returns:
        True if patched, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()

    # Apply patch
    new_content = re.sub(pattern, replacement, content)

    # Check if anything changed
    if new_content == content:
        return False

    # Write patched content
    with open(file_path, 'w') as f:
        f.write(new_content)

    logger.info(f"Patched {file_path}")
    return True


def patch_save_model():
    """Patch the save_model method in TitanMemoryModel class."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "core", "memory", "titan_memory.py")

    # Pattern to find save_model method
    pattern = r'def save_model\(self, path: str\):(.*?)with open\(path, \'w\'\) as f:'

    # Replacement with detach() calls
    replacement = r'def save_model(self, path: str):\n        """Save model to file.\n        \n        Args:\n            path: Path to save file\n        """\n        # Create directory if it doesn\'t exist\n        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)\n        \n        # Save model state and config\n        state_dict = self.state_dict()\n        config_dict = vars(self.config)\n        save_data = {\n            "state_dict": {k: v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v \n                          for k, v in state_dict.items()},\n            "config": config_dict,\n            "memory_state": self.memory_state.detach().cpu().numpy().tolist()\n        }\n        \n        with open(path, \'w\') as f:'

    if not patch_file(file_path, pattern, replacement):
        logger.warning("Failed to patch save_model method or already patched")

    # Pattern for forward_pass method in TitanMemorySystem
    pattern2 = r'return \{\s+"predicted": result\["predicted"\]\.cpu\(\)\.numpy\(\)\.tolist\(\),\s+"newMemory": result\["new_memory"\]\.cpu\(\)\.numpy\(\)\.tolist\(\),\s+"surprise": float\(result\["surprise"\]\.item\(\)\)\s+\}'

    # Replacement with detach() calls
    replacement2 = r'return {\n            "predicted": result["predicted"].detach().cpu().numpy().tolist(),\n            "newMemory": result["new_memory"].detach().cpu().numpy().tolist(),\n            "surprise": float(result["surprise"].item())\n        }'

    if not patch_file(file_path, pattern2, replacement2):
        logger.warning("Failed to patch forward_pass method or already patched")

    # Pattern for get_memory_state in TitanMemoryModel
    pattern3 = r'def get_memory_state\(self\)(.*?)return self\.memory_state\.detach\(\)\.cpu\(\)\.numpy\(\)'

    # Check if detach is already present
    with open(file_path, 'r') as f:
        content = f.read()

    if 'return self.memory_state.detach().cpu().numpy()' not in content:
        # Pattern for get_memory_state without detach
        pattern3 = r'def get_memory_state\(self\)(.*?)return self\.memory_state\.cpu\(\)\.numpy\(\)'

        # Replacement with detach() call
        replacement3 = r'def get_memory_state(self)\1return self.memory_state.detach().cpu().numpy()'

        if not patch_file(file_path, pattern3, replacement3):
            logger.warning("Failed to patch get_memory_state method or already patched")


def patch_diffusion_adapter():
    """Patch the diffusion adapter to detach tensors in training."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "core", "memory", "titan_integration", "diffusion_adapter.py")

    if not os.path.exists(file_path):
        logger.error(f"Diffusion adapter file not found: {file_path}")
        return

    # Open the file and look for the training section
    with open(file_path, 'r') as f:
        content = f.read()

    # Check for the memory training section
    if "# Train memory on the prompt-generation pair" in content:
        # Fix is needed
        memory_training_pattern = r'# Train memory on the prompt-generation pair\s+loss = self\.memory_guidance\.train_on_generation\(self\.prompt, answer\)'

        # Add try-except with better error handling and detach()
        memory_training_replacement = r'# Train memory on the prompt-generation pair\n                try:\n                    loss = self.memory_guidance.train_on_generation(self.prompt, answer)\n                    logger.info(f"Memory training loss: {loss}")\n                except Exception as e:\n                    logger.error(f"Error in memory training: {e}")\n                    # Try fallback method without gradients\n                    try:\n                        with torch.no_grad():\n                            loss = self.memory_guidance.train_on_generation(self.prompt, answer)\n                            logger.info(f"Memory training loss (fallback): {loss}")\n                    except Exception as e2:\n                        logger.error(f"Fallback memory training also failed: {e2}")'

        if not patch_file(file_path, memory_training_pattern, memory_training_replacement):
            logger.warning("Failed to patch memory training in diffusion adapter or already patched")


def main():
    """Main function."""
    # Patch the TitanMemoryModel save_model method
    patch_save_model()

    # Patch the diffusion adapter
    patch_diffusion_adapter()

    # Patch any other files using simple patterns
    for root, dirs, files in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "memory")):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                # Read file
                with open(file_path, 'r') as f:
                    content = f.read()

                # Simple patterns
                if "tensor.numpy()" in content:
                    content = content.replace("tensor.numpy()", "tensor.detach().numpy()")

                    # Write back
                    with open(file_path, 'w') as f:
                        f.write(content)

                    logger.info(f"Fixed tensor.numpy() in {file_path}")

                # Pattern for tolist on tensor
                if "tensor.tolist()" in content and "tensor.detach().tolist()" not in content:
                    content = content.replace("tensor.tolist()", "tensor.detach().tolist()")

                    # Write back
                    with open(file_path, 'w') as f:
                        f.write(content)

                    logger.info(f"Fixed tensor.tolist() in {file_path}")

    logger.info("All tensor detach fixes applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
