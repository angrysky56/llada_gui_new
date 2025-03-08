#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for Tensor detach issues in memory integration.

This script fixes the error: "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
"""

import logging
import os
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tensor_detach_fix")


def patch_tensor_detach():
    """Patch tensor detach issues in the memory integration code."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Locate potential files with tensor operations
    files_to_check = [
        os.path.join(script_dir, "core", "memory", "memory_integration.py"),
        os.path.join(script_dir, "core", "memory", "memory_server", "server.py"),
        os.path.join(script_dir, "direct_memory_fix.py")
    ]

    fixes_applied = 0

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        logger.info(f"Checking file: {file_path}")

        # Read file content
        with open(file_path, "r") as f:
            content = f.read()

        # Check for tensor.numpy() without detach
        original_content = content

        # Pattern: tensor.numpy() -> tensor.detach().numpy()
        pattern1 = r'([a-zA-Z0-9_]+)\.numpy\(\)'
        content = re.sub(pattern1, r'\1.detach().numpy()', content)

        # Pattern: np.array(tensor) -> np.array(tensor.detach())
        pattern2 = r'np\.array\(([a-zA-Z0-9_]+)\)'
        content = re.sub(pattern2, r'np.array(\1.detach())', content)

        # Pattern: tensor.tolist() -> tensor.detach().tolist()
        pattern3 = r'([a-zA-Z0-9_]+)\.tolist\(\)'
        content = re.sub(pattern3, r'\1.detach().tolist()', content)

        # Save changes if modified
        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            logger.info(f"Applied tensor detach fixes to: {file_path}")
            fixes_applied += 1

    # Add specific fix for common memory server implementations
    server_py = os.path.join(script_dir, "core", "memory", "memory_server", "server.py")
    if os.path.exists(server_py):
        with open(server_py, "r") as f:
            content = f.read()

        # Check if it's a PyTorch implementation
        if "torch" in content and "nn.Module" in content:
            # Add explicit detach calls for PyTorch model outputs
            original_content = content

            # Pattern: modify forward method to detach outputs
            pattern = r'(def forward\([^)]*\).*?return\s*{[^}]*})'

            if "detach()" not in content:
                # Add detach for predicted, newMemory, etc.
                modified_forward = re.sub(pattern,
                                          lambda m: m.group(0).replace('"predicted": predicted',
                                                                       '"predicted": predicted.detach()')
                                          .replace('"newMemory": new_memory',
                                                   '"newMemory": new_memory.detach()'),
                                          content, flags=re.DOTALL)

                if modified_forward != content:
                    with open(server_py, "w") as f:
                        f.write(modified_forward)
                    logger.info(f"Applied PyTorch-specific tensor detach fixes to: {server_py}")
                    fixes_applied += 1

    return fixes_applied > 0


def main():
    """Main function."""
    if patch_tensor_detach():
        logger.info("Successfully fixed tensor detach issues")
        return 0
    else:
        logger.warning("No tensor detach issues found or fixed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
