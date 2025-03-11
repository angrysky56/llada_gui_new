#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for train2.py - Adds missing add_gumbel_noise function
"""

import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_train2")

# Path to train2.py
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE = os.path.join(SCRIPT_DIR, "train", "train2.py")

# The Gumbel noise function to be added
GUMBEL_NOISE_FUNCTION = '''
def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Add Gumbel noise to logits for sampling."""
    if temperature == 0:
        return logits
    
    # Sample Gumbel noise
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
    
    # Apply temperature and add noise
    return logits + gumbel_noise * temperature
'''

def fix_train2_file():
    """Add missing add_gumbel_noise function to train2.py"""
    
    if not os.path.exists(TARGET_FILE):
        logger.error(f"Target file not found: {TARGET_FILE}")
        return False
        
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Check if the function already exists
    if "def add_gumbel_noise" in content:
        logger.info("add_gumbel_noise function already exists in the file, no need to add it.")
        return True
    
    # Find a good place to add the function - after the linear_beta_schedule function
    linear_beta_pattern = r'def linear_beta_schedule.*?\)'
    linear_beta_match = re.search(linear_beta_pattern, content)
    
    if linear_beta_match:
        # Find the end of the function
        function_start = linear_beta_match.start()
        lines = content[function_start:].split('\n')
        
        # Find the first blank line after the function
        insert_pos = function_start
        for i, line in enumerate(lines):
            insert_pos += len(line) + 1  # +1 for the newline character
            if i > 0 and line.strip() == '':
                break
                
        # Insert the Gumbel noise function
        modified_content = content[:insert_pos] + GUMBEL_NOISE_FUNCTION + content[insert_pos:]
        
        # Write back the modified content
        with open(TARGET_FILE, 'w') as f:
            f.write(modified_content)
            
        logger.info(f"Successfully added add_gumbel_noise function to {TARGET_FILE}")
        return True
    else:
        logger.error("Could not find the right position to add the function")
        return False

if __name__ == "__main__":
    fix_train2_file()
