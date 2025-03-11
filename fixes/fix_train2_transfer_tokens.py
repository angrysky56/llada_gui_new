#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix for train2.py - Adds missing get_num_transfer_tokens function
"""

import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_train2_transfer")

# Path to train2.py
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_FILE = os.path.join(SCRIPT_DIR, "train", "train2.py")

# The transfer tokens function to be added
TRANSFER_TOKENS_FUNCTION = '''
def get_num_transfer_tokens(mask_indices: torch.Tensor, steps: int) -> torch.Tensor:
    """Calculate how many masked tokens to unmask at each step."""
    # Count the total number of tokens that need to be unmasked
    total_tokens = mask_indices.sum(dim=1)
    
    # Allocate approximately equal tokens per step
    token_rates = total_tokens.float() / steps
    
    # Create a tensor of the number of tokens to transfer at each step
    # We'll do this by calculating a sequence of cumulative tokens and taking the differences
    cumulative_tokens = torch.outer(token_rates, torch.arange(steps + 1, device=token_rates.device))
    cumulative_tokens = torch.floor(cumulative_tokens).long()
    
    # Ensure we don't exceed the total tokens
    cumulative_tokens = torch.minimum(cumulative_tokens, total_tokens.unsqueeze(1).expand(-1, steps + 1))
    
    # Get the token transfers per step by taking differences
    num_tokens_per_step = cumulative_tokens[:, 1:] - cumulative_tokens[:, :-1]
    
    return num_tokens_per_step
'''

def fix_train2_file():
    """Add missing get_num_transfer_tokens function to train2.py"""
    
    if not os.path.exists(TARGET_FILE):
        logger.error(f"Target file not found: {TARGET_FILE}")
        return False
        
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Check if the function already exists
    if "def get_num_transfer_tokens" in content:
        logger.info("get_num_transfer_tokens function already exists in the file, no need to add it.")
        return True
    
    # Find a good place to add the function - after the add_gumbel_noise function
    gumbel_noise_pattern = r'def add_gumbel_noise.*?\n\n'
    gumbel_noise_match = re.search(gumbel_noise_pattern, content, re.DOTALL)
    
    if gumbel_noise_match:
        # Insert position is right after the add_gumbel_noise function
        insert_pos = gumbel_noise_match.end()
        
        # Insert the transfer tokens function
        modified_content = content[:insert_pos] + TRANSFER_TOKENS_FUNCTION + content[insert_pos:]
        
        # Write back the modified content
        with open(TARGET_FILE, 'w') as f:
            f.write(modified_content)
            
        logger.info(f"Successfully added get_num_transfer_tokens function to {TARGET_FILE}")
        return True
    else:
        # Try adding it after the forward_process function as a backup
        forward_process_pattern = r'def forward_process.*?\n\n'
        forward_match = re.search(forward_process_pattern, content, re.DOTALL)
        
        if forward_match:
            insert_pos = forward_match.end()
            modified_content = content[:insert_pos] + TRANSFER_TOKENS_FUNCTION + content[insert_pos:]
            
            with open(TARGET_FILE, 'w') as f:
                f.write(modified_content)
                
            logger.info(f"Successfully added get_num_transfer_tokens function to {TARGET_FILE} after forward_process")
            return True
        else:
            logger.error("Could not find a suitable position to add the function")
            return False

if __name__ == "__main__":
    fix_train2_file()
