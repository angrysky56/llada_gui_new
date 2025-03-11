#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the memory integration.

This script tests the memory integration to ensure it's working correctly
after the code reorganization.
"""

import sys
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_memory_integration")

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_initialization():
    """Test memory initialization."""
    from core.memory.memory_init import initialize_memory, get_memory_interface

    print("Testing memory initialization...")
    success = initialize_memory(start_server=True)
    
    if success:
        print("✅ Memory initialization successful")
        memory_interface = get_memory_interface()
        print(f"Memory interface: {memory_interface}")
        print(f"Memory initialized: {memory_interface.initialized}")
        print(f"Memory state shape: {memory_interface.get_memory_state().shape}")
    else:
        print("❌ Memory initialization failed")
    
    return success

def test_memory_operations():
    """Test memory operations."""
    from core.memory.memory_init import get_memory_interface, reset_memory
    
    memory_interface = get_memory_interface()
    if not memory_interface or not memory_interface.initialized:
        print("❌ Memory interface not initialized, skipping operations test")
        return False
    
    print("Testing memory operations...")
    
    # Create a test vector
    test_vector = np.random.randn(memory_interface.input_dim)
    
    # Run forward pass
    print("Running forward pass...")
    result = memory_interface.forward_pass(test_vector)
    print(f"Forward pass result: {result.keys()}")
    
    # Reset memory
    print("Resetting memory...")
    reset_memory()
    print(f"Memory state after reset: {np.sum(memory_interface.get_memory_state())}")
    
    print("✅ Memory operations test passed")
    return True

def test_gui_enhancement():
    """Test GUI enhancement function."""
    try:
        from core.memory.memory_ui import enhance_llada_gui
        
        # We're not actually creating the GUI here, just checking that
        # the function is available and importable
        if enhance_llada_gui:
            print("✅ GUI enhancement function available")
            return True
        else:
            print("❌ GUI enhancement function not available")
            return False
    except ImportError as e:
        print(f"❌ Failed to import GUI enhancement function: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing memory integration...")
    
    # Test initialization
    init_success = test_initialization()
    
    # Test memory operations if initialization succeeded
    if init_success:
        test_memory_operations()
    
    # Test GUI enhancement
    test_gui_enhancement()
    
    print("Tests complete.")

if __name__ == "__main__":
    main()
