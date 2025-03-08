#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory update helper for LLaDA GUI.

This module provides functionality to update the memory system after generation,
enabling automatic training and data flow.
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)


def update_memory_after_generation(gui_instance, prompt, generated_text):
    """Update memory system after generation.
    
    This function should be called after generation completes to:
    1. Store the generation data for future training
    2. Enable the training button
    3. Perform automatic training if enabled
    
    Args:
        gui_instance: LLaDAGUI instance
        prompt: Input prompt
        generated_text: Generated text
    """
    if not hasattr(gui_instance, 'memory_viz'):
        logger.warning("GUI has no memory_viz attribute, cannot update memory")
        return

    memory_viz = gui_instance.memory_viz

    # Store data
    if hasattr(memory_viz, 'set_generation_data'):
        print("Storing generation data in memory system")
        memory_viz.set_generation_data(prompt, generated_text)

    # Enable training button
    if hasattr(memory_viz, 'train_btn'):
        memory_viz.train_btn.setEnabled(True)
        memory_viz.training_status.setText("Training data available")

    # Check if automatic training is enabled (only using Titan memory)
    if hasattr(gui_instance, 'worker') and hasattr(gui_instance.worker, 'memory_guidance'):
        memory_guidance = gui_instance.worker.memory_guidance
        if hasattr(memory_guidance, 'auto_train') and memory_guidance.auto_train:
            print("Auto-training enabled, training memory on generation")
            if hasattr(memory_viz, 'train_memory'):
                try:
                    memory_viz.train_memory()
                except Exception as e:
                    logger.error(f"Error in auto-training: {e}")
