#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory handler for integrating Titan Memory with LLaDA GUI.

This module provides the integration point between the memory system
and the GUI, handling the creation and management of memory-guided generation.
"""

import logging

from PyQt6.QtWidgets import QMessageBox

from .diffusion_adapter import MemoryGuidedDiffusionWorker
# Import memory components
from .memory_guidance import TitanMemoryGuidance
from ..titan_memory import TitanMemorySystem

# Set up logging
logger = logging.getLogger(__name__)


def handle_memory_integration(gui_instance, prompt, config):
    """Handle memory integration for generation.
    
    Args:
        gui_instance: LLaDAGUI instance
        prompt: Input prompt
        config: Generation configuration
        
    Returns:
        True if memory integration was handled, False otherwise
    """
    # Try to use the Titan memory integration
    try:
        # Get memory weight from config
        memory_weight = config.get('memory_weight', 0.3)

        # Check if memory_viz exists to get memory influence
        if hasattr(gui_instance, 'memory_viz'):
            if hasattr(gui_instance.memory_viz, 'get_memory_influence'):
                memory_weight = gui_instance.memory_viz.get_memory_influence()
                logger.info(f"Using memory weight from visualization: {memory_weight}")

            # Check for auto-train setting and pass to memory_guidance
            if hasattr(gui_instance.memory_viz, 'auto_train'):
                auto_train = gui_instance.memory_viz.auto_train.isChecked()
                logger.info(f"Auto-training setting from UI: {'enabled' if auto_train else 'disabled'}")

        # Create memory system
        memory_system = TitanMemorySystem()

        # Initialize Titan memory guidance
        memory_guidance = TitanMemoryGuidance(
            memory_system=memory_system,
            memory_weight=memory_weight
        )

        # Set auto-train based on UI if available
        if hasattr(gui_instance, 'memory_viz') and hasattr(gui_instance.memory_viz, 'auto_train'):
            memory_guidance.auto_train = gui_instance.memory_viz.auto_train.isChecked()

        # Ensure it has a valid mask_id (needed by torch.full)
        if memory_guidance.mask_id is None:
            logger.warning("Memory guidance has no mask_id, using default 126336")
            memory_guidance.mask_id = 126336

        # Create worker
        worker = MemoryGuidedDiffusionWorker(prompt, config, memory_guidance)
        gui_instance.worker = worker

        # Connect signals
        worker.progress.connect(gui_instance.update_progress)

        # Connect visualization update if available
        if hasattr(gui_instance, 'update_visualization'):
            worker.step_update.connect(gui_instance.update_visualization)
        else:
            logger.warning("GUI instance does not have update_visualization method")

        # Connect memory visualization if available
        if hasattr(gui_instance, 'memory_viz'):
            if hasattr(gui_instance.memory_viz, 'display_memory_state'):
                worker.memory_update.connect(gui_instance.memory_viz.display_memory_state)
            else:
                logger.warning("Memory visualization widget does not have display_memory_state method")
        else:
            logger.warning("GUI instance does not have memory_viz attribute")

        worker.finished.connect(gui_instance.generation_finished)
        worker.error.connect(gui_instance.generation_error)

        # Start worker
        worker.start()
        logger.info("Titan memory-guided worker started successfully")

        return True

    except Exception as e:
        logger.error(f"Error with Titan memory integration: {e}")
        logger.warning("Falling back to legacy memory integration")

        # Fall back to legacy memory integration if available
        try:
            from ..memory_integration import MemoryGuidanceDiffusionWorker as LegacyWorker
            from ..memory_integration import get_memory_interface, initialize_memory

            # Get memory interface
            memory_interface = get_memory_interface()

            # Check if memory interface exists and is initialized
            if memory_interface is None or not memory_interface.initialized:
                # Try to initialize
                logger.info("Memory interface not initialized, attempting to initialize")
                success = initialize_memory(start_server=True)

                if not success:
                    logger.error("Failed to initialize memory system after all attempts")
                    QMessageBox.warning(
                        gui_instance,
                        "Memory Integration Error",
                        "Could not initialize memory system. Running without memory integration."
                    )
                    return False

                memory_interface = get_memory_interface()
                if memory_interface is None:
                    logger.error("Memory interface is still None after initialization")
                    return False

            # Create legacy worker
            memory_weight = config.get('memory_weight', 0.3)
            logger.info(f"Creating legacy memory-guided worker with weight {memory_weight}")

            worker = LegacyWorker(prompt, config, memory_interface)
            gui_instance.worker = worker

            # Connect signals
            worker.progress.connect(gui_instance.update_progress)

            if hasattr(gui_instance, 'update_visualization'):
                worker.step_update.connect(gui_instance.update_visualization)

            if hasattr(gui_instance, 'memory_viz'):
                if hasattr(gui_instance.memory_viz, 'display_memory_state'):
                    worker.memory_update.connect(gui_instance.memory_viz.display_memory_state)

            worker.finished.connect(gui_instance.generation_finished)
            worker.error.connect(gui_instance.generation_error)

            # Start worker
            worker.start()
            logger.info("Legacy memory-guided worker started successfully")

            return True

        except Exception as e:
            logger.error(f"Error with legacy memory integration: {e}")
            QMessageBox.warning(
                gui_instance,
                "Memory Integration Error",
                f"Error integrating memory: {str(e)}\n\nFalling back to standard generation."
            )
            return False
