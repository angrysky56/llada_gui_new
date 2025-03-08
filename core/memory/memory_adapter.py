#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory adapter for LLaDA GUI.

This module provides the memory visualization and integration with the GUI.
"""

import logging
import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QTabWidget, QMessageBox

# Import memory update helper
try:
    from .memory_update import update_memory_after_generation

    HAS_MEMORY_UPDATE = True
except ImportError:
    HAS_MEMORY_UPDATE = False
    print("Warning: Memory update module not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_adapter.log'
)
logger = logging.getLogger("memory_adapter")

# Memory interfaces
HAS_MEMORY_INTERFACE = False
HAS_TITAN_MEMORY = False

# Try to import Titan memory first (preferred)
try:
    from .titan_integration import TitanMemoryGuidance, MemoryGuidedDiffusionWorker
    from .titan_integration.memory_handler import handle_memory_integration as titan_handle_memory_integration

    HAS_TITAN_MEMORY = True
    logger.info("Titan memory integration available")
except ImportError as e:
    logger.warning(f"Titan memory integration not available: {e}")
    print(f"Titan memory integration not available: {e}")

# Fall back to legacy memory interface if Titan not available
try:
    from .memory_integration import (
        MemoryVisualizationWidget, get_memory_interface, initialize_memory,
        MCPTitanMemoryInterface, MemoryGuidanceDiffusionWorker,
        try_python_server_fallback
    )

    HAS_MEMORY_INTERFACE = True
    logger.info("Legacy memory interface available")
except ImportError as e:
    logger.error(f"Legacy memory interface not available: {e}")
    print(f"Legacy memory interface not available: {e}")


def add_memory_visualization_tab(gui_instance):
    """Add memory visualization tab to the GUI.
    
    Args:
        gui_instance: LLaDAGUI instance
        
    Returns:
        True if tab was added successfully, False otherwise
    """
    if not HAS_MEMORY_INTERFACE:
        logger.error("Memory interface not available, cannot add visualization tab")
        return False

    # Get memory interface
    memory_interface = get_memory_interface()
    if memory_interface is None:
        # Create a new one
        memory_interface = MCPTitanMemoryInterface()
        logger.info("Created new memory interface")

    # Check if tab already exists
    if hasattr(gui_instance, 'memory_viz'):
        # Tab already exists, no need to create it again
        logger.info("Memory visualization tab already exists")
        return True

    # Create visualization widget
    try:
        memory_viz = MemoryVisualizationWidget(memory_interface)
        gui_instance.memory_viz = memory_viz
        logger.info("Created memory visualization widget")
    except Exception as e:
        logger.error(f"Error creating memory visualization widget: {e}")
        QMessageBox.warning(
            gui_instance,
            "Memory Visualization Error",
            f"Could not create memory visualization: {str(e)}"
        )
        return False

    # Add to tab widget if available
    if hasattr(gui_instance, 'tab_widget') and isinstance(gui_instance.tab_widget, QTabWidget):
        try:
            memory_tab_index = gui_instance.tab_widget.addTab(memory_viz, "Memory")
            logger.info(f"Added memory tab at index {memory_tab_index}")

            # Schedule memory connection attempt
            QTimer.singleShot(1000, lambda: _delayed_memory_connection(gui_instance))

            return True
        except Exception as e:
            logger.error(f"Error adding memory tab to tab widget: {e}")
            return False
    else:
        logger.error("Tab widget not found in GUI instance")
        return False


def _delayed_memory_connection(gui_instance):
    """Attempt to connect to memory server after a delay.
    
    Args:
        gui_instance: LLaDAGUI instance
    """
    if hasattr(gui_instance, 'memory_viz') and hasattr(gui_instance.memory_viz, 'connect_memory'):
        try:
            logger.info("Attempting delayed memory connection")
            connected = gui_instance.memory_viz.connect_memory()

            # Enable training if connected
            if connected and hasattr(gui_instance, 'memory_viz'):
                logger.info("Memory connected, enabling training")
                gui_instance.memory_viz.train_btn.setEnabled(True)

        except Exception as e:
            logger.error(f"Error in delayed memory connection: {e}")


def handle_memory_integration(gui_instance, prompt, config):
    """Handle memory integration for generation.
    
    Args:
        gui_instance: LLaDAGUI instance
        prompt: Input prompt
        config: Generation configuration
        
    Returns:
        True if memory integration was handled, False otherwise
    """
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
    # Try Titan memory integration first (if available)
    if HAS_TITAN_MEMORY:
        logger.info("Using Titan memory integration")
        try:
            success = titan_handle_memory_integration(gui_instance, prompt, config)
            if success:
                return True
            logger.warning("Titan memory integration failed, falling back to legacy")
        except Exception as e:
            logger.error(f"Error in Titan memory integration: {e}")
            logger.warning("Falling back to legacy memory integration")

    # Fall back to legacy memory interface
    if not HAS_MEMORY_INTERFACE:
        logger.error("No memory interface available for integration")
        return False

    # Get legacy memory interface
    memory_interface = get_memory_interface()

    # Check if memory interface exists and is initialized
    if memory_interface is None or not memory_interface.initialized:
        # Try to initialize
        logger.info("Memory interface not initialized, attempting to initialize")
        success = initialize_memory(start_server=True)
        if not success:
            # Try Python fallback as a last resort
            logger.warning("Standard initialization failed, trying Python fallback")
            if try_python_server_fallback():
                # Try to initialize again with the fallback server
                time.sleep(1)  # Brief pause to ensure server is ready
                success = initialize_memory(start_server=False)  # Don't try to start server again

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

    try:
        # Create memory-guided diffusion worker
        memory_weight = config.get('memory_weight', 0.3)  # Default weight
        logger.info(f"Creating legacy memory-guided worker with weight {memory_weight}")

        # Create worker
        worker = MemoryGuidanceDiffusionWorker(prompt, config, memory_interface)
        gui_instance.worker = worker

        # Connect signals
        worker.progress.connect(gui_instance.update_progress)

        # Check if update_visualization exists (for compatibility with different GUI versions)
        if hasattr(gui_instance, 'update_visualization'):
            worker.step_update.connect(gui_instance.update_visualization)
        else:
            logger.warning("GUI instance does not have update_visualization method")

        # Check if memory_viz exists
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
        logger.info("Legacy memory-guided worker started successfully")

        return True
    except Exception as e:
        logger.error(f"Error creating or starting legacy memory-guided worker: {e}")
        QMessageBox.warning(
            gui_instance,
            "Memory Integration Error",
            f"Error integrating memory: {str(e)}\n\nFalling back to standard generation."
        )
        return False
