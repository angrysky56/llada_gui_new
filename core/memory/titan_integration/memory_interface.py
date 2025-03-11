#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCP Titan Memory Interface.

This module provides the interface to the MCP Titan Memory system via HTTP API.
"""

import time
import logging
import numpy as np
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='memory_server.log'
)
logger = logging.getLogger("memory_interface")

class MCPTitanMemoryInterface:
    """Interface to the MCP Titan Memory system via HTTP API."""

    def __init__(self, api_url="http://localhost:3000"):
        """Initialize the memory interface.
        
        Args:
            api_url: URL of the MCP Titan Memory API
        """
        self.api_url = api_url
        self.memory_state = None
        self.input_dim = 64  # Default, will be updated from model
        self.memory_dim = 64  # Default, will be updated from model
        self.initialized = False
        self.connection_timeout = 3  # Seconds
        self.connection_retries = 2  # Number of retries
        self._api_endpoints = {
            'status': ['/status', '/api/status'],
            'init': ['/init', '/api/init_model'],
            'forward': ['/forward', '/api/forward_pass'],
            'train': ['/trainStep', '/api/train_step'],
            'save': ['/save', '/api/save_model'],
            'load': ['/load', '/api/load_model'],
            'reset': ['/reset', '/api/reset_memory'],
            'memory': ['/memory', '/api/memory_state']
        }

    def initialize(self, input_dim=64, memory_dim=64):
        """Initialize the memory model.
        
        Args:
            input_dim: Dimension of input vectors
            memory_dim: Dimension of memory vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if server is running by trying multiple endpoints
            server_running = False
            connection_error = None

            for endpoint in self._api_endpoints['status']:
                try:
                    for retry in range(self.connection_retries):
                        try:
                            response = requests.get(
                                f"{self.api_url}{endpoint}",
                                timeout=self.connection_timeout
                            )
                            if response.status_code == 200:
                                server_running = True
                                logger.info(f"Memory server is running (detected via {endpoint})")
                                break
                        except requests.exceptions.RequestException as e:
                            connection_error = e
                            logger.debug(f"Connection attempt {retry + 1} to {endpoint} failed: {e}")
                            time.sleep(0.5)  # Brief delay between retries

                    if server_running:
                        break
                except Exception as e:
                    logger.debug(f"Error checking endpoint {endpoint}: {e}")

            if not server_running:
                logger.warning(f"Could not connect to memory server: {connection_error}")
                return False

            # Initialize the model - try both init endpoints
            init_success = False
            for endpoint in self._api_endpoints['init']:
                try:
                    response = requests.post(
                        f"{self.api_url}{endpoint}",
                        json={"inputDim": input_dim, "outputDim": memory_dim},
                        timeout=self.connection_timeout
                    )
                    if response.status_code == 200:
                        init_success = True
                        logger.info(f"Model initialized via {endpoint}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Init request failed on {endpoint}: {e}")

            if not init_success:
                logger.error("Failed to initialize model on any endpoint")
                return False

            self.input_dim = input_dim
            self.memory_dim = memory_dim

            # Initialize memory state to zeros
            self.memory_state = np.zeros(memory_dim)
            self.initialized = True

            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            return False

    def forward_pass(self, input_vector):
        """Run forward pass through the memory model.
        
        Args:
            input_vector: Input vector of shape [input_dim]
            
        Returns:
            dict with predicted, newMemory, and surprise
        """
        if not self.initialized:
            logger.warning("Memory not initialized. Attempting to initialize now.")
            if not self.initialize():
                logger.error("Failed to initialize memory")
                # Return default values
                return {
                    "predicted": np.zeros(self.input_dim).tolist(),
                    "newMemory": np.zeros(self.memory_dim).tolist(),
                    "surprise": 0.0
                }

        # Prepare request data
        request_data = {
            "x": input_vector.tolist() if isinstance(input_vector, np.ndarray) else input_vector,
            "memoryState": self.memory_state.tolist() if isinstance(self.memory_state,
                                                                    np.ndarray) else self.memory_state
        }

        # Try each forward endpoint
        for endpoint in self._api_endpoints['forward']:
            try:
                for retry in range(self.connection_retries):
                    try:
                        response = requests.post(
                            f"{self.api_url}{endpoint}",
                            json=request_data,
                            timeout=self.connection_timeout
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Update memory state
                            if "newMemory" in result:
                                self.memory_state = np.array(result["newMemory"])
                            elif "memory" in result:
                                # In case the API returns 'memory' instead of 'newMemory'
                                self.memory_state = np.array(result["memory"])

                            logger.debug(f"Forward pass successful via {endpoint}")
                            return result
                    except Exception as e:
                        logger.debug(f"Forward pass retry {retry + 1} failed on {endpoint}: {e}")
                        if retry == self.connection_retries - 1:
                            # Last retry failed, continue to next endpoint
                            logger.warning(f"All retries failed for endpoint {endpoint}")
            except Exception as e:
                logger.debug(f"Error in forward pass with endpoint {endpoint}: {e}")

        # All endpoints failed
        logger.error("All forward pass endpoints failed")

        # If memory is initialized but server not responding, consider it as uninitialized for next time
        self.initialized = False

        # Return default values
        return {
            "predicted": np.zeros(self.input_dim).tolist(),
            "newMemory": self.memory_state.tolist() if isinstance(self.memory_state, np.ndarray) else self.memory_state,
            "surprise": 0.0
        }

    def get_memory_state(self):
        """Get the current memory state."""
        return self.memory_state if self.initialized else np.zeros(self.memory_dim)

    def reset(self):
        """Reset the memory state."""
        if self.initialized:
            self.memory_state = np.zeros(self.memory_dim)
