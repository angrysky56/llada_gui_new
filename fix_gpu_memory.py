#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fix GPU memory offloading issues in the LLaDA memory integration system.

This script applies patches to the memory_integration_auto.py file to improve
GPU memory management and fix any API endpoint mismatches.

Usage:
    python fix_gpu_memory.py
    
    # Or to apply fixes and launch the application:
    python fix_gpu_memory.py --launch
"""

import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to apply fixes and optionally launch the application."""
    logger.info("Applying GPU memory offloading fixes...")
    
    # Import the memory patch module
    try:
        # First make sure memory_integration_auto is installed
        import memory_integration_auto
        
        # Write the memory patch module if it doesn't exist
        patch_path = os.path.join(os.path.dirname(__file__), "memory_patch.py")
        if not os.path.exists(patch_path):
            from memory_patch_code import patch_code
            with open(patch_path, "w") as f:
                f.write(patch_code)
            logger.info(f"Created memory patch at: {patch_path}")
        
        # Import the patch module
        import memory_patch
        
        # Apply the fixes
        success = memory_patch.apply_gpu_offloading_fixes()
        
        if success:
            logger.info("GPU memory offloading fixes applied successfully!")
        else:
            logger.error("Failed to apply GPU memory offloading fixes")
            return 1
        
        # Optionally launch the application
        if len(sys.argv) > 1 and sys.argv[1] == "--launch":
            logger.info("Launching LLaDA with memory integration and GPU fixes...")
            from memory_integration_auto import main as launch_app
            launch_app()
        
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

# Include the patch code as a string to avoid having to create a separate file
memory_patch_code = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Patch file to fix GPU offloading issues in memory_integration_auto.py
\"\"\"

import os
import sys
import logging
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_gpu_offloading_fixes():
    \"\"\"Apply fixes to the memory integration system for better GPU offloading.\"\"\"
    try:
        # 1. Import the memory interface
        from memory_integration_auto import MCPTitanMemoryInterface
        
        # 2. Fix API endpoint paths
        original_init = MCPTitanMemoryInterface.initialize
        
        def patched_initialize(self, input_dim=64, memory_dim=64):
            \"\"\"Patched initialize method with corrected endpoint.\"\"\"
            logger.info("Using patched initialize method with improved GPU handling")
            # First, make sure the server is running
            if not self.server_manager.check_server():
                if not self.server_manager.start_server():
                    logger.error("Failed to start memory server")
                    return False
            
            try:
                # Use the correct endpoint with improved error handling
                import requests
                
                # Free up GPU memory before making request
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try different endpoint paths to ensure compatibility
                endpoints = [
                    f"{self.api_url}/init",
                    f"{self.api_url}/init_model"
                ]
                
                success = False
                for endpoint in endpoints:
                    try:
                        logger.info(f"Trying to initialize with endpoint: {endpoint}")
                        response = requests.post(
                            endpoint, 
                            json={"inputDim": input_dim, "outputDim": memory_dim},
                            timeout=5  # 5 second timeout
                        )
                        response.raise_for_status()
                        success = True
                        logger.info(f"Successfully initialized with endpoint: {endpoint}")
                        break
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed with endpoint {endpoint}: {e}")
                
                if not success:
                    logger.error("All initialization endpoints failed")
                    return False
                
                self.input_dim = input_dim
                self.memory_dim = memory_dim
                
                # Initialize memory state to zeros
                self.memory_state = torch.zeros(memory_dim, dtype=torch.float32)
                if torch.cuda.is_available():
                    # Keep on CPU to save GPU memory
                    self.memory_state = self.memory_state.cpu().numpy()
                else:
                    self.memory_state = self.memory_state.numpy()
                
                self.initialized = True
                
                return True
            except Exception as e:
                logger.error(f"Failed to initialize memory: {str(e)}")
                return False
        
        # 3. Fix forward_pass method for better GPU handling
        original_forward = MCPTitanMemoryInterface.forward_pass
        
        def patched_forward_pass(self, input_vector):
            \"\"\"Patched forward_pass method with improved GPU memory handling.\"\"\"
            if not self.initialized:
                raise ValueError("Memory not initialized. Call initialize() first.")
                
            try:
                # Convert input to CPU numpy array if it's a GPU tensor
                if isinstance(input_vector, torch.Tensor) and input_vector.device.type == "cuda":
                    input_vector = input_vector.detach().cpu().numpy()
                elif isinstance(input_vector, torch.Tensor):
                    input_vector = input_vector.detach().numpy()
                
                # Use the forward endpoint with proper error handling
                import requests
                
                # Free some GPU memory before making request
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Try different endpoint paths to ensure compatibility
                endpoints = [
                    f"{self.api_url}/forward",
                    f"{self.api_url}/forward_pass"
                ]
                
                success = False
                result = None
                
                for endpoint in endpoints:
                    try:
                        response = requests.post(
                            endpoint,
                            json={
                                "x": input_vector.tolist() if isinstance(input_vector, np.ndarray) else input_vector,
                            },
                            timeout=5  # 5 second timeout
                        )
                        response.raise_for_status()
                        result = response.json()
                        success = True
                        break
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed with endpoint {endpoint}: {e}")
                
                if not success or result is None:
                    logger.error("All forward pass endpoints failed")
                    # Return default values
                    return {
                        "predicted": np.zeros(self.input_dim).tolist(),
                        "newMemory": self.memory_state.tolist() if isinstance(self.memory_state, np.ndarray) else self.memory_state,
                        "surprise": 0.0
                    }
                
                # Update memory state using numpy to save GPU memory
                self.memory_state = np.array(result.get("memory", result.get("newMemory", [])))
                
                return {
                    "predicted": result.get("predicted", []),
                    "newMemory": result.get("memory", result.get("newMemory", [])),
                    "surprise": result.get("surprise", 0.0)
                }
            except Exception as e:
                logger.error(f"Memory forward pass error: {str(e)}")
                # Return default values
                return {
                    "predicted": np.zeros(self.input_dim).tolist(),
                    "newMemory": self.memory_state.tolist() if isinstance(self.memory_state, np.ndarray) else self.memory_state,
                    "surprise": 0.0
                }
        
        # 4. Create improved server manager for better thread management
        from memory_integration_auto import MemoryServerManager
        
        original_start_server = MemoryServerManager.start_server
        
        def patched_start_server(self):
            \"\"\"Patched start_server method with better thread and memory management.\"\"\"
            if self.is_running:
                logger.info("Memory server is already running")
                return True
            
            try:
                logger.info("Starting memory server thread with improved GPU management")
                
                # Free GPU memory before starting server
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Parse the host and port from the API URL
                parts = self.api_url.split('://')
                if len(parts) > 1:
                    host_port = parts[1].split(':')
                    host = host_port[0]
                    if len(host_port) > 1:
                        port = int(host_port[1].split('/')[0])
                    else:
                        port = 80
                else:
                    host = "localhost"
                    port = 3000
                
                # Import here to avoid circular imports
                import threading
                from memory_server.server import start_server
                
                # Start server in a thread with better exception handling
                def run_server():
                    try:
                        logger.info(f"Starting memory server on {host}:{port}")
                        start_server(host, port)
                    except Exception as e:
                        logger.error(f"Error in memory server thread: {str(e)}")
                
                self.server_thread = threading.Thread(
                    target=run_server,
                    daemon=True
                )
                self.server_thread.start()
                
                # Wait for server to start (up to 30 seconds)
                import time
                import requests
                
                start_time = time.time()
                max_wait = 30
                is_ready = False
                
                logger.info("Waiting for memory server to initialize...")
                while time.time() - start_time < max_wait:
                    try:
                        # Check if server is running by making a status request
                        response = requests.get(f"{self.api_url}/status", timeout=1)
                        if response.status_code == 200:
                            is_ready = True
                            break
                    except requests.exceptions.RequestException:
                        # Server not ready yet, wait and retry
                        time.sleep(1)
                
                if is_ready:
                    logger.info("Memory server started successfully")
                    self.is_running = True
                    return True
                else:
                    logger.error(f"Memory server failed to start in {max_wait} seconds")
                    self.stop_server()
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to start memory server: {str(e)}")
                self.stop_server()
                return False
        
        # 5. Apply all the patches
        MCPTitanMemoryInterface.initialize = patched_initialize
        MCPTitanMemoryInterface.forward_pass = patched_forward_pass
        MemoryServerManager.start_server = patched_start_server
        
        logger.info("Successfully applied GPU offloading fixes to memory integration")
        return True
    
    except Exception as e:
        logger.error(f"Failed to apply GPU offloading fixes: {str(e)}")
        return False

# Apply fixes when imported
if __name__ != "__main__":
    apply_gpu_offloading_fixes()

# Run as script
if __name__ == "__main__":
    success = apply_gpu_offloading_fixes()
    print(f"Applied GPU offloading fixes: {'Success' if success else 'Failed'}")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--launch":
        # Launch the memory integration with fixes
        from memory_integration_auto import main
        main()
"""

# Write the memory patch code to a module
def write_memory_patch():
    """Write the memory patch code to a module."""
    patch_path = os.path.join(os.path.dirname(__file__), "memory_patch.py")
    with open(patch_path, "w") as f:
        f.write(memory_patch_code)
    return patch_path

# Define patch code for external use
patch_code = memory_patch_code

if __name__ == "__main__":
    # Write the memory patch module
    patch_path = write_memory_patch()
    logger.info(f"Created memory patch at: {patch_path}")
    
    # Run the main function
    sys.exit(main())
