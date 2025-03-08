#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct patcher for memory widget in LLaDA GUI.

This script directly modifies the memory visualization widget to fix connection issues.
"""

import logging
import os
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_patch")


def patch_memory_widget():
    """Patch the memory visualization widget."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find memory integration file
    memory_file = os.path.join(script_dir, "core", "memory", "memory_integration.py")
    if not os.path.exists(memory_file):
        logger.error(f"Memory integration file not found: {memory_file}")
        return False

    # Read file content
    with open(memory_file, "r") as f:
        content = f.read()

    # Look for MemoryVisualizationWidget connect_memory method
    connect_pattern = r'def connect_memory\(self\):.*?return True'
    connect_match = re.search(connect_pattern, content, re.DOTALL)

    if not connect_match:
        logger.error("Could not find connect_memory method in MemoryVisualizationWidget")
        return False

    # Get original connect method
    original_connect = connect_match.group(0)

    # Create new connect method with direct memory server start
    new_connect = """def connect_memory(self):
        \"\"\"Connect or disconnect the memory system.\"\"\"
        if self.status_label.text() == "Memory System: Connected":
            # Currently connected, disconnect
            self.update_memory_status(False)
            self.reset_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            
            # Try to properly stop the server if we started it
            try:
                # Use server manager if available
                if SERVER_MANAGER_AVAILABLE:
                    server_manager = get_server_manager()
                    if server_manager:
                        server_manager.stop()
                else:
                    # Manually kill processes
                    import subprocess
                    subprocess.run(['pkill', '-f', 'server.py'], check=False)
                    subprocess.run(['pkill', '-f', 'server.js'], check=False)
            except Exception as e:
                print(f"Warning: Error stopping memory server: {e}")
                
            return False
        else:
            # Kill any existing processes first
            try:
                import subprocess
                import os
                import sys
                import time
                
                # Kill any process using port 3000
                subprocess.run(['pkill', '-f', 'server.py'], check=False)
                subprocess.run(['pkill', '-f', 'server.js'], check=False)
                subprocess.run(['pkill', '-f', 'memory_server'], check=False)
                
                try:
                    # Find processes using port 3000 with lsof
                    result = subprocess.run(['lsof', '-i', ':3000', '-t'], 
                                           capture_output=True, text=True)
                    if result.stdout.strip():
                        pids = result.stdout.strip().split('\\n')
                        for pid in pids:
                            if pid.strip():
                                print(f"Killing process {pid} using port 3000")
                                subprocess.run(['kill', '-9', pid.strip()], check=False)
                except Exception as e:
                    print(f"Error finding processes with lsof: {e}")
                
                # Wait for processes to terminate
                time.sleep(1)
                
                # Start memory server directly from this process
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                direct_script = os.path.join(project_root, 'direct_memory_fix.py')
                
                if os.path.isfile(direct_script):
                    print(f"Starting memory server directly: {direct_script}")
                    # Start in background
                    process = subprocess.Popen(
                        [sys.executable, direct_script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True
                    )
                    
                    # Wait for server to start
                    print("Waiting for memory server to start...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"Error setting up memory server: {e}")
            
            # Initialize vector database
            if VECTOR_DB_AVAILABLE:
                try:
                    self.vector_db = get_vector_db()
                    print("Vector database initialized successfully")
                except Exception as e:
                    print(f"Error initializing vector database: {e}")
                    self.vector_db = None
                    
            # Try to connect
            if initialize_memory(True):  # Try to start server if needed
                self.update_memory_status(True)
                self.reset_btn.setEnabled(True)
                self.save_btn.setEnabled(True)
                self.load_btn.setEnabled(True)
                # Display initial memory state
                self.display_memory_state(self.memory_interface.get_memory_state())
                return True"""

    # Replace the connect method
    new_content = content.replace(original_connect, new_connect)

    # Write back to file
    with open(memory_file, "w") as f:
        f.write(new_content)

    logger.info("Successfully patched memory widget connect_memory method")
    return True


def main():
    """Main function."""
    # Patch memory widget
    if not patch_memory_widget():
        logger.error("Failed to patch memory widget")
        return 1

    logger.info("Memory widget successfully patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
