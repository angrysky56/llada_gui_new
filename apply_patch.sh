#!/bin/bash

# Script to apply the extreme mode patch to LLaDA GUI

# Check if we're in the right directory
if [ ! -f "llada_gui.py" ]; then
  echo "Error: llada_gui.py not found in the current directory."
  echo "Please run this script from the LLaDA GUI repository root."
  exit 1
fi

# Create the patch file
echo "Creating patch file..."
cat > extreme_mode_patch.py << 'EOL'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch for removing hard limits from extreme mode in LLaDA GUI.

This script modifies the LLaDA GUI to replace hard limits with warnings
when using extreme memory mode.
"""

import os
import sys
import re
from pathlib import Path

def patch_gui_file():
    """
    Patch the llada_gui.py file to replace hard limits with warnings.
    """
    # Find the GUI file
    gui_path = Path("llada_gui.py")
    if not os.path.exists(gui_path):
        print(f"Error: Could not find GUI file at {gui_path}")
        return False
    
    print(f"Patching GUI file at {gui_path}")
    
    # Read the file content
    with open(gui_path, "r") as f:
        content = f.read()
    
    # Create a backup
    backup_path = gui_path.with_suffix(".py.extreme_backup")
    if not backup_path.exists():
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup at {backup_path}")
    
    # Find the extreme mode section
    extreme_mode_pattern = re.compile(
        r"# Add warning for extreme mode.*?"
        r"if self\.extreme_mode\.isChecked\(\):.*?"
        r"(\s+# Reduce parameters for extreme mode.*?"
        r"\s+if config\['gen_length'\] > 64:.*?"
        r"\s+self\.gen_length_spin\.setValue\(64\).*?"
        r"\s+config\['gen_length'\] = 64.*?"
        r"\s+if config\['steps'\] > 64:.*?"
        r"\s+self\.steps_spin\.setValue\(64\).*?"
        r"\s+config\['steps'\] = 64.*?"
        r"\s+if config\['block_length'\] > 32:.*?"
        r"\s+self\.block_length_spin\.setValue\(32\).*?"
        r"\s+config\['block_length'\] = 32)",
        re.DOTALL
    )
    
    # Replacement code with warnings but no forced limits
    replacement = """        # Show warnings for extreme mode without forcing limits
        warnings = []
        
        if config['gen_length'] > 64:
            warnings.append(f"- Generation Length: {config['gen_length']} (recommended: 64 or less)")
            
        if config['steps'] > 64:
            warnings.append(f"- Sampling Steps: {config['steps']} (recommended: 64 or less)")
            
        if config['block_length'] > 32:
            warnings.append(f"- Block Length: {config['block_length']} (recommended: 32 or less)")
        
        if warnings:
            result = QMessageBox.warning(
                self,
                "Extreme Memory Mode Warning",
                "You're using parameters that exceed the recommended limits for Extreme Memory Mode:\\n\\n" +
                "\\n".join(warnings) + 
                "\\n\\nThese settings may cause out-of-memory errors on GPUs with limited VRAM (8-12GB)." +
                "\\n\\nDo you want to continue with these settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if result == QMessageBox.StandardButton.No:
                # User chose to use recommended values
                if config['gen_length'] > 64:
                    self.gen_length_spin.setValue(64)
                    config['gen_length'] = 64
                    
                if config['steps'] > 64:
                    self.steps_spin.setValue(64)
                    config['steps'] = 64
                    
                if config['block_length'] > 32:
                    self.block_length_spin.setValue(32)
                    config['block_length'] = 32"""
    
    # Apply the patch
    if extreme_mode_pattern.search(content):
        new_content = extreme_mode_pattern.sub(replacement, content)
        
        # Write the updated content
        with open(gui_path, "w") as f:
            f.write(new_content)
        
        print("Successfully patched the GUI file to replace hard limits with warnings.")
        return True
    else:
        print("Could not find the extreme mode section in the GUI file.")
        return False

def main():
    """Main function."""
    print("LLaDA GUI Extreme Mode Patch")
    print("============================")
    print("This patch replaces hard limits with warnings when using extreme memory mode.")
    
    success = patch_gui_file()
    
    if success:
        print("\nPatch applied successfully!")
        print("You can now use higher step counts with extreme memory mode if your GPU can handle it.")
        print("If you encounter out-of-memory errors, try reducing the parameters.")
    else:
        print("\nFailed to apply the patch.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
EOL

# Make the script executable
chmod +x extreme_mode_patch.py

# Run the patch script
echo "Applying patch..."
python extreme_mode_patch.py

# Check the result
if [ $? -eq 0 ]; then
  echo "Patch applied successfully!"
  echo "You can now run LLaDA GUI with extreme mode and higher step counts."
else
  echo "Failed to apply the patch."
  exit 1
fi

echo "Done!"
