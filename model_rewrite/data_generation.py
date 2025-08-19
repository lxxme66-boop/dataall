#!/usr/bin/env python3
"""Wrapper for data generation that handles multiple interfaces"""

import sys
import os
import subprocess

# Check if this is being called with the old interface
if '--input' in sys.argv:
    # Use the simple rewrite script for the old interface
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simple_script = os.path.join(script_dir, 'simple_rewrite.py')
    
    # Replace argv[0] with the simple script path
    new_argv = [sys.executable, simple_script] + sys.argv[1:]
    
    # Execute the simple script
    result = subprocess.run(new_argv)
    sys.exit(result.returncode)
else:
    # Use the original data_generation script for the new interface
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_script = os.path.join(script_dir, 'data_generation_original.py')
    
    # Replace argv[0] with the original script path
    new_argv = [sys.executable, original_script] + sys.argv[1:]
    
    # Execute the original script
    result = subprocess.run(new_argv)
    sys.exit(result.returncode)