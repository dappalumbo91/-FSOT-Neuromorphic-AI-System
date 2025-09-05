"""Web interface wrapper for main directory"""
import sys
import os
from pathlib import Path

# Add clean system to path if not already there
clean_system_path = Path(__file__).parent.parent / "FSOT_Clean_System"
clean_system_str = str(clean_system_path)

if clean_system_str not in sys.path:
    sys.path.insert(0, clean_system_str)

# Change to the clean system directory temporarily
original_cwd = os.getcwd()
try:
    os.chdir(clean_system_str)
    # Import the class directly
    from interfaces import web_interface as wi_module
    WebInterface = wi_module.WebInterface
finally:
    os.chdir(original_cwd)

__all__ = ['WebInterface']
