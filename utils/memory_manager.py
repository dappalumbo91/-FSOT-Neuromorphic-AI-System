"""Memory manager wrapper for main directory"""
import sys
import os
from pathlib import Path
import importlib.util

# Add clean system to path if not already there
clean_system_path = Path(__file__).parent.parent / "FSOT_Clean_System"
clean_system_str = str(clean_system_path)

if clean_system_str not in sys.path:
    sys.path.insert(0, clean_system_str)

# Load the module directly using importlib to avoid circular imports
module_path = clean_system_path / "utils" / "memory_manager.py"
spec = importlib.util.spec_from_file_location("clean_memory_manager", module_path)

if spec is not None and spec.loader is not None:
    mm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm_module)
    # Get the memory_manager instance
    memory_manager = mm_module.memory_manager
else:
    raise ImportError("Could not load memory_manager module")

__all__ = ['memory_manager']
