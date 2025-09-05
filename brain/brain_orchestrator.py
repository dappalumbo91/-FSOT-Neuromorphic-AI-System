"""Brain orchestrator wrapper for main directory"""
import sys
import os
from pathlib import Path

def get_brain_orchestrator():
    """Get BrainOrchestrator class from FSOT_Clean_System."""
    # Add clean system to path if not already there
    clean_system_path = Path(__file__).parent.parent / "FSOT_Clean_System"
    clean_system_str = str(clean_system_path)

    if clean_system_str not in sys.path:
        sys.path.insert(0, clean_system_str)

    # Change to the clean system directory temporarily
    original_cwd = os.getcwd()
    try:
        os.chdir(clean_system_str)
        # Now import normally
        from brain import brain_orchestrator as bo_module
        return bo_module.BrainOrchestrator
    finally:
        os.chdir(original_cwd)

# Use lazy loading to avoid circular imports
BrainOrchestrator = None

def __getattr__(name):
    """Lazy loading for BrainOrchestrator."""
    global BrainOrchestrator
    if name == 'BrainOrchestrator':
        if BrainOrchestrator is None:
            BrainOrchestrator = get_brain_orchestrator()
        return BrainOrchestrator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['BrainOrchestrator']
