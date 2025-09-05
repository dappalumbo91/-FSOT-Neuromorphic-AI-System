#!/usr/bin/env python3
"""
Environment Validation Script for FSOT 2.0 Clean System
Checks all dependencies and reports system readiness
"""

import sys
import importlib
from typing import List, Tuple

def check_dependency(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Check if a dependency is available"""
    try:
        importlib.import_module(module_name)
        return True, f"✅ {module_name} - {description}"
    except ImportError as e:
        return False, f"❌ {module_name} - {description} - Error: {e}"

def validate_environment() -> bool:
    """Validate the entire environment"""
    print("🧠⚡ FSOT 2.0 Environment Validation")
    print("=" * 50)
    
    # Core dependencies
    core_deps = [
        ("asyncio", "Async support (built-in)"),
        ("json", "JSON support (built-in)"),
        ("logging", "Logging support (built-in)"),
        ("pathlib", "Path utilities (built-in)"),
    ]
    
    # Optional dependencies
    optional_deps = [
        ("numpy", "Scientific computing"),
        ("torch", "Deep learning framework"),
        ("fastapi", "Web framework"),
        ("uvicorn", "ASGI server"),
        ("pytest", "Testing framework"),
        ("yaml", "YAML configuration"),
    ]
    
    # Project modules
    project_modules = [
        ("core.fsot_engine", "FSOT mathematical engine"),
        ("core.neural_signal", "Neural communication"),
        ("core.consciousness", "Consciousness monitoring"),
        ("brain.brain_orchestrator", "Brain coordination"),
        ("brain.frontal_cortex", "Frontal cortex module"),
        ("config.settings", "Configuration management"),
        ("interfaces.cli_interface", "CLI interface"),
        ("interfaces.web_interface", "Web interface"),
    ]
    
    all_good = True
    
    print("\n📦 Core Dependencies:")
    for module, desc in core_deps:
        success, msg = check_dependency(module, desc)
        print(f"  {msg}")
        if not success:
            all_good = False
    
    print("\n🔧 Optional Dependencies:")
    for module, desc in optional_deps:
        success, msg = check_dependency(module, desc)
        print(f"  {msg}")
        # Optional deps don't fail validation
    
    print("\n🧠 Project Modules:")
    for module, desc in project_modules:
        success, msg = check_dependency(module, desc)
        print(f"  {msg}")
        if not success:
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ Environment validation PASSED")
        print("🚀 FSOT 2.0 System is ready to run!")
    else:
        print("❌ Environment validation FAILED")
        print("⚠️  Please fix the issues above before running the system")
    
    print(f"\n🐍 Python Version: {sys.version}")
    print(f"📍 Python Path: {sys.executable}")
    
    return all_good

if __name__ == "__main__":
    validate_environment()
