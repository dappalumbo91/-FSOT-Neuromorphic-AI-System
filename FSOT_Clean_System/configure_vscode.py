#!/usr/bin/env python3
"""
Python Interpreter Selection Helper for VS Code
This script helps VS Code find the correct Python interpreter with all dependencies
"""

import sys
import subprocess
import json
from pathlib import Path

def get_python_info():
    """Get current Python interpreter information"""
    info = {
        "executable": sys.executable,
        "version": sys.version,
        "paths": sys.path[:10],  # First 10 paths for brevity
    }
    
    # Check key dependencies
    dependencies = ["fastapi", "uvicorn", "pytest", "numpy", "torch"]
    available_deps = {}
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            available_deps[dep] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
                "location": getattr(module, "__file__", "unknown")
            }
        except ImportError:
            available_deps[dep] = {"available": False}
    
    info["dependencies"] = available_deps
    return info

def create_vscode_python_config():
    """Create VS Code Python configuration"""
    python_path = sys.executable
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Create launch.json for debugging
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "FSOT 2.0 Main",
                "type": "python",
                "request": "launch", 
                "program": "${workspaceFolder}/main.py",
                "console": "integratedTerminal",
                "python": python_path,
                "cwd": "${workspaceFolder}"
            },
            {
                "name": "FSOT 2.0 Startup",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/start.py", 
                "console": "integratedTerminal",
                "python": python_path,
                "cwd": "${workspaceFolder}"
            },
            {
                "name": "Environment Validation",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/validate_environment.py",
                "console": "integratedTerminal", 
                "python": python_path,
                "cwd": "${workspaceFolder}"
            }
        ]
    }
    
    with open(vscode_dir / "launch.json", "w") as f:
        json.dump(launch_config, f, indent=4)
    
    print(f"✅ Created VS Code configuration with Python: {python_path}")

def main():
    """Main function"""
    print("🐍 Python Interpreter Information for VS Code")
    print("=" * 60)
    
    info = get_python_info()
    
    print(f"📍 Python Executable: {info['executable']}")
    print(f"🔢 Python Version: {info['version'].split()[0]}")
    
    print("\n📦 Dependencies Status:")
    for dep, details in info["dependencies"].items():
        if details["available"]:
            version = details.get("version", "unknown")
            print(f"  ✅ {dep:<12} v{version}")
        else:
            print(f"  ❌ {dep:<12} not available")
    
    print(f"\n📁 Module Search Paths (top 5):")
    for i, path in enumerate(info["paths"][:5], 1):
        print(f"  {i}. {path}")
    
    # Create VS Code configuration
    create_vscode_python_config()
    
    print(f"\n🎯 Recommended VS Code Settings:")
    print(f'   "python.defaultInterpreterPath": "{info["executable"]}",')
    print(f'   "python.pythonPath": "{info["executable"]}",')
    
    print(f"\n✅ Configuration complete!")
    print(f"💡 To apply in VS Code:")
    print(f"   1. Press Ctrl+Shift+P")
    print(f"   2. Type 'Python: Select Interpreter'")
    print(f"   3. Choose: {info['executable']}")
    print(f"   4. Reload VS Code window (Ctrl+Shift+P -> 'Developer: Reload Window')")

if __name__ == "__main__":
    main()
