#!/usr/bin/env python3
"""
Pylance Import Debug Helper
Diagnoses and fixes import resolution issues for VS Code Pylance
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def test_imports():
    """Test all the problematic imports that Pylance is complaining about"""
    print("🔍 Testing Import Resolution")
    print("=" * 50)
    
    imports_to_test = [
        ("fastapi", "FastAPI web framework"),
        ("fastapi.responses", "FastAPI responses module"),
        ("fastapi.staticfiles", "FastAPI static files module"),
        ("uvicorn", "ASGI server"),
        ("pytest", "Testing framework")
    ]
    
    results = {}
    
    for module_name, description in imports_to_test:
        try:
            # Try to import the module
            if "." in module_name:
                # Handle submodule imports
                parent_module = module_name.split(".")[0]
                submodule = module_name.split(".")[1]
                parent = importlib.import_module(parent_module)
                module = getattr(parent, submodule, None)
                if module is None:
                    module = importlib.import_module(module_name)
            else:
                module = importlib.import_module(module_name)
            
            # Get module information
            module_file = getattr(module, "__file__", "Built-in")
            module_version = getattr(module, "__version__", "Unknown")
            
            results[module_name] = {
                "status": "✅ Success",
                "file": module_file,
                "version": module_version,
                "description": description
            }
            
        except ImportError as e:
            results[module_name] = {
                "status": "❌ Failed", 
                "error": str(e),
                "description": description
            }
        except Exception as e:
            results[module_name] = {
                "status": "⚠️  Error",
                "error": str(e),
                "description": description
            }
    
    # Print results
    for module_name, result in results.items():
        print(f"\n📦 {module_name} - {result['description']}")
        print(f"   Status: {result['status']}")
        if "file" in result:
            print(f"   Location: {result['file']}")
            print(f"   Version: {result['version']}")
        elif "error" in result:
            print(f"   Error: {result['error']}")
    
    return results

def generate_pylance_config():
    """Generate configuration to help Pylance"""
    print(f"\n🔧 Generating Pylance Configuration")
    print("=" * 50)
    
    # Get site-packages location
    try:
        import site
        site_packages = site.getsitepackages()
        user_site = site.getusersitepackages()
        
        print(f"📁 Site packages locations:")
        for i, path in enumerate(site_packages, 1):
            print(f"   {i}. {path}")
        print(f"   User site: {user_site}")
        
        # Create pyrightconfig.json for better Pylance support
        pyright_config = {
            "include": [
                "core",
                "brain", 
                "config",
                "interfaces",
                "utils",
                "tests"
            ],
            "exclude": [
                "**/__pycache__",
                "**/.pytest_cache"
            ],
            "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
            "pythonPlatform": "Windows",
            "executionEnvironments": [
                {
                    "root": ".",
                    "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "pythonPlatform": "Windows",
                    "extraPaths": site_packages + [user_site]
                }
            ],
            "typeCheckingMode": "basic",
            "useLibraryCodeForTypes": True,
            "autoImportCompletions": True,
            "autoSearchPaths": True
        }
        
        with open("pyrightconfig.json", "w") as f:
            import json
            json.dump(pyright_config, f, indent=2)
        
        print(f"✅ Created pyrightconfig.json")
        
    except Exception as e:
        print(f"❌ Error creating Pylance config: {e}")

def check_python_environment():
    """Check Python environment details"""
    print(f"\n🐍 Python Environment Details")
    print("=" * 50)
    
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check if we're in a virtual environment
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"Virtual Environment: {venv}")
    else:
        print("Virtual Environment: None (using system Python)")
    
    # Check PATH for python executables
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    python_paths = []
    for dir_path in path_dirs:
        python_exe = Path(dir_path) / "python.exe"
        if python_exe.exists():
            python_paths.append(str(python_exe))
    
    print(f"\n🔍 Python executables in PATH:")
    for i, path in enumerate(python_paths, 1):
        current = " (CURRENT)" if path == sys.executable else ""
        print(f"   {i}. {path}{current}")

def create_import_test_file():
    """Create a test file to verify imports work"""
    test_content = '''#!/usr/bin/env python3
"""
Import Test File - Auto-generated for Pylance debugging
This file should import successfully if all dependencies are working
"""

# Test basic imports
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Test optional web dependencies
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    print("✅ FastAPI imports successful")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

# Test testing framework
try:
    import pytest
    print("✅ pytest import successful")
except ImportError as e:
    print(f"❌ pytest import failed: {e}")

# Test project modules
try:
    from core.fsot_engine import FSOTEngine
    from brain.brain_orchestrator import BrainOrchestrator
    print("✅ Project modules import successful")
except ImportError as e:
    print(f"❌ Project modules import failed: {e}")

if __name__ == "__main__":
    print("🧪 Import test completed")
'''
    
    with open("test_imports.py", "w") as f:
        f.write(test_content)
    
    print(f"✅ Created test_imports.py")

def main():
    """Main diagnostic function"""
    print("🔧 Pylance Import Resolution Diagnostics")
    print("=" * 60)
    
    # Run all diagnostics
    test_imports()
    check_python_environment()
    generate_pylance_config()
    create_import_test_file()
    
    print(f"\n🎯 Summary & Next Steps:")
    print(f"1. ✅ Configuration files created (pyrightconfig.json)")
    print(f"2. ✅ Test file created (test_imports.py)")
    print(f"3. 🔄 In VS Code, press Ctrl+Shift+P")
    print(f"4. 🔄 Type 'Python: Select Interpreter'")
    print(f"5. 🔄 Choose: {sys.executable}")
    print(f"6. 🔄 Run 'Developer: Reload Window'")
    print(f"7. 🧪 Test with: python test_imports.py")
    
    print(f"\n💡 If issues persist:")
    print(f"   - Restart VS Code completely")
    print(f"   - Check VS Code Python extension is latest version")
    print(f"   - Disable/re-enable Python extension")

if __name__ == "__main__":
    main()
