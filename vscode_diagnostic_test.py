#!/usr/bin/env python3
"""
VS Code Problems Panel Diagnostic
==================================
Tests if the main import issues have been resolved.
"""

print("🔍 VS Code Import Diagnostic Test")
print("=" * 40)

# Test major external libraries
test_imports = [
    "numpy", "matplotlib", "scipy", "pandas", "networkx",
    "seaborn", "PIL", "cv2", "yaml", "schedule", "nltk",
    "fastapi", "uvicorn", "pytest", "bs4", "flask",
    "sklearn", "torch", "sympy", "mpmath"
]

successful_imports = []
failed_imports = []

for module in test_imports:
    try:
        __import__(module)
        successful_imports.append(module)
        print(f"✅ {module}")
    except ImportError as e:
        failed_imports.append((module, str(e)))
        print(f"❌ {module}: {e}")

print(f"\n📊 RESULTS:")
print(f"✅ Successful: {len(successful_imports)}/{len(test_imports)}")
print(f"❌ Failed: {len(failed_imports)}")

if failed_imports:
    print(f"\n🔧 FAILED IMPORTS:")
    for module, error in failed_imports:
        print(f"   {module}: {error}")

# Test core FSOT system
print(f"\n🧠 FSOT CORE SYSTEM TEST:")
try:
    import brain_system
    print("✅ brain_system imports successfully")
except Exception as e:
    print(f"❌ brain_system: {e}")

try:
    import neural_network
    print("✅ neural_network imports successfully")
except Exception as e:
    print(f"❌ neural_network: {e}")

try:
    import fsot_simulations
    print("✅ fsot_simulations imports successfully")
except Exception as e:
    print(f"❌ fsot_simulations: {e}")

print(f"\n🎯 RECOMMENDATION:")
if len(failed_imports) == 0:
    print("🎉 All major dependencies resolved!")
    print("📋 VS Code Problems panel should now show significantly fewer errors.")
    print("🔄 Try reloading VS Code window if problems persist.")
else:
    print(f"⚠️ {len(failed_imports)} packages still need attention.")
    print("🔧 Install missing packages to resolve remaining import errors.")

print(f"\n💡 To refresh VS Code:")
print("1. Press Ctrl+Shift+P")
print("2. Type 'Python: Restart Language Server'")
print("3. Or reload window with Ctrl+Shift+P -> 'Developer: Reload Window'")
