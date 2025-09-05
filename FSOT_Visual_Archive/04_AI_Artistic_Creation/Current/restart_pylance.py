#!/usr/bin/env python3
"""
FSOT Pylance Restart Utility
============================

Simple utility to restart Pylance language server when FSOT changes cause type issues.
This can be run from anywhere in the project to force a language server refresh.
"""

import sys
import os

# Add FSOT_Clean_System to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FSOT_Clean_System'))

try:
    from fsot_hardwiring import force_pylance_restart, vscode_integration
    
    def main():
        """Main restart function"""
        print("🔄 FSOT Pylance Restart Utility")
        print("=" * 40)
        
        # Check if VS Code is available
        if vscode_integration.is_vscode_available():
            print("✅ VS Code environment detected")
            print("🔄 Restarting Pylance language server...")
            
            if force_pylance_restart():
                print("✅ Pylance restart successful!")
                print("   Your type checking should now be refreshed.")
            else:
                print("❌ Pylance restart failed.")
                print("   Try manually: Ctrl+Shift+P → 'Python: Restart Language Server'")
        else:
            print("⚠️  VS Code not detected")
            print("   This utility only works within VS Code environment")
            print("   Manual restart: Ctrl+Shift+P → 'Python: Restart Language Server'")
        
        print("\n🌟 FSOT hardwiring continues to enforce theoretical compliance")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Error importing FSOT modules: {e}")
    print("Please ensure you're running from the project root directory")
