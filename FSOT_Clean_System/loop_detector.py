#!/usr/bin/env python3
"""
Minimal System Test - Find Endless Loop Location
"""

import sys
import time
import signal
import threading
from datetime import datetime

def timeout_handler():
    """Force exit after timeout"""
    time.sleep(10)  # 10 second timeout
    print("\n❌ TIMEOUT - FORCING EXIT")
    print("This indicates an endless loop somewhere in the system")
    import os
    os._exit(1)

def test_imports_only():
    """Test just imports to see if loop is in import chain"""
    print("🔍 Testing imports only...")
    
    try:
        print("1. Importing FSOT foundation...")
        from fsot_2_0_foundation import FSOTCore
        print("✅ FSOT foundation imported")
        
        print("2. Importing config...")
        from config import config
        print("✅ Config imported")
        
        print("3. Importing main...")
        from main import FSOTHardwiredSystem
        print("✅ Main imported")
        
        print("4. Creating system object...")
        system = FSOTHardwiredSystem()
        print("✅ System object created")
        
        print("✅ All imports successful - loop is NOT in imports")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🚨 ENDLESS LOOP DETECTOR")
    print("=" * 30)
    print(f"Started at: {datetime.now()}")
    
    # Start timeout thread
    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()
    
    # Test imports only (safest test)
    success = test_imports_only()
    
    if success:
        print("\n🎯 CONCLUSION: Endless loop is NOT in imports")
        print("💡 The loop is likely in:")
        print("   - System initialization")
        print("   - CLI run loop")
        print("   - Brain orchestrator coordination loop")
        print("   - Web interface startup")
    else:
        print("\n🎯 CONCLUSION: Problem found in imports")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("✅ Test completed without endless loop")

if __name__ == "__main__":
    main()
