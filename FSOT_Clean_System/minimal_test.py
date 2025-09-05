#!/usr/bin/env python3
"""
Minimal FSOT System Test - Find the endless loop
"""

import sys
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_step_by_step():
    """Test each component step by step"""
    
    print("🔍 TESTING STEP BY STEP...")
    
    try:
        print("1. Testing FSOT foundation import...")
        from fsot_2_0_foundation import FSOTCore, FSOTDomain
        print("✅ FSOT foundation OK")
        
        print("2. Testing config import...")
        from config import config
        print("✅ Config OK")
        
        print("3. Testing core import...")
        from core import consciousness_monitor
        print("✅ Core OK")
        
        print("4. Testing brain orchestrator import...")
        from brain.brain_orchestrator import BrainOrchestrator
        print("✅ Brain orchestrator import OK")
        
        print("5. Creating brain orchestrator...")
        brain = BrainOrchestrator()
        print("✅ Brain orchestrator created")
        
        print("6. Testing CLI import...")
        from interfaces.cli_interface import CLIInterface
        print("✅ CLI interface import OK")
        
        print("7. Creating CLI interface...")
        cli = CLIInterface(brain)
        print("✅ CLI interface created")
        
        print("8. Testing main system import...")
        from main import FSOTHardwiredSystem
        print("✅ Main system import OK")
        
        print("9. Creating main system...")
        system = FSOTHardwiredSystem()
        print("✅ Main system created")
        
        print("\n🎉 ALL TESTS PASSED - No endless loop in basic creation")
        return True
        
    except Exception as e:
        print(f"❌ ERROR at step: {e}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_step_by_step()
    if success:
        print("\n✅ System components can be created without loops")
        print("The endless loop must be in the run/CLI loop itself")
    else:
        print("\n❌ Found the problem - see error above")
    
    sys.exit(0 if success else 1)
