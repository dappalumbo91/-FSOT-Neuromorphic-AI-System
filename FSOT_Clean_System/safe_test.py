#!/usr/bin/env python3
"""
Safe FSOT System Launcher - No Endless Loops
"""

import asyncio
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def safe_system_test():
    """Safely test the system without endless loops"""
    print("🧠 FSOT 2.0 SAFE SYSTEM TEST")
    print("=" * 40)
    
    try:
        # Import and create system
        from main import FSOTHardwiredSystem
        
        print("✅ Creating FSOT system...")
        system = FSOTHardwiredSystem()
        
        print("✅ System created successfully")
        print(f"   Name: {system.name}")
        print(f"   Domain: {system.domain.name}")
        print(f"   FSOT Scalar: {system.fsot_scalar:.6f}")
        
        # Initialize system with timeout
        print("\n🔧 Initializing system (with 30s timeout)...")
        try:
            await asyncio.wait_for(system.initialize(), timeout=30.0)
            print("✅ System initialized successfully")
            
            # Get system status
            print("\n📊 Getting system status...")
            status = await system.get_system_status()
            
            print("✅ System Status:")
            sys_info = status.get('system', {})
            print(f"   Running: {sys_info.get('running', False)}")
            print(f"   FSOT Compliance: {sys_info.get('fsot_compliance', False)}")
            print(f"   Modules: {len(status.get('brain', {}).get('modules', {}))}")
            
            # Clean shutdown
            print("\n🔄 Shutting down cleanly...")
            await system.shutdown()
            print("✅ System shutdown complete")
            
            return True
            
        except asyncio.TimeoutError:
            print("❌ System initialization timed out after 30 seconds")
            print("   This suggests an endless loop in initialization")
            await system.shutdown()
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def safe_cli_test():
    """Test CLI interface safely"""
    print("\n🖥️ SAFE CLI TEST")
    print("=" * 20)
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        from interfaces.cli_interface import CLIInterface
        
        # Create brain
        brain = BrainOrchestrator()
        await brain.initialize()
        
        # Create CLI but don't run it
        cli = CLIInterface(brain)
        print("✅ CLI interface created successfully")
        
        # Test a single command processing
        print("📝 Testing single command processing...")
        
        # Simulate help command
        await cli._process_command("help")
        print("✅ Command processing works")
        
        # Cleanup
        await brain.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

async def main():
    """Main safe test function"""
    print(f"🕐 Test started at: {datetime.now()}")
    
    # Test 1: System creation and initialization
    test1_success = await safe_system_test()
    
    # Test 2: CLI interface (non-interactive)
    test2_success = await safe_cli_test()
    
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    print(f"System Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"CLI Test:    {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed!")
        print("💡 The endless loop is likely in the interactive CLI input() loop")
        print("💡 Solution: Add proper error handling and timeouts to CLI")
    else:
        print("\n⚠️ Some tests failed - check errors above")
    
    print(f"\n🕐 Test completed at: {datetime.now()}")

if __name__ == "__main__":
    # Run with overall timeout to prevent hanging
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=120.0))  # 2 minute max
    except asyncio.TimeoutError:
        print("\n❌ CRITICAL: Test timed out after 2 minutes")
        print("   This confirms there's an endless loop in the system")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    
    print("\n✅ Test script completed - no endless loops here")
