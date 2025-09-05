#!/usr/bin/env python3
"""
Safe CLI Test - Test interactive CLI without endless loops
"""

import asyncio
import sys
from datetime import datetime

async def test_cli_safely():
    """Test CLI in a controlled manner"""
    print("🖥️ SAFE CLI TESTING")
    print("=" * 30)
    
    try:
        from main import FSOTHardwiredSystem
        
        # Create system
        system = FSOTHardwiredSystem()
        print("✅ System created")
        
        # Initialize with timeout
        await asyncio.wait_for(system.initialize(), timeout=60.0)
        print("✅ System initialized")
        
        # Create CLI interface
        from interfaces.cli_interface import CLIInterface
        cli = CLIInterface(system.brain_orchestrator)
        print("✅ CLI interface created")
        
        # Test CLI command processing directly (non-interactive)
        print("\n🧪 Testing CLI commands directly:")
        
        # Test help command
        print("Testing 'help' command...")
        await cli._process_command("help")
        
        # Test status command
        print("\nTesting 'status' command...")
        await cli._process_command("status")
        
        # Test consciousness command
        print("\nTesting 'consciousness' command...")
        await cli._process_command("consciousness")
        
        # Test history command
        print("\nTesting 'history' command...")
        await cli._process_command("history")
        
        # Test a simple query
        print("\nTesting simple query...")
        await cli._process_command("What is the golden ratio?")
        
        print("\n✅ All CLI commands tested successfully!")
        
        # Shutdown
        await system.shutdown()
        print("✅ System shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def main():
    print(f"🕐 CLI Test started at: {datetime.now()}")
    
    try:
        success = await asyncio.wait_for(test_cli_safely(), timeout=180.0)
        
        if success:
            print("\n🎉 CLI TEST SUCCESSFUL!")
            print("💡 The CLI interface is working properly")
            print("💡 Interactive mode should now work without endless loops")
        else:
            print("\n❌ CLI test had issues")
        
    except asyncio.TimeoutError:
        print("\n⏰ CLI test timed out")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    
    print(f"\n🕐 CLI Test completed at: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
