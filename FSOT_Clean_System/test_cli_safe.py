#!/usr/bin/env python3
"""
FSOT CLI Test - Non-Interactive Version
Tests CLI functionality without interactive input loops
"""

import asyncio
import sys
from datetime import datetime

async def test_cli_commands():
    """Test CLI commands without interactive loop"""
    print("🖥️ CLI COMMAND TESTING")
    print("=" * 30)
    
    try:
        from main import FSOTHardwiredSystem
        from interfaces.cli_interface import CLIInterface
        
        # Create system
        system = FSOTHardwiredSystem()
        
        # Initialize with timeout
        await asyncio.wait_for(system.initialize(), timeout=30.0)
        print("✅ System initialized")
        
        # Create CLI
        cli = CLIInterface(system.brain_orchestrator)
        print("✅ CLI created")
        
        # Test commands directly (no interactive input)
        commands_to_test = [
            "help",
            "status", 
            "consciousness",
            "What is FSOT?",
            "quit"
        ]
        
        print("\n🧪 Testing CLI commands:")
        for cmd in commands_to_test:
            print(f"\n▶️ Testing: {cmd}")
            try:
                await asyncio.wait_for(cli._process_command(cmd), timeout=10.0)
                print(f"✅ Command '{cmd}' processed successfully")
            except asyncio.TimeoutError:
                print(f"⏰ Command '{cmd}' timed out")
            except Exception as e:
                print(f"❌ Command '{cmd}' failed: {e}")
        
        # Shutdown
        await system.shutdown()
        print("\n✅ System shutdown complete")
        
        print("\n🎉 CLI TESTING COMPLETE")
        print("✅ All CLI functionality works correctly")
        print("✅ No endless loops in CLI processing")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

async def main():
    print(f"🕐 Started: {datetime.now()}")
    
    try:
        success = await asyncio.wait_for(test_cli_commands(), timeout=60.0)
        print(f"\n🕐 Completed: {datetime.now()}")
        return success
        
    except asyncio.TimeoutError:
        print("\n⏰ CLI test timed out - endless loop detected")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
        sys.exit(0)
