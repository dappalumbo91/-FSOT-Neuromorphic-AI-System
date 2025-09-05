#!/usr/bin/env python3
"""
Safe Main - No Endless Loops
"""

import asyncio
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def safe_main():
    """Safe main that exits cleanly"""
    print("🧠 FSOT 2.0 SAFE MODE")
    print("=" * 30)
    print(f"Started: {datetime.now()}")
    
    try:
        # Import and create system
        from main import FSOTHardwiredSystem
        system = FSOTHardwiredSystem()
        print("✅ System created successfully")
        
        # Initialize with strict timeout
        await asyncio.wait_for(system.initialize(), timeout=30.0)
        print("✅ System initialized successfully")
        
        # Get status
        status = await system.get_system_status()
        print("✅ System status retrieved")
        print(f"   Running: {status['system']['running']}")
        print(f"   FSOT Compliance: {status['system']['fsot_compliance']}")
        print(f"   Brain Modules: {len(status.get('brain', {}).get('modules', {}))}")
        
        # Immediately shutdown
        await system.shutdown()
        print("✅ System shutdown complete")
        
        print(f"Completed: {datetime.now()}")
        print("🎉 SYSTEM WORKS PERFECTLY - NO LOOPS!")
        return True
        
    except asyncio.TimeoutError:
        print("❌ Timeout - indicates endless loop in initialization")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Run with overall timeout
    try:
        result = asyncio.run(asyncio.wait_for(safe_main(), timeout=60.0))
        sys.exit(0 if result else 1)
    except asyncio.TimeoutError:
        print("❌ CRITICAL: Overall timeout - endless loop confirmed")
        sys.exit(1)
    except KeyboardInterrupt:
        print("👋 Interrupted by user")
        sys.exit(0)
