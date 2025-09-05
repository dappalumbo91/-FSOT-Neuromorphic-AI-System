#!/usr/bin/env python3
"""
Simple FSOT System Test
Debug what's causing the endless loop
"""

import sys
import logging
import asyncio
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test basic imports step by step"""
    logger.info("Testing basic imports...")
    
    try:
        logger.info("1. Testing FSOT foundation...")
        from fsot_2_0_foundation import FSOTCore, FSOTDomain
        logger.info("✅ FSOT foundation imported")
        
        logger.info("2. Testing config...")
        from config import config
        logger.info("✅ Config imported")
        
        logger.info("3. Testing core modules...")
        from core import consciousness_monitor, neural_hub
        logger.info("✅ Core modules imported")
        
        logger.info("4. Testing brain orchestrator...")
        from brain.brain_orchestrator import BrainOrchestrator
        logger.info("✅ Brain orchestrator imported")
        
        logger.info("5. Testing CLI interface...")
        from interfaces.cli_interface import CLIInterface
        logger.info("✅ CLI interface imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_fsot_core():
    """Test FSOT core functionality"""
    logger.info("Testing FSOT core...")
    
    try:
        from fsot_2_0_foundation import FSOTCore, FSOTDomain
        
        fsot_core = FSOTCore()
        scalar = fsot_core.compute_universal_scalar(
            d_eff=12,
            domain=FSOTDomain.AI_TECH,
            observed=True
        )
        logger.info(f"✅ FSOT scalar computed: {scalar:.6f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ FSOT core test failed: {e}")
        return False

async def test_brain_creation():
    """Test brain orchestrator creation"""
    logger.info("Testing brain orchestrator creation...")
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        
        brain = BrainOrchestrator()
        logger.info("✅ Brain orchestrator created")
        
        # Try basic initialization
        logger.info("Testing brain initialization...")
        await brain.initialize()
        logger.info("✅ Brain orchestrator initialized")
        
        # Test status
        status = await brain.get_status()
        logger.info(f"✅ Brain status: {len(status.get('modules', {}))} modules")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Brain creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_system_creation():
    """Test main system creation"""
    logger.info("Testing main system creation...")
    
    try:
        from main import FSOTHardwiredSystem
        
        logger.info("Creating FSOT system...")
        system = FSOTHardwiredSystem()
        logger.info("✅ FSOT system created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_cli_interface():
    """Test CLI interface creation"""
    logger.info("Testing CLI interface...")
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        from interfaces.cli_interface import CLIInterface
        
        brain = BrainOrchestrator()
        cli = CLIInterface(brain)
        logger.info("✅ CLI interface created")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CLI interface test failed: {e}")
        return False

async def main():
    logger.info("🔍 FSOT SYSTEM DIAGNOSTIC TEST")
    logger.info("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("FSOT Core", test_fsot_core),
        ("Brain Creation", test_brain_creation),
        ("System Creation", test_system_creation),
        ("CLI Interface", test_cli_interface)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 30)
        
        try:
            result = await test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            logger.error(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    logger.info("\n📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    logger.info(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed - system should work normally")
    else:
        logger.info("⚠️ Some tests failed - check errors above")

if __name__ == "__main__":
    asyncio.run(main())
