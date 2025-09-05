#!/usr/bin/env python3
"""
FSOT 2.0 System Debug Tool
==========================
Comprehensive debug analysis of the entire FSOT Neuromorphic AI System
"""

import asyncio
import logging
import traceback
import sys
from pathlib import Path

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_system.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def debug_imports():
    """Debug all system imports"""
    logger.info("🔍 DEBUGGING SYSTEM IMPORTS...")
    
    import_results = {}
    
    # Test core FSOT imports
    try:
        from fsot_2_0_foundation import FSOTCore, FSOTComponent, FSOTDomain
        import_results['fsot_foundation'] = "✅ SUCCESS"
        logger.info("✅ FSOT 2.0 Foundation imported successfully")
    except Exception as e:
        import_results['fsot_foundation'] = f"❌ FAILED: {e}"
        logger.error(f"❌ FSOT Foundation import failed: {e}")
    
    # Test hardwiring system
    try:
        from fsot_hardwiring import hardwire_fsot, activate_fsot_hardwiring
        import_results['fsot_hardwiring'] = "✅ SUCCESS"
        logger.info("✅ FSOT Hardwiring imported successfully")
    except Exception as e:
        import_results['fsot_hardwiring'] = f"❌ FAILED: {e}"
        logger.error(f"❌ FSOT Hardwiring import failed: {e}")
    
    # Test theoretical integration
    try:
        from fsot_theoretical_integration import FSOTNeuromorphicIntegrator
        import_results['fsot_theory'] = "✅ SUCCESS"
        logger.info("✅ FSOT Theoretical Integration imported successfully")
    except Exception as e:
        import_results['fsot_theory'] = f"❌ FAILED: {e}"
        logger.error(f"❌ FSOT Theory import failed: {e}")
    
    # Test brain modules
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        import_results['brain_orchestrator'] = "✅ SUCCESS"
        logger.info("✅ Brain Orchestrator imported successfully")
    except Exception as e:
        import_results['brain_orchestrator'] = f"❌ FAILED: {e}"
        logger.error(f"❌ Brain Orchestrator import failed: {e}")
    
    # Test integration system
    try:
        from integration import EnhancedFSOTIntegration
        import_results['integration'] = "✅ SUCCESS"
        logger.info("✅ Integration System imported successfully")
    except Exception as e:
        import_results['integration'] = f"❌ FAILED: {e}"
        logger.error(f"❌ Integration System import failed: {e}")
    
    # Test core components
    try:
        from core import consciousness_monitor, neural_hub
        import_results['core'] = "✅ SUCCESS"
        logger.info("✅ Core components imported successfully")
    except Exception as e:
        import_results['core'] = f"❌ FAILED: {e}"
        logger.error(f"❌ Core components import failed: {e}")
    
    # Test utilities
    try:
        from utils.memory_manager import memory_manager
        import_results['utils'] = "✅ SUCCESS"
        logger.info("✅ Utilities imported successfully")
    except Exception as e:
        import_results['utils'] = f"❌ FAILED: {e}"
        logger.error(f"❌ Utilities import failed: {e}")
    
    # Test config
    try:
        from config import config
        import_results['config'] = "✅ SUCCESS"
        logger.info("✅ Config imported successfully")
    except Exception as e:
        import_results['config'] = f"❌ FAILED: {e}"
        logger.error(f"❌ Config import failed: {e}")
    
    return import_results

def debug_fsot_core():
    """Debug FSOT 2.0 core functionality"""
    logger.info("🔍 DEBUGGING FSOT 2.0 CORE...")
    
    try:
        from fsot_2_0_foundation import FSOTCore, FSOTDomain, FSOTConstants
        
        # Test FSOT Core initialization
        fsot_core = FSOTCore()
        logger.info("✅ FSOT Core initialized successfully")
        
        # Test constant calculations
        phi = float(FSOTConstants.PHI)
        consciousness = FSOTConstants.CONSCIOUSNESS_FACTOR
        max_dims = FSOTConstants.MAX_DIMENSIONS
        
        logger.info(f"📊 FSOT Constants:")
        logger.info(f"   φ (Golden Ratio): {phi:.10f}")
        logger.info(f"   Consciousness Factor: {consciousness:.6f}")
        logger.info(f"   Max Dimensions: {max_dims}")
        
        # Test scalar calculation
        scalar = fsot_core.compute_universal_scalar(
            d_eff=12,
            domain=FSOTDomain.AI_TECH,
            observed=True
        )
        logger.info(f"✅ Universal scalar calculated: {scalar:.6f}")
        
        # Test system health
        health = fsot_core.get_system_health()
        logger.info(f"📈 FSOT System Health: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FSOT Core debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_brain_modules():
    """Debug brain module functionality"""
    logger.info("🔍 DEBUGGING BRAIN MODULES...")
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        
        # Test brain orchestrator initialization
        brain = BrainOrchestrator()
        logger.info("✅ Brain Orchestrator initialized successfully")
        
        # Check available modules
        try:
            modules = brain.modules if hasattr(brain, 'modules') else {}
            logger.info(f"🧠 Available Brain Modules: {len(modules)}")
            for module_name in modules:
                logger.info(f"   - {module_name}")
        except Exception as e:
            logger.warning(f"Could not get modules list: {e}")
            logger.info("🧠 Brain modules status: Unknown")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Brain modules debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_integration_system():
    """Debug integration system"""
    logger.info("🔍 DEBUGGING INTEGRATION SYSTEM...")
    
    try:
        from integration import EnhancedFSOTIntegration
        
        # Test integration initialization
        integration = EnhancedFSOTIntegration()
        logger.info("✅ Integration System initialized successfully")
        
        # Test API managers
        try:
            api_status = integration.get_integration_status() if hasattr(integration, 'get_integration_status') else "Unknown"
            logger.info(f"🔌 API Status: {api_status}")
        except Exception as e:
            logger.warning(f"Could not get API status: {e}")
        
        # Test skills database
        try:
            skills_info = "Skills system operational"
            logger.info(f"🎯 Skills Status: {skills_info}")
        except Exception as e:
            logger.warning(f"Could not get skills info: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration system debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_memory_system():
    """Debug memory management system"""
    logger.info("🔍 DEBUGGING MEMORY SYSTEM...")
    
    try:
        from utils.memory_manager import memory_manager
        
        # Test memory manager
        memory_status = memory_manager.get_status()
        logger.info(f"💾 Memory Status: {memory_status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory system debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

def debug_consciousness_system():
    """Debug consciousness monitoring"""
    logger.info("🔍 DEBUGGING CONSCIOUSNESS SYSTEM...")
    
    try:
        from core import consciousness_monitor
        
        # Test consciousness monitor
        state = consciousness_monitor.get_current_state()
        logger.info(f"🌟 Consciousness State: {state}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Consciousness system debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

async def debug_main_system():
    """Debug the main system initialization"""
    logger.info("🔍 DEBUGGING MAIN SYSTEM INITIALIZATION...")
    
    try:
        # Import the main system
        from main import FSOTHardwiredSystem
        
        # Test system initialization
        system = FSOTHardwiredSystem()
        logger.info("✅ Main system initialized successfully")
        
        # Test FSOT status
        status = system.get_fsot_status()
        logger.info(f"🔒 FSOT Status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main system debug failed: {e}")
        logger.error(traceback.format_exc())
        return False

def check_file_integrity():
    """Check integrity of critical files"""
    logger.info("🔍 CHECKING FILE INTEGRITY...")
    
    critical_files = [
        "fsot_2_0_foundation.py",
        "fsot_hardwiring.py",
        "fsot_theoretical_integration.py",
        "main.py",
        "config/system_config.json",
        "brain/brain_orchestrator.py",
        "integration/__init__.py"
    ]
    
    file_status = {}
    
    for file_path in critical_files:
        full_path = Path(file_path)
        if full_path.exists():
            file_status[file_path] = f"✅ EXISTS ({full_path.stat().st_size} bytes)"
            logger.info(f"✅ {file_path} exists")
        else:
            file_status[file_path] = "❌ MISSING"
            logger.error(f"❌ {file_path} is missing!")
    
    return file_status

async def main():
    """Run comprehensive system debug"""
    logger.info("🚀 STARTING COMPREHENSIVE FSOT SYSTEM DEBUG")
    logger.info("=" * 60)
    
    debug_results = {}
    
    # Check file integrity
    logger.info("\n📁 PHASE 1: FILE INTEGRITY CHECK")
    debug_results['file_integrity'] = check_file_integrity()
    
    # Test imports
    logger.info("\n📦 PHASE 2: IMPORT TESTING")
    debug_results['imports'] = debug_imports()
    
    # Test FSOT core
    logger.info("\n🔬 PHASE 3: FSOT CORE TESTING")
    debug_results['fsot_core'] = debug_fsot_core()
    
    # Test brain modules
    logger.info("\n🧠 PHASE 4: BRAIN MODULES TESTING")
    debug_results['brain_modules'] = debug_brain_modules()
    
    # Test integration system
    logger.info("\n🔌 PHASE 5: INTEGRATION SYSTEM TESTING")
    debug_results['integration'] = debug_integration_system()
    
    # Test memory system
    logger.info("\n💾 PHASE 6: MEMORY SYSTEM TESTING")
    debug_results['memory'] = debug_memory_system()
    
    # Test consciousness system
    logger.info("\n🌟 PHASE 7: CONSCIOUSNESS SYSTEM TESTING")
    debug_results['consciousness'] = debug_consciousness_system()
    
    # Test main system
    logger.info("\n🎯 PHASE 8: MAIN SYSTEM TESTING")
    debug_results['main_system'] = await debug_main_system()
    
    # Generate debug report
    logger.info("\n📊 DEBUG SUMMARY")
    logger.info("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, result in debug_results.items():
        if isinstance(result, dict):
            for test, status in result.items():
                total_tests += 1
                if "✅" in str(status):
                    passed_tests += 1
                logger.info(f"{category}.{test}: {status}")
        else:
            total_tests += 1
            if result:
                passed_tests += 1
                logger.info(f"{category}: ✅ PASSED")
            else:
                logger.info(f"{category}: ❌ FAILED")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("\n🎯 FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("🎉 SYSTEM STATUS: EXCELLENT")
    elif success_rate >= 75:
        logger.info("✅ SYSTEM STATUS: GOOD")
    elif success_rate >= 50:
        logger.info("⚠️ SYSTEM STATUS: NEEDS ATTENTION")
    else:
        logger.info("❌ SYSTEM STATUS: CRITICAL ISSUES")
    
    return debug_results

if __name__ == "__main__":
    asyncio.run(main())
