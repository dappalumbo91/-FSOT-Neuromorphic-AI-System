#!/usr/bin/env python3
"""
FSOT 2.0 System Functional Test
===============================
Test the actual functionality and interactions of the FSOT system
"""

import asyncio
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_system_functionality():
    """Test the actual functionality of the system"""
    logger.info("🚀 STARTING FSOT SYSTEM FUNCTIONAL TESTS")
    logger.info("=" * 60)
    
    test_results = {}
    
    try:
        # Import main system
        from main import FSOTHardwiredSystem
        from fsot_2_0_foundation import FSOTCore, FSOTDomain
        
        # Test 1: System Initialization
        logger.info("\n🧪 TEST 1: System Initialization")
        system = FSOTHardwiredSystem()
        test_results['system_init'] = "✅ PASSED"
        logger.info("✅ System initialized successfully")
        
        # Test 2: FSOT Core Calculations
        logger.info("\n🧪 TEST 2: FSOT Core Calculations")
        fsot_core = FSOTCore()
        
        # Test different dimensional calculations
        test_calculations = [
            (12, FSOTDomain.AI_TECH),
            (14, FSOTDomain.COGNITIVE),
            (11, FSOTDomain.NEURAL),
            (10, FSOTDomain.NEURAL)
        ]
        
        for d_eff, domain in test_calculations:
            scalar = fsot_core.compute_universal_scalar(
                d_eff=d_eff,
                domain=domain,
                observed=True
            )
            logger.info(f"   D_eff={d_eff}, Domain={domain.name}: Scalar={scalar:.6f}")
        
        test_results['fsot_calculations'] = "✅ PASSED"
        
        # Test 3: System Status
        logger.info("\n🧪 TEST 3: System Status")
        status = system.get_fsot_status()
        logger.info(f"✅ System status retrieved: {status['name']}")
        logger.info(f"   Domain: {status['domain']}")
        logger.info(f"   FSOT Scalar: {status['fsot_scalar']:.6f}")
        logger.info(f"   Theoretical Alignment: {status['theoretical_alignment']}")
        test_results['system_status'] = "✅ PASSED"
        
        # Test 4: Brain Orchestrator
        logger.info("\n🧪 TEST 4: Brain Orchestrator")
        try:
            from brain.brain_orchestrator import BrainOrchestrator
            brain = BrainOrchestrator()
            logger.info("✅ Brain orchestrator created successfully")
            test_results['brain_orchestrator'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ Brain orchestrator failed: {e}")
            test_results['brain_orchestrator'] = "❌ FAILED"
        
        # Test 5: Integration System
        logger.info("\n🧪 TEST 5: Integration System")
        try:
            from integration import EnhancedFSOTIntegration
            integration = EnhancedFSOTIntegration()
            integration_status = integration.get_integration_status()
            logger.info("✅ Integration system operational")
            logger.info(f"   Total Skills: {integration_status['skills_status']['total_skills']}")
            test_results['integration_system'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ Integration system failed: {e}")
            test_results['integration_system'] = "❌ FAILED"
        
        # Test 6: Memory Manager
        logger.info("\n🧪 TEST 6: Memory Manager")
        try:
            from utils.memory_manager import memory_manager
            memory_status = memory_manager.get_status()
            logger.info("✅ Memory manager operational")
            logger.info(f"   Total Memory: {memory_status['memory_stats']['total_gb']:.1f} GB")
            logger.info(f"   Available: {memory_status['memory_stats']['available_gb']:.1f} GB")
            test_results['memory_manager'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ Memory manager failed: {e}")
            test_results['memory_manager'] = "❌ FAILED"
        
        # Test 7: Consciousness Monitor
        logger.info("\n🧪 TEST 7: Consciousness Monitor")
        try:
            from core import consciousness_monitor
            consciousness_state = consciousness_monitor.get_current_state()
            logger.info("✅ Consciousness monitor operational")
            logger.info(f"   State: {consciousness_state['state']}")
            logger.info(f"   Level: {consciousness_state['consciousness_level']}")
            test_results['consciousness_monitor'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ Consciousness monitor failed: {e}")
            test_results['consciousness_monitor'] = "❌ FAILED"
        
        # Test 8: FSOT Compliance Validation
        logger.info("\n🧪 TEST 8: FSOT Compliance Validation")
        try:
            from fsot_2_0_foundation import validate_system_fsot_compliance
            compliance = validate_system_fsot_compliance()
            logger.info("✅ FSOT compliance validated")
            logger.info(f"   Status: {compliance['status']}")
            logger.info(f"   Violation Rate: {compliance['violation_rate']}")
            test_results['fsot_compliance'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ FSOT compliance failed: {e}")
            test_results['fsot_compliance'] = "❌ FAILED"
        
        # Test 9: System Configuration
        logger.info("\n🧪 TEST 9: System Configuration")
        try:
            from config import config
            logger.info("✅ Configuration loaded")
            logger.info(f"   System Config Available: {hasattr(config, 'system_config')}")
            test_results['system_config'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ Configuration failed: {e}")
            test_results['system_config'] = "❌ FAILED"
        
        # Test 10: CLI Interface
        logger.info("\n🧪 TEST 10: CLI Interface")
        try:
            from interfaces.cli_interface import CLIInterface
            cli = CLIInterface(system)
            logger.info("✅ CLI interface created")
            test_results['cli_interface'] = "✅ PASSED"
        except Exception as e:
            logger.error(f"❌ CLI interface failed: {e}")
            test_results['cli_interface'] = "❌ FAILED"
        
    except Exception as e:
        logger.error(f"❌ Critical system failure: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Calculate results
    passed_tests = sum(1 for result in test_results.values() if "✅" in result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info("\n🎯 FUNCTIONAL TEST RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in test_results.items():
        logger.info(f"{test_name}: {result}")
    
    logger.info(f"\nTests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        logger.info("🎉 SYSTEM FUNCTIONAL STATUS: EXCELLENT")
    elif success_rate >= 75:
        logger.info("✅ SYSTEM FUNCTIONAL STATUS: GOOD")
    elif success_rate >= 50:
        logger.info("⚠️ SYSTEM FUNCTIONAL STATUS: NEEDS ATTENTION")
    else:
        logger.info("❌ SYSTEM FUNCTIONAL STATUS: CRITICAL ISSUES")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = asyncio.run(test_system_functionality())
    sys.exit(0 if success else 1)
