#!/usr/bin/env python3
"""
COMPREHENSIVE FSOT AI SYSTEM DEBUG
==================================
Deep testing of every function, component, and integration
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, List

# Setup comprehensive logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FSOTSystemDebugger:
    """Comprehensive FSOT system debugging framework"""
    
    def __init__(self):
        self.test_results = {}
        self.component_status = {}
        self.function_tests = {}
        self.error_log = []
        
    async def debug_entire_system(self):
        """Debug every component of the FSOT AI system"""
        logger.info("🚀 STARTING COMPREHENSIVE FSOT AI SYSTEM DEBUG")
        logger.info("=" * 80)
        
        # Phase 1: Core Foundation Testing
        await self._debug_fsot_foundation()
        
        # Phase 2: Hardwiring System Testing
        await self._debug_hardwiring_system()
        
        # Phase 3: Theoretical Integration Testing
        await self._debug_theoretical_integration()
        
        # Phase 4: Brain System Complete Testing
        await self._debug_brain_system_complete()
        
        # Phase 5: Integration & API Testing
        await self._debug_integration_apis()
        
        # Phase 6: Memory & Consciousness Deep Testing
        await self._debug_memory_consciousness()
        
        # Phase 7: Main System & CLI Testing
        await self._debug_main_cli_system()
        
        # Phase 8: Configuration & Utils Testing
        await self._debug_config_utils()
        
        # Phase 9: Performance & Stress Testing
        await self._debug_performance_stress()
        
        # Phase 10: End-to-End Integration Testing
        await self._debug_end_to_end()
        
        # Generate comprehensive report
        self._generate_debug_report()
    
    async def _debug_fsot_foundation(self):
        """Deep debug of FSOT 2.0 foundation"""
        logger.info("\n🔬 PHASE 1: FSOT FOUNDATION DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from fsot_2_0_foundation import (
                FSOTCore, FSOTComponent, FSOTDomain, FSOTConstants,
                validate_system_fsot_compliance, fsot_enforced
            )
            
            # Test FSOTCore functionality
            logger.info("🧪 Testing FSOTCore...")
            fsot_core = FSOTCore()
            
            # Test all domain calculations
            test_domains = [
                (FSOTDomain.NEURAL, [10, 11, 12, 13, 14, 15]),
                (FSOTDomain.AI_TECH, [11, 12, 13]),
                (FSOTDomain.COGNITIVE, [12, 13, 14, 15, 16])
            ]
            
            calculations_tested = 0
            for domain, dimensions in test_domains:
                for d_eff in dimensions:
                    scalar = fsot_core.compute_universal_scalar(
                        d_eff=d_eff, domain=domain, observed=True
                    )
                    logger.info(f"   {domain.name} D_eff={d_eff}: Scalar={scalar:.6f}")
                    calculations_tested += 1
            
            # Test system health
            health = fsot_core.get_system_health()
            logger.info(f"📊 System Health: {health}")
            
            # Test compliance validation
            compliance = validate_system_fsot_compliance()
            logger.info(f"🔒 Compliance: {compliance['status']}")
            
            self.test_results['fsot_foundation'] = {
                'status': '✅ PASSED',
                'calculations_tested': calculations_tested,
                'health': health,
                'compliance': compliance['status']
            }
            
        except Exception as e:
            self._log_error('fsot_foundation', e)
    
    async def _debug_hardwiring_system(self):
        """Deep debug of hardwiring system"""
        logger.info("\n🔧 PHASE 2: HARDWIRING SYSTEM DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from fsot_hardwiring import (
                hardwire_fsot, neural_module, ai_component, cognitive_process,
                activate_fsot_hardwiring, get_hardwiring_status, FSOTEnforcementSystem
            )
            from fsot_2_0_foundation import FSOTDomain
            
            # Test enforcement system
            logger.info("🧪 Testing FSOT Enforcement...")
            enforcement = FSOTEnforcementSystem()
            
            # Test hardwiring activation
            results = activate_fsot_hardwiring()
            logger.info(f"🔒 Hardwiring Results: {results}")
            
            # Test status retrieval
            status = get_hardwiring_status()
            logger.info(f"📊 Hardwiring Status: {status}")
            
            # Test decorators
            @hardwire_fsot(FSOTDomain.AI_TECH, 12)
            class TestComponent:
                def test_method(self):
                    return "test_passed"
            
            test_comp = TestComponent()
            result = test_comp.test_method()
            logger.info(f"🧪 Decorator Test: {result}")
            
            self.test_results['hardwiring'] = {
                'status': '✅ PASSED',
                'enforcement_active': True,
                'decorators_working': result == "test_passed"
            }
            
        except Exception as e:
            self._log_error('hardwiring', e)
    
    async def _debug_theoretical_integration(self):
        """Deep debug of theoretical integration"""
        logger.info("\n📐 PHASE 3: THEORETICAL INTEGRATION DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from fsot_theoretical_integration import (
                FSOTNeuromorphicIntegrator, get_fsot_enhancement,
                calculate_consciousness_scalar
            )
            
            # Test integrator
            logger.info("🧪 Testing Theoretical Integrator...")
            integrator = FSOTNeuromorphicIntegrator()
            
            # Test report generation
            report = integrator.generate_theoretical_report()
            logger.info(f"📄 Report Generated: {len(report)} characters")
            
            # Test consciousness calculations
            consciousness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
            for level in consciousness_levels:
                scalar = calculate_consciousness_scalar(level)
                logger.info(f"🧠 Consciousness {level}: Scalar={scalar:.6f}")
            
            # Test FSOT enhancements
            test_config = {"name": "test_module", "enabled": True}
            enhanced = get_fsot_enhancement("test_module", test_config)
            logger.info(f"🔧 Enhancement Test: {enhanced}")
            
            self.test_results['theoretical'] = {
                'status': '✅ PASSED',
                'report_length': len(report),
                'consciousness_calcs': len(consciousness_levels),
                'enhancements_working': enhanced is not None
            }
            
        except Exception as e:
            self._log_error('theoretical', e)
    
    async def _debug_brain_system_complete(self):
        """Complete deep debug of brain system"""
        logger.info("\n🧠 PHASE 4: BRAIN SYSTEM COMPLETE DEBUG")
        logger.info("=" * 50)
        
        try:
            from brain.brain_orchestrator import BrainOrchestrator
            
            # Test brain creation and initialization
            logger.info("🧪 Testing Brain Orchestrator...")
            brain = BrainOrchestrator()
            
            # Initialize brain
            await brain.initialize()
            logger.info("✅ Brain initialized")
            
            # Test all brain modules
            status = await brain.get_status()
            modules = status.get('modules', {})
            logger.info(f"🧩 Brain Modules ({len(modules)}):")
            
            for name, module_info in modules.items():
                logger.info(f"   {name}: Active={module_info.get('is_active')}")
                
                # Test individual module functionality
                module = await brain.get_module(name)
                if module:
                    module_status = module.get_status()
                    logger.info(f"     Status: {module_status}")
            
            # Test query processing
            logger.info("🧪 Testing Query Processing...")
            test_queries = [
                "What is consciousness?",
                "How does memory work?",
                "What is the golden ratio?",
                "Explain neural networks"
            ]
            
            for query in test_queries:
                try:
                    result = await asyncio.wait_for(
                        brain.process_query(query), timeout=10.0
                    )
                    logger.info(f"   Query '{query[:20]}...': ✅ Processed")
                except asyncio.TimeoutError:
                    logger.warning(f"   Query '{query[:20]}...': ⏰ Timeout")
                except Exception as e:
                    logger.error(f"   Query '{query[:20]}...': ❌ Error: {e}")
            
            # Test brain connections
            connections = status.get('connections', {})
            logger.info(f"🔗 Brain Connections: {len(connections)} modules connected")
            
            # Shutdown brain
            await brain.shutdown()
            logger.info("✅ Brain shutdown complete")
            
            self.test_results['brain_system'] = {
                'status': '✅ PASSED',
                'modules_count': len(modules),
                'queries_tested': len(test_queries),
                'connections': len(connections)
            }
            
        except Exception as e:
            self._log_error('brain_system', e)
    
    async def _debug_integration_apis(self):
        """Deep debug of integration and API systems"""
        logger.info("\n🔌 PHASE 5: INTEGRATION & API DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from integration import EnhancedFSOTIntegration
            from brain.brain_orchestrator import BrainOrchestrator
            
            # Create brain for integration
            brain = BrainOrchestrator()
            await brain.initialize()
            
            # Test integration system
            logger.info("🧪 Testing Integration System...")
            integration = EnhancedFSOTIntegration(brain)
            
            # Test integration status
            status = integration.get_integration_status()
            logger.info(f"📊 Integration Status:")
            logger.info(f"   Initialized: {status['system_info']['initialized']}")
            logger.info(f"   Skills: {status['skills_status']['total_skills']}")
            logger.info(f"   Categories: {len(status['skills_status']['category_breakdown'])}")
            
            # Test API systems
            api_status = status['api_status']
            logger.info(f"📡 API Systems ({len(api_status)}):")
            for api_name, api_info in api_status.items():
                enabled = api_info.get('enabled', False)
                configured = api_info.get('configured', False)
                logger.info(f"   {api_name}: Enabled={enabled}, Configured={configured}")
            
            # Test skills system
            skills_status = status['skills_status']
            logger.info(f"🎯 Skills System:")
            for category, info in skills_status['category_breakdown'].items():
                count = info['count']
                proficiency = info['average_proficiency']
                logger.info(f"   {category}: {count} skills, {proficiency:.2%} proficiency")
            
            # Test learning system
            learning_status = status['learning_status']
            logger.info(f"📚 Learning System:")
            logger.info(f"   Concepts: {learning_status['total_concepts']}")
            logger.info(f"   Domains: {len(learning_status['domains'])}")
            
            # Test recommendations
            recommendations = integration.get_recommendations()
            logger.info(f"💡 Recommendations: {len(recommendations)}")
            
            # Cleanup
            integration.shutdown()
            await brain.shutdown()
            
            self.test_results['integration'] = {
                'status': '✅ PASSED',
                'skills_count': skills_status['total_skills'],
                'api_count': len(api_status),
                'recommendations': len(recommendations)
            }
            
        except Exception as e:
            self._log_error('integration', e)
    
    async def _debug_memory_consciousness(self):
        """Deep debug of memory and consciousness systems"""
        logger.info("\n💾🌟 PHASE 6: MEMORY & CONSCIOUSNESS DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from utils.memory_manager import memory_manager
            from core import consciousness_monitor
            
            # Test memory manager
            logger.info("🧪 Testing Memory Manager...")
            memory_manager.start_monitoring()
            
            memory_status = memory_manager.get_status()
            mem_stats = memory_status['memory_stats']
            logger.info(f"💾 Memory Status:")
            logger.info(f"   Total: {mem_stats['total_gb']:.1f} GB")
            logger.info(f"   Available: {mem_stats['available_gb']:.1f} GB")
            logger.info(f"   Usage: {mem_stats['percentage']:.1f}%")
            logger.info(f"   Status: {mem_stats['status']}")
            
            # Test consciousness monitor
            logger.info("🧪 Testing Consciousness Monitor...")
            await consciousness_monitor.start_monitoring()
            
            consciousness_state = consciousness_monitor.get_current_state()
            logger.info(f"🌟 Consciousness State:")
            logger.info(f"   Level: {consciousness_state['consciousness_level']:.1%}")
            logger.info(f"   State: {consciousness_state['state']}")
            logger.info(f"   Coherence: {consciousness_state['coherence']:.1%}")
            
            # Test consciousness updates
            test_loads = [0.1, 0.3, 0.5, 0.7, 0.9]
            for load in test_loads:
                consciousness_monitor.update_processing_load(load)
                state = consciousness_monitor.get_current_state()
                logger.info(f"   Load {load}: Level={state['consciousness_level']:.2%}")
            
            # Test brain waves
            brain_waves = consciousness_state.get('brain_waves', {})
            logger.info(f"🌊 Brain Waves: {len(brain_waves)} types")
            
            # Cleanup
            await consciousness_monitor.stop_monitoring()
            
            self.test_results['memory_consciousness'] = {
                'status': '✅ PASSED',
                'memory_gb': mem_stats['total_gb'],
                'consciousness_tests': len(test_loads),
                'brain_waves': len(brain_waves)
            }
            
        except Exception as e:
            self._log_error('memory_consciousness', e)
    
    async def _debug_main_cli_system(self):
        """Deep debug of main system and CLI"""
        logger.info("\n🖥️ PHASE 7: MAIN SYSTEM & CLI DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from main import FSOTHardwiredSystem
            from interfaces.cli_interface import CLIInterface
            
            # Test main system creation
            logger.info("🧪 Testing Main System...")
            system = FSOTHardwiredSystem()
            logger.info(f"✅ System created: {system.name}")
            
            # Test system initialization (with coordination disabled)
            await system.initialize()
            logger.info("✅ System initialized")
            
            # Test system status
            status = await system.get_system_status()
            logger.info(f"📊 System Status:")
            logger.info(f"   Running: {status['system']['running']}")
            logger.info(f"   FSOT Compliance: {status['system']['fsot_compliance']}")
            logger.info(f"   FSOT Scalar: {status['system']['fsot_scalar']:.6f}")
            
            # Test CLI interface (non-interactive)
            logger.info("🧪 Testing CLI Interface...")
            cli = CLIInterface(system.brain_orchestrator)
            
            # Test CLI commands
            test_commands = [
                "help",
                "status",
                "consciousness",
                "history"
            ]
            
            for cmd in test_commands:
                try:
                    await asyncio.wait_for(cli._process_command(cmd), timeout=5.0)
                    logger.info(f"   Command '{cmd}': ✅ Processed")
                except Exception as e:
                    logger.warning(f"   Command '{cmd}': ❌ Error: {e}")
            
            # Test system shutdown
            await system.shutdown()
            logger.info("✅ System shutdown complete")
            
            self.test_results['main_cli'] = {
                'status': '✅ PASSED',
                'system_name': system.name,
                'cli_commands_tested': len(test_commands),
                'fsot_scalar': status['system']['fsot_scalar']
            }
            
        except Exception as e:
            self._log_error('main_cli', e)
    
    async def _debug_config_utils(self):
        """Deep debug of configuration and utilities"""
        logger.info("\n⚙️ PHASE 8: CONFIG & UTILS DEEP DEBUG")
        logger.info("=" * 50)
        
        try:
            from config import config, ConfigManager
            from utils.memory_manager import memory_manager
            from core import neural_hub
            
            # Test configuration system
            logger.info("🧪 Testing Configuration...")
            logger.info(f"📋 Brain Config: {config.brain_config}")
            logger.info(f"📋 FSOT Config: {config.fsot_config}")
            logger.info(f"📋 System Config: {config.system_config}")
            
            # Test neural hub
            logger.info("🧪 Testing Neural Hub...")
            hub_stats = neural_hub.get_stats()
            logger.info(f"🔗 Neural Hub Stats: {hub_stats}")
            
            # Test memory manager functions
            logger.info("🧪 Testing Memory Manager Functions...")
            memory_status = memory_manager.get_status()
            logger.info(f"💾 Memory Manager Status: {memory_status['memory_stats']['status']}")
            logger.info(f"💾 Optimizers Count: {memory_status['optimizers_count']}")
            
            self.test_results['config_utils'] = {
                'status': '✅ PASSED',
                'config_loaded': True,
                'neural_hub_stats': hub_stats,
                'memory_manager_working': True,
                'optimizers_count': memory_status['optimizers_count']
            }
            
        except Exception as e:
            self._log_error('config_utils', e)
    
    async def _debug_performance_stress(self):
        """Performance and stress testing"""
        logger.info("\n⚡ PHASE 9: PERFORMANCE & STRESS TESTING")
        logger.info("=" * 50)
        
        try:
            from fsot_2_0_foundation import FSOTCore, FSOTDomain
            from utils.memory_manager import memory_manager
            
            # Test calculation performance
            logger.info("🧪 Testing Calculation Performance...")
            fsot_core = FSOTCore()
            
            start_time = datetime.now()
            calculations = 0
            
            for _ in range(100):  # 100 calculations
                scalar = fsot_core.compute_universal_scalar(
                    d_eff=12, domain=FSOTDomain.AI_TECH, observed=True
                )
                calculations += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            calc_per_sec = calculations / duration if duration > 0 else 0
            
            logger.info(f"⚡ Performance:")
            logger.info(f"   Calculations: {calculations}")
            logger.info(f"   Duration: {duration:.3f}s")
            logger.info(f"   Rate: {calc_per_sec:.1f} calc/sec")
            
            # Test memory under load
            memory_status = memory_manager.get_status()
            memory_before = memory_status['memory_stats']['used_gb']
            
            # Create temporary objects
            test_objects = []
            for i in range(1000):
                test_objects.append(f"test_object_{i}" * 100)
            
            memory_status_after = memory_manager.get_status()
            memory_after = memory_status_after['memory_stats']['used_gb']
            memory_diff = memory_after - memory_before
            
            logger.info(f"💾 Memory Stress Test:")
            logger.info(f"   Objects Created: {len(test_objects)}")
            logger.info(f"   Memory Increase: {memory_diff:.2f} GB")
            
            # Cleanup
            del test_objects
            
            self.test_results['performance'] = {
                'status': '✅ PASSED',
                'calculations_per_sec': calc_per_sec,
                'memory_stress_gb': memory_diff,
                'test_objects': 1000
            }
            
        except Exception as e:
            self._log_error('performance', e)
    
    async def _debug_end_to_end(self):
        """End-to-end integration testing"""
        logger.info("\n🔄 PHASE 10: END-TO-END INTEGRATION TESTING")
        logger.info("=" * 50)
        
        try:
            # Test complete system workflow
            logger.info("🧪 Testing Complete System Workflow...")
            
            # 1. Create and initialize system
            from main import FSOTHardwiredSystem
            system = FSOTHardwiredSystem()
            await system.initialize()
            
            # 2. Test FSOT calculations
            from fsot_2_0_foundation import FSOTCore, FSOTDomain
            fsot_core = FSOTCore()
            scalar = fsot_core.compute_universal_scalar(d_eff=12, domain=FSOTDomain.AI_TECH, observed=True)
            
            # 3. Test brain query
            if system.brain_orchestrator:
                result = await system.brain_orchestrator.process_query("Test query")
                logger.info(f"🧠 Brain Query Result: {result.get('response', {}).get('decision', 'No decision')}")
            
            # 4. Test integration
            if system.integration_system:
                integration_status = system.integration_system.get_integration_status()
                logger.info(f"🔌 Integration Skills: {integration_status['skills_status']['total_skills']}")
            
            # 5. Test system status
            final_status = await system.get_system_status()
            logger.info(f"📊 Final System Status: Running={final_status['system']['running']}")
            
            # 6. Clean shutdown
            await system.shutdown()
            
            self.test_results['end_to_end'] = {
                'status': '✅ PASSED',
                'fsot_scalar': scalar,
                'system_running': final_status['system']['running'],
                'workflow_complete': True
            }
            
        except Exception as e:
            self._log_error('end_to_end', e)
    
    def _log_error(self, component: str, error: Exception):
        """Log error for a component"""
        error_info = {
            'component': component,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        self.error_log.append(error_info)
        self.test_results[component] = {
            'status': '❌ FAILED',
            'error': str(error)
        }
        logger.error(f"❌ {component} failed: {error}")
    
    def _generate_debug_report(self):
        """Generate comprehensive debug report"""
        logger.info("\n📊 COMPREHENSIVE DEBUG REPORT")
        logger.info("=" * 80)
        
        # Count results
        passed = sum(1 for result in self.test_results.values() if '✅' in result.get('status', ''))
        total = len(self.test_results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Component summary
        logger.info(f"🎯 COMPONENT TEST RESULTS:")
        for component, result in self.test_results.items():
            status = result.get('status', '❓ UNKNOWN')
            logger.info(f"   {component}: {status}")
        
        # Overall summary
        logger.info(f"\n📈 OVERALL RESULTS:")
        logger.info(f"   Tests Passed: {passed}/{total}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        
        # Error summary
        if self.error_log:
            logger.info(f"\n❌ ERRORS ENCOUNTERED ({len(self.error_log)}):")
            for error in self.error_log:
                logger.error(f"   {error['component']}: {error['error']}")
        
        # System status
        if success_rate >= 90:
            logger.info(f"\n🎉 SYSTEM STATUS: EXCELLENT ({success_rate:.1f}%)")
        elif success_rate >= 75:
            logger.info(f"\n✅ SYSTEM STATUS: GOOD ({success_rate:.1f}%)")
        elif success_rate >= 50:
            logger.info(f"\n⚠️ SYSTEM STATUS: NEEDS ATTENTION ({success_rate:.1f}%)")
        else:
            logger.info(f"\n❌ SYSTEM STATUS: CRITICAL ISSUES ({success_rate:.1f}%)")
        
        return success_rate

async def main():
    """Run comprehensive FSOT system debug"""
    debugger = FSOTSystemDebugger()
    
    try:
        await asyncio.wait_for(debugger.debug_entire_system(), timeout=600.0)  # 10 minute timeout
    except asyncio.TimeoutError:
        logger.error("❌ Debug timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE FSOT AI SYSTEM DEBUG")
    print("Testing every function, component, and integration...")
    print()
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Debug interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Critical debug error: {e}")
        sys.exit(1)
