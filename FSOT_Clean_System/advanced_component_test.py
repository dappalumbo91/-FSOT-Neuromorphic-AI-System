#!/usr/bin/env python3
"""
Advanced Component Testing for FSOT System
Detailed testing of individual components
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_brain_orchestrator_detailed():
    """Detailed brain orchestrator testing"""
    print("\n🧠 ADVANCED BRAIN ORCHESTRATOR TESTING")
    print("=" * 50)
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        
        brain = BrainOrchestrator()
        print("✅ Brain orchestrator created")
        
        # Initialize with timeout
        await asyncio.wait_for(brain.initialize(), timeout=30.0)
        print("✅ Brain orchestrator initialized")
        
        # Get detailed status
        status = await brain.get_status()
        print(f"🧠 Brain Status:")
        print(f"   Modules: {len(status.get('modules', {}))}")
        print(f"   Initialized: {status.get('initialized', False)}")
        print(f"   Activation: {status.get('overall_activation', 0):.1%}")
        print(f"   Processing Load: {status.get('processing_load', 0):.1%}")
        print(f"   Queries Processed: {status.get('queries_processed', 0)}")
        
        # Test module listing
        modules = status.get('modules', {})
        print(f"\n🧩 Brain Modules ({len(modules)}):")
        for name, module_info in modules.items():
            active = "🟢" if module_info.get('is_active') else "🔴"
            activation = module_info.get('activation_level', 0)
            print(f"   {active} {name}: {activation:.1%} activation")
        
        # Test a simple query
        print(f"\n💭 Testing query processing...")
        result = await asyncio.wait_for(
            brain.process_query("What is consciousness?"), 
            timeout=30.0
        )
        print(f"✅ Query processed successfully")
        print(f"   Response available: {'response' in result}")
        print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
        
        # Cleanup
        await brain.shutdown()
        print("✅ Brain orchestrator shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Brain orchestrator test failed: {e}")
        return False

async def test_integration_system_detailed():
    """Detailed integration system testing"""
    print("\n🔌 ADVANCED INTEGRATION SYSTEM TESTING")
    print("=" * 50)
    
    try:
        from integration import EnhancedFSOTIntegration
        from brain.brain_orchestrator import BrainOrchestrator
        
        # Create brain for integration
        brain = BrainOrchestrator()
        await brain.initialize()
        
        # Create integration system
        integration = EnhancedFSOTIntegration(brain)
        print("✅ Integration system created")
        
        # Get detailed status
        status = integration.get_integration_status()
        print(f"🔌 Integration Status:")
        print(f"   Initialized: {status['system_info']['initialized']}")
        print(f"   Learning Active: {status['system_info']['learning_active']}")
        print(f"   Total Skills: {status['skills_status']['total_skills']}")
        print(f"   Skill Categories: {len(status['skills_status']['category_breakdown'])}")
        
        # Test API status
        api_status = status['api_status']
        print(f"\n📡 API Status:")
        for api_name, api_info in api_status.items():
            enabled = "✅" if api_info.get('enabled') else "❌"
            configured = "🔧" if api_info.get('configured') else "⚙️"
            print(f"   {enabled} {configured} {api_name}")
        
        # Test skills
        skills_info = status['skills_status']
        print(f"\n🎯 Skills Breakdown:")
        for category, info in skills_info['category_breakdown'].items():
            count = info['count']
            avg_prof = info['average_proficiency']
            print(f"   {category}: {count} skills, {avg_prof:.1%} avg proficiency")
        
        # Test recommendations
        recommendations = integration.get_recommendations()
        print(f"\n💡 Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec['title']} ({rec['priority']} priority)")
        
        # Cleanup
        integration.shutdown()
        await brain.shutdown()
        print("✅ Integration system test complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration system test failed: {e}")
        return False

async def test_fsot_theoretical_integration():
    """Test FSOT theoretical integration"""
    print("\n🔬 ADVANCED FSOT THEORETICAL TESTING")
    print("=" * 50)
    
    try:
        from fsot_theoretical_integration import FSOTNeuromorphicIntegrator
        from fsot_2_0_foundation import FSOTCore, FSOTDomain
        
        # Create integrator
        integrator = FSOTNeuromorphicIntegrator()
        print("✅ FSOT integrator created")
        
        # Test various dimensional calculations
        test_cases = [
            (10, FSOTDomain.NEURAL, "Basic neural processing"),
            (12, FSOTDomain.AI_TECH, "AI technology optimal"),
            (14, FSOTDomain.COGNITIVE, "Consciousness emergence"),
            (16, FSOTDomain.COGNITIVE, "High consciousness"),
            (8, FSOTDomain.NEURAL, "Simple neural"),
            (25, FSOTDomain.AI_TECH, "Maximum dimensions")
        ]
        
        print(f"\n📊 FSOT Scalar Calculations:")
        fsot_core = FSOTCore()
        
        for d_eff, domain, description in test_cases:
            scalar = fsot_core.compute_universal_scalar(
                d_eff=d_eff,
                domain=domain,
                observed=True
            )
            mode = "🌟 EMERGING" if scalar > 0 else "🛡️ DAMPED" if scalar < 0 else "⚖️ BALANCED"
            print(f"   {description}: D_eff={d_eff}, Scalar={scalar:.6f} {mode}")
        
        # Test theoretical report generation
        print(f"\n📄 Generating theoretical report...")
        report = integrator.generate_theoretical_report()
        print(f"✅ Report generated ({len(report)} characters)")
        
        # Test consciousness calculations
        print(f"\n🧠 Consciousness Testing:")
        from fsot_theoretical_integration import calculate_consciousness_scalar
        
        consciousness_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for level in consciousness_levels:
            c_scalar = calculate_consciousness_scalar(level)
            print(f"   Level {level:.1f}: Scalar={c_scalar:.6f}")
        
        print("✅ FSOT theoretical integration test complete")
        return True
        
    except Exception as e:
        print(f"❌ FSOT theoretical test failed: {e}")
        return False

async def test_memory_and_consciousness():
    """Test memory manager and consciousness monitor"""
    print("\n💾🌟 MEMORY & CONSCIOUSNESS TESTING")
    print("=" * 50)
    
    try:
        from utils.memory_manager import memory_manager
        from core import consciousness_monitor
        
        # Test memory manager
        print("💾 Memory Manager Testing:")
        memory_manager.start_monitoring()
        print("✅ Memory monitoring started")
        
        status = memory_manager.get_status()
        mem_stats = status['memory_stats']
        print(f"   Total Memory: {mem_stats['total_gb']:.1f} GB")
        print(f"   Available: {mem_stats['available_gb']:.1f} GB")
        print(f"   Used: {mem_stats['used_gb']:.1f} GB")
        print(f"   Usage: {mem_stats['percentage']:.1f}%")
        print(f"   Status: {mem_stats['status']}")
        
        # Test consciousness monitor
        print(f"\n🌟 Consciousness Monitor Testing:")
        await consciousness_monitor.start_monitoring()
        print("✅ Consciousness monitoring started")
        
        state = consciousness_monitor.get_current_state()
        print(f"   Level: {state['consciousness_level']:.1%}")
        print(f"   State: {state['state']}")
        print(f"   Coherence: {state['coherence']:.1%}")
        print(f"   Focus: {state['attention_focus']:.1%}")
        
        # Test consciousness update
        consciousness_monitor.update_processing_load(0.5)
        print("✅ Consciousness updated with processing load")
        
        # Cleanup
        await consciousness_monitor.stop_monitoring()
        print("✅ Memory & consciousness test complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory & consciousness test failed: {e}")
        return False

async def main():
    """Run all advanced component tests"""
    print("🚀 ADVANCED COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("Brain Orchestrator", test_brain_orchestrator_detailed),
        ("Integration System", test_integration_system_detailed),
        ("FSOT Theoretical", test_fsot_theoretical_integration),
        ("Memory & Consciousness", test_memory_and_consciousness)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = await asyncio.wait_for(test_func(), timeout=120.0)
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except asyncio.TimeoutError:
            results[test_name] = "⏰ TIMEOUT"
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    # Results summary
    print(f"\n📊 ADVANCED TESTING RESULTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    passed = sum(1 for r in results.values() if "✅" in r)
    total = len(results)
    
    print(f"\nAdvanced Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
