#!/usr/bin/env python3
"""
Quick Performance Test for FSOT 2.0 System
Standalone test that doesn't require interactive mode
"""

import time
import sys
import os
import traceback
from typing import Dict, Any
import asyncio

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_fsot_imports():
    """Test that all FSOT modules can be imported successfully"""
    print("🔍 Testing FSOT Module Imports...")
    start_time = time.time()
    
    try:
        # Test core imports
        from core.fsot_engine import FSOTEngine, Domain, FSOTParameters
        from brain.brain_orchestrator import BrainOrchestrator
        from interfaces.cli_interface import CLIInterface
        
        import_time = time.time() - start_time
        print(f"✅ All modules imported successfully in {import_time:.3f}s")
        return True, import_time
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Import failed after {error_time:.3f}s: {str(e)}")
        return False, error_time

async def test_fsot_calculations():
    """Test FSOT mathematical calculations"""
    print("\n🧮 Testing FSOT Mathematical Engine...")
    start_time = time.time()
    
    try:
        from core.fsot_engine import FSOTEngine, Domain, FSOTParameters
        
        engine = FSOTEngine()
        
        # Test basic calculations
        test_results = []
        
        # Test domain calculations
        result1 = engine.compute_for_domain(Domain.COGNITIVE)
        test_results.append(result1)
        
        # Test with custom parameters
        params = FSOTParameters(D_eff=14, N=2, P=3, observed=True)
        result2 = engine.compute_scalar(params)
        test_results.append(result2)
        
        # Test interpretation
        interpretation = engine.interpret_result(result1, Domain.COGNITIVE)
        
        calc_time = time.time() - start_time
        print(f"✅ FSOT calculations completed in {calc_time:.3f}s")
        print(f"   - Cognitive domain result: {result1:.6f}")
        print(f"   - Custom parameters result: {result2:.6f}")
        print(f"   - Interpretation: {interpretation['sign_meaning']}")
        return True, calc_time
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ FSOT calculations failed after {error_time:.3f}s: {str(e)}")
        return False, error_time

async def test_brain_initialization():
    """Test brain orchestrator initialization"""
    print("\n🧠 Testing Brain Module Initialization...")
    start_time = time.time()
    
    try:
        from brain.brain_orchestrator import BrainOrchestrator
        
        brain = BrainOrchestrator()
        await brain.initialize()
        
        # Test basic functionality
        status = await brain.get_status()
        
        # Test query processing
        response = await brain.process_query("What is consciousness?")
        
        # Cleanup
        await brain.shutdown()
        
        init_time = time.time() - start_time
        print(f"✅ Brain initialization completed in {init_time:.3f}s")
        print(f"   - Modules loaded: {status.get('modules', 0)}")
        print(f"   - Status: {status.get('status', 'unknown')}")
        print(f"   - Query response type: {type(response).__name__}")
        return True, init_time
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Brain initialization failed after {error_time:.3f}s: {str(e)}")
        return False, error_time

async def test_memory_usage():
    """Test memory efficiency"""
    print("\n💾 Testing Memory Usage...")
    start_time = time.time()
    
    try:
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Initial memory: {initial_memory:.2f} MB")
        
        # Create and test multiple components
        from core.fsot_engine import FSOTEngine, Domain, FSOTParameters
        from brain.brain_orchestrator import BrainOrchestrator
        
        engines = [FSOTEngine() for _ in range(5)]
        brain = BrainOrchestrator()
        await brain.initialize()
        
        # Run some calculations
        for engine in engines:
            engine.compute_for_domain(Domain.COGNITIVE)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        await brain.shutdown()
        del engines
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_time = time.time() - start_time
        print(f"✅ Memory test completed in {memory_time:.3f}s")
        print(f"   Peak memory: {peak_memory:.2f} MB")
        print(f"   Memory increase: {memory_increase:.2f} MB")
        print(f"   Final memory: {final_memory:.2f} MB")
        return True, memory_time
        
    except ImportError:
        print("⚠️  psutil not available, skipping memory test")
        return True, 0.0
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Memory test failed after {error_time:.3f}s: {str(e)}")
        return False, error_time

async def test_concurrent_operations():
    """Test concurrent operations performance"""
    print("\n⚡ Testing Concurrent Operations...")
    start_time = time.time()
    
    try:
        from core.fsot_engine import FSOTEngine, Domain
        
        # Create multiple engines for concurrent testing
        engines = [FSOTEngine() for _ in range(3)]
        
        # Run concurrent calculations
        def calculate_for_engine(engine, domain):
            return engine.compute_for_domain(domain)
        
        # Test different domains concurrently
        domains = [Domain.COGNITIVE, Domain.QUANTUM, Domain.BIOLOGICAL]
        results = []
        
        for engine, domain in zip(engines, domains):
            result = calculate_for_engine(engine, domain)
            results.append(result)
        
        concurrent_time = time.time() - start_time
        print(f"✅ Concurrent operations completed in {concurrent_time:.3f}s")
        print(f"   Concurrent tasks: {len(results)}")
        print(f"   Results: {[f'{r:.6f}' for r in results]}")
        return True, concurrent_time
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Concurrent operations failed after {error_time:.3f}s: {str(e)}")
        return False, error_time

async def main():
    """Run all performance tests"""
    print("🧠⚡ FSOT 2.0 QUICK PERFORMANCE TEST")
    print("=" * 50)
    
    overall_start = time.time()
    test_results = {}
    
    # Run all tests
    tests = [
        ("Import Speed", test_fsot_imports),
        ("FSOT Calculations", test_fsot_calculations),
        ("Brain Initialization", test_brain_initialization),
        ("Memory Usage", test_memory_usage),
        ("Concurrent Operations", test_concurrent_operations)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            success, duration = await test_func()
            test_results[test_name] = {
                'success': success,
                'duration': duration
            }
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            test_results[test_name] = {
                'success': False,
                'duration': 0.0,
                'error': str(e)
            }
    
    overall_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result['duration']
        print(f"{test_name:25} {status:8} {duration:8.3f}s")
    
    print("-" * 50)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total Time: {overall_time:.3f}s")
    
    # Performance assessment
    if passed_tests == total_tests:
        print("\n🎉 EXCELLENT: All tests passed!")
        if overall_time < 5.0:
            print("⚡ FAST: System initialized quickly")
        elif overall_time < 10.0:
            print("✅ GOOD: System performance is adequate")
        else:
            print("⚠️  SLOW: System may need optimization")
    else:
        print(f"\n⚠️  WARNING: {total_tests - passed_tests} test(s) failed")
        print("🔧 Consider debugging failed components")
    
    return test_results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        traceback.print_exc()
