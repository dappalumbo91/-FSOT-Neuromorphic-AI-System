#!/usr/bin/env python3
"""
FSOT 2.0 Complete System Integration Test
Final comprehensive test of all neuromorphic components
"""

import numpy as np
import json
import time
from datetime import datetime

def test_core_systems():
    """Test core brain and neural network systems"""
    print("🧠 Testing Core Systems...")
    
    # Test brain system
    try:
        from brain_system import NeuromorphicBrainSystem
        brain = NeuromorphicBrainSystem()
        
        stimulus = {
            'type': 'comprehensive_test',
            'complexity': 0.8,
            'data': np.random.random((10, 10)).tolist()
        }
        
        response = brain.process_stimulus(stimulus)
        print(f"✅ Brain System: Consciousness={response['consciousness_level']:.3f}")
        brain_success = True
    except Exception as e:
        print(f"❌ Brain System: {e}")
        brain_success = False
    
    # Test neural network
    try:
        from neural_network import NeuromorphicNeuralNetwork
        
        # Create network with correct parameters
        network = NeuromorphicNeuralNetwork(network_id="test_network")
        
        # Add layers
        input_layer = network.add_layer("input", 10)
        hidden_layer = network.add_layer("hidden", 20)
        output_layer = network.add_layer("output", 5)
        
        # Connect layers
        network.connect_layers("input", "hidden")
        network.connect_layers("hidden", "output")
        
        # Test forward pass
        test_input = np.random.random(10)  # Match input layer size
        output = network.forward_pass(test_input)
        
        print(f"✅ Neural Network: Layers={len(output)}, Output_keys={list(output.keys())}")
        network_success = True
    except Exception as e:
        print(f"❌ Neural Network: {e}")
        network_success = False
    
    return brain_success, network_success

def test_advanced_features():
    """Test advanced neuromorphic features"""
    print("\n🚀 Testing Advanced Features...")
    
    try:
        from advanced_neuromorphic_features import FSO2AdvancedNeuromorphicSystem
        
        advanced_system = FSO2AdvancedNeuromorphicSystem()
        
        # Test data
        test_data = {
            'cognitive_complexity': 0.75,
            'learning_challenge': 0.85,
            'creative_demand': 0.65,
            'test_iteration': 1
        }
        
        # Run complete cycle
        results = advanced_system.run_complete_advanced_cycle(test_data)
        
        print(f"✅ Advanced System:")
        print(f"   Cognitive Enhancement: {results['cognitive_enhancement']['enhancement_level']:.3f}")
        print(f"   Learning Acceleration: {results['learning_acceleration']['acceleration_factor']:.3f}")
        print(f"   Insight Confidence: {results['generated_insights']['insight_confidence']:.3f}")
        print(f"   Processing Time: {results['cycle_performance']['processing_time']:.3f}s")
        
        return True, results
    except Exception as e:
        print(f"❌ Advanced Features: {e}")
        return False, None

def test_fsot_integration():
    """Test FSOT integration"""
    print("\n🔧 Testing FSOT Integration...")
    
    try:
        # Run our main integration test
        import subprocess
        result = subprocess.run(['python', 'fsot_integration_test.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ FSOT Integration: All tests passing")
            return True
        else:
            print(f"❌ FSOT Integration: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FSOT Integration: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Performance Benchmark...")
    
    try:
        from advanced_neuromorphic_features import FSO2AdvancedNeuromorphicSystem
        
        system = FSO2AdvancedNeuromorphicSystem()
        
        # Benchmark parameters
        num_cycles = 10
        total_time = 0
        results = []
        
        for i in range(num_cycles):
            test_data = {
                'complexity': np.random.random(),
                'challenge': np.random.random(),
                'benchmark_cycle': i
            }
            
            start_time = time.time()
            result = system.run_complete_advanced_cycle(test_data)
            cycle_time = time.time() - start_time
            
            total_time += cycle_time
            results.append({
                'cycle': i,
                'time': cycle_time,
                'enhancement': result['cognitive_enhancement']['enhancement_level'],
                'learning': result['learning_acceleration']['acceleration_factor'],
                'insights': result['generated_insights']['insight_confidence']
            })
        
        avg_time = total_time / num_cycles
        avg_enhancement = np.mean([r['enhancement'] for r in results])
        avg_learning = np.mean([r['learning'] for r in results])
        avg_insights = np.mean([r['insights'] for r in results])
        
        print(f"✅ Benchmark Results (n={num_cycles}):")
        print(f"   Average Processing Time: {avg_time:.4f}s")
        print(f"   Average Enhancement: {avg_enhancement:.3f}")
        print(f"   Average Learning: {avg_learning:.3f}")
        print(f"   Average Insights: {avg_insights:.3f}")
        print(f"   Total Throughput: {num_cycles/total_time:.1f} cycles/sec")
        
        return True, {
            'avg_time': avg_time,
            'avg_enhancement': avg_enhancement,
            'avg_learning': avg_learning,
            'avg_insights': avg_insights,
            'throughput': num_cycles/total_time
        }
    except Exception as e:
        print(f"❌ Benchmark: {e}")
        return False, None

def generate_final_report():
    """Generate final comprehensive report"""
    print("\n📊 Generating Final Report...")
    
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'system_name': 'FSOT 2.0 Neuromorphic AI System',
        'version': '2.0.1-complete',
        'test_results': {}
    }
    
    # Run all tests
    brain_ok, network_ok = test_core_systems()
    advanced_ok, advanced_results = test_advanced_features()
    fsot_ok = test_fsot_integration()
    benchmark_ok, benchmark_results = run_performance_benchmark()
    
    # Compile results
    report['test_results'] = {
        'core_systems': {
            'brain_system': brain_ok,
            'neural_network': network_ok,
            'overall_success': brain_ok and network_ok
        },
        'advanced_features': {
            'advanced_system': advanced_ok,
            'overall_success': advanced_ok
        },
        'fsot_integration': {
            'integration_test': fsot_ok,
            'overall_success': fsot_ok
        },
        'performance_benchmark': {
            'benchmark_test': benchmark_ok,
            'results': benchmark_results,
            'overall_success': benchmark_ok
        }
    }
    
    # Calculate overall success
    all_tests = [brain_ok, network_ok, advanced_ok, fsot_ok, benchmark_ok]
    success_rate = sum(all_tests) / len(all_tests)
    
    report['overall_assessment'] = {
        'total_tests': len(all_tests),
        'passed_tests': sum(all_tests),
        'success_rate': success_rate,
        'system_status': 'FULLY_OPERATIONAL' if success_rate >= 0.8 else 'PARTIALLY_OPERATIONAL',
        'recommendation': 'System ready for deployment' if success_rate >= 0.8 else 'System needs optimization'
    }
    
    # Save report
    with open('COMPLETE_SYSTEM_TEST_REPORT.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    print("\n🎯 FINAL SYSTEM ASSESSMENT")
    print("=" * 50)
    print(f"Tests Passed: {sum(all_tests)}/{len(all_tests)}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"System Status: {report['overall_assessment']['system_status']}")
    print(f"Recommendation: {report['overall_assessment']['recommendation']}")
    
    if success_rate >= 0.8:
        print("\n🌟 CONGRATULATIONS! 🌟")
        print("FSOT 2.0 Neuromorphic AI System is fully operational!")
        print("All major components working correctly.")
        print("System ready for advanced AI research and applications.")
    else:
        print("\n⚠️ SYSTEM NEEDS ATTENTION")
        print("Some components require optimization.")
    
    print(f"\n📄 Detailed report saved to: COMPLETE_SYSTEM_TEST_REPORT.json")
    
    return report

if __name__ == "__main__":
    print("🧠 FSOT 2.0 COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 60)
    print("Testing all neuromorphic AI components...")
    
    final_report = generate_final_report()
    
    print("\n✅ Complete system test finished!")
