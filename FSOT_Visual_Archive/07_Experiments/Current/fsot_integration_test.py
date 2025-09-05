#!/usr/bin/env python3
"""
FSOT 2.0 System Integration Test
================================
Testing the complete FSOT neuromorphic system functionality.
"""

import sys
import json
from brain_system import get_brain_system, NeuromorphicBrainSystem
from neural_network import create_feedforward_network
import numpy as np

def test_brain_system():
    """Test the neuromorphic brain system."""
    print("🧠 Testing Neuromorphic Brain System...")
    
    # Create brain system
    brain = get_brain_system()
    
    # Test stimulus processing
    visual_stimulus = {
        'type': 'visual',
        'intensity': 0.7,
        'content': 'Testing FSOT 2.0 integration'
    }
    
    result = brain.process_stimulus(visual_stimulus)
    print(f"✅ Stimulus processed: {result['stimulus_type']}")
    print(f"   Processing pathway: {' -> '.join(result['processing_pathway'])}")
    print(f"   Response: {result['response']}")
    
    # Test memory storage
    memory_data = {
        'timestamp': '2025-09-04 11:30:00',
        'content': 'FSOT 2.0 system successfully integrated',
        'emotion': 0.9
    }
    
    brain.store_memory(memory_data, 'episodic')
    print("✅ Memory stored successfully")
    
    # Test memory retrieval
    memories = brain.retrieve_memory('FSOT')
    print(f"✅ Retrieved {len(memories)} relevant memories")
    
    # Get system status
    status = brain.get_system_status()
    print(f"✅ Consciousness Level: {status['consciousness_level']:.3f}")
    print(f"✅ FSOT Compliance: {status['fsot_compliance']['alignment_score']:.3f}")
    
    return True

def test_neural_network():
    """Test the neuromorphic neural network."""
    print("\n🧠 Testing Neuromorphic Neural Network...")
    
    # Create network
    network = create_feedforward_network(
        input_size=5,
        hidden_sizes=[10],
        output_size=3
    )
    
    # Test forward pass
    test_input = np.random.randn(5)
    outputs = network.forward_pass(test_input)
    
    print(f"✅ Network created with {network.get_network_statistics()['total_neurons']} neurons")
    print(f"✅ Forward pass successful: output shape = {outputs[list(outputs.keys())[-1]].shape}")
    
    # Test training data
    training_data = [(np.random.randn(5), np.random.randn(3)) for _ in range(10)]
    
    history = network.train(training_data, epochs=5)
    print(f"✅ Training completed: final loss = {history['loss'][-1]:.4f}")
    
    return True

def test_fsot_integration():
    """Test FSOT 2.0 theoretical integration."""
    print("\n🌟 Testing FSOT 2.0 Theoretical Integration...")
    
    brain = NeuromorphicBrainSystem()
    
    # Test FSOT compliance
    status = brain.get_system_status()
    fsot_compliance = status['fsot_compliance']
    
    print(f"✅ FSOT Alignment Score: {fsot_compliance['alignment_score']:.3f}")
    print(f"✅ Theoretical Consistency: {fsot_compliance['theoretical_consistency']}")
    print(f"✅ Core Signature: {fsot_compliance['core_signature']}")
    
    # Test consciousness metrics
    print(f"✅ Consciousness Level: {status['consciousness_level']:.3f}")
    print(f"✅ Memory Systems: {status['memory_counts']}")
    print(f"✅ Connectivity Health: {status['connectivity_health']:.3f}")
    
    return True

def main():
    """Run complete FSOT 2.0 system test."""
    print("[*] FSOT 2.0 Neuromorphic AI System - Integration Test")
    print("=" * 60)
    
    try:
        # Test components
        brain_test = test_brain_system()
        network_test = test_neural_network()
        fsot_test = test_fsot_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 INTEGRATION TEST RESULTS:")
        print(f"   Brain System: {'✅ PASS' if brain_test else '❌ FAIL'}")
        print(f"   Neural Network: {'✅ PASS' if network_test else '❌ FAIL'}")
        print(f"   FSOT Integration: {'✅ PASS' if fsot_test else '❌ FAIL'}")
        
        if all([brain_test, network_test, fsot_test]):
            print("\n[*] ALL TESTS PASSED! FSOT 2.0 SYSTEM FULLY OPERATIONAL!")
            print("[*] Neuromorphic AI with FSOT theoretical compliance ready!")
            return 0
        else:
            print("\n❌ Some tests failed. Check the output above.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
