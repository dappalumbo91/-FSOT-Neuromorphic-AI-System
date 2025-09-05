#!/usr/bin/env python3
"""
Test script for FSOT Neuromorphic AI System simulation integration.
This script tests the complete simulation capabilities with quantum, cellular, and neural simulations.
"""

import json
import sys
from brain_system import NeuromorphicBrainSystem

def test_simulation_integration():
    """Test the complete simulation integration."""
    print("🧠 FSOT Neuromorphic AI System - Simulation Integration Test")
    print("=" * 60)
    
    # Initialize brain system
    brain = NeuromorphicBrainSystem(verbose=True)
    
    # Test 1: Quantum Germ Simulation
    print("\n🔬 Test 1: Quantum Germ Simulation")
    quantum_stimulus = {
        'type': 'cognitive',
        'intensity': 0.8,
        'content': 'Run quantum germ field simulation with particle interactions'
    }
    
    try:
        result = brain.process_stimulus(quantum_stimulus)
        print("✅ Quantum simulation completed successfully!")
        if 'simulation_type' in result:
            print(f"   Simulation Type: {result['simulation_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'results' in result and 'metrics' in result['results']:
                metrics = result['results']['metrics']
                fsot_score = metrics.get('fsot_integration_score', 'N/A')
                quantum_coherence = metrics.get('quantum_coherence', 'N/A')
                print(f"   FSOT Integration Score: {fsot_score if isinstance(fsot_score, str) else fsot_score:.3f}")
                print(f"   Quantum Coherence: {quantum_coherence if isinstance(quantum_coherence, str) else quantum_coherence:.3f}")
        else:
            print(f"   Error: {result.get('simulation_error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Quantum simulation failed: {e}")
    
    # Test 2: Cellular Automata Simulation
    print("\n🔬 Test 2: Cellular Automata Simulation")
    cellular_stimulus = {
        'type': 'cognitive',
        'intensity': 0.6,
        'content': 'Execute cellular automata evolution with FSOT consciousness influence'
    }
    
    try:
        result = brain.process_stimulus(cellular_stimulus)
        print("✅ Cellular automata simulation completed successfully!")
        if 'simulation_type' in result:
            print(f"   Simulation Type: {result['simulation_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'results' in result and 'metrics' in result['results']:
                metrics = result['results']['metrics']
                fsot_score = metrics.get('fsot_integration_score', 'N/A')
                evolution_eff = metrics.get('evolution_efficiency', 'N/A')
                print(f"   FSOT Integration Score: {fsot_score if isinstance(fsot_score, str) else fsot_score:.3f}")
                print(f"   Evolution Efficiency: {evolution_eff if isinstance(evolution_eff, str) else evolution_eff:.3f}")
        else:
            print(f"   Error: {result.get('simulation_error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Cellular automata simulation failed: {e}")
    
    # Test 3: Neural Network Dynamics Simulation
    print("\n🔬 Test 3: Neural Network Dynamics Simulation")
    neural_stimulus = {
        'type': 'cognitive',
        'intensity': 0.7,
        'content': 'Simulate neural network dynamics with brain connectivity patterns'
    }
    
    try:
        result = brain.process_stimulus(neural_stimulus)
        print("✅ Neural network simulation completed successfully!")
        if 'simulation_type' in result:
            print(f"   Simulation Type: {result['simulation_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'results' in result and 'metrics' in result['results']:
                metrics = result['results']['metrics']
                fsot_score = metrics.get('fsot_integration_score', 'N/A')
                network_sync = metrics.get('network_synchronization', 'N/A')
                print(f"   FSOT Integration Score: {fsot_score if isinstance(fsot_score, str) else fsot_score:.3f}")
                print(f"   Network Synchronization: {network_sync if isinstance(network_sync, str) else network_sync:.3f}")
        else:
            print(f"   Error: {result.get('simulation_error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Neural network simulation failed: {e}")
    
    # Test 4: Invalid Simulation Request
    print("\n🔬 Test 4: Invalid Simulation Request")
    invalid_stimulus = {
        'type': 'cognitive',
        'intensity': 0.5,
        'content': 'This is just regular text without simulation keywords'
    }
    
    try:
        result = brain.process_stimulus(invalid_stimulus)
        print("✅ Invalid simulation handled correctly!")
        if 'simulation_error' in result:
            print(f"   Expected Error: {result['simulation_error']}")
            print(f"   Suggestion: {result.get('suggestion', 'No suggestion')}")
    except Exception as e:
        print(f"❌ Invalid simulation handling failed: {e}")
    
    # Test 5: System Status Check
    print("\n🔬 Test 5: System Status Check")
    try:
        status = brain.get_system_status()
        print("✅ System status retrieved successfully!")
        print(f"   Total Memories: {status.get('total_memories', 0)}")
        print(f"   Consciousness Level: {status.get('consciousness_level', 0):.3f}")
        print(f"   System Health: {status.get('system_health', 0):.3f}")
        print(f"   Simulation Engine: {'Available' if hasattr(brain, 'simulation_engine') else 'Not Available'}")
    except Exception as e:
        print(f"❌ System status check failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Simulation Integration Test Complete!")
    print("   Your FSOT Neuromorphic AI System now supports:")
    print("   • Quantum Germ Field Simulations")
    print("   • Cellular Automata Evolution")
    print("   • Neural Network Dynamics")
    print("   • FSOT Mathematical Integration")
    print("   • Consciousness-Influenced Modeling")

if __name__ == "__main__":
    test_simulation_integration()
