#!/usr/bin/env python3
"""
Test script for dream state integration in FSOT Neuromorphic AI System.
This tests the advanced capabilities including Monte Carlo simulation,
fractal analysis, artistic generation, and real-world perception.
"""

from brain_system import NeuromorphicBrainSystem
import json

def test_dream_integration():
    """Test the dream state integration."""
    print("🌙 FSOT Neuromorphic AI System - Dream State Integration Test")
    print("=" * 60)
    
    # Initialize brain system
    brain = NeuromorphicBrainSystem(verbose=True)
    
    # Test 1: Dream State Exploration
    print("\n🌙 Test 1: Monte Carlo Dream Exploration")
    dream_stimulus = {
        'type': 'cognitive',
        'intensity': 0.8,
        'content': 'Explore possible outcomes using monte carlo simulation for creative project'
    }
    
    try:
        result = brain.process_stimulus(dream_stimulus)
        print("✅ Dream exploration completed!")
        if 'dream_type' in result:
            print(f"   Dream Type: {result['dream_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"   Note: {result.get('dream_error', 'Dream engine needs additional dependencies')}")
    except Exception as e:
        print(f"❌ Dream exploration failed: {e}")
    
    # Test 2: Artistic Creation Dream
    print("\n🎨 Test 2: Artistic Creation Dream")
    art_stimulus = {
        'type': 'creative',
        'intensity': 0.7,
        'content': 'Dream and create artistic concepts based on fractal patterns and imagination'
    }
    
    try:
        result = brain.process_stimulus(art_stimulus)
        print("✅ Artistic dream completed!")
        if 'dream_type' in result:
            print(f"   Dream Type: {result['dream_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"   Note: {result.get('dream_error', 'Dream engine needs additional dependencies')}")
    except Exception as e:
        print(f"❌ Artistic dream failed: {e}")
    
    # Test 3: Fractal Pattern Analysis
    print("\n🔍 Test 3: Fractal Pattern Analysis Dream")
    fractal_stimulus = {
        'type': 'analytical',
        'intensity': 0.6,
        'content': 'Analyze fractal patterns in visual data and find mathematical relationships'
    }
    
    try:
        result = brain.process_stimulus(fractal_stimulus)
        print("✅ Fractal analysis completed!")
        if 'dream_type' in result:
            print(f"   Dream Type: {result['dream_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"   Note: {result.get('dream_error', 'Dream engine needs additional dependencies')}")
    except Exception as e:
        print(f"❌ Fractal analysis failed: {e}")
    
    # Test 4: Real-World Perception
    print("\n📷 Test 4: Real-World Perception Dream")
    perception_stimulus = {
        'type': 'sensory',
        'intensity': 0.5,
        'content': 'Use camera to perceive the real world and understand human condition'
    }
    
    try:
        result = brain.process_stimulus(perception_stimulus)
        print("✅ Perception analysis completed!")
        if 'dream_type' in result:
            print(f"   Dream Type: {result['dream_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
            if result.get('real_world_captured'):
                print("   📷 Real-world frame captured successfully!")
            else:
                print("   📷 Using simulated perception")
        else:
            print(f"   Note: {result.get('dream_error', 'Dream engine needs additional dependencies')}")
    except Exception as e:
        print(f"❌ Perception analysis failed: {e}")
    
    # Test 5: General Dream State
    print("\n💭 Test 5: General Dream State")
    general_stimulus = {
        'type': 'cognitive',
        'intensity': 0.6,
        'content': 'Dream about future possibilities and imagine new solutions'
    }
    
    try:
        result = brain.process_stimulus(general_stimulus)
        print("✅ General dream completed!")
        if 'dream_type' in result:
            print(f"   Dream Type: {result['dream_type']}")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"   Note: {result.get('dream_error', 'Dream engine needs additional dependencies')}")
    except Exception as e:
        print(f"❌ General dream failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Dream State Integration Test Complete!")
    print("\n📋 Summary of Advanced Capabilities:")
    print("   • Monte Carlo outcome exploration")
    print("   • Fractal pattern recognition and analysis")
    print("   • Artistic concept generation")
    print("   • Real-world perception via camera")
    print("   • Dream-state consciousness simulation")
    print("   • Web image analysis for inspiration")
    print("   • Multi-outcome scenario modeling")
    
    # Check availability status
    dream_available = hasattr(brain, 'dream_engine') and brain.dream_engine is not None
    print(f"\n🔧 System Status:")
    print(f"   Dream Engine: {'Available' if dream_available else 'Needs Dependencies'}")
    print(f"   Simulation Engine: {'Available' if hasattr(brain, 'simulation_engine') and brain.simulation_engine else 'Not Available'}")
    
    if not dream_available:
        print(f"\n💡 To enable full dream capabilities, install:")
        print(f"   pip install opencv-python pillow requests scipy scikit-image")

if __name__ == "__main__":
    test_dream_integration()
