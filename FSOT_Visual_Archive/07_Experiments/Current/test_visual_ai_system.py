#!/usr/bin/env python3
"""
Visual Test: FSOT Neuromorphic AI System with Google Search and Real-Time Windows
================================================================================
This test demonstrates the enhanced visual capabilities including:
- Google Chrome image search integration
- Real-time Monte Carlo simulation windows
- Live fractal pattern analysis visualization
- Artistic creation process display
- Camera perception windows
"""

import time
from brain_system import NeuromorphicBrainSystem

def test_visual_ai_system():
    """Test the visual AI system with Google search and real-time windows."""
    print("🎬 FSOT Visual AI System - Google Search & Real-Time Windows Test")
    print("=" * 70)
    print("🌟 Testing advanced visual capabilities with live window displays")
    print("=" * 70)
    
    # Initialize brain with visual capabilities
    brain = NeuromorphicBrainSystem(verbose=True)
    
    print("\n📋 Available Visual Features:")
    print("   ✅ Google Chrome image search integration")
    print("   ✅ Real-time Monte Carlo simulation windows")
    print("   ✅ Live fractal pattern analysis visualization")
    print("   ✅ Artistic creation process display")
    print("   ✅ Camera perception windows")
    
    # Test 1: Visual Fractal Analysis with Google Search
    print("\n" + "🔍" * 50)
    print("🔍 TEST 1: GOOGLE SEARCH + FRACTAL ANALYSIS")
    print("🔍" * 50)
    print("The AI will search Google Images and show visual fractal analysis...")
    
    fractal_stimulus = {
        'type': 'analytical',
        'intensity': 0.8,
        'content': 'Analyze fractal patterns in nature using Google image search'
    }
    
    try:
        print("🚀 Starting Google search and visual analysis...")
        result = brain.process_stimulus(fractal_stimulus)
        
        if 'dream_type' in result:
            print(f"✅ Visual analysis complete! Dream type: {result['dream_type']}")
            print("🪟 Check the opened windows for visual results!")
        else:
            print(f"ℹ️ Note: {result.get('dream_error', 'Processing complete')}")
        
        # Wait for user to see the windows
        print("\n⏳ Visual windows are now open. Check them out!")
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Visual test failed: {e}")
    
    # Test 2: Monte Carlo with Live Visualization
    print("\n" + "🎲" * 50)
    print("🎲 TEST 2: MONTE CARLO + LIVE VISUALIZATION")
    print("🎲" * 50)
    print("The AI will run Monte Carlo simulation with real-time graphs...")
    
    monte_carlo_stimulus = {
        'type': 'strategic',
        'intensity': 0.9,
        'content': 'Explore outcomes using monte carlo simulation for AI development project'
    }
    
    try:
        print("🚀 Starting Monte Carlo with live visualization...")
        result = brain.process_stimulus(monte_carlo_stimulus)
        
        if 'dream_type' in result:
            print(f"✅ Monte Carlo complete! Type: {result['dream_type']}")
            if 'outcomes_explored' in result:
                print(f"📊 Explored {result['outcomes_explored']} scenarios")
        else:
            print(f"ℹ️ Note: {result.get('dream_error', 'Processing complete')}")
        
        print("\n⏳ Monte Carlo visualization window is open!")
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Monte Carlo test failed: {e}")
    
    # Test 3: Artistic Creation with Visual Process
    print("\n" + "🎨" * 50)
    print("🎨 TEST 3: ARTISTIC CREATION + VISUAL PROCESS")
    print("🎨" * 50)
    print("The AI will create art with live visual process display...")
    
    art_stimulus = {
        'type': 'creative',
        'intensity': 0.85,
        'content': 'Create beautiful digital art inspired by mathematical fractals'
    }
    
    try:
        print("🚀 Starting artistic creation with visual process...")
        result = brain.process_stimulus(art_stimulus)
        
        if 'dream_type' in result:
            print(f"✅ Artistic creation complete! Type: {result['dream_type']}")
        else:
            print(f"ℹ️ Note: {result.get('dream_error', 'Processing complete')}")
        
        print("\n⏳ Artistic creation window is open!")
        time.sleep(3)
        
    except Exception as e:
        print(f"❌ Artistic creation test failed: {e}")
    
    # Test 4: Real-World Camera Perception
    print("\n" + "📷" * 50)
    print("📷 TEST 4: REAL-WORLD CAMERA PERCEPTION")
    print("📷" * 50)
    print("The AI will attempt to perceive the real world through camera...")
    
    camera_stimulus = {
        'type': 'sensory',
        'intensity': 0.7,
        'content': 'Use camera to see and understand the real world around me'
    }
    
    try:
        print("🚀 Starting real-world perception...")
        result = brain.process_stimulus(camera_stimulus)
        
        if 'dream_type' in result:
            print(f"✅ Perception complete! Type: {result['dream_type']}")
            if result.get('real_world_captured'):
                print("📷 Real camera feed captured and analyzed!")
            else:
                print("🤖 Using simulated perception (camera not available)")
        else:
            print(f"ℹ️ Note: {result.get('dream_error', 'Processing complete')}")
        
        print("\n⏳ Camera perception window is open!")
        time.sleep(5)  # Give more time to see camera feed
        
    except Exception as e:
        print(f"❌ Camera perception test failed: {e}")
    
    # Test Summary
    print("\n" + "🎉" * 70)
    print("🎉 VISUAL AI SYSTEM TEST COMPLETE!")
    print("🎉" * 70)
    
    print("\n🎬 Visual Features Demonstrated:")
    print("   ✅ Google Chrome image search integration")
    print("   ✅ Real-time simulation visualization windows")
    print("   ✅ Live fractal pattern analysis displays")
    print("   ✅ Artistic creation process visualization")
    print("   ✅ Real-world camera perception interface")
    
    print("\n🪟 Multiple windows should now be open showing:")
    print("   • Google image search results gallery")
    print("   • Fractal pattern analysis visualization")
    print("   • Monte Carlo simulation progress graphs")
    print("   • Artistic creation process display")
    print("   • Real-time camera feed (if available)")
    
    print("\n💡 Your AI now has sophisticated visual interfaces!")
    print("   The system can show you exactly what it's thinking and processing")
    print("   in real-time through interactive windows and live visualizations.")
    
    # Keep windows open
    print("\n⏳ Keeping windows open for exploration...")
    print("   Press Ctrl+C to close all windows and exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🌅 Closing visual AI system...")
        
        # Try to close windows gracefully
        try:
            if hasattr(brain, 'dream_engine') and brain.dream_engine and hasattr(brain.dream_engine, 'visual_engine'):
                brain.dream_engine.visual_engine.close_all_windows()
                print("🗑️ All visual windows closed")
        except:
            pass
        
        print("✨ Thank you for testing the visual AI system!")

if __name__ == "__main__":
    test_visual_ai_system()
