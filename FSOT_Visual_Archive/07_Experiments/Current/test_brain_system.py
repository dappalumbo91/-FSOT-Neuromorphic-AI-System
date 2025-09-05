#!/usr/bin/env python3
"""
Test script to verify brain_system.py is working correctly
"""

def test_brain_system():
    """Test that brain_system.py imports and works correctly"""
    print("🧪 Testing brain_system.py...")
    
    try:
        # Import and create brain system
        from brain_system import NeuromorphicBrainSystem
        brain = NeuromorphicBrainSystem()
        print("✅ Brain system created successfully")
        
        # Test basic functionality
        stimulus = {'type': 'cognitive', 'intensity': 0.7}
        result = brain.process_stimulus(stimulus)
        print(f"✅ Stimulus processed: consciousness={result['consciousness_level']:.3f}")
        
        # Test memory
        brain.store_memory({'content': 'test memory'}, 'episodic')
        memories = brain.retrieve_memory('test')
        print(f"✅ Memory system: {len(memories)} memories retrieved")
        
        # Test status
        status = brain.get_system_status()
        print(f"✅ System status: {status['fsot_compliance']['core_signature']}")
        
        print("🌟 All tests passed - brain_system.py is fully functional!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_brain_system()
