#!/usr/bin/env python3
"""
Test script to verify frontal cortex fixes
"""

import sys
import asyncio
from pathlib import Path

# Add the path to our system
sys.path.insert(0, str(Path(__file__).parent))

def test_frontal_cortex_import():
    """Test that frontal cortex imports correctly"""
    print("🧠 Testing Frontal Cortex Import...")
    
    try:
        from brain.frontal_cortex import FrontalCortex
        print("✅ Import successful")
        
        # Create instance
        frontal_cortex = FrontalCortex()
        print("✅ Instantiation successful")
        
        # Check that all required methods exist
        required_methods = [
            '_update_goals',
            '_apply_inhibition', 
            '_executive_override',
            '_provide_status',
            '_retrieve_from_working_memory',
            '_clear_working_memory'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(frontal_cortex, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print(f"✅ All {len(required_methods)} required methods present")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_frontal_cortex_functionality():
    """Test basic frontal cortex functionality"""
    print("🧪 Testing Frontal Cortex Functionality...")
    
    try:
        from brain.frontal_cortex import FrontalCortex
        from core import NeuralSignal, SignalType, Priority
        
        frontal_cortex = FrontalCortex()
        
        # Test decision making
        decision_signal = NeuralSignal(
            source="test",
            target="frontal_cortex",
            signal_type=SignalType.COGNITIVE,
            data={
                'decision_request': {
                    'options': ['option_a', 'option_b', 'option_c'],
                    'context': {'urgency': 'normal'},
                    'urgency': 'normal'
                }
            }
        )
        
        result = await frontal_cortex._process_signal_impl(decision_signal)
        if result and 'decision' in result.data:
            print("✅ Decision making works")
        else:
            print("❌ Decision making failed")
            return False
        
        # Test working memory
        memory_signal = NeuralSignal(
            source="test",
            target="frontal_cortex", 
            signal_type=SignalType.MEMORY,
            data={
                'store': {
                    'key': 'test_item',
                    'value': 'test_value'
                }
            }
        )
        
        result = await frontal_cortex._process_signal_impl(memory_signal)
        if result and result.data.get('status') == 'stored':
            print("✅ Working memory storage works")
        else:
            print("❌ Working memory storage failed")
            return False
        
        # Test executive functions
        exec_signal = NeuralSignal(
            source="test",
            target="frontal_cortex",
            signal_type=SignalType.EXECUTIVE,
            data={
                'status_request': {}
            }
        )
        
        result = await frontal_cortex._process_signal_impl(exec_signal)
        if result and 'executive_summary' in result.data:
            print("✅ Executive status works")
        else:
            print("❌ Executive status failed")
            return False
        
        print("✅ All functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

async def main():
    """Main test execution"""
    print("🚀 FRONTAL CORTEX METHOD FIX VERIFICATION")
    print("=" * 50)
    
    # Run tests
    import_success = test_frontal_cortex_import()
    functionality_success = await test_frontal_cortex_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   Import & Methods: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   Functionality: {'✅ PASS' if functionality_success else '❌ FAIL'}")
    
    if import_success and functionality_success:
        print("\n🎉 ALL TESTS PASSED! Frontal cortex fixes are working correctly.")
        print("   Pylance attribute access warnings should now be resolved.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
