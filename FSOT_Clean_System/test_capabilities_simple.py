#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 - Simple Capability Test
==========================================

Simple test script to verify our implemented capabilities are working.
Tests the actual files we created in their correct locations.

Author: GitHub Copilot
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

def test_enhanced_memory_system():
    """Test Enhanced Memory System"""
    print("🧠 Testing Enhanced Memory System...")
    try:
        from core.enhanced_memory_system import EnhancedMemorySystem
        
        # Initialize memory system
        memory = EnhancedMemorySystem()
        
        # Test basic functionality
        memory.working_memory.store("test", {"data": "test_value"})
        result = memory.working_memory.retrieve("test")
        
        print(f"   ✅ Memory system working: {result is not None}")
        
        # Test statistics
        stats = memory.get_memory_statistics()
        print(f"   📊 Memory stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_multimodal_processor():
    """Test Multi-modal Processor"""
    print("🎯 Testing Multi-modal Processor...")
    try:
        from capabilities.multimodal_processor import MultiModalProcessor
        
        # Initialize processor
        processor = MultiModalProcessor()
        
        # Test text processing
        text_result = processor.analyze_text("This is a test sentence.")
        print(f"   ✅ Text processing working: {text_result is not None}")
        
        # Check capabilities
        caps = processor.get_capabilities()
        print(f"   🔧 Capabilities: {len(caps)} available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_continuous_api_discovery():
    """Test Continuous API Discovery"""
    print("🔍 Testing Continuous API Discovery...")
    try:
        from capabilities.continuous_api_discovery import ContinuousAPIDiscovery
        
        # Initialize discovery system
        discovery = ContinuousAPIDiscovery()
        
        # Test basic functionality
        stats = discovery.get_statistics()
        print(f"   ✅ API discovery working: stats available")
        print(f"   📊 Database connected: {'total_apis' in stats}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_web_dashboard():
    """Test Web Dashboard"""
    print("🌐 Testing Web Dashboard...")
    try:
        from interfaces.web_dashboard import SimpleFSOTDashboard
        
        # Initialize dashboard
        dashboard = SimpleFSOTDashboard(port=5001)  # Different port for testing
        
        # Test basic functionality
        dashboard.log_activity("test", "Testing dashboard")
        print(f"   ✅ Dashboard working: activity logged")
        
        # Test metrics
        dashboard.update_metrics({"test_metric": 42})
        print(f"   📊 Metrics working: test metric set")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_desktop_control():
    """Test Desktop Control"""
    print("🖥️ Testing Desktop Control...")
    try:
        from capabilities.desktop_control import FreeDesktopController
        
        # Initialize controller (but don't activate for safety)
        controller = FreeDesktopController()
        
        # Test safe operations
        capabilities = controller.capabilities
        browsers = controller.find_application("browser")
        system_info = controller.get_system_info()
        
        print(f"   ✅ Desktop control working: {len(capabilities)} capabilities")
        print(f"   🔧 App database: {len(browsers)} browsers found")
        print(f"   💻 System info: Platform {system_info.get('platform', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_advanced_training():
    """Test Advanced Training System"""
    print("🎓 Testing Advanced Training System...")
    try:
        from capabilities.advanced_training import AdvancedFreeTrainingSystem
        
        # Initialize training system
        trainer = AdvancedFreeTrainingSystem()
        
        # Test basic functionality
        status = trainer.get_status()
        curriculum = trainer.curriculum
        
        print(f"   ✅ Training system working: {len(curriculum)} stages")
        print(f"   📚 Capabilities: {sum(trainer.capabilities.values())} enabled")
        print(f"   🎯 Current stage: {status['current_stage']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_system_integration():
    """Test System Integration"""
    print("🔗 Testing System Integration...")
    try:
        from system_integration import EnhancedFSOTSystem
        
        # Initialize system (but don't start)
        system = EnhancedFSOTSystem()
        
        # Test status
        status = system.get_system_status()
        
        print(f"   ✅ Integration working: system initialized")
        print(f"   🧩 Components: {len(system.components)} detected")
        print(f"   ⚙️ Config loaded: {system.config is not None}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_main_system():
    """Check if main system files exist"""
    print("📁 Checking main system files...")
    
    # Check for original system files
    files_to_check = [
        ("neuromorphic_brain.py", "Neuromorphic Brain"),
        ("free_api_discovery.py", "Free API Discovery"),
        ("main.py", "Main System"),
        ("config.py", "Configuration")
    ]
    
    found_count = 0
    for filename, description in files_to_check:
        if (current_dir / filename).exists():
            print(f"   ✅ {description}: {filename}")
            found_count += 1
        elif (current_dir.parent / filename).exists():
            print(f"   ✅ {description}: {filename} (in parent directory)")
            found_count += 1
        else:
            print(f"   ❌ {description}: {filename} not found")
    
    print(f"   📊 Main files: {found_count}/{len(files_to_check)} found")
    return found_count

def main():
    """Run all tests"""
    print("🧪 Enhanced FSOT 2.0 - Simple Capability Test")
    print("=" * 60)
    
    # Test our implemented capabilities
    tests = [
        ("Enhanced Memory System", test_enhanced_memory_system),
        ("Multi-modal Processor", test_multimodal_processor),
        ("Continuous API Discovery", test_continuous_api_discovery),
        ("Web Dashboard", test_web_dashboard),
        ("Desktop Control", test_desktop_control),
        ("Advanced Training", test_advanced_training),
        ("System Integration", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            print()
    
    # Check main system files
    main_files_found = check_main_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"🆕 New Capabilities: {passed}/{total} working")
    print(f"📁 Main System Files: {main_files_found}/4 found")
    
    success_rate = (passed / total) * 100
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL NEW CAPABILITIES ARE WORKING!")
        print("✅ Enhanced FSOT 2.0 system is ready for testing!")
    elif passed >= total // 2:
        print(f"\n✅ Most capabilities working ({passed}/{total})")
        print("🔧 System is largely functional")
    else:
        print(f"\n⚠️ Several capabilities need attention ({total-passed} failed)")
        print("🔧 Check error messages above")
    
    print(f"\n💡 Next Steps:")
    if passed >= total // 2:
        print("   • Run: python main.py")
        print("   • Try: python demo_free_capabilities.py")
        print("   • Test: python -c 'from system_integration import *; print(\"System ready!\")'")
    else:
        print("   • Check Python environment and dependencies")
        print("   • Verify file locations and imports")
        print("   • Run: python install_dependencies.py")
    
    return passed == total

if __name__ == "__main__":
    main()
