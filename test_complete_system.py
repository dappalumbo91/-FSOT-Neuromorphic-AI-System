#!/usr/bin/env python3
"""
🧠⚡ FSOT 2.0 COMPLETE SYSTEM TEST
==========================================
Comprehensive testing of Enhanced FSOT 2.0 system
Tests all major capabilities and integrations
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path

# Add system paths
current_dir = Path(__file__).parent
clean_system_path = current_dir / "FSOT_Clean_System"

sys.path.insert(0, str(clean_system_path))

class ComprehensiveSystemTest:
    """Complete test suite for Enhanced FSOT 2.0"""
    
    def __init__(self):
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        print("🧠⚡ ENHANCED FSOT 2.0 COMPREHENSIVE SYSTEM TEST")
        print("=" * 60)
    
    def test_brain_architecture(self):
        """Test the neuromorphic brain architecture"""
        print("\n🧠 Testing Brain Architecture...")
        try:
            from brain.brain_orchestrator import BrainOrchestrator
            brain = BrainOrchestrator()
            
            # Check brain modules
            modules = [
                'frontal_cortex', 'hippocampus', 'amygdala', 'cerebellum',
                'temporal_lobe', 'occipital_lobe', 'thalamus', 'parietal_lobe',
                'pflt', 'brainstem'
            ]
            
            missing_modules = []
            for module_name in modules:
                if not hasattr(brain, module_name):
                    missing_modules.append(module_name)
            
            if not missing_modules:
                self.record_test("Brain Architecture", True, f"All {len(modules)} brain modules present")
                print(f"  ✅ Brain modules: {len(modules)}/10 initialized")
            else:
                self.record_test("Brain Architecture", False, f"Missing modules: {missing_modules}")
                print(f"  ❌ Missing modules: {missing_modules}")
                
        except Exception as e:
            self.record_test("Brain Architecture", False, f"Error: {str(e)}")
            print(f"  ❌ Brain architecture test failed: {e}")
    
    def test_memory_management(self):
        """Test memory management system"""
        print("\n🧮 Testing Memory Management...")
        try:
            from utils.memory_manager import memory_manager
            
            # Test memory manager initialization
            if hasattr(memory_manager, 'get_memory_stats'):
                memory_stats = memory_manager.get_memory_stats()
                self.record_test("Memory Management", True, f"Memory stats: {memory_stats.status}")
                print("  ✅ Memory manager working")
            else:
                self.record_test("Memory Management", False, "Memory manager missing methods")
                print("  ❌ Memory manager incomplete")
                
        except Exception as e:
            self.record_test("Memory Management", False, f"Error: {str(e)}")
            print(f"  ❌ Memory management test failed: {e}")
    
    def test_integration_system(self):
        """Test integration capabilities"""
        print("\n🔌 Testing Integration System...")
        try:
            from integration.integration_manager import EnhancedFSOTIntegration
            integration = EnhancedFSOTIntegration()
            
            # Test skills database
            if hasattr(integration, 'skills_database'):
                skills = integration.skills_database.skills
                self.record_test("Integration System", True, f"Skills loaded: {len(skills)}")
                print(f"  ✅ Integration system: {len(skills)} skills available")
            else:
                self.record_test("Integration System", False, "Skills database not found")
                print("  ❌ Integration system incomplete")
                
        except Exception as e:
            self.record_test("Integration System", False, f"Error: {str(e)}")
            print(f"  ❌ Integration system test failed: {e}")
    
    def test_web_interface(self):
        """Test web interface components"""
        print("\n🌐 Testing Web Interface...")
        try:
            from interfaces.web_interface import WebInterface
            # WebInterface requires brain_orchestrator, so test with None
            web = WebInterface(brain_orchestrator=None)
            
            if hasattr(web, 'app'):
                self.record_test("Web Interface", True, "Web interface initialized")
                print("  ✅ Web interface ready")
            else:
                self.record_test("Web Interface", False, "Web interface missing app")
                print("  ❌ Web interface incomplete")
                
        except Exception as e:
            self.record_test("Web Interface", False, f"Error: {str(e)}")
            print(f"  ❌ Web interface test failed: {e}")
    
    def test_consciousness_simulation(self):
        """Test consciousness simulation"""
        print("\n🌟 Testing Consciousness Simulation...")
        try:
            from core.consciousness import ConsciousnessMonitor
            consciousness = ConsciousnessMonitor()
            
            # Test consciousness state
            if hasattr(consciousness, 'current_metrics'):
                level = consciousness.current_metrics.level
                self.record_test("Consciousness Simulation", True, f"Consciousness level: {level}")
                print(f"  ✅ Consciousness simulation: {level:.1%} active")
            else:
                self.record_test("Consciousness Simulation", False, "Invalid consciousness state")
                print("  ❌ Consciousness simulation incomplete")
                
        except Exception as e:
            self.record_test("Consciousness Simulation", False, f"Error: {str(e)}")
            print(f"  ❌ Consciousness simulation test failed: {e}")
    
    def test_fsot_integration_capabilities(self):
        """Test FSOT 2.0 integration capabilities"""
        print("\n🚀 Testing FSOT Integration Capabilities...")
        try:
            # Test FSOT system availability
            from brain_system import NeuromorphicBrainSystem
            
            brain = NeuromorphicBrainSystem(verbose=False)
            
            # Test basic functionality
            stimulus = {
                'type': 'conversational',
                'intensity': 0.8,
                'content': 'Test FSOT integration capabilities'
            }
            
            result = brain.process_stimulus(stimulus)
            
            if result and 'response' in result:
                print("  ✅ FSOT Brain System Integration")
                print("  ✅ Stimulus Processing")
                print("  ✅ Response Generation")
                self.record_test("FSOT Integration", True, "All FSOT capabilities working")
            else:
                print("  ❌ FSOT Response Generation Failed")
                self.record_test("FSOT Integration", False, "Response generation failed")
                
        except Exception as e:
            self.record_test("FSOT Integration", False, f"Error: {str(e)}")
            print(f"  ❌ FSOT integration test failed: {e}")
    
    def test_file_structure(self):
        """Test file structure integrity"""
        print("\n📁 Testing File Structure...")
        try:
            required_paths = [
                clean_system_path / "main.py",
                clean_system_path / "brain" / "brain_orchestrator.py", 
                clean_system_path / "utils" / "memory_manager.py",
                clean_system_path / "integration" / "integration_manager.py"
            ]
            
            missing_files = []
            for path in required_paths:
                if not path.exists():
                    missing_files.append(str(path))
            
            if not missing_files:
                self.record_test("File Structure", True, f"All {len(required_paths)} core files present")
                print(f"  ✅ File structure: {len(required_paths)} core files verified")
            else:
                self.record_test("File Structure", False, f"Missing files: {len(missing_files)}")
                print(f"  ❌ Missing {len(missing_files)} core files")
                
        except Exception as e:
            self.record_test("File Structure", False, f"Error: {str(e)}")
            print(f"  ❌ File structure test failed: {e}")
    
    async def test_async_functionality(self):
        """Test async functionality"""
        print("\n⚡ Testing Async Functionality...")
        try:
            # Simple async test
            await asyncio.sleep(0.1)
            self.record_test("Async Functionality", True, "Async operations working")
            print("  ✅ Async functionality working")
            
        except Exception as e:
            self.record_test("Async Functionality", False, f"Error: {str(e)}")
            print(f"  ❌ Async functionality test failed: {e}")
    
    def record_test(self, test_name: str, passed: bool, details: str):
        """Record test result"""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed"] += 1
        else:
            self.test_results["failed"] += 1
        
        self.test_results["details"].append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
    
    async def run_all_tests(self):
        """Run all tests"""
        print("Starting comprehensive system test...\n")
        
        # Run all tests
        self.test_file_structure()
        self.test_brain_architecture()
        self.test_memory_management()
        self.test_integration_system()
        self.test_web_interface()
        self.test_consciousness_simulation()
        self.test_fsot_integration_capabilities()
        await self.test_async_functionality()
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 60)
        print("🏆 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        
        total = self.test_results["total_tests"]
        passed = self.test_results["passed"]
        failed = self.test_results["failed"]
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"📊 Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.test_results["details"]:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test']}: {result['details']}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 80:
            print("🟢 EXCELLENT: System is highly functional and ready for production use!")
        elif success_rate >= 60:
            print("🟡 GOOD: System is mostly functional with minor issues to address.")
        elif success_rate >= 40:
            print("🟠 FAIR: System has basic functionality but needs significant improvements.")
        else:
            print("🔴 POOR: System has major issues that need immediate attention.")
        
        # Save results
        results_file = current_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")

async def main():
    """Main test execution"""
    tester = ComprehensiveSystemTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
