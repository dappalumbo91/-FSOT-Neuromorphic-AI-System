#!/usr/bin/env python3
"""
FSOT 2.0 Performance Testing Suite
Comprehensive evaluation of system capabilities and performance
"""

import asyncio
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

class FSOTPerformanceTester:
    """Comprehensive performance testing for FSOT 2.0 system"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def run_command_test(self, command: str, description: str):
        """Test a CLI command and measure performance"""
        print(f"\n🧪 Testing: {description}")
        print(f"   Command: {command}")
        
        start_time = time.time()
        try:
            # Simulate sending command to the system
            result = subprocess.run(
                f'echo "{command}" | python main.py',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd="."
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"   ✅ Success in {execution_time:.3f}s")
                status = "success"
            else:
                print(f"   ⚠️  Completed with warnings in {execution_time:.3f}s")
                status = "warning"
                
            self.test_results["tests"][command] = {
                "description": description,
                "status": status,
                "execution_time": execution_time,
                "output_length": len(result.stdout) if result.stdout else 0
            }
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout after 10s")
            self.test_results["tests"][command] = {
                "description": description,
                "status": "timeout",
                "execution_time": 10.0
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.test_results["tests"][command] = {
                "description": description,
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def test_system_commands(self):
        """Test core system commands"""
        print("🔬 TESTING CORE SYSTEM COMMANDS")
        print("=" * 50)
        
        test_commands = [
            ("help", "Display help system"),
            ("status", "System status display"),
            ("brain", "Brain module information"),
            ("consciousness", "Consciousness monitoring"),
            ("config", "Configuration display"),
        ]
        
        for cmd, desc in test_commands:
            self.run_command_test(cmd, desc)
    
    def test_fsot_calculations(self):
        """Test FSOT mathematical calculations"""
        print("\n🧮 TESTING FSOT CALCULATIONS")
        print("=" * 50)
        
        calc_tests = [
            ("calculate phi", "Golden ratio calculation"),
            ("calculate e", "Euler's number calculation"), 
            ("calculate pi", "Pi calculation"),
            ("calculate gamma", "Euler-Mascheroni constant"),
        ]
        
        for cmd, desc in calc_tests:
            self.run_command_test(cmd, desc)
    
    def test_brain_functionality(self):
        """Test brain module functionality"""
        print("\n🧠 TESTING BRAIN FUNCTIONALITY")
        print("=" * 50)
        
        brain_tests = [
            ("brain status", "Brain status check"),
            ("brain modules", "Brain module listing"),
            ("brain frontal_cortex", "Frontal cortex query"),
        ]
        
        for cmd, desc in brain_tests:
            self.run_command_test(cmd, desc)
    
    def test_initialization_speed(self):
        """Test system initialization performance"""
        print("\n⚡ TESTING INITIALIZATION SPEED")
        print("=" * 50)
        
        times = []
        for i in range(3):
            print(f"   Run {i+1}/3...")
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    'echo "quit" | python main.py',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd="."
                )
                
                init_time = time.time() - start_time
                times.append(init_time)
                print(f"   ✅ Initialization: {init_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Initialization Performance:")
            print(f"   Average: {avg_time:.3f}s")
            print(f"   Fastest: {min_time:.3f}s") 
            print(f"   Slowest: {max_time:.3f}s")
            
            self.test_results["initialization"] = {
                "average": avg_time,
                "min": min_time,
                "max": max_time,
                "runs": len(times)
            }
    
    def test_stress_load(self):
        """Test system under stress"""
        print("\n💪 TESTING STRESS LOAD")
        print("=" * 50)
        
        print("   Running rapid command sequence...")
        start_time = time.time()
        
        try:
            commands = ["status", "brain", "consciousness", "help", "config"] * 2
            command_string = ";".join(commands) + ";quit"
            
            result = subprocess.run(
                f'echo "{command_string}" | python main.py',
                shell=True,
                capture_output=True,
                text=True,
                timeout=20,
                cwd="."
            )
            
            total_time = time.time() - start_time
            commands_per_second = len(commands) / total_time
            
            print(f"   ✅ {len(commands)} commands in {total_time:.3f}s")
            print(f"   📈 Throughput: {commands_per_second:.1f} commands/sec")
            
            self.test_results["stress_test"] = {
                "commands": len(commands),
                "total_time": total_time,
                "throughput": commands_per_second,
                "status": "success"
            }
            
        except Exception as e:
            print(f"   ❌ Stress test failed: {e}")
            self.test_results["stress_test"] = {
                "status": "failed",
                "error": str(e)
            }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 GENERATING PERFORMANCE REPORT")
        print("=" * 50)
        
        # Calculate summary statistics
        successful_tests = sum(1 for test in self.test_results["tests"].values() 
                             if test.get("status") == "success")
        total_tests = len(self.test_results["tests"])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        avg_execution_time = sum(test.get("execution_time", 0) 
                               for test in self.test_results["tests"].values()) / total_tests
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time
        }
        
        # Save detailed report
        with open("performance_report.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Print summary
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Tests Run: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Response Time: {avg_execution_time:.3f}s")
        
        if "initialization" in self.test_results:
            init = self.test_results["initialization"]
            print(f"   Avg Init Time: {init['average']:.3f}s")
        
        if "stress_test" in self.test_results:
            stress = self.test_results["stress_test"]
            if stress.get("status") == "success":
                print(f"   Command Throughput: {stress['throughput']:.1f} cmd/s")
        
        print(f"\n💾 Detailed report saved to: performance_report.json")
        
        # Performance rating
        if success_rate >= 90 and avg_execution_time < 2.0:
            rating = "EXCELLENT"
            emoji = "🏆"
        elif success_rate >= 80 and avg_execution_time < 3.0:
            rating = "GOOD"
            emoji = "👍"
        elif success_rate >= 70:
            rating = "ACCEPTABLE"
            emoji = "👌"
        else:
            rating = "NEEDS IMPROVEMENT"
            emoji = "⚠️"
        
        print(f"\n{emoji} OVERALL PERFORMANCE RATING: {rating}")
        return rating
    
    def run_full_test_suite(self):
        """Run complete performance test suite"""
        print("🚀 FSOT 2.0 PERFORMANCE TEST SUITE")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.test_system_commands()
        self.test_fsot_calculations()
        self.test_brain_functionality()
        self.test_initialization_speed()
        self.test_stress_load()
        
        # Generate final report
        rating = self.generate_performance_report()
        
        print("\n" + "=" * 60)
        print("🎯 PERFORMANCE TESTING COMPLETE!")
        print(f"🧠⚡ FSOT 2.0 System Performance: {rating}")
        
        return self.test_results

def main():
    """Main testing function"""
    tester = FSOTPerformanceTester()
    results = tester.run_full_test_suite()
    return results

if __name__ == "__main__":
    main()
