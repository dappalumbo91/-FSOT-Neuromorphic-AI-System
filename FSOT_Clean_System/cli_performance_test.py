#!/usr/bin/env python3
"""
Interactive CLI Performance Test
Tests the CLI response time and functionality
"""

import time
import sys
import os
import asyncio
from typing import Dict, Any

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_cli_commands():
    """Test CLI command performance"""
    print("🖥️ Testing CLI Interface Performance...")
    
    try:
        from interfaces.cli_interface import CLIInterface
        from brain.brain_orchestrator import BrainOrchestrator
        from core.fsot_engine import FSOTEngine
        
        # Initialize components
        brain = BrainOrchestrator()
        await brain.initialize()
        
        fsot_engine = FSOTEngine()
        cli = CLIInterface(brain, fsot_engine)
        
        # Test commands
        test_commands = [
            "status",
            "help",
            "brain status",
            "fsot compute cognitive"
        ]
        
        results = []
        
        for command in test_commands:
            print(f"\n   Testing command: '{command}'")
            start_time = time.time()
            
            try:
                # Process command
                response = await cli.process_command(command)
                duration = time.time() - start_time
                
                print(f"   ✅ Response in {duration:.3f}s")
                print(f"   📝 Response type: {type(response).__name__}")
                
                results.append({
                    'command': command,
                    'duration': duration,
                    'success': True,
                    'response_type': type(response).__name__
                })
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"   ❌ Failed in {duration:.3f}s: {str(e)}")
                results.append({
                    'command': command,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })
        
        # Cleanup
        await brain.shutdown()
        
        return results
        
    except Exception as e:
        print(f"❌ CLI test failed: {str(e)}")
        return []

async def test_system_responsiveness():
    """Test overall system responsiveness"""
    print("\n📊 Testing System Responsiveness...")
    
    try:
        from core.fsot_engine import FSOTEngine, Domain
        from brain.brain_orchestrator import BrainOrchestrator
        
        # Initialize
        brain = BrainOrchestrator()
        fsot = FSOTEngine()
        
        start_time = time.time()
        await brain.initialize()
        init_time = time.time() - start_time
        
        # Test multiple operations
        operations = []
        
        # FSOT calculations
        start_time = time.time()
        for domain in [Domain.COGNITIVE, Domain.QUANTUM, Domain.BIOLOGICAL]:
            result = fsot.compute_for_domain(domain)
            operations.append(f"FSOT {domain.name}: {result:.6f}")
        fsot_time = time.time() - start_time
        
        # Brain queries
        start_time = time.time()
        query_result = await brain.process_query("Test query for performance")
        brain_time = time.time() - start_time
        operations.append(f"Brain query processed in {brain_time:.3f}s")
        
        # Cleanup
        await brain.shutdown()
        
        print(f"   ✅ Brain initialization: {init_time:.3f}s")
        print(f"   ✅ FSOT calculations: {fsot_time:.3f}s")
        print(f"   ✅ Brain query: {brain_time:.3f}s")
        
        for op in operations:
            print(f"      {op}")
        
        return {
            'init_time': init_time,
            'fsot_time': fsot_time,
            'brain_time': brain_time,
            'operations': operations
        }
        
    except Exception as e:
        print(f"❌ Responsiveness test failed: {str(e)}")
        return {}

async def main():
    """Run CLI performance tests"""
    print("🧠⚡ FSOT 2.0 CLI PERFORMANCE TEST")
    print("=" * 50)
    
    overall_start = time.time()
    
    # Test CLI commands
    cli_results = await test_cli_commands()
    
    # Test system responsiveness
    responsiveness_results = await test_system_responsiveness()
    
    overall_time = time.time() - overall_start
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CLI PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if cli_results:
        successful_commands = sum(1 for r in cli_results if r['success'])
        total_commands = len(cli_results)
        avg_response_time = sum(r['duration'] for r in cli_results if r['success']) / max(successful_commands, 1)
        
        print(f"Commands tested: {total_commands}")
        print(f"Successful: {successful_commands}")
        print(f"Success rate: {(successful_commands/total_commands)*100:.1f}%")
        print(f"Average response time: {avg_response_time:.3f}s")
        
        print("\nCommand Details:")
        for result in cli_results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} '{result['command']}' - {result['duration']:.3f}s")
    
    if responsiveness_results:
        print(f"\nSystem Responsiveness:")
        print(f"  Init time: {responsiveness_results.get('init_time', 0):.3f}s")
        print(f"  FSOT time: {responsiveness_results.get('fsot_time', 0):.3f}s")
        print(f"  Brain time: {responsiveness_results.get('brain_time', 0):.3f}s")
    
    print(f"\nTotal test time: {overall_time:.3f}s")
    
    # Performance assessment
    if cli_results and responsiveness_results:
        avg_time = sum(r['duration'] for r in cli_results if r['success']) / max(len(cli_results), 1)
        if avg_time < 0.1:
            print("🚀 EXCELLENT: Very fast response times")
        elif avg_time < 0.5:
            print("✅ GOOD: Acceptable response times")
        else:
            print("⚠️  SLOW: Consider optimization")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {str(e)}")
        import traceback
        traceback.print_exc()
