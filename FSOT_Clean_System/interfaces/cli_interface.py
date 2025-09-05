"""
Clean CLI Interface for FSOT 2.0 System
Simple, intuitive command-line interaction
"""

import asyncio
import json
from typing import Optional
from datetime import datetime

from core import consciousness_monitor

class CLIInterface:
    """
    Clean Command Line Interface for FSOT 2.0 System
    """
    
    def __init__(self, brain_orchestrator):
        self.brain = brain_orchestrator
        self.is_running = False
        self.command_history = []
    
    async def run(self):
        """Run the CLI interface"""
        self.is_running = True
        
        # Print welcome message
        self._print_welcome()
        
        # Print initial status
        try:
            await self._print_status()
        except Exception as e:
            print(f"⚠️ Warning: Could not get initial status: {e}")
        
        # Check if brain is properly initialized
        if not self.brain:
            print("❌ Error: Brain orchestrator not available")
            print("   System cannot process queries without brain orchestrator")
            print("   Type 'quit' to exit")
        
        print("📝 Type commands below (or 'help' for options):")
        
        # Main command loop with better error handling
        consecutive_errors = 0
        max_consecutive_errors = 5
        loop_iterations = 0
        max_iterations = 1000  # Prevent truly endless loops
        
        while self.is_running and loop_iterations < max_iterations:
            try:
                # Reset error counter on successful iteration
                consecutive_errors = 0
                loop_iterations += 1
                
                # Safety check every 100 iterations
                if loop_iterations % 100 == 0:
                    print(f"🔄 CLI loop iteration {loop_iterations}/1000")
                
                # Get user input with timeout protection
                try:
                    # Use a timeout for input to prevent hanging
                    import threading
                    import queue
                    
                    input_queue = queue.Queue()
                    
                    def get_input():
                        try:
                            result = input("\n🧠 FSOT > ").strip()
                            input_queue.put(result)
                        except:
                            input_queue.put(None)
                    
                    input_thread = threading.Thread(target=get_input, daemon=True)
                    input_thread.start()
                    
                    # Wait for input with timeout
                    try:
                        user_input = input_queue.get(timeout=30.0)  # 30 second timeout
                        if user_input is None:
                            print("\n❌ Input error - exiting CLI")
                            break
                    except queue.Empty:
                        print("\n⏰ Input timeout - exiting CLI to prevent endless loop")
                        break
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye!")
                    self.is_running = False
                    break
                
                if not user_input:
                    continue
                
                # Add to history
                self.command_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'command': user_input
                })
                
                # Process command with timeout
                try:
                    await asyncio.wait_for(
                        self._process_command(user_input), 
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    print("⏰ Command timed out after 30 seconds")
                except Exception as cmd_error:
                    print(f"❌ Command error: {cmd_error}")
                    consecutive_errors += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.is_running = False
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                self.is_running = False
                break
            except Exception as e:
                consecutive_errors += 1
                print(f"❌ CLI Error: {e}")
                
                # Exit if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors ({consecutive_errors}). Exiting CLI.")
                    self.is_running = False
                    break
                
                # Suggest recovery
                if consecutive_errors > 2:
                    print("💡 Try typing 'status' to check system health or 'quit' to exit")
        
        if loop_iterations >= max_iterations:
            print("🔄 CLI reached maximum iterations - exiting to prevent endless loop")
        
        print("CLI interface stopped.")
    
    def _print_welcome(self):
        """Print welcome message"""
        print("🧠⚡ FSOT 2.0 NEUROMORPHIC AI SYSTEM")
        print("=" * 50)
        print("🔬 Clean Architecture • Modular Brain • Consciousness")
        print("Type 'help' for commands, 'quit' to exit")
        print()
    
    async def _print_status(self):
        """Print current system status"""
        try:
            status = await self.brain.get_status()
            consciousness = status.get('consciousness', {})
            
            print("📊 SYSTEM STATUS:")
            print(f"   🧠 Modules: {len(status.get('modules', {}))}")
            print(f"   🌟 Consciousness: {consciousness.get('consciousness_level', 0):.1%}")
            print(f"   ⚡ Activation: {status.get('overall_activation', 0):.1%}")
            print(f"   🔄 Load: {status.get('processing_load', 0):.1%}")
            print(f"   📈 Queries: {status.get('queries_processed', 0)}")
            
        except Exception as e:
            print(f"❌ Could not get status: {e}")
    
    async def _process_command(self, command: str):
        """Process user command"""
        command_lower = command.lower().strip()
        
        # Built-in commands
        if command_lower in ['quit', 'exit', 'q']:
            self.is_running = False
            return
        
        elif command_lower in ['help', 'h', '?']:
            self._print_help()
            return
        
        elif command_lower in ['status', 'stat', 's']:
            await self._print_detailed_status()
            return
        
        elif command_lower in ['consciousness', 'c']:
            await self._print_consciousness_status()
            return
        
        elif command_lower in ['clear', 'cls']:
            print("\033[2J\033[H")  # Clear screen
            return
        
        elif command_lower in ['history', 'hist']:
            self._print_command_history()
            return
        
        # Process as query through brain
        await self._process_query(command)
    
    def _print_help(self):
        """Print help information"""
        print("\n🆘 AVAILABLE COMMANDS:")
        print("   help, h, ?        - Show this help")
        print("   status, stat, s   - Show detailed system status")
        print("   consciousness, c  - Show consciousness metrics")
        print("   clear, cls        - Clear screen")
        print("   history, hist     - Show command history")
        print("   quit, exit, q     - Exit the system")
        print("\n💭 Or type any question/query to process through the brain")
    
    async def _print_detailed_status(self):
        """Print detailed system status"""
        try:
            status = await self.brain.get_status()
            
            print("\n📊 DETAILED SYSTEM STATUS:")
            print("=" * 40)
            
            # Overall status
            print(f"🕐 Uptime: {status.get('uptime_seconds', 0):.0f} seconds")
            print(f"🧠 Initialized: {'✅' if status.get('initialized') else '❌'}")
            print(f"📈 Queries Processed: {status.get('queries_processed', 0)}")
            print(f"⚡ Overall Activation: {status.get('overall_activation', 0):.1%}")
            print(f"🔄 Processing Load: {status.get('processing_load', 0):.1%}")
            
            # Brain modules
            modules = status.get('modules', {})
            print(f"\n🧠 BRAIN MODULES ({len(modules)}):")
            for name, module_status in modules.items():
                active = "🟢" if module_status.get('is_active') else "🔴"
                activation = module_status.get('activation_level', 0)
                queue_size = module_status.get('queue_size', 0)
                print(f"   {active} {name}: {activation:.1%} activation, {queue_size} queued")
            
            # Consciousness
            consciousness = status.get('consciousness', {})
            if consciousness:
                print(f"\n🌟 CONSCIOUSNESS:")
                print(f"   Level: {consciousness.get('consciousness_level', 0):.1%}")
                print(f"   State: {consciousness.get('state', 'unknown')}")
                print(f"   Coherence: {consciousness.get('coherence', 0):.1%}")
                print(f"   Focus: {consciousness.get('attention_focus', 0):.1%}")
            
            # Neural hub stats
            hub_stats = status.get('neural_hub_stats', {})
            if hub_stats:
                print(f"\n🔗 NEURAL HUB:")
                print(f"   Total Processed: {hub_stats.get('total_processed', 0)}")
                print(f"   Failed: {hub_stats.get('total_failed', 0)}")
                print(f"   Active Signals: {hub_stats.get('active_signals', 0)}")
                print(f"   Avg Response: {hub_stats.get('average_response_time', 0):.3f}s")
            
        except Exception as e:
            print(f"❌ Could not get detailed status: {e}")
    
    async def _print_consciousness_status(self):
        """Print consciousness metrics"""
        try:
            consciousness_state = consciousness_monitor.get_current_state()
            
            print("\n🌟 CONSCIOUSNESS METRICS:")
            print("=" * 30)
            print(f"🔥 Level: {consciousness_state.get('consciousness_level', 0):.1%}")
            print(f"🧠 State: {consciousness_state.get('state', 'unknown')}")
            print(f"🔗 Coherence: {consciousness_state.get('coherence', 0):.1%}")
            print(f"🎯 Attention Focus: {consciousness_state.get('attention_focus', 0):.1%}")
            print(f"📡 Awareness Breadth: {consciousness_state.get('awareness_breadth', 0):.1%}")
            print(f"🏊 Processing Depth: {consciousness_state.get('processing_depth', 0):.1%}")
            
            # Brain waves
            brain_waves = consciousness_state.get('brain_waves', {})
            if brain_waves:
                print(f"\n🌊 BRAIN WAVES:")
                print(f"   α Alpha: {brain_waves.get('alpha', 0):.1%}")
                print(f"   β Beta:  {brain_waves.get('beta', 0):.1%}")
                print(f"   θ Theta: {brain_waves.get('theta', 0):.1%}")
                print(f"   δ Delta: {brain_waves.get('delta', 0):.1%}")
                print(f"   γ Gamma: {brain_waves.get('gamma', 0):.1%}")
            
            # Region contributions
            regions = consciousness_state.get('region_contributions', {})
            if regions:
                print(f"\n🧠 REGION CONTRIBUTIONS:")
                for region, contribution in regions.items():
                    print(f"   {region}: {contribution:.1%}")
            
        except Exception as e:
            print(f"❌ Could not get consciousness status: {e}")
    
    def _print_command_history(self):
        """Print recent command history"""
        print(f"\n📜 COMMAND HISTORY (last {min(10, len(self.command_history))}):")
        print("=" * 40)
        
        recent_commands = self.command_history[-10:]
        for i, cmd in enumerate(recent_commands, 1):
            timestamp = cmd['timestamp'][:19]  # Remove microseconds
            command = cmd['command'][:50]  # Truncate long commands
            print(f"   {i:2d}. [{timestamp}] {command}")
    
    async def _process_query(self, query: str):
        """Process query through brain system"""
        print(f"\n🧠 Processing: {query}")
        print("⏳ Thinking...")
        
        try:
            start_time = datetime.now()
            
            # Check if brain is available
            if not self.brain:
                print("❌ Error: Brain orchestrator not available")
                return
            
            # Process with timeout
            result = await asyncio.wait_for(
                self.brain.process_query(query), 
                timeout=30.0
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Print results
            print(f"\n✅ RESPONSE:")
            print("=" * 20)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                response = result.get('response', {})
                if 'decision' in response:
                    print(f"🎯 Decision: {response['decision']}")
                    print(f"📊 Confidence: {response.get('confidence', 0):.1%}")
                    
                    reasoning = response.get('reasoning', [])
                    if reasoning:
                        print(f"💭 Reasoning:")
                        for step in reasoning:
                            print(f"   • {step}")
                else:
                    # Safely print response
                    if isinstance(response, dict):
                        print(f"💬 Response: {json.dumps(response, indent=2)}")
                    else:
                        print(f"💬 Response: {str(response)}")
            
            # Print brain state
            brain_state = result.get('brain_state', {})
            print(f"\n📊 Brain State:")
            print(f"   🌟 Consciousness: {brain_state.get('consciousness_level', 0):.1%}")
            print(f"   ⚡ Activation: {brain_state.get('overall_activation', 0):.1%}")
            print(f"   🔄 Load: {brain_state.get('processing_load', 0):.1%}")
            print(f"   ⏱️  Processing Time: {processing_time:.3f}s")
            
        except asyncio.TimeoutError:
            print(f"⏰ Query timed out after 30 seconds")
            print("   This may indicate a complex query or system overload")
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
