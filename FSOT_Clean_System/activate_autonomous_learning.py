#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 Autonomous Learning System Activator
=====================================================

This script activates and manages the autonomous learning system with continuous
knowledge acquisition, web search integration, and brain module coordination.

Author: GitHub Copilot
"""

import os
import sys
import json
import time
import threading
import schedule
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class AutonomousLearningActivator:
    """Activates and manages autonomous learning system"""
    
    def __init__(self):
        self.config_dir = project_root / "config"
        self.autonomous_config_file = self.config_dir / "autonomous_config.json"
        self.api_config_file = self.config_dir / "api_config.json"
        
        self.running = False
        self.learning_thread = None
        self.schedule_thread = None
        
        # Load configurations
        self.load_configs()
    
    def load_configs(self):
        """Load autonomous learning and API configurations"""
        try:
            # Load autonomous learning config
            if self.autonomous_config_file.exists():
                with open(self.autonomous_config_file, 'r') as f:
                    self.autonomous_config = json.load(f)
            else:
                self.autonomous_config = self.get_default_autonomous_config()
            
            # Load API config
            if self.api_config_file.exists():
                with open(self.api_config_file, 'r') as f:
                    self.api_config = json.load(f)
            else:
                self.api_config = {}
                
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            self.autonomous_config = self.get_default_autonomous_config()
            self.api_config = {}
    
    def get_default_autonomous_config(self):
        """Get default autonomous learning configuration"""
        return {
            "autonomous_learning": {
                "enabled": True,
                "mode": "continuous",
                "schedule": {
                    "interval_minutes": 30,
                    "daily_sessions": 16,
                    "learning_duration_minutes": 15,
                    "break_duration_minutes": 15
                }
            },
            "training_schedules": {"enabled": True},
            "skills_development": {"enabled": True},
            "background_operations": {"enabled": True}
        }
    
    def check_prerequisites(self):
        """Check if all prerequisites are met for autonomous learning"""
        print("🔍 Checking Prerequisites...")
        
        prerequisites_met = True
        
        # Check if main system files exist
        main_file = project_root / "main.py"
        if not main_file.exists():
            print("❌ main.py not found")
            prerequisites_met = False
        else:
            print("✅ Main system file found")
        
        # Check integration modules
        integration_dir = project_root / "integration"
        required_modules = [
            "autonomous_learning.py",
            "web_search_engine.py",
            "skills_database.py",
            "training_facility.py"
        ]
        
        for module in required_modules:
            module_file = integration_dir / module
            if module_file.exists():
                print(f"✅ {module} found")
            else:
                print(f"❌ {module} not found")
                prerequisites_met = False
        
        # Check API configuration
        configured_apis = 0
        total_apis = len(self.api_config)
        
        for service, config in self.api_config.items():
            if config.get("enabled", False):
                api_key = config.get("api_key", "")
                if api_key and not api_key.startswith("your-"):
                    configured_apis += 1
                    print(f"✅ {service.upper()} API configured")
                else:
                    print(f"⚠️  {service.upper()} API enabled but no key")
            else:
                print(f"❌ {service.upper()} API disabled")
        
        if configured_apis > 0:
            print(f"📊 API Status: {configured_apis}/{total_apis} services configured")
        else:
            print("⚠️  No external APIs configured - limited functionality")
        
        return prerequisites_met
    
    def activate_autonomous_learning(self):
        """Activate the autonomous learning system"""
        print("\\n🚀 Activating Autonomous Learning System...")
        print("=" * 50)
        
        # Update configuration to enable autonomous learning
        self.autonomous_config["autonomous_learning"]["enabled"] = True
        self.autonomous_config["training_schedules"]["enabled"] = True
        self.autonomous_config["skills_development"]["enabled"] = True
        self.autonomous_config["background_operations"]["enabled"] = True
        
        # Save updated configuration
        try:
            with open(self.autonomous_config_file, 'w') as f:
                json.dump(self.autonomous_config, f, indent=2)
            print("✅ Autonomous learning configuration saved")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
        
        # Start autonomous learning
        self.running = True
        
        # Start learning thread
        self.learning_thread = threading.Thread(
            target=self.continuous_learning_loop, 
            daemon=True
        )
        self.learning_thread.start()
        print("✅ Autonomous learning thread started")
        
        # Start schedule management thread
        self.schedule_thread = threading.Thread(
            target=self.schedule_management_loop,
            daemon=True
        )
        self.schedule_thread.start()
        print("✅ Schedule management thread started")
        
        # Setup training schedules
        self.setup_training_schedules()
        
        print("\\n🎉 Autonomous Learning System Activated!")
        print("🧠 The system will now continuously learn and improve")
        
        return True
    
    def continuous_learning_loop(self):
        """Main continuous learning loop"""
        learning_config = self.autonomous_config["autonomous_learning"]
        interval_minutes = learning_config["schedule"]["interval_minutes"]
        
        print(f"🔄 Starting continuous learning loop (every {interval_minutes} minutes)")
        
        while self.running:
            try:
                # Trigger learning session
                self.trigger_learning_session()
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                print(f"❌ Error in learning loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def trigger_learning_session(self):
        """Trigger a learning session"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\\n📚 [{timestamp}] Starting Learning Session...")
        
        try:
            # Import and run autonomous learning
            sys.path.insert(0, str(project_root / "integration"))
            from autonomous_learning import AutonomousLearningSystem
            
            # Initialize learning system
            learning_system = AutonomousLearningSystem()
            
            # Trigger learning
            if hasattr(learning_system, 'learn_autonomously'):
                results = learning_system.learn_autonomously()
                print(f"✅ Learning session completed: {len(results.get('discoveries', []))} discoveries")
            else:
                print("⚠️  Learning system not fully configured")
                
        except ImportError as e:
            print(f"⚠️  Could not import learning system: {e}")
        except Exception as e:
            print(f"❌ Learning session error: {e}")
    
    def setup_training_schedules(self):
        """Setup automated training schedules"""
        if not self.autonomous_config["training_schedules"]["enabled"]:
            return
        
        schedules = self.autonomous_config["training_schedules"]["schedules"]
        
        # Daily evaluation
        if schedules.get("daily_evaluation", {}).get("enabled", False):
            eval_time = schedules["daily_evaluation"]["time"]
            schedule.every().day.at(eval_time).do(self.run_daily_evaluation)
            print(f"📅 Daily evaluation scheduled at {eval_time}")
        
        # Weekly deep training
        if schedules.get("weekly_deep_training", {}).get("enabled", False):
            day = schedules["weekly_deep_training"]["day"]
            time_str = schedules["weekly_deep_training"]["time"]
            getattr(schedule.every(), day.lower()).at(time_str).do(self.run_weekly_training)
            print(f"📅 Weekly training scheduled on {day} at {time_str}")
        
        # Monthly comprehensive review
        if schedules.get("monthly_comprehensive_review", {}).get("enabled", False):
            schedule.every().month.do(self.run_monthly_review)
            print("📅 Monthly review scheduled")
    
    def schedule_management_loop(self):
        """Manage scheduled tasks"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"❌ Schedule management error: {e}")
                time.sleep(60)
    
    def run_daily_evaluation(self):
        """Run daily evaluation"""
        print("\\n📊 Running Daily Evaluation...")
        try:
            # Import and run training facility
            sys.path.insert(0, str(project_root / "integration"))
            from training_facility import BrainTrainingFacility
            
            facility = BrainTrainingFacility()
            if hasattr(facility, 'evaluate_all_modules'):
                results = facility.evaluate_all_modules()
                print(f"✅ Daily evaluation completed: {results.get('overall_score', 'N/A')}")
        except Exception as e:
            print(f"❌ Daily evaluation error: {e}")
    
    def run_weekly_training(self):
        """Run weekly deep training"""
        print("\\n🏋️  Running Weekly Deep Training...")
        try:
            # Run intensive training session
            self.trigger_learning_session()
            self.run_daily_evaluation()
            print("✅ Weekly training completed")
        except Exception as e:
            print(f"❌ Weekly training error: {e}")
    
    def run_monthly_review(self):
        """Run monthly comprehensive review"""
        print("\\n📋 Running Monthly Comprehensive Review...")
        try:
            # Run comprehensive analysis
            self.run_daily_evaluation()
            print("✅ Monthly review completed")
        except Exception as e:
            print(f"❌ Monthly review error: {e}")
    
    def show_status(self):
        """Show current autonomous learning status"""
        print("\\n📊 Autonomous Learning Status:")
        print("-" * 40)
        
        if self.running:
            print("🟢 Status: ACTIVE")
            print(f"🧵 Learning Thread: {'Running' if self.learning_thread and self.learning_thread.is_alive() else 'Stopped'}")
            print(f"⏰ Schedule Thread: {'Running' if self.schedule_thread and self.schedule_thread.is_alive() else 'Stopped'}")
        else:
            print("🔴 Status: INACTIVE")
        
        # Show configuration
        learning_config = self.autonomous_config["autonomous_learning"]
        if learning_config["enabled"]:
            interval = learning_config["schedule"]["interval_minutes"]
            print(f"⏱️  Learning Interval: {interval} minutes")
            print(f"🎯 Mode: {learning_config['mode']}")
        
        # Show scheduled tasks
        pending_jobs = schedule.jobs
        if pending_jobs:
            print(f"📅 Scheduled Tasks: {len(pending_jobs)}")
            for job in pending_jobs:
                print(f"   - {job}")
    
    def stop_autonomous_learning(self):
        """Stop the autonomous learning system"""
        print("\\n🛑 Stopping Autonomous Learning System...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        if self.schedule_thread and self.schedule_thread.is_alive():
            self.schedule_thread.join(timeout=5)
        
        # Clear scheduled jobs
        schedule.clear()
        
        print("✅ Autonomous learning system stopped")

def main():
    """Main entry point"""
    print("🧠 Enhanced FSOT 2.0 Autonomous Learning Activator")
    print("=" * 60)
    
    activator = AutonomousLearningActivator()
    
    try:
        # Check prerequisites
        if not activator.check_prerequisites():
            print("\\n⚠️  Some prerequisites not met. Continuing with limited functionality...")
        
        # Ask user for activation
        choice = input("\\n🚀 Activate autonomous learning? (y/n): ").lower().strip()
        
        if choice == 'y':
            # Activate autonomous learning
            if activator.activate_autonomous_learning():
                print("\\n✅ Autonomous learning system is now active!")
                print("💡 The system will continuously learn and improve in the background.")
                print("\\n📋 Commands:")
                print("   - 'status': Show system status")
                print("   - 'stop': Stop autonomous learning")
                print("   - 'quit': Exit program")
                
                # Interactive loop
                while activator.running:
                    try:
                        command = input("\\n> ").lower().strip()
                        
                        if command == 'status':
                            activator.show_status()
                        elif command == 'stop':
                            activator.stop_autonomous_learning()
                            break
                        elif command in ['quit', 'exit']:
                            activator.stop_autonomous_learning()
                            break
                        else:
                            print("Unknown command. Try: status, stop, quit")
                            
                    except KeyboardInterrupt:
                        print("\\n\\n🛑 Interrupted by user")
                        activator.stop_autonomous_learning()
                        break
            else:
                print("❌ Failed to activate autonomous learning")
        else:
            print("⏭️  Autonomous learning activation skipped")
    
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        sys.exit(1)
    
    print("\\n👋 Goodbye!")

if __name__ == "__main__":
    main()
