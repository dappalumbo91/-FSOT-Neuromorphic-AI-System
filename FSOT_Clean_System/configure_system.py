#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 System Configuration & Activation Script
This script helps configure API keys and activate advanced features.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class FSOTConfigurator:
    """
    FSOT 2.0 System Configuration Manager.
    
    This class handles the configuration of API keys, autonomous settings,
    and system parameters for the FSOT Neuromorphic AI System. It provides
    interactive configuration setup and validation for all core components.
    
    Attributes:
        config_dir: Directory containing configuration files
        api_config_file: Path to API configuration file
        autonomous_config_file: Path to autonomous configuration file
    """
    def __init__(self):
        self.config_dir = Path(__file__).parent / "config"
        self.api_config_file = self.config_dir / "api_config.json"
        self.autonomous_config_file = self.config_dir / "autonomous_config.json"
        
    def print_banner(self):
        """Print the configuration banner"""
        print("=" * 80)
        print("🧠 Enhanced FSOT 2.0 Neuromorphic AI System Configuration")
        print("=" * 80)
        print("🚀 Welcome to the Enhanced FSOT 2.0 System Configuration!")
        print("📝 This script will help you configure API keys and activate advanced features.")
        print()
    
    def print_api_instructions(self):
        """Print instructions for obtaining API keys"""
        print("🔑 API Key Configuration Instructions:")
        print("-" * 50)
        print("1. OpenAI API Key:")
        print("   • Visit: https://platform.openai.com/api-keys")
        print("   • Create a new API key")
        print("   • Format: sk-...")
        print()
        print("2. GitHub Personal Access Token:")
        print("   • Visit: https://github.com/settings/tokens")
        print("   • Generate new token (classic)")
        print("   • Select 'repo' and 'user' scopes")
        print("   • Format: ghp_...")
        print()
        print("3. Wolfram Alpha App ID:")
        print("   • Visit: https://developer.wolframalpha.com/portal/myapps")
        print("   • Get an AppID (free tier available)")
        print("   • Format: alphanumeric string")
        print()
        print("4. HuggingFace API Token:")
        print("   • Visit: https://huggingface.co/settings/tokens")
        print("   • Create new token")
        print("   • Format: hf_...")
        print()
        print("5. Google Custom Search API:")
        print("   • Visit: https://console.cloud.google.com/")
        print("   • Enable Custom Search API")
        print("   • Create credentials (API key)")
        print("   • Visit: https://cse.google.com/cse/")
        print("   • Create a custom search engine")
        print("   • Get Search Engine ID (cx parameter)")
        print()
    
    def configure_api_keys(self):
        """Interactive API key configuration"""
        print("🛠️  API Key Configuration:")
        print("-" * 30)
        
        # Load current config
        with open(self.api_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Configure each service
        services = [
            ("OpenAI", "openai", "sk-"),
            ("GitHub", "github", "ghp_"),
            ("Wolfram Alpha", "wolfram", ""),
            ("HuggingFace", "huggingface", "hf_"),
            ("Google Custom Search", "google_custom_search", "")
        ]
        
        updated = False
        for service_name, config_key, prefix in services:
            current_key = config.get(config_key, {}).get("api_key", "")
            if current_key and not current_key.startswith("your-"):
                print(f"✅ {service_name}: Already configured")
                continue
            
            print(f"\n🔧 Configure {service_name}:")
            choice = input(f"   Configure {service_name}? (y/n/skip): ").lower().strip()
            
            if choice == 'y':
                if config_key == "google_custom_search":
                    api_key = input(f"   Enter Google API Key: ").strip()
                    search_engine_id = input(f"   Enter Search Engine ID: ").strip()
                    if api_key and search_engine_id:
                        config[config_key]["api_key"] = api_key
                        config[config_key]["search_engine_id"] = search_engine_id
                        config[config_key]["enabled"] = True
                        updated = True
                        print(f"   ✅ {service_name} configured!")
                else:
                    api_key = input(f"   Enter {service_name} API Key: ").strip()
                    if api_key:
                        config[config_key]["api_key"] = api_key
                        config[config_key]["enabled"] = True
                        updated = True
                        print(f"   ✅ {service_name} configured!")
            elif choice == 'skip':
                continue
            else:
                print(f"   ⏭️  Skipping {service_name}")
        
        # Save updated config
        if updated:
            with open(self.api_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"\n💾 API configuration saved to: {self.api_config_file}")
        else:
            print("\n📝 No API keys were updated.")
    
    def activate_autonomous_learning(self):
        """Activate autonomous learning system"""
        print("\n🤖 Autonomous Learning Activation:")
        print("-" * 35)
        
        choice = input("Activate autonomous learning? (y/n): ").lower().strip()
        if choice == 'y':
            # Load autonomous config
            with open(self.autonomous_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Enable autonomous learning
            config["autonomous_learning"]["enabled"] = True
            config["training_schedules"]["enabled"] = True
            config["skills_development"]["enabled"] = True
            config["background_operations"]["enabled"] = True
            
            # Configure learning schedule
            print("\n⏰ Learning Schedule Configuration:")
            interval = input("   Learning interval (minutes, default 30): ").strip()
            if interval.isdigit():
                config["autonomous_learning"]["schedule"]["interval_minutes"] = int(interval)
            
            # Save config
            with open(self.autonomous_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Autonomous learning system activated!")
            print(f"📊 Learning will occur every {config['autonomous_learning']['schedule']['interval_minutes']} minutes")
        else:
            print("⏭️  Autonomous learning activation skipped.")
    
    def show_system_status(self):
        """Show current system configuration status"""
        print("\n📊 System Configuration Status:")
        print("-" * 35)
        
        # Check API configuration
        with open(self.api_config_file, 'r', encoding='utf-8') as f:
            api_config = json.load(f)
        
        configured_apis = 0
        total_apis = 0
        for service, config in api_config.items():
            total_apis += 1
            api_key = config.get("api_key", "")
            if api_key and not api_key.startswith("your-") and config.get("enabled", False):
                configured_apis += 1
                print(f"✅ {service.upper()}: Configured & Enabled")
            elif config.get("enabled", False):
                print(f"⚠️  {service.upper()}: Enabled but no API key")
            else:
                print(f"❌ {service.upper()}: Disabled")
        
        print(f"\n📈 API Services: {configured_apis}/{total_apis} configured")
        
        # Check autonomous learning
        with open(self.autonomous_config_file, 'r', encoding='utf-8') as f:
            auto_config = json.load(f)
        
        if auto_config["autonomous_learning"]["enabled"]:
            print("🤖 Autonomous Learning: ✅ Enabled")
            interval = auto_config["autonomous_learning"]["schedule"]["interval_minutes"]
            print(f"⏰ Learning Interval: {interval} minutes")
        else:
            print("🤖 Autonomous Learning: ❌ Disabled")
        
        if auto_config["training_schedules"]["enabled"]:
            print("🏋️  Training Schedules: ✅ Enabled")
        else:
            print("🏋️  Training Schedules: ❌ Disabled")
    
    def create_launch_script(self):
        """Create a launch script for the enhanced system"""
        launch_script = '''#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 System Launcher
Launch the system with all advanced features enabled.
"""

import subprocess
import sys
import time
from pathlib import Path

def launch_enhanced_fsot():
    """Launch the Enhanced FSOT 2.0 system"""
    print("🚀 Launching Enhanced FSOT 2.0 Neuromorphic AI System...")
    print("=" * 60)
    
    # Launch main system
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\\n🛑 System shutdown requested by user.")
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(launch_enhanced_fsot())
'''
        
        launch_file = Path(__file__).parent / "launch_enhanced_fsot.py"
        with open(launch_file, 'w', encoding='utf-8') as f:
            f.write(launch_script)
        
        print(f"\n🚀 Launch script created: {launch_file}")
        print("   You can now run: python launch_enhanced_fsot.py")
    
    def run_configuration(self):
        """Run the complete configuration process"""
        self.print_banner()
        
        # Show instructions
        show_instructions = input("Show API key instructions? (y/n): ").lower().strip()
        if show_instructions == 'y':
            self.print_api_instructions()
            input("\\nPress Enter to continue...")
        
        # Configure API keys
        self.configure_api_keys()
        
        # Activate autonomous learning
        self.activate_autonomous_learning()
        
        # Show system status
        self.show_system_status()
        
        # Create launch script
        create_launcher = input("\\nCreate system launcher script? (y/n): ").lower().strip()
        if create_launcher == 'y':
            self.create_launch_script()
        
        print("\\n🎉 Enhanced FSOT 2.0 configuration complete!")
        print("🚀 Your neuromorphic AI system is ready for deployment!")
        print("\\n📋 Next steps:")
        print("   1. Ensure you have all required API keys configured")
        print("   2. Run: python main.py")
        print("   3. Or use: python launch_enhanced_fsot.py")
        print("\\n💡 Tip: The system will start autonomous learning automatically!")

def main():
    """Main entry point"""
    try:
        configurator = FSOTConfigurator()
        configurator.run_configuration()
    except KeyboardInterrupt:
        print("\\n\\n🛑 Configuration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Configuration error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
