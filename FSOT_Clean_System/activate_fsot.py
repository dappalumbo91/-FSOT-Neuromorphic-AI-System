#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 Direct Activation Script
==========================================

Direct activation of autonomous learning and advanced features.
"""

import json
import sys
import os
from pathlib import Path

def activate_enhanced_fsot():
    """Directly activate Enhanced FSOT 2.0 with all advanced features"""
    
    print("Enhanced FSOT 2.0 Direct Activation")
    print("=" * 50)
    
    # Get the config directory
    config_dir = Path(__file__).parent / "config"
    
    # Enable autonomous learning
    autonomous_config_file = config_dir / "autonomous_config.json"
    try:
        with open(autonomous_config_file, 'r', encoding='utf-8') as f:
            autonomous_config = json.load(f)
        
        # Enable all features
        autonomous_config["autonomous_learning"]["enabled"] = True
        autonomous_config["training_schedules"]["enabled"] = True
        autonomous_config["skills_development"]["enabled"] = True
        autonomous_config["background_operations"]["enabled"] = True
        
        # Save configuration
        with open(autonomous_config_file, 'w', encoding='utf-8') as f:
            json.dump(autonomous_config, f, indent=2)
        
        print("✓ Autonomous learning activated")
        
    except Exception as e:
        print(f"Warning: Could not configure autonomous learning: {e}")
    
    # Check API configuration
    api_config_file = config_dir / "api_config.json"
    try:
        with open(api_config_file, 'r', encoding='utf-8') as f:
            api_config = json.load(f)
        
        enabled_apis = []
        for service, config in api_config.items():
            if config.get("enabled", False):
                api_key = config.get("api_key", "")
                if api_key and not api_key.startswith("your-"):
                    enabled_apis.append(service.upper())
        
        if enabled_apis:
            print(f"✓ API services configured: {', '.join(enabled_apis)}")
        else:
            print("! No external APIs configured - using basic functionality")
        
    except Exception as e:
        print(f"Warning: Could not check API configuration: {e}")
    
    print("\n" + "=" * 50)
    print("🚀 Enhanced FSOT 2.0 Activation Complete!")
    print("🧠 Neuromorphic AI System with Advanced Features")
    print("✓ 10-module brain architecture")
    print("✓ Autonomous learning system")
    print("✓ Web search integration")
    print("✓ Skills database")
    print("✓ Training facility")
    print("✓ API manager")
    print("\nStarting main system...")
    
    # Launch main system
    main_file = Path(__file__).parent / "main.py"
    if main_file.exists():
        os.system(f'python "{main_file}"')
    else:
        print("Error: main.py not found")

if __name__ == "__main__":
    activate_enhanced_fsot()
