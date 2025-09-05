"""
FSOT 2.0 System Configuration Management
Clean, centralized configuration for the entire system
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class BrainConfig:
    """Brain module configuration"""
    consciousness_threshold: float = 0.5
    learning_rate: float = 0.001
    memory_capacity: int = 10000
    neural_signal_timeout: float = 1.0
    auto_cleanup: bool = True

@dataclass
class FSOTConfig:
    """FSOT 2.0 engine configuration"""
    max_dimensions: int = 25
    default_dimensions: int = 12
    scaling_constant: float = 0.4202
    consciousness_factor: float = 0.288
    use_observer_effects: bool = True

@dataclass
class SystemConfig:
    """Overall system configuration"""
    log_level: str = "INFO"
    web_port: int = 8000
    enable_web_interface: bool = True
    auto_save_interval: int = 300  # seconds
    max_concurrent_tasks: int = 10

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        self.brain_config = BrainConfig()
        self.fsot_config = FSOTConfig()
        self.system_config = SystemConfig()
        
        # Load configurations from files if they exist
        self.load_configs()
    
    def load_configs(self):
        """Load configurations from JSON files"""
        brain_config_file = self.config_dir / "brain_config.json"
        if brain_config_file.exists():
            with open(brain_config_file, 'r') as f:
                data = json.load(f)
                self.brain_config = BrainConfig(**data.get('brain', {}))
                self.fsot_config = FSOTConfig(**data.get('fsot', {}))
                self.system_config = SystemConfig(**data.get('system', {}))
    
    def save_configs(self):
        """Save current configurations to JSON file"""
        config_data = {
            'brain': asdict(self.brain_config),
            'fsot': asdict(self.fsot_config),
            'system': asdict(self.system_config)
        }
        
        brain_config_file = self.config_dir / "brain_config.json"
        with open(brain_config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def get_env_var(self, key: str, default: Any = None) -> Any:
        """Get environment variable with fallback"""
        return os.getenv(key, default)
    
    def update_brain_config(self, **kwargs):
        """Update brain configuration"""
        for key, value in kwargs.items():
            if hasattr(self.brain_config, key):
                setattr(self.brain_config, key, value)
        self.save_configs()
    
    def update_fsot_config(self, **kwargs):
        """Update FSOT configuration"""
        for key, value in kwargs.items():
            if hasattr(self.fsot_config, key):
                setattr(self.fsot_config, key, value)
        self.save_configs()

# Global configuration instance
config = ConfigManager()
