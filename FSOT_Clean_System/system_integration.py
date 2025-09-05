#!/usr/bin/env python3
"""
System Integration Hub for Enhanced FSOT 2.0
===========================================

Central integration point that combines all system capabilities into a unified interface.
Manages communication between neuromorphic brain, API system, and new capabilities.

Author: GitHub Copilot
"""

import os
import sys
import json
import time
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all system components
try:
    from brain.brain_orchestrator import BrainOrchestrator
    from core.fsot_engine import FSOTEngine
    from core.consciousness import ConsciousnessMonitor
    from capabilities.desktop_control import FreeDesktopController
    from capabilities.advanced_training import AdvancedFreeTrainingSystem
    from interfaces.web_dashboard import SimpleFSOTDashboard
    from interfaces.cli_interface import CLIInterface
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.warning(f"Some components not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFSOTSystem:
    """
    Enhanced Free Self-Organizing Transformative (FSOT) System 2.0
    ==============================================================
    
    Complete integration of all system capabilities:
    - Neuromorphic Brain with synaptic learning
    - Free API Discovery and management
    - Enhanced memory systems (working, long-term, episodic, etc.)
    - Multi-modal processing (vision, audio, text)
    - Continuous API discovery
    - Web dashboard interface
    - Desktop control automation
    - Advanced training and learning
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/system_config.json"
        self.system_id = f"fsot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # Component instances
        self.components = {}
        self.component_status = {}
        
        # System metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "learning_sessions": 0,
            "apis_discovered": 0,
            "memory_entries": 0,
            "automation_actions": 0,
            "uptime_seconds": 0,
            "last_activity": None
        }
        
        # Event system for component communication
        self.event_listeners = {}
        self.event_queue = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Enhanced FSOT System {self.system_id} initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "components": {
                "neuromorphic_brain": {"enabled": True, "learning_rate": 0.1},
                "api_discovery": {"enabled": True, "max_apis": 50},
                "memory_system": {"enabled": True, "max_working_memory": 100},
                "multimodal_processor": {"enabled": True, "vision_enabled": True},
                "continuous_discovery": {"enabled": True, "discovery_interval": 3600},
                "web_dashboard": {"enabled": True, "port": 5000},
                "desktop_control": {"enabled": False, "safety_mode": True},
                "training_system": {"enabled": True, "auto_training": False}
            },
            "system": {
                "auto_start": False,
                "log_level": "INFO",
                "max_log_entries": 10000,
                "backup_interval": 3600
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for section in default_config:
                        if section in loaded_config:
                            default_config[section].update(loaded_config[section])
                        
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def _initialize_components(self):
        """Initialize all system components"""
        if not COMPONENTS_AVAILABLE:
            logger.error("Required components not available")
            return
        
        try:
            # Initialize Brain Orchestrator (replaces Neuromorphic Brain)
            if self.config["components"]["neuromorphic_brain"]["enabled"]:
                self.components["brain"] = BrainOrchestrator()
                self.component_status["brain"] = "initialized"
                logger.info("✅ Brain Orchestrator initialized")
            
            # Initialize FSOT Engine (replaces API Discovery)
            if self.config["components"]["api_discovery"]["enabled"]:
                self.components["fsot_engine"] = FSOTEngine()
                self.component_status["fsot_engine"] = "initialized"
                logger.info("✅ FSOT Engine initialized")
            
            # Initialize Consciousness Monitor (replaces Enhanced Memory System)
            if self.config["components"]["memory_system"]["enabled"]:
                self.components["consciousness"] = ConsciousnessMonitor()
                self.component_status["consciousness"] = "initialized"
                logger.info("✅ Consciousness Monitor initialized")
            
            # Initialize CLI Interface (replaces Multi-modal Processor)
            if self.config["components"]["multimodal_processor"]["enabled"]:
                self.components["cli"] = CLIInterface()
                self.component_status["cli"] = "initialized"
                logger.info("✅ CLI Interface initialized")
            
            # Initialize Desktop Controller (available component)
            if self.config["components"]["continuous_discovery"]["enabled"]:
                self.components["desktop_control"] = FreeDesktopController()
                self.component_status["desktop_control"] = "initialized"
                logger.info("✅ Desktop Controller initialized")
            
            # Initialize Web Dashboard
            if self.config["components"]["web_dashboard"]["enabled"]:
                self.components["dashboard"] = SimpleFSOTDashboard(
                    port=self.config["components"]["web_dashboard"].get("port", 5000)
                )
                self.component_status["dashboard"] = "initialized"
                logger.info("✅ Web Dashboard initialized")
            
            # Initialize Desktop Control (disabled by default for safety)
            if self.config["components"]["desktop_control"]["enabled"]:
                self.components["desktop"] = FreeDesktopController()
                if self.config["components"]["desktop_control"]["safety_mode"]:
                    self.components["desktop"].enable_safety_mode(True)
                self.component_status["desktop"] = "initialized"
                logger.info("✅ Desktop Control initialized (safety mode)")
            
            # Initialize Training System
            if self.config["components"]["training_system"]["enabled"]:
                self.components["training"] = AdvancedFreeTrainingSystem()
                self.component_status["training"] = "initialized"
                logger.info("✅ Advanced Training System initialized")
            
            self.is_initialized = True
            logger.info(f"🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            self.is_initialized = False
    
    def start_system(self):
        """Start the complete FSOT system"""
        if not self.is_initialized:
            logger.error("System not initialized - cannot start")
            return False
        
        if self.is_running:
            logger.warning("System already running")
            return True
        
        try:
            self.start_time = datetime.now()
            self.is_running = True
            
            # Start individual components
            self._start_components()
            
            # Start system monitoring
            self._start_monitoring()
            
            # Start event processing
            self._start_event_processing()
            
            # Send startup event
            self._emit_event("system_started", {
                "system_id": self.system_id,
                "timestamp": self.start_time.isoformat(),
                "components": list(self.components.keys())
            })
            
            logger.info(f"🚀 Enhanced FSOT System started successfully!")
            return True
            
        except Exception as e:
            logger.error(f"System startup error: {e}")
            self.is_running = False
            return False
    
    def _start_components(self):
        """Start individual components"""
        
        # Start dashboard
        if "dashboard" in self.components:
            self.components["dashboard"].start_dashboard()
            self.component_status["dashboard"] = "running"
        
        # Start continuous API discovery
        if "continuous_discovery" in self.components:
            self.components["continuous_discovery"].start_discovery()
            self.component_status["continuous_discovery"] = "running"
        
        # Activate desktop control if enabled
        if "desktop" in self.components:
            self.components["desktop"].activate()
            self.component_status["desktop"] = "running"
        
        # Start auto-training if enabled
        if ("training" in self.components and 
            self.config["components"]["training_system"].get("auto_training", False)):
            self.components["training"].start_training_session()
            self.component_status["training"] = "training"
    
    def _start_monitoring(self):
        """Start system monitoring thread"""
        def monitor_loop():
            while self.is_running:
                try:
                    self._update_metrics()
                    self._check_component_health()
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("📊 System monitoring started")
    
    def _start_event_processing(self):
        """Start event processing thread"""
        def event_loop():
            while self.is_running:
                try:
                    if self.event_queue:
                        event = self.event_queue.pop(0)
                        self._process_event(event)
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Event processing error: {e}")
                    time.sleep(1)
        
        event_thread = threading.Thread(target=event_loop, daemon=True)
        event_thread.start()
        logger.info("📨 Event processing started")
    
    def _update_metrics(self):
        """Update system metrics"""
        if self.start_time:
            self.metrics["uptime_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        # Collect metrics from components
        if "memory" in self.components:
            memory_stats = self.components["memory"].get_memory_statistics()
            self.metrics["memory_entries"] = memory_stats.get("total_memories", 0)
        
        if "training" in self.components:
            training_status = self.components["training"].get_status()
            self.metrics["learning_sessions"] = training_status["metrics"]["training_sessions"]
        
        if "continuous_discovery" in self.components:
            discovery_stats = self.components["continuous_discovery"].get_statistics()
            self.metrics["apis_discovered"] = discovery_stats.get("total_apis", 0)
        
        if "desktop" in self.components:
            desktop_status = self.components["desktop"].get_status()
            self.metrics["automation_actions"] = desktop_status.get("total_actions", 0)
    
    def _check_component_health(self):
        """Check health of all components"""
        for component_name, component in self.components.items():
            try:
                if hasattr(component, "get_status"):
                    status = component.get_status()
                    if status.get("error"):
                        self.component_status[component_name] = "error"
                        logger.warning(f"Component {component_name} has errors")
                    else:
                        if self.component_status[component_name] == "error":
                            self.component_status[component_name] = "running"
                            logger.info(f"Component {component_name} recovered")
            except Exception as e:
                logger.error(f"Health check error for {component_name}: {e}")
                self.component_status[component_name] = "error"
    
    def stop_system(self):
        """Stop the complete FSOT system"""
        if not self.is_running:
            logger.warning("System not running")
            return
        
        logger.info("🛑 Stopping Enhanced FSOT System...")
        
        try:
            self.is_running = False
            
            # Stop components
            if "dashboard" in self.components:
                self.components["dashboard"].stop_dashboard()
            
            if "continuous_discovery" in self.components:
                self.components["continuous_discovery"].stop_discovery()
            
            if "desktop" in self.components:
                self.components["desktop"].deactivate()
            
            if "training" in self.components:
                self.components["training"].stop_training_session()
            
            # Send shutdown event
            self._emit_event("system_stopped", {
                "system_id": self.system_id,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": self.metrics["uptime_seconds"]
            })
            
            logger.info("✅ Enhanced FSOT System stopped successfully")
            
        except Exception as e:
            logger.error(f"System shutdown error: {e}")
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query through the complete system"""
        try:
            self.metrics["total_queries"] += 1
            self.metrics["last_activity"] = datetime.now().isoformat()
            
            query_context = context or {}
            query_context["query"] = query
            query_context["timestamp"] = datetime.now().isoformat()
            
            # Store query in memory
            if "memory" in self.components:
                self.components["memory"].store_memory(
                    "query",
                    query,
                    {"context": query_context}
                )
            
            # Process through brain
            brain_response = None
            if "brain" in self.components:
                brain_response = self.components["brain"].process_input(query)
            
            # Search APIs for additional information
            api_results = []
            if "api_discovery" in self.components:
                apis = self.components["api_discovery"].get_available_apis()
                # Use first few APIs for quick response
                for api_name in list(apis.keys())[:3]:
                    try:
                        result = self.components["api_discovery"].search_with_api(api_name, query)
                        if result and result.get("success"):
                            api_results.append({
                                "source": api_name,
                                "data": result["data"]
                            })
                    except Exception as e:
                        logger.error(f"API search error: {e}")
            
            # Process through multi-modal if available
            multimodal_analysis = None
            if "multimodal" in self.components:
                multimodal_analysis = self.components["multimodal"].analyze_text(query)
            
            # Combine all responses
            combined_response = self._combine_responses(
                query, brain_response, api_results, multimodal_analysis
            )
            
            # Store response in memory
            if "memory" in self.components:
                self.components["memory"].store_memory(
                    "response",
                    combined_response["content"],
                    {"query": query, "sources": combined_response["sources"]}
                )
            
            # Emit query event
            self._emit_event("query_processed", {
                "query": query,
                "response_length": len(combined_response["content"]),
                "sources": combined_response["sources"],
                "success": True
            })
            
            self.metrics["successful_queries"] += 1
            return combined_response
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            
            # Emit error event
            self._emit_event("query_error", {
                "query": query,
                "error": str(e),
                "success": False
            })
            
            return {
                "content": f"I encountered an error processing your query: {str(e)}",
                "sources": ["error_handler"],
                "success": False,
                "error": str(e)
            }
    
    def _combine_responses(self, query: str, brain_response: Any, 
                          api_results: List[Dict], multimodal_analysis: Any) -> Dict[str, Any]:
        """Combine responses from different components"""
        
        response_parts = []
        sources = []
        
        # Add brain response
        if brain_response:
            if hasattr(brain_response, 'response') and brain_response.response:
                response_parts.append(f"Brain analysis: {brain_response.response}")
                sources.append("neuromorphic_brain")
        
        # Add API results
        for api_result in api_results:
            if isinstance(api_result["data"], str):
                response_parts.append(f"From {api_result['source']}: {api_result['data']}")
            elif isinstance(api_result["data"], dict):
                # Extract meaningful content from dict
                if "summary" in api_result["data"]:
                    response_parts.append(f"From {api_result['source']}: {api_result['data']['summary']}")
                elif "description" in api_result["data"]:
                    response_parts.append(f"From {api_result['source']}: {api_result['data']['description']}")
            sources.append(api_result["source"])
        
        # Add multimodal analysis
        if multimodal_analysis and multimodal_analysis.get("analysis"):
            response_parts.append(f"Text analysis: {multimodal_analysis['analysis']}")
            sources.append("multimodal_processor")
        
        # Combine all parts
        if response_parts:
            combined_content = "\n\n".join(response_parts)
        else:
            combined_content = f"I understand you're asking about '{query}'. Let me search my knowledge base for relevant information."
        
        return {
            "content": combined_content,
            "sources": sources,
            "success": True,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit system event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "system_id": self.system_id
        }
        
        self.event_queue.append(event)
        
        # Also log important events
        if event_type in ["system_started", "system_stopped", "query_error"]:
            logger.info(f"Event: {event_type} - {data}")
    
    def _process_event(self, event: Dict[str, Any]):
        """Process system event"""
        event_type = event["type"]
        
        # Call registered listeners
        if event_type in self.event_listeners:
            for listener in self.event_listeners[event_type]:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Event listener error: {e}")
        
        # Update dashboard if available
        if "dashboard" in self.components:
            dashboard = self.components["dashboard"]
            
            if event_type == "query_processed":
                dashboard.log_activity("query", f"Processed query: {event['data']['query'][:50]}...")
            elif event_type == "system_started":
                dashboard.log_activity("system", "Enhanced FSOT System started")
            elif event_type == "system_stopped":
                dashboard.log_activity("system", "Enhanced FSOT System stopped")
    
    def register_event_listener(self, event_type: str, listener: Callable):
        """Register event listener"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_id": self.system_id,
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": self.metrics["uptime_seconds"],
            "components": {},
            "metrics": self.metrics,
            "config": self.config
        }
        
        # Get individual component status
        for component_name, component in self.components.items():
            try:
                if hasattr(component, "get_status"):
                    component_status = component.get_status()
                else:
                    component_status = {"status": "unknown"}
                
                status["components"][component_name] = {
                    "status": self.component_status.get(component_name, "unknown"),
                    "details": component_status
                }
            except Exception as e:
                status["components"][component_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_overview": self.get_system_status(),
            "component_reports": {}
        }
        
        # Get detailed reports from each component
        if "memory" in self.components:
            report["component_reports"]["memory"] = self.components["memory"].get_memory_statistics()
        
        if "training" in self.components:
            report["component_reports"]["training"] = self.components["training"].generate_training_report()
        
        if "continuous_discovery" in self.components:
            report["component_reports"]["api_discovery"] = self.components["continuous_discovery"].get_statistics()
        
        if "brain" in self.components:
            report["component_reports"]["brain"] = {
                "synapses": len(getattr(self.components["brain"], 'synapses', {})),
                "learning_rate": getattr(self.components["brain"], 'learning_rate', 0.1)
            }
        
        return report
    
    def save_system_state(self, filepath: Optional[str] = None):
        """Save current system state to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"system_state_{timestamp}.json"
        
        try:
            state = self.get_system_status()
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"System state saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            return None

# Global system instance
_fsot_system = None

def get_fsot_system() -> EnhancedFSOTSystem:
    """Get or create global FSOT system instance"""
    global _fsot_system
    if _fsot_system is None:
        _fsot_system = EnhancedFSOTSystem()
    return _fsot_system

def main():
    """Main function for running the complete system"""
    print("🧠 Enhanced FSOT 2.0 - Complete System")
    print("=" * 50)
    
    # Create and start system
    system = EnhancedFSOTSystem()
    
    if not system.is_initialized:
        print("❌ System initialization failed")
        return
    
    print("✅ System initialized successfully")
    
    # Start system
    if system.start_system():
        print("🚀 System started successfully")
        print(f"📊 Dashboard available at: http://localhost:{system.config['components']['web_dashboard']['port']}")
        
        try:
            # Interactive loop
            while system.is_running:
                user_input = input("\n💬 Enter query (or 'quit' to exit): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    print("🤔 Processing query...")
                    response = system.process_query(user_input)
                    print(f"\n🤖 Response: {response['content']}")
                    print(f"📚 Sources: {', '.join(response['sources'])}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Received interrupt signal")
        
        finally:
            # Stop system
            system.stop_system()
            print("👋 System stopped. Goodbye!")
    
    else:
        print("❌ System startup failed")

if __name__ == "__main__":
    main()
