#!/usr/bin/env python3
"""
ENHANCED FSOT 2.0 INTEGRATION MANAGER
=====================================

Central integration manager that coordinates all advanced capabilities:
- API Manager: External API access
- Autonomous Learning: Sophisticated learning system
- Web Search Engine: Production-grade search
- Skills Database: Learning progression tracking
- Training Facility: 10-module evaluation system

Author: GitHub Copilot
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import integration modules
from .api_manager import APIManager
from .autonomous_learning import AutonomousLearningSystem
from .web_search_engine import EnhancedWebSearchEngine
from .skills_database import SkillsDatabase, create_default_skills
from .training_facility import EnhancedTrainingFacility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFSOTIntegration:
    """Enhanced FSOT 2.0 integration manager for all advanced capabilities"""
    
    def __init__(self, brain_orchestrator=None):
        self.brain_orchestrator = brain_orchestrator
        self.start_time = datetime.now()
        
        # Initialize all integration components
        logger.info("Initializing Enhanced FSOT 2.0 Integration...")
        
        # API Manager
        self.api_manager = APIManager()
        if brain_orchestrator:
            self.api_manager.set_brain_context(brain_orchestrator)
        
        # Web Search Engine
        self.web_search = EnhancedWebSearchEngine(brain_orchestrator, self.api_manager)
        
        # Autonomous Learning System
        self.autonomous_learning = AutonomousLearningSystem(
            brain_orchestrator, self.api_manager, self.web_search
        )
        
        # Skills Database
        self.skills_database = SkillsDatabase(brain_orchestrator=brain_orchestrator)
        
        # Training Facility
        self.training_facility = EnhancedTrainingFacility(brain_orchestrator)
        
        # Integration status
        self.integration_status = {
            "initialized": True,
            "api_enabled": self.api_manager.get_api_status(),
            "learning_active": False,
            "training_active": False
        }
        
        # Initialize default skills if none exist
        if not self.skills_database.skills:
            self._initialize_default_skills()
        
        logger.info("Enhanced FSOT 2.0 Integration initialized successfully")
    
    def _initialize_default_skills(self):
        """Initialize default skills for the neuromorphic system"""
        logger.info("Initializing default skills...")
        default_skills = create_default_skills()
        
        for skill in default_skills:
            self.skills_database.add_skill(skill)
        
        logger.info(f"Initialized {len(default_skills)} default skills")
    
    def start_autonomous_learning(self, background: bool = True) -> bool:
        """Start autonomous learning system"""
        try:
            if background:
                self.autonomous_learning.start_background_learning()
            else:
                # Run single learning cycle
                self.autonomous_learning.autonomous_learning_cycle(duration_minutes=15)
            
            self.integration_status["learning_active"] = True
            logger.info("Autonomous learning started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start autonomous learning: {e}")
            return False
    
    def stop_autonomous_learning(self) -> bool:
        """Stop autonomous learning system"""
        try:
            self.autonomous_learning.stop_background_learning()
            self.integration_status["learning_active"] = False
            logger.info("Autonomous learning stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop autonomous learning: {e}")
            return False
    
    def search_and_learn(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Search for information and learn from results"""
        try:
            # Perform web search
            search_results = self.web_search.search(query, num_results=5)
            
            if not search_results:
                return {"success": False, "reason": "No search results"}
            
            # Learn from search results
            learning_result = self.autonomous_learning.learn_from_search(query, domain)
            
            # Extract key concepts for skills tracking
            if learning_result["success"]:
                # Find relevant skills to practice
                relevant_skills = self._find_relevant_skills(query, domain)
                
                # Practice relevant skills (simulated)
                for skill_id in relevant_skills[:3]:  # Top 3 relevant skills
                    self.skills_database.practice_skill(
                        skill_id, 
                        duration_minutes=5, 
                        quality_score=0.8,
                        notes=f"Learned from search: {query}"
                    )
            
            return {
                "success": True,
                "search_results": len(search_results),
                "learning_result": learning_result,
                "skills_practiced": len(relevant_skills[:3]) if learning_result["success"] else 0
            }
            
        except Exception as e:
            logger.error(f"Search and learn failed: {e}")
            return {"success": False, "reason": str(e)}
    
    def _find_relevant_skills(self, query: str, domain: str = None) -> List[str]:
        """Find skills relevant to the query/domain"""
        relevant_skills = []
        query_lower = query.lower()
        
        for skill in self.skills_database.skills.values():
            # Check if skill name or category matches query
            if (skill.name.lower() in query_lower or 
                query_lower in skill.name.lower() or
                (domain and skill.category.value == domain.lower())):
                relevant_skills.append(skill.id)
        
        return relevant_skills
    
    def run_comprehensive_training(self) -> Dict[str, Any]:
        """Run comprehensive training and evaluation"""
        try:
            self.integration_status["training_active"] = True
            
            # Run training facility evaluation
            training_results = self.training_facility.run_comprehensive_evaluation()
            
            # Update skills based on training results
            self._update_skills_from_training(training_results)
            
            self.integration_status["training_active"] = False
            
            return {
                "success": True,
                "training_results": training_results,
                "overall_score": training_results["overall_performance"]["overall_score"]
            }
            
        except Exception as e:
            logger.error(f"Comprehensive training failed: {e}")
            self.integration_status["training_active"] = False
            return {"success": False, "reason": str(e)}
    
    def _update_skills_from_training(self, training_results: Dict[str, Any]):
        """Update skills based on training results"""
        overall_score = training_results["overall_performance"]["overall_score"]
        
        # Practice skills based on training performance
        for module_name, module_result in training_results["modules"].items():
            module_score = module_result.get("score", 0)
            
            # Find skills related to this brain module
            related_skills = []
            for skill in self.skills_database.skills.values():
                if module_name in skill.brain_modules_involved:
                    related_skills.append(skill.id)
            
            # Practice related skills with quality based on module performance
            for skill_id in related_skills:
                self.skills_database.practice_skill(
                    skill_id,
                    duration_minutes=10,
                    quality_score=module_score,
                    notes=f"Training evaluation: {module_name}"
                )
    
    def enable_api(self, api_name: str, api_key: str) -> bool:
        """Enable an API with the provided key"""
        try:
            # Update API configuration
            if api_name in self.api_manager.config:
                self.api_manager.config[api_name]["api_key"] = api_key
                self.api_manager.config[api_name]["enabled"] = True
                
                # Save configuration
                config_path = Path(self.api_manager.config_path)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w') as f:
                    json.dump(self.api_manager.config, f, indent=2)
                
                # Update status
                self.integration_status["api_enabled"] = self.api_manager.get_api_status()
                
                logger.info(f"Enabled API: {api_name}")
                return True
            else:
                logger.error(f"Unknown API: {api_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to enable API {api_name}: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        # Learning status
        learning_status = self.autonomous_learning.get_learning_status()
        
        # Skills status
        skills_status = self.skills_database.get_skill_statistics()
        
        # Training status
        training_status = self.training_facility.get_training_statistics()
        
        # Web search status
        search_status = self.web_search.get_search_statistics()
        
        return {
            "system_info": {
                "initialized": self.integration_status["initialized"],
                "uptime": str(datetime.now() - self.start_time),
                "learning_active": self.integration_status["learning_active"],
                "training_active": self.integration_status["training_active"]
            },
            "api_status": self.integration_status["api_enabled"],
            "learning_status": learning_status,
            "skills_status": skills_status,
            "training_status": training_status,
            "search_status": search_status
        }
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get comprehensive system recommendations"""
        recommendations = []
        
        # API recommendations
        api_status = self.api_manager.get_api_status()
        for api_name, status in api_status.items():
            if not status["enabled"]:
                recommendations.append({
                    "type": "api",
                    "priority": "medium",
                    "title": f"Enable {api_name} API",
                    "description": f"Add API key to enable {api_name} integration"
                })
        
        # Learning recommendations
        learning_recs = self.skills_database.get_learning_recommendations()
        for rec in learning_recs[:3]:  # Top 3
            recommendations.append({
                "type": "learning",
                "priority": rec.get("priority", "medium"),
                "title": rec.get("skill", ""),
                "description": rec.get("reason", "")
            })
        
        # Training recommendations
        if not self.integration_status["training_active"]:
            recommendations.append({
                "type": "training",
                "priority": "low",
                "title": "Run comprehensive training",
                "description": "Evaluate all brain modules for optimization opportunities"
            })
        
        return recommendations[:10]  # Top 10 recommendations
    
    def shutdown(self):
        """Shutdown integration system gracefully"""
        logger.info("Shutting down Enhanced FSOT 2.0 Integration...")
        
        # Stop autonomous learning
        self.stop_autonomous_learning()
        
        # Save all data
        self.skills_database._save_database()
        self.autonomous_learning._save_knowledge_state()
        
        logger.info("Enhanced FSOT 2.0 Integration shutdown complete")

def demo_integration():
    """Demonstration of the integration system"""
    print("Enhanced FSOT 2.0 Integration System Demo")
    print("=" * 50)
    
    # Initialize integration (without brain orchestrator for demo)
    integration = EnhancedFSOTIntegration()
    
    # Get status
    status = integration.get_integration_status()
    print(f"\nSystem Status:")
    print(f"  Initialized: {status['system_info']['initialized']}")
    print(f"  Uptime: {status['system_info']['uptime']}")
    print(f"  Total Skills: {status['skills_status']['total_skills']}")
    
    # Show API status
    print(f"\nAPI Status:")
    for api, info in status['api_status'].items():
        enabled = "✅" if info['enabled'] else "❌"
        print(f"  {api}: {enabled}")
    
    # Show recommendations
    recommendations = integration.get_recommendations()
    print(f"\nRecommendations ({len(recommendations)}):")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. {rec['title']} ({rec['priority']} priority)")
    
    # Demo search and learn
    print(f"\nDemo: Search and Learn")
    result = integration.search_and_learn("artificial intelligence advancements")
    print(f"  Success: {result['success']}")
    if result['success']:
        print(f"  Search Results: {result['search_results']}")
        print(f"  Skills Practiced: {result['skills_practiced']}")
    
    # Demo training
    print(f"\nDemo: Comprehensive Training")
    training_result = integration.run_comprehensive_training()
    print(f"  Success: {training_result['success']}")
    if training_result['success']:
        print(f"  Overall Score: {training_result['overall_score']:.3f}")
    
    # Shutdown
    integration.shutdown()
    print(f"\nDemo completed!")

if __name__ == "__main__":
    demo_integration()
