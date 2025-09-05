"""
Enhanced FSOT 2.0 Integration Package
=====================================

This package provides comprehensive integration capabilities for the
Enhanced FSOT 2.0 neuromorphic AI system, including:

- API Manager: External API access and coordination
- Autonomous Learning: Sophisticated multi-domain learning system
- Web Search Engine: Production-grade search capabilities
- Skills Database: Learning progression tracking
- Training Facility: 10-module evaluation system
- Integration Manager: Central coordination system

Author: GitHub Copilot
"""

from .api_manager import APIManager, create_api_config_template
from .autonomous_learning import AutonomousLearningSystem, LearningDomain, KnowledgeGraph
from .web_search_engine import EnhancedWebSearchEngine, SearchResult, WebSearchCache
from .skills_database import SkillsDatabase, Skill, SkillLevel, SkillCategory, create_default_skills
from .training_facility import EnhancedTrainingFacility, BrainModuleEvaluator
from .integration_manager import EnhancedFSOTIntegration

__all__ = [
    # API Manager
    'APIManager',
    'create_api_config_template',
    
    # Autonomous Learning
    'AutonomousLearningSystem',
    'LearningDomain',
    'KnowledgeGraph',
    
    # Web Search Engine
    'EnhancedWebSearchEngine',
    'SearchResult',
    'WebSearchCache',
    
    # Skills Database
    'SkillsDatabase',
    'Skill',
    'SkillLevel',
    'SkillCategory',
    'create_default_skills',
    
    # Training Facility
    'EnhancedTrainingFacility',
    'BrainModuleEvaluator',
    
    # Integration Manager
    'EnhancedFSOTIntegration'
]

__version__ = "2.0.0"
__author__ = "GitHub Copilot"
