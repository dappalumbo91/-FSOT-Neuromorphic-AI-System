#!/usr/bin/env python3
"""
ENHANCED FSOT 2.0 SKILLS DATABASE
=================================

JSON-based skills tracking system with learning progression, proficiency metrics,
and brain module integration for the enhanced neuromorphic architecture.

Author: GitHub Copilot
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import logging
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"

class SkillCategory(Enum):
    """Skill categories aligned with brain modules"""
    ANALYTICAL = "analytical"          # Frontal cortex
    CREATIVE = "creative"             # Frontal cortex + temporal
    MEMORY = "memory"                 # Hippocampus
    EMOTIONAL = "emotional"           # Amygdala
    MOTOR = "motor"                   # Cerebellum
    SENSORY = "sensory"              # Occipital, temporal
    LINGUISTIC = "linguistic"        # PFLT, temporal
    SPATIAL = "spatial"              # Parietal lobe
    LOGICAL = "logical"              # Frontal cortex
    SOCIAL = "social"                # Amygdala + frontal

class Skill:
    """Individual skill with learning progression"""
    
    def __init__(self, name: str, description: str, category: SkillCategory,
                 complexity: float = 0.5, prerequisites: List[str] = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.category = category
        self.complexity = complexity  # 0.0 to 1.0
        self.prerequisites = prerequisites or []
        
        # Learning progression
        self.current_level = SkillLevel.NOVICE
        self.proficiency_score = 0.0  # 0.0 to 1.0
        self.learning_sessions = []
        self.mastery_milestones = []
        
        # Brain integration
        self.brain_modules_involved = self._determine_brain_modules()
        self.consciousness_integration = 0.0
        
        # Metadata
        self.created_date = datetime.now()
        self.last_practiced = None
        self.total_practice_time = 0  # in minutes
        
    def _determine_brain_modules(self) -> List[str]:
        """Determine which brain modules are involved in this skill"""
        module_mapping = {
            SkillCategory.ANALYTICAL: ["frontal_cortex", "thalamus"],
            SkillCategory.CREATIVE: ["frontal_cortex", "temporal_lobe"],
            SkillCategory.MEMORY: ["hippocampus", "thalamus"],
            SkillCategory.EMOTIONAL: ["amygdala", "frontal_cortex"],
            SkillCategory.MOTOR: ["cerebellum", "brainstem"],
            SkillCategory.SENSORY: ["occipital_lobe", "thalamus"],
            SkillCategory.LINGUISTIC: ["pflt", "temporal_lobe"],
            SkillCategory.SPATIAL: ["parietal_lobe", "occipital_lobe"],
            SkillCategory.LOGICAL: ["frontal_cortex", "parietal_lobe"],
            SkillCategory.SOCIAL: ["amygdala", "frontal_cortex", "temporal_lobe"]
        }
        return module_mapping.get(self.category, ["frontal_cortex"])
    
    def practice(self, duration_minutes: int, quality_score: float = 0.8, 
                notes: str = "", brain_orchestrator=None) -> bool:
        """Record a practice session"""
        
        # Create practice session
        session = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "quality_score": quality_score,  # 0.0 to 1.0
            "notes": notes,
            "proficiency_before": self.proficiency_score
        }
        
        # Update proficiency based on practice
        improvement = self._calculate_improvement(duration_minutes, quality_score)
        self.proficiency_score = min(1.0, self.proficiency_score + improvement)
        
        # Update level if threshold reached
        self._update_skill_level()
        
        # Update metadata
        self.last_practiced = datetime.now()
        self.total_practice_time += duration_minutes
        
        # Record session
        session["proficiency_after"] = self.proficiency_score
        session["level_after"] = self.current_level.value
        self.learning_sessions.append(session)
        
        # Brain integration
        if brain_orchestrator:
            self._send_brain_signals(brain_orchestrator, "practice", session)
        
        logger.info(f"Practiced {self.name}: {self.proficiency_score:.2f} proficiency ({self.current_level.value})")
        return True
    
    def _calculate_improvement(self, duration: int, quality: float) -> float:
        """Calculate skill improvement from practice session"""
        base_improvement = (duration / 60) * quality * 0.05  # Base 5% per hour of quality practice
        
        # Complexity adjustment
        complexity_factor = 1.0 - (self.complexity * 0.3)  # Harder skills improve slower
        
        # Diminishing returns as proficiency increases
        diminishing_factor = 1.0 - (self.proficiency_score * 0.5)
        
        # FSOT consciousness enhancement
        consciousness_factor = 1.0 + (self.consciousness_integration * 0.2)
        
        improvement = base_improvement * complexity_factor * diminishing_factor * consciousness_factor
        return max(0.001, improvement)  # Minimum improvement
    
    def _update_skill_level(self):
        """Update skill level based on proficiency score"""
        level_thresholds = {
            0.0: SkillLevel.NOVICE,
            0.2: SkillLevel.BEGINNER,
            0.4: SkillLevel.INTERMEDIATE,
            0.6: SkillLevel.ADVANCED,
            0.8: SkillLevel.EXPERT,
            0.95: SkillLevel.MASTER
        }
        
        new_level = SkillLevel.NOVICE
        for threshold, level in level_thresholds.items():
            if self.proficiency_score >= threshold:
                new_level = level
        
        # Check for level up
        if new_level != self.current_level:
            milestone = {
                "timestamp": datetime.now().isoformat(),
                "old_level": self.current_level.value,
                "new_level": new_level.value,
                "proficiency_score": self.proficiency_score
            }
            self.mastery_milestones.append(milestone)
            self.current_level = new_level
            logger.info(f"Skill level up! {self.name}: {new_level.value}")
    
    def _send_brain_signals(self, brain_orchestrator, action: str, data: Dict):
        """Send signals to relevant brain modules"""
        for module in self.brain_modules_involved:
            signal = {
                "type": f"skill_{action}",
                "skill_id": self.id,
                "skill_name": self.name,
                "category": self.category.value,
                "proficiency": self.proficiency_score,
                "level": self.current_level.value,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            brain_orchestrator.send_signal(module, signal)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Generate learning insights for this skill"""
        insights = []
        
        # Practice frequency
        if self.learning_sessions:
            recent_sessions = [s for s in self.learning_sessions 
                             if datetime.fromisoformat(s["timestamp"]) > datetime.now() - timedelta(days=7)]
            if len(recent_sessions) >= 3:
                insights.append("Good practice frequency this week")
            elif len(recent_sessions) < 1:
                insights.append("Consider practicing more regularly")
        
        # Quality assessment
        if self.learning_sessions:
            avg_quality = sum(s["quality_score"] for s in self.learning_sessions) / len(self.learning_sessions)
            if avg_quality > 0.8:
                insights.append("High quality practice sessions")
            elif avg_quality < 0.5:
                insights.append("Focus on practice quality improvement")
        
        # Proficiency progression
        if self.proficiency_score > 0.7:
            insights.append("Approaching expert level")
        elif self.proficiency_score > 0.4:
            insights.append("Good intermediate progress")
        else:
            insights.append("Building foundational knowledge")
        
        # Complexity vs proficiency
        if self.complexity > 0.8 and self.proficiency_score > 0.5:
            insights.append("Mastering complex skill well")
        
        return {
            "insights": insights,
            "recommendation": self._get_practice_recommendation(),
            "next_milestone": self._get_next_milestone()
        }
    
    def _get_practice_recommendation(self) -> str:
        """Get personalized practice recommendation"""
        if not self.learning_sessions:
            return "Start with short, focused practice sessions"
        
        if self.last_practiced and (datetime.now() - self.last_practiced).days > 7:
            return "Consider regular practice to maintain proficiency"
        
        if self.proficiency_score < 0.3:
            return "Focus on fundamental concepts and regular practice"
        elif self.proficiency_score < 0.6:
            return "Increase practice complexity and duration"
        elif self.proficiency_score < 0.8:
            return "Apply skills in real-world scenarios"
        else:
            return "Mentor others and tackle advanced challenges"
    
    def _get_next_milestone(self) -> str:
        """Get next learning milestone"""
        current_score = self.proficiency_score
        
        milestones = [
            (0.2, "Beginner level"),
            (0.4, "Intermediate level"),
            (0.6, "Advanced level"),
            (0.8, "Expert level"),
            (0.95, "Master level")
        ]
        
        for threshold, description in milestones:
            if current_score < threshold:
                progress = (current_score / threshold) * 100
                return f"{description} ({progress:.1f}% progress)"
        
        return "Master level achieved!"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "complexity": self.complexity,
            "prerequisites": self.prerequisites,
            "current_level": self.current_level.value,
            "proficiency_score": self.proficiency_score,
            "brain_modules_involved": self.brain_modules_involved,
            "consciousness_integration": self.consciousness_integration,
            "created_date": self.created_date.isoformat(),
            "last_practiced": self.last_practiced.isoformat() if self.last_practiced else None,
            "total_practice_time": self.total_practice_time,
            "learning_sessions": self.learning_sessions,
            "mastery_milestones": self.mastery_milestones
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create skill from dictionary"""
        skill = cls(
            name=data["name"],
            description=data["description"],
            category=SkillCategory(data["category"]),
            complexity=data["complexity"],
            prerequisites=data["prerequisites"]
        )
        
        # Restore saved data
        skill.id = data["id"]
        skill.current_level = SkillLevel(data["current_level"])
        skill.proficiency_score = data["proficiency_score"]
        skill.consciousness_integration = data.get("consciousness_integration", 0.0)
        skill.created_date = datetime.fromisoformat(data["created_date"])
        skill.last_practiced = datetime.fromisoformat(data["last_practiced"]) if data["last_practiced"] else None
        skill.total_practice_time = data["total_practice_time"]
        skill.learning_sessions = data["learning_sessions"]
        skill.mastery_milestones = data["mastery_milestones"]
        
        return skill

class SkillsDatabase:
    """Enhanced skills database with learning progression tracking"""
    
    def __init__(self, database_path: str = "data/skills_database.json", brain_orchestrator=None):
        self.database_path = Path(database_path)
        self.brain_orchestrator = brain_orchestrator
        self.skills = {}  # skill_id -> Skill
        self.skill_categories = {}  # category -> List[skill_ids]
        
        # Learning analytics
        self.learning_analytics = {
            "total_practice_time": 0,
            "skills_mastered": 0,
            "learning_streaks": [],
            "skill_connections": {}
        }
        
        # Load existing database
        self._load_database()
    
    def _load_database(self):
        """Load skills database from JSON file"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                
                # Load skills
                for skill_data in data.get("skills", []):
                    skill = Skill.from_dict(skill_data)
                    self.skills[skill.id] = skill
                    
                    # Update category mapping
                    category = skill.category
                    if category not in self.skill_categories:
                        self.skill_categories[category] = []
                    self.skill_categories[category].append(skill.id)
                
                # Load analytics
                self.learning_analytics = data.get("learning_analytics", self.learning_analytics)
                
                logger.info(f"Loaded {len(self.skills)} skills from database")
                
            except Exception as e:
                logger.error(f"Failed to load skills database: {e}")
    
    def _save_database(self):
        """Save skills database to JSON file"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        skills_data = [skill.to_dict() for skill in self.skills.values()]
        
        data = {
            "skills": skills_data,
            "learning_analytics": self.learning_analytics,
            "last_updated": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        try:
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Skills database saved successfully")
        except Exception as e:
            logger.error(f"Failed to save skills database: {e}")
    
    def add_skill(self, skill: Skill) -> bool:
        """Add skill to database"""
        if skill.id not in self.skills:
            self.skills[skill.id] = skill
            
            # Update category mapping
            category = skill.category
            if category not in self.skill_categories:
                self.skill_categories[category] = []
            self.skill_categories[category].append(skill.id)
            
            # Brain integration
            if self.brain_orchestrator:
                signal = {
                    "type": "skill_added",
                    "skill_id": skill.id,
                    "skill_name": skill.name,
                    "category": skill.category.value,
                    "timestamp": datetime.now().isoformat()
                }
                self.brain_orchestrator.send_signal("hippocampus", signal)
            
            self._save_database()
            logger.info(f"Added skill: {skill.name}")
            return True
        
        return False
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID"""
        return self.skills.get(skill_id)
    
    def get_skills_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get all skills in a category"""
        skill_ids = self.skill_categories.get(category, [])
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]
    
    def practice_skill(self, skill_id: str, duration_minutes: int, 
                      quality_score: float = 0.8, notes: str = "") -> bool:
        """Record skill practice session"""
        skill = self.get_skill(skill_id)
        if skill:
            result = skill.practice(duration_minutes, quality_score, notes, self.brain_orchestrator)
            if result:
                # Update analytics
                self.learning_analytics["total_practice_time"] += duration_minutes
                self._update_learning_analytics()
                self._save_database()
            return result
        return False
    
    def _update_learning_analytics(self):
        """Update learning analytics"""
        # Count mastered skills
        mastered = sum(1 for skill in self.skills.values() 
                      if skill.current_level in [SkillLevel.EXPERT, SkillLevel.MASTER])
        self.learning_analytics["skills_mastered"] = mastered
        
        # Calculate learning streaks
        # (Implementation for streak calculation would go here)
    
    def get_learning_recommendations(self) -> List[Dict[str, Any]]:
        """Get personalized learning recommendations"""
        recommendations = []
        
        # Skills that need practice
        for skill in self.skills.values():
            if skill.last_practiced and (datetime.now() - skill.last_practiced).days > 7:
                recommendations.append({
                    "type": "maintenance",
                    "skill": skill.name,
                    "reason": "Haven't practiced recently",
                    "priority": "medium"
                })
        
        # Skills ready for advancement
        for skill in self.skills.values():
            if skill.proficiency_score > 0.7 and skill.current_level != SkillLevel.MASTER:
                recommendations.append({
                    "type": "advancement",
                    "skill": skill.name,
                    "reason": "Ready for next level",
                    "priority": "high"
                })
        
        # New skills to learn (based on prerequisites)
        mastered_skills = {skill.name for skill in self.skills.values() 
                          if skill.proficiency_score > 0.6}
        
        return recommendations[:10]  # Top 10 recommendations
    
    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get comprehensive skill statistics"""
        if not self.skills:
            return {"total_skills": 0}
        
        # Category distribution
        category_stats = {}
        for category in SkillCategory:
            skills_in_category = self.get_skills_by_category(category)
            if skills_in_category:
                avg_proficiency = sum(s.proficiency_score for s in skills_in_category) / len(skills_in_category)
                category_stats[category.value] = {
                    "count": len(skills_in_category),
                    "average_proficiency": avg_proficiency,
                    "mastered": sum(1 for s in skills_in_category if s.proficiency_score > 0.8)
                }
        
        # Overall statistics
        all_skills = list(self.skills.values())
        avg_proficiency = sum(s.proficiency_score for s in all_skills) / len(all_skills)
        
        return {
            "total_skills": len(self.skills),
            "average_proficiency": avg_proficiency,
            "skills_mastered": sum(1 for s in all_skills if s.proficiency_score > 0.8),
            "total_practice_time": self.learning_analytics["total_practice_time"],
            "category_breakdown": category_stats,
            "most_practiced_skill": max(all_skills, key=lambda s: s.total_practice_time).name if all_skills else None,
            "highest_proficiency_skill": max(all_skills, key=lambda s: s.proficiency_score).name if all_skills else None
        }

def create_default_skills() -> List[Skill]:
    """Create default skills for the neuromorphic system"""
    default_skills = [
        # Analytical Skills (Frontal Cortex)
        Skill("Pattern Recognition", "Identifying patterns in data and information", 
              SkillCategory.ANALYTICAL, complexity=0.6),
        Skill("Logical Reasoning", "Apply logical principles to solve problems", 
              SkillCategory.LOGICAL, complexity=0.7),
        Skill("Critical Thinking", "Evaluate information objectively and make reasoned judgments", 
              SkillCategory.ANALYTICAL, complexity=0.8),
        
        # Creative Skills
        Skill("Creative Problem Solving", "Generate innovative solutions to complex problems", 
              SkillCategory.CREATIVE, complexity=0.75),
        Skill("Artistic Expression", "Create and appreciate artistic works", 
              SkillCategory.CREATIVE, complexity=0.6),
        
        # Memory Skills (Hippocampus)
        Skill("Information Retention", "Store and recall information effectively", 
              SkillCategory.MEMORY, complexity=0.5),
        Skill("Associative Memory", "Create connections between different pieces of information", 
              SkillCategory.MEMORY, complexity=0.7),
        
        # Linguistic Skills (PFLT + Temporal)
        Skill("Language Processing", "Understand and generate natural language", 
              SkillCategory.LINGUISTIC, complexity=0.8),
        Skill("Translation", "Convert between different languages", 
              SkillCategory.LINGUISTIC, complexity=0.85),
        
        # Spatial Skills (Parietal Lobe)
        Skill("Spatial Reasoning", "Understand and manipulate spatial relationships", 
              SkillCategory.SPATIAL, complexity=0.7),
        Skill("Mathematical Computation", "Perform mathematical calculations and analysis", 
              SkillCategory.SPATIAL, complexity=0.75),
        
        # Social/Emotional Skills (Amygdala)
        Skill("Emotional Intelligence", "Understand and manage emotions effectively", 
              SkillCategory.EMOTIONAL, complexity=0.8),
        Skill("Social Interaction", "Communicate and collaborate effectively with others", 
              SkillCategory.SOCIAL, complexity=0.7)
    ]
    
    return default_skills

if __name__ == "__main__":
    # Test skills database
    db = SkillsDatabase()
    
    # Add default skills if database is empty
    if not db.skills:
        for skill in create_default_skills():
            db.add_skill(skill)
    
    print("Skills Database Status:")
    stats = db.get_skill_statistics()
    print(f"Total Skills: {stats['total_skills']}")
    print(f"Average Proficiency: {stats['average_proficiency']:.2f}")
    print(f"Skills Mastered: {stats['skills_mastered']}")
    
    print("\nSkill Categories:")
    for category, info in stats.get('category_breakdown', {}).items():
        print(f"  {category}: {info['count']} skills, {info['average_proficiency']:.2f} avg proficiency")
