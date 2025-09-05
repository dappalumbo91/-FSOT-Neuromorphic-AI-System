"""
FSOT 2.0 Cerebellum Brain Module
Motor Control and Coordination
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority

logger = logging.getLogger(__name__)

class MotorState(Enum):
    """Motor control states"""
    IDLE = "idle"
    COORDINATING = "coordinating"
    LEARNING = "learning"
    EXECUTING = "executing"
    BALANCING = "balancing"

class SkillLevel(Enum):
    """Skill proficiency levels"""
    NOVICE = 1
    DEVELOPING = 2
    PROFICIENT = 3
    ADVANCED = 4
    EXPERT = 5

@dataclass
class MotorSkill:
    """Represents a learned motor skill"""
    name: str
    level: SkillLevel
    practice_count: int
    success_rate: float
    last_practiced: datetime
    complexity: float  # 0.0 to 1.0

@dataclass
class MotorCommand:
    """Motor command execution data"""
    action: str
    parameters: Dict[str, Any]
    precision_required: float  # 0.0 to 1.0
    timing_critical: bool
    timestamp: datetime

class Cerebellum(BrainModule):
    """
    Cerebellum Brain Module - Motor Control and Coordination
    
    Responsibilities:
    - Motor control and coordination
    - Skill learning and refinement
    - Balance and precision
    - Movement timing and sequencing
    - Error correction and adaptation
    """
    
    def __init__(self):
        super().__init__(
            name="cerebellum",
            anatomical_region="hindbrain",
            functions=[
                "motor_control",
                "skill_learning",
                "balance_coordination",
                "precision_timing",
                "error_correction",
                "movement_planning"
            ]
        )
        
        # Motor control state
        self.current_state = MotorState.IDLE
        self.motor_skills: Dict[str, MotorSkill] = {}
        self.active_commands: List[MotorCommand] = []
        
        # Performance metrics
        self.commands_executed = 0
        self.skills_learned = 0
        self.coordination_accuracy = 0.85
        self.response_time = 0.05  # seconds
        
        # Learning parameters
        self.learning_rate = 0.1
        self.skill_decay_rate = 0.01
        self.practice_threshold = 10
        
        # Initialize basic motor skills
        self._initialize_basic_skills()
    
    def _initialize_basic_skills(self):
        """Initialize basic motor skills"""
        basic_skills = [
            ("typing", SkillLevel.PROFICIENT, 0.92),
            ("navigation", SkillLevel.ADVANCED, 0.88),
            ("interaction", SkillLevel.PROFICIENT, 0.85),
            ("coordination", SkillLevel.DEVELOPING, 0.75)
        ]
        
        for skill_name, level, success_rate in basic_skills:
            self.motor_skills[skill_name] = MotorSkill(
                name=skill_name,
                level=level,
                practice_count=100,
                success_rate=success_rate,
                last_practiced=datetime.now(),
                complexity=0.5
            )
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signals"""
        try:
            if signal.signal_type == SignalType.MOTOR:
                return await self._execute_motor_command(signal)
            elif signal.signal_type == SignalType.SKILL_LEARNING:
                return await self._learn_skill(signal)
            elif signal.signal_type == SignalType.COORDINATION_REQUEST:
                return await self._coordinate_movement(signal)
            elif signal.signal_type == SignalType.BALANCE_CHECK:
                return await self._check_balance(signal)
            else:
                # Analyze all signals for motor learning opportunities
                return await self._analyze_for_motor_patterns(signal)
                
        except Exception as e:
            logger.error(f"Error processing signal in cerebellum: {e}")
            return None
    
    async def _execute_motor_command(self, signal: NeuralSignal) -> NeuralSignal:
        """Execute motor command with coordination"""
        command_data = signal.data.get('command', {})
        action = command_data.get('action', 'unknown')
        parameters = command_data.get('parameters', {})
        precision = command_data.get('precision_required', 0.5)
        timing_critical = command_data.get('timing_critical', False)
        
        # Create motor command
        motor_command = MotorCommand(
            action=action,
            parameters=parameters,
            precision_required=precision,
            timing_critical=timing_critical,
            timestamp=datetime.now()
        )
        
        # Execute with cerebellum coordination
        execution_result = await self._coordinate_execution(motor_command)
        
        # Update learning
        skill_name = self._get_skill_category(action)
        if skill_name in self.motor_skills:
            await self._update_skill_from_execution(skill_name, execution_result)
        
        self.commands_executed += 1
        self.current_state = MotorState.EXECUTING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.MOTOR_RESULT,
            data={
                'execution_result': execution_result,
                'coordination_accuracy': self.coordination_accuracy,
                'response_time': self.response_time,
                'motor_state': self.current_state.value
            },
            priority=Priority.HIGH if timing_critical else Priority.NORMAL
        )
    
    async def _coordinate_execution(self, command: MotorCommand) -> Dict[str, Any]:
        """Coordinate motor command execution"""
        start_time = datetime.now()
        
        # Simulate cerebellum processing
        processing_time = self.response_time * (1 + command.precision_required)
        await asyncio.sleep(processing_time / 100)  # Scale down for simulation
        
        # Calculate success based on skill level and precision requirement
        skill_name = self._get_skill_category(command.action)
        skill = self.motor_skills.get(skill_name)
        
        if skill:
            base_success = skill.success_rate
            precision_modifier = 1.0 - (command.precision_required * 0.3)
            success_probability = base_success * precision_modifier
        else:
            success_probability = 0.7  # Default for unknown skills
        
        success = success_probability > 0.75
        actual_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': success,
            'accuracy': success_probability,
            'execution_time': actual_time,
            'precision_achieved': command.precision_required * success_probability,
            'coordination_quality': self.coordination_accuracy
        }
    
    async def _learn_skill(self, signal: NeuralSignal) -> NeuralSignal:
        """Learn or improve a motor skill"""
        skill_data = signal.data.get('skill', {})
        skill_name = skill_data.get('name', 'unknown')
        complexity = skill_data.get('complexity', 0.5)
        practice_data = skill_data.get('practice_data', {})
        
        # Get or create skill
        if skill_name not in self.motor_skills:
            self.motor_skills[skill_name] = MotorSkill(
                name=skill_name,
                level=SkillLevel.NOVICE,
                practice_count=0,
                success_rate=0.5,
                last_practiced=datetime.now(),
                complexity=complexity
            )
            self.skills_learned += 1
        
        skill = self.motor_skills[skill_name]
        
        # Update skill based on practice
        learning_progress = await self._process_skill_learning(skill, practice_data)
        
        self.current_state = MotorState.LEARNING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.SKILL_LEARNING_RESULT,
            data={
                'skill_name': skill_name,
                'current_level': skill.level.name,
                'success_rate': skill.success_rate,
                'practice_count': skill.practice_count,
                'learning_progress': learning_progress
            },
            priority=Priority.NORMAL
        )
    
    async def _process_skill_learning(self, skill: MotorSkill, practice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process skill learning and improvement"""
        success = practice_data.get('success', True)
        difficulty = practice_data.get('difficulty', 0.5)
        
        # Update practice count
        skill.practice_count += 1
        skill.last_practiced = datetime.now()
        
        # Update success rate with learning
        if success:
            improvement = self.learning_rate * (1.0 - skill.success_rate)
            skill.success_rate = min(1.0, skill.success_rate + improvement)
        else:
            decline = self.learning_rate * 0.1
            skill.success_rate = max(0.1, skill.success_rate - decline)
        
        # Update skill level based on practice and success
        old_level = skill.level
        if skill.practice_count >= self.practice_threshold and skill.success_rate >= 0.8:
            if skill.level.value < SkillLevel.EXPERT.value:
                skill.level = SkillLevel(skill.level.value + 1)
        
        return {
            'improvement': skill.success_rate,
            'level_changed': old_level != skill.level,
            'new_level': skill.level.name,
            'practices_needed': max(0, self.practice_threshold - skill.practice_count)
        }
    
    async def _coordinate_movement(self, signal: NeuralSignal) -> NeuralSignal:
        """Coordinate complex movement sequences"""
        movement_data = signal.data.get('movement', {})
        sequence = movement_data.get('sequence', [])
        timing_requirements = movement_data.get('timing', {})
        
        # Analyze movement sequence for coordination
        coordination_plan = await self._plan_movement_coordination(sequence, timing_requirements)
        
        self.current_state = MotorState.COORDINATING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.COORDINATION_RESULT,
            data={
                'coordination_plan': coordination_plan,
                'estimated_execution_time': coordination_plan.get('total_time', 0),
                'coordination_accuracy': self.coordination_accuracy,
                'sequence_complexity': len(sequence)
            },
            priority=Priority.HIGH
        )
    
    async def _plan_movement_coordination(self, sequence: List[Dict], timing: Dict[str, Any]) -> Dict[str, Any]:
        """Plan coordination for movement sequence"""
        total_time = 0
        coordination_steps = []
        
        for i, movement in enumerate(sequence):
            action = movement.get('action', 'unknown')
            skill_name = self._get_skill_category(action)
            skill = self.motor_skills.get(skill_name)
            
            step_time = self.response_time
            if skill:
                # Adjust time based on skill level
                skill_modifier = 1.0 - (skill.level.value * 0.1)
                step_time *= skill_modifier
            
            coordination_steps.append({
                'step': i,
                'action': action,
                'estimated_time': step_time,
                'coordination_level': skill.success_rate if skill else 0.7
            })
            
            total_time += step_time
        
        return {
            'steps': coordination_steps,
            'total_time': total_time,
            'overall_coordination': self.coordination_accuracy,
            'timing_precision': timing.get('precision', 0.8)
        }
    
    async def _check_balance(self, signal: NeuralSignal) -> NeuralSignal:
        """Check and maintain balance during operations"""
        balance_data = signal.data.get('balance', {})
        current_state = balance_data.get('current_state', 'stable')
        disturbance = balance_data.get('disturbance_level', 0.0)
        
        # Calculate balance correction
        balance_correction = await self._calculate_balance_correction(current_state, disturbance)
        
        self.current_state = MotorState.BALANCING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.BALANCE_RESULT,
            data={
                'balance_status': 'stable' if disturbance < 0.3 else 'adjusting',
                'correction_needed': balance_correction,
                'stability_confidence': max(0.0, 1.0 - disturbance),
                'response_time': self.response_time
            },
            priority=Priority.URGENT if disturbance > 0.7 else Priority.HIGH
        )
    
    async def _calculate_balance_correction(self, state: str, disturbance: float) -> Dict[str, Any]:
        """Calculate balance correction parameters"""
        if disturbance < 0.1:
            return {'type': 'none', 'magnitude': 0.0}
        elif disturbance < 0.5:
            return {'type': 'minor_adjustment', 'magnitude': disturbance * 0.5}
        else:
            return {'type': 'major_correction', 'magnitude': min(1.0, disturbance * 0.8)}
    
    async def _analyze_for_motor_patterns(self, signal: NeuralSignal) -> NeuralSignal:
        """Analyze signals for motor learning opportunities"""
        # Extract potential motor patterns
        content = str(signal.data.get('content', ''))
        motor_keywords = ['move', 'control', 'coordinate', 'execute', 'perform', 'action']
        
        has_motor_content = any(keyword in content.lower() for keyword in motor_keywords)
        
        if has_motor_content:
            self.current_state = MotorState.COORDINATING
            
        # Return original signal (pass-through with state update)
        return signal
    
    def _get_skill_category(self, action: str) -> str:
        """Categorize action into skill type"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ['type', 'input', 'keyboard']):
            return 'typing'
        elif any(word in action_lower for word in ['move', 'navigate', 'go']):
            return 'navigation'
        elif any(word in action_lower for word in ['click', 'select', 'interact']):
            return 'interaction'
        else:
            return 'coordination'
    
    async def _update_skill_from_execution(self, skill_name: str, result: Dict[str, Any]):
        """Update skill based on execution result"""
        if skill_name in self.motor_skills:
            skill = self.motor_skills[skill_name]
            success = result.get('success', False)
            accuracy = result.get('accuracy', 0.5)
            
            # Update success rate
            if success:
                skill.success_rate = min(1.0, skill.success_rate + self.learning_rate * 0.1)
            else:
                skill.success_rate = max(0.1, skill.success_rate - self.learning_rate * 0.05)
            
            skill.practice_count += 1
            skill.last_practiced = datetime.now()
    
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Apply skill decay for unused skills
        current_time = datetime.now()
        for skill in self.motor_skills.values():
            time_since_practice = (current_time - skill.last_practiced).days
            if time_since_practice > 7:  # Week without practice
                decay = self.skill_decay_rate * time_since_practice
                skill.success_rate = max(0.1, skill.success_rate - decay)
        
        # Clean old commands
        self.active_commands = [
            cmd for cmd in self.active_commands
            if (current_time - cmd.timestamp).seconds < 300  # 5 minutes
        ]
        
        # Reset state if idle
        if self.current_state != MotorState.IDLE:
            self.current_state = MotorState.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """Get cerebellum status"""
        base_status = super().get_status()
        
        cerebellum_status = {
            'motor_state': self.current_state.value,
            'commands_executed': self.commands_executed,
            'skills_learned': self.skills_learned,
            'coordination_accuracy': self.coordination_accuracy,
            'response_time': self.response_time,
            'active_skills': len(self.motor_skills),
            'skill_summary': {
                name: {
                    'level': skill.level.name,
                    'success_rate': round(skill.success_rate, 3),
                    'practice_count': skill.practice_count
                }
                for name, skill in self.motor_skills.items()
            }
        }
        
        base_status.update(cerebellum_status)
        return base_status
