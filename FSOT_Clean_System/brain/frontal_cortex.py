"""
Frontal Cortex - Executive Functions and Decision Making
Clean implementation of frontal lobe functionality
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority, Domain

class FrontalCortex(BrainModule):
    """
    Frontal Cortex Brain Module
    
    Functions:
    - Executive control and decision making
    - Planning and goal management
    - Working memory coordination
    - Impulse control and inhibition
    """
    
    def __init__(self):
        super().__init__(
            name="frontal_cortex",
            anatomical_region="cerebrum",
            functions=["decision_making", "planning", "executive_control", "working_memory"]
        )
        
        # Frontal cortex specific state
        self.working_memory: Dict[str, Any] = {}
        self.current_goals: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.inhibition_signals: Dict[str, float] = {}
        
        # FSOT domain for cognitive processing
        self.fsot_domain = Domain.COGNITIVE
        
        # Configuration
        self.max_working_memory = 10
        self.max_goals = 5
        self.decision_confidence_threshold = 0.7
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process signals through frontal cortex"""
        
        if signal.signal_type == SignalType.COGNITIVE:
            return await self._process_cognitive_signal(signal)
        elif signal.signal_type == SignalType.EXECUTIVE:
            return await self._process_executive_signal(signal)
        elif signal.signal_type == SignalType.MEMORY:
            return await self._process_memory_signal(signal)
        else:
            # Default processing - executive oversight
            return await self._provide_executive_oversight(signal)
    
    async def _process_cognitive_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process cognitive requests (decisions, planning)"""
        data = signal.data
        
        if 'decision_request' in data:
            return await self._make_decision(data['decision_request'], signal)
        elif 'planning_request' in data:
            return await self._create_plan(data['planning_request'], signal)
        elif 'goal_update' in data:
            return await self._update_goals(data['goal_update'], signal)
        
        return None
    
    async def _process_executive_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process executive control signals"""
        data = signal.data
        
        if 'inhibit' in data:
            return await self._apply_inhibition(data['inhibit'], signal)
        elif 'override' in data:
            return await self._executive_override(data['override'], signal)
        elif 'status_request' in data:
            return await self._provide_status(signal)
        
        return None
    
    async def _process_memory_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process working memory operations"""
        data = signal.data
        
        if 'store' in data:
            return await self._store_in_working_memory(data['store'], signal)
        elif 'retrieve' in data:
            return await self._retrieve_from_working_memory(data['retrieve'], signal)
        elif 'clear' in data:
            return await self._clear_working_memory(data['clear'], signal)
        
        return None
    
    async def _make_decision(self, decision_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Make executive decision"""
        options = decision_data.get('options', [])
        context = decision_data.get('context', {})
        urgency = decision_data.get('urgency', 'normal')
        
        # Calculate FSOT scalar for decision context
        fsot_params = {
            'observed': True,
            'delta_psi': self.activation_level,
            'recent_hits': 1 if urgency == 'urgent' else 0
        }
        decision_scalar = self.get_fsot_scalar(**fsot_params)
        
        # Simple decision algorithm
        if not options:
            selected_option = "no_action"
            confidence = 0.0
        elif len(options) == 1:
            selected_option = options[0]
            confidence = 0.9
        else:
            # Use FSOT scalar to influence decision
            option_index = abs(int(decision_scalar * 100)) % len(options)
            selected_option = options[option_index]
            confidence = min(0.95, abs(decision_scalar) + 0.5)
        
        # Create decision record
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'options': options,
            'selected': selected_option,
            'confidence': confidence,
            'context': context,
            'fsot_scalar': decision_scalar,
            'reasoning': self._generate_reasoning(options, selected_option, confidence)
        }
        
        # Store in decision history
        self.decision_history.append(decision_record)
        if len(self.decision_history) > 100:  # Keep recent decisions
            self.decision_history = self.decision_history[-100:]
        
        # Create response signal
        response_data = {
            'decision': selected_option,
            'confidence': confidence,
            'reasoning': decision_record['reasoning'],
            'executive_approval': confidence >= self.decision_confidence_threshold
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.EXECUTIVE,
            data=response_data,
            priority=Priority.HIGH if urgency == 'urgent' else Priority.NORMAL
        )
    
    async def _create_plan(self, planning_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Create execution plan"""
        goal = planning_data.get('goal', '')
        constraints = planning_data.get('constraints', [])
        resources = planning_data.get('resources', [])
        
        # Simple planning algorithm
        plan_steps = [
            f"Analyze goal: {goal}",
            "Assess available resources",
            "Identify potential obstacles",
            "Create step-by-step approach",
            "Execute with monitoring"
        ]
        
        # Consider constraints
        if constraints:
            plan_steps.insert(2, f"Account for constraints: {', '.join(constraints)}")
        
        plan = {
            'goal': goal,
            'steps': plan_steps,
            'estimated_duration': len(plan_steps) * 5,  # 5 minutes per step
            'resources_needed': resources,
            'confidence': 0.8,
            'created_by': self.name
        }
        
        # Add to current goals if not already present
        if not any(g['goal'] == goal for g in self.current_goals):
            self.current_goals.append(plan)
            if len(self.current_goals) > self.max_goals:
                self.current_goals = self.current_goals[-self.max_goals:]
        
        response_data = {
            'plan': plan,
            'status': 'plan_created',
            'executive_approval': True
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.COGNITIVE,
            data=response_data
        )
    
    async def _store_in_working_memory(self, store_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Store information in working memory"""
        key = store_data.get('key', f"item_{len(self.working_memory)}")
        value = store_data.get('value')
        duration = store_data.get('duration', 300)  # 5 minutes default
        
        # Store with timestamp
        self.working_memory[key] = {
            'value': value,
            'stored_at': datetime.now(),
            'duration': duration,
            'access_count': 0
        }
        
        # Cleanup old items if memory is full
        if len(self.working_memory) > self.max_working_memory:
            # Remove oldest item
            oldest_key = min(self.working_memory.keys(), 
                           key=lambda k: self.working_memory[k]['stored_at'])
            del self.working_memory[oldest_key]
        
        response_data = {
            'status': 'stored',
            'key': key,
            'working_memory_size': len(self.working_memory)
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.MEMORY,
            data=response_data
        )
    
    def _generate_reasoning(self, options: List[str], selected: str, confidence: float) -> List[str]:
        """Generate reasoning steps for decision"""
        reasoning = [
            f"Evaluated {len(options)} available options",
            f"Selected '{selected}' based on executive analysis",
            f"Decision confidence: {confidence:.1%}"
        ]
        
        if confidence < self.decision_confidence_threshold:
            reasoning.append("⚠️ Low confidence - consider additional information")
        
        return reasoning
    
    async def _update_goals(self, goal_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Update current goals"""
        action = goal_data.get('action', 'add')
        goal_info = goal_data.get('goal', {})
        
        if action == 'add':
            self.current_goals.append(goal_info)
            if len(self.current_goals) > self.max_goals:
                self.current_goals = self.current_goals[-self.max_goals:]
            status = 'goal_added'
        elif action == 'remove':
            goal_id = goal_info.get('id') or goal_info.get('goal')
            self.current_goals = [g for g in self.current_goals 
                                if g.get('id') != goal_id and g.get('goal') != goal_id]
            status = 'goal_removed'
        elif action == 'update':
            goal_id = goal_info.get('id') or goal_info.get('goal')
            for i, g in enumerate(self.current_goals):
                if g.get('id') == goal_id or g.get('goal') == goal_id:
                    self.current_goals[i].update(goal_info)
                    break
            status = 'goal_updated'
        else:
            status = 'unknown_action'
        
        response_data = {
            'status': status,
            'active_goals': len(self.current_goals),
            'goals': self.current_goals
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.COGNITIVE,
            data=response_data
        )
    
    async def _apply_inhibition(self, inhibition_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Apply inhibitory control"""
        target = inhibition_data.get('target', 'unknown')
        strength = inhibition_data.get('strength', 0.5)
        duration = inhibition_data.get('duration', 60)  # seconds
        
        # Store inhibition signal
        self.inhibition_signals[target] = {
            'strength': strength,
            'applied_at': datetime.now(),
            'duration': duration
        }
        
        response_data = {
            'status': 'inhibition_applied',
            'target': target,
            'strength': strength,
            'message': f"Inhibitory control applied to {target} with strength {strength}"
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.EXECUTIVE,
            data=response_data,
            priority=Priority.HIGH
        )
    
    async def _executive_override(self, override_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Apply executive override"""
        target = override_data.get('target', 'unknown')
        action = override_data.get('action', 'stop')
        reason = override_data.get('reason', 'Executive intervention required')
        
        override_record = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'action': action,
            'reason': reason,
            'authority': self.name
        }
        
        # Could store override history if needed
        
        response_data = {
            'status': 'override_applied',
            'target': target,
            'action': action,
            'reason': reason,
            'authority': 'frontal_cortex',
            'message': f"Executive override: {action} applied to {target}"
        }
        
        return NeuralSignal(
            source=self.name,
            target=target if target != 'unknown' else original_signal.source,
            signal_type=SignalType.EXECUTIVE,
            data=response_data,
            priority=Priority.URGENT
        )
    
    async def _provide_status(self, original_signal: NeuralSignal) -> NeuralSignal:
        """Provide executive status report"""
        status_data = {
            'module': self.name,
            'activation_level': self.activation_level,
            'working_memory_usage': f"{len(self.working_memory)}/{self.max_working_memory}",
            'active_goals': len(self.current_goals),
            'recent_decisions': len([d for d in self.decision_history 
                                  if (datetime.now() - datetime.fromisoformat(d['timestamp'])).seconds < 3600]),
            'inhibition_targets': list(self.inhibition_signals.keys()),
            'executive_summary': self.get_executive_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.EXECUTIVE,
            data=status_data
        )
    
    async def _retrieve_from_working_memory(self, retrieve_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Retrieve information from working memory"""
        key = retrieve_data.get('key')
        pattern = retrieve_data.get('pattern')  # For pattern matching
        
        if key and key in self.working_memory:
            # Direct key retrieval
            item = self.working_memory[key]
            item['access_count'] += 1
            
            response_data = {
                'status': 'found',
                'key': key,
                'value': item['value'],
                'stored_at': item['stored_at'].isoformat(),
                'access_count': item['access_count']
            }
        elif pattern:
            # Pattern-based search
            matching_items = {}
            for k, v in self.working_memory.items():
                if pattern.lower() in str(v['value']).lower() or pattern.lower() in k.lower():
                    v['access_count'] += 1
                    matching_items[k] = v['value']
            
            response_data = {
                'status': 'found' if matching_items else 'not_found',
                'pattern': pattern,
                'matches': matching_items,
                'count': len(matching_items)
            }
        else:
            # Return all items if no specific request
            all_items = {k: v['value'] for k, v in self.working_memory.items()}
            response_data = {
                'status': 'all_items',
                'items': all_items,
                'count': len(all_items)
            }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.MEMORY,
            data=response_data
        )
    
    async def _clear_working_memory(self, clear_data: Dict[str, Any], original_signal: NeuralSignal) -> NeuralSignal:
        """Clear working memory"""
        scope = clear_data.get('scope', 'all')
        key = clear_data.get('key')
        
        if scope == 'all':
            cleared_count = len(self.working_memory)
            self.working_memory.clear()
            message = f"Cleared all {cleared_count} items from working memory"
        elif scope == 'expired':
            # Clear expired items
            now = datetime.now()
            expired_keys = []
            for k, v in self.working_memory.items():
                if (now - v['stored_at']).seconds > v['duration']:
                    expired_keys.append(k)
            
            for k in expired_keys:
                del self.working_memory[k]
            
            message = f"Cleared {len(expired_keys)} expired items from working memory"
        elif key and key in self.working_memory:
            del self.working_memory[key]
            message = f"Cleared item '{key}' from working memory"
        else:
            message = "No items cleared - invalid scope or key"
        
        response_data = {
            'status': 'cleared',
            'scope': scope,
            'message': message,
            'remaining_items': len(self.working_memory)
        }
        
        return NeuralSignal(
            source=self.name,
            target=original_signal.source,
            signal_type=SignalType.MEMORY,
            data=response_data
        )
    
    async def _provide_executive_oversight(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Provide executive oversight for any signal"""
        # Simple oversight - just acknowledge and potentially inhibit
        oversight_data = {
            'oversight_provided': True,
            'executive_assessment': 'approved',
            'oversight_notes': f"Signal from {signal.source} reviewed and approved"
        }
        
        # Don't send response unless specifically requested
        if signal.response_expected:
            return NeuralSignal(
                source=self.name,
                target=signal.source,
                signal_type=SignalType.EXECUTIVE,
                data=oversight_data
            )
        
        return None
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary of frontal cortex state"""
        return {
            'working_memory_items': len(self.working_memory),
            'active_goals': len(self.current_goals),
            'recent_decisions': len([d for d in self.decision_history 
                                  if (datetime.now() - datetime.fromisoformat(d['timestamp'])).seconds < 3600]),
            'average_decision_confidence': sum(d['confidence'] for d in self.decision_history[-10:]) / max(len(self.decision_history[-10:]), 1),
            'current_goals': [g['goal'] for g in self.current_goals],
            'executive_state': 'active' if self.activation_level > 0.5 else 'monitoring'
        }
