"""
FSOT 2.0 Thalamus Brain Module
Central Relay and Consciousness Coordination
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority

logger = logging.getLogger(__name__)

class RelayMode(Enum):
    """Thalamus relay modes"""
    IDLE = "idle"
    ROUTING = "routing"
    FILTERING = "filtering"
    INTEGRATING = "integrating"
    COORDINATING = "coordinating"

class ConsciousnessLevel(Enum):
    """Consciousness coordination levels"""
    UNCONSCIOUS = 0
    SUBLIMINAL = 1
    AWARE = 2
    FOCUSED = 3
    HYPERAWARE = 4

@dataclass
class SignalRoute:
    """Represents a signal routing path"""
    source_module: str
    target_module: str
    signal_type: SignalType
    priority_boost: float
    routing_rules: Dict[str, Any]
    success_rate: float
    timestamp: datetime

@dataclass
class ConsciousnessState:
    """Current consciousness state"""
    level: ConsciousnessLevel
    focus_target: Optional[str]
    attention_distribution: Dict[str, float]
    global_workspace: Dict[str, Any]
    timestamp: datetime

class Thalamus(BrainModule):
    """
    Thalamus Brain Module - Central Relay and Consciousness Coordination
    
    Responsibilities:
    - Central signal routing and relay
    - Consciousness coordination and global workspace
    - Attention management and focus control
    - Sensory integration and filtering
    - Inter-module communication optimization
    - Global brain state management
    """
    
    def __init__(self):
        super().__init__(
            name="thalamus",
            anatomical_region="diencephalon",
            functions=[
                "signal_routing",
                "consciousness_coordination",
                "attention_management",
                "sensory_integration",
                "global_workspace",
                "brain_state_management",
                "communication_optimization"
            ]
        )
        
        # Relay and coordination state
        self.current_mode = RelayMode.IDLE
        self.consciousness_state = ConsciousnessState(
            level=ConsciousnessLevel.AWARE,
            focus_target=None,
            attention_distribution={},
            global_workspace={},
            timestamp=datetime.now()
        )
        
        # Routing and relay capabilities
        self.routing_table: Dict[str, List[SignalRoute]] = defaultdict(list)
        self.active_routes: Set[Tuple[str, str]] = set()
        self.signal_history: deque = deque(maxlen=1000)
        
        # Brain module registry
        self.registered_modules: Dict[str, Dict[str, Any]] = {}
        self.module_health: Dict[str, float] = {}
        self.communication_matrix: Dict[Tuple[str, str], float] = {}
        
        # Performance metrics
        self.signals_routed = 0
        self.consciousness_updates = 0
        self.attention_shifts = 0
        self.integration_events = 0
        
        # Coordination parameters
        self.attention_threshold = 0.7
        self.consciousness_decay_rate = 0.05
        self.routing_optimization_interval = 100  # signals
        self.global_workspace_capacity = 20  # items
        
        # Initialize thalamus systems
        self._initialize_thalamus_systems()
    
    def _initialize_thalamus_systems(self):
        """Initialize thalamus coordination systems"""
        # Default routing rules
        self.default_routing_rules = {
            'priority_boost': {
                Priority.VITAL: 2.0,
                Priority.URGENT: 1.5,
                Priority.HIGH: 1.2,
                Priority.NORMAL: 1.0,
                Priority.LOW: 0.8
            },
            'consciousness_filters': {
                'safety_signals': {'priority_boost': 2.0, 'consciousness_required': True},
                'memory_signals': {'priority_boost': 1.3, 'consciousness_required': False},
                'motor_signals': {'priority_boost': 1.1, 'consciousness_required': False},
                'sensory_signals': {'priority_boost': 1.4, 'consciousness_required': True}
            },
            'attention_weights': {
                'frontal_cortex': 0.3,
                'amygdala': 0.25,
                'hippocampus': 0.2,
                'temporal_lobe': 0.15,
                'occipital_lobe': 0.1
            }
        }
        
        # Global workspace categories
        self.workspace_categories = {
            'current_focus': {},
            'active_goals': {},
            'environmental_state': {},
            'system_status': {},
            'recent_events': deque(maxlen=10)
        }
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signals"""
        try:
            # Log signal for routing analysis
            self._log_signal(signal)
            
            if signal.signal_type == SignalType.CONSCIOUSNESS_UPDATE:
                return await self._update_consciousness(signal)
            elif signal.signal_type == SignalType.ATTENTION_CONTROL:
                return await self._control_attention(signal)
            elif signal.signal_type == SignalType.BRAIN_STATE_QUERY:
                return await self._provide_brain_state(signal)
            elif signal.signal_type == SignalType.MODULE_REGISTRATION:
                return await self._register_module(signal)
            elif signal.signal_type == SignalType.ROUTING_REQUEST:
                return await self._route_signal(signal)
            else:
                # All signals pass through thalamus for coordination
                return await self._coordinate_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing signal in thalamus: {e}")
            return None
    
    def _log_signal(self, signal: NeuralSignal):
        """Log signal for analysis and routing optimization"""
        signal_record = {
            'timestamp': datetime.now(),
            'source': signal.source,
            'target': signal.target,
            'signal_type': signal.signal_type.value,
            'priority': signal.priority.value,
            'data_size': len(str(signal.data))
        }
        self.signal_history.append(signal_record)
        self.signals_routed += 1
    
    async def _update_consciousness(self, signal: NeuralSignal) -> NeuralSignal:
        """Update global consciousness state"""
        consciousness_data = signal.data.get('consciousness', {})
        update_type = consciousness_data.get('type', 'level_update')
        
        if update_type == 'level_update':
            result = await self._update_consciousness_level(consciousness_data)
        elif update_type == 'focus_update':
            result = await self._update_focus(consciousness_data)
        elif update_type == 'workspace_update':
            result = await self._update_global_workspace(consciousness_data)
        else:
            result = {'error': f'Unknown consciousness update type: {update_type}'}
        
        self.consciousness_updates += 1
        self.current_mode = RelayMode.COORDINATING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.CONSCIOUSNESS_UPDATE_RESULT,
            data={
                'consciousness_result': result,
                'current_level': self.consciousness_state.level.name,
                'focus_target': self.consciousness_state.focus_target,
                'global_workspace_size': len(self.consciousness_state.global_workspace)
            },
            priority=Priority.HIGH
        )
    
    async def _update_consciousness_level(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update consciousness level"""
        level_change = data.get('level_change', 0)
        trigger_event = data.get('trigger', 'unknown')
        
        # Calculate new consciousness level
        current_value = self.consciousness_state.level.value
        new_value = max(0, min(4, current_value + level_change))
        new_level = ConsciousnessLevel(new_value)
        
        old_level = self.consciousness_state.level
        self.consciousness_state.level = new_level
        self.consciousness_state.timestamp = datetime.now()
        
        # Update global workspace based on consciousness level
        await self._adjust_workspace_for_consciousness_level()
        
        return {
            'level_changed': old_level != new_level,
            'old_level': old_level.name,
            'new_level': new_level.name,
            'trigger_event': trigger_event,
            'consciousness_value': new_value
        }
    
    async def _update_focus(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update attention focus"""
        new_focus = data.get('focus_target')
        focus_strength = data.get('strength', 1.0)
        
        old_focus = self.consciousness_state.focus_target
        self.consciousness_state.focus_target = new_focus
        
        # Redistribute attention based on new focus
        if new_focus:
            await self._redistribute_attention(new_focus, focus_strength)
            self.attention_shifts += 1
        
        return {
            'focus_changed': old_focus != new_focus,
            'old_focus': old_focus,
            'new_focus': new_focus,
            'attention_distribution': dict(self.consciousness_state.attention_distribution)
        }
    
    async def _redistribute_attention(self, focus_target: str, strength: float):
        """Redistribute attention across brain modules"""
        # Reset attention distribution
        total_modules = len(self.default_routing_rules['attention_weights'])
        base_attention = (1.0 - strength) / max(1, total_modules - 1)
        
        self.consciousness_state.attention_distribution = {}
        
        for module, default_weight in self.default_routing_rules['attention_weights'].items():
            if module == focus_target:
                self.consciousness_state.attention_distribution[module] = strength
            else:
                self.consciousness_state.attention_distribution[module] = base_attention
    
    async def _update_global_workspace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update global workspace content"""
        workspace_updates = data.get('workspace_updates', {})
        
        for category, content in workspace_updates.items():
            if category in self.workspace_categories:
                if category == 'recent_events':
                    self.workspace_categories[category].append({
                        'event': content,
                        'timestamp': datetime.now()
                    })
                else:
                    self.workspace_categories[category].update(content)
        
        # Update consciousness state global workspace
        self.consciousness_state.global_workspace = self._compile_global_workspace()
        
        return {
            'workspace_updated': True,
            'categories_updated': list(workspace_updates.keys()),
            'workspace_size': len(self.consciousness_state.global_workspace)
        }
    
    def _compile_global_workspace(self) -> Dict[str, Any]:
        """Compile current global workspace"""
        workspace = {}
        
        # Compile from all categories
        for category, content in self.workspace_categories.items():
            if category == 'recent_events':
                workspace[category] = list(content)
            else:
                workspace[category] = dict(content)
        
        # Add system-level information
        workspace['system_info'] = {
            'consciousness_level': self.consciousness_state.level.name,
            'focus_target': self.consciousness_state.focus_target,
            'active_modules': len(self.registered_modules),
            'signal_load': len(self.signal_history)
        }
        
        return workspace
    
    async def _adjust_workspace_for_consciousness_level(self):
        """Adjust global workspace based on consciousness level"""
        level = self.consciousness_state.level
        
        if level == ConsciousnessLevel.UNCONSCIOUS:
            # Minimal workspace - only vital functions
            self.workspace_categories['current_focus'] = {}
        elif level == ConsciousnessLevel.SUBLIMINAL:
            # Basic awareness - simple processing
            pass  # Keep current workspace
        elif level == ConsciousnessLevel.AWARE:
            # Normal consciousness - full workspace
            pass  # Keep current workspace
        elif level == ConsciousnessLevel.FOCUSED:
            # Enhanced focus - prioritize current focus
            if self.consciousness_state.focus_target:
                focus_content = self.workspace_categories['current_focus'].get(
                    self.consciousness_state.focus_target, {}
                )
                # Enhance focus content
                focus_content['enhanced'] = True
        elif level == ConsciousnessLevel.HYPERAWARE:
            # Maximum awareness - expand workspace
            self.workspace_categories['environmental_state']['hyperaware_mode'] = True
    
    async def _control_attention(self, signal: NeuralSignal) -> NeuralSignal:
        """Control attention and focus"""
        attention_data = signal.data.get('attention', {})
        command = attention_data.get('command', 'focus')
        target = attention_data.get('target')
        parameters = attention_data.get('parameters', {})
        
        if command == 'focus':
            result = await self._focus_attention(target, parameters)
        elif command == 'distribute':
            result = await self._distribute_attention(parameters)
        elif command == 'shift':
            result = await self._shift_attention(target, parameters)
        else:
            result = {'error': f'Unknown attention command: {command}'}
        
        self.attention_shifts += 1
        self.current_mode = RelayMode.COORDINATING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.ATTENTION_CONTROL_RESULT,
            data={
                'attention_result': result,
                'current_focus': self.consciousness_state.focus_target,
                'attention_distribution': dict(self.consciousness_state.attention_distribution)
            },
            priority=Priority.HIGH
        )
    
    async def _focus_attention(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Focus attention on specific target"""
        intensity = parameters.get('intensity', 0.8)
        duration = parameters.get('duration', None)
        
        old_focus = self.consciousness_state.focus_target
        self.consciousness_state.focus_target = target
        
        await self._redistribute_attention(target, intensity)
        
        return {
            'focus_set': True,
            'target': target,
            'intensity': intensity,
            'previous_focus': old_focus
        }
    
    async def _distribute_attention(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute attention across multiple targets"""
        distribution = parameters.get('distribution', {})
        
        # Normalize distribution
        total_weight = sum(distribution.values())
        if total_weight > 0:
            normalized_distribution = {
                target: weight / total_weight
                for target, weight in distribution.items()
            }
            self.consciousness_state.attention_distribution = normalized_distribution
            self.consciousness_state.focus_target = None  # No single focus
        
        return {
            'distribution_set': True,
            'targets': list(distribution.keys()),
            'distribution': normalized_distribution
        }
    
    async def _shift_attention(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Shift attention from current focus to new target"""
        transition_speed = parameters.get('speed', 'normal')
        
        old_target = self.consciousness_state.focus_target
        
        # Gradual shift based on speed
        if transition_speed == 'instant':
            await self._focus_attention(target, parameters)
        else:
            # Implement gradual shift
            await self._focus_attention(target, parameters)
        
        return {
            'attention_shifted': True,
            'from_target': old_target,
            'to_target': target,
            'transition_speed': transition_speed
        }
    
    async def _provide_brain_state(self, signal: NeuralSignal) -> NeuralSignal:
        """Provide comprehensive brain state information"""
        query = signal.data.get('query', {})
        detail_level = query.get('detail_level', 'summary')
        
        if detail_level == 'summary':
            brain_state = await self._get_brain_state_summary()
        elif detail_level == 'detailed':
            brain_state = await self._get_detailed_brain_state()
        elif detail_level == 'full':
            brain_state = await self._get_full_brain_state()
        else:
            brain_state = {'error': f'Unknown detail level: {detail_level}'}
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.BRAIN_STATE_RESULT,
            data={
                'brain_state': brain_state,
                'detail_level': detail_level,
                'timestamp': datetime.now().isoformat()
            },
            priority=Priority.NORMAL
        )
    
    async def _get_brain_state_summary(self) -> Dict[str, Any]:
        """Get summary brain state"""
        return {
            'consciousness_level': self.consciousness_state.level.name,
            'focus_target': self.consciousness_state.focus_target,
            'active_modules': len(self.registered_modules),
            'signal_activity': len(self.signal_history),
            'attention_focused': self.consciousness_state.focus_target is not None
        }
    
    async def _get_detailed_brain_state(self) -> Dict[str, Any]:
        """Get detailed brain state"""
        summary = await self._get_brain_state_summary()
        
        detailed_state = {
            **summary,
            'attention_distribution': dict(self.consciousness_state.attention_distribution),
            'global_workspace': dict(self.consciousness_state.global_workspace),
            'module_health': dict(self.module_health),
            'recent_signal_types': self._analyze_recent_signals(),
            'routing_efficiency': self._calculate_routing_efficiency()
        }
        
        return detailed_state
    
    async def _get_full_brain_state(self) -> Dict[str, Any]:
        """Get full brain state with all details"""
        detailed_state = await self._get_detailed_brain_state()
        
        full_state = {
            **detailed_state,
            'registered_modules': dict(self.registered_modules),
            'routing_table': {k: len(v) for k, v in self.routing_table.items()},
            'signal_history_sample': list(self.signal_history)[-10:],  # Last 10 signals
            'performance_metrics': {
                'signals_routed': self.signals_routed,
                'consciousness_updates': self.consciousness_updates,
                'attention_shifts': self.attention_shifts,
                'integration_events': self.integration_events
            }
        }
        
        return full_state
    
    async def _register_module(self, signal: NeuralSignal) -> NeuralSignal:
        """Register a brain module with thalamus"""
        module_data = signal.data.get('module', {})
        module_name = module_data.get('name', signal.source)
        module_info = module_data.get('info', {})
        
        # Register module
        self.registered_modules[module_name] = {
            'info': module_info,
            'registered_at': datetime.now(),
            'last_seen': datetime.now(),
            'signal_count': 0
        }
        
        # Initialize module health
        self.module_health[module_name] = 1.0
        
        # Setup default routing for module
        await self._setup_module_routing(module_name, module_info)
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.MODULE_REGISTRATION_RESULT,
            data={
                'registration_result': {
                    'registered': True,
                    'module_name': module_name,
                    'total_modules': len(self.registered_modules)
                }
            },
            priority=Priority.NORMAL
        )
    
    async def _setup_module_routing(self, module_name: str, module_info: Dict[str, Any]):
        """Setup routing rules for a registered module"""
        module_functions = module_info.get('functions', [])
        
        # Create routing rules based on module functions
        for function in module_functions:
            route = SignalRoute(
                source_module='any',
                target_module=module_name,
                signal_type=self._map_function_to_signal_type(function),
                priority_boost=1.0,
                routing_rules={'function': function},
                success_rate=1.0,
                timestamp=datetime.now()
            )
            self.routing_table[module_name].append(route)
    
    def _map_function_to_signal_type(self, function: str) -> SignalType:
        """Map function name to signal type"""
        function_mapping = {
            'memory': SignalType.MEMORY,
            'safety': SignalType.SAFETY_CHECK,
            'motor': SignalType.MOTOR,
            'language': SignalType.LANGUAGE_COMPREHENSION,
            'visual': SignalType.VISUAL_PROCESSING,
            'decision': SignalType.EXECUTIVE
        }
        
        for key, signal_type in function_mapping.items():
            if key in function.lower():
                return signal_type
        
        return SignalType.COGNITIVE  # Default
    
    async def _route_signal(self, signal: NeuralSignal) -> NeuralSignal:
        """Route signal using thalamus routing table"""
        routing_request = signal.data.get('routing', {})
        target_module = routing_request.get('target_module')
        signal_to_route = routing_request.get('signal')
        
        if not target_module or not signal_to_route:
            return NeuralSignal(
                source=self.name,
                target=signal.source,
                signal_type=SignalType.ROUTING_RESULT,
                data={'error': 'Invalid routing request'},
                priority=Priority.LOW
            )
        
        # Find optimal route
        route_result = await self._find_optimal_route(signal_to_route, target_module)
        
        self.current_mode = RelayMode.ROUTING
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.ROUTING_RESULT,
            data={
                'routing_result': route_result,
                'route_found': route_result.get('success', False)
            },
            priority=Priority.NORMAL
        )
    
    async def _find_optimal_route(self, signal_data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """Find optimal routing path for signal"""
        if target in self.registered_modules:
            # Update module health and routing success
            self.module_health[target] = min(1.0, self.module_health[target] + 0.01)
            
            return {
                'success': True,
                'target_module': target,
                'route_quality': self.module_health[target],
                'estimated_latency': 0.001  # Very fast internal routing
            }
        else:
            return {
                'success': False,
                'reason': f'Module {target} not registered',
                'available_modules': list(self.registered_modules.keys())
            }
    
    async def _coordinate_signal(self, signal: NeuralSignal) -> NeuralSignal:
        """Coordinate and potentially modify signal based on global state"""
        # Update module tracking
        if signal.source in self.registered_modules:
            self.registered_modules[signal.source]['last_seen'] = datetime.now()
            self.registered_modules[signal.source]['signal_count'] += 1
        
        # Apply consciousness-based filtering and boosting
        modified_signal = await self._apply_consciousness_coordination(signal)
        
        # Update global workspace if signal is significant
        if signal.priority.value <= Priority.HIGH.value:
            await self._update_workspace_from_signal(signal)
        
        self.current_mode = RelayMode.INTEGRATING
        self.integration_events += 1
        
        return modified_signal
    
    async def _apply_consciousness_coordination(self, signal: NeuralSignal) -> NeuralSignal:
        """Apply consciousness-based coordination to signal"""
        # Check if signal requires consciousness
        signal_category = self._categorize_signal(signal)
        consciousness_rules = self.default_routing_rules['consciousness_filters'].get(signal_category, {})
        
        # Boost priority if consciousness is focused on source
        if (self.consciousness_state.focus_target == signal.source and 
            signal.source in self.consciousness_state.attention_distribution):
            
            attention_level = self.consciousness_state.attention_distribution[signal.source]
            if attention_level > self.attention_threshold:
                # Boost signal priority
                new_priority_value = max(1, signal.priority.value - 1)
                signal.priority = Priority(new_priority_value)
        
        return signal
    
    def _categorize_signal(self, signal: NeuralSignal) -> str:
        """Categorize signal for consciousness processing"""
        signal_type = signal.signal_type.value
        
        if 'safety' in signal_type:
            return 'safety_signals'
        elif 'memory' in signal_type:
            return 'memory_signals'
        elif 'motor' in signal_type:
            return 'motor_signals'
        elif any(word in signal_type for word in ['visual', 'auditory', 'sensory']):
            return 'sensory_signals'
        else:
            return 'cognitive_signals'
    
    async def _update_workspace_from_signal(self, signal: NeuralSignal):
        """Update global workspace based on significant signal"""
        if signal.signal_type in [SignalType.SAFETY_CHECK, SignalType.THREAT_DETECTION]:
            self.workspace_categories['environmental_state']['safety_alert'] = {
                'source': signal.source,
                'timestamp': datetime.now(),
                'priority': signal.priority.name
            }
        elif signal.signal_type == SignalType.MEMORY:
            self.workspace_categories['recent_events'].append({
                'type': 'memory_operation',
                'source': signal.source,
                'timestamp': datetime.now()
            })
    
    def _analyze_recent_signals(self) -> Dict[str, Any]:
        """Analyze recent signal patterns"""
        if not self.signal_history:
            return {'no_data': True}
        
        recent_signals = list(self.signal_history)[-50:]  # Last 50 signals
        signal_types = [s['signal_type'] for s in recent_signals]
        sources = [s['source'] for s in recent_signals]
        
        return {
            'total_recent': len(recent_signals),
            'most_common_type': max(set(signal_types), key=signal_types.count),
            'most_active_source': max(set(sources), key=sources.count),
            'average_priority': sum(s['priority'] for s in recent_signals) / len(recent_signals)
        }
    
    def _calculate_routing_efficiency(self) -> float:
        """Calculate overall routing efficiency"""
        if not self.module_health:
            return 0.0
        
        return sum(self.module_health.values()) / len(self.module_health)
    
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        current_time = datetime.now()
        
        # Apply consciousness decay
        if self.consciousness_state.level.value > ConsciousnessLevel.AWARE.value:
            decay = self.consciousness_decay_rate
            new_value = max(ConsciousnessLevel.AWARE.value, 
                          self.consciousness_state.level.value - decay)
            self.consciousness_state.level = ConsciousnessLevel(int(new_value))
        
        # Clean old routing table entries
        for module_routes in self.routing_table.values():
            # Remove routes older than 1 hour with low success rate
            module_routes[:] = [
                route for route in module_routes
                if (current_time - route.timestamp).total_seconds() / 3600 < 1 or route.success_rate > 0.5
            ]
        
        # Update module health based on activity
        for module_name, module_info in self.registered_modules.items():
            time_since_seen = (current_time - module_info['last_seen']).seconds
            if time_since_seen > 300:  # 5 minutes
                self.module_health[module_name] *= 0.95  # Gradual decay
        
        # Clean global workspace
        for category, content in self.workspace_categories.items():
            if category == 'recent_events':
                # Keep only last 10 events
                continue
            elif isinstance(content, dict):
                # Remove entries older than 1 hour
                expired_keys = [
                    key for key, value in content.items()
                    if (isinstance(value, dict) and 
                        'timestamp' in value and
                        (current_time - value['timestamp']).total_seconds() / 3600 > 1)
                ]
                for key in expired_keys:
                    del content[key]
        
        # Reset mode if idle
        if self.current_mode != RelayMode.IDLE:
            self.current_mode = RelayMode.IDLE
    
    def get_status(self) -> Dict[str, Any]:
        """Get thalamus status"""
        base_status = super().get_status()
        
        thalamus_status = {
            'relay_mode': self.current_mode.value,
            'consciousness_level': self.consciousness_state.level.name,
            'focus_target': self.consciousness_state.focus_target,
            'registered_modules': len(self.registered_modules),
            'active_routes': len(self.active_routes),
            'signals_routed': self.signals_routed,
            'consciousness_updates': self.consciousness_updates,
            'attention_shifts': self.attention_shifts,
            'integration_events': self.integration_events,
            'routing_efficiency': self._calculate_routing_efficiency(),
            'global_workspace_items': len(self.consciousness_state.global_workspace),
            'attention_distribution': dict(self.consciousness_state.attention_distribution),
            'module_health_average': sum(self.module_health.values()) / max(1, len(self.module_health))
        }
        
        base_status.update(thalamus_status)
        return base_status
