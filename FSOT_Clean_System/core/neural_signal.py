"""
Neural Signal Communication System
Clean implementation of brain module communication
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

class SignalType(Enum):
    """Types of neural signals"""
    SENSORY = "sensory_input"
    COGNITIVE = "cognitive_processing"
    MEMORY = "memory_operation"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_RESULT = "memory_result"
    LEARNING = "learning"
    MOTOR = "motor_command"
    EMOTIONAL = "emotional_response"
    EXECUTIVE = "executive_control"
    CONSCIOUSNESS = "consciousness_update"
    SAFETY_CHECK = "safety_check"
    SAFETY_RESULT = "safety_result"
    SAFETY_BLOCK = "safety_block"
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    EMOTIONAL_RESULT = "emotional_result"
    THREAT_DETECTION = "threat_detection"
    THREAT_RESULT = "threat_result"
    
    # Extended brain module signals
    MOTOR_COORDINATION = "motor_coordination"
    MOTOR_RESULT = "motor_result"
    SKILL_LEARNING = "skill_learning" 
    SKILL_LEARNING_RESULT = "skill_learning_result"
    BALANCE_CORRECTION = "balance_correction"
    BALANCE_CHECK = "balance_check"
    BALANCE_RESULT = "balance_result"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESULT = "coordination_result"
    
    # Brainstem specific signals
    CIRCADIAN_REGULATION = "circadian_regulation"
    CIRCADIAN_REGULATION_RESULT = "circadian_regulation_result"
    HOMEOSTATIC_CONTROL = "homeostatic_control"
    HOMEOSTATIC_CONTROL_RESULT = "homeostatic_control_result"
    CONSCIOUSNESS_REGULATION = "consciousness_regulation"
    CONSCIOUSNESS_REGULATION_RESULT = "consciousness_regulation_result"
    VITAL_FUNCTION_CONTROL = "vital_function_control"
    VITAL_FUNCTION_RESPONSE = "vital_function_response"
    AUTONOMIC_REGULATION = "autonomic_regulation"
    AUTONOMIC_REGULATION_RESULT = "autonomic_regulation_result"
    REFLEX_ACTIVATION = "reflex_activation"
    REFLEX_ACTIVATION_RESULT = "reflex_activation_result"
    LANGUAGE_COMPREHENSION = "language_comprehension"
    LANGUAGE_COMPREHENSION_RESULT = "language_comprehension_result"
    LANGUAGE_GENERATION = "language_generation"
    LANGUAGE_GENERATION_RESULT = "language_generation_result"
    CONVERSATION_MANAGEMENT = "conversation_management"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    VISUAL_PROCESSING = "visual_processing"
    VISUAL_PROCESSING_RESULT = "visual_processing_result"
    PATTERN_RECOGNITION = "pattern_recognition"
    PATTERN_RECOGNITION_RESULT = "pattern_recognition_result"
    OBJECT_DETECTION = "object_detection"
    OBJECT_DETECTION_RESULT = "object_detection_result"
    MOTION_TRACKING = "motion_tracking"
    MOTION_TRACKING_RESULT = "motion_tracking_result"
    AUDITORY_PROCESSING = "auditory_processing"
    AUDITORY_PROCESSING_RESULT = "auditory_processing_result"
    CONVERSATION_MANAGEMENT_RESULT = "conversation_management_result"
    SEMANTIC_ANALYSIS_RESULT = "semantic_analysis_result"
    VISUAL_MEMORY = "visual_memory"
    VISUAL_MEMORY_RESULT = "visual_memory_result"
    
    # New Parietal Lobe signals
    SPATIAL_REASONING = "spatial_reasoning"
    SPATIAL_REASONING_RESULT = "spatial_reasoning_result"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    MATHEMATICAL_COMPUTATION_RESULT = "mathematical_computation_result"
    COORDINATE_TRANSFORMATION = "coordinate_transformation"
    COORDINATE_TRANSFORMATION_RESULT = "coordinate_transformation_result"
    SENSORY_INTEGRATION = "sensory_integration"
    SENSORY_INTEGRATION_RESULT = "sensory_integration_result"
    PATTERN_ANALYSIS = "pattern_analysis"
    PATTERN_ANALYSIS_RESULT = "pattern_analysis_result"
    SPATIAL_MAPPING = "spatial_mapping"
    SPATIAL_MAPPING_RESULT = "spatial_mapping_result"
    PARIETAL_PROCESSING_RESULT = "parietal_processing_result"
    
    # New PFLT (Language) signals
    LANGUAGE_TRANSLATION = "language_translation"
    LANGUAGE_TRANSLATION_RESULT = "language_translation_result"
    PHONEME_ANALYSIS = "phoneme_analysis"
    PHONEME_ANALYSIS_RESULT = "phoneme_analysis_result"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    CREATIVE_GENERATION = "creative_generation"
    CREATIVE_GENERATION_RESULT = "creative_generation_result"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    SEMANTIC_UNDERSTANDING_RESULT = "semantic_understanding_result"
    LANGUAGE_DETECTION = "language_detection"
    LANGUAGE_DETECTION_RESULT = "language_detection_result"
    LINGUISTIC_ANALYSIS_RESULT = "linguistic_analysis_result"
    PFLT_PROCESSING_RESULT = "pflt_processing_result"
    
    # New Brainstem signals
    BRAINSTEM_MONITORING = "brainstem_monitoring"
    
    # Generic result signals
    ACKNOWLEDGMENT = "acknowledgment"
    
    # Thalamus coordination signals
    CONSCIOUSNESS_UPDATE = "consciousness_update"
    CONSCIOUSNESS_UPDATE_RESULT = "consciousness_update_result"
    ATTENTION_CONTROL = "attention_control"
    ATTENTION_CONTROL_RESULT = "attention_control_result"
    BRAIN_STATE_QUERY = "brain_state_query"
    BRAIN_STATE_RESULT = "brain_state_result"
    MODULE_REGISTRATION = "module_registration"
    MODULE_REGISTRATION_RESULT = "module_registration_result"
    ROUTING_REQUEST = "routing_request"
    ROUTING_RESULT = "routing_result"

class Priority(Enum):
    """Signal processing priorities"""
    VITAL = 1      # Life-critical functions
    URGENT = 2     # Immediate response required
    HIGH = 3       # Important processing
    NORMAL = 4     # Standard processing
    LOW = 5        # Background processing

@dataclass
class NeuralSignal:
    """
    Represents a signal passing between brain modules
    Clean, focused implementation
    """
    source: str
    target: str
    signal_type: SignalType
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    response_expected: bool = False
    timeout: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'source': self.source,
            'target': self.target,
            'signal_type': self.signal_type.value,
            'data': self.data,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'response_expected': self.response_expected,
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralSignal':
        """Create from dictionary"""
        return cls(
            source=data['source'],
            target=data['target'],
            signal_type=SignalType(data['signal_type']),
            data=data['data'],
            priority=Priority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            signal_id=data['signal_id'],
            response_expected=data.get('response_expected', False),
            timeout=data.get('timeout', 1.0)
        )

class NeuralCommunicationHub:
    """
    Central hub for neural signal routing and management
    """
    
    def __init__(self):
        self.registered_modules: Dict[str, Callable] = {}
        self.signal_history: List[NeuralSignal] = []
        self.active_signals: Dict[str, NeuralSignal] = {}
        self.signal_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'average_response_time': 0.0
        }
        self._lock = asyncio.Lock()
    
    def register_module(self, module_name: str, signal_handler: Callable):
        """Register a brain module for signal processing"""
        self.registered_modules[module_name] = signal_handler
    
    def unregister_module(self, module_name: str):
        """Unregister a brain module"""
        if module_name in self.registered_modules:
            del self.registered_modules[module_name]
    
    async def send_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """
        Send a neural signal to target module
        
        Args:
            signal: Neural signal to send
            
        Returns:
            Response signal if response_expected, otherwise None
        """
        async with self._lock:
            self.signal_history.append(signal)
            self.active_signals[signal.signal_id] = signal
        
        try:
            # Check if target module is registered
            if signal.target not in self.registered_modules:
                raise ValueError(f"Target module '{signal.target}' not registered")
            
            # Get handler for target module
            handler = self.registered_modules[signal.target]
            
            # Process signal with timeout
            start_time = datetime.now()
            
            if signal.response_expected:
                response = await asyncio.wait_for(
                    handler(signal), 
                    timeout=signal.timeout
                )
                
                # Update stats
                response_time = (datetime.now() - start_time).total_seconds()
                self._update_stats(response_time, success=True)
                
                return response
            else:
                # Fire and forget
                asyncio.create_task(handler(signal))
                self._update_stats(0.0, success=True)
                return None
                
        except asyncio.TimeoutError:
            self._update_stats(signal.timeout, success=False)
            raise TimeoutError(f"Signal {signal.signal_id} timed out")
            
        except Exception as e:
            self._update_stats(0.0, success=False)
            raise RuntimeError(f"Signal processing failed: {e}")
            
        finally:
            # Clean up active signal
            async with self._lock:
                if signal.signal_id in self.active_signals:
                    del self.active_signals[signal.signal_id]
    
    async def broadcast_signal(self, signal: NeuralSignal, targets: List[str]) -> Dict[str, Optional[NeuralSignal]]:
        """
        Broadcast signal to multiple targets
        
        Args:
            signal: Signal to broadcast
            targets: List of target module names
            
        Returns:
            Dictionary mapping target names to their responses
        """
        responses = {}
        tasks = []
        
        for target in targets:
            # Create individual signal for each target
            target_signal = NeuralSignal(
                source=signal.source,
                target=target,
                signal_type=signal.signal_type,
                data=signal.data.copy(),
                priority=signal.priority,
                response_expected=signal.response_expected,
                timeout=signal.timeout
            )
            
            task = asyncio.create_task(self.send_signal(target_signal))
            tasks.append((target, task))
        
        # Wait for all responses
        for target, task in tasks:
            try:
                response = await task
                responses[target] = response
            except Exception as e:
                responses[target] = None
                print(f"Failed to get response from {target}: {e}")
        
        return responses
    
    def _update_stats(self, response_time: float, success: bool):
        """Update signal processing statistics"""
        if success:
            self.signal_stats['total_processed'] += 1
            # Update rolling average
            total = self.signal_stats['total_processed']
            current_avg = self.signal_stats['average_response_time']
            self.signal_stats['average_response_time'] = (current_avg * (total - 1) + response_time) / total
        else:
            self.signal_stats['total_failed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            **self.signal_stats,
            'active_signals': len(self.active_signals),
            'registered_modules': list(self.registered_modules.keys()),
            'total_history': len(self.signal_history)
        }
    
    def clear_history(self, keep_recent: int = 100):
        """Clear signal history, keeping recent signals"""
        if len(self.signal_history) > keep_recent:
            self.signal_history = self.signal_history[-keep_recent:]

# Global communication hub instance
neural_hub = NeuralCommunicationHub()

def create_signal(source: str, target: str, signal_type: SignalType, 
                 data: Dict[str, Any], **kwargs) -> NeuralSignal:
    """
    Convenience function to create neural signals
    
    Args:
        source: Source module name
        target: Target module name
        signal_type: Type of signal
        data: Signal data
        **kwargs: Additional signal parameters
        
    Returns:
        Created neural signal
    """
    return NeuralSignal(
        source=source,
        target=target,
        signal_type=signal_type,
        data=data,
        **kwargs
    )
