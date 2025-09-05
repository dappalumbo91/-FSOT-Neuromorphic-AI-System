"""Core package initialization"""
from .fsot_engine import FSOTEngine, FSOTParameters, Domain
from .neural_signal import NeuralSignal, SignalType, Priority, neural_hub, create_signal
from .consciousness import ConsciousnessMonitor, ConsciousnessState, ConsciousnessMetrics, consciousness_monitor

__all__ = [
    'FSOTEngine', 'FSOTParameters', 'Domain',
    'NeuralSignal', 'SignalType', 'Priority', 'neural_hub', 'create_signal',
    'ConsciousnessMonitor', 'ConsciousnessState', 'ConsciousnessMetrics', 'consciousness_monitor'
]
