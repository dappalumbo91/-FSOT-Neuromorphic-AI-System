"""
Base Brain Module Class
Clean, extensible foundation for all brain modules
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import logging

from core import NeuralSignal, SignalType, Priority, neural_hub, consciousness_monitor
from core import FSOTEngine, Domain

logger = logging.getLogger(__name__)

class BrainModule(ABC):
    """
    Abstract base class for all brain modules
    Provides clean, consistent interface for modular brain architecture
    """
    
    def __init__(self, name: str, anatomical_region: str, functions: List[str]):
        self.name = name
        self.anatomical_region = anatomical_region
        self.functions = functions
        
        # Module state
        self.activation_level = 0.0
        self.is_active = False
        self.last_activity = datetime.now()
        
        # Processing state
        self.processing_queue: List[NeuralSignal] = []
        self.max_queue_size = 100
        self.processing_timeout = 1.0
        
        # Connections to other modules
        self.connections: Set[str] = set()
        self.connection_strengths: Dict[str, float] = {}
        
        # FSOT integration
        self.fsot_engine = FSOTEngine()
        self.fsot_domain = Domain.COGNITIVE  # Default domain
        
        # Performance metrics
        self.signals_processed = 0
        self.processing_errors = 0
        self.average_response_time = 0.0
        
        # Register with neural hub
        neural_hub.register_module(self.name, self.process_signal)
        
        # Start background processing - will be started by orchestrator
        self.processing_task: Optional[asyncio.Task] = None
    
    async def start_processing(self):
        """Start background signal processing"""
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._processing_loop())
    
    async def stop_processing(self):
        """Stop background processing"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def _processing_loop(self):
        """Background processing loop"""
        while True:
            try:
                if self.processing_queue:
                    signal = self.processing_queue.pop(0)
                    await self._internal_process_signal(signal)
                else:
                    await asyncio.sleep(0.01)  # Small delay when no signals
            except Exception as e:
                logger.error(f"{self.name} processing error: {e}")
                self.processing_errors += 1
                await asyncio.sleep(0.1)  # Back off on error
    
    async def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """
        Main signal processing entry point
        Called by neural hub when signal is received
        """
        try:
            # Add to processing queue if busy
            if len(self.processing_queue) >= self.max_queue_size:
                logger.warning(f"{self.name} queue full, dropping signal")
                return None
            
            # If expecting response, process immediately
            if signal.response_expected:
                return await self._internal_process_signal(signal)
            else:
                # Add to queue for background processing
                self.processing_queue.append(signal)
                return None
                
        except Exception as e:
            logger.error(f"{self.name} signal processing error: {e}")
            self.processing_errors += 1
            return None
    
    async def _internal_process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Internal signal processing with metrics and activation updates"""
        start_time = datetime.now()
        
        try:
            # Update activation
            self._update_activation(signal)
            
            # Process signal using module-specific logic
            response = await self._process_signal_impl(signal)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=True)
            
            # Update consciousness contribution
            self._update_consciousness_contribution()
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=False)
            logger.error(f"{self.name} internal processing error: {e}")
            return None
    
    @abstractmethod
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """
        Module-specific signal processing implementation
        Must be implemented by subclasses
        """
        pass
    
    def _update_activation(self, signal: NeuralSignal):
        """Update module activation based on signal"""
        # Simple activation model - can be overridden
        signal_strength = 0.1
        
        # Boost activation based on signal priority
        if signal.priority == Priority.VITAL:
            signal_strength = 0.8
        elif signal.priority == Priority.URGENT:
            signal_strength = 0.6
        elif signal.priority == Priority.HIGH:
            signal_strength = 0.4
        
        # Update activation with decay
        decay_factor = 0.95
        self.activation_level = (self.activation_level * decay_factor) + signal_strength
        self.activation_level = min(1.0, self.activation_level)
        
        # Update activity status
        self.is_active = self.activation_level > 0.1
        self.last_activity = datetime.now()
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update processing metrics"""
        if success:
            self.signals_processed += 1
            # Update rolling average response time
            alpha = 0.1  # Smoothing factor
            self.average_response_time = (
                (1 - alpha) * self.average_response_time + 
                alpha * processing_time
            )
        else:
            self.processing_errors += 1
    
    def _update_consciousness_contribution(self):
        """Update contribution to global consciousness"""
        # Calculate contribution based on activation and function complexity
        base_contribution = self.activation_level * 0.1
        function_complexity = len(self.functions) * 0.02
        contribution = min(0.3, base_contribution + function_complexity)
        
        consciousness_monitor.update_region_contribution(self.name, contribution)
    
    def connect_to(self, module_name: str, strength: float = 1.0):
        """Establish connection to another module"""
        self.connections.add(module_name)
        self.connection_strengths[module_name] = strength
    
    def disconnect_from(self, module_name: str):
        """Remove connection to another module"""
        self.connections.discard(module_name)
        if module_name in self.connection_strengths:
            del self.connection_strengths[module_name]
    
    async def send_to_module(self, target_module: str, signal_type: SignalType, 
                           data: Dict[str, Any], **kwargs) -> Optional[NeuralSignal]:
        """Send signal to another module"""
        if target_module not in self.connections:
            logger.warning(f"{self.name} not connected to {target_module}")
            return None
        
        signal = NeuralSignal(
            source=self.name,
            target=target_module,
            signal_type=signal_type,
            data=data,
            **kwargs
        )
        
        return await neural_hub.send_signal(signal)
    
    async def broadcast_to_connections(self, signal_type: SignalType, 
                                     data: Dict[str, Any], **kwargs) -> Dict[str, Optional[NeuralSignal]]:
        """Broadcast signal to all connected modules"""
        if not self.connections:
            return {}
        
        signal = NeuralSignal(
            source=self.name,
            target="",  # Will be set for each target
            signal_type=signal_type,
            data=data,
            **kwargs
        )
        
        return await neural_hub.broadcast_signal(signal, list(self.connections))
    
    def get_fsot_scalar(self, **overrides) -> float:
        """Get FSOT scalar for current module state"""
        # Include activation level in FSOT calculation
        overrides['delta_psi'] = overrides.get('delta_psi', self.activation_level)
        overrides['observed'] = overrides.get('observed', self.is_active)
        
        return self.fsot_engine.compute_for_domain(self.fsot_domain, **overrides)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current module status"""
        return {
            'name': self.name,
            'anatomical_region': self.anatomical_region,
            'functions': self.functions,
            'activation_level': self.activation_level,
            'is_active': self.is_active,
            'last_activity': self.last_activity.isoformat(),
            'connections': list(self.connections),
            'queue_size': len(self.processing_queue),
            'metrics': {
                'signals_processed': self.signals_processed,
                'processing_errors': self.processing_errors,
                'average_response_time': self.average_response_time,
                'error_rate': self.processing_errors / max(1, self.signals_processed)
            },
            'fsot_scalar': self.get_fsot_scalar()
        }
    
    async def perform_maintenance(self):
        """
        Perform periodic maintenance tasks for the brain module
        
        This method is called periodically to perform cleanup and optimization
        tasks. Base implementation provides common maintenance functionality
        that can be extended by subclasses.
        """
        current_time = datetime.now()
        
        # Update last activity timestamp
        self.last_activity = current_time
        
        # Clean up old processed signals if queue is getting large
        if len(self.processing_queue) > self.max_queue_size * 0.8:
            # Remove oldest 25% of signals
            num_to_remove = len(self.processing_queue) // 4
            self.processing_queue = self.processing_queue[num_to_remove:]
            logger.debug(f"{self.name}: Cleaned up {num_to_remove} old signals from queue")
        
        # Update performance metrics
        if self.signals_processed > 0:
            error_rate = self.processing_errors / self.signals_processed
            if error_rate > 0.1:  # More than 10% error rate
                logger.warning(f"{self.name}: High error rate detected: {error_rate:.2%}")
        
        # Reset counters if they get too large to prevent overflow
        if self.signals_processed > 1000000:
            self.signals_processed = self.signals_processed // 2
            self.processing_errors = self.processing_errors // 2
            logger.debug(f"{self.name}: Reset performance counters to prevent overflow")
    
    async def shutdown(self):
        """Shutdown module gracefully"""
        await self.stop_processing()
        neural_hub.unregister_module(self.name)
        consciousness_monitor.update_region_contribution(self.name, 0.0)
        logger.info(f"{self.name} module shutdown complete")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', active={self.is_active}, activation={self.activation_level:.2f})"
