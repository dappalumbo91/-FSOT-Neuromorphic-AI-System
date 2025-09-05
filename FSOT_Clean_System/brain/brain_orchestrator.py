"""
Brain Orchestrator - Coordinates All Brain Modules
Clean brain coordination and management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_module import BrainModule
from .frontal_cortex import FrontalCortex
from .hippocampus import Hippocampus
from .amygdala import Amygdala
from .cerebellum import Cerebellum
from .temporal_lobe import TemporalLobe
from .occipital_lobe import OccipitalLobe
from .thalamus import Thalamus
from .parietal_lobe import ParietalLobe
from .pflt import PFLT
from .brainstem import Brainstem

from core import NeuralSignal, SignalType, Priority, neural_hub, consciousness_monitor

logger = logging.getLogger(__name__)

class BrainOrchestrator:
    """
    Central orchestrator for all brain modules
    Manages module lifecycle, connections, and coordination
    """
    
    def __init__(self):
        self.modules: Dict[str, BrainModule] = {}
        self.module_connections: Dict[str, List[str]] = {}
        self.is_initialized = False
        self.start_time = datetime.now()
        
        # Brain state
        self.overall_activation = 0.0
        self.processing_load = 0.0
        self.queries_processed = 0
    
    async def initialize(self):
        """Initialize all brain modules and establish connections"""
        logger.info("🧠 Initializing brain modules...")
        
        try:
            # Initialize core brain modules
            await self._initialize_modules()
            
            # Establish inter-module connections
            await self._establish_connections()
            
            # Start coordination monitoring
            await self._start_coordination_monitoring()
            
            self.is_initialized = True
            logger.info("✅ Brain orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Brain initialization failed: {e}")
            raise
    
    async def _initialize_modules(self):
        """Initialize individual brain modules"""
        # Core Brain Modules
        logger.info("Initializing frontal cortex...")
        self.modules['frontal_cortex'] = FrontalCortex()
        
        logger.info("Initializing hippocampus...")
        self.modules['hippocampus'] = Hippocampus()
        
        logger.info("Initializing amygdala...")
        self.modules['amygdala'] = Amygdala()
        
        # Extended Brain Modules
        logger.info("Initializing cerebellum...")
        self.modules['cerebellum'] = Cerebellum()
        
        logger.info("Initializing temporal lobe...")
        self.modules['temporal_lobe'] = TemporalLobe()
        
        logger.info("Initializing occipital lobe...")
        self.modules['occipital_lobe'] = OccipitalLobe()
        
        logger.info("Initializing thalamus...")
        self.modules['thalamus'] = Thalamus()
        
        # New Advanced Brain Modules
        logger.info("Initializing parietal lobe...")
        self.modules['parietal_lobe'] = ParietalLobe()
        
        logger.info("Initializing PFLT (language processing)...")
        self.modules['pflt'] = PFLT()
        
        logger.info("Initializing brainstem...")
        self.modules['brainstem'] = Brainstem()
        
        # Start processing for all modules
        for module in self.modules.values():
            await module.start_processing()
        
        logger.info(f"Initialized {len(self.modules)} brain modules (comprehensive neuromorphic brain)")
        
        
        # Register all modules with thalamus for coordination
        await self._register_modules_with_thalamus()
    
    async def _establish_connections(self):
        """Establish connections between brain modules"""
        # Define comprehensive brain module connections
        connections = {
            'frontal_cortex': ['hippocampus', 'amygdala', 'thalamus', 'temporal_lobe', 'parietal_lobe', 'pflt'],
            'hippocampus': ['frontal_cortex', 'amygdala', 'thalamus', 'temporal_lobe'],
            'amygdala': ['frontal_cortex', 'hippocampus', 'thalamus', 'brainstem'],
            'cerebellum': ['frontal_cortex', 'thalamus', 'parietal_lobe', 'brainstem'],
            'temporal_lobe': ['frontal_cortex', 'hippocampus', 'thalamus', 'occipital_lobe', 'pflt'],
            'occipital_lobe': ['frontal_cortex', 'temporal_lobe', 'thalamus', 'parietal_lobe'],
            'thalamus': ['frontal_cortex', 'hippocampus', 'amygdala', 'cerebellum', 'temporal_lobe', 'occipital_lobe', 'parietal_lobe', 'pflt', 'brainstem'],
            'parietal_lobe': ['frontal_cortex', 'thalamus', 'occipital_lobe', 'cerebellum'],
            'pflt': ['frontal_cortex', 'temporal_lobe', 'thalamus'],
            'brainstem': ['thalamus', 'amygdala', 'cerebellum']
        }
        
        # Establish connections
        for source_module, targets in connections.items():
            if source_module in self.modules:
                for target in targets:
                    if target in self.modules:
                        self.modules[source_module].connect_to(target)
                        logger.debug(f"Connected {source_module} → {target}")
        
        self.module_connections = connections
        logger.info(f"Complete brain module connections established ({len(self.modules)} modules fully connected)")
    
    async def _register_modules_with_thalamus(self):
        """Register all modules with thalamus for centralized coordination"""
        if 'thalamus' not in self.modules:
            return
        
        thalamus = self.modules['thalamus']
        
        for module_name, module in self.modules.items():
            if module_name != 'thalamus':
                # Create registration signal
                registration_signal = NeuralSignal(
                    source=module_name,
                    target='thalamus',
                    signal_type=SignalType.MODULE_REGISTRATION,
                    data={
                        'module': {
                            'name': module_name,
                            'info': {
                                'functions': module.functions,
                                'anatomical_region': module.anatomical_region,
                                'capabilities': getattr(module, 'capabilities', [])
                            }
                        }
                    },
                    priority=Priority.HIGH
                )
                
                # Register with thalamus
                await thalamus.process_signal(registration_signal)
                logger.debug(f"Registered {module_name} with thalamus")
        
        logger.info("All modules registered with thalamus for centralized coordination")
    
    async def _start_coordination_monitoring(self):
        """Start background coordination monitoring - DISABLED to prevent loops"""
        logger.info("⚠️ Coordination monitoring DISABLED to prevent endless loops")
        # DON'T start the coordination loop - this prevents endless loops
        # asyncio.create_task(self._coordination_loop())
    
    async def _coordination_loop(self):
        """Background loop for brain coordination"""
        loop_count = 0
        max_loops = 10000  # Prevent truly endless loops
        
        while self.is_initialized and loop_count < max_loops:
            try:
                await self._update_brain_state()
                await asyncio.sleep(0.5)  # Update every 500ms
                loop_count += 1
                
                # Check for shutdown every 100 iterations
                if loop_count % 100 == 0:
                    logger.debug(f"Coordination loop iteration {loop_count}")
                    
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(1.0)
                
        if loop_count >= max_loops:
            logger.warning("Coordination loop reached maximum iterations - stopping")
            self.is_initialized = False
    
    async def _update_brain_state(self):
        """Update overall brain state"""
        if not self.modules:
            return
        
        # Calculate overall activation
        activations = [module.activation_level for module in self.modules.values()]
        self.overall_activation = sum(activations) / len(activations)
        
        # Calculate processing load
        queue_sizes = [len(module.processing_queue) for module in self.modules.values()]
        max_queue = max(queue_sizes) if queue_sizes else 0
        self.processing_load = min(1.0, max_queue / 50.0)  # Normalize to 0-1
        
        # Update consciousness monitor
        consciousness_monitor.update_processing_load(self.processing_load)
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query through the brain system
        
        Args:
            query: Input query string
            context: Optional context information
            
        Returns:
            Processing results from brain modules
        """
        if not self.is_initialized:
            raise RuntimeError("Brain orchestrator not initialized")
        
        self.queries_processed += 1
        start_time = datetime.now()
        
        logger.info(f"🧠 Processing query: {query[:50]}...")
        
        try:
            # Create cognitive signal for frontal cortex
            query_data = {
                'query': query,
                'context': context or {},
                'timestamp': start_time.isoformat()
            }
            
            # Send to frontal cortex for executive processing
            if 'frontal_cortex' in self.modules:
                signal = NeuralSignal(
                    source='brain_orchestrator',
                    target='frontal_cortex',
                    signal_type=SignalType.COGNITIVE,
                    data={
                        'decision_request': {
                            'options': ['process_query', 'request_clarification', 'defer_processing'],
                            'context': query_data,
                            'urgency': 'normal'
                        }
                    },
                    priority=Priority.NORMAL,
                    response_expected=True,
                    timeout=2.0
                )
                
                response = await neural_hub.send_signal(signal)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    'query': query,
                    'response': response.data if response else {'error': 'No response from frontal cortex'},
                    'processing_time': processing_time,
                    'modules_involved': ['frontal_cortex'],
                    'brain_state': {
                        'overall_activation': self.overall_activation,
                        'processing_load': self.processing_load,
                        'consciousness_level': consciousness_monitor.current_metrics.level
                    }
                }
                
                logger.info(f"✅ Query processed in {processing_time:.2f}s")
                return result
            else:
                raise RuntimeError("No brain modules available for processing")
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                'query': query,
                'error': str(e),
                'processing_time': processing_time,
                'modules_involved': [],
                'brain_state': {
                    'overall_activation': self.overall_activation,
                    'processing_load': self.processing_load
                }
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive brain status"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        module_statuses = {}
        for name, module in self.modules.items():
            module_statuses[name] = module.get_status()
        
        return {
            'initialized': self.is_initialized,
            'uptime_seconds': uptime,
            'modules': module_statuses,
            'connections': self.module_connections,
            'overall_activation': self.overall_activation,
            'processing_load': self.processing_load,
            'queries_processed': self.queries_processed,
            'consciousness': consciousness_monitor.get_current_state(),
            'neural_hub_stats': neural_hub.get_stats()
        }
    
    async def get_module(self, module_name: str) -> Optional[BrainModule]:
        """Get specific brain module"""
        return self.modules.get(module_name)
    
    def send_signal(self, module_name: str, signal: Dict[str, Any]):
        """Send signal to specific brain module (for integration compatibility)"""
        if module_name in self.modules:
            try:
                # Process signal through the module
                module = self.modules[module_name]
                # For integration purposes, we'll log the signal
                logger.debug(f"Signal sent to {module_name}: {signal.get('type', 'unknown')}")
                return True
            except Exception as e:
                logger.warning(f"Failed to send signal to {module_name}: {e}")
                return False
        else:
            logger.warning(f"Module {module_name} not found for signal routing")
            return False
    
    async def shutdown(self):
        """Shutdown all brain modules gracefully"""
        logger.info("🔄 Shutting down brain orchestrator...")
        
        self.is_initialized = False
        
        # Shutdown all modules
        for name, module in self.modules.items():
            logger.info(f"Shutting down {name}...")
            await module.shutdown()
        
        self.modules.clear()
        logger.info("✅ Brain orchestrator shutdown complete")
