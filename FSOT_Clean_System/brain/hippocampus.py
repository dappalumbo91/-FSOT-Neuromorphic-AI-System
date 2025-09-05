"""
FSOT 2.0 Hippocampus Brain Module
Memory formation, consolidation, and retrieval
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base_module import BrainModule
from core import NeuralSignal, SignalType, Priority

logger = logging.getLogger(__name__)

@dataclass
class MemoryTrace:
    """Represents a memory trace in the hippocampus"""
    content: Any
    timestamp: datetime
    importance: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    associations: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []

class Hippocampus(BrainModule):
    """
    Hippocampus Brain Module - Memory and Learning
    
    Responsibilities:
    - Short-term to long-term memory consolidation
    - Pattern recognition and association
    - Spatial and temporal context processing
    - Learning from experience
    """
    
    def __init__(self):
        super().__init__(
            name="hippocampus",
            anatomical_region="temporal_lobe",
            functions=[
                "memory_consolidation",
                "pattern_recognition", 
                "spatial_navigation",
                "episodic_memory",
                "learning"
            ]
        )
        
        # Memory storage
        self.short_term_memory: Dict[str, MemoryTrace] = {}
        self.long_term_memory: Dict[str, MemoryTrace] = {}
        self.memory_associations: Dict[str, List[str]] = {}
        
        # Learning parameters
        self.consolidation_threshold = 0.7  # Importance threshold for long-term storage
        self.max_short_term_items = 100     # Maximum short-term memory items
        self.decay_rate = 0.95              # Memory decay rate per day
        
        # Performance metrics
        self.memories_stored = 0
        self.memories_retrieved = 0
        self.consolidations_performed = 0
    
    async def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signals"""
        try:
            if signal.signal_type == SignalType.MEMORY_STORE:
                return await self._store_memory(signal)
            elif signal.signal_type == SignalType.MEMORY_RETRIEVE:
                return await self._retrieve_memory(signal)
            elif signal.signal_type == SignalType.LEARNING:
                return await self._process_learning(signal)
            else:
                # Forward to other modules
                return signal
                
        except Exception as e:
            logger.error(f"Error processing signal in hippocampus: {e}")
            return None
    
    async def _process_signal_impl(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Module-specific signal processing implementation"""
        return await self.process_signal(signal)
    
    async def _store_memory(self, signal: NeuralSignal) -> NeuralSignal:
        """Store new memory trace"""
        content = signal.data.get('content')
        importance = signal.data.get('importance', 0.5)
        context = signal.data.get('context', {})
        
        # Create memory trace
        memory_id = f"mem_{datetime.now().timestamp()}"
        memory_trace = MemoryTrace(
            content=content,
            timestamp=datetime.now(),
            importance=importance
        )
        
        # Store in short-term memory first
        self.short_term_memory[memory_id] = memory_trace
        self.memories_stored += 1
        
        # Check if should be consolidated to long-term
        if importance >= self.consolidation_threshold:
            await self._consolidate_memory(memory_id, memory_trace)
        
        # Manage short-term memory capacity
        await self._manage_memory_capacity()
        
        logger.debug(f"Stored memory: {memory_id} (importance: {importance})")
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.ACKNOWLEDGMENT,
            data={'memory_id': memory_id, 'stored': True},
            priority=Priority.LOW
        )
    
    async def _retrieve_memory(self, signal: NeuralSignal) -> NeuralSignal:
        """Retrieve memory based on query"""
        query = signal.data.get('query', '')
        context = signal.data.get('context', {})
        
        # Search both short-term and long-term memory
        results = []
        
        # Search short-term memory
        for mem_id, trace in self.short_term_memory.items():
            if self._matches_query(trace, query):
                trace.access_count += 1
                trace.last_accessed = datetime.now()
                results.append({
                    'id': mem_id,
                    'content': trace.content,
                    'relevance': self._calculate_relevance(trace, query),
                    'type': 'short_term'
                })
        
        # Search long-term memory
        for mem_id, trace in self.long_term_memory.items():
            if self._matches_query(trace, query):
                trace.access_count += 1
                trace.last_accessed = datetime.now()
                results.append({
                    'id': mem_id,
                    'content': trace.content,
                    'relevance': self._calculate_relevance(trace, query),
                    'type': 'long_term'
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        self.memories_retrieved += 1
        
        logger.debug(f"Retrieved {len(results)} memories for query: {query}")
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.MEMORY_RESULT,
            data={'results': results, 'query': query},
            priority=Priority.HIGH
        )
    
    async def _process_learning(self, signal: NeuralSignal) -> NeuralSignal:
        """Process learning signal to strengthen memory associations"""
        experience = signal.data.get('experience')
        outcome = signal.data.get('outcome')
        success = signal.data.get('success', True)
        
        # Find related memories
        related_memories = []
        for mem_id, trace in self.short_term_memory.items():
            if self._is_related_to_experience(trace, experience):
                related_memories.append((mem_id, trace))
        
        # Strengthen or weaken associations based on outcome
        learning_factor = 1.1 if success else 0.9
        
        for mem_id, trace in related_memories:
            trace.importance *= learning_factor
            
            # If importance increased significantly, consider consolidation
            if trace.importance >= self.consolidation_threshold and mem_id in self.short_term_memory:
                await self._consolidate_memory(mem_id, trace)
        
        logger.debug(f"Processed learning from experience: {experience}")
        
        return NeuralSignal(
            source=self.name,
            target=signal.source,
            signal_type=SignalType.ACKNOWLEDGMENT,
            data={'learning_processed': True, 'memories_affected': len(related_memories)},
            priority=Priority.LOW
        )
    
    async def _consolidate_memory(self, memory_id: str, memory_trace: MemoryTrace):
        """Consolidate memory from short-term to long-term storage"""
        if memory_id in self.short_term_memory:
            # Move to long-term memory
            self.long_term_memory[memory_id] = memory_trace
            del self.short_term_memory[memory_id]
            
            self.consolidations_performed += 1
            logger.debug(f"Consolidated memory to long-term: {memory_id}")
    
    async def _manage_memory_capacity(self):
        """Manage short-term memory capacity"""
        if len(self.short_term_memory) > self.max_short_term_items:
            # Remove least important/oldest memories
            items = list(self.short_term_memory.items())
            items.sort(key=lambda x: (x[1].importance, x[1].timestamp))
            
            # Remove oldest 10% of items
            remove_count = len(items) // 10
            for i in range(remove_count):
                mem_id, _ = items[i]
                del self.short_term_memory[mem_id]
    
    def _matches_query(self, memory_trace: MemoryTrace, query: str) -> bool:
        """Check if memory trace matches query"""
        if not query:
            return True
        
        # Simple text matching - could be enhanced with embeddings
        content_str = str(memory_trace.content).lower()
        query_lower = query.lower()
        
        return query_lower in content_str
    
    def _calculate_relevance(self, memory_trace: MemoryTrace, query: str) -> float:
        """Calculate relevance score for memory trace"""
        # Simple relevance calculation - could be enhanced
        base_relevance = memory_trace.importance
        
        # Boost relevance for recently accessed memories
        if memory_trace.last_accessed:
            time_factor = 1.0 - min(1.0, (datetime.now() - memory_trace.last_accessed).days / 7)
            base_relevance *= (1.0 + time_factor * 0.2)
        
        # Boost relevance for frequently accessed memories
        access_factor = min(1.0, memory_trace.access_count / 10)
        base_relevance *= (1.0 + access_factor * 0.1)
        
        return base_relevance
    
    def _is_related_to_experience(self, memory_trace: MemoryTrace, experience: Any) -> bool:
        """Check if memory is related to current experience"""
        # Simple relation check - could be enhanced with semantic similarity
        return str(experience).lower() in str(memory_trace.content).lower()
    
    async def perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        await super().perform_maintenance()
        
        # Decay memory importance over time
        current_time = datetime.now()
        
        for trace in list(self.short_term_memory.values()):
            days_old = (current_time - trace.timestamp).days
            if days_old > 0:
                trace.importance *= (self.decay_rate ** days_old)
        
        for trace in list(self.long_term_memory.values()):
            days_old = (current_time - trace.timestamp).days
            if days_old > 0:
                trace.importance *= (self.decay_rate ** (days_old * 0.1))  # Slower decay for long-term
        
        # Remove very low importance memories
        self.short_term_memory = {
            mem_id: trace for mem_id, trace in self.short_term_memory.items()
            if trace.importance > 0.1
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get hippocampus status"""
        base_status = super().get_status()
        
        hippocampus_status = {
            'short_term_memories': len(self.short_term_memory),
            'long_term_memories': len(self.long_term_memory),
            'total_memories_stored': self.memories_stored,
            'total_memories_retrieved': self.memories_retrieved,
            'consolidations_performed': self.consolidations_performed,
            'consolidation_threshold': self.consolidation_threshold
        }
        
        base_status.update(hippocampus_status)
        return base_status
