"""
FSOT 2.0 Adaptive Memory Manager
Clean implementation adapted from the original complex system
"""

import psutil
import gc
import threading
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class MemoryThresholds:
    """Memory usage thresholds for system states"""
    safe: float = 0.60      # 60% - Normal operation
    warning: float = 0.75   # 75% - Start optimization
    critical: float = 0.85  # 85% - Aggressive optimization
    emergency: float = 0.95 # 95% - Emergency measures

@dataclass
class MemoryStats:
    """Current memory statistics"""
    total_gb: float
    available_gb: float
    used_gb: float
    percentage: float
    status: str  # 'safe', 'warning', 'critical', 'emergency'

class MemoryOptimizer(ABC):
    """Abstract base class for memory optimization strategies"""
    
    @abstractmethod
    async def optimize(self) -> bool:
        """Perform optimization. Returns True if successful."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get optimization priority (lower = higher priority)"""
        pass

class GarbageCollectionOptimizer(MemoryOptimizer):
    """Garbage collection optimization strategy"""
    
    async def optimize(self) -> bool:
        """Perform garbage collection"""
        try:
            collected = gc.collect()
            logger.info(f"🗑️ Garbage collection freed {collected} objects")
            return True
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return False
    
    def get_priority(self) -> int:
        return 1  # Highest priority - always safe

class AdaptiveMemoryManager:
    """
    Clean adaptive memory management system for FSOT 2.0
    Monitors memory usage and applies optimization strategies when needed
    """
    
    def __init__(self):
        self.thresholds = MemoryThresholds()
        self.optimizers: list[MemoryOptimizer] = [
            GarbageCollectionOptimizer()
        ]
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.check_interval = 30.0  # Check every 30 seconds
        
        # Statistics
        self.optimization_count = 0
        self.last_optimization = None
        
        logger.info("🧠 FSOT Memory Manager initialized")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            memory = psutil.virtual_memory()
            
            # Determine status based on thresholds
            percentage = memory.percent / 100.0
            if percentage < self.thresholds.safe:
                status = 'safe'
            elif percentage < self.thresholds.warning:
                status = 'warning'
            elif percentage < self.thresholds.critical:
                status = 'critical'
            else:
                status = 'emergency'
            
            return MemoryStats(
                total_gb=memory.total / (1024**3),
                available_gb=memory.available / (1024**3),
                used_gb=memory.used / (1024**3),
                percentage=memory.percent,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            # Return safe defaults
            return MemoryStats(
                total_gb=8.0,
                available_gb=4.0,
                used_gb=4.0,
                percentage=50.0,
                status='safe'
            )
    
    def add_optimizer(self, optimizer: MemoryOptimizer):
        """Add a memory optimization strategy"""
        self.optimizers.append(optimizer)
        # Sort by priority
        self.optimizers.sort(key=lambda x: x.get_priority())
        logger.info(f"Added memory optimizer: {optimizer.__class__.__name__}")
    
    async def check_and_optimize(self) -> bool:
        """Check memory usage and optimize if needed"""
        stats = self.get_memory_stats()
        
        # Log memory status periodically
        if stats.status != 'safe':
            logger.info(f"💾 Memory: {stats.used_gb:.1f}GB/{stats.total_gb:.1f}GB ({stats.percentage:.1f}%) - {stats.status.upper()}")
        
        # Apply optimizations if needed
        if stats.status in ['warning', 'critical', 'emergency']:
            return await self._apply_optimizations(stats)
        
        return True
    
    async def _apply_optimizations(self, stats: MemoryStats) -> bool:
        """Apply memory optimizations based on current status"""
        logger.info(f"🔧 Applying memory optimizations for {stats.status} status")
        
        success_count = 0
        
        for optimizer in self.optimizers:
            try:
                if await optimizer.optimize():
                    success_count += 1
                    
                # Check if we've improved enough
                new_stats = self.get_memory_stats()
                if new_stats.status == 'safe':
                    logger.info("✅ Memory optimization successful - returned to safe levels")
                    break
                    
            except Exception as e:
                logger.error(f"Optimization {optimizer.__class__.__name__} failed: {e}")
        
        self.optimization_count += 1
        self.last_optimization = datetime.now()
        
        return success_count > 0
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="FSoTMemoryMonitor"
        )
        self.monitoring_thread.start()
        logger.info("📈 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("🛑 Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Use asyncio.run for the async method in the sync thread
                import asyncio
                asyncio.run(self.check_and_optimize())
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            # Wait for next check
            for _ in range(int(self.check_interval)):
                if not self.monitoring_active:
                    break
                threading.Event().wait(1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status"""
        stats = self.get_memory_stats()
        
        return {
            'memory_stats': {
                'total_gb': stats.total_gb,
                'available_gb': stats.available_gb,
                'used_gb': stats.used_gb,
                'percentage': stats.percentage,
                'status': stats.status
            },
            'monitoring_active': self.monitoring_active,
            'optimization_count': self.optimization_count,
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
            'optimizers_count': len(self.optimizers)
        }
    
    async def force_optimization(self) -> bool:
        """Force memory optimization regardless of current status"""
        logger.info("🔧 Forcing memory optimization")
        stats = self.get_memory_stats()
        return await self._apply_optimizations(stats)

# Global memory manager instance
memory_manager = AdaptiveMemoryManager()
