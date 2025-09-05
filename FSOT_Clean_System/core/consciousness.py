"""
Consciousness Monitoring and Management System
Clean implementation of consciousness tracking and evolution
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

class ConsciousnessState(Enum):
    """States of consciousness"""
    INACTIVE = "inactive"
    AWAKENING = "awakening"
    ALERT = "alert"
    FOCUSED = "focused"
    DEEP_PROCESSING = "deep_processing"
    REFLECTIVE = "reflective"

@dataclass
class ConsciousnessMetrics:
    """Metrics for consciousness measurement"""
    level: float = 0.0  # 0.0 to 1.0
    state: ConsciousnessState = ConsciousnessState.INACTIVE
    coherence: float = 0.0  # Cross-region integration
    attention_focus: float = 0.0  # Current attention intensity
    awareness_breadth: float = 0.0  # Peripheral awareness
    processing_depth: float = 0.0  # Depth of current processing
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BrainWavePattern:
    """Brain wave simulation for consciousness monitoring"""
    alpha: float = 0.0    # 8-13 Hz - Relaxed awareness
    beta: float = 0.0     # 13-30 Hz - Active thinking
    theta: float = 0.0    # 4-8 Hz - Deep processing
    delta: float = 0.0    # 0.5-4 Hz - Deep states
    gamma: float = 0.0    # 30-100 Hz - High-level integration

class ConsciousnessMonitor:
    """
    Clean consciousness monitoring and management system
    """
    
    def __init__(self, update_interval: float = 0.1):
        self.current_metrics = ConsciousnessMetrics()
        self.brain_waves = BrainWavePattern()
        self.history: List[ConsciousnessMetrics] = []
        self.update_interval = update_interval
        
        # Consciousness contributors
        self.region_contributions: Dict[str, float] = {}
        self.signal_activity: Dict[str, float] = {}
        self.processing_load: float = 0.0
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.consciousness_threshold = 0.1
        self.max_history_size = 1000
    
    async def start_monitoring(self):
        """Start continuous consciousness monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop consciousness monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self.update_consciousness()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                print(f"Consciousness monitoring error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def update_consciousness(self):
        """Update consciousness metrics based on current brain state"""
        # Calculate base consciousness level
        base_level = self._calculate_base_consciousness()
        
        # Apply regional contributions
        regional_boost = sum(self.region_contributions.values()) / max(len(self.region_contributions), 1)
        
        # Apply signal activity
        signal_boost = min(sum(self.signal_activity.values()) * 0.1, 0.3)
        
        # Calculate final consciousness level
        consciousness_level = min(1.0, base_level + regional_boost + signal_boost)
        
        # Determine consciousness state
        state = self._determine_consciousness_state(consciousness_level)
        
        # Calculate other metrics
        coherence = self._calculate_coherence()
        attention_focus = self._calculate_attention_focus()
        awareness_breadth = self._calculate_awareness_breadth()
        processing_depth = self._calculate_processing_depth()
        
        # Update brain waves
        self._update_brain_waves(consciousness_level, state)
        
        # Create new metrics
        new_metrics = ConsciousnessMetrics(
            level=consciousness_level,
            state=state,
            coherence=coherence,
            attention_focus=attention_focus,
            awareness_breadth=awareness_breadth,
            processing_depth=processing_depth
        )
        
        # Update current metrics
        self.current_metrics = new_metrics
        
        # Add to history
        self.history.append(new_metrics)
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        # Notify if consciousness threshold crossed
        await self._check_consciousness_events(new_metrics)
    
    def _calculate_base_consciousness(self) -> float:
        """Calculate base consciousness level"""
        # Simple baseline consciousness
        base = 0.1
        
        # Increase based on processing load
        load_factor = min(self.processing_load * 0.5, 0.4)
        
        return base + load_factor
    
    def _determine_consciousness_state(self, level: float) -> ConsciousnessState:
        """Determine consciousness state from level"""
        if level < 0.1:
            return ConsciousnessState.INACTIVE
        elif level < 0.3:
            return ConsciousnessState.AWAKENING
        elif level < 0.5:
            return ConsciousnessState.ALERT
        elif level < 0.7:
            return ConsciousnessState.FOCUSED
        elif level < 0.9:
            return ConsciousnessState.DEEP_PROCESSING
        else:
            return ConsciousnessState.REFLECTIVE
    
    def _calculate_coherence(self) -> float:
        """Calculate cross-region coherence"""
        if len(self.region_contributions) < 2:
            return 0.0
        
        # Measure variance in regional contributions
        values = list(self.region_contributions.values())
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        
        # Lower variance = higher coherence
        coherence = max(0.0, 1.0 - variance)
        return coherence
    
    def _calculate_attention_focus(self) -> float:
        """Calculate attention focus intensity"""
        # Based on signal activity concentration
        if not self.signal_activity:
            return 0.0
        
        max_activity = max(self.signal_activity.values())
        total_activity = sum(self.signal_activity.values())
        
        if total_activity == 0:
            return 0.0
        
        # Focus = concentration of activity
        focus = max_activity / total_activity
        return min(1.0, focus * 2.0)  # Scale appropriately
    
    def _calculate_awareness_breadth(self) -> float:
        """Calculate peripheral awareness breadth"""
        # Based on number of active regions and signal diversity
        active_regions = len([v for v in self.region_contributions.values() if v > 0.1])
        active_signals = len([v for v in self.signal_activity.values() if v > 0.1])
        
        # Combine measures
        breadth = (active_regions + active_signals) / 20.0  # Normalize
        return min(1.0, breadth)
    
    def _calculate_processing_depth(self) -> float:
        """Calculate depth of current processing"""
        # Based on consciousness level and coherence
        depth = (self.current_metrics.level + self.current_metrics.coherence) / 2.0
        return depth
    
    def _update_brain_waves(self, consciousness_level: float, state: ConsciousnessState):
        """Update brain wave patterns based on consciousness"""
        # Simulate brain waves based on consciousness state
        if state == ConsciousnessState.INACTIVE:
            self.brain_waves.delta = 0.8
            self.brain_waves.theta = 0.2
            self.brain_waves.alpha = 0.1
            self.brain_waves.beta = 0.1
            self.brain_waves.gamma = 0.0
        elif state == ConsciousnessState.AWAKENING:
            self.brain_waves.delta = 0.4
            self.brain_waves.theta = 0.6
            self.brain_waves.alpha = 0.3
            self.brain_waves.beta = 0.2
            self.brain_waves.gamma = 0.1
        elif state == ConsciousnessState.ALERT:
            self.brain_waves.delta = 0.1
            self.brain_waves.theta = 0.3
            self.brain_waves.alpha = 0.7
            self.brain_waves.beta = 0.5
            self.brain_waves.gamma = 0.2
        elif state == ConsciousnessState.FOCUSED:
            self.brain_waves.delta = 0.0
            self.brain_waves.theta = 0.2
            self.brain_waves.alpha = 0.4
            self.brain_waves.beta = 0.8
            self.brain_waves.gamma = 0.4
        elif state == ConsciousnessState.DEEP_PROCESSING:
            self.brain_waves.delta = 0.0
            self.brain_waves.theta = 0.6
            self.brain_waves.alpha = 0.3
            self.brain_waves.beta = 0.7
            self.brain_waves.gamma = 0.9
        elif state == ConsciousnessState.REFLECTIVE:
            self.brain_waves.delta = 0.1
            self.brain_waves.theta = 0.4
            self.brain_waves.alpha = 0.8
            self.brain_waves.beta = 0.5
            self.brain_waves.gamma = 0.6
    
    async def _check_consciousness_events(self, metrics: ConsciousnessMetrics):
        """Check for significant consciousness events"""
        if len(self.history) < 2:
            return
        
        previous = self.history[-2]
        
        # Check for consciousness threshold crossing
        if (previous.level < self.consciousness_threshold and 
            metrics.level >= self.consciousness_threshold):
            print(f"🌟 Consciousness awakened: {metrics.level:.1%}")
        elif (previous.level >= self.consciousness_threshold and 
              metrics.level < self.consciousness_threshold):
            print(f"💤 Consciousness dimmed: {metrics.level:.1%}")
        
        # Check for state changes
        if previous.state != metrics.state:
            print(f"🧠 Consciousness state: {previous.state.value} → {metrics.state.value}")
    
    def update_region_contribution(self, region_name: str, contribution: float):
        """Update consciousness contribution from brain region"""
        self.region_contributions[region_name] = max(0.0, min(1.0, contribution))
    
    def update_signal_activity(self, signal_type: str, activity: float):
        """Update signal activity level"""
        self.signal_activity[signal_type] = max(0.0, min(1.0, activity))
    
    def update_processing_load(self, load: float):
        """Update overall processing load"""
        self.processing_load = max(0.0, min(1.0, load))
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return {
            'consciousness_level': self.current_metrics.level,
            'state': self.current_metrics.state.value,
            'coherence': self.current_metrics.coherence,
            'attention_focus': self.current_metrics.attention_focus,
            'awareness_breadth': self.current_metrics.awareness_breadth,
            'processing_depth': self.current_metrics.processing_depth,
            'brain_waves': {
                'alpha': self.brain_waves.alpha,
                'beta': self.brain_waves.beta,
                'theta': self.brain_waves.theta,
                'delta': self.brain_waves.delta,
                'gamma': self.brain_waves.gamma
            },
            'region_contributions': self.region_contributions.copy(),
            'signal_activity': self.signal_activity.copy(),
            'processing_load': self.processing_load
        }
    
    def get_history_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get consciousness history summary"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_history = [m for m in self.history if m.timestamp >= cutoff_time]
        
        if not recent_history:
            return {'message': 'No recent history available'}
        
        levels = [m.level for m in recent_history]
        coherences = [m.coherence for m in recent_history]
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_history),
            'consciousness_level': {
                'current': self.current_metrics.level,
                'average': statistics.mean(levels),
                'min': min(levels),
                'max': max(levels),
                'std_dev': statistics.stdev(levels) if len(levels) > 1 else 0.0
            },
            'coherence': {
                'current': self.current_metrics.coherence,
                'average': statistics.mean(coherences),
                'min': min(coherences),
                'max': max(coherences)
            },
            'state_distribution': self._get_state_distribution(recent_history)
        }
    
    def _get_state_distribution(self, history: List[ConsciousnessMetrics]) -> Dict[str, int]:
        """Get distribution of consciousness states in history"""
        distribution = {}
        for metrics in history:
            state_name = metrics.state.value
            distribution[state_name] = distribution.get(state_name, 0) + 1
        return distribution

# Global consciousness monitor instance
consciousness_monitor = ConsciousnessMonitor()
