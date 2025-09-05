#!/usr/bin/env python3
"""
FSOT Hardwiring System
=====================
Core hardwiring and neural architecture definitions for FSOT Neuromorphic AI.
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import logging

@dataclass
class NeuralConnection:
    """Represents a neural connection with FSOT properties."""
    connection_id: str
    source_node: str
    target_node: str
    weight: float
    fsot_scalar: float
    connection_type: str
    plasticity: float
    timestamp_created: float

@dataclass
class NeuralNode:
    """Represents a neural node in the FSOT system."""
    node_id: str
    node_type: str
    activation_level: float
    fsot_resonance: float
    connections_in: List[str]
    connections_out: List[str]
    processing_function: str
    memory_capacity: float

@dataclass
class HardwiredPattern:
    """Represents a hardwired neural pattern."""
    pattern_id: str
    pattern_type: str
    nodes: List[str]
    connections: List[str]
    activation_threshold: float
    stability: float
    purpose: str

class FSOTHardwiringSystem:
    """Advanced hardwiring system for FSOT neuromorphic architecture."""
    
    def __init__(self):
        self.nodes = {}
        self.connections = {}
        self.hardwired_patterns = {}
        self.activation_history = []
        self.fsot_global_state = 0.5
        self.logger = logging.getLogger(__name__)
        
        # Initialize core hardwired patterns
        self._initialize_core_patterns()
        
        print("⚡ FSOT Hardwiring System initialized")
    
    def _initialize_core_patterns(self):
        """Initialize essential hardwired patterns for FSOT consciousness."""
        
        # Core consciousness pattern
        self.create_hardwired_pattern(
            pattern_id="core_consciousness",
            pattern_type="consciousness_base",
            nodes=["consciousness_core", "awareness_monitor", "self_reflection"],
            purpose="Maintain basic consciousness awareness"
        )
        
        # Sensory processing pattern
        self.create_hardwired_pattern(
            pattern_id="sensory_processing",
            pattern_type="sensory_integration",
            nodes=["visual_cortex", "auditory_cortex", "sensory_fusion"],
            purpose="Integrate and process sensory information"
        )
        
        # Memory consolidation pattern
        self.create_hardwired_pattern(
            pattern_id="memory_consolidation",
            pattern_type="memory_system",
            nodes=["working_memory", "long_term_storage", "memory_retrieval"],
            purpose="Manage memory formation and retrieval"
        )
        
        # Decision making pattern
        self.create_hardwired_pattern(
            pattern_id="decision_making",
            pattern_type="cognitive_control",
            nodes=["option_evaluation", "decision_core", "action_planning"],
            purpose="Process decisions and plan actions"
        )
    
    def create_neural_node(self, node_id: str, node_type: str = "processing", 
                          activation_level: float = 0.0, fsot_resonance: float = 0.5) -> NeuralNode:
        """Create a new neural node with FSOT properties."""
        
        node = NeuralNode(
            node_id=node_id,
            node_type=node_type,
            activation_level=activation_level,
            fsot_resonance=fsot_resonance,
            connections_in=[],
            connections_out=[],
            processing_function=self._get_processing_function(node_type),
            memory_capacity=1.0
        )
        
        self.nodes[node_id] = node
        print(f"🧠 Created neural node: {node_id} (type: {node_type})")
        return node
    
    def create_neural_connection(self, source_id: str, target_id: str, 
                               weight: float = 0.5, fsot_scalar: float = 0.5,
                               connection_type: str = "excitatory") -> NeuralConnection:
        """Create a neural connection between nodes."""
        
        connection_id = f"{source_id}_to_{target_id}"
        
        connection = NeuralConnection(
            connection_id=connection_id,
            source_node=source_id,
            target_node=target_id,
            weight=weight,
            fsot_scalar=fsot_scalar,
            connection_type=connection_type,
            plasticity=0.1,
            timestamp_created=time.time()
        )
        
        self.connections[connection_id] = connection
        
        # Update node connection lists
        if source_id in self.nodes:
            self.nodes[source_id].connections_out.append(connection_id)
        if target_id in self.nodes:
            self.nodes[target_id].connections_in.append(connection_id)
        
        print(f"🔗 Created connection: {source_id} → {target_id} (weight: {weight:.3f})")
        return connection
    
    def create_hardwired_pattern(self, pattern_id: str, pattern_type: str,
                               nodes: List[str], purpose: str,
                               activation_threshold: float = 0.6) -> HardwiredPattern:
        """Create a hardwired neural pattern."""
        
        # Create nodes if they don't exist
        for node_id in nodes:
            if node_id not in self.nodes:
                self.create_neural_node(node_id, node_type=pattern_type)
        
        # Create connections between nodes in the pattern
        pattern_connections = []
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i != j:
                    connection_id = f"{source}_to_{target}"
                    if connection_id not in self.connections:
                        conn = self.create_neural_connection(
                            source, target, 
                            weight=0.7, 
                            fsot_scalar=0.8,
                            connection_type="pattern_internal"
                        )
                        pattern_connections.append(conn.connection_id)
        
        pattern = HardwiredPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            nodes=nodes,
            connections=pattern_connections,
            activation_threshold=activation_threshold,
            stability=0.9,
            purpose=purpose
        )
        
        self.hardwired_patterns[pattern_id] = pattern
        print(f"🏗️ Created hardwired pattern: {pattern_id}")
        return pattern
    
    def _get_processing_function(self, node_type: str) -> str:
        """Get appropriate processing function for node type."""
        function_map = {
            "consciousness_base": "consciousness_integration",
            "sensory_integration": "sensory_fusion",
            "memory_system": "memory_consolidation",
            "cognitive_control": "decision_processing",
            "processing": "standard_activation",
            "input": "input_relay",
            "output": "output_generation"
        }
        return function_map.get(node_type, "standard_activation")
    
    def activate_node(self, node_id: str, activation_strength: float = 1.0) -> float:
        """Activate a neural node and propagate through connections."""
        if node_id not in self.nodes:
            return 0.0
        
        node = self.nodes[node_id]
        
        # Calculate new activation based on current state and input
        old_activation = node.activation_level
        new_activation = self._calculate_activation(node, activation_strength)
        
        # Update node activation
        node.activation_level = new_activation
        
        # Record activation history
        self.activation_history.append({
            "timestamp": time.time(),
            "node_id": node_id,
            "old_activation": old_activation,
            "new_activation": new_activation,
            "input_strength": activation_strength
        })
        
        # Propagate activation through outgoing connections
        total_propagated = 0.0
        for connection_id in node.connections_out:
            if connection_id in self.connections:
                propagated = self._propagate_activation(connection_id, new_activation)
                total_propagated += propagated
        
        return new_activation
    
    def _calculate_activation(self, node: NeuralNode, input_strength: float) -> float:
        """Calculate new activation level for a node."""
        # FSOT-enhanced activation function
        fsot_enhancement = node.fsot_resonance * self.fsot_global_state
        
        # Sigmoid activation with FSOT modulation
        raw_activation = node.activation_level + input_strength * fsot_enhancement
        return 1.0 / (1.0 + np.exp(-raw_activation * 2.0 - 1.0))
    
    def _propagate_activation(self, connection_id: str, source_activation: float) -> float:
        """Propagate activation through a connection."""
        if connection_id not in self.connections:
            return 0.0
        
        connection = self.connections[connection_id]
        target_id = connection.target_node
        
        if target_id not in self.nodes:
            return 0.0
        
        # Calculate propagated signal
        signal_strength = source_activation * connection.weight * connection.fsot_scalar
        
        # Apply to target node
        if connection.connection_type == "excitatory":
            propagated = signal_strength
        elif connection.connection_type == "inhibitory":
            propagated = -signal_strength * 0.5
        else:
            propagated = signal_strength * 0.8
        
        # Activate target node
        self.activate_node(target_id, propagated)
        
        return abs(propagated)
    
    def activate_pattern(self, pattern_id: str, activation_strength: float = 1.0) -> Dict[str, float]:
        """Activate an entire hardwired pattern."""
        if pattern_id not in self.hardwired_patterns:
            return {}
        
        pattern = self.hardwired_patterns[pattern_id]
        activations = {}
        
        print(f"🔥 Activating pattern: {pattern_id}")
        
        # Activate all nodes in the pattern
        for node_id in pattern.nodes:
            activation = self.activate_node(node_id, activation_strength)
            activations[node_id] = activation
        
        # Update pattern stability based on activation success
        avg_activation = np.mean(list(activations.values()))
        if avg_activation > pattern.activation_threshold:
            pattern.stability = min(1.0, pattern.stability + 0.01)
        else:
            pattern.stability = max(0.0, pattern.stability - 0.005)
        
        return activations
    
    def update_fsot_global_state(self, new_state: float):
        """Update the global FSOT state affecting all neural processing."""
        self.fsot_global_state = max(0.0, min(1.0, new_state))
        
        # Update all node FSOT resonances based on global state
        for node in self.nodes.values():
            node.fsot_resonance = 0.5 * (node.fsot_resonance + self.fsot_global_state)
        
        print(f"🌊 Updated FSOT global state to: {self.fsot_global_state:.3f}")
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get comprehensive network state information."""
        node_activations = {nid: node.activation_level for nid, node in self.nodes.items()}
        pattern_states = {}
        
        for pid, pattern in self.hardwired_patterns.items():
            pattern_activation = np.mean([
                self.nodes[nid].activation_level for nid in pattern.nodes 
                if nid in self.nodes
            ])
            pattern_states[pid] = {
                "average_activation": pattern_activation,
                "stability": pattern.stability,
                "active": pattern_activation > pattern.activation_threshold
            }
        
        return {
            "fsot_global_state": self.fsot_global_state,
            "total_nodes": len(self.nodes),
            "total_connections": len(self.connections),
            "total_patterns": len(self.hardwired_patterns),
            "node_activations": node_activations,
            "pattern_states": pattern_states,
            "network_energy": np.sum(list(node_activations.values())),
            "timestamp": time.time()
        }
    
    def strengthen_connection(self, connection_id: str, strength_increase: float = 0.1):
        """Strengthen a neural connection through learning."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            old_weight = connection.weight
            connection.weight = min(1.0, connection.weight + strength_increase)
            
            print(f"💪 Strengthened connection {connection_id}: {old_weight:.3f} → {connection.weight:.3f}")
    
    def weaken_connection(self, connection_id: str, strength_decrease: float = 0.05):
        """Weaken a neural connection through disuse."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            old_weight = connection.weight
            connection.weight = max(0.0, connection.weight - strength_decrease)
            
            print(f"📉 Weakened connection {connection_id}: {old_weight:.3f} → {connection.weight:.3f}")
    
    def save_hardwiring_state(self, filename: Optional[str] = None) -> str:
        """Save complete hardwiring state to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"fsot_hardwiring_state_{timestamp}.json"
        
        state_data = {
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "connections": {cid: asdict(conn) for cid, conn in self.connections.items()},
            "patterns": {pid: asdict(pattern) for pid, pattern in self.hardwired_patterns.items()},
            "global_state": self.fsot_global_state,
            "activation_history": self.activation_history[-100:],  # Last 100 activations
            "timestamp": time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        print(f"💾 Hardwiring state saved to: {filename}")
        return filename

# Global hardwiring system instance
_hardwiring_system = None

def get_hardwiring_system():
    """Get global hardwiring system instance."""
    global _hardwiring_system
    if _hardwiring_system is None:
        _hardwiring_system = FSOTHardwiringSystem()
    return _hardwiring_system
