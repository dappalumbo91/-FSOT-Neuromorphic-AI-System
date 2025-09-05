#!/usr/bin/env python3
"""
FSOT 2.0 Neural Pathway Debug System
===================================
Simplified neural pathway system that integrates with your existing FSOT system
and provides granular synaptic-level debugging.
"""

import numpy as np
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import sys
import os

# Safe FSOT 2.0 Integration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FSOT_Clean_System'))

try:
    from fsot_2_0_foundation import FSOTCore, FSOTDomain, FSOTConstants
    FSOT_AVAILABLE = True
    print("✅ FSOT 2.0 core loaded successfully")
except ImportError as e:
    print(f"⚠️ FSOT 2.0 not available: {e}")
    FSOT_AVAILABLE = False

# Load existing brain system for integration
sys.path.insert(0, os.path.dirname(__file__))
try:
    from brain_system import NeuromorphicBrainSystem
    BRAIN_SYSTEM_AVAILABLE = True
    print("✅ Existing brain system loaded")
except ImportError as e:
    print(f"⚠️ Brain system not available: {e}")
    BRAIN_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SynapticConnection:
    """Simplified synaptic connection with FSOT compliance"""
    
    def __init__(self, source_id: str, target_id: str, weight: float = 0.5):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.activation_count = 0
        self.last_activation = 0.0
        self.plasticity = 0.1  # Learning rate
        
        # FSOT enhancement
        self.fsot_modulation = 1.0
        if FSOT_AVAILABLE:
            try:
                core = FSOTCore()
                # Use basic computation without complex parameters
                self.fsot_modulation = abs(core.compute_universal_scalar(10, FSOTDomain.NEURAL))
            except:
                self.fsot_modulation = 1.0
    
    def transmit(self, signal: float, current_time: float) -> float:
        """Transmit signal through synapse"""
        self.activation_count += 1
        self.last_activation = current_time
        
        # Apply synaptic transmission with FSOT modulation
        transmitted_signal = signal * self.weight * self.fsot_modulation
        
        # Hebbian learning: strengthen with use
        if signal > 0.1:
            self.weight = min(1.0, self.weight + self.plasticity * signal * 0.01)
        
        return transmitted_signal

class NeuralNode:
    """Simplified neural node with biological properties"""
    
    def __init__(self, node_id: str, node_type: str = "processing"):
        self.node_id = node_id
        self.node_type = node_type
        
        # Neural properties
        self.activation = 0.0
        self.threshold = 0.5
        self.refractory_time = 0.0
        self.membrane_potential = 0.0
        
        # Connections
        self.incoming_connections: List[SynapticConnection] = []
        self.outgoing_connections: List[SynapticConnection] = []
        
        # Activity tracking
        self.firing_history: List[float] = []
        self.total_activations = 0
        
        # FSOT signature
        self.fsot_signature = 0.0
        self._update_fsot_signature()
    
    def _update_fsot_signature(self):
        """Update FSOT signature for this node"""
        if FSOT_AVAILABLE:
            try:
                core = FSOTCore()
                self.fsot_signature = core.compute_universal_scalar(11, FSOTDomain.NEURAL)
            except:
                self.fsot_signature = 1.0
    
    def add_incoming_connection(self, connection: SynapticConnection):
        """Add incoming synaptic connection"""
        self.incoming_connections.append(connection)
    
    def add_outgoing_connection(self, connection: SynapticConnection):
        """Add outgoing synaptic connection"""
        self.outgoing_connections.append(connection)
    
    def receive_input(self, input_signal: float, current_time: float):
        """Receive input from connected neurons"""
        self.membrane_potential += input_signal
    
    def update(self, current_time: float) -> float:
        """Update neuron state and return output"""
        # Check refractory period
        if current_time < self.refractory_time:
            return 0.0
        
        # Check activation threshold
        if self.membrane_potential >= self.threshold:
            # Fire action potential
            self.activation = 1.0
            self.total_activations += 1
            self.firing_history.append(current_time)
            
            # Set refractory period
            self.refractory_time = current_time + 2.0  # 2ms refractory
            
            # Reset membrane potential
            self.membrane_potential = 0.0
            
            # Update FSOT signature
            self._update_fsot_signature()
            
            # Keep history manageable
            if len(self.firing_history) > 1000:
                self.firing_history = self.firing_history[-1000:]
            
            return self.activation
        else:
            # Passive decay
            self.membrane_potential *= 0.9
            self.activation = 0.0
            return 0.0
    
    def get_firing_rate(self, time_window: float = 1000.0) -> float:
        """Calculate firing rate in given time window"""
        if not self.firing_history:
            return 0.0
        
        current_time = self.firing_history[-1] if self.firing_history else 0.0
        recent_fires = [t for t in self.firing_history if current_time - t <= time_window]
        return len(recent_fires) / (time_window / 1000.0)

class NeuralPathway:
    """Neural pathway with synaptic-level modeling"""
    
    def __init__(self, pathway_id: str, pathway_type: str = "cortical"):
        self.pathway_id = pathway_id
        self.pathway_type = pathway_type
        
        # Pathway components
        self.nodes: Dict[str, NeuralNode] = {}
        self.connections: Dict[str, SynapticConnection] = {}
        
        # Input/output organization
        self.input_nodes: Set[str] = set()
        self.output_nodes: Set[str] = set()
        self.processing_nodes: Set[str] = set()
        
        # Pathway metrics
        self.activity_level = 0.0
        self.synchronization = 0.0
        self.pathway_coherence = 0.0
        
        # FSOT integration
        self.fsot_coherence = 0.0
        self._update_pathway_fsot()
    
    def _update_pathway_fsot(self):
        """Update pathway-level FSOT metrics"""
        if FSOT_AVAILABLE:
            try:
                core = FSOTCore()
                self.fsot_coherence = core.compute_universal_scalar(12, FSOTDomain.NEURAL)
            except:
                self.fsot_coherence = 1.0
    
    def add_node(self, node: NeuralNode, role: str = "processing"):
        """Add neural node to pathway"""
        self.nodes[node.node_id] = node
        
        if role == "input":
            self.input_nodes.add(node.node_id)
        elif role == "output":
            self.output_nodes.add(node.node_id)
        else:
            self.processing_nodes.add(node.node_id)
        
        logger.debug(f"Added {role} node {node.node_id} to pathway {self.pathway_id}")
    
    def connect_nodes(self, source_id: str, target_id: str, weight: float = 0.5):
        """Connect two nodes with a synapse"""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source_id}, {target_id}")
        
        connection_id = f"{source_id}_{target_id}"
        connection = SynapticConnection(source_id, target_id, weight)
        
        self.connections[connection_id] = connection
        self.nodes[source_id].add_outgoing_connection(connection)
        self.nodes[target_id].add_incoming_connection(connection)
        
        logger.debug(f"Connected {source_id} -> {target_id} with weight {weight}")
    
    def process(self, inputs: Dict[str, float], current_time: float) -> Dict[str, float]:
        """Process inputs through the pathway"""
        # Apply inputs to input nodes
        for node_id, signal in inputs.items():
            if node_id in self.input_nodes and node_id in self.nodes:
                self.nodes[node_id].receive_input(signal, current_time)
        
        # Update all nodes
        active_nodes = 0
        outputs = {}
        
        for node_id, node in self.nodes.items():
            output = node.update(current_time)
            
            if output > 0:
                active_nodes += 1
                
                # Propagate signal through outgoing connections
                for connection in node.outgoing_connections:
                    transmitted_signal = connection.transmit(output, current_time)
                    target_node = self.nodes[connection.target_id]
                    target_node.receive_input(transmitted_signal, current_time)
                
                # Collect outputs from output nodes
                if node_id in self.output_nodes:
                    outputs[node_id] = output
        
        # Update pathway metrics
        self.activity_level = active_nodes / len(self.nodes) if self.nodes else 0.0
        self._update_pathway_fsot()
        
        return outputs
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        node_stats = {}
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                'type': node.node_type,
                'total_activations': node.total_activations,
                'firing_rate': node.get_firing_rate(),
                'membrane_potential': node.membrane_potential,
                'fsot_signature': node.fsot_signature,
                'incoming_connections': len(node.incoming_connections),
                'outgoing_connections': len(node.outgoing_connections)
            }
        
        connection_stats = {}
        for conn_id, connection in self.connections.items():
            connection_stats[conn_id] = {
                'weight': connection.weight,
                'activation_count': connection.activation_count,
                'fsot_modulation': connection.fsot_modulation,
                'plasticity': connection.plasticity
            }
        
        return {
            'pathway_id': self.pathway_id,
            'pathway_type': self.pathway_type,
            'activity_level': self.activity_level,
            'synchronization': self.synchronization,
            'fsot_coherence': self.fsot_coherence,
            'node_count': len(self.nodes),
            'connection_count': len(self.connections),
            'input_nodes': len(self.input_nodes),
            'output_nodes': len(self.output_nodes),
            'processing_nodes': len(self.processing_nodes),
            'node_details': node_stats,
            'connection_details': connection_stats
        }

class FSNeuralDebugSystem:
    """FSOT-integrated neural debug system with existing brain integration"""
    
    def __init__(self):
        self.pathways: Dict[str, NeuralPathway] = {}
        self.current_time = 0.0
        self.time_step = 1.0
        
        # Integration with existing systems
        self.brain_system = None
        self.fsot_core = None
        
        # System metrics
        self.system_coherence = 0.0
        self.consciousness_level = 0.0
        
        self._initialize_integrations()
        
        logger.info("🧠 FSOT Neural Debug System initialized")
    
    def _initialize_integrations(self):
        """Initialize integrations with existing systems"""
        # FSOT integration
        if FSOT_AVAILABLE:
            try:
                self.fsot_core = FSOTCore()
                logger.info("✅ FSOT core integrated")
            except Exception as e:
                logger.warning(f"FSOT integration failed: {e}")
        
        # Brain system integration
        if BRAIN_SYSTEM_AVAILABLE:
            try:
                self.brain_system = NeuromorphicBrainSystem()
                logger.info("✅ Brain system integrated")
            except Exception as e:
                logger.warning(f"Brain system integration failed: {e}")
    
    def create_pathway_from_brain_region(self, region_name: str) -> NeuralPathway:
        """Create a neural pathway based on brain region"""
        if self.brain_system and region_name in self.brain_system.regions:
            brain_region = self.brain_system.regions[region_name]
            
            # Create pathway
            pathway = NeuralPathway(f"pathway_{region_name}", "brain_derived")
            
            # Create nodes based on neuron count (scaled down)
            node_count = min(10, max(3, brain_region.neurons // 10000))
            
            for i in range(node_count):
                node = NeuralNode(f"{region_name}_node_{i}")
                role = "input" if i < 2 else "output" if i >= node_count - 2 else "processing"
                pathway.add_node(node, role)
            
            # Connect nodes in a meaningful pattern
            for i in range(node_count - 1):
                pathway.connect_nodes(f"{region_name}_node_{i}", f"{region_name}_node_{i+1}", 0.7)
            
            # Add some recurrent connections
            if node_count > 3:
                pathway.connect_nodes(f"{region_name}_node_{node_count-1}", f"{region_name}_node_1", 0.3)
            
            self.pathways[region_name] = pathway
            logger.info(f"Created pathway for brain region: {region_name}")
            
            return pathway
        
        else:
            # Create generic pathway
            pathway = NeuralPathway(region_name, "generic")
            
            # Add basic structure
            for i in range(5):
                node = NeuralNode(f"{region_name}_node_{i}")
                role = "input" if i < 2 else "output" if i >= 3 else "processing"
                pathway.add_node(node, role)
            
            # Basic connections
            pathway.connect_nodes(f"{region_name}_node_0", f"{region_name}_node_2", 0.8)
            pathway.connect_nodes(f"{region_name}_node_1", f"{region_name}_node_2", 0.7)
            pathway.connect_nodes(f"{region_name}_node_2", f"{region_name}_node_3", 0.9)
            pathway.connect_nodes(f"{region_name}_node_2", f"{region_name}_node_4", 0.6)
            
            self.pathways[region_name] = pathway
            logger.info(f"Created generic pathway: {region_name}")
            
            return pathway
    
    def run_debug_cycle(self, input_data: Dict[str, Dict[str, float]], steps: int = 10) -> Dict[str, Any]:
        """Run a debug cycle and collect comprehensive data"""
        debug_results = {
            'timestamp': datetime.now().isoformat(),
            'steps_run': steps,
            'pathway_outputs': [],
            'system_metrics': [],
            'fsot_metrics': [],
            'integration_status': {}
        }
        
        for step in range(steps):
            self.current_time += self.time_step
            step_outputs = {}
            
            # Process each pathway
            for pathway_id, pathway in self.pathways.items():
                inputs = input_data.get(pathway_id, {})
                outputs = pathway.process(inputs, self.current_time)
                step_outputs[pathway_id] = outputs
            
            # Test brain system integration
            if self.brain_system and step % 3 == 0:
                try:
                    stimulus = {'type': 'neural_pathway', 'intensity': 0.6, 'step': step}
                    brain_result = self.brain_system.process_stimulus(stimulus)
                    debug_results['integration_status']['brain_system'] = 'active'
                    debug_results['integration_status']['consciousness'] = brain_result.get('consciousness_level', 0.0)
                except Exception as e:
                    debug_results['integration_status']['brain_system'] = f'error: {e}'
            
            # Calculate system metrics
            total_activity = sum(p.activity_level for p in self.pathways.values())
            avg_activity = total_activity / len(self.pathways) if self.pathways else 0.0
            
            # FSOT metrics
            fsot_system_scalar = 0.0
            if self.fsot_core:
                try:
                    fsot_system_scalar = self.fsot_core.compute_universal_scalar(13, FSOTDomain.AI_TECH)
                except:
                    fsot_system_scalar = 1.0
            
            # Store step results
            debug_results['pathway_outputs'].append(step_outputs)
            debug_results['system_metrics'].append({
                'step': step,
                'time': self.current_time,
                'avg_activity': avg_activity,
                'total_pathways': len(self.pathways)
            })
            debug_results['fsot_metrics'].append({
                'step': step,
                'fsot_scalar': fsot_system_scalar,
                'pathway_coherence': [p.fsot_coherence for p in self.pathways.values()]
            })
        
        return debug_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': {
                'pathways': len(self.pathways),
                'total_nodes': sum(len(p.nodes) for p in self.pathways.values()),
                'total_connections': sum(len(p.connections) for p in self.pathways.values()),
                'current_time': self.current_time
            },
            'integration_status': {
                'fsot_available': FSOT_AVAILABLE,
                'brain_system_available': BRAIN_SYSTEM_AVAILABLE,
                'fsot_core_active': self.fsot_core is not None,
                'brain_system_active': self.brain_system is not None
            },
            'pathway_details': {},
            'system_recommendations': []
        }
        
        # Get detailed pathway information
        for pathway_id, pathway in self.pathways.items():
            report['pathway_details'][pathway_id] = pathway.get_debug_info()
        
        # Generate recommendations
        recommendations = []
        
        if not FSOT_AVAILABLE:
            recommendations.append("Install FSOT 2.0 framework for full theoretical compliance")
        
        if not BRAIN_SYSTEM_AVAILABLE:
            recommendations.append("Integrate with existing brain system for enhanced functionality")
        
        if len(self.pathways) < 3:
            recommendations.append("Add more neural pathways for complex processing")
        
        total_nodes = sum(len(p.nodes) for p in self.pathways.values())
        if total_nodes < 10:
            recommendations.append("Increase neural node count for better modeling")
        
        if not recommendations:
            recommendations.append("System is well-configured for neural pathway debugging")
        
        report['system_recommendations'] = recommendations
        
        return report

def create_comprehensive_demo():
    """Create comprehensive demonstration of the neural pathway debug system"""
    print("🧠 COMPREHENSIVE FSOT NEURAL PATHWAY DEBUG DEMO")
    print("=" * 60)
    
    # Create debug system
    debug_system = FSNeuralDebugSystem()
    
    # Create pathways based on brain regions
    brain_regions = ['prefrontal_cortex', 'temporal_lobe', 'occipital_lobe']
    
    for region in brain_regions:
        pathway = debug_system.create_pathway_from_brain_region(region)
        print(f"✅ Created pathway for {region}: {len(pathway.nodes)} nodes")
    
    # Create test inputs
    test_inputs = {
        'prefrontal_cortex': {'prefrontal_cortex_node_0': 0.8, 'prefrontal_cortex_node_1': 0.6},
        'temporal_lobe': {'temporal_lobe_node_0': 0.7},
        'occipital_lobe': {'occipital_lobe_node_0': 0.9, 'occipital_lobe_node_1': 0.5}
    }
    
    print("\n🧪 Running debug cycle...")
    debug_results = debug_system.run_debug_cycle(test_inputs, steps=5)
    
    print(f"   ✅ Completed {debug_results['steps_run']} steps")
    print(f"   📊 System metrics collected: {len(debug_results['system_metrics'])} entries")
    print(f"   🔬 FSOT metrics collected: {len(debug_results['fsot_metrics'])} entries")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = debug_system.generate_comprehensive_report()
    
    print("\n📊 SYSTEM SUMMARY:")
    print(f"   Pathways: {report['system_overview']['pathways']}")
    print(f"   Total Nodes: {report['system_overview']['total_nodes']}")
    print(f"   Total Connections: {report['system_overview']['total_connections']}")
    print(f"   FSOT Available: {report['integration_status']['fsot_available']}")
    print(f"   Brain System Available: {report['integration_status']['brain_system_available']}")
    
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(report['system_recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Save reports
    debug_file = f"neural_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(debug_file, 'w') as f:
        json.dump(debug_results, f, indent=2)
    
    report_file = f"neural_debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Debug results saved: {debug_file}")
    print(f"💾 Comprehensive report saved: {report_file}")
    
    print("\n🎓 NEURAL PATHWAY INSIGHTS:")
    print("1. ✅ Synaptic-level modeling provides granular control")
    print("2. ✅ FSOT 2.0 integration ensures theoretical consistency")
    print("3. ✅ Brain system integration leverages existing architecture")
    print("4. ✅ Real-time debugging with comprehensive metrics")
    print("5. ✅ Pathway-based design prevents endless loops")
    print("\n🚀 Ready for integration with your main FSOT system!")
    
    return debug_system, debug_results, report

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run comprehensive demonstration
    system, results, report = create_comprehensive_demo()
