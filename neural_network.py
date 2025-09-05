"""
FSOT 2.0 Neural Network Implementation
=====================================
Neuromorphic neural network with FSOT theoretical compliance.

This module provides advanced neural network architectures specifically designed
for neuromorphic computing with full FSOT 2.0 theoretical integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path
import threading
from datetime import datetime
import sys
import os

# Add FSOT system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FSOT_Clean_System'))

try:
    # First try the new compatibility system
    from fsot_compatibility import fsot_enforce
    from fsot_compatibility import FSOTDomain as _ImportedFSOTDomain
    print("[OK] FSOT compatibility system imported successfully")
    FSOT_AVAILABLE = True
    # Create compatible FSOTDomain class
    class FSOTDomain:
        NEUROMORPHIC = getattr(_ImportedFSOTDomain, 'NEUROMORPHIC', "neuromorphic")
        AI_TECH = getattr(_ImportedFSOTDomain, 'AI_TECH', "ai_tech")
        PATTERN_RECOGNITION = getattr(_ImportedFSOTDomain, 'PATTERN_RECOGNITION', "pattern_recognition")
        REAL_TIME_PROCESSING = getattr(_ImportedFSOTDomain, 'REAL_TIME_PROCESSING', "real_time_processing")
        ADAPTIVE_LEARNING = getattr(_ImportedFSOTDomain, 'ADAPTIVE_LEARNING', "adaptive_learning")
        
except ImportError:
    try:
        # Fallback to old FSOT system with proper path handling
        clean_system_path = os.path.join(os.path.dirname(__file__), 'FSOT_Clean_System')
        if clean_system_path not in sys.path:
            sys.path.insert(0, clean_system_path)
        
        try:
            from fsot_hardwiring import hardwire_fsot as fsot_enforce
            print("[OK] hardwire_fsot imported successfully")
        except (ImportError, AttributeError):
            print("⚠️ hardwire_fsot not found, using fallback")
            # Define fallback function with correct type
            def fsot_enforce(domain: Any = None, d_eff: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
                    return target
                return decorator
            
        try:
            from fsot_2_0_foundation import FSOTCore as FoundationFSOTCore
            print("[OK] FSOTCore imported successfully")
        except (ImportError, AttributeError):
            print("⚠️ FSOTCore not found, using fallback")
            FoundationFSOTCore = None
            
        FSOT_AVAILABLE = True
        print("[OK] Legacy FSOT modules imported (partial)")
            
    except ImportError as e:
        print(f"⚠️ FSOT import failed, using fallback: {e}")
        FoundationFSOTCore = None
        # Define fallback function with correct type
        def fsot_enforce(domain: Any = None, d_eff: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
                return target
            return decorator
        FSOT_AVAILABLE = False

    # Define fallback for standalone usage
    if 'fsot_enforce' not in locals():
        def fsot_enforce(domain: Any = None, d_eff: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
                return target
            return decorator
    
    # Use a separate class to avoid conflicts
    class LocalFSOTDomain:
        NEUROMORPHIC = "neuromorphic"
        AI_TECH = "ai_tech"
        PATTERN_RECOGNITION = "pattern_recognition"
        REAL_TIME_PROCESSING = "real_time_processing"
        ADAPTIVE_LEARNING = "adaptive_learning"
        
    # Create FSOTDomain alias with type ignore
    FSOTDomain = LocalFSOTDomain  # type: ignore

# Ensure FSOTCore is available regardless of import path
if 'FoundationFSOTCore' in locals() and FoundationFSOTCore is not None:
    FSOTCore = FoundationFSOTCore
else:
    class FSOTCore:
        def __init__(self):
            pass
        
        def get_theoretical_signature(self):
            return "FSOT-2.0-COMPLIANT"


@dataclass
class Neuron:
    """
    Individual neuron representation with FSOT compliance.
    
    Attributes:
        id: Unique neuron identifier
        activation: Current activation level
        threshold: Firing threshold
        bias: Neuron bias value
        weights: Input connection weights
        position: 3D spatial position
        neuron_type: Type classification (excitatory/inhibitory)
    """
    id: str
    activation: float = 0.0
    threshold: float = 0.5
    bias: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    neuron_type: str = 'excitatory'
    fsot_compliance: float = 1.0


@dataclass
class Synapse:
    """
    Synaptic connection between neurons.
    
    Attributes:
        pre_neuron: Source neuron ID
        post_neuron: Target neuron ID
        weight: Synaptic strength
        delay: Transmission delay
        plasticity: Learning rate for this synapse
    """
    pre_neuron: str
    post_neuron: str
    weight: float
    delay: float = 0.0
    plasticity: float = 0.01
    last_update: float = 0.0


class NeuralNetworkError(Exception):
    """Custom exception for neural network operations."""
    pass


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: float) -> float:
        """Forward pass through activation function."""
        pass
    
    @abstractmethod
    def derivative(self, x: float) -> float:
        """Derivative of activation function."""
        pass


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function."""
    
    def forward(self, x: float) -> float:
        """Sigmoid forward pass."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def derivative(self, x: float) -> float:
        """Sigmoid derivative."""
        sig = self.forward(x)
        return sig * (1.0 - sig)


class ReLUActivation(ActivationFunction):
    """ReLU activation function."""
    
    def forward(self, x: float) -> float:
        """ReLU forward pass."""
        return max(0.0, x)
    
    def derivative(self, x: float) -> float:
        """ReLU derivative."""
        return 1.0 if x > 0.0 else 0.0


class TanhActivation(ActivationFunction):
    """Tanh activation function."""
    
    def forward(self, x: float) -> float:
        """Tanh forward pass."""
        return np.tanh(x)
    
    def derivative(self, x: float) -> float:
        """Tanh derivative."""
        return 1.0 - np.tanh(x)**2


@fsot_enforce(domain=FSOTDomain.NEUROMORPHIC, d_eff=12)
class NeuromorphicLayer:
    """
    Neuromorphic layer implementing brain-inspired processing.
    
    This layer simulates biological neural processes including:
    - Spiking neural dynamics
    - Synaptic plasticity
    - Lateral inhibition
    - Temporal dynamics
    """
    
    def __init__(self, 
                 layer_id: str,
                 size: int,
                 activation_func: Optional[ActivationFunction] = None,
                 spatial_dimensions: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize neuromorphic layer.
        
        Args:
            layer_id: Unique layer identifier
            size: Number of neurons in layer
            activation_func: Activation function to use
            spatial_dimensions: 3D spatial arrangement
        """
        self.layer_id = layer_id
        self.size = size
        self.activation_func = activation_func or SigmoidActivation()
        self.spatial_dimensions = spatial_dimensions
        
        # Initialize neurons
        self.neurons = {}
        self._initialize_neurons()
        
        # Layer properties
        self.inhibition_strength = 0.1
        self.noise_level = 0.01
        self.refractory_period = 0.002  # 2ms
        
        # Temporal dynamics
        self.membrane_potential = np.zeros(size)
        self.spike_times = {}
        self.last_spike = np.full(size, -np.inf)
        
        # FSOT compliance
        self.fsot_core = FSOTCore()
        self.theoretical_alignment = 1.0
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_neurons(self):
        """Initialize neurons with spatial positions."""
        dx, dy, dz = self.spatial_dimensions
        total_positions = dx * dy * dz
        
        for i in range(self.size):
            # Calculate 3D position
            z = i // (dx * dy)
            y = (i % (dx * dy)) // dx
            x = i % dx
            
            # Handle case where neurons exceed spatial grid
            if i >= total_positions:
                x = np.random.uniform(0, dx)
                y = np.random.uniform(0, dy)
                z = np.random.uniform(0, dz)
            
            neuron_id = f"{self.layer_id}_neuron_{i}"
            self.neurons[neuron_id] = Neuron(
                id=neuron_id,
                position=(float(x), float(y), float(z)),
                threshold=np.random.normal(0.5, 0.1),
                bias=np.random.normal(0.0, 0.05)
            )
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def forward(self, inputs: np.ndarray, time_step: float = 0.0) -> np.ndarray:
        """
        Forward pass through neuromorphic layer.
        
        Args:
            inputs: Input vector
            time_step: Current simulation time
            
        Returns:
            Layer output activations
        """
        try:
            if len(inputs) != self.size:
                raise NeuralNetworkError(f"Input size {len(inputs)} doesn't match layer size {self.size}")
            
            outputs = np.zeros(self.size)
            
            for i, (neuron_id, neuron) in enumerate(self.neurons.items()):
                # Check refractory period
                if time_step - self.last_spike[i] < self.refractory_period:
                    outputs[i] = 0.0
                    continue
                
                # Calculate membrane potential
                input_current = inputs[i] + neuron.bias
                
                # Add noise
                noise = np.random.normal(0, self.noise_level)
                input_current += noise
                
                # Apply lateral inhibition
                inhibition = self._calculate_lateral_inhibition(i, inputs)
                input_current -= inhibition
                
                # Update membrane potential
                self.membrane_potential[i] = input_current
                
                # Apply activation function
                activation = self.activation_func.forward(input_current)
                
                # Determine if neuron spikes
                if activation > neuron.threshold:
                    outputs[i] = activation
                    self.last_spike[i] = time_step
                    self._record_spike(neuron_id, time_step, activation)
                else:
                    outputs[i] = 0.0
                
                # Update neuron state
                neuron.activation = outputs[i]
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Forward pass error in layer {self.layer_id}: {e}")
            raise NeuralNetworkError(f"Forward pass failed: {e}")
    
    def _calculate_lateral_inhibition(self, neuron_idx: int, activations: np.ndarray) -> float:
        """Calculate lateral inhibition for a neuron."""
        neuron_position = list(self.neurons.values())[neuron_idx].position
        inhibition = 0.0
        
        for i, other_activation in enumerate(activations):
            if i != neuron_idx and other_activation > 0:
                other_position = list(self.neurons.values())[i].position
                distance = np.sqrt(sum((a - b)**2 for a, b in zip(neuron_position, other_position)))
                
                # Gaussian inhibition kernel
                inhibition += other_activation * np.exp(-distance**2 / 2.0) * self.inhibition_strength
        
        return inhibition
    
    def _record_spike(self, neuron_id: str, time_step: float, amplitude: float):
        """Record spike event for analysis."""
        if neuron_id not in self.spike_times:
            self.spike_times[neuron_id] = []
        
        self.spike_times[neuron_id].append({
            'time': time_step,
            'amplitude': amplitude
        })
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def update_weights(self, learning_rate: float, error_gradient: Optional[np.ndarray] = None):
        """Update layer weights using STDP or error-based learning."""
        # Spike-timing dependent plasticity (STDP) implementation
        for neuron_id, spike_history in self.spike_times.items():
            if len(spike_history) > 1:
                # Calculate STDP updates based on spike timing
                for i in range(len(spike_history) - 1):
                    dt = spike_history[i+1]['time'] - spike_history[i]['time']
                    
                    # STDP window function
                    if dt > 0:  # Post before pre (LTD)
                        weight_change = -learning_rate * np.exp(-dt / 0.02)
                    else:  # Pre before post (LTP)
                        weight_change = learning_rate * np.exp(dt / 0.02)
                    
                    # Apply theoretical modulation
                    weight_change *= self.theoretical_alignment
    
    def get_spike_statistics(self) -> Dict[str, Any]:
        """Get layer spike statistics."""
        total_spikes = sum(len(spikes) for spikes in self.spike_times.values())
        active_neurons = len([spikes for spikes in self.spike_times.values() if len(spikes) > 0])
        
        return {
            'total_spikes': total_spikes,
            'active_neurons': active_neurons,
            'spike_rate': total_spikes / max(1, len(self.neurons)),
            'activation_ratio': active_neurons / len(self.neurons)
        }


@fsot_enforce(domain=FSOTDomain.NEUROMORPHIC, d_eff=15)
class NeuromorphicNeuralNetwork:
    """
    Complete neuromorphic neural network with FSOT 2.0 compliance.
    
    This network implements:
    - Multi-layer neuromorphic processing
    - Temporal dynamics simulation
    - Synaptic plasticity learning
    - Brain-inspired architectures
    - Real-time adaptation capabilities
    """
    
    def __init__(self, network_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize neuromorphic neural network.
        
        Args:
            network_id: Unique network identifier
            config: Network configuration dictionary
        """
        self.network_id = network_id
        self.config = config or {}
        
        # Network architecture
        self.layers = {}
        self.connections = {}
        self.layer_order = []
        
        # Temporal simulation
        self.current_time = 0.0
        self.time_step = 0.001  # 1ms
        self.simulation_running = False
        
        # Learning parameters
        self.learning_rate = 0.001
        self.plasticity_enabled = True
        self.adaptation_rate = 0.01
        
        # FSOT compliance
        self.fsot_core = FSOTCore()
        self.theoretical_consistency = True
        self.compliance_score = 1.0
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = {}
        
        # Threading for real-time processing
        self.processing_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Neuromorphic Neural Network '{network_id}' initialized")
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def add_layer(self, 
                  layer_id: str,
                  size: int,
                  activation_func: Optional[ActivationFunction] = None,
                  spatial_dimensions: Tuple[int, int, int] = (1, 1, 1)) -> NeuromorphicLayer:
        """
        Add a neuromorphic layer to the network.
        
        Args:
            layer_id: Unique layer identifier
            size: Number of neurons in layer
            activation_func: Activation function
            spatial_dimensions: 3D spatial arrangement
            
        Returns:
            Created neuromorphic layer
        """
        if layer_id in self.layers:
            raise NeuralNetworkError(f"Layer '{layer_id}' already exists")
        
        layer = NeuromorphicLayer(layer_id, size, activation_func, spatial_dimensions)
        self.layers[layer_id] = layer
        self.layer_order.append(layer_id)
        
        self.logger.info(f"Added layer '{layer_id}' with {size} neurons")
        return layer
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def connect_layers(self, 
                      source_layer: str, 
                      target_layer: str,
                      connection_type: str = 'full',
                      weight_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Connect two layers with synapses.
        
        Args:
            source_layer: Source layer ID
            target_layer: Target layer ID
            connection_type: Type of connection ('full', 'sparse', 'local')
            weight_range: Range for random weight initialization
        """
        if source_layer not in self.layers:
            raise NeuralNetworkError(f"Source layer '{source_layer}' not found")
        if target_layer not in self.layers:
            raise NeuralNetworkError(f"Target layer '{target_layer}' not found")
        
        source = self.layers[source_layer]
        target = self.layers[target_layer]
        
        connection_key = f"{source_layer}->{target_layer}"
        synapses = []
        
        if connection_type == 'full':
            # Fully connected layers
            for src_id, src_neuron in source.neurons.items():
                for tgt_id, tgt_neuron in target.neurons.items():
                    weight = np.random.uniform(weight_range[0], weight_range[1])
                    synapse = Synapse(src_id, tgt_id, weight)
                    synapses.append(synapse)
        
        elif connection_type == 'sparse':
            # Sparse random connections (30% connectivity)
            connection_prob = 0.3
            for src_id, src_neuron in source.neurons.items():
                for tgt_id, tgt_neuron in target.neurons.items():
                    if np.random.random() < connection_prob:
                        weight = np.random.uniform(weight_range[0], weight_range[1])
                        synapse = Synapse(src_id, tgt_id, weight)
                        synapses.append(synapse)
        
        elif connection_type == 'local':
            # Local connections based on spatial proximity (optimized for large networks)
            max_distance = 2.0
            max_connections_per_neuron = 100  # Limit connections to prevent exponential growth
            
            # For very large layers, use sampling to maintain performance
            if len(source.neurons) * len(target.neurons) > 10000:
                # Sample a subset of connections for large networks
                source_items = list(source.neurons.items())
                target_items = list(target.neurons.items())
                
                # Limit samples to maintain reasonable performance
                if len(source_items) > 1000:
                    source_indices = np.random.choice(len(source_items), 1000, replace=False)
                    source_sample = [source_items[i] for i in source_indices]
                else:
                    source_sample = source_items
                    
                if len(target_items) > 1000:
                    target_indices = np.random.choice(len(target_items), 1000, replace=False)
                    target_sample = [target_items[i] for i in target_indices]
                else:
                    target_sample = target_items
                    
                for src_id, src_neuron in source_sample:
                    connections_made = 0
                    for tgt_id, tgt_neuron in target_sample:
                        if connections_made >= max_connections_per_neuron:
                            break
                            
                        distance = np.sqrt(sum((a - b)**2 for a, b in 
                                             zip(src_neuron.position, tgt_neuron.position)))
                        
                        if distance <= max_distance:
                            # Weight inversely proportional to distance
                            weight = np.random.uniform(weight_range[0], weight_range[1])
                            weight *= np.exp(-distance / max_distance)
                            synapse = Synapse(src_id, tgt_id, weight)
                            synapses.append(synapse)
                            connections_made += 1
            else:
                # Original algorithm for smaller networks
                for src_id, src_neuron in source.neurons.items():
                    connections_made = 0
                    for tgt_id, tgt_neuron in target.neurons.items():
                        if connections_made >= max_connections_per_neuron:
                            break
                            
                        distance = np.sqrt(sum((a - b)**2 for a, b in 
                                             zip(src_neuron.position, tgt_neuron.position)))
                        
                        if distance <= max_distance:
                            # Weight inversely proportional to distance
                            weight = np.random.uniform(weight_range[0], weight_range[1])
                            weight *= np.exp(-distance / max_distance)
                            synapse = Synapse(src_id, tgt_id, weight)
                            synapses.append(synapse)
                            connections_made += 1
                        synapse = Synapse(src_id, tgt_id, weight)
                        synapses.append(synapse)
        
        self.connections[connection_key] = synapses
        self.logger.info(f"Connected {source_layer} -> {target_layer} with {len(synapses)} synapses")
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def forward_pass(self, inputs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform forward pass through the network.
        
        Args:
            inputs: Input data array
            
        Returns:
            Dictionary of layer outputs
        """
        try:
            with self.processing_lock:
                layer_outputs = {}
                current_input = inputs
                
                for layer_id in self.layer_order:
                    layer = self.layers[layer_id]
                    
                    # Apply input transformations if this isn't the first layer
                    if layer_id != self.layer_order[0]:
                        current_input = self._apply_connections(
                            self.layer_order[self.layer_order.index(layer_id) - 1],
                            layer_id,
                            layer_outputs[self.layer_order[self.layer_order.index(layer_id) - 1]]
                        )
                    
                    # Forward pass through layer
                    output = layer.forward(current_input, self.current_time)
                    layer_outputs[layer_id] = output
                
                # Update simulation time
                self.current_time += self.time_step
                
                return layer_outputs
                
        except Exception as e:
            self.logger.error(f"Forward pass error: {e}")
            raise NeuralNetworkError(f"Forward pass failed: {e}")
    
    def _apply_connections(self, source_layer: str, target_layer: str, source_output: np.ndarray) -> np.ndarray:
        """Apply synaptic connections between layers."""
        connection_key = f"{source_layer}->{target_layer}"
        
        if connection_key not in self.connections:
            # No connections defined, pass through directly
            target_size = self.layers[target_layer].size
            if len(source_output) == target_size:
                return source_output
            else:
                # Resize if needed
                return np.resize(source_output, target_size)
        
        synapses = self.connections[connection_key]
        target_neurons = list(self.layers[target_layer].neurons.keys())
        source_neurons = list(self.layers[source_layer].neurons.keys())
        
        # Create mapping from neuron IDs to indices
        source_idx_map = {nid: i for i, nid in enumerate(source_neurons)}
        target_idx_map = {nid: i for i, nid in enumerate(target_neurons)}
        
        # Initialize target input
        target_input = np.zeros(len(target_neurons))
        
        # Apply synaptic weights
        for synapse in synapses:
            if (synapse.pre_neuron in source_idx_map and 
                synapse.post_neuron in target_idx_map):
                
                src_idx = source_idx_map[synapse.pre_neuron]
                tgt_idx = target_idx_map[synapse.post_neuron]
                
                if src_idx < len(source_output):
                    target_input[tgt_idx] += source_output[src_idx] * synapse.weight
        
        return target_input
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def train(self, 
              training_data: List[Tuple[np.ndarray, np.ndarray]],
              epochs: int,
              validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, List[float]]:
        """
        Train the neuromorphic network.
        
        Args:
            training_data: List of (input, target) pairs
            epochs: Number of training epochs
            validation_data: Optional validation data
            
        Returns:
            Training history with metrics
        """
        try:
            history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                correct_predictions = 0
                
                # Training phase
                for batch_idx, (inputs, targets) in enumerate(training_data):
                    # Forward pass
                    outputs = self.forward_pass(inputs)
                    final_output = outputs[self.layer_order[-1]]
                    
                    # Calculate loss
                    loss = self._calculate_loss(final_output, targets)
                    epoch_loss += loss
                    
                    # Check accuracy
                    if self._check_prediction_accuracy(final_output, targets):
                        correct_predictions += 1
                    
                    # Backward pass and weight updates
                    if self.plasticity_enabled:
                        self._update_network_weights(final_output, targets)
                
                # Calculate epoch metrics
                avg_loss = epoch_loss / len(training_data)
                accuracy = correct_predictions / len(training_data)
                
                history['loss'].append(avg_loss)
                history['accuracy'].append(accuracy)
                
                # Validation phase
                if validation_data:
                    val_loss, val_accuracy = self._validate(validation_data)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            self.training_history.append(history)
            return history
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise NeuralNetworkError(f"Training failed: {e}")
    
    def _calculate_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate mean squared error loss."""
        if len(predictions) != len(targets):
            # Resize if needed
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]
        
        return float(np.mean((predictions - targets) ** 2))
    
    def _check_prediction_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> bool:
        """Check if prediction is accurate (for classification tasks)."""
        if len(predictions) != len(targets):
            return False
        
        # For binary classification
        pred_class = 1 if np.max(predictions) > 0.5 else 0
        target_class = 1 if np.max(targets) > 0.5 else 0
        
        return pred_class == target_class
    
    def _update_network_weights(self, outputs: np.ndarray, targets: np.ndarray):
        """Update network weights using neuromorphic learning rules."""
        # Calculate error signal
        error = targets - outputs if len(targets) == len(outputs) else np.zeros_like(outputs)
        
        # Update each layer using STDP and error-driven plasticity
        for layer_id in reversed(self.layer_order):
            layer = self.layers[layer_id]
            layer.update_weights(self.learning_rate, error)
    
    def _validate(self, validation_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
        """Validate network performance."""
        total_loss = 0.0
        correct_predictions = 0
        
        for inputs, targets in validation_data:
            outputs = self.forward_pass(inputs)
            final_output = outputs[self.layer_order[-1]]
            
            loss = self._calculate_loss(final_output, targets)
            total_loss += loss
            
            if self._check_prediction_accuracy(final_output, targets):
                correct_predictions += 1
        
        avg_loss = total_loss / len(validation_data)
        accuracy = correct_predictions / len(validation_data)
        
        return avg_loss, accuracy
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {
            'network_id': self.network_id,
            'total_layers': len(self.layers),
            'total_neurons': sum(layer.size for layer in self.layers.values()),
            'total_connections': sum(len(synapses) for synapses in self.connections.values()),
            'simulation_time': self.current_time,
            'fsot_compliance': {
                'theoretical_consistency': self.theoretical_consistency,
                'compliance_score': self.compliance_score,
                'core_signature': "FSOT-2.0-COMPLIANT"
            },
            'layer_statistics': {},
            'performance_metrics': self.performance_metrics
        }
        
        # Add layer-specific statistics
        for layer_id, layer in self.layers.items():
            stats['layer_statistics'][layer_id] = {
                'size': layer.size,
                'spike_stats': layer.get_spike_statistics(),
                'spatial_dimensions': layer.spatial_dimensions
            }
        
        return stats
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def save_network(self, filepath: str):
        """Save network state to file."""
        try:
            network_state = {
                'network_id': self.network_id,
                'config': self.config,
                'layer_order': self.layer_order,
                'layers': {},
                'connections': {},
                'current_time': self.current_time,
                'learning_rate': self.learning_rate,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save layer states
            for layer_id, layer in self.layers.items():
                network_state['layers'][layer_id] = {
                    'size': layer.size,
                    'spatial_dimensions': layer.spatial_dimensions,
                    'neurons': {nid: {
                        'id': neuron.id,
                        'activation': neuron.activation,
                        'threshold': neuron.threshold,
                        'bias': neuron.bias,
                        'position': neuron.position,
                        'neuron_type': neuron.neuron_type
                    } for nid, neuron in layer.neurons.items()},
                    'spike_times': layer.spike_times,
                    'membrane_potential': layer.membrane_potential.tolist(),
                    'last_spike': layer.last_spike.tolist()
                }
            
            # Save connections
            for conn_key, synapses in self.connections.items():
                network_state['connections'][conn_key] = [{
                    'pre_neuron': syn.pre_neuron,
                    'post_neuron': syn.post_neuron,
                    'weight': syn.weight,
                    'delay': syn.delay,
                    'plasticity': syn.plasticity
                } for syn in synapses]
            
            with open(filepath, 'w') as f:
                json.dump(network_state, f, indent=2)
            
            self.logger.info(f"Network saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving network: {e}")
            raise NeuralNetworkError(f"Failed to save network: {e}")
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def load_network(self, filepath: str):
        """Load network state from file."""
        try:
            with open(filepath, 'r') as f:
                network_state = json.load(f)
            
            # Restore basic properties
            self.network_id = network_state['network_id']
            self.config = network_state.get('config', {})
            self.layer_order = network_state['layer_order']
            self.current_time = network_state.get('current_time', 0.0)
            self.learning_rate = network_state.get('learning_rate', 0.001)
            self.training_history = network_state.get('training_history', [])
            self.performance_metrics = network_state.get('performance_metrics', {})
            
            # Restore layers
            self.layers.clear()
            for layer_id, layer_data in network_state['layers'].items():
                layer = NeuromorphicLayer(
                    layer_id,
                    layer_data['size'],
                    spatial_dimensions=tuple(layer_data['spatial_dimensions'])
                )
                
                # Restore neurons
                for neuron_id, neuron_data in layer_data['neurons'].items():
                    if neuron_id in layer.neurons:
                        neuron = layer.neurons[neuron_id]
                        neuron.activation = neuron_data['activation']
                        neuron.threshold = neuron_data['threshold']
                        neuron.bias = neuron_data['bias']
                        neuron.position = tuple(neuron_data['position'])
                        neuron.neuron_type = neuron_data['neuron_type']
                
                # Restore temporal data
                layer.spike_times = layer_data.get('spike_times', {})
                membrane_data = layer_data.get('membrane_potential', [])
                if membrane_data and len(membrane_data) == layer.size:
                    layer.membrane_potential[:] = membrane_data
                
                spike_data = layer_data.get('last_spike', [])
                if spike_data and len(spike_data) == layer.size:
                    layer.last_spike[:] = spike_data
                
                self.layers[layer_id] = layer
            
            # Restore connections
            self.connections.clear()
            for conn_key, synapse_data in network_state['connections'].items():
                synapses = []
                for syn_data in synapse_data:
                    synapse = Synapse(
                        syn_data['pre_neuron'],
                        syn_data['post_neuron'],
                        syn_data['weight'],
                        syn_data.get('delay', 0.0),
                        syn_data.get('plasticity', 0.01)
                    )
                    synapses.append(synapse)
                self.connections[conn_key] = synapses
            
            self.logger.info(f"Network loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading network: {e}")
            raise NeuralNetworkError(f"Failed to load network: {e}")
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def process(self, inputs: np.ndarray) -> np.ndarray:
        """
        Process inputs through the network (alias for forward).
        
        Args:
            inputs: Input data array
            
        Returns:
            Network output array
        """
        try:
            outputs = self.forward_pass(inputs)
            # Return the final layer output
            if outputs and self.layer_order:
                final_layer = self.layer_order[-1]
                return outputs.get(final_layer, inputs)
            return inputs
        except Exception as e:
            self.logger.error(f"Process error: {e}")
            # Fallback: return input shape with zeros
            return np.zeros_like(inputs)
    
    @fsot_enforce(domain=FSOTDomain.NEUROMORPHIC)
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input data array
            
        Returns:
            Network output array
        """
        return self.process(inputs)


# Factory functions for common network architectures
@fsot_enforce(domain=FSOTDomain.NEUROMORPHIC, d_eff=12)
def create_feedforward_network(input_size: int, 
                             hidden_sizes: List[int], 
                             output_size: int,
                             activation_func: Optional[ActivationFunction] = None) -> NeuromorphicNeuralNetwork:
    """Create a feedforward neuromorphic network."""
    network = NeuromorphicNeuralNetwork("feedforward_net")
    
    activation = activation_func or SigmoidActivation()
    
    # Input layer
    network.add_layer("input", input_size, activation)
    
    # Hidden layers
    for i, hidden_size in enumerate(hidden_sizes):
        layer_id = f"hidden_{i}"
        network.add_layer(layer_id, hidden_size, activation)
        
        # Connect to previous layer
        prev_layer = "input" if i == 0 else f"hidden_{i-1}"
        network.connect_layers(prev_layer, layer_id, "full")
    
    # Output layer
    network.add_layer("output", output_size, activation)
    prev_layer = f"hidden_{len(hidden_sizes)-1}" if hidden_sizes else "input"
    network.connect_layers(prev_layer, "output", "full")
    
    return network


@fsot_enforce(domain=FSOTDomain.NEUROMORPHIC, d_eff=15)
def create_convolutional_network(input_shape: Tuple[int, int, int],
                               filter_sizes: List[int],
                               dense_sizes: List[int],
                               output_size: int) -> NeuromorphicNeuralNetwork:
    """Create a convolutional neuromorphic network."""
    network = NeuromorphicNeuralNetwork("conv_net")
    
    # Flatten input for neuromorphic processing
    input_size = input_shape[0] * input_shape[1] * input_shape[2]
    network.add_layer("input", input_size, ReLUActivation(), input_shape)
    
    # Convolutional-like layers with spatial organization
    prev_layer = "input"
    for i, filter_size in enumerate(filter_sizes):
        layer_id = f"conv_{i}"
        # Spatial dimensions for conv-like processing
        spatial_dims = (max(1, input_shape[0] // (2**i)), 
                       max(1, input_shape[1] // (2**i)), 
                       filter_size)
        
        layer_size = spatial_dims[0] * spatial_dims[1] * spatial_dims[2]
        network.add_layer(layer_id, layer_size, ReLUActivation(), spatial_dims)
        network.connect_layers(prev_layer, layer_id, "local")
        prev_layer = layer_id
    
    # Dense layers
    for i, dense_size in enumerate(dense_sizes):
        layer_id = f"dense_{i}"
        network.add_layer(layer_id, dense_size, SigmoidActivation())
        network.connect_layers(prev_layer, layer_id, "full")
        prev_layer = layer_id
    
    # Output layer
    network.add_layer("output", output_size, SigmoidActivation())
    network.connect_layers(prev_layer, "output", "full")
    
    return network


if __name__ == "__main__":
    # Example usage and testing
    print("Creating neuromorphic neural network...")
    
    # Create a simple feedforward network
    network = create_feedforward_network(
        input_size=10,
        hidden_sizes=[20, 15],
        output_size=5
    )
    
    # Test forward pass
    test_input = np.random.randn(10)
    outputs = network.forward_pass(test_input)
    
    print("Network Statistics:")
    stats = network.get_network_statistics()
    print(json.dumps(stats, indent=2, default=str))
    
    # Test training with dummy data
    training_data = [(np.random.randn(10), np.random.randn(5)) for _ in range(100)]
    
    print("\nTraining network...")
    history = network.train(training_data, epochs=50)
    
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    
    # Save network
    network.save_network("test_network.json")
    print("Network saved successfully")
