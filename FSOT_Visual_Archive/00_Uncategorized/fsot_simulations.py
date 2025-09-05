"""
FSOT 2.0 Simulation Engine
==========================
Advanced simulation capabilities integrated with FSOT theoretical framework.

This module provides various simulation types including quantum, biological,
physics, and neuromorphic simulations with FSOT scalar integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Optional advanced simulation imports
try:
    import networkx as nx
    from scipy.integrate import odeint
    from scipy.spatial.distance import pdist, squareform
    import sympy as sp
    ADVANCED_SIMS_AVAILABLE = True
except ImportError:
    ADVANCED_SIMS_AVAILABLE = False

class FSOTSimulationEngine:
    """
    Advanced simulation engine integrated with FSOT 2.0 mathematics.
    Provides quantum, biological, physics, and complex system simulations.
    """
    
    def __init__(self, output_dir: str = "simulation_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.simulation_history = []
        
    def run_simulation(self, sim_type: str, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """Main simulation dispatcher"""
        
        sim_type = sim_type.lower().replace(" ", "_")
        
        if "quantum" in sim_type and "germ" in sim_type:
            return self.quantum_germ_simulation(params, fsot_params)
        elif "cellular" in sim_type or "biological" in sim_type:
            return self.cellular_automata_simulation(params, fsot_params)
        elif "network" in sim_type or "neural" in sim_type:
            return self.neural_network_simulation(params, fsot_params)
        elif "physics" in sim_type or "particle" in sim_type:
            return self.particle_physics_simulation(params, fsot_params)
        elif "ecosystem" in sim_type or "evolution" in sim_type:
            return self.ecosystem_simulation(params, fsot_params)
        else:
            return self.default_fsot_simulation(params, fsot_params)
    
    def quantum_germ_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Quantum germ simulation with FSOT field interactions.
        Models quantum-scale entities interacting through FSOT-derived fields.
        """
        # Parameters
        num_germs = params.get('num_germs', 25)
        steps = params.get('steps', 100)
        grid_size = params.get('grid_size', 100)
        
        # FSOT-derived parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        d_eff = fsot_params.get('d_eff', 6)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Generate FSOT-influenced quantum field
        field_strength = s_scalar * consciousness_factor
        quantum_field = np.random.uniform(-field_strength, field_strength, (grid_size, grid_size))
        
        # Add FSOT dimensional efficiency patterns
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, grid_size), np.linspace(0, 2*np.pi, grid_size))
        dimensional_pattern = np.sin(d_eff * x) * np.cos(d_eff * y) * field_strength * 0.1
        quantum_field += dimensional_pattern
        
        # Initialize quantum germs
        germ_positions = np.random.uniform(0, grid_size, (num_germs, 2))
        germ_states = np.array(np.random.uniform(0, 1, num_germs))  # Explicit array conversion
        germ_history = [germ_positions.copy()]
        
        # Simulation loop
        for step in range(steps):
            # Get field influence at current positions
            field_x = np.clip(germ_positions[:, 0].astype(int), 0, grid_size-1)
            field_y = np.clip(germ_positions[:, 1].astype(int), 0, grid_size-1)
            field_influence = quantum_field[field_y, field_x]
            
            # FSOT-guided movement
            movement_scale = 0.5 * s_scalar
            random_movement = np.random.uniform(-movement_scale, movement_scale, germ_positions.shape)
            field_movement = field_influence[:, np.newaxis] * np.random.uniform(-0.3, 0.3, germ_positions.shape)
            
            # Update positions
            germ_positions += random_movement + field_movement
            germ_positions = np.mod(germ_positions, grid_size)  # Periodic boundary
            
            # Update quantum states based on FSOT interactions
            state_influence = field_influence * consciousness_factor
            germ_states = np.clip(germ_states + state_influence * 0.1, 0, 1)
            
            # Record history every 10 steps
            if step % 10 == 0:
                germ_history.append(germ_positions.copy())
        
        # Generate visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot quantum field
        im = ax1.imshow(quantum_field, cmap='viridis', alpha=0.7)
        ax1.scatter(germ_positions[:, 0], germ_positions[:, 1], 
                   c=germ_states, s=50, cmap='plasma', edgecolors='white', linewidth=1)
        ax1.set_title(f'FSOT Quantum Germ Field (S={s_scalar:.3f}, D_eff={d_eff})')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im, ax=ax1, label='Field Strength')
        
        # Plot trajectories
        for i, history in enumerate(zip(*germ_history)):
            trajectory = np.array(history)
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.6, linewidth=1)
        ax2.scatter(germ_positions[:, 0], germ_positions[:, 1], 
                   c=germ_states, s=50, cmap='plasma', edgecolors='black', linewidth=1)
        ax2.set_title('Quantum Germ Trajectories')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_xlim(0, grid_size)
        ax2.set_ylim(0, grid_size)
        
        plt.tight_layout()
        
        # Save simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_germ_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate FSOT metrics
        avg_state = np.mean(germ_states)
        field_variance = np.var(quantum_field)
        coherence_measure = 1 - np.std(germ_states)
        
        result = {
            'simulation_type': 'Quantum Germ Simulation',
            'parameters': {
                'num_germs': num_germs,
                'steps': steps,
                'grid_size': grid_size,
                's_scalar': s_scalar,
                'd_eff': d_eff,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'average_quantum_state': float(avg_state),
                'field_variance': float(field_variance),
                'coherence_measure': float(coherence_measure),
                'final_positions_spread': float(np.std(germ_positions))
            },
            'output_file': str(filepath),
            'data': {
                'final_positions': germ_positions.tolist(),
                'final_states': list(germ_states) if isinstance(germ_states, np.ndarray) else [float(germ_states)],
                'field_sample': quantum_field[::10, ::10].tolist()  # Downsampled field
            },
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def cellular_automata_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Cellular automata with FSOT-influenced rules.
        Models biological/cellular systems with FSOT mathematical integration.
        """
        # Parameters
        grid_size = params.get('grid_size', 80)
        generations = params.get('generations', 50)
        initial_density = params.get('initial_density', 0.3)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        d_eff = fsot_params.get('d_eff', 8)
        
        # Initialize cellular grid
        # Initialize grid with integer values (0 or 1) instead of boolean
        grid = (np.random.random((grid_size, grid_size)) < initial_density).astype(int)
        history = [grid.copy()]
        
        # FSOT-influenced cellular rules
        survival_threshold = 2 + s_scalar * 2  # FSOT influences survival
        birth_threshold = 3 + s_scalar * 1
        
        for gen in range(generations):
            new_grid = grid.copy()
            
            for i in range(1, grid_size-1):
                for j in range(1, grid_size-1):
                    # Count neighbors (now using integer grid)
                    neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
                    
                    # FSOT-modified rules
                    if grid[i, j] == 1:  # Cell is alive
                        if neighbors < survival_threshold or neighbors > 3 + s_scalar:
                            new_grid[i, j] = 0
                    else:  # Cell is dead
                        if abs(neighbors - birth_threshold) < 0.5:
                            new_grid[i, j] = 1
            
            grid = new_grid
            if gen % 5 == 0:  # Save every 5th generation
                history.append(grid.copy())
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (gen_idx, gen_grid) in enumerate(zip([0, 10, 20, 30, 40, 49], history)):
            if idx < 6:
                axes[idx].imshow(gen_grid, cmap='binary')
                axes[idx].set_title(f'Generation {gen_idx}')
                axes[idx].axis('off')
        
        plt.suptitle(f'FSOT Cellular Automata (S={s_scalar:.3f}, D_eff={d_eff})')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cellular_automata_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        final_density = np.mean(grid)
        stability = 1 - np.mean([np.sum(np.abs(history[i] - history[i-1])) 
                                for i in range(1, len(history))]) / (grid_size**2)
        
        result = {
            'simulation_type': 'Cellular Automata Simulation',
            'parameters': {
                'grid_size': grid_size,
                'generations': generations,
                'initial_density': initial_density,
                's_scalar': s_scalar,
                'd_eff': d_eff
            },
            'metrics': {
                'final_density': float(final_density),
                'stability_measure': float(stability),
                'pattern_complexity': float(np.std(grid.astype(float)))
            },
            'output_file': str(filepath),
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def particle_physics_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Particle physics simulation with FSOT field interactions.
        Models particle dynamics in FSOT-influenced fields.
        """
        # Parameters
        num_particles = params.get('num_particles', 30)
        steps = params.get('steps', 200)
        field_size = params.get('field_size', 10.0)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        d_eff = fsot_params.get('d_eff', 8)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Initialize particles
        positions = np.random.uniform(-field_size/2, field_size/2, (num_particles, 2))
        velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))
        masses = np.array(np.random.uniform(0.5, 2.0, num_particles))  # Explicit array conversion
        
        # Simulation history
        position_history = [positions.copy()]
        
        # FSOT field function
        def fsot_field_force(pos, s_scalar, d_eff, consciousness_factor):
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            if r == 0:
                return np.array([0.0, 0.0])
            
            # FSOT-derived force
            field_strength = s_scalar * np.exp(-r/d_eff) * consciousness_factor
            force_magnitude = field_strength / (r + 0.1)  # Avoid division by zero
            force_direction = -pos / r  # Attractive force toward center
            
            return force_magnitude * force_direction
        
        # Simulation loop
        dt = 0.01
        for step in range(steps):
            forces = np.zeros_like(positions)
            
            # Calculate FSOT forces for each particle
            for i in range(num_particles):
                fsot_force = fsot_field_force(positions[i], s_scalar, d_eff, consciousness_factor)
                forces[i] = fsot_force
                
                # Add particle-particle interactions
                for j in range(num_particles):
                    if i != j:
                        r_vec = positions[j] - positions[i]
                        r_mag = np.linalg.norm(r_vec)
                        if r_mag > 0.1:  # Avoid singularity
                            # Simple repulsive force
                            force_mag = 0.01 / (r_mag**2)
                            forces[i] -= force_mag * r_vec / r_mag
            
            # Update velocities and positions
            accelerations = forces / masses[:, np.newaxis]
            velocities += accelerations * dt
            velocities *= 0.99  # Damping
            positions += velocities * dt
            
            # Boundary conditions (reflective)
            for i in range(num_particles):
                for dim in range(2):
                    if abs(positions[i, dim]) > field_size/2:
                        positions[i, dim] = np.sign(positions[i, dim]) * field_size/2
                        velocities[i, dim] *= -0.8  # Damped reflection
            
            # Record history every 10 steps
            if step % 10 == 0:
                position_history.append(positions.copy())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final particle positions with field
        x_field = np.linspace(-field_size/2, field_size/2, 50)
        y_field = np.linspace(-field_size/2, field_size/2, 50)
        X_field, Y_field = np.meshgrid(x_field, y_field)
        
        # Calculate field strength
        field_strength = np.zeros_like(X_field)
        for i in range(X_field.shape[0]):
            for j in range(X_field.shape[1]):
                r = np.sqrt(X_field[i,j]**2 + Y_field[i,j]**2)
                field_strength[i,j] = s_scalar * np.exp(-r/d_eff) * consciousness_factor
        
        # Plot field and particles
        im = ax1.contourf(X_field, Y_field, field_strength, levels=20, cmap='viridis', alpha=0.7)
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                             c=masses, s=50*masses, cmap='plasma', 
                             edgecolors='white', linewidth=1)
        ax1.set_title(f'FSOT Particle Field (S={s_scalar:.3f})')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        plt.colorbar(im, ax=ax1, label='Field Strength')
        
        # Plot trajectories
        for i in range(num_particles):
            trajectory = np.array([pos[i] for pos in position_history])
            ax2.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.6, linewidth=1)
        
        ax2.scatter(positions[:, 0], positions[:, 1], 
                   c=masses, s=50*masses, cmap='plasma', 
                   edgecolors='black', linewidth=1)
        ax2.set_title('Particle Trajectories')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_xlim(-field_size/2, field_size/2)
        ax2.set_ylim(-field_size/2, field_size/2)
        
        plt.tight_layout()
        
        # Save simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"particle_physics_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        final_kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        center_of_mass = np.mean(positions, axis=0)
        particle_spread = np.std(positions, axis=0)
        
        result = {
            'simulation_type': 'Particle Physics Simulation',
            'parameters': {
                'num_particles': num_particles,
                'steps': steps,
                'field_size': field_size,
                's_scalar': s_scalar,
                'd_eff': d_eff,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'final_kinetic_energy': float(final_kinetic_energy),
                'center_of_mass_x': float(center_of_mass[0]),
                'center_of_mass_y': float(center_of_mass[1]),
                'particle_spread_x': float(particle_spread[0]),
                'particle_spread_y': float(particle_spread[1])
            },
            'output_file': str(filepath),
            'data': {
                'final_positions': positions.tolist(),
                'final_velocities': velocities.tolist(),
                'masses': list(masses) if isinstance(masses, np.ndarray) else [float(masses)]
            },
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def ecosystem_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Ecosystem evolution simulation with FSOT consciousness factors.
        Models predator-prey dynamics with FSOT-influenced evolution.
        """
        # Parameters
        grid_size = params.get('grid_size', 60)
        generations = params.get('generations', 100)
        initial_prey = params.get('initial_prey', 200)
        initial_predators = params.get('initial_predators', 50)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Initialize ecosystem
        prey_positions = np.random.uniform(0, grid_size, (initial_prey, 2))
        predator_positions = np.random.uniform(0, grid_size, (initial_predators, 2))
        
        prey_history = [len(prey_positions)]
        predator_history = [len(predator_positions)]
        
        # FSOT-influenced ecosystem parameters
        prey_reproduction_rate = 0.1 + s_scalar * 0.05
        predator_efficiency = 0.3 + consciousness_factor * 0.2
        
        for gen in range(generations):
            # Prey reproduction (FSOT-enhanced)
            if len(prey_positions) > 0:
                reproduction_prob = prey_reproduction_rate * (1 + np.random.uniform(-0.1, 0.1))
                new_prey_count = int(len(prey_positions) * reproduction_prob)
                
                if new_prey_count > 0:
                    new_prey = np.random.uniform(0, grid_size, (new_prey_count, 2))
                    prey_positions = np.vstack([prey_positions, new_prey])
            
            # Predation (FSOT consciousness affects hunting success)
            if len(prey_positions) > 0 and len(predator_positions) > 0:
                prey_to_remove = []
                
                for pred_idx, pred_pos in enumerate(predator_positions):
                    # Find nearby prey
                    distances = np.sqrt(np.sum((prey_positions - pred_pos)**2, axis=1))
                    nearby_prey = np.where(distances < 2.0)[0]
                    
                    if len(nearby_prey) > 0:
                        # FSOT consciousness affects hunting success
                        hunt_success_rate = predator_efficiency * consciousness_factor
                        if np.random.random() < hunt_success_rate:
                            # Catch closest prey
                            closest_prey = nearby_prey[np.argmin(distances[nearby_prey])]
                            prey_to_remove.append(closest_prey)
                
                # Remove caught prey
                if prey_to_remove:
                    prey_positions = np.delete(prey_positions, prey_to_remove, axis=0)
            
            # Predator survival and reproduction
            if len(predator_positions) > 0:
                # Survival depends on prey availability
                survival_rate = min(0.9, len(prey_positions) / (len(predator_positions) * 10))
                survivors = np.random.random(len(predator_positions)) < survival_rate
                predator_positions = predator_positions[survivors]
                
                # Reproduction if well-fed
                if len(prey_positions) > len(predator_positions) * 5:
                    reproduction_rate = 0.05 * s_scalar
                    new_predator_count = max(1, int(len(predator_positions) * reproduction_rate))
                    new_predators = np.random.uniform(0, grid_size, (new_predator_count, 2))
                    predator_positions = np.vstack([predator_positions, new_predators])
            
            # Movement (FSOT-influenced random walk)
            if len(prey_positions) > 0:
                movement_scale = 1.0 + s_scalar * 0.5
                prey_movement = np.random.uniform(-movement_scale, movement_scale, prey_positions.shape)
                prey_positions = np.clip(prey_positions + prey_movement, 0, grid_size)
            
            if len(predator_positions) > 0:
                movement_scale = 1.5 + consciousness_factor * 0.5
                predator_movement = np.random.uniform(-movement_scale, movement_scale, predator_positions.shape)
                predator_positions = np.clip(predator_positions + predator_movement, 0, grid_size)
            
            # Record population
            prey_history.append(len(prey_positions))
            predator_history.append(len(predator_positions))
            
            # Prevent extinction
            if len(prey_positions) == 0:
                prey_positions = np.random.uniform(0, grid_size, (10, 2))
            if len(predator_positions) == 0 and len(prey_positions) > 20:
                predator_positions = np.random.uniform(0, grid_size, (5, 2))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final ecosystem state
        if len(prey_positions) > 0:
            ax1.scatter(prey_positions[:, 0], prey_positions[:, 1], 
                       c='green', s=20, alpha=0.7, label='Prey')
        if len(predator_positions) > 0:
            ax1.scatter(predator_positions[:, 0], predator_positions[:, 1], 
                       c='red', s=50, alpha=0.8, label='Predators')
        
        ax1.set_title(f'FSOT Ecosystem (S={s_scalar:.3f})')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.legend()
        ax1.set_xlim(0, grid_size)
        ax1.set_ylim(0, grid_size)
        
        # Population dynamics
        generations_range = range(len(prey_history))
        ax2.plot(generations_range, prey_history, 'g-', label='Prey', linewidth=2)
        ax2.plot(generations_range, predator_history, 'r-', label='Predators', linewidth=2)
        ax2.set_title('Population Dynamics')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Population')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save simulation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ecosystem_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate ecosystem metrics
        final_prey = len(prey_positions)
        final_predators = len(predator_positions)
        population_stability = 1 - (np.std(prey_history[-20:]) / np.mean(prey_history[-20:]) if np.mean(prey_history[-20:]) > 0 else 0)
        
        result = {
            'simulation_type': 'Ecosystem Evolution Simulation',
            'parameters': {
                'grid_size': grid_size,
                'generations': generations,
                'initial_prey': initial_prey,
                'initial_predators': initial_predators,
                's_scalar': s_scalar,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'final_prey_count': final_prey,
                'final_predator_count': final_predators,
                'population_stability': float(population_stability),
                'predator_prey_ratio': float(final_predators / max(1, final_prey))
            },
            'output_file': str(filepath),
            'data': {
                'prey_history': prey_history,
                'predator_history': predator_history,
                'final_prey_positions': prey_positions.tolist() if len(prey_positions) > 0 else [],
                'final_predator_positions': predator_positions.tolist() if len(predator_positions) > 0 else []
            },
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result

    def neural_network_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Neural network dynamics with FSOT consciousness integration.
        """
        if not ADVANCED_SIMS_AVAILABLE:
            return {'error': 'Advanced simulation libraries not available'}
        
        # Parameters
        num_nodes = params.get('num_nodes', 50)
        connection_prob = params.get('connection_prob', 0.1)
        simulation_time = params.get('simulation_time', 100)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Create network
        G = nx.erdos_renyi_graph(num_nodes, connection_prob)
        
        # Initialize node states (neural activations)
        node_states = np.array(np.random.uniform(0, 1, num_nodes))  # Explicit array conversion
        activation_history = [node_states.copy()]
        
        # FSOT-influenced network dynamics
        for t in range(simulation_time):
            new_states = node_states.copy()
            
            for node in G.nodes():
                # Get neighbor influences
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_influence = np.mean([node_states[n] for n in neighbors])
                    
                    # FSOT consciousness factor affects neural dynamics
                    consciousness_boost = consciousness_factor * s_scalar
                    decay_rate = 0.1 * (1 - s_scalar)
                    
                    # Update activation
                    new_states[node] = ((1 - decay_rate) * node_states[node] +
                                       0.1 * neighbor_influence +
                                       consciousness_boost * np.random.uniform(-0.1, 0.1))
                    
                    # Apply sigmoid activation
                    new_states[node] = 1 / (1 + np.exp(-5 * (new_states[node] - 0.5)))
            
            node_states = new_states
            if t % 10 == 0:
                activation_history.append(node_states.copy())
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Network structure
        pos = nx.spring_layout(G)
        node_colors = node_states
        nx.draw(G, pos, node_color=node_colors, node_size=100, 
                cmap='viridis', ax=ax1, with_labels=False)
        ax1.set_title(f'FSOT Neural Network (S={s_scalar:.3f})')
        
        # Activation dynamics
        time_steps = np.arange(0, len(activation_history) * 10, 10)
        for i in range(min(10, num_nodes)):  # Plot first 10 nodes
            ax2.plot(time_steps, [h[i] for h in activation_history], alpha=0.7)
        ax2.set_title('Neural Activation Dynamics')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Activation Level')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neural_network_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate network metrics
        avg_activation = np.mean(node_states)
        network_coherence = 1 - np.std(node_states)
        connectivity = nx.average_clustering(G)
        
        result = {
            'simulation_type': 'Neural Network Simulation',
            'parameters': {
                'num_nodes': num_nodes,
                'connection_prob': connection_prob,
                'simulation_time': simulation_time,
                's_scalar': s_scalar,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'average_activation': float(avg_activation),
                'network_coherence': float(network_coherence),
                'connectivity': float(connectivity),
                'final_variance': float(np.var(node_states))
            },
            'output_file': str(filepath),
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def default_fsot_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Default FSOT mathematical visualization.
        """
        # Parameters
        resolution = params.get('resolution', 100)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.5)
        d_eff = fsot_params.get('d_eff', 12)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Generate FSOT field visualization
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)
        
        # FSOT field equation (simplified)
        Z = (s_scalar * np.exp(-(X**2 + Y**2) / d_eff) *
             np.sin(consciousness_factor * np.sqrt(X**2 + Y**2)))
        
        # Create visualization
        fig = plt.figure(figsize=(12, 5))
        
        # 2D field
        ax1 = fig.add_subplot(121)
        contour = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax1.set_title(f'FSOT Field (S={s_scalar:.3f}, D_eff={d_eff})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(contour, ax=ax1)
        
        # 3D surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax2.set_title('FSOT 3D Field')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Field Strength')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fsot_field_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        result = {
            'simulation_type': 'FSOT Field Simulation',
            'parameters': {
                'resolution': resolution,
                's_scalar': s_scalar,
                'd_eff': d_eff,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'field_strength_max': float(np.max(Z)),
                'field_strength_min': float(np.min(Z)),
                'field_variance': float(np.var(Z))
            },
            'output_file': str(filepath),
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of all simulations run"""
        return {
            'total_simulations': len(self.simulation_history),
            'simulation_types': list(set([sim['simulation_type'] for sim in self.simulation_history])),
            'recent_simulations': self.simulation_history[-5:] if len(self.simulation_history) >= 5 else self.simulation_history,
            'output_directory': str(self.output_dir)
        }
