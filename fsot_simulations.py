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
        germ_states = np.random.uniform(0, 1, (num_germs,))  # Quantum state probability
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
                'final_states': germ_states.tolist(),
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
        node_states = np.random.uniform(0, 1, (num_nodes,))
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
    
    def particle_physics_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Particle physics simulation with FSOT field interactions.
        Models particle dynamics under FSOT scalar field influence.
        """
        
        # Extract parameters
        num_particles = params.get('num_particles', 100)
        simulation_time = params.get('simulation_time', 200)
        particle_mass = params.get('particle_mass', 1.0)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.618)
        d_eff = fsot_params.get('d_eff', 12)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        
        # Initialize particles
        positions = np.random.uniform(-10, 10, (num_particles, 3))
        velocities = np.random.normal(0, 1, (num_particles, 3))
        masses = np.full(num_particles, particle_mass)
        
        position_history = [positions.copy()]
        energy_history = []
        
        # Physics simulation with FSOT influence
        dt = 0.01
        for t in range(simulation_time):
            # FSOT field influence on particles
            fsot_field = s_scalar * np.exp(-np.linalg.norm(positions, axis=1) / d_eff)
            
            # Calculate forces
            forces = np.zeros_like(positions)
            
            # Inter-particle forces (simplified gravitational)
            for i in range(num_particles):
                for j in range(i+1, num_particles):
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 0.1:  # Avoid singularity
                        force_magnitude = masses[i] * masses[j] / (r_mag**2) * consciousness_factor
                        force_vec = force_magnitude * r_vec / r_mag
                        forces[i] += force_vec
                        forces[j] -= force_vec
            
            # FSOT field force
            for i in range(num_particles):
                field_force = fsot_field[i] * s_scalar * positions[i] / np.linalg.norm(positions[i])
                forces[i] += field_force * consciousness_factor
            
            # Update velocities and positions
            accelerations = forces / masses[:, np.newaxis]
            velocities += accelerations * dt
            positions += velocities * dt
            
            # Calculate total energy
            kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
            potential_energy = np.sum(fsot_field) * s_scalar
            total_energy = kinetic_energy + potential_energy
            
            position_history.append(positions.copy())
            energy_history.append(total_energy)
        
        # Create visualization
        if ADVANCED_SIMS_AVAILABLE:
            fig = plt.figure(figsize=(15, 5))
            
            # 3D particle trajectories
            ax1 = fig.add_subplot(131, projection='3d')
            for i in range(min(10, num_particles)):  # Show only first 10 for clarity
                trajectory = np.array([pos[i] for pos in position_history])
                ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.7)
            ax1.set_title('Particle Trajectories')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Energy evolution
            ax2 = fig.add_subplot(132)
            ax2.plot(energy_history)
            ax2.set_title('System Energy Evolution')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Total Energy')
            ax2.grid(True, alpha=0.3)
            
            # Final particle distribution
            ax3 = fig.add_subplot(133)
            ax3.scatter(positions[:, 0], positions[:, 1], s=masses*10, alpha=0.6)
            ax3.set_title('Final Particle Distribution')
            ax3.set_xlabel('X Position')
            ax3.set_ylabel('Y Position')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(energy_history)
            ax.set_title('Particle Physics Simulation - Energy Evolution')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Total Energy')
            ax.grid(True, alpha=0.3)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"particle_physics_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        result = {
            'simulation_type': 'Particle Physics Simulation',
            'parameters': {
                'num_particles': num_particles,
                'simulation_time': simulation_time,
                'particle_mass': particle_mass,
                's_scalar': s_scalar,
                'd_eff': d_eff,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'final_energy': float(energy_history[-1]),
                'energy_variance': float(np.var(energy_history)),
                'particle_spread': float(np.std(np.linalg.norm(positions, axis=1)))
            },
            'output_file': str(filepath),
            'data': {
                'energy_history': energy_history,
                'final_positions': positions.tolist()
            },
            'timestamp': timestamp
        }
        
        self.simulation_history.append(result)
        return result
    
    def ecosystem_simulation(self, params: Dict[str, Any], fsot_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Ecosystem evolution simulation with FSOT consciousness factors.
        Models species interactions and evolution under FSOT influence.
        """
        
        # Extract parameters
        num_species = params.get('num_species', 5)
        population_size = params.get('population_size', 1000)
        simulation_time = params.get('simulation_time', 500)
        mutation_rate = params.get('mutation_rate', 0.01)
        
        # FSOT parameters
        s_scalar = fsot_params.get('s_scalar', 0.618)
        consciousness_factor = fsot_params.get('consciousness_factor', 0.3)
        d_eff = fsot_params.get('d_eff', 12)
        
        # Initialize species with traits
        species_populations = np.random.randint(50, 200, num_species)
        species_traits = np.random.uniform(0, 1, (num_species, 3))  # [aggression, cooperation, adaptation]
        species_fitness = np.ones(num_species)
        
        population_history = [species_populations.copy()]
        fitness_history = [species_fitness.copy()]
        trait_history = [species_traits.copy()]
        
        # Ecosystem simulation with FSOT influence
        for t in range(simulation_time):
            # FSOT consciousness influence on species behavior
            consciousness_boost = consciousness_factor * s_scalar * np.exp(-t / d_eff)
            
            # Calculate species interactions
            new_populations = species_populations.copy().astype(float)
            new_traits = species_traits.copy()
            
            for i in range(num_species):
                if species_populations[i] > 0:
                    # Environmental carrying capacity influenced by FSOT
                    carrying_capacity = 150 * (1 + consciousness_boost)
                    
                    # Growth rate based on traits and FSOT consciousness
                    cooperation_bonus = species_traits[i, 1] * consciousness_boost
                    adaptation_bonus = species_traits[i, 2] * s_scalar
                    growth_rate = 0.1 * (1 + cooperation_bonus + adaptation_bonus)
                    
                    # Population dynamics
                    growth = growth_rate * species_populations[i] * (1 - species_populations[i] / carrying_capacity)
                    
                    # Competition with other species
                    competition = 0
                    for j in range(num_species):
                        if i != j and species_populations[j] > 0:
                            trait_difference = np.linalg.norm(species_traits[i] - species_traits[j])
                            competition += species_populations[j] * (1 - trait_difference) * 0.01
                    
                    # Update population
                    new_populations[i] = max(0, species_populations[i] + growth - competition)
                    
                    # Trait evolution (mutation influenced by consciousness)
                    if np.random.random() < mutation_rate * (1 + consciousness_boost):
                        mutation_strength = 0.05 * consciousness_factor
                        new_traits[i] += np.random.normal(0, mutation_strength, 3)
                        new_traits[i] = np.clip(new_traits[i], 0, 1)
            
            # Update fitness based on population stability and traits
            for i in range(num_species):
                if new_populations[i] > 0:
                    stability = 1 - abs(new_populations[i] - species_populations[i]) / max(species_populations[i], 1)
                    trait_balance = 1 - np.std(new_traits[i])
                    species_fitness[i] = 0.5 * stability + 0.3 * trait_balance + 0.2 * consciousness_boost
                else:
                    species_fitness[i] = 0
            
            species_populations = new_populations.astype(int)
            species_traits = new_traits
            
            population_history.append(species_populations.copy())
            fitness_history.append(species_fitness.copy())
            trait_history.append(species_traits.copy())
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Population evolution
        for i in range(num_species):
            pops = [pop[i] for pop in population_history]
            ax1.plot(pops, label=f'Species {i+1}', alpha=0.8)
        ax1.set_title('Species Population Evolution')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Population')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fitness evolution
        for i in range(num_species):
            fitness = [fit[i] for fit in fitness_history]
            ax2.plot(fitness, label=f'Species {i+1}', alpha=0.8)
        ax2.set_title('Species Fitness Evolution')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final trait distribution
        final_traits = trait_history[-1]
        trait_names = ['Aggression', 'Cooperation', 'Adaptation']
        x = np.arange(len(trait_names))
        width = 0.15
        for i in range(num_species):
            if species_populations[i] > 0:
                ax3.bar(x + i*width, final_traits[i], width, label=f'Species {i+1}', alpha=0.8)
        ax3.set_title('Final Trait Distribution')
        ax3.set_xlabel('Traits')
        ax3.set_ylabel('Trait Value')
        ax3.set_xticks(x + width * (num_species-1) / 2)
        ax3.set_xticklabels(trait_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Biodiversity over time
        biodiversity = [np.sum(pop > 0) for pop in population_history]
        ax4.plot(biodiversity, 'g-', linewidth=2)
        ax4.set_title('Biodiversity Evolution')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Number of Active Species')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ecosystem_sim_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        result = {
            'simulation_type': 'Ecosystem Evolution Simulation',
            'parameters': {
                'num_species': num_species,
                'initial_population': population_size,
                'simulation_time': simulation_time,
                'mutation_rate': mutation_rate,
                's_scalar': s_scalar,
                'consciousness_factor': consciousness_factor
            },
            'metrics': {
                'final_biodiversity': int(np.sum(species_populations > 0)),
                'total_population': int(np.sum(species_populations)),
                'average_fitness': float(np.mean(species_fitness[species_populations > 0])) if np.any(species_populations > 0) else 0.0,
                'trait_diversity': float(np.std(final_traits[species_populations > 0])) if np.any(species_populations > 0) else 0.0
            },
            'output_file': str(filepath),
            'data': {
                'population_history': [pop.tolist() for pop in population_history[-10:]],  # Last 10 steps
                'final_populations': species_populations.tolist(),
                'final_traits': final_traits.tolist()
            },
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
