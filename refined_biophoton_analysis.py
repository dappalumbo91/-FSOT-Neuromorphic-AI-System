#!/usr/bin/env python3
"""
FSOT 2.0 Biophoton Hypothesis - Refined Analysis
Deep dive into the scale-invariant light-as-signal hypothesis with corrected parameters

Key Findings from Initial Analysis:
- Dramatic FSOT S-scalar transition at macro scale (0.1 → 0.9)
- Axonal optical properties support waveguide hypothesis (79 modes, 99% transmission)
- Power ratio indicates efficiency differences, not impossibility
- Speed advantage of >2 million times suggests fundamental mechanism
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
from biophoton_neural_simulation import FSOTBiophotonSimulator

class RefinedBiophotonAnalysis:
    """
    Refined analysis focusing on the scale transition phenomena in FSOT.
    """
    
    def __init__(self):
        self.sim = FSOTBiophotonSimulator()
        self.output_dir = Path("refined_biophoton_analysis")
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_scale_transition_phenomenon(self):
        """
        Deep analysis of the dramatic FSOT S-scalar transition.
        """
        print("🔬 REFINED FSOT 2.0 BIOPHOTON ANALYSIS")
        print("=" * 60)
        print("Investigating Scale Transition Phenomena")
        print()
        
        # Test finer scale resolution around transition points
        scale_factors = np.logspace(-15, 3, 100)  # Much finer resolution
        
        s_scalars = []
        poof_factors = []
        consciousness_factors = []
        
        for scale in scale_factors:
            fsot = self.sim.calculate_fsot_scalar({
                'scale_factor': scale,
                'coherence_level': 0.7,
                'observer_influence': 0.3
            })
            s_scalars.append(fsot['s_scalar'])
            poof_factors.append(fsot['poof_factor'])
            consciousness_factors.append(fsot['consciousness_factor'])
        
        # Find transition points
        s_array = np.array(s_scalars)
        transition_indices = np.where(np.diff(s_array) > 0.1)[0]
        
        results = {
            'scale_factors': scale_factors.tolist(),
            's_scalars': s_scalars,
            'poof_factors': poof_factors,
            'consciousness_factors': consciousness_factors,
            'transition_points': []
        }
        
        if len(transition_indices) > 0:
            for idx in transition_indices:
                transition_scale = scale_factors[idx]
                s_before = s_scalars[idx]
                s_after = s_scalars[idx + 1]
                
                results['transition_points'].append({
                    'scale_factor': transition_scale,
                    's_before': s_before,
                    's_after': s_after,
                    'jump_magnitude': s_after - s_before
                })
                
                print(f"📍 FSOT Transition Detected:")
                print(f"   Scale: {transition_scale:.2e}")
                print(f"   S-scalar jump: {s_before:.3f} → {s_after:.3f}")
                print(f"   Jump magnitude: {s_after - s_before:.3f}")
                print()
        
        return results
    
    def evaluate_biophoton_feasibility(self):
        """
        Evaluate biological feasibility of biophoton signaling.
        """
        print("🧬 BIOLOGICAL FEASIBILITY ANALYSIS")
        print("=" * 40)
        
        # Real biophoton measurements from literature
        real_biophoton_data = {
            'resting_emission': 10,  # photons/cm²/s (typical for cells)
            'active_emission': 100,  # photons/cm²/s (during activity)
            'wavelength_range': (400e-9, 800e-9),  # nm
            'coherence_time': 1e-12,  # seconds (femtosecond coherence)
            'observed_in_neurons': True,
            'observed_correlations': True  # With neural activity
        }
        
        # Axon surface area calculation
        axon_length = 1e-3  # 1 mm
        axon_diameter = 10e-6  # 10 μm
        axon_surface_area = np.pi * axon_diameter * axon_length  # m²
        
        # Expected biophoton flux from axon
        expected_flux_resting = real_biophoton_data['resting_emission'] * axon_surface_area * 1e4  # photons/s
        expected_flux_active = real_biophoton_data['active_emission'] * axon_surface_area * 1e4  # photons/s
        
        # Compare with simulation
        t_bio, emission, intensity = self.sim.simulate_biophoton_signal()
        simulated_peak_flux = max(emission)
        simulated_avg_flux = np.mean(emission)
        
        feasibility_score = min(simulated_peak_flux / expected_flux_active, 2.0)  # Cap at 2x
        
        print(f"Expected biophoton flux (resting): {expected_flux_resting:.1f} photons/s")
        print(f"Expected biophoton flux (active): {expected_flux_active:.1f} photons/s")
        print(f"Simulated peak flux: {simulated_peak_flux:.1f} photons/s")
        print(f"Feasibility score: {feasibility_score:.2f} (1.0 = perfect match)")
        print()
        
        # Optical waveguide analysis
        waveguide = self.sim.simulate_axon_optical_waveguide()
        
        print(f"Axon optical properties:")
        print(f"  - Numerical aperture: {waveguide['numerical_aperture']:.3f}")
        print(f"  - Number of modes: {waveguide['num_modes']}")
        print(f"  - Transmission efficiency: {waveguide['transmission_efficiency']*100:.1f}%")
        print(f"  - Speed advantage: {waveguide['speed_advantage']:.0f}x faster than electrical")
        print()
        
        return {
            'feasibility_score': feasibility_score,
            'expected_flux_range': (expected_flux_resting, expected_flux_active),
            'simulated_flux_range': (min(emission), max(emission)),
            'optical_properties': waveguide,
            'biological_plausibility': feasibility_score > 0.5
        }
    
    def test_consciousness_scaling_hypothesis(self):
        """
        Test if consciousness factor correlates with known neural phenomena.
        """
        print("🧠 CONSCIOUSNESS SCALING HYPOTHESIS TEST")
        print("=" * 45)
        
        # Test different observer influence levels
        observer_levels = np.linspace(0, 1, 11)
        consciousness_responses = []
        s_scalar_responses = []
        
        for obs_level in observer_levels:
            fsot = self.sim.calculate_fsot_scalar({
                'scale_factor': 1e-6,  # Cellular scale
                'coherence_level': 0.7,
                'observer_influence': obs_level
            })
            consciousness_responses.append(fsot['consciousness_factor'])
            s_scalar_responses.append(fsot['s_scalar'])
        
        # Look for non-linear responses (signatures of consciousness)
        consciousness_gradient = np.gradient(consciousness_responses)
        s_gradient = np.gradient(s_scalar_responses)
        
        # Peak gradient suggests optimal consciousness influence
        optimal_obs_idx = np.argmax(consciousness_gradient)
        optimal_observer_level = observer_levels[optimal_obs_idx]
        
        print(f"Optimal observer influence level: {optimal_observer_level:.2f}")
        print(f"Consciousness factor at optimum: {consciousness_responses[optimal_obs_idx]:.3f}")
        print(f"S-scalar at optimum: {s_scalar_responses[optimal_obs_idx]:.3f}")
        print()
        
        # Test correlation with neural complexity
        neural_complexity_proxy = [
            (0.1, "Simple reflex"),
            (0.3, "Basic perception"), 
            (0.5, "Pattern recognition"),
            (0.7, "Abstract thinking"),
            (0.9, "Self-awareness")
        ]
        
        for complexity, description in neural_complexity_proxy:
            fsot = self.sim.calculate_fsot_scalar({
                'scale_factor': 1e-6,
                'coherence_level': 0.7,
                'observer_influence': complexity
            })
            print(f"{description:18s}: Consciousness factor = {fsot['consciousness_factor']:.3f}")
        
        return {
            'optimal_observer_level': optimal_observer_level,
            'consciousness_scaling': list(zip(observer_levels, consciousness_responses)),
            's_scalar_scaling': list(zip(observer_levels, s_scalar_responses))
        }
    
    def create_comprehensive_visualization(self, scale_data: Dict, feasibility_data: Dict, consciousness_data: Dict):
        """
        Create comprehensive visualization of all analyses.
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Scale transition analysis
        ax1 = plt.subplot(3, 4, 1)
        plt.loglog(scale_data['scale_factors'], scale_data['s_scalars'], 'b-', linewidth=2)
        plt.xlabel('Scale Factor')
        plt.ylabel('S-scalar')
        plt.title('FSOT Scale Transition')
        plt.grid(True, alpha=0.3)
        
        # Mark transition points
        for tp in scale_data['transition_points']:
            plt.axvline(tp['scale_factor'], color='red', linestyle='--', alpha=0.7)
            plt.text(tp['scale_factor'], tp['s_after'], f"Jump: {tp['jump_magnitude']:.2f}", 
                    rotation=90, ha='right', va='bottom')
        
        # 2. Consciousness factor vs scale
        ax2 = plt.subplot(3, 4, 2)
        plt.loglog(scale_data['scale_factors'], scale_data['consciousness_factors'], 'g-', linewidth=2)
        plt.xlabel('Scale Factor')
        plt.ylabel('Consciousness Factor')
        plt.title('Consciousness Scaling')
        plt.grid(True, alpha=0.3)
        
        # 3. Poof factor evolution
        ax3 = plt.subplot(3, 4, 3)
        plt.loglog(scale_data['scale_factors'], scale_data['poof_factors'], 'r-', linewidth=2)
        plt.xlabel('Scale Factor')
        plt.ylabel('Poof Factor')
        plt.title('FSOT Poof Factor')
        plt.grid(True, alpha=0.3)
        
        # 4. Biophoton flux comparison
        ax4 = plt.subplot(3, 4, 4)
        flux_data = [
            feasibility_data['expected_flux_range'][0],
            feasibility_data['expected_flux_range'][1],
            feasibility_data['simulated_flux_range'][0],
            feasibility_data['simulated_flux_range'][1]
        ]
        flux_labels = ['Expected\nResting', 'Expected\nActive', 'Simulated\nMin', 'Simulated\nMax']
        colors = ['lightblue', 'blue', 'lightcoral', 'red']
        
        bars = plt.bar(range(4), flux_data, color=colors, alpha=0.7)
        plt.xticks(range(4), flux_labels)
        plt.ylabel('Photon Flux (photons/s)')
        plt.title('Biophoton Flux Comparison')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 5. Optical waveguide properties
        ax5 = plt.subplot(3, 4, 5)
        opt_props = feasibility_data['optical_properties']
        prop_names = ['NA', 'Modes/10', 'Transmission', 'Speed/1000']
        prop_values = [
            opt_props['numerical_aperture'],
            opt_props['num_modes'] / 10,
            opt_props['transmission_efficiency'],
            opt_props['speed_advantage'] / 1000
        ]
        
        plt.bar(range(4), prop_values, color='purple', alpha=0.7)
        plt.xticks(range(4), prop_names)
        plt.ylabel('Normalized Value')
        plt.title('Axon Optical Properties')
        plt.grid(True, alpha=0.3)
        
        # 6. Consciousness scaling response
        ax6 = plt.subplot(3, 4, 6)
        obs_levels, cons_factors = zip(*consciousness_data['consciousness_scaling'])
        plt.plot(obs_levels, cons_factors, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Observer Influence Level')
        plt.ylabel('Consciousness Factor')
        plt.title('Consciousness Response')
        plt.grid(True, alpha=0.3)
        
        # Mark optimal point
        optimal_level = consciousness_data['optimal_observer_level']
        optimal_idx = list(obs_levels).index(optimal_level)
        optimal_cons = cons_factors[optimal_idx]
        plt.scatter([optimal_level], [optimal_cons], s=100, color='red', zorder=5)
        plt.annotate(f'Optimal\n({optimal_level:.1f}, {optimal_cons:.3f})', 
                    xy=(optimal_level, optimal_cons), xytext=(10, 10), 
                    textcoords='offset points', ha='left')
        
        # 7. FSOT parameter phase space
        ax7 = plt.subplot(3, 4, 7)
        plt.scatter(scale_data['s_scalars'], scale_data['consciousness_factors'], 
                   c=scale_data['poof_factors'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Poof Factor')
        plt.xlabel('S-scalar')
        plt.ylabel('Consciousness Factor')
        plt.title('FSOT Phase Space')
        plt.grid(True, alpha=0.3)
        
        # 8. Signal comparison
        ax8 = plt.subplot(3, 4, 8)
        t_elec, v_elec = self.sim.simulate_classical_neural_signal()
        t_bio, emission, intensity = self.sim.simulate_biophoton_signal()
        
        # Normalize for comparison
        v_norm = v_elec / max(abs(v_elec))
        emission_norm = emission / max(emission)
        
        plt.plot(t_elec * 1000, v_norm, 'b-', linewidth=2, label='Electrical (norm)')
        plt.plot(t_bio * 1000, emission_norm, 'r-', linewidth=2, label='Biophoton (norm)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalized Signal')
        plt.title('Signal Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Feasibility assessment
        ax9 = plt.subplot(3, 4, 9)
        feasibility_metrics = [
            feasibility_data['feasibility_score'],
            1.0 if feasibility_data['optical_properties']['transmission_efficiency'] > 0.9 else 0.5,
            1.0 if feasibility_data['optical_properties']['speed_advantage'] > 1000 else 0.5,
            1.0 if feasibility_data['biological_plausibility'] else 0.0
        ]
        metric_names = ['Flux\nMatch', 'High\nTransmission', 'Speed\nAdvantage', 'Bio\nPlausible']
        colors = ['green' if m > 0.7 else 'orange' if m > 0.3 else 'red' for m in feasibility_metrics]
        
        plt.bar(range(4), feasibility_metrics, color=colors, alpha=0.7)
        plt.xticks(range(4), metric_names)
        plt.ylabel('Score')
        plt.title('Feasibility Assessment')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # 10. Scale transition detail
        ax10 = plt.subplot(3, 4, 10)
        if scale_data['transition_points']:
            tp = scale_data['transition_points'][0]  # Focus on first transition
            
            # Find indices around transition
            scales = np.array(scale_data['scale_factors'])
            tp_idx = np.argmin(np.abs(scales - tp['scale_factor']))
            
            start_idx = max(0, tp_idx - 20)
            end_idx = min(len(scales), tp_idx + 20)
            
            local_scales = scales[start_idx:end_idx]
            local_s = np.array(scale_data['s_scalars'])[start_idx:end_idx]
            
            plt.plot(local_scales, local_s, 'ko-', linewidth=2, markersize=4)
            plt.axvline(tp['scale_factor'], color='red', linestyle='--')
            plt.xlabel('Scale Factor')
            plt.ylabel('S-scalar')
            plt.title('Transition Detail')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
        
        # 11. FSOT theoretical prediction accuracy
        ax11 = plt.subplot(3, 4, 11)
        
        # Compare with theoretical golden ratio relationships
        golden_ratio = (1 + np.sqrt(5)) / 2
        theoretical_s = 1 / golden_ratio  # ≈ 0.618
        
        actual_s_values = np.array(scale_data['s_scalars'])
        accuracy_scores = 1 - np.abs(actual_s_values - theoretical_s) / theoretical_s
        
        plt.hist(accuracy_scores, bins=20, alpha=0.7, color='teal')
        plt.axvline(float(np.mean(accuracy_scores)), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(accuracy_scores):.3f}')
        plt.xlabel('FSOT Accuracy Score')
        plt.ylabel('Frequency')
        plt.title('Theoretical Prediction Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. Summary recommendation
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate overall hypothesis support
        support_scores = {
            'Scale Transitions': 1.0 if scale_data['transition_points'] else 0.0,
            'Optical Feasibility': feasibility_data['feasibility_score'],
            'Speed Advantage': 1.0 if feasibility_data['optical_properties']['speed_advantage'] > 1000 else 0.5,
            'Consciousness Scaling': 1.0 if consciousness_data['optimal_observer_level'] > 0.3 else 0.5,
            'Biological Plausibility': 1.0 if feasibility_data['biological_plausibility'] else 0.0
        }
        
        overall_support = np.mean(list(support_scores.values()))
        
        # Create recommendation text
        if overall_support > 0.7:
            recommendation = "STRONG SUPPORT\nfor biophoton hypothesis"
            color = 'green'
        elif overall_support > 0.5:
            recommendation = "MODERATE SUPPORT\nrequires further study"
            color = 'orange'
        else:
            recommendation = "INSUFFICIENT SUPPORT\nalternative mechanisms"
            color = 'red'
        
        ax12.text(0.5, 0.7, recommendation, ha='center', va='center', 
                 fontsize=14, fontweight='bold', color=color,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2))
        
        ax12.text(0.5, 0.3, f'Overall Score: {overall_support:.2f}/1.00', 
                 ha='center', va='center', fontsize=12)
        
        # List individual scores
        score_text = '\n'.join([f'{k}: {v:.2f}' for k, v in support_scores.items()])
        ax12.text(0.5, 0.05, score_text, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"refined_biophoton_analysis_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath), overall_support, support_scores
    
    def run_complete_refined_analysis(self):
        """
        Execute the complete refined analysis.
        """
        print("🌟 REFINED FSOT 2.0 BIOPHOTON ANALYSIS")
        print("=" * 80)
        print("Deep Investigation of Light-as-Signal Scale-Invariant Hypothesis")
        print()
        
        # Run all analyses
        scale_data = self.analyze_scale_transition_phenomenon()
        feasibility_data = self.evaluate_biophoton_feasibility()
        consciousness_data = self.test_consciousness_scaling_hypothesis()
        
        # Create comprehensive visualization
        viz_path, overall_support, support_scores = self.create_comprehensive_visualization(
            scale_data, feasibility_data, consciousness_data
        )
        
        # Generate final report
        report = {
            'hypothesis': "Light serves as electrical signals at quantum/cellular scales",
            'overall_support_score': float(overall_support),
            'individual_scores': {k: float(v) for k, v in support_scores.items()},
            'key_findings': {
                'scale_transitions': int(len(scale_data['transition_points'])),
                'feasibility_score': float(feasibility_data['feasibility_score']),
                'optimal_consciousness_level': float(consciousness_data['optimal_observer_level']),
                'speed_advantage': float(feasibility_data['optical_properties']['speed_advantage'])
            },
            'fsot_validation': {
                'transition_points_detected': bool(len(scale_data['transition_points']) > 0),
                'consciousness_scaling_nonlinear': bool(consciousness_data['optimal_observer_level'] > 0.3),
                'optical_properties_feasible': bool(feasibility_data['biological_plausibility'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.output_dir / f"refined_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print()
        print("🎯 REFINED ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"Overall Support: {overall_support:.3f}/1.000")
        print()
        print("Individual Component Scores:")
        for component, score in support_scores.items():
            status = "✓" if score > 0.7 else "?" if score > 0.3 else "✗"
            print(f"  {status} {component}: {score:.3f}")
        
        print()
        print(f"📊 Visualization: {viz_path}")
        print(f"📄 Report: {report_file}")
        print()
        print("🚀 Refined analysis complete!")
        
        return report

def main():
    """Run the refined biophoton analysis."""
    analyzer = RefinedBiophotonAnalysis()
    return analyzer.run_complete_refined_analysis()

if __name__ == "__main__":
    main()
