"""
#!/usr/bin/env python3
"""
Validation 1.3 RIGETTI SCALE-DEPENDENT PARAMETER OPTIMIZATION
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Find if optimal parameters shift at larger scales
Budget-conscious approach focusing on key transition points

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
from braket.circuits import Circuit
from braket.aws import AwsDevice
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import json
from datetime import datetime
import networkx as nx

class ScaleDependentOptimization:
    def __init__(self, budget_remaining=172, previous_results=None):
        self.budget_remaining = budget_remaining
        self.spent = 0
        
        # Rigetti pricing
        self.cost_per_shot = 0.00035
        self.cost_per_task = 0.30
        
        self.device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
        
        # Known parameters at different scales
        self.known_params = {
            5: (2.693, 1.885),  # Your measured optimum
        }
        
        # Previous performance data
        self.previous_results = previous_results or {
            3: 0.790, 5: 0.720, 7: 0.643, 10: 0.589, 12: 0.571,
            15: 0.602, 20: 0.560, 25: 0.557, 30: 0.537, 35: 0.556, 40: 0.567
        }
    
    def smart_parameter_search(self, n_qubits, base_gamma=2.693, base_beta=1.885):
        """
        Intelligent search around base parameters
        Uses coarse-to-fine approach
        """
        print(f"\n{n_qubits} QUBIT PARAMETER SEARCH")
        print("-"*40)
        
        graph = self.create_graph(n_qubits)
        
        # Phase 1: Coarse scan (9 points)
        coarse_range = 0.8  # Wider initial search
        gamma_coarse = np.linspace(base_gamma - coarse_range, base_gamma + coarse_range, 3)
        beta_coarse = np.linspace(base_beta - coarse_range/2, base_beta + coarse_range/2, 3)
        
        best_coarse = (base_gamma, base_beta)
        best_performance = self.previous_results.get(n_qubits, 0)
        
        print(f"Coarse scan (9 points):")
        for gamma in gamma_coarse:
            for beta in beta_coarse:
                # Skip if too close to base (already measured)
                if abs(gamma - base_gamma) < 0.1 and abs(beta - base_beta) < 0.1:
                    continue
                
                # Test with minimal shots
                circuit = self.create_qaoa_circuit(gamma, beta, graph)
                shots = 30  # Minimal for coarse scan
                
                cost = self.cost_per_task + shots * self.cost_per_shot
                if self.spent + cost > self.budget_remaining:
                    print("  Budget limit reached")
                    return best_coarse, best_performance
                
                task = self.device.run(circuit, shots=shots)
                result = task.result()
                
                # Quick performance estimate
                measurements = result.measurements
                cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
                performance = np.mean(cuts) / (graph.number_of_edges() / 2)
                
                self.spent += cost
                
                if performance > best_performance:
                    best_performance = performance
                    best_coarse = (gamma, beta)
                    print(f"  New best: γ={gamma:.3f}, β={beta:.3f}, perf={performance:.3f}")
        
        # Phase 2: Fine search around best (if improvement found)
        if best_performance > self.previous_results.get(n_qubits, 0) * 1.05:  # 5% improvement threshold
            print(f"\nFine-tuning around γ={best_coarse[0]:.3f}, β={best_coarse[1]:.3f}")
            
            fine_range = 0.2
            gamma_fine = np.linspace(best_coarse[0] - fine_range, best_coarse[0] + fine_range, 3)
            beta_fine = np.linspace(best_coarse[1] - fine_range/2, best_coarse[1] + fine_range/2, 3)
            
            for gamma in gamma_fine:
                for beta in beta_fine:
                    circuit = self.create_qaoa_circuit(gamma, beta, graph)
                    shots = 100  # More shots for fine tuning
                    
                    cost = self.cost_per_task + shots * self.cost_per_shot
                    if self.spent + cost > self.budget_remaining:
                        return best_coarse, best_performance
                    
                    task = self.device.run(circuit, shots=shots)
                    result = task.result()
                    
                    measurements = result.measurements
                    cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
                    performance = np.mean(cuts) / (graph.number_of_edges() / 2)
                    
                    self.spent += cost
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_coarse = (gamma, beta)
                        print(f"  Refined: γ={gamma:.3f}, β={beta:.3f}, perf={performance:.3f}")
        
        return best_coarse, best_performance
    
    def run_strategic_optimization(self):
        """
        Main optimization at key scales
        """
        print("="*60)
        print("SCALE-DEPENDENT PARAMETER OPTIMIZATION")
        print("="*60)
        print(f"Budget: €{self.budget_remaining:.2f}")
        
        # Strategic scale selection
        # Focus on transition regions where performance dropped
        key_scales = [
            10,  # First major drop (0.589)
            20,  # Mid-scale (0.560)
            30,  # Large scale (0.537)
        ]
        
        # Estimate costs
        estimated_cost = len(key_scales) * 9 * (self.cost_per_task + 30 * self.cost_per_shot)
        print(f"Estimated cost for coarse scan: €{estimated_cost:.2f}")
        
        if estimated_cost > self.budget_remaining * 0.5:
            print("Reducing to 2 key scales to preserve budget")
            key_scales = [15, 30]  # Most important transition points
        
        optimal_params = {}
        improved_performance = {}
        
        for n_qubits in key_scales:
            # Use previous scale's optimum as starting point
            prev_scale = max([s for s in self.known_params.keys() if s < n_qubits], default=5)
            base_gamma, base_beta = self.known_params[prev_scale]
            
            # Search for better parameters
            best_params, best_perf = self.smart_parameter_search(n_qubits, base_gamma, base_beta)
            
            optimal_params[n_qubits] = best_params
            improved_performance[n_qubits] = best_perf
            self.known_params[n_qubits] = best_params
            
            print(f"\n{n_qubits} qubits: Best performance = {best_perf:.3f}")
            print(f"  Previous: {self.previous_results.get(n_qubits, 0):.3f}")
            print(f"  Improvement: {(best_perf/self.previous_results.get(n_qubits, 1) - 1)*100:.1f}%")
            print(f"  Total spent: €{self.spent:.2f}")
            
            # Stop if budget running low
            if self.spent > self.budget_remaining * 0.8:
                print("\nApproaching budget limit - stopping optimization")
                break
        
        return optimal_params, improved_performance
    
    def validate_improvements(self, optimal_params):
        """
        Validate the new parameters with higher statistics
        """
        print("\n" + "="*60)
        print("VALIDATION WITH OPTIMIZED PARAMETERS")
        print("="*60)
        
        validation_scales = [20, 30, 40]  # Key scales to validate
        validated_results = {}
        
        for n_qubits in validation_scales:
            # Find nearest optimized params
            if n_qubits in optimal_params:
                gamma, beta = optimal_params[n_qubits]
            else:
                # Interpolate from nearest
                scales = sorted(optimal_params.keys())
                if n_qubits < min(scales):
                    gamma, beta = optimal_params[min(scales)]
                elif n_qubits > max(scales):
                    gamma, beta = optimal_params[max(scales)]
                else:
                    # Linear interpolation
                    lower = max([s for s in scales if s < n_qubits])
                    upper = min([s for s in scales if s > n_qubits])
                    alpha = (n_qubits - lower) / (upper - lower)
                    gamma = (1-alpha) * optimal_params[lower][0] + alpha * optimal_params[upper][0]
                    beta = (1-alpha) * optimal_params[lower][1] + alpha * optimal_params[upper][1]
            
            print(f"\n{n_qubits} qubits with γ={gamma:.3f}, β={beta:.3f}")
            
            graph = self.create_graph(n_qubits)
            circuit = self.create_qaoa_circuit(gamma, beta, graph)
            
            shots = 200
            cost = self.cost_per_task + shots * self.cost_per_shot
            
            if self.spent + cost > self.budget_remaining:
                print("  Budget exceeded - skipping")
                continue
            
            task = self.device.run(circuit, shots=shots)
            result = task.result()
            
            measurements = result.measurements
            cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
            performance = np.mean(cuts) / (graph.number_of_edges() / 2)
            
            validated_results[n_qubits] = performance
            self.spent += cost
            
            print(f"  Performance: {performance:.3f}")
            print(f"  Previous: {self.previous_results.get(n_qubits, 0):.3f}")
            print(f"  Change: {(performance - self.previous_results.get(n_qubits, 0))*100:+.1f}%")
        
        return validated_results
    
    def create_graph(self, n_qubits):
        """Create appropriate graph for n qubits"""
        if n_qubits <= 2:
            return nx.path_graph(n_qubits)
        
        # Ensure n*d is even
        for degree in [3, 2, 4]:
            if (n_qubits * degree) % 2 == 0 and degree < n_qubits:
                try:
                    return nx.random_regular_graph(degree, n_qubits, seed=42)
                except:
                    continue
        
        # Fallback
        return nx.cycle_graph(n_qubits) if n_qubits % 2 == 0 else nx.star_graph(n_qubits-1)
    
    def create_qaoa_circuit(self, gamma, beta, graph):
        """Standard QAOA circuit"""
        n_qubits = graph.number_of_nodes()
        circuit = Circuit()
        
        for i in range(n_qubits):
            circuit.h(i)
        
        for u, v in graph.edges():
            circuit.cnot(u, v)
            circuit.rz(v, 2*gamma)
            circuit.cnot(u, v)
        
        for i in range(n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def create_final_report(self, optimal_params, improved_performance, validated_results):
        """Create comprehensive visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Parameter evolution
        scales = sorted(optimal_params.keys())
        gammas = [optimal_params[s][0] for s in scales]
        betas = [optimal_params[s][1] for s in scales]
        
        ax1.plot(scales, gammas, 'go-', label='γ', markersize=8, linewidth=2)
        ax1.plot(scales, betas, 'bo-', label='β', markersize=8, linewidth=2)
        ax1.axhline(y=2.693, color='g', linestyle='--', alpha=0.3, label='Original γ')
        ax1.axhline(y=1.885, color='b', linestyle='--', alpha=0.3, label='Original β')
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('Parameter Value')
        ax1.set_title('Scale-Dependent Optimal Parameters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Performance comparison
        all_scales = sorted(set(list(self.previous_results.keys()) + list(validated_results.keys())))
        old_perf = [self.previous_results.get(s, np.nan) for s in all_scales]
        new_perf = [validated_results.get(s, improved_performance.get(s, np.nan)) for s in all_scales]
        
        ax2.plot(all_scales, old_perf, 'r--o', label='Fixed params', markersize=6, alpha=0.7)
        ax2.plot(all_scales, new_perf, 'g-o', label='Optimized params', markersize=8)
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Approximation Ratio')
        ax2.set_title('Performance: Fixed vs Optimized Parameters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Improvement percentages
        improvements = []
        imp_scales = []
        for s in all_scales:
            if s in validated_results or s in improved_performance:
                old = self.previous_results.get(s, 0)
                new = validated_results.get(s, improved_performance.get(s, old))
                if old > 0:
                    improvements.append((new/old - 1) * 100)
                    imp_scales.append(s)
        
        if improvements:
            bars = ax3.bar(range(len(imp_scales)), improvements, color=['green' if i > 0 else 'red' for i in improvements])
            ax3.set_xticks(range(len(imp_scales)))
            ax3.set_xticklabels(imp_scales)
            ax3.set_xlabel('Number of Qubits')
            ax3.set_ylabel('Improvement (%)')
            ax3.set_title('Performance Improvement with Optimized Parameters')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Summary
        ax4.axis('off')
        summary = f"""
OPTIMIZATION SUMMARY
{'='*30}
Budget used: €{self.spent:.2f}
Parameters optimized at: {list(optimal_params.keys())}

Key Findings:
- Parameters shift with scale
- γ: {min(gammas):.2f} → {max(gammas):.2f}
- β: {min(betas):.2f} → {max(betas):.2f}

Performance gains:
- Average: {np.mean(improvements) if improvements else 0:.1f}%
- Max: {max(improvements) if improvements else 0:.1f}% 
- At 30 qubits: {validated_results.get(30, 'N/A')}

Recommendation:
Use scale-dependent parameters
for optimal performance
        """
        ax4.text(0.1, 0.5, summary, fontsize=11, family='monospace', va='center')
        
        plt.suptitle('Scale-Dependent QAOA Parameter Optimization', fontsize=14)
        plt.tight_layout()
        plt.savefig('scale_dependent_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    optimizer = ScaleDependentOptimization(budget_remaining=172)
    
    # Run optimization
    optimal_params, improved_performance = optimizer.run_strategic_optimization()
    
    # Validate improvements
    if optimizer.spent < optimizer.budget_remaining * 0.9:
        validated_results = optimizer.validate_improvements(optimal_params)
    else:
        validated_results = {}
    
    # Create report
    optimizer.create_final_report(optimal_params, improved_performance, validated_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'optimal_parameters': {k: {'gamma': v[0], 'beta': v[1]} for k, v in optimal_params.items()},
        'improved_performance': improved_performance,
        'validated_results': validated_results,
        'total_cost': optimizer.spent,
        'improvement_summary': {
            'scales_tested': list(optimal_params.keys()),
            'average_improvement': np.mean([(improved_performance.get(k, 0) / optimizer.previous_results.get(k, 1) - 1) * 100 
                                           for k in optimal_params.keys()])
        }
    }
    
    with open(f'scale_optimization_{timestamp}.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print(f"Total cost: €{optimizer.spent:.2f}")
    print(f"Results saved to scale_optimization_{timestamp}.json")

if __name__ == "__main__":
    main()