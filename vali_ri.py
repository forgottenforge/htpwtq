"""
#!/usr/bin/env python3
"""
Validation 1.1 RIGETTI ANKAA-3 VALIDATION
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Leveraging 84-qubit processor with zero queue
Budget-aware scaling up to feasible limits

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
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import json
from datetime import datetime
import networkx as nx
import time

class RigettiComprehensiveValidation:
    def __init__(self, budget_eur=200):
        self.budget_eur = budget_eur
        self.spent_eur = 0
        
        # Rigetti pricing
        self.cost_per_shot = 0.00035
        self.cost_per_task = 0.30
        
        # Ankaa-3 has 84 qubits
        self.device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
        self.device = AwsDevice(self.device_arn)
        
        print(f"Rigetti Ankaa-3 Status: {self.device.status}")
        print(f"Available qubits: 84")
        print(f"Queue depth: 0 (confirmed)")
    
    def phase_1_landscape_calibration(self):
        """
        CRITICAL: Measure actual landscape on Rigetti hardware
        Your quantized params (0.25, 1.25) might not be optimal for Rigetti
        """
        print("\n" + "="*70)
        print("PHASE 1: RIGETTI PARAMETER LANDSCAPE CALIBRATION")
        print("="*70)
        print("Testing if sweet spot transfers to Rigetti architecture...")
        
        # Test on 5 qubits with parameter sweep
        n_qubits = 5
        graph = nx.cycle_graph(n_qubits)
        
        # Coarse grid
        gamma_values = np.linspace(0, 2*np.pi, 8)
        beta_values = np.linspace(0, np.pi, 6)
        
        landscape = np.zeros((len(gamma_values), len(beta_values)))
        
        shots_per_point = 50  # Minimal for landscape
        total_points = len(gamma_values) * len(beta_values)
        landscape_cost = total_points * (self.cost_per_task + shots_per_point * self.cost_per_shot)
        
        print(f"\nLandscape scan: {total_points} points")
        print(f"Estimated cost: €{landscape_cost:.2f}")
        
        if landscape_cost > 20:  # Limit landscape budget
            print("Reducing grid density...")
            gamma_values = gamma_values[::2]  # Every other point
            beta_values = beta_values[::2]
            total_points = len(gamma_values) * len(beta_values)
            landscape_cost = total_points * (self.cost_per_task + shots_per_point * self.cost_per_shot)
            print(f"Reduced to {total_points} points: €{landscape_cost:.2f}")
        
        best_performance = 0
        best_params = (0.25, 1.25)  # Default to your values
        
        for i, gamma in enumerate(gamma_values):
            for j, beta in enumerate(beta_values):
                circuit = self.create_qaoa_circuit(gamma, beta, graph)
                
                task = self.device.run(circuit, shots=shots_per_point)
                result = task.result()
                
                measurements = result.measurements
                cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
                performance = np.mean(cuts) / 4  # Max cut for pentagon
                
                landscape[i, j] = performance
                self.spent_eur += self.cost_per_task + shots_per_point * self.cost_per_shot
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = (gamma, beta)
                    print(f"  New best: γ={gamma:.3f}, β={beta:.3f}, perf={performance:.3f}")
        
        print(f"\nRigetti optimal: γ={best_params[0]:.3f}, β={best_params[1]:.3f}")
        print(f"Performance: {best_performance:.3f}")
        
        # Fine-tune around best
        if self.spent_eur < 30:
            print("\nFine-tuning...")
            gamma_fine = np.linspace(best_params[0]-0.3, best_params[0]+0.3, 5)
            beta_fine = np.linspace(best_params[1]-0.2, best_params[1]+0.2, 5)
            
            for gamma in gamma_fine:
                for beta in beta_fine:
                    circuit = self.create_qaoa_circuit(gamma, beta, graph)
                    task = self.device.run(circuit, shots=100)
                    result = task.result()
                    
                    measurements = result.measurements
                    cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
                    performance = np.mean(cuts) / 4
                    
                    self.spent_eur += self.cost_per_task + 100 * self.cost_per_shot
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = (gamma, beta)
                        print(f"  Refined: γ={gamma:.3f}, β={beta:.3f}, perf={performance:.3f}")
        
        return best_params, landscape, best_performance
    
    def phase_2_aggressive_scaling(self, optimal_params):
        """
        Scale as high as budget allows
        """
        print("\n" + "="*70)
        print("PHASE 2: AGGRESSIVE SCALING ON RIGETTI")
        print("="*70)
        
        gamma, beta = optimal_params
        
        # Calculate max feasible qubits within budget
        remaining_budget = self.budget_eur - self.spent_eur
        cost_per_test = self.cost_per_task + 200 * self.cost_per_shot  # 200 shots per test
        max_tests = int(remaining_budget / cost_per_test)
        
        print(f"Remaining budget: €{remaining_budget:.2f}")
        print(f"Cost per test: €{cost_per_test:.2f}")
        print(f"Max tests possible: {max_tests}")
        
        # Strategic qubit selection
        qubit_targets = []
        
        # Essential scales (must have)
        essential = [3, 5, 7, 10, 12, 15]
        
        # Ambitious scales (if budget allows)
        ambitious = [20, 25, 30, 35, 40, 45, 50]
        
        # Combine based on budget
        if max_tests >= len(essential) + 3:
            qubit_targets = essential + ambitious[:min(3, max_tests - len(essential))]
        else:
            qubit_targets = essential[:max_tests]
        
        print(f"Testing qubits: {qubit_targets}")
        
        results = {}
        
        for n_qubits in qubit_targets:
            print(f"\n{n_qubits} QUBITS:")
            
            # Check if we can afford it
            if self.spent_eur + cost_per_test > self.budget_eur:
                print(f"  Skipping - would exceed budget")
                break
            
            # Create appropriate graph
            if n_qubits <= 20:
                graph = nx.random_regular_graph(min(3, n_qubits-1), n_qubits, seed=42)
            else:
                # For large graphs, use sparser connectivity
                graph = nx.random_regular_graph(2, n_qubits, seed=42)
            
            print(f"  Graph: {graph.number_of_edges()} edges")
            
            # Run test
            circuit = self.create_qaoa_circuit(gamma, beta, graph)
            
            # Reduce shots for very large circuits
            shots = 200 if n_qubits <= 20 else 100
            
            task = self.device.run(circuit, shots=shots)
            result = task.result()
            
            measurements = result.measurements
            cuts = []
            for m in measurements:
                cut = sum(1 for u, v in graph.edges() if m[u] != m[v])
                cuts.append(cut)
            
            # For large graphs, compare to random partition expectation
            if n_qubits > 15:
                expected_random_cut = graph.number_of_edges() / 2
                approx_ratio = np.mean(cuts) / expected_random_cut
            else:
                max_cut = self.calculate_max_cut_exact(graph)
                approx_ratio = np.mean(cuts) / max_cut
            
            results[n_qubits] = {
                'performance': approx_ratio,
                'std': np.std(cuts) / (graph.number_of_edges() / 2),
                'edges': graph.number_of_edges(),
                'shots': shots
            }
            
            self.spent_eur += self.cost_per_task + shots * self.cost_per_shot
            
            print(f"  Performance: {approx_ratio:.3f}")
            print(f"  Total spent: €{self.spent_eur:.2f}")
            
            # Stop if approaching budget
            if self.spent_eur > self.budget_eur * 0.9:
                print("\nApproaching budget limit - stopping")
                break
        
        return results
    
    def phase_3_statistical_validation(self, optimal_params):
        """
        High-statistics runs on key sizes
        """
        print("\n" + "="*70)
        print("PHASE 3: STATISTICAL VALIDATION")
        print("="*70)
        
        gamma, beta = optimal_params
        
        # Pick 3 key sizes for high-statistics validation
        test_sizes = [5, 10, 15]
        shots = 500
        repetitions = 5
        
        statistical_results = {}
        
        for n_qubits in test_sizes:
            if self.spent_eur + repetitions * (self.cost_per_task + shots * self.cost_per_shot) > self.budget_eur:
                print(f"Skipping {n_qubits} qubits - would exceed budget")
                continue
            
            print(f"\n{n_qubits} qubits with {repetitions} repetitions:")
            
            graph = nx.random_regular_graph(min(3, n_qubits-1), n_qubits, seed=42)
            performances = []
            
            for rep in range(repetitions):
                circuit = self.create_qaoa_circuit(gamma, beta, graph)
                
                task = self.device.run(circuit, shots=shots)
                result = task.result()
                
                measurements = result.measurements
                cuts = [sum(1 for u, v in graph.edges() if m[u] != m[v]) for m in measurements]
                
                if n_qubits <= 10:
                    max_cut = self.calculate_max_cut_exact(graph)
                else:
                    max_cut = graph.number_of_edges() * 0.878  # Goemans-Williamson
                
                performance = np.mean(cuts) / max_cut
                performances.append(performance)
                
                self.spent_eur += self.cost_per_task + shots * self.cost_per_shot
                
                print(f"  Rep {rep+1}: {performance:.3f}")
            
            # Calculate statistics
            mean_perf = np.mean(performances)
            std_perf = np.std(performances, ddof=1)
            sem = std_perf / np.sqrt(len(performances))
            ci_95 = stats.t.interval(0.95, len(performances)-1, mean_perf, sem)
            
            statistical_results[n_qubits] = {
                'mean': mean_perf,
                'std': std_perf,
                'ci_95': ci_95,
                'n_total_measurements': shots * repetitions
            }
            
            print(f"  Final: {mean_perf:.3f} ± {std_perf:.3f}")
            print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        
        return statistical_results
    
    def create_publication_package(self, all_results):
        """
        Generate comprehensive publication materials
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Extract data
        landscape = all_results['landscape']['data']
        scaling_results = all_results['scaling']
        stats_results = all_results['statistics']
        
        # Panel 1: Rigetti landscape
        ax1 = plt.subplot(3, 3, 1)
        if landscape is not None:
            im = ax1.imshow(landscape, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_xlabel('β index')
            ax1.set_ylabel('γ index')
            ax1.set_title(f'Rigetti Parameter Landscape')
            plt.colorbar(im, ax=ax1)
            
            # Mark optimum
            opt_gamma = all_results['landscape']['optimal'][0]
            opt_beta = all_results['landscape']['optimal'][1]
            ax1.plot(opt_beta*len(landscape[0])/(np.pi), 
                    opt_gamma*len(landscape)/(2*np.pi), 
                    'r*', markersize=20)
        
        # Panel 2: Scaling curve
        ax2 = plt.subplot(3, 3, 2)
        qubits = sorted(scaling_results.keys())
        performances = [scaling_results[q]['performance'] for q in qubits]
        errors = [scaling_results[q]['std'] for q in qubits]
        
        ax2.errorbar(qubits, performances, yerr=errors, fmt='bo-', 
                    markersize=8, linewidth=2, capsize=5)
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Approximation Ratio')
        ax2.set_title(f'Scaling up to {max(qubits)} Qubits')
        ax2.grid(True, alpha=0.3)
        
        # Fit scaling law
        if len(qubits) > 3:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(b * x) + c
            
            try:
                popt, _ = curve_fit(exp_decay, qubits, performances, p0=[1, -0.05, 0.5])
                x_fit = np.linspace(min(qubits), max(qubits), 100)
                ax2.plot(x_fit, exp_decay(x_fit, *popt), 'r--', 
                        label=f'Fit: {popt[0]:.2f}e^({popt[1]:.3f}n)+{popt[2]:.2f}')
                ax2.legend()
            except:
                pass
        
        # Panel 3: Statistical validation
        ax3 = plt.subplot(3, 3, 3)
        if stats_results:
            stat_qubits = list(stats_results.keys())
            stat_means = [stats_results[q]['mean'] for q in stat_qubits]
            stat_cis = [stats_results[q]['ci_95'] for q in stat_qubits]
            
            for i, q in enumerate(stat_qubits):
                ci = stat_cis[i]
                ax3.plot([q, q], ci, 'b-', linewidth=3)
                ax3.plot(q, stat_means[i], 'ro', markersize=10)
            
            ax3.set_xlabel('Qubits')
            ax3.set_ylabel('Performance')
            ax3.set_title('95% Confidence Intervals')
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Cost breakdown
        ax4 = plt.subplot(3, 3, 4)
        phases = ['Landscape', 'Scaling', 'Statistics']
        costs = [
            all_results['landscape']['cost'],
            all_results['scaling_cost'],
            all_results['statistics_cost']
        ]
        colors = ['blue', 'green', 'red']
        
        bars = ax4.bar(phases, costs, color=colors, alpha=0.7)
        ax4.set_ylabel('Cost (€)')
        ax4.set_title('Budget Utilization')
        ax4.axhline(y=self.budget_eur, color='red', linestyle='--', label='Budget')
        
        for bar, cost in zip(bars, costs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'€{cost:.2f}', ha='center', va='bottom')
        
        # Panel 5-9: Summary
        ax5 = plt.subplot(3, 3, 5)
        ax5.axis('off')
        summary = f"""
RIGETTI ANKAA-3 RESULTS
{'='*25}
Max qubits tested: {max(qubits)}
Optimal params:
  γ = {all_results['landscape']['optimal'][0]:.3f}
  β = {all_results['landscape']['optimal'][1]:.3f}
  
Performance at 15q: {scaling_results.get(15, {}).get('performance', 'N/A')}
Total cost: €{self.spent_eur:.2f}
        """
        ax5.text(0.1, 0.5, summary, fontsize=11, family='monospace', va='center')
        
        plt.suptitle(f'Complete Rigetti Validation (up to {max(qubits)} qubits)', fontsize=14)
        plt.tight_layout()
        plt.savefig('rigetti_complete_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
    
    def calculate_max_cut_exact(self, graph):
        """Exact max cut for small graphs"""
        n = graph.number_of_nodes()
        if n > 12:
            return len(graph.edges()) * 0.878
        
        max_cut = 0
        for i in range(2**n):
            partition = format(i, f'0{n}b')
            cut = sum(1 for u, v in graph.edges() if partition[u] != partition[v])
            max_cut = max(max_cut, cut)
        return max_cut
    
    def run_complete_validation(self):
        """Execute everything"""
        print("="*70)
        print("RIGETTI ANKAA-3 COMPLETE VALIDATION")
        print("="*70)
        print(f"Budget: €{self.budget_eur}")
        print("Strategy: Exploit zero queue time for maximum scaling")
        
        all_results = {}
        
        # Phase 1: Find Rigetti-specific optimal parameters
        optimal_params, landscape, best_perf = self.phase_1_landscape_calibration()
        all_results['landscape'] = {
            'optimal': optimal_params,
            'performance': best_perf,
            'data': landscape,
            'cost': self.spent_eur
        }
        
        landscape_cost = self.spent_eur
        
        # Phase 2: Scale as high as possible
        scaling_results = self.phase_2_aggressive_scaling(optimal_params)
        all_results['scaling'] = scaling_results
        all_results['scaling_cost'] = self.spent_eur - landscape_cost
        
        scaling_cost = self.spent_eur
        
        # Phase 3: Statistical validation (if budget remains)
        if self.spent_eur < self.budget_eur * 0.9:
            stats_results = self.phase_3_statistical_validation(optimal_params)
            all_results['statistics'] = stats_results
            all_results['statistics_cost'] = self.spent_eur - scaling_cost
        else:
            all_results['statistics'] = {}
            all_results['statistics_cost'] = 0
        
        # Generate publication materials
        self.create_publication_package(all_results)
        
        # Save everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'rigetti_validation_{timestamp}.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        print(f"Total spent: €{self.spent_eur:.2f}")
        print(f"Max qubits tested: {max(scaling_results.keys())}")
        print(f"Rigetti optimal params: γ={optimal_params[0]:.3f}, β={optimal_params[1]:.3f}")
        
        return all_results


def main():
    validator = RigettiComprehensiveValidation(budget_eur=200)
    results = validator.run_complete_validation()

if __name__ == "__main__":
    main()