"""
#!/usr/bin/env python3
"""
Validation 1.2 RIGETTI ANKAA-3 VALIDATION fixed
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
from braket.aws import AwsDevice
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime
import networkx as nx

class RigettiValidationFixed:
    def __init__(self, budget_eur=200, already_spent=23.99):
        self.budget_eur = budget_eur
        self.spent_eur = already_spent  # Account for what you already spent
        
        self.cost_per_shot = 0.00035
        self.cost_per_task = 0.30
        
        self.device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
        self.device = AwsDevice(self.device_arn)
        
        # Use the parameters you already found!
        self.optimal_gamma = 2.693
        self.optimal_beta = 1.885
        
        print(f"Resuming with optimal params: γ={self.optimal_gamma:.3f}, β={self.optimal_beta:.3f}")
        print(f"Already spent: €{self.spent_eur:.2f}")
        print(f"Remaining budget: €{budget_eur - already_spent:.2f}")
    
    def create_appropriate_graph(self, n_qubits):
        """Create graph ensuring n*d is even"""
        if n_qubits <= 2:
            return nx.path_graph(n_qubits)
        
        # Try regular graph with degree 3
        if (n_qubits * 3) % 2 == 0:
            return nx.random_regular_graph(3, n_qubits, seed=42)
        
        # Try degree 2
        elif (n_qubits * 2) % 2 == 0:
            return nx.random_regular_graph(2, n_qubits, seed=42)
        
        # Try degree 4
        elif n_qubits >= 5 and (n_qubits * 4) % 2 == 0:
            return nx.random_regular_graph(4, n_qubits, seed=42)
        
        # Fallback: use cycle or star
        else:
            if n_qubits % 2 == 0:
                return nx.cycle_graph(n_qubits)
            else:
                return nx.star_graph(n_qubits-1)
    
    def continue_scaling(self):
        """Continue from where we left off"""
        print("\n" + "="*70)
        print("CONTINUING SCALING STUDY")
        print("="*70)
        
        remaining_budget = self.budget_eur - self.spent_eur
        cost_per_test = self.cost_per_task + 200 * self.cost_per_shot
        
        # Start from 5 qubits since 3 was already done
        qubit_targets = [5, 7, 10, 12, 15, 20, 25, 30, 35, 40]
        
        # Filter by budget
        max_tests = int(remaining_budget / cost_per_test)
        qubit_targets = qubit_targets[:max_tests]
        
        print(f"Testing: {qubit_targets}")
        
        results = {
            3: {'performance': 0.790, 'already_measured': True}  # Your existing result
        }
        
        for n_qubits in qubit_targets:
            if self.spent_eur + cost_per_test > self.budget_eur:
                print(f"\nStopping at {n_qubits-1} qubits - budget limit")
                break
            
            print(f"\n{n_qubits} QUBITS:")
            
            # Create appropriate graph
            graph = self.create_appropriate_graph(n_qubits)
            print(f"  Graph type: {type(graph).__name__}, edges: {graph.number_of_edges()}")
            
            # Run test
            circuit = self.create_qaoa_circuit(self.optimal_gamma, self.optimal_beta, graph)
            
            shots = 200 if n_qubits <= 20 else 100
            
            task = self.device.run(circuit, shots=shots)
            result = task.result()
            
            measurements = result.measurements
            cuts = []
            for m in measurements:
                cut = sum(1 for u, v in graph.edges() if m[u] != m[v])
                cuts.append(cut)
            
            # Calculate performance
            if n_qubits <= 12:
                max_cut = self.calculate_max_cut_exact(graph)
            else:
                # Use approximation for large graphs
                max_cut = graph.number_of_edges() * 0.878
            
            performance = np.mean(cuts) / max_cut if max_cut > 0 else 0
            
            results[n_qubits] = {
                'performance': performance,
                'std': np.std(cuts) / max_cut if max_cut > 0 else 0,
                'graph_edges': graph.number_of_edges(),
                'shots': shots
            }
            
            self.spent_eur += self.cost_per_task + shots * self.cost_per_shot
            
            print(f"  Performance: {performance:.3f}")
            print(f"  Total spent: €{self.spent_eur:.2f}")
        
        return results
    
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
            return graph.number_of_edges() * 0.878
        
        max_cut = 0
        for i in range(2**n):
            partition = format(i, f'0{n}b')
            cut = sum(1 for u, v in graph.edges() if partition[u] != partition[v])
            max_cut = max(max_cut, cut)
        return max_cut
    
    def create_results_figure(self, results):
        """Create publication figure"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scaling plot
        qubits = sorted(results.keys())
        performances = [results[q]['performance'] for q in qubits]
        
        ax1.plot(qubits, performances, 'bo-', markersize=10, linewidth=2)
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('Approximation Ratio')
        ax1.set_title(f'Rigetti Scaling to {max(qubits)} Qubits')
        ax1.grid(True, alpha=0.3)
        
        # Add exponential fit if enough points
        if len(qubits) > 3:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(b * np.array(x)) + c
            
            try:
                popt, _ = curve_fit(exp_decay, qubits, performances, p0=[0.5, -0.05, 0.3])
                x_fit = np.linspace(min(qubits), max(qubits), 100)
                ax1.plot(x_fit, exp_decay(x_fit, *popt), 'r--', alpha=0.5,
                        label=f'Fit: {popt[0]:.2f}e^({popt[1]:.3f}n)+{popt[2]:.2f}')
                ax1.legend()
            except:
                pass
        
        # Summary stats
        ax2.axis('off')
        summary = f"""
RIGETTI ANKAA-3 RESULTS
{'='*30}
Optimal Parameters:
  γ = {self.optimal_gamma:.3f}
  β = {self.optimal_beta:.3f}

Qubits tested: {min(qubits)} - {max(qubits)}
Best performance: {max(performances):.3f}
at {qubits[performances.index(max(performances))]} qubits

Total cost: €{self.spent_eur:.2f}

Key Finding:
Performance remains >50%
up to {max([q for q, r in results.items() if r['performance'] > 0.5])} qubits
        """
        ax2.text(0.1, 0.5, summary, fontsize=11, family='monospace', va='center')
        
        plt.suptitle('Rigetti Quantum Processor Validation', fontsize=14)
        plt.tight_layout()
        plt.savefig('rigetti_scaling_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # Resume from where you left off
    validator = RigettiValidationFixed(budget_eur=200, already_spent=23.99)
    results = validator.continue_scaling()
    
    # Create figure
    validator.create_results_figure(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        'timestamp': timestamp,
        'optimal_params': {
            'gamma': validator.optimal_gamma,
            'beta': validator.optimal_beta
        },
        'results': results,
        'total_cost': validator.spent_eur
    }
    
    with open(f'rigetti_results_{timestamp}.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print(f"Final cost: €{validator.spent_eur:.2f}")
    print(f"Max qubits tested: {max(results.keys())}")

if __name__ == "__main__":
    main()