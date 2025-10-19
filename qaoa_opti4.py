"""
#!/usr/bin/env python3
"""
Quantum Circuit Auto-Optimizer v2.4 - Sweet Spot Edition 
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
from scipy import stats
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
import matplotlib.pyplot as plt
from datetime import datetime
import json
import itertools

class QuantumSweetSpotStudy:
    """
    Wissenschaftlich valide Studie zur Sweet Spot Methodik
    """
    
    def __init__(self, platform='simulator'):
        self.platform = platform
        
        if platform == 'iqm':
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
            self.shots_per_test = 100  # Weniger für Kosten
            print("IQM Emerald initialized")
        else:
            self.device = LocalSimulator("braket_dm")
            self.shots_per_test = 500
            print("Simulator initialized")
        
        self.results = {
            'qaoa': {},
            'vqe': {},
            'grover': {}
        }
    
    # ============ ALGORITHMEN ============
    
    def create_qaoa(self, gamma, beta, graph_edges):
        """Standard QAOA"""
        n_qubits = max(max(e) for e in graph_edges) + 1
        circuit = Circuit()
        
        for i in range(n_qubits):
            circuit.h(i)
        
        for u,v in graph_edges:
            circuit.cnot(u, v)
            circuit.rz(v, 2*gamma)
            circuit.cnot(u, v)
        
        for i in range(n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def create_vqe(self, theta, phi, n_qubits=3):
        """Simple VQE ansatz"""
        circuit = Circuit()
        
        # Layer 1
        for i in range(n_qubits):
            circuit.ry(i, theta)
        
        # Entanglement
        for i in range(n_qubits-1):
            circuit.cnot(i, i+1)
        
        # Layer 2
        for i in range(n_qubits):
            circuit.ry(i, phi)
        
        return circuit
    
    def create_grover(self, target='11', with_dd=False):
        """2-qubit Grover"""
        circuit = Circuit()
        
        circuit.h(0)
        circuit.h(1)
        
        if with_dd:
            circuit.x(0)
            circuit.x(0)
        
        # Oracle
        if target == '11':
            circuit.cz(0, 1)
        
        # Diffusion
        circuit.h(0)
        circuit.h(1)
        circuit.x(0)
        circuit.x(1)
        circuit.cz(0, 1)
        circuit.x(0)
        circuit.x(1)
        circuit.h(0)
        circuit.h(1)
        
        return circuit
    
    # ============ PARAMETER SETS ============
    
    def get_parameter_sets(self):
        """Verschiedene Parameter-Sets für Vergleich"""
        return {
            'qaoa': {
                'sweet_spot': (0.217, 1.284),  # Vom Simulator optimiert
                'theoretical': (np.pi/4, np.pi/8),  # Theoretisch optimal
                'random_1': (np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)),
                'random_2': (np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)),
                'random_3': (np.random.uniform(0, np.pi), np.random.uniform(0, np.pi)),
                'grid_search': None  # Wird durch Grid Search gefunden
            },
            'vqe': {
                'sweet_spot': (np.pi/3, np.pi/4),  # Zu finden
                'theoretical': (np.pi/2, np.pi/2),
                'random_1': (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)),
                'random_2': (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)),
                'random_3': (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))
            },
            'grover': {
                'standard': False,  # Ohne DD
                'with_dd': True     # Mit DD
            }
        }
    
    # ============ GRID SEARCH ============
    
    def grid_search_optimal(self, algorithm, test_graph=None):
        """Finde optimale Parameter durch Grid Search"""
        print(f"\nGrid search for {algorithm}...")
        
        if algorithm == 'qaoa':
            gammas = np.linspace(0.1, np.pi, 10)
            betas = np.linspace(0.1, np.pi/2, 10)
            
            best_performance = 0
            best_params = None
            
            for gamma, beta in itertools.product(gammas, betas):
                circuit = self.create_qaoa(gamma, beta, test_graph)
                result = self.device.run(circuit, shots=20).result()
                measurements = result.measurements
                
                # Performance metric
                cuts = [sum(1 for u,v in test_graph if m[u] != m[v]) 
                       for m in measurements]
                performance = np.mean(cuts) / 2  # Max cut = 2 für Triangle
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = (gamma, beta)
            
            print(f"  Grid search found: γ={best_params[0]:.3f}, β={best_params[1]:.3f}")
            return best_params
        
        # Ähnlich für VQE...
        return None
    
    # ============ TESTING ============
    
    def test_algorithm(self, algorithm, params, graph=None, repetitions=5):
        """Test mit mehreren Wiederholungen für Statistik"""
        performances = []
        
        for rep in range(repetitions):
            if algorithm == 'qaoa':
                circuit = self.create_qaoa(params[0], params[1], graph)
                max_value = 2  # Max cut für Triangle
            elif algorithm == 'vqe':
                circuit = self.create_vqe(params[0], params[1])
                max_value = 1  # Normalisiert
            elif algorithm == 'grover':
                circuit = self.create_grover(with_dd=params)
                max_value = 1  # Success rate
            
            # Run
            result = self.device.run(circuit, shots=self.shots_per_test).result()
            measurements = result.measurements
            
            # Calculate performance
            if algorithm == 'qaoa':
                cuts = [sum(1 for u,v in graph if m[u] != m[v]) for m in measurements]
                performance = np.mean(cuts) / max_value
            elif algorithm == 'grover':
                success = sum(1 for m in measurements if all(m)) / len(measurements)
                performance = success
            else:  # VQE
                # Simplified: measure expectation value
                performance = np.mean([(-1)**sum(m) for m in measurements])
                performance = (performance + 1) / 2  # Normalize to [0,1]
            
            performances.append(performance)
        
        return performances
    
    # ============ HAUPTSTUDIE ============
    
    def run_comprehensive_study(self):
        """Vollständige statistische Studie"""
        print("\n" + "="*70)
        print("QUANTUM SWEET SPOT VALIDATION STUDY")
        print("="*70)
        print(f"Platform: {self.platform}")
        print(f"Shots per test: {self.shots_per_test}")
        print("Repetitions per condition: 5")
        
        # Test graph für QAOA
        triangle_graph = [(0,1), (1,2), (0,2)]
        
        # Get parameter sets
        param_sets = self.get_parameter_sets()
        
        # Grid search für QAOA
        if self.platform == 'simulator':
            grid_params = self.grid_search_optimal('qaoa', triangle_graph)
            param_sets['qaoa']['grid_search'] = grid_params
        
        # ============ QAOA TESTS ============
        print("\n" + "-"*70)
        print("TESTING QAOA")
        print("-"*70)
        
        qaoa_results = {}
        for name, params in param_sets['qaoa'].items():
            if params is None:
                continue
            print(f"\nTesting {name}: γ={params[0]:.3f}, β={params[1]:.3f}")
            performances = self.test_algorithm('qaoa', params, triangle_graph)
            qaoa_results[name] = performances
            print(f"  Mean: {np.mean(performances):.3f} ± {np.std(performances):.3f}")
        
        # ============ GROVER TESTS ============
        print("\n" + "-"*70)
        print("TESTING GROVER")
        print("-"*70)
        
        grover_results = {}
        for name, use_dd in param_sets['grover'].items():
            print(f"\nTesting {name}")
            performances = self.test_algorithm('grover', use_dd)
            grover_results[name] = performances
            print(f"  Mean: {np.mean(performances):.3f} ± {np.std(performances):.3f}")
        
        # ============ STATISTICAL ANALYSIS ============
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        
        # ANOVA für QAOA
        if len(qaoa_results) > 2:
            print("\nQAOA Parameter Comparison (ANOVA):")
            f_stat, p_value = stats.f_oneway(*qaoa_results.values())
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  ✓ Significant difference between parameter sets (p < 0.05)")
                
                # Post-hoc: Sweet Spot vs Others
                if 'sweet_spot' in qaoa_results:
                    sweet_spot_perf = qaoa_results['sweet_spot']
                    for name, perf in qaoa_results.items():
                        if name != 'sweet_spot':
                            t_stat, t_p = stats.ttest_ind(sweet_spot_perf, perf)
                            if t_p < 0.05:
                                print(f"  Sweet Spot vs {name}: p={t_p:.4f} (significant)")
            else:
                print("  ✗ No significant difference")
        
        # T-test für Grover
        if len(grover_results) == 2:
            print("\nGrover DD Comparison (t-test):")
            t_stat, p_value = stats.ttest_ind(
                grover_results['with_dd'],
                grover_results['standard']
            )
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  ✓ DD significantly improves Grover (p < 0.05)")
        
        # ============ VISUALIZATION ============
        self.plot_results(qaoa_results, grover_results)
        
        # ============ SAVE RESULTS ============
        save_data = {
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'qaoa_results': {k: {'mean': np.mean(v), 'std': np.std(v), 'values': v} 
                            for k, v in qaoa_results.items()},
            'grover_results': {k: {'mean': np.mean(v), 'std': np.std(v), 'values': v}
                              for k, v in grover_results.items()},
            'statistical_tests': {
                'qaoa_anova_p': p_value if len(qaoa_results) > 2 else None
            }
        }
        
        filename = f'sweet_spot_validation_{self.platform}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")
        
        # ============ CONCLUSION ============
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        
        # Best QAOA
        best_qaoa = max(qaoa_results.items(), key=lambda x: np.mean(x[1]))
        print(f"Best QAOA: {best_qaoa[0]} with {np.mean(best_qaoa[1]):.1%}")
        
        # Sweet Spot validation
        if 'sweet_spot' in qaoa_results:
            sweet_mean = np.mean(qaoa_results['sweet_spot'])
            other_means = [np.mean(v) for k,v in qaoa_results.items() if k != 'sweet_spot']
            if sweet_mean > max(other_means):
                print("✓ Sweet Spot method validated!")
            else:
                print("✗ Sweet Spot not optimal in this test")
        
        return save_data
    
    def plot_results(self, qaoa_results, grover_results):
        """Visualisiere mit Error Bars"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # QAOA
        names = list(qaoa_results.keys())
        means = [np.mean(qaoa_results[n]) for n in names]
        stds = [np.std(qaoa_results[n]) for n in names]
        
        colors = ['green' if 'sweet' in n else 'red' if 'random' in n else 'blue' for n in names]
        
        ax1.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Performance')
        ax1.set_title('QAOA Parameter Comparison')
        ax1.set_ylim(0, 1)
        
        # Grover
        grover_names = list(grover_results.keys())
        grover_means = [np.mean(grover_results[n]) for n in grover_names]
        grover_stds = [np.std(grover_results[n]) for n in grover_names]
        
        ax2.bar(range(len(grover_names)), grover_means, yerr=grover_stds, 
               capsize=5, color=['blue', 'green'], alpha=0.7)
        ax2.set_xticks(range(len(grover_names)))
        ax2.set_xticklabels(grover_names)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Grover Comparison')
        ax2.set_ylim(0, 1)
        
        plt.suptitle(f'Sweet Spot Validation Study - {self.platform.upper()}')
        plt.tight_layout()
        plt.show()


def main():
    print("QUANTUM SWEET SPOT VALIDATION")
    print("="*70)
    
    print("\n1. Run on Simulator (free, full study)")
    print("2. Run on IQM Emerald (~$10-15 for full study)")
    print("3. Quick IQM test (~$3, reduced)")
    
    choice = input("\nChoice: ")
    
    if choice == '2':
        platform = 'iqm'
        print("\n⚠️  Full study will cost ~$10-15")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    elif choice == '3':
        platform = 'iqm'
        print("\nQuick test mode selected")
        # Reduziere Tests
    else:
        platform = 'simulator'
    
    study = QuantumSweetSpotStudy(platform)
    results = study.run_comprehensive_study()


if __name__ == "__main__":
    main()