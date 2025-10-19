"""
#!/usr/bin/env python3
"""
QAOA PRODUCTION TEST
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
from collections import Counter

class QAOAProductionTest:
    """
    Production-ready QAOA with actual measurements
    """
    
    def __init__(self, platform='simulator'):
        self.platform = platform
        
        if platform == 'iqm':
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
            self.shots_default = 256
            print(f"IQM Emerald initialized (20 qubits)")
            print(f"Cost per circuit: ~${0.30 + 256*0.00035:.2f}")
        else:
            self.device = LocalSimulator("braket_dm")
            self.shots_default = 1000
            print("Density matrix simulator initialized")
        
        # Test graphs - starting with proven 3-qubit triangle
        self.graphs = {
            'triangle': {
                'edges': [(0,1), (1,2), (0,2)],
                'n_qubits': 3,
                'max_cut': 2  # Verified by brute force
            }
        }
        
        # Parameter sets based on your actual results
        self.param_sets = {
            'sweet_spot': (0.217, 1.284),      # Your 94.5% result
            'quantized': (0.25, 1.25),         # Hardware-friendly
            'theoretical': (np.pi/4, np.pi/8), # ~0.785, ~0.393
            'vermicular_v1': (0.3, 0.65),      # From your optimizer
            'landscape_peak': (0.317, 1.151)   # From landscape analysis
        }
        
        # Add random controls
        np.random.seed(42)
        for i in range(3):
            self.param_sets[f'random_{i+1}'] = (
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, np.pi)
            )
    
    def create_qaoa_circuit(self, gamma, beta, edges, n_qubits):
        """
        Standard QAOA - this is what actually works
        """
        circuit = Circuit()
        
        # Initial superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # Cost operator (phase separator)
        for u, v in edges:
            circuit.cnot(u, v)
            circuit.rz(v, 2 * gamma)
            circuit.cnot(u, v)
        
        # Mixing operator
        for i in range(n_qubits):
            circuit.rx(i, 2 * beta)
        
        return circuit
    
    def measure_cut_value(self, bitstring, edges):
        """
        Actually calculate the cut value for a measurement
        """
        cut = 0
        for u, v in edges:
            if bitstring[u] != bitstring[v]:
                cut += 1
        return cut
    
    def run_circuit_and_measure(self, circuit, edges, shots=None):
        """
        Run circuit and get REAL cut measurements
        """
        if shots is None:
            shots = self.shots_default
        
        # Run the circuit
        result = self.device.run(circuit, shots=shots).result()
        
        # Get measurements - this is an array of bitstrings
        measurements = result.measurements
        
        # Calculate cut for each measurement
        cuts = []
        for measurement in measurements:
            cut = self.measure_cut_value(measurement, edges)
            cuts.append(cut)
        
        return cuts
    
    def analyze_cuts(self, cuts, max_cut):
        """
        Comprehensive analysis of cut distribution
        """
        cuts_array = np.array(cuts)
        
        return {
            'mean_cut': np.mean(cuts_array),
            'std_cut': np.std(cuts_array),
            'max_found': np.max(cuts_array),
            'min_found': np.min(cuts_array),
            'approximation_ratio': np.mean(cuts_array) / max_cut,
            'success_rate': np.sum(cuts_array == max_cut) / len(cuts),
            'distribution': Counter(cuts),
            'raw_cuts': cuts
        }
    
    def test_parameter_set(self, name, params, graph_name='triangle', repetitions=5):
        """
        Test a parameter set with multiple repetitions for statistics
        """
        graph = self.graphs[graph_name]
        edges = graph['edges']
        n_qubits = graph['n_qubits']
        max_cut = graph['max_cut']
        
        gamma, beta = params
        
        print(f"\nTesting {name}: γ={gamma:.4f}, β={beta:.4f}")
        
        all_results = []
        
        for rep in range(repetitions):
            # Create circuit
            circuit = self.create_qaoa_circuit(gamma, beta, edges, n_qubits)
            
            # Run and measure
            cuts = self.run_circuit_and_measure(circuit, edges)
            
            # Analyze
            analysis = self.analyze_cuts(cuts, max_cut)
            all_results.append(analysis)
            
            print(f"  Rep {rep+1}: {analysis['approximation_ratio']:.3f} "
                  f"(success: {analysis['success_rate']:.1%})")
        
        # Aggregate statistics
        approx_ratios = [r['approximation_ratio'] for r in all_results]
        success_rates = [r['success_rate'] for r in all_results]
        
        aggregate = {
            'name': name,
            'gamma': gamma,
            'beta': beta,
            'mean_approx_ratio': np.mean(approx_ratios),
            'std_approx_ratio': np.std(approx_ratios),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'all_results': all_results,
            'approx_ratios': approx_ratios
        }
        
        print(f"  Final: {aggregate['mean_approx_ratio']:.3f} ± {aggregate['std_approx_ratio']:.3f}")
        
        return aggregate
    
    def run_complete_comparison(self):
        """
        Complete statistical comparison of all parameter sets
        """
        print("\n" + "="*70)
        print("QAOA PRODUCTION TEST - COMPLETE COMPARISON")
        print("="*70)
        print(f"Platform: {self.platform}")
        print(f"Graph: Triangle (3 qubits, max cut = 2)")
        print(f"Parameter sets: {len(self.param_sets)}")
        
        # Test all parameter sets
        all_results = {}
        
        # On hardware, test only the important ones
        if self.platform == 'iqm':
            test_sets = ['sweet_spot', 'quantized', 'theoretical']
            repetitions = 3
            print(f"\nHardware mode: Testing {len(test_sets)} sets")
            print(f"Estimated cost: ${len(test_sets) * repetitions * 0.39:.2f}")
        else:
            test_sets = list(self.param_sets.keys())
            repetitions = 5
        
        for name in test_sets:
            all_results[name] = self.test_parameter_set(
                name, self.param_sets[name], repetitions=repetitions
            )
        
        # Statistical analysis
        self.statistical_analysis(all_results)
        
        # Visualization
        self.visualize_results(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def statistical_analysis(self, results):
        """
        Comprehensive statistical tests
        """
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        
        # Prepare data for ANOVA
        groups = []
        labels = []
        
        for name, data in results.items():
            groups.append(data['approx_ratios'])
            labels.append(name)
        
        # 1. Test normality
        print("\n1. Normality Tests (Shapiro-Wilk):")
        for name, group in zip(labels, groups):
            if len(group) >= 3:
                stat, p = stats.shapiro(group)
                print(f"   {name}: p={p:.4f} {'✓ Normal' if p > 0.05 else '✗ Not normal'}")
        
        # 2. ANOVA or Kruskal-Wallis
        if len(groups) > 2:
            print("\n2. Omnibus Test:")
            # Check if all groups are normal
            all_normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
            
            if all_normal:
                f_stat, p_val = stats.f_oneway(*groups)
                print(f"   ANOVA: F={f_stat:.3f}, p={p_val:.6f}")
            else:
                h_stat, p_val = stats.kruskal(*groups)
                print(f"   Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.6f}")
            
            if p_val < 0.05:
                print("   ✓ Significant differences found (p < 0.05)")
            else:
                print("   ✗ No significant differences")
        
        # 3. Pairwise comparisons for key pairs
        print("\n3. Key Pairwise Comparisons:")
        
        key_comparisons = [
            ('sweet_spot', 'theoretical'),
            ('sweet_spot', 'quantized'),
            ('quantized', 'theoretical')
        ]
        
        for name1, name2 in key_comparisons:
            if name1 in results and name2 in results:
                group1 = results[name1]['approx_ratios']
                group2 = results[name2]['approx_ratios']
                
                # T-test
                t_stat, p_val = stats.ttest_ind(group1, group2)
                
                # Effect size (Cohen's d)
                mean1, mean2 = np.mean(group1), np.mean(group2)
                pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
                cohen_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                print(f"   {name1} vs {name2}:")
                print(f"      Means: {mean1:.3f} vs {mean2:.3f}")
                print(f"      t={t_stat:.2f}, p={p_val:.6f}")
                print(f"      Cohen's d={cohen_d:.2f} {'(large)' if abs(cohen_d) > 0.8 else '(medium)' if abs(cohen_d) > 0.5 else '(small)'}")
        
        # 4. Best performer
        print("\n4. Best Performer:")
        best = max(results.items(), key=lambda x: x[1]['mean_approx_ratio'])
        print(f"   {best[0]}: {best[1]['mean_approx_ratio']:.3f} ± {best[1]['std_approx_ratio']:.3f}")
        
        # Check if sweet spot is actually best
        if 'sweet_spot' in results:
            sweet_perf = results['sweet_spot']['mean_approx_ratio']
            others = [r['mean_approx_ratio'] for n, r in results.items() if n != 'sweet_spot']
            if sweet_perf > max(others):
                print("   ✓ Sweet spot parameters validated as best!")
    
    def visualize_results(self, results):
        """
        Professional visualization of results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sort by performance
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1]['mean_approx_ratio'], 
                               reverse=True)
        
        names = [r[0] for r in sorted_results]
        means = [r[1]['mean_approx_ratio'] for r in sorted_results]
        stds = [r[1]['std_approx_ratio'] for r in sorted_results]
        
        # 1. Bar chart with error bars
        colors = ['green' if 'sweet' in n or 'quant' in n else 
                 'blue' if 'theo' in n else 
                 'gray' for n in names]
        
        bars = ax1.bar(range(len(names)), means, yerr=stds, 
                      capsize=5, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Approximation Ratio')
        ax1.set_title('Performance Comparison')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% threshold')
        
        # Add values on bars
        for bar, mean, std in zip(bars, means, stds):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + std + 0.01,
                    f'{mean:.1%}', ha='center', va='bottom')
        
        # 2. Success rates
        success_means = [r[1]['mean_success_rate'] for r in sorted_results]
        ax2.bar(range(len(names)), success_means, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Success Rate (Finding Max Cut)')
        ax2.set_title('Optimal Cut Frequency')
        
        # 3. Parameter space
        for name, result in results.items():
            color = 'green' if 'sweet' in name else 'gold' if 'quant' in name else 'blue' if 'theo' in name else 'gray'
            marker = 'o' if 'sweet' in name else 's' if 'quant' in name else '^' if 'theo' in name else '.'
            ax3.scatter(result['gamma'], result['beta'], 
                       s=200*result['mean_approx_ratio'], 
                       c=color, marker=marker, alpha=0.7, 
                       label=f"{name[:10]}: {result['mean_approx_ratio']:.1%}")
        
        ax3.set_xlabel('γ')
        ax3.set_ylabel('β')
        ax3.set_title('Parameter Space (size = performance)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution for best performer
        best_name = max(results.items(), key=lambda x: x[1]['mean_approx_ratio'])[0]
        best_data = results[best_name]
        
        # Aggregate all cuts from all repetitions
        all_cuts = []
        for rep in best_data['all_results']:
            all_cuts.extend(rep['raw_cuts'])
        
        cut_counts = Counter(all_cuts)
        cuts_sorted = sorted(cut_counts.keys())
        frequencies = [cut_counts[c] for c in cuts_sorted]
        
        ax4.bar(cuts_sorted, frequencies, alpha=0.7, color='green')
        ax4.axvline(x=2, color='r', linestyle='--', label='Max Cut')
        ax4.set_xlabel('Cut Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Cut Distribution - {best_name}')
        ax4.legend()
        
        plt.suptitle(f'QAOA Results - {self.platform.upper()}', fontsize=14)
        plt.tight_layout()
        
        filename = f'qaoa_production_{self.platform}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
        plt.show()
    
    def save_results(self, results):
        """
        Save complete results with all metadata
        """
        save_data = {
            'metadata': {
                'platform': self.platform,
                'timestamp': datetime.now().isoformat(),
                'shots_per_circuit': self.shots_default,
                'graph': 'triangle_3_qubit'
            },
            'results': {}
        }
        
        for name, data in results.items():
            save_data['results'][name] = {
                'gamma': data['gamma'],
                'beta': data['beta'],
                'mean_approx_ratio': data['mean_approx_ratio'],
                'std_approx_ratio': data['std_approx_ratio'],
                'mean_success_rate': data['mean_success_rate'],
                'approx_ratios': data['approx_ratios']
            }
        
        filename = f'qaoa_production_{self.platform}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Results saved to: {filename}")


def main():
    """
    Main execution with cost control
    """
    print("QAOA PRODUCTION TEST")
    print("="*70)
    print("\nThis will run REAL quantum circuits and measure ACTUAL cuts")
    print("No placeholders, no models - just real quantum computing\n")
    
    print("1. Simulator (free, full test)")
    print("2. IQM Hardware (~$3.50 for essential tests)")
    
    choice = input("\nChoice: ")
    
    if choice == '2':
        platform = 'iqm'
        print("\n⚠️  Hardware test will run:")
        print("  - Sweet spot (0.217, 1.284)")
        print("  - Quantized (0.25, 1.25)")
        print("  - Theoretical (π/4, π/8)")
        print("  - 3 repetitions each")
        print(f"  - Estimated cost: ${3 * 3 * 0.39:.2f}")
        
        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return
    else:
        platform = 'simulator'
    
    # Run test
    tester = QAOAProductionTest(platform)
    results = tester.run_complete_comparison()
    
    # Summary
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    best = max(results.items(), key=lambda x: x[1]['mean_approx_ratio'])
    print(f"\nBest parameters: {best[0]}")
    print(f"Performance: {best[1]['mean_approx_ratio']:.1%}")
    
    if platform == 'iqm':
        print(f"\nActual cost: ~${len(results) * 3 * 0.39:.2f}")


if __name__ == "__main__":
    main()