"""
#!/usr/bin/env python3
"""
IQM QUANTIZED PARAMETER TEST
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Test the optimal quantized parameters that achieved 99.2% on simulator
Compare with our previously tested "sweet spot" parameters

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
from datetime import datetime
import json

class IQMQuantizedTest:
    """Direct comparison of parameter sets on IQM"""
    
    def __init__(self, platform='simulator'):
        self.platform = platform
        
        if platform == 'iqm':
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
            print("IQM Emerald initialized")
            print("Cost per test: ~$0.40")
        else:
            self.device = LocalSimulator("braket_dm")
            print("Simulator initialized")
        
        # Triangle graph
        self.edges = [(0,1), (1,2), (0,2)]
        self.n_qubits = 3
        self.max_cut = 2
        
        # Parameter sets to compare
        self.param_sets = {
            'sweet_spot': {
                'gamma': 0.217,
                'beta': 1.284,
                'description': 'Found via optimization (tested: 92.6% on IQM)'
            },
            'iqm_quantized': {
                'gamma': 0.250,  # = 1/4
                'beta': 1.250,   # = 5/4
                'description': 'Hardware-friendly quantized (99.2% simulator, untested on IQM)'
            },
            'theoretical': {
                'gamma': np.pi/4,  # = 0.785
                'beta': np.pi/8,   # = 0.393
                'description': 'Theoretical optimum (tested: 39.6% on IQM)'
            }
        }
    
    def create_qaoa(self, gamma, beta):
        """Standard QAOA circuit"""
        circuit = Circuit()
        
        # Initial superposition
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Cost operator
        for u, v in self.edges:
            circuit.cnot(u, v)
            circuit.rz(v, 2*gamma)
            circuit.cnot(u, v)
        
        # Mixing operator  
        for i in range(self.n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def test_parameters(self, name, params, shots=256, repetitions=3):
        """Test specific parameter set with repetitions"""
        print(f"\nTesting {name}:")
        print(f"  γ = {params['gamma']:.4f}")
        print(f"  β = {params['beta']:.4f}")
        print(f"  Description: {params['description']}")
        
        performances = []
        cut_distributions = []
        
        for rep in range(repetitions):
            circuit = self.create_qaoa(params['gamma'], params['beta'])
            
            # Run circuit
            result = self.device.run(circuit, shots=shots).result()
            measurements = result.measurements
            
            # Calculate cuts
            cuts = []
            for m in measurements:
                cut = sum(1 for u,v in self.edges if m[u] != m[v])
                cuts.append(cut)
            
            # Performance metrics
            avg_cut = np.mean(cuts)
            approx_ratio = avg_cut / self.max_cut
            optimal_rate = cuts.count(self.max_cut) / shots
            
            performances.append(approx_ratio)
            cut_distributions.append(cuts)
            
            print(f"  Rep {rep+1}: {approx_ratio:.1%} (optimal rate: {optimal_rate:.1%})")
        
        # Statistics
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        
        print(f"  Final: {mean_perf:.1%} ± {std_perf:.1%}")
        
        return {
            'mean': mean_perf,
            'std': std_perf,
            'performances': performances,
            'cut_distributions': cut_distributions
        }
    
    def run_comparison(self):
        """Run complete comparison"""
        print("\n" + "="*70)
        print("IQM QUANTIZED PARAMETER COMPARISON")
        print("="*70)
        print(f"Platform: {self.platform.upper()}")
        print(f"Graph: Triangle (3 qubits, 3 edges, max cut = 2)")
        
        results = {}
        
        # Test each parameter set
        if self.platform == 'iqm':
            # On hardware: test only the important ones
            test_sets = ['sweet_spot', 'iqm_quantized']
            shots = 256
            reps = 3
        else:
            # On simulator: test all for comparison
            test_sets = list(self.param_sets.keys())
            shots = 1000
            reps = 5
        
        for name in test_sets:
            results[name] = self.test_parameters(
                name, 
                self.param_sets[name],
                shots=shots,
                repetitions=reps
            )
        
        # Statistical comparison
        self.analyze_results(results)
        
        # Visualization
        self.plot_comparison(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def analyze_results(self, results):
        """Statistical analysis"""
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        
        if 'iqm_quantized' in results and 'sweet_spot' in results:
            # Compare IQM quantized vs sweet spot
            iqm_q = results['iqm_quantized']
            sweet = results['sweet_spot']
            
            diff = iqm_q['mean'] - sweet['mean']
            
            print(f"\nIQM Quantized vs Sweet Spot:")
            print(f"  IQM Quantized: {iqm_q['mean']:.1%} ± {iqm_q['std']:.1%}")
            print(f"  Sweet Spot: {sweet['mean']:.1%} ± {sweet['std']:.1%}")
            print(f"  Difference: {diff:+.1%}")
            
            if abs(diff) > 0.02:  # >2% difference
                if diff > 0:
                    print("  → IQM Quantized is BETTER!")
                    print("  → Hardware prefers clean quantized values")
                else:
                    print("  → Sweet Spot is better (surprising!)")
            else:
                print("  → No significant difference")
                print("  → Both found the same optimum")
        
        # If theoretical tested, show improvement
        if 'theoretical' in results:
            theo = results['theoretical']
            best_name = max(results.keys(), key=lambda k: results[k]['mean'])
            best = results[best_name]
            
            improvement = best['mean'] / theo['mean']
            print(f"\n{best_name} vs Theoretical:")
            print(f"  Improvement: {improvement:.1f}x")
    
    def plot_comparison(self, results):
        """Visualize results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart with error bars
        names = list(results.keys())
        means = [results[n]['mean'] for n in names]
        stds = [results[n]['std'] for n in names]
        
        colors = []
        for n in names:
            if 'quantized' in n:
                colors.append('gold')
            elif 'sweet' in n:
                colors.append('green')
            else:
                colors.append('gray')
        
        bars = ax1.bar(range(len(names)), means, yerr=stds, 
                      capsize=5, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Approximation Ratio')
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        ax1.set_title('Parameter Set Performance')
        
        # Add values on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{mean:.1%}', ha='center', va='bottom')
        
        # Parameter space visualization
        ax2.scatter(self.param_sets['sweet_spot']['gamma'],
                   self.param_sets['sweet_spot']['beta'],
                   s=200, c='green', marker='o', 
                   label='Sweet Spot', edgecolors='black')
        ax2.scatter(self.param_sets['iqm_quantized']['gamma'],
                   self.param_sets['iqm_quantized']['beta'],
                   s=200, c='gold', marker='s',
                   label='IQM Quantized', edgecolors='black')
        ax2.scatter(self.param_sets['theoretical']['gamma'],
                   self.param_sets['theoretical']['beta'],
                   s=200, c='gray', marker='^',
                   label='Theoretical', edgecolors='black')
        
        ax2.set_xlabel('γ')
        ax2.set_ylabel('β')
        ax2.set_title('Parameter Space')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'IQM Quantized vs Sweet Spot - {self.platform.upper()}')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'iqm_quantized_test_{self.platform}_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nPlot saved to {filename}")
    
    def save_results(self, results):
        """Save results to JSON"""
        save_data = {
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'parameter_sets': self.param_sets,
            'results': {
                name: {
                    'mean': float(data['mean']),
                    'std': float(data['std']),
                    'performances': [float(p) for p in data['performances']]
                }
                for name, data in results.items()
            }
        }
        
        if self.platform == 'iqm' and 'iqm_quantized' in results:
            # Add special note if IQM quantized performs best
            if results['iqm_quantized']['mean'] > 0.95:
                save_data['breakthrough'] = "IQM Quantized parameters achieve >95% on hardware!"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'iqm_quantized_results_{self.platform}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    print("IQM QUANTIZED PARAMETER TEST")
    print("="*70)
    print("\nThis tests the 99.2% simulator parameters on real hardware")
    print("Theory: Hardware prefers quantized values (0.25, 1.25)")
    
    print("\n1. Test on Simulator (verify)")
    print("2. Test on IQM Emerald (~$1)")
    
    choice = input("\nChoice: ")
    
    if choice == '2':
        platform = 'iqm'
        print("\n⚠️  Will test on IQM Emerald")
        print("Testing: Sweet Spot (0.217, 1.284) vs IQM Quantized (0.25, 1.25)")
        confirm = input("Cost ~$1. Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    else:
        platform = 'simulator'
    
    tester = IQMQuantizedTest(platform)
    results = tester.run_comparison()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    
    if platform == 'iqm' and 'iqm_quantized' in results:
        if results['iqm_quantized']['mean'] > 0.95:
            print("🎉 BREAKTHROUGH: IQM Quantized >95% on hardware!")
            print("This confirms hardware prefers clean quantized values")
    
    print("="*70)


if __name__ == "__main__":
    main()