"""
Multi-Layer QAOA Sweet Spot Finder
Automatically finds optimal parameters for p=1,2,3,4 and validates them
"""

import numpy as np
from braket.circuits import Circuit
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime
import itertools

class MultiLayerSweetSpotFinder:
    def __init__(self, platform='simulator', shots_per_test=256):
        self.platform = platform
        self.shots_per_test = shots_per_test
        self.spent_eur = 0
        self.budget_eur = 30.0
        
        self.pricing = {
            'emerald': {'per_task': 0.30, 'per_shot': 0.00145},
            'rigetti': {'per_task': 0.30, 'per_shot': 0.00035},
            'ionq': {'per_task': 0.30, 'per_shot': 0.01}
        }
        
        if platform == 'simulator':
            self.device = LocalSimulator("braket_dm")
        elif platform == 'emerald':
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
        elif platform == 'rigetti':
            self.device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
        elif platform == 'ionq':
            self.device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1")
    
    def create_qaoa_circuit(self, gamma_list, beta_list, graph, p):
        """Create p-layer QAOA circuit"""
        n_qubits = max(max(edge) for edge in graph) + 1
        circuit = Circuit()
        
        # Initial superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # p QAOA layers
        for layer in range(p):
            # Cost Hamiltonian
            for u, v in graph:
                circuit.cnot(u, v)
                circuit.rz(v, 2 * gamma_list[layer])
                circuit.cnot(u, v)
            
            # Mixer Hamiltonian
            for i in range(n_qubits):
                circuit.rx(i, 2 * beta_list[layer])
        
        return circuit
    
    def evaluate_qaoa(self, gamma_list, beta_list, graph, p, shots=None):
        """Evaluate QAOA performance"""
        if shots is None:
            shots = self.shots_per_test
        
        circuit = self.create_qaoa_circuit(gamma_list, beta_list, graph, p)
        
        # Cost calculation for hardware
        if self.platform != 'simulator':
            cost = self.pricing[self.platform]['per_task'] + shots * self.pricing[self.platform]['per_shot']
            if self.spent_eur + cost > self.budget_eur:
                return None
            self.spent_eur += cost
        
        result = self.device.run(circuit, shots=shots).result()
        measurements = result.measurements
        
        # Calculate cut values
        max_cut = len(graph)
        cuts = [sum(1 for u, v in graph if m[u] != m[v]) for m in measurements]
        performance = np.mean(cuts) / max_cut
        
        return performance
    
    def grid_search_sweet_spot(self, p, graph, grid_resolution=10):
        """
        Find sweet spot for given p using 2-phase grid search
        Phase 1: Coarse grid
        Phase 2: Fine refinement around best point
        """
        print(f"\n{'='*70}")
        print(f"GRID SEARCH FOR p={p}")
        print(f"{'='*70}")
        
        # Phase 1: Coarse grid search
        print(f"\n📍 Phase 1: Coarse Grid ({grid_resolution}x{grid_resolution} per parameter)")
        
        # Parameter ranges based on p=1 sweet spot knowledge
        if p == 1:
            gamma_range = np.linspace(0.1, 0.5, grid_resolution)
            beta_range = np.linspace(0.8, 1.6, grid_resolution)
        else:
            # For p>1, scale from p=1 sweet spot
            gamma_range = np.linspace(0.1, 0.4, grid_resolution)
            beta_range = np.linspace(0.7, 1.8, grid_resolution)
        
        best_performance = 0
        best_params = None
        
        # For p>1, we need multiple parameters
        param_combinations = []
        for _ in range(p):
            param_combinations.append((gamma_range, beta_range))
        
        # Test all combinations (simplified for p>1)
        if p == 1:
            total_tests = len(gamma_range) * len(beta_range)
            tested = 0
            
            for gamma in gamma_range:
                for beta in beta_range:
                    tested += 1
                    if tested % 10 == 0:
                        print(f"  Progress: {tested}/{total_tests} ({100*tested/total_tests:.1f}%)")
                    
                    perf = self.evaluate_qaoa([gamma], [beta], graph, p, shots=50)
                    
                    if perf and perf > best_performance:
                        best_performance = perf
                        best_params = ([gamma], [beta])
        
        else:
            # For p>1, use heuristic: scale from p=1 sweet spot
            print(f"  Using heuristic scaling for p={p}")
            
            # Try different scaling strategies
            strategies = [
                'linear',     # γ_i = γ_1 * i/p
                'decreasing', # γ_i = γ_1 * (1 - 0.1*i)
                'increasing', # γ_i = γ_1 * (1 + 0.1*i)
            ]
            
            for strategy in strategies:
                for gamma_base in gamma_range[::2]:  # Sample every other
                    for beta_base in beta_range[::2]:
                        
                        if strategy == 'linear':
                            gammas = [gamma_base * (i+1)/p for i in range(p)]
                            betas = [beta_base * (i+1)/p for i in range(p)]
                        elif strategy == 'decreasing':
                            gammas = [gamma_base * (1 - 0.1*i) for i in range(p)]
                            betas = [beta_base * (1 - 0.1*i) for i in range(p)]
                        else:  # increasing
                            gammas = [gamma_base * (1 + 0.1*i) for i in range(p)]
                            betas = [beta_base * (1 + 0.1*i) for i in range(p)]
                        
                        perf = self.evaluate_qaoa(gammas, betas, graph, p, shots=50)
                        
                        if perf and perf > best_performance:
                            best_performance = perf
                            best_params = (gammas, betas)
        
        print(f"\n✓ Coarse grid best: Performance = {best_performance:.3f}")
        print(f"  Parameters: γ = {[f'{g:.3f}' for g in best_params[0]]}")
        print(f"               β = {[f'{b:.3f}' for b in best_params[1]]}")
        
        # Phase 2: Fine refinement
        if p == 1:
            print(f"\n📍 Phase 2: Fine Refinement (7x7 around best)")
            
            delta = 0.1
            gamma_fine = np.linspace(best_params[0][0] - delta, best_params[0][0] + delta, 7)
            beta_fine = np.linspace(best_params[1][0] - delta, best_params[1][0] + delta, 7)
            
            for gamma in gamma_fine:
                for beta in beta_fine:
                    perf = self.evaluate_qaoa([gamma], [beta], graph, p, shots=200)
                    
                    if perf and perf > best_performance:
                        best_performance = perf
                        best_params = ([gamma], [beta])
        
        print(f"\n✅ SWEET SPOT FOUND!")
        print(f"   Performance: {best_performance:.3f}")
        print(f"   γ = {best_params[0]}")
        print(f"   β = {best_params[1]}")
        
        return best_params, best_performance
    
    def get_theoretical_params(self, p):
        """Get theoretical parameters for p layers"""
        # Standard QAOA theoretical params
        if p == 1:
            return ([np.pi/4], [np.pi/8])
        else:
            # Linear interpolation for p>1 (common theoretical approach)
            gammas = [np.pi/4 * (i+1)/p for i in range(p)]
            betas = [np.pi/8 * (i+1)/p for i in range(p)]
            return (gammas, betas)
    
    def run_complete_study(self, max_p=4):
        """
        Main study: Find and validate sweet spots for all p
        """
        print("="*70)
        print("MULTI-LAYER QAOA SWEET SPOT FINDER")
        print("="*70)
        print(f"Platform: {self.platform}")
        print(f"Max layers: p={max_p}")
        print(f"Budget: €{self.budget_eur}")
        print()
        
        # Triangle graph
        graph = [(0,1), (1,2), (0,2)]
        
        results = {}
        
        # For each depth p
        for p in range(1, max_p + 1):
            print(f"\n{'#'*70}")
            print(f"# TESTING p={p}")
            print(f"{'#'*70}")
            
            # 1. Find sweet spot
            sweet_params, sweet_perf = self.grid_search_sweet_spot(p, graph)
            
            # 2. Get theoretical params
            theo_params = self.get_theoretical_params(p)
            
            # 3. Validate both with multiple repetitions
            print(f"\n🔬 VALIDATION WITH {self.shots_per_test} SHOTS")
            
            # Sweet spot validation
            print(f"\nTesting Sweet Spot...")
            sweet_performances = []
            for rep in range(3 if self.platform != 'simulator' else 5):
                perf = self.evaluate_qaoa(sweet_params[0], sweet_params[1], graph, p)
                if perf is None:
                    print("⚠️ Budget exhausted!")
                    break
                sweet_performances.append(perf)
                print(f"  Rep {rep+1}: {perf:.3f}")
            
            # Theoretical validation
            print(f"\nTesting Theoretical...")
            theo_performances = []
            for rep in range(3 if self.platform != 'simulator' else 5):
                perf = self.evaluate_qaoa(theo_params[0], theo_params[1], graph, p)
                if perf is None:
                    break
                theo_performances.append(perf)
                print(f"  Rep {rep+1}: {perf:.3f}")
            
            # Statistical comparison
            if len(sweet_performances) > 0 and len(theo_performances) > 0:
                sweet_mean = np.mean(sweet_performances)
                sweet_std = np.std(sweet_performances, ddof=1)
                theo_mean = np.mean(theo_performances)
                theo_std = np.std(theo_performances, ddof=1)
                
                # t-test
                if len(sweet_performances) > 1 and len(theo_performances) > 1:
                    t_stat, p_val = stats.ttest_ind(sweet_performances, theo_performances)
                    
                    print(f"\n📊 RESULTS for p={p}:")
                    print(f"   Sweet Spot: {sweet_mean:.3f} ± {sweet_std:.3f}")
                    print(f"   Theoretical: {theo_mean:.3f} ± {theo_std:.3f}")
                    print(f"   Improvement: {(sweet_mean/theo_mean - 1)*100:.1f}%")
                    print(f"   t-test p-value: {p_val:.4f}")
                    if p_val < 0.05:
                        print(f"   ✓ Significantly better!")
                    
                    results[p] = {
                        'sweet_params': (sweet_params[0], sweet_params[1]),
                        'sweet_mean': sweet_mean,
                        'sweet_std': sweet_std,
                        'sweet_performances': sweet_performances,
                        'theo_params': (theo_params[0], theo_params[1]),
                        'theo_mean': theo_mean,
                        'theo_std': theo_std,
                        'theo_performances': theo_performances,
                        'improvement': sweet_mean / theo_mean,
                        'p_value': p_val
                    }
            
            if self.spent_eur >= self.budget_eur * 0.9:
                print(f"\n⚠️ Approaching budget limit (€{self.spent_eur:.2f}/€{self.budget_eur})")
                break
        
        # Final visualization
        self.plot_all_results(results)
        self.save_results(results)
        
        return results
    
    def plot_all_results(self, results):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))
        
        p_values = sorted(results.keys())
        
        # 1. Performance comparison
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(p_values))
        width = 0.35
        
        sweet_means = [results[p]['sweet_mean'] for p in p_values]
        sweet_stds = [results[p]['sweet_std'] for p in p_values]
        theo_means = [results[p]['theo_mean'] for p in p_values]
        theo_stds = [results[p]['theo_std'] for p in p_values]
        
        ax1.bar(x - width/2, sweet_means, width, yerr=sweet_stds, label='Sweet Spot',
                color='green', alpha=0.7, capsize=5)
        ax1.bar(x + width/2, theo_means, width, yerr=theo_stds, label='Theoretical',
                color='red', alpha=0.7, capsize=5)
        ax1.set_xlabel('QAOA Depth (p)', fontsize=12)
        ax1.set_ylabel('Approximation Ratio', fontsize=12)
        ax1.set_title('Sweet Spot vs Theoretical', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'p={p}' for p in p_values])
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # 2. Improvement factor
        ax2 = plt.subplot(2, 3, 2)
        improvements = [(results[p]['improvement'] - 1) * 100 for p in p_values]
        colors = ['green' if i > 0 else 'red' for i in improvements]
        ax2.bar(p_values, improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('QAOA Depth (p)', fontsize=12)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_title('Performance Gain Over Theoretical', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. Statistical significance
        ax3 = plt.subplot(2, 3, 3)
        p_vals = [results[p]['p_value'] for p in p_values]
        colors = ['green' if p < 0.05 else 'orange' for p in p_vals]
        ax3.bar(p_values, [-np.log10(p) for p in p_vals], color=colors, alpha=0.7)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05', linewidth=2)
        ax3.set_xlabel('QAOA Depth (p)', fontsize=12)
        ax3.set_ylabel('-log10(p-value)', fontsize=12)
        ax3.set_title('Statistical Significance', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. Box plots of distributions
        ax4 = plt.subplot(2, 3, 4)
        sweet_data = [results[p]['sweet_performances'] for p in p_values]
        theo_data = [results[p]['theo_performances'] for p in p_values]
        
        positions = []
        all_data = []
        colors_list = []
        for i, p in enumerate(p_values):
            positions.extend([i*2, i*2+0.8])
            all_data.extend([sweet_data[i], theo_data[i]])
            colors_list.extend(['green', 'red'])
        
        bp = ax4.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax4.set_xlabel('QAOA Depth (p)', fontsize=12)
        ax4.set_ylabel('Approximation Ratio', fontsize=12)
        ax4.set_title('Performance Distributions', fontsize=14, fontweight='bold')
        ax4.set_xticks([i*2+0.4 for i in range(len(p_values))])
        ax4.set_xticklabels([f'p={p}' for p in p_values])
        
        # 5. Parameter values
        ax5 = plt.subplot(2, 3, 5)
        for p in p_values:
            gammas = results[p]['sweet_params'][0]
            betas = results[p]['sweet_params'][1]
            for i, (g, b) in enumerate(zip(gammas, betas)):
                ax5.scatter(g, b, s=200, alpha=0.7, label=f'p={p}, layer={i+1}')
        ax5.set_xlabel('γ (rad)', fontsize=12)
        ax5.set_ylabel('β (rad)', fontsize=12)
        ax5.set_title('Optimal Parameters', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Summary text
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary = f"SUMMARY\n{'='*40}\n\n"
        summary += f"Platform: {self.platform.upper()}\n"
        summary += f"Total spent: €{self.spent_eur:.2f}\n\n"
        
        for p in p_values:
            summary += f"p={p}:\n"
            summary += f"  Sweet: {results[p]['sweet_mean']:.1%} ± {results[p]['sweet_std']:.1%}\n"
            summary += f"  Theo:  {results[p]['theo_mean']:.1%} ± {results[p]['theo_std']:.1%}\n"
            summary += f"  Gain:  {(results[p]['improvement']-1)*100:+.1f}%\n"
            if results[p]['p_value'] < 0.05:
                summary += f"  ✓ Significant (p={results[p]['p_value']:.4f})\n"
            summary += "\n"
        
        ax6.text(0.1, 0.9, summary, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax6.transAxes)
        
        plt.suptitle(f'Multi-Layer QAOA Sweet Spot Analysis - {self.platform.upper()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"multi_layer_sweet_spots_{self.platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved: {filename}")
        plt.show()
    
    def save_results(self, results):
        """Save to JSON"""
        save_data = {
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'spent_eur': self.spent_eur,
            'results': {}
        }
        
        for p, data in results.items():
            save_data['results'][f'p={p}'] = {
                'sweet_params': {
                    'gamma': [float(x) for x in data['sweet_params'][0]],
                    'beta': [float(x) for x in data['sweet_params'][1]]
                },
                'sweet_mean': float(data['sweet_mean']),
                'sweet_std': float(data['sweet_std']),
                'theo_mean': float(data['theo_mean']),
                'theo_std': float(data['theo_std']),
                'improvement_factor': float(data['improvement']),
                'p_value': float(data['p_value'])
            }
        
        filename = f"multi_layer_results_{self.platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Results saved: {filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-LAYER QAOA SWEET SPOT FINDER")
    print("="*70)
    print("\nAutomatically finds optimal parameters for p=1,2,3,4")
    print("and compares them with theoretical predictions.\n")
    
    print("Platform options:")
    print("  1. Simulator (free, recommended for testing)")
    print("  2. IQM Emerald (€15-20 for full study)")
    print("  3. Rigetti Ankaa-3 (€5-8 for full study)")
    print("  4. IonQ Forte-1 (€50+ for full study)")
    
    choice = input("\nChoose platform (1-4): ").strip()
    
    platform_map = {
        '1': 'simulator',
        '2': 'emerald',
        '3': 'rigetti',
        '4': 'ionq'
    }
    
    platform = platform_map.get(choice, 'simulator')
    
    if platform != 'simulator':
        print(f"\n⚠️ Hardware testing selected!")
        print(f"Estimated cost: €15-25 for complete study")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            exit()
    
    max_p = int(input("\nMaximum p value to test (1-4): ") or "4")
    
    finder = MultiLayerSweetSpotFinder(platform=platform, shots_per_test=256)
    results = finder.run_complete_study(max_p=max_p)
    
    print("\n" + "="*70)
    print("✅ STUDY COMPLETE!")
    print("="*70)