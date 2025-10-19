"""
#!/usr/bin/env python3
"""
sigma c validation
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
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import cirq
import networkx as nx
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: POWER ANALYSIS
# ============================================================================

class PowerAnalysis:
    def __init__(self, shots=256):
        try:
            from statsmodels.stats.power import TTestIndPower
            self.power_calc = TTestIndPower()
        except:
            self.power_calc = None
        self.shots = shots
    
    def calculate_power_scipy(self, effect_size, n, alpha=0.05):
        """Scipy fallback for power calculation"""
        if effect_size > 10:  # Very large effects
            return 1.0
        
        df = 2 * n - 2
        ncp = effect_size * np.sqrt(n / 2)
        t_crit = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        return min(max(power, 0), 1)
    
    def hedges_g_conservative(self, mean1, mean2, sd1, sd2, n1, n2):
        """Ultra-conservative: Use larger of measured SD or shot noise, plus margin"""
        shot_noise1 = np.sqrt(mean1 * (1 - mean1) / self.shots)
        shot_noise2 = np.sqrt(mean2 * (1 - mean2) / self.shots)
        
        sd1_total = max(sd1, shot_noise1) * 1.5
        sd2_total = max(sd2, shot_noise2) * 1.5
        
        sp = np.sqrt(((n1-1)*sd1_total**2 + (n2-1)*sd2_total**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / sp
        correction = 1 - (3 / (4*(n1 + n2) - 9))
        
        return d * correction, sp
    
    def analyze_comparisons(self):
        comparisons = {
            'VERM vs Base (IQM)': {
                'mean1': 0.930, 'sd1': 0.016, 'n1': 5,
                'mean2': 0.183, 'sd2': 0.024, 'n2': 5,
                'category': 'Primary'
            },
            'VERM vs Base (Rigetti)': {
                'mean1': 0.854, 'sd1': 0.023, 'n1': 5,
                'mean2': 0.157, 'sd2': 0.021, 'n2': 5,
                'category': 'Primary'
            },
            'QAOA Opt vs Theory (IQM)': {
                'mean1': 0.945, 'sd1': 0.011, 'n1': 5,
                'mean2': 0.396, 'sd2': 0.041, 'n2': 5,
                'category': 'Primary'
            },
            'QAOA Opt vs Theory (Rig)': {
                'mean1': 0.790, 'sd1': 0.031, 'n1': 3,
                'mean2': 0.371, 'sd2': 0.038, 'n2': 3,
                'category': 'Primary'
            },
            'IQM Quant vs Sim': {
                'mean1': 0.945, 'sd1': 0.011, 'n1': 5,
                'mean2': 0.926, 'sd2': 0.020, 'n2': 5,
                'category': 'Secondary'
            },
            'Rig-Opt vs Sim': {
                'mean1': 0.785, 'sd1': 0.033, 'n1': 3,
                'mean2': 0.790, 'sd2': 0.031, 'n2': 3,
                'category': 'Secondary'
            },
            '20q vs 30q': {
                'mean1': 0.561, 'sd1': 0.026, 'n1': 3,
                'mean2': 0.537, 'sd2': 0.031, 'n2': 3,
                'category': 'Scaling'
            },
            '30q vs 40q': {
                'mean1': 0.537, 'sd1': 0.031, 'n1': 3,
                'mean2': 0.561, 'sd2': 0.028, 'n2': 3,
                'category': 'Scaling'
            },
        }
        
        results = []
        
        for name, data in comparisons.items():
            g, _ = self.hedges_g_conservative(
                data['mean1'], data['mean2'],
                data['sd1'], data['sd2'],
                data['n1'], data['n2']
            )
            
            power = self.calculate_power_scipy(abs(g), data['n1'])
            
            results.append({
                'Comparison': name,
                'Category': data['category'],
                'N': data['n1'],
                'g': abs(g),
                'Power': power
            })
        
        return pd.DataFrame(results)

# ============================================================================
# PART 2: NOISE VALIDATION (FIXED)
# ============================================================================

class NoiseValidation:
    def __init__(self):
        self.noise_levels = np.linspace(0, 0.3, 20)
        self.shots = 1000
    
    def run_validation(self):
        print("\n=== NOISE VALIDATION (Depolarizing Model) ===\n")
        
        results = {}
        simulator = cirq.Simulator()
        
        for circuit_type in ['bell', 'product']:
            qubits = cirq.LineQubit.range(2)
            
            # Create base circuit WITHOUT measurement
            if circuit_type == 'bell':
                base_circuit = cirq.Circuit(
                    cirq.H(qubits[0]),
                    cirq.CNOT(qubits[0], qubits[1])
                )
            else:
                base_circuit = cirq.Circuit()
            
            fidelities = []
            
            for epsilon in tqdm(self.noise_levels, desc=f"{circuit_type:8s}", leave=False):
                # Apply noise
                noisy_circuit = base_circuit.with_noise(cirq.depolarize(epsilon))
                
                # Add measurement to COPY
                measured_circuit = noisy_circuit.copy()
                measured_circuit.append(cirq.measure(*qubits, key='m'))
                
                # Run simulation
                result = simulator.run(measured_circuit, repetitions=self.shots)
                counts = result.histogram(key='m')
                
                # Calculate fidelity
                if circuit_type == 'bell':
                    fid = (counts.get(0, 0) + counts.get(3, 0)) / self.shots
                else:
                    fid = counts.get(0, 0) / self.shots
                
                fidelities.append(fid)
            
            # Compute σ_c
            fidelities = np.array(fidelities)
            smoothed = gaussian_filter1d(fidelities, sigma=1.0)
            gradient = np.gradient(smoothed, self.noise_levels)
            sigma_c_idx = np.argmax(np.abs(gradient))
            sigma_c = self.noise_levels[sigma_c_idx]
            
            # Bootstrap CI
            sigma_c_samples = []
            for _ in range(1000):
                indices = np.random.choice(len(fidelities), len(fidelities), replace=True)
                resampled = fidelities[indices]
                smoothed_boot = gaussian_filter1d(resampled, sigma=1.0)
                gradient_boot = np.gradient(smoothed_boot, self.noise_levels)
                sc_boot = self.noise_levels[np.argmax(np.abs(gradient_boot))]
                sigma_c_samples.append(sc_boot)
            
            ci_lower = np.percentile(sigma_c_samples, 2.5)
            ci_upper = np.percentile(sigma_c_samples, 97.5)
            
            results[circuit_type] = {
                'sigma_c': sigma_c,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'fidelities': fidelities.tolist()
            }
            
            print(f"{circuit_type:8s}: σ_c = {sigma_c:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        return results

# ============================================================================
# PART 3: QAOA SEED SENSITIVITY (FIXED)
# ============================================================================

class QAOASeedTest:
    def __init__(self):
        self.seeds = [42, 123, 456, 789, 1001]
        self.shots = 500
    
    def create_qaoa_circuit(self, gamma, beta):
        """Create QAOA circuit for 3-qubit triangle MaxCut"""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()
        
        # Initial state
        circuit.append([cirq.H(q) for q in qubits])
        
        # Cost layer (triangle: edges 0-1, 1-2, 2-0)
        edges = [(0, 1), (1, 2), (2, 0)]
        for i, j in edges:
            circuit.append(cirq.CNOT(qubits[i], qubits[j]))
            circuit.append(cirq.rz(2 * gamma)(qubits[j]))
            circuit.append(cirq.CNOT(qubits[i], qubits[j]))
        
        # Mixing layer
        circuit.append([cirq.rx(2 * beta)(q) for q in qubits])
        
        # Measurement
        circuit.append(cirq.measure(*qubits, key='m'))
        
        return circuit
    
    def maxcut_cost(self, bitstring):
        """Calculate cut size for triangle graph"""
        edges = [(0, 1), (1, 2), (2, 0)]
        return sum(1 for i, j in edges if bitstring[i] != bitstring[j])
    
    def evaluate(self, gamma, beta, seed):
        """Evaluate QAOA at given parameters"""
        np.random.seed(seed)
        circuit = self.create_qaoa_circuit(gamma, beta)
        
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=self.shots)
        
        counts = result.histogram(key='m')
        total_cost = 0
        
        for bitstring_int, count in counts.items():
            bitstring = tuple(int(b) for b in f"{bitstring_int:03b}")
            cost = self.maxcut_cost(bitstring)
            total_cost += cost * count
        
        avg_cost = total_cost / self.shots
        max_cut = 2  # Triangle max cut
        
        return avg_cost / max_cut
    
    def run_test(self):
        print("\n=== SEED SENSITIVITY (10×10 Coarse Grid) ===\n")
        
        gamma_range = np.linspace(0, np.pi, 10)
        beta_range = np.linspace(0, np.pi/2, 10)
        
        results = []
        
        for seed in self.seeds:
            best_ratio = 0
            best_gamma = 0
            best_beta = 0
            
            # Grid search
            for gamma in tqdm(gamma_range, desc=f"Seed {seed:4d}", leave=False):
                for beta in beta_range:
                    ratio = self.evaluate(gamma, beta, seed)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_gamma = gamma
                        best_beta = beta
            
            results.append({
                'seed': seed,
                'gamma': best_gamma,
                'beta': best_beta,
                'ratio': best_ratio
            })
            
            print(f"Seed {seed:4d}: γ={best_gamma:.3f}, β={best_beta:.3f}, r={best_ratio:.3f}")
        
        df = pd.DataFrame(results)
        
        stats = {
            'gamma_mean': df['gamma'].mean(),
            'gamma_std': df['gamma'].std(),
            'gamma_cv': df['gamma'].std() / df['gamma'].mean() * 100 if df['gamma'].mean() > 0 else 0,
            'beta_mean': df['beta'].mean(),
            'beta_std': df['beta'].std(),
            'beta_cv': df['beta'].std() / df['beta'].mean() * 100 if df['beta'].mean() > 0.01 else 0,
            'ratio_mean': df['ratio'].mean(),
            'ratio_std': df['ratio'].std(),
            'ratio_cv': df['ratio'].std() / df['ratio'].mean() * 100
        }
        
        return df, stats

# ============================================================================
# PLOTTING
# ============================================================================

def create_plots(power_df, noise_results, seed_df):
    """Create summary plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Power by category
    ax1 = axes[0, 0]
    categories = power_df['Category'].unique()
    colors = {'Primary': 'green', 'Secondary': 'orange', 'Scaling': 'red'}
    
    for cat in categories:
        cat_df = power_df[power_df['Category'] == cat]
        ax1.scatter(cat_df['g'], cat_df['Power'], 
                   label=cat, s=100, alpha=0.7, c=colors[cat])
    
    ax1.axhline(0.8, color='red', linestyle='--', label='80% threshold')
    ax1.set_xlabel('Effect Size (g)', fontweight='bold')
    ax1.set_ylabel('Statistical Power', fontweight='bold')
    ax1.set_title('Power Analysis', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, max(power_df['g']) * 1.1])
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Noise curves
    ax2 = axes[0, 1]
    for circuit_type in ['bell', 'product']:
        fids = noise_results[circuit_type]['fidelities']
        noise_levels = np.linspace(0, 0.3, len(fids))
        ax2.plot(noise_levels, fids, 'o-', label=circuit_type, alpha=0.7)
        
        sc = noise_results[circuit_type]['sigma_c']
        ax2.axvline(sc, linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Noise Level ε', fontweight='bold')
    ax2.set_ylabel('Fidelity', fontweight='bold')
    ax2.set_title('Noise Degradation', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Seed parameter scatter
    ax3 = axes[1, 0]
    ax3.scatter(seed_df['gamma'], seed_df['beta'], s=150, alpha=0.7)
    
    for _, row in seed_df.iterrows():
        ax3.annotate(f"{int(row['seed'])}", 
                    (row['gamma'], row['beta']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('γ (rad)', fontweight='bold')
    ax3.set_ylabel('β (rad)', fontweight='bold')
    ax3.set_title('Parameter Stability', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Ratio distribution
    ax4 = axes[1, 1]
    ax4.hist(seed_df['ratio'], bins=8, alpha=0.7, edgecolor='black')
    ax4.axvline(seed_df['ratio'].mean(), color='red', linestyle='--', 
               label=f"Mean={seed_df['ratio'].mean():.3f}")
    ax4.set_xlabel('Approximation Ratio', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Performance Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('revision_validation.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: revision_validation.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("REVISION VALIDATION - FINAL WORKING VERSION")
    print("="*80)
    start = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}\n")
    
    # Part 1
    print("[1/3] POWER ANALYSIS")
    print("-" * 40)
    power = PowerAnalysis()
    power_df = power.analyze_comparisons()
    
    print(f"\n{'Comparison':<30} {'g':>6} {'Power':>7}")
    print("-" * 45)
    for _, row in power_df.iterrows():
        power_str = f">{row['Power']:.3f}" if row['Power'] > 0.999 else f"{row['Power']:.3f}"
        print(f"{row['Comparison']:<30} {row['g']:>6.2f} {power_str:>7}")
    
    # Part 2
    print("\n[2/3] NOISE VALIDATION")
    print("-" * 40)
    noise = NoiseValidation()
    noise_results = noise.run_validation()
    
    # Part 3
    print("\n[3/3] SEED SENSITIVITY")
    print("-" * 40)
    qaoa = QAOASeedTest()
    seed_df, seed_stats = qaoa.run_test()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n1. POWER ANALYSIS:")
    primary = power_df[power_df['Category'] == 'Primary']
    print(f"   Primary comparisons: {len(primary)}")
    print(f"   Average g: {primary['g'].mean():.2f}")
    print(f"   All well-powered (>0.80): {(primary['Power'] > 0.8).all()}")
    
    print("\n2. NOISE VALIDATION:")
    bell_sc = noise_results['bell']['sigma_c']
    prod_sc = noise_results['product']['sigma_c']
    print(f"   Bell σ_c: {bell_sc:.3f}")
    print(f"   Product σ_c: {prod_sc:.3f}")
    if prod_sc > 0.01:
        print(f"   Ratio: {bell_sc/prod_sc:.1f}×")
    
    print("\n3. SEED SENSITIVITY:")
    print(f"   γ: {seed_stats['gamma_mean']:.3f} ± {seed_stats['gamma_std']:.3f} (CV={seed_stats['gamma_cv']:.1f}%)")
    print(f"   β: {seed_stats['beta_mean']:.3f} ± {seed_stats['beta_std']:.3f} (CV={seed_stats['beta_cv']:.1f}%)")
    print(f"   r: {seed_stats['ratio_mean']:.3f} ± {seed_stats['ratio_std']:.3f} (CV={seed_stats['ratio_cv']:.1f}%)")
    
    # Save
    results = {
        'power_analysis': power_df.to_dict('records'),
        'noise_validation': {
            'bell': {k: v for k, v in noise_results['bell'].items() if k != 'fidelities'},
            'product': {k: v for k, v in noise_results['product'].items() if k != 'fidelities'}
        },
        'seed_sensitivity': {
            'data': seed_df.to_dict('records'),
            'statistics': seed_stats
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (datetime.now() - start).total_seconds() / 60
        }
    }
    
    with open('revision_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved: revision_results.json")
    
    # Plots
    create_plots(power_df, noise_results, seed_df)
    
    end = datetime.now()
    runtime = (end - start).total_seconds() / 60
    print(f"\nEnd: {end.strftime('%H:%M:%S')}")
    print(f"Runtime: {runtime:.1f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()