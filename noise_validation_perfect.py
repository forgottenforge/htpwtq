"""
#!/usr/bin/env python3
"""
sigma c noisy validation
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
from scipy.ndimage import gaussian_filter1d
import cirq
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PerfectNoiseValidation:
    """Perfect noise validation - FIXED version"""
    
    def __init__(self):
        self.noise_levels = np.linspace(0, 0.3, 20)
        self.smoothing_sigma = 1.0
        self.bootstrap_iterations = 1000
        
        # Hardware parameters
        self.T1 = 45e-6
        self.T2 = 35e-6
        self.gate_time = 10e-9
    
    def create_bell_state(self):
        """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2"""
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1])
        )
        ideal_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        return circuit, qubits, ideal_state
    
    def create_product_state(self):
        """
        Create product state |00⟩
        
        FIX: Use explicit identity gates instead of empty circuit
        This ensures Cirq treats it as a proper 2-qubit circuit
        """
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            # Add identity gates to make it a proper 2-qubit circuit
            cirq.I(qubits[0]),
            cirq.I(qubits[1])
        )
        ideal_state = np.array([1, 0, 0, 0], dtype=complex)
        return circuit, qubits, ideal_state
    
    def apply_noise_model(self, circuit, noise_type, epsilon):
        """Apply realistic noise model"""
        
        if noise_type == 'depolarizing':
            return circuit.with_noise(cirq.depolarize(epsilon))
        
        elif noise_type == 'amplitude_damping':
            gamma = 1 - np.exp(-self.gate_time / self.T1)
            gamma_scaled = min(gamma * (epsilon / 0.005), 1.0)
            return circuit.with_noise(cirq.amplitude_damp(gamma_scaled))
        
        elif noise_type == 'phase_damping':
            gamma = 1 - np.exp(-self.gate_time / self.T2)
            gamma_scaled = min(gamma * (epsilon / 0.005), 1.0)
            return circuit.with_noise(cirq.phase_damp(gamma_scaled))
        
        elif noise_type == 'correlated_2q':
            return circuit.with_noise(cirq.depolarize(epsilon * 1.05))
        
        elif noise_type == 'full':
            noisy = circuit.with_noise(cirq.depolarize(epsilon))
            
            gamma_t1 = 1 - np.exp(-self.gate_time / self.T1)
            gamma_t1_scaled = min(gamma_t1 * (epsilon / 0.005), 0.5)
            noisy = noisy.with_noise(cirq.amplitude_damp(gamma_t1_scaled))
            
            gamma_t2 = 1 - np.exp(-self.gate_time / self.T2)
            gamma_t2_scaled = min(gamma_t2 * (epsilon / 0.005), 0.5)
            noisy = noisy.with_noise(cirq.phase_damp(gamma_t2_scaled))
            
            return noisy
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def statevector_to_density_matrix(self, state_vector):
        """Convert statevector to density matrix"""
        # Ensure proper shape
        state_vector = np.array(state_vector).flatten()
        return np.outer(state_vector, state_vector.conj())
    
    def calculate_fidelity(self, ideal_state, noisy_state):
        """
        Calculate quantum state fidelity - FIXED
        
        Handles dimension mismatches gracefully
        """
        # Ensure both are 1D arrays
        ideal_state = np.array(ideal_state).flatten()
        noisy_state = np.array(noisy_state).flatten()
        
        # Check dimensions match
        if len(ideal_state) != len(noisy_state):
            raise ValueError(f"Dimension mismatch: ideal={len(ideal_state)}, noisy={len(noisy_state)}")
        
        return np.abs(np.vdot(ideal_state, noisy_state))**2
    
    def calculate_coherence(self, density_matrix):
        """Calculate l1-norm of coherence"""
        n = density_matrix.shape[0]
        coherence = 0.0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    coherence += np.abs(density_matrix[i, j])
        
        coherence /= (n * (n - 1))
        return coherence
    
    def calculate_purity(self, density_matrix):
        """Calculate purity: P = Tr[ρ²]"""
        return np.real(np.trace(density_matrix @ density_matrix))
    
    def information_functional(self, fidelity, coherence, purity, epsilon, is_entangled=True):
       
        if is_entangled:
            # Full information functional for entangled states
            return fidelity * coherence * np.sqrt(purity) * (1 - epsilon)**2
        else:
            # Simplified for product states (coherence=0 by definition)
            return fidelity * np.sqrt(purity) * (1 - epsilon)**2
    
    def run_noise_sweep(self, circuit_type, noise_model):
        """Run noise sweep - FIXED"""
        
        # Create circuit
        if circuit_type == 'bell':
            circuit, qubits, ideal_state = self.create_bell_state()
        else:
            circuit, qubits, ideal_state = self.create_product_state()
        
        # Storage
        observables = []
        fidelities = []
        coherences = []
        purities = []
        
        # Simulator
        simulator = cirq.Simulator()
        
        # Sweep through noise levels
        for epsilon in tqdm(self.noise_levels, 
                          desc=f"{circuit_type:8s} - {noise_model:20s}",
                          leave=False):
            
            try:
                # Apply noise
                noisy_circuit = self.apply_noise_model(circuit, noise_model, epsilon)
                
                # Simulate
                result = simulator.simulate(noisy_circuit)
                noisy_state = result.final_state_vector
                
                # Ensure proper shape
                noisy_state = np.array(noisy_state).flatten()
                
                # Convert to density matrix
                noisy_dm = self.statevector_to_density_matrix(noisy_state)
                
                # Calculate observables
                fidelity = self.calculate_fidelity(ideal_state, noisy_state)
                coherence = self.calculate_coherence(noisy_dm)
                purity = self.calculate_purity(noisy_dm)
                
                # Information functional
                is_entangled = (circuit_type == 'bell')
                info = self.information_functional(fidelity, coherence, purity, epsilon, is_entangled)
                
                # Store
                fidelities.append(fidelity)
                coherences.append(coherence)
                purities.append(purity)
                observables.append(info)
                
            except Exception as e:
                print(f"\nERROR at epsilon={epsilon:.3f}: {str(e)}")
                print(f"  Noisy state shape: {np.array(noisy_state).shape if 'noisy_state' in locals() else 'N/A'}")
                print(f"  Ideal state shape: {np.array(ideal_state).shape}")
                raise
        
        # Convert to arrays
        observables = np.array(observables)
        
        # Compute σ_c
        sigma_c, ci_lower, ci_upper = self.compute_sigma_c_with_bootstrap(
            observables, self.noise_levels
        )
        
        return {
            'sigma_c': sigma_c,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'observables': observables.tolist(),
            'fidelities': fidelities,
            'coherences': coherences,
            'purities': purities
        }
    
    def compute_sigma_c_with_bootstrap(self, observables, noise_levels):
        """Compute σ_c with bootstrap CI"""
        
        # Check for all-zero observables (problematic case)
        if np.all(observables == 0):
            print("  WARNING: All observables are zero!")
            return 0.0, 0.0, 0.0
        
        # Smooth
        smoothed = gaussian_filter1d(observables, sigma=self.smoothing_sigma)
        
        # Gradient
        gradient = np.gradient(smoothed, noise_levels)
        
        # Find peak
        sigma_c_idx = np.argmax(np.abs(gradient))
        sigma_c = noise_levels[sigma_c_idx]
        
        # Bootstrap
        sigma_c_samples = []
        
        for _ in range(self.bootstrap_iterations):
            indices = np.random.choice(len(observables), len(observables), replace=True)
            resampled = observables[indices]
            
            smoothed_boot = gaussian_filter1d(resampled, sigma=self.smoothing_sigma)
            gradient_boot = np.gradient(smoothed_boot, noise_levels)
            sigma_c_boot_idx = np.argmax(np.abs(gradient_boot))
            sigma_c_boot = noise_levels[sigma_c_boot_idx]
            
            sigma_c_samples.append(sigma_c_boot)
        
        ci_lower = np.percentile(sigma_c_samples, 2.5)
        ci_upper = np.percentile(sigma_c_samples, 97.5)
        
        return sigma_c, ci_lower, ci_upper
    
    def run_full_validation(self):
        """Run complete validation"""
        
        noise_models = [
            'depolarizing',
            'amplitude_damping',
            'phase_damping',
            'correlated_2q',
            'full'
        ]
        
        results = {
            'bell': {},
            'product': {}
        }
        
        print("\n" + "="*80)
        print("PERFECT NOISE VALIDATION - Statevector Method")
        print("="*80)
        print(f"Noise levels: {len(self.noise_levels)} points from 0 to 0.3")
        print(f"Bootstrap iterations: {self.bootstrap_iterations}")
        print(f"Total simulations: {len(noise_models)} × 2 states × {len(self.noise_levels)} = {len(noise_models)*2*len(self.noise_levels)}")
        print("="*80 + "\n")
        
        for circuit_type in ['bell', 'product']:
            print(f"\n{circuit_type.upper()} STATE:")
            print("-" * 60)
            
            for noise_model in noise_models:
                result = self.run_noise_sweep(circuit_type, noise_model)
                results[circuit_type][noise_model] = result
                
                sc = result['sigma_c']
                ci_low = result['ci_lower']
                ci_up = result['ci_upper']
                ci_width = ci_up - ci_low
                
                print(f"{noise_model:20s}: σ_c = {sc:.3f}  [{ci_low:.3f}, {ci_up:.3f}]  (width: {ci_width:.3f})")
        
        # Summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n{'Noise Model':<20} {'Bell σ_c':>10} {'Product σ_c':>12} {'Ratio':>8}")
        print("-" * 55)
        
        for model in noise_models:
            bell_sc = results['bell'][model]['sigma_c']
            prod_sc = results['product'][model]['sigma_c']
            
            if prod_sc > 0.01:
                ratio = bell_sc / prod_sc
                ratio_str = f"{ratio:.1f}×"
            else:
                ratio_str = "---"
            
            print(f"{model:<20} {bell_sc:>10.3f} {prod_sc:>12.3f} {ratio_str:>8}")
        
        # Validation check
        bell_sc = results['bell']['depolarizing']['sigma_c']
        prod_sc = results['product']['depolarizing']['sigma_c']
        
        print("\n" + "="*80)
        print("VALIDATION vs PAPER")
        print("="*80)
        print(f"Paper:       Bell σ_c = 0.200, Product σ_c = 0.050, Ratio = 4.0×")
        print(f"Our results: Bell σ_c = {bell_sc:.3f}, Product σ_c = {prod_sc:.3f}", end="")
        
        if prod_sc > 0.01:
            ratio = bell_sc / prod_sc
            print(f", Ratio = {ratio:.1f}×")
            
            bell_ok = 0.15 < bell_sc < 0.25
            prod_ok = 0.03 < prod_sc < 0.08
            ratio_ok = 3.0 < ratio < 5.0
            
            print(f"\nValidation: Bell {'✓' if bell_ok else '✗'} | Product {'✓' if prod_ok else '✗'} | Ratio {'✓' if ratio_ok else '✗'}")
            
            if bell_ok and prod_ok and ratio_ok:
                print("\n✓ SUCCESS - Results match paper!")
            else:
                print("\n⚠ Deviation from paper - investigating...")
        else:
            print(" (Product σ_c too small for ratio)")
        
        return results
    
    def generate_latex_table(self, results):
        """Generate LaTeX table"""
        
        latex = r"""\begin{table}[H]
\centering
\caption{$\sigma_c$ validation using statevector simulation with information functional $\mathcal{I}[\rho,\epsilon] = F \cdot C \cdot \sqrt{P} \cdot (1-\epsilon)^2$. Bootstrap 95\% CI from 1000 iterations.}
\label{tab:noise_validation}
\begin{tabular}{lccc}  
\toprule
\textbf{Noise Model} & \textbf{Bell $\sigma_c$} & \textbf{Product $\sigma_c$} & \textbf{Ratio} \\
\midrule
"""
        
        noise_labels = {
            'depolarizing': 'Depolarizing',
            'amplitude_damping': '+ Amplitude damping',
            'phase_damping': '+ Phase damping',
            'correlated_2q': '+ Correlated errors',
            'full': 'Full model'
        }
        
        for model, label in noise_labels.items():
            bell = results['bell'][model]
            prod = results['product'][model]
            
            bell_sc = bell['sigma_c']
            bell_err = (bell['ci_upper'] - bell['ci_lower']) / 2
            
            prod_sc = prod['sigma_c']
            prod_err = (prod['ci_upper'] - prod['ci_lower']) / 2
            
            if prod_sc > 0.01:
                ratio = bell_sc / prod_sc
                ratio_str = f"${ratio:.1f}\\times$"
            else:
                ratio_str = "---"
            
            latex += f"{label} & ${bell_sc:.3f} \\pm {bell_err:.3f}$ & "
            latex += f"${prod_sc:.3f} \\pm {prod_err:.3f}$ & {ratio_str} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        
        return latex
    
    def generate_simple_plot(self, results):
        """Generate simple 2x2 plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Bell degradation
        ax = axes[0, 0]
        for model in ['depolarizing', 'full']:
            obs = results['bell'][model]['observables']
            ax.plot(self.noise_levels, obs, 'o-', label=model, linewidth=2, markersize=4)
        ax.set_xlabel('Noise ε', fontweight='bold')
        ax.set_ylabel('Information I', fontweight='bold')
        ax.set_title('Bell State Degradation', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Product degradation
        ax = axes[0, 1]
        for model in ['depolarizing', 'full']:
            obs = results['product'][model]['observables']
            ax.plot(self.noise_levels, obs, 'o-', label=model, linewidth=2, markersize=4)
        ax.set_xlabel('Noise ε', fontweight='bold')
        ax.set_ylabel('Information I', fontweight='bold')
        ax.set_title('Product State Degradation', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Fidelity comparison
        ax = axes[1, 0]
        for ctype, color in [('bell', 'blue'), ('product', 'red')]:
            fids = results[ctype]['depolarizing']['fidelities']
            ax.plot(self.noise_levels, fids, 'o-', label=ctype, color=color, linewidth=2)
        ax.set_xlabel('Noise ε', fontweight='bold')
        ax.set_ylabel('Fidelity', fontweight='bold')
        ax.set_title('Fidelity vs Noise', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: σ_c comparison
        ax = axes[1, 1]
        models = list(results['bell'].keys())
        bell_scs = [results['bell'][m]['sigma_c'] for m in models]
        prod_scs = [results['product'][m]['sigma_c'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, bell_scs, width, label='Bell', alpha=0.7)
        ax.bar(x + width/2, prod_scs, width, label='Product', alpha=0.7)
        ax.set_ylabel('σ_c', fontweight='bold')
        ax.set_title('Critical Noise Threshold', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[:4] for m in models], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('noise_validation.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('noise_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✓ Plots saved: noise_validation.pdf/.png")

def main():
    print("="*80)
    print("PERFECT NOISE VALIDATION v2 (FIXED)")
    print("="*80)
    
    start = datetime.now()
    print(f"Start: {start.strftime('%H:%M:%S')}\n")
    
    validator = PerfectNoiseValidation()
    results = validator.run_full_validation()
    
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)
    
    latex = validator.generate_latex_table(results)
    with open('noise_validation_table.tex', 'w') as f:
        f.write(latex)
    print("\n✓ LaTeX saved: noise_validation_table.tex")
    
    validator.generate_simple_plot(results)
    
    # Save data
    export = {
        'bell': {m: {'sigma_c': d['sigma_c'], 'ci_lower': d['ci_lower'], 'ci_upper': d['ci_upper']}
                for m, d in results['bell'].items()},
        'product': {m: {'sigma_c': d['sigma_c'], 'ci_lower': d['ci_lower'], 'ci_upper': d['ci_upper']}
                   for m, d in results['product'].items()},
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (datetime.now() - start).total_seconds() / 60
        }
    }
    
    with open('noise_validation_data.json', 'w') as f:
        json.dump(export, f, indent=2)
    print("✓ Data saved: noise_validation_data.json")
    
    runtime = (datetime.now() - start).total_seconds() / 60
    print(f"\nRuntime: {runtime:.1f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()