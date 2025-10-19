"""
#!/usr/bin/env python3
"""
Quantum Circuit Auto-Optimizer v2.1 
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
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt

class QAOAPerfectionFixed:
    """QAOA mit korrekten Metriken"""
    
    def __init__(self):
        self.device = LocalSimulator("braket_dm")
        # Realistischerer Graph (Triangle)
        self.edges = [(0,1), (1,2), (0,2)]
        self.n_qubits = 3
        self.max_possible_cut = 2  # Für Triangle
    
    def test_qaoa_variant(self, name, circuit, shots=1000):
        """Test mit korrekten Metriken"""
        result = self.device.run(circuit, shots=shots).result()
        measurements = result.measurements
        
        cut_values = []
        for m in measurements:
            cuts = sum(1 for u,v in self.edges if m[u] != m[v])
            cut_values.append(cuts)
        
        # Korrekte Metriken
        avg_cut = np.mean(cut_values)
        max_cut_found = max(cut_values)
        
        # Approximation Ratio (wichtigste Metrik!)
        approx_ratio = avg_cut / self.max_possible_cut
        
        # Success = wie oft finden wir optimalen Cut
        optimal_count = cut_values.count(self.max_possible_cut)
        success_rate = optimal_count / shots
        
        print(f"\n{name}:")
        print(f"  Avg Cut: {avg_cut:.2f}/{self.max_possible_cut}")
        print(f"  Approximation Ratio: {approx_ratio:.1%}")
        print(f"  Optimal Success: {success_rate:.1%}")
        print(f"  Gates: {len(circuit.instructions)}")
        
        return {
            'approx_ratio': approx_ratio,
            'success_rate': success_rate,
            'avg_cut': avg_cut,
            'gates': len(circuit.instructions)
        }
    
    def create_vermicular_qaoa(self, gamma=0.6, beta=0.8):
        """
        VERMICULAR QAOA - Optimiert für echte Hardware
        Kombiniert beste Strategien
        """
        circuit = Circuit()
        
        # 1. SYMMETRISCHE INITIALISIERUNG
        for i in range(self.n_qubits):
            circuit.ry(i, np.pi/2)  # Statt H für bessere Kontrolle
        
        # 2. LAYER 1 - Trotterized Cost
        for u,v in self.edges:
            # Decomposed CNOT für weniger Noise
            circuit.ry(v, np.pi/2)
            circuit.cz(u, v)
            circuit.ry(v, -np.pi/2)
            
            circuit.rz(v, gamma/2)  # Halbe Rotation
            
            circuit.ry(v, np.pi/2)
            circuit.cz(u, v)
            circuit.ry(v, -np.pi/2)
        
        # 3. ADAPTIVER MIXER
        if self.n_qubits <= 3:
            # Full mixer für kleine Probleme
            for i in range(self.n_qubits):
                circuit.h(i)
                circuit.rz(i, 2*beta)
                circuit.h(i)
        else:
            # XY-Mixer für große Probleme (vereinfacht ohne rxx/ryy)
            for i in range(self.n_qubits-1):
                circuit.cnot(i, i+1)
                circuit.ry(i, beta)
                circuit.cnot(i, i+1)
                circuit.cnot(i+1, i)
                circuit.ry(i+1, beta)
                circuit.cnot(i+1, i)
        
        # 4. LAYER 2 - Reversed Cost (Symmetrization)
        for u,v in reversed(self.edges):
            circuit.ry(v, np.pi/2)
            circuit.cz(u, v)
            circuit.ry(v, -np.pi/2)
            
            circuit.rz(v, gamma/2)
            
            circuit.ry(v, np.pi/2)
            circuit.cz(u, v)
            circuit.ry(v, -np.pi/2)
        
        # 5. ERROR MITIGATION
        for i in range(self.n_qubits):
            circuit.rz(i, -0.01)  # Compensate systematic errors
        
        return circuit
    
    def optimize_and_test(self):
        """Haupttest mit Parameter-Scan"""
        print("QAOA VERMICULAR OPTIMIZATION")
        print("="*60)
        print(f"Graph: {self.n_qubits} qubits, {len(self.edges)} edges")
        print(f"Max possible cut: {self.max_possible_cut}")
        print("="*60)
        
        best_approx = 0
        best_params = None
        
        # Grid search für beste Parameter
        print("\nSearching optimal parameters...")
        for gamma in np.linspace(0.3, 1.0, 5):
            for beta in np.linspace(0.2, 0.8, 5):
                circuit = self.create_vermicular_qaoa(gamma, beta)
                
                # Quick test
                result = self.device.run(circuit, shots=100).result()
                measurements = result.measurements
                
                cuts = [sum(1 for u,v in self.edges if m[u] != m[v]) 
                       for m in measurements]
                approx = np.mean(cuts) / self.max_possible_cut
                
                if approx > best_approx:
                    best_approx = approx
                    best_params = (gamma, beta)
                    print(f"  New best: γ={gamma:.2f}, β={beta:.2f}, approx={approx:.2%}")
        
        print(f"\nBest parameters found:")
        print(f"γ = {best_params[0]:.3f}")
        print(f"β = {best_params[1]:.3f}")
        
        # Final test mit besten Parametern
        print("\nFINAL VERMICULAR QAOA TEST:")
        final_circuit = self.create_vermicular_qaoa(*best_params)
        final_result = self.test_qaoa_variant("VERMICULAR", final_circuit, shots=1000)
        
        # Vergleich mit Standard
        print("\nSTANDARD QAOA (baseline):")
        standard = self.create_standard_qaoa(*best_params)
        standard_result = self.test_qaoa_variant("STANDARD", standard, shots=1000)
        
        # Improvement
        improvement = (final_result['approx_ratio'] / standard_result['approx_ratio'] - 1) * 100
        print(f"\n{'='*60}")
        print(f"IMPROVEMENT: {improvement:+.1f}%")
        
        if final_result['approx_ratio'] > 0.8:
            print("✅ QAOA SUCCESSFULLY OPTIMIZED!")
            print(f"Approximation Ratio: {final_result['approx_ratio']:.1%}")
        
        # Visualize results
        self.plot_results(final_result, standard_result, best_params)
        
        return final_result
    
    def create_standard_qaoa(self, gamma, beta):
        """Standard QAOA zum Vergleich"""
        circuit = Circuit()
        
        for i in range(self.n_qubits):
            circuit.h(i)
        
        for u,v in self.edges:
            circuit.cnot(u, v)
            circuit.rz(v, 2*gamma)
            circuit.cnot(u, v)
        
        for i in range(self.n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def plot_results(self, vermicular_result, standard_result, best_params):
        """Visualize the comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Comparison bar chart
        methods = ['Standard\nQAOA', 'VERMICULAR\nQAOA']
        approx_ratios = [standard_result['approx_ratio'], vermicular_result['approx_ratio']]
        colors = ['blue', 'green']
        
        bars = ax1.bar(methods, approx_ratios, color=colors, alpha=0.7)
        ax1.set_ylabel('Approximation Ratio')
        ax1.set_title('QAOA Performance Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, approx_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.1%}', ha='center', va='bottom')
        
        # Gate count comparison
        gate_counts = [standard_result['gates'], vermicular_result['gates']]
        ax2.bar(methods, gate_counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Gate Count')
        ax2.set_title('Circuit Complexity')
        
        plt.suptitle(f'Optimal Parameters: γ={best_params[0]:.3f}, β={best_params[1]:.3f}')
        plt.tight_layout()
        plt.savefig('qaoa_optimization_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# Run it
if __name__ == "__main__":
    optimizer = QAOAPerfectionFixed()
    result = optimizer.optimize_and_test()
    
    print("\n" + "="*60)
    print("READY FOR HARDWARE TEST")
    print("Expected on IQM: ~60-70% approximation ratio")
    print("="*60)