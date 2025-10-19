"""
#!/usr/bin/env python3
"""
Quantum Circuit Auto-Optimizer v2.3 
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
from braket.aws import AwsDevice
import matplotlib.pyplot as plt
from datetime import datetime
import json

class QAOARealityCheck:
    """
    Ehrliche QAOA-Tests ohne Bullshit
    """
    
    def __init__(self, platform='simulator'):
        self.platform = platform
        
        if platform == 'iqm':
            # EMERALD nicht Garnet!
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald")
            print("IQM Emerald initialized (Garnet is offline)")
        else:
            self.device = LocalSimulator("braket_dm")
            print("Simulator initialized")
        
        self.edges = [(0,1), (1,2), (0,2)]
        self.n_qubits = 3
        self.max_cut = 2
    
    def create_standard_qaoa(self, gamma, beta):
        """Standard QAOA - offenbar der wahre Champion"""
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
    
    def create_simplified_vermicular(self, gamma, beta):
        """Vereinfachte Version - weniger ist mehr"""
        circuit = Circuit()
        
        # Initial state - nur H, keine extras
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Cost operator mit CZ statt CNOT
        for u,v in self.edges:
            circuit.cz(u, v)
            circuit.rz(v, 2*gamma)
            circuit.cz(u, v)
        
        # Mixer 
        for i in range(self.n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def create_minimal_qaoa(self, gamma, beta):
        """Absolut minimal"""
        circuit = Circuit()
        
        # Init
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Single cost application
        for u,v in self.edges:
            circuit.cz(u, v)
        
        for i in range(self.n_qubits):
            circuit.rz(i, gamma)
        
        # Mixer
        for i in range(self.n_qubits):
            circuit.rx(i, 2*beta)
        
        return circuit
    
    def find_truth(self):
        """Was funktioniert WIRKLICH?"""
        print("\n" + "="*70)
        print("QAOA REALITY CHECK")
        print("="*70)
        
        # Die "optimalen" Parameter vom Simulator
        test_params = [
            (0.217, 1.284, "Simulator optimal"),
            (0.3, 0.65, "Original v1"),
            (0.317, 1.151, "Landscape peak"),
            (np.pi/4, np.pi/8, "Theoretical"),
            (0.25, 1.25, "IQM quantized")
        ]
        
        results = {}
        
        print("\nTesting parameter sets on all variants...")
        print("-"*50)
        
        for gamma, beta, name in test_params:
            print(f"\nParameter set: {name}")
            print(f"γ={gamma:.3f}, β={beta:.3f}")
            
            # Test alle Varianten
            variants = {
                'Standard': self.create_standard_qaoa,
                'Simplified': self.create_simplified_vermicular,
                'Minimal': self.create_minimal_qaoa
            }
            
            for variant_name, create_func in variants.items():
                circuit = create_func(gamma, beta)
                
                # Test
                shots = 200 if self.platform == 'iqm' else 500
                result = self.device.run(circuit, shots=shots).result()
                measurements = result.measurements
                
                # Analyse
                cuts = [sum(1 for u,v in self.edges if m[u] != m[v]) 
                       for m in measurements]
                approx_ratio = np.mean(cuts) / self.max_cut
                
                key = f"{name}_{variant_name}"
                results[key] = {
                    'gamma': gamma,
                    'beta': beta,
                    'variant': variant_name,
                    'approx_ratio': approx_ratio,
                    'gates': len(circuit.instructions)
                }
                
                print(f"  {variant_name}: {approx_ratio:.1%} ({len(circuit.instructions)} gates)")
        
        # Finde den ECHTEN Gewinner
        best = max(results.items(), key=lambda x: x[1]['approx_ratio'])
        
        print("\n" + "="*70)
        print("THE TRUTH:")
        print(f"Best: {best[0]}")
        print(f"Performance: {best[1]['approx_ratio']:.1%}")
        print(f"Parameters: γ={best[1]['gamma']:.3f}, β={best[1]['beta']:.3f}")
        print("="*70)
        
        return results
    
    def test_on_iqm(self, gamma, beta):
        """Direkter IQM Test mit besten Parametern"""
        print("\n" + "="*70)
        print("IQM EMERALD TEST")
        print("="*70)
        
        circuit = self.create_standard_qaoa(gamma, beta)
        
        print(f"Testing Standard QAOA with γ={gamma:.3f}, β={beta:.3f}")
        print(f"Circuit: {len(circuit.instructions)} gates")
        
        shots = 256
        print(f"Running {shots} shots on IQM Emerald...")
        
        start_time = datetime.now()
        result = self.device.run(circuit, shots=shots).result()
        runtime = (datetime.now() - start_time).total_seconds()
        
        measurements = result.measurements
        
        # Analyse
        cuts = [sum(1 for u,v in self.edges if m[u] != m[v]) 
               for m in measurements]
        approx_ratio = np.mean(cuts) / self.max_cut
        success_rate = cuts.count(self.max_cut) / shots
        
        print(f"\nResults:")
        print(f"Approximation Ratio: {approx_ratio:.1%}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Runtime: {runtime:.1f}s")
        
        # Distribution plot
        plt.figure(figsize=(8, 5))
        plt.hist(cuts, bins=range(max(cuts)+2), alpha=0.7, edgecolor='black')
        plt.xlabel('Cut Value')
        plt.ylabel('Frequency')
        plt.title(f'IQM Results - Approx Ratio: {approx_ratio:.1%}')
        plt.axvline(self.max_cut, color='r', linestyle='--', label='Optimal')
        plt.legend()
        plt.show()
        
        return approx_ratio
    
    def run(self):
        """Hauptausführung"""
        if self.platform == 'simulator':
            # Auf Simulator: Finde die Wahrheit
            results = self.find_truth()
            
            # Save
            filename = f'qaoa_reality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {filename}")
            
        else:
            # Auf IQM: Teste nur das Beste
            print("Using best parameters from simulator...")
            
            # Die ECHTEN besten Parameter (Standard QAOA)
            gamma = 0.217
            beta = 1.284
            
            approx = self.test_on_iqm(gamma, beta)
            
            if approx > 0.5:
                print("\n🎉 SUCCESS! >50% on real hardware!")
            
            # Cost estimate
            cost = 256 * 0.00035 + 0.30
            print(f"\nActual cost: ${cost:.2f}")
        
        return True


def main():
    print("QAOA REALITY CHECK")
    print("="*70)
    print("\nTime to find out what REALLY works!")
    
    print("\n1. Test on Simulator")
    print("2. Test on IQM Emerald (~$1)")
    
    choice = input("\nChoice: ")
    
    if choice == '2':
        platform = 'iqm'
        print("\n⚠️  Will test on IQM Emerald (Garnet is offline)")
        confirm = input("Cost ~$1. Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    else:
        platform = 'simulator'
    
    checker = QAOARealityCheck(platform)
    checker.run()


if __name__ == "__main__":
    main()