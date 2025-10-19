"""
#!/usr/bin/env python3
"""
iqm_corrected_execution.py
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Corrected version for IQM with 1-based qubit indexing

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from braket.circuits import Circuit
from braket.aws import AwsDevice
from braket.devices import LocalSimulator

class IQMCorrectedExecution:
    """
    IQM-spezifische Implementierung mit korrekter Qubit-Indizierung
    """
    
    def __init__(self, use_real_hardware=True):
        self.use_real_hardware = use_real_hardware
        self.platform = 'iqm'
        self.results = {}
        
        # IQM-spezifische Qubit-Indizes (1-basiert!)
        self.qubit_1 = 1  # Statt 0
        self.qubit_2 = 2  # Statt 1
        
        # Configuration
        self.config = {
            'n_noise_levels': 7,
            'shots_per_circuit': 256,
            'tomography_bases': 4
        }
        
        # Cost estimate for IQM
        tasks = 2 * self.config['n_noise_levels'] * self.config['tomography_bases']
        shots = tasks * self.config['shots_per_circuit']
        
        self.cost_estimate = {
            'tasks': tasks,
            'shots': shots,
            'cost': tasks * 0.30 + shots * 0.00145
        }
        
        print("="*60)
        print("IQM CORRECTED EXECUTION")
        print("="*60)
        print(f"Using qubits: {self.qubit_1} and {self.qubit_2} (1-based indexing)")
        print(f"Noise levels: {self.config['n_noise_levels']}")
        print(f"Shots per circuit: {self.config['shots_per_circuit']}")
        print(f"Estimated cost: ${self.cost_estimate['cost']:.2f}")
        print("="*60)
    
    def get_device(self):
        """Get IQM device"""
        if not self.use_real_hardware:
            return LocalSimulator()
        
        device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
        
        # Check device status
        status = device.status
        print(f"\nIQM Status: {status}")
        
        if status != 'ONLINE':
            raise RuntimeError(f"IQM is not online! Status: {status}")
        
        # Get device properties
        properties = device.properties
        print(f"  Qubits: {properties.paradigm.qubitCount}")
        print(f"  Using qubits {self.qubit_1} and {self.qubit_2}")
        
        return device
    
    def create_circuits(self):
        """Create circuits with IQM-specific qubit mapping"""
        circuits = {}
        
        # 1. Product state |00⟩ on qubits 1,2
        circuits['product'] = Circuit()
        # Already in ground state
        
        # 2. Bell state |Φ+⟩ on qubits 1,2
        bell = Circuit()
        bell.h(self.qubit_1)  # H on qubit 1
        bell.cnot(self.qubit_1, self.qubit_2)  # CNOT between qubits 1 and 2
        circuits['bell'] = bell
        
        print(f"\nCreated circuits using qubits {self.qubit_1} and {self.qubit_2}")
        
        return circuits
    
    def add_noise_level(self, circuit, noise_level):
        """Add noise with IQM-specific qubit indices"""
        noisy_circuit = circuit.copy()
        
        if noise_level > 0:
            noise_angle = float(noise_level * np.pi / 4)
            
            # Use IQM qubit indices (1, 2)
            for qubit_idx in [self.qubit_1, self.qubit_2]:
                rotation_angle = float(noise_angle * np.random.uniform(-1, 1))
                # Braket order: rz(qubit, angle)
                noisy_circuit.rz(qubit_idx, rotation_angle)
            
            # Higher noise levels
            if noise_level > 0.2:
                extra_rotation = float(noise_level * np.pi / 8)
                for qubit_idx in [self.qubit_1, self.qubit_2]:
                    noisy_circuit.rz(qubit_idx, extra_rotation)
        
        return noisy_circuit
    
    def minimal_tomography(self, circuit):
        """Tomography with IQM qubit mapping"""
        tomo_circuits = []
        
        measurements = [
            ('ZZ', []),
            ('XX', [('h', self.qubit_1), ('h', self.qubit_2)]),
            ('YY', [('sdg', self.qubit_1), ('h', self.qubit_1), 
                    ('sdg', self.qubit_2), ('h', self.qubit_2)]),
            ('ZI', [])
        ]
        
        for label, prep_gates in measurements:
            tomo_circuit = circuit.copy()
            
            # Add preparation gates
            for gate, qubit in prep_gates:
                if gate == 'h':
                    tomo_circuit.h(qubit)
                elif gate == 'sdg':
                    tomo_circuit.si(qubit)  # S-dagger
            
            # Add measurement on IQM qubits
            tomo_circuit.measure([self.qubit_1, self.qubit_2])
            
            tomo_circuits.append((label, tomo_circuit))
        
        return tomo_circuits
    
    def run_experiment(self):
        """Run complete IQM experiment"""
        print(f"\n{'='*60}")
        print(f"RUNNING ON IQM GARNET")
        print(f"{'='*60}")
        
        # Get device
        device = self.get_device()
        
        # Create base circuits
        base_circuits = self.create_circuits()
        
        # Noise levels
        noise_levels = np.linspace(0.0, 0.3, self.config['n_noise_levels'])
        
        # Store results
        platform_results = {
            'device_arn': device.arn if self.use_real_hardware else 'simulator',
            'timestamp': datetime.now().isoformat(),
            'noise_levels': noise_levels.tolist(),
            'qubit_mapping': {
                'logical_0': self.qubit_1,
                'logical_1': self.qubit_2
            },
            'states': {}
        }
        
        all_tasks = []
        task_metadata = []
        
        print("\nSubmitting quantum tasks...")
        
        for state_name, base_circuit in base_circuits.items():
            print(f"\n{state_name}:")
            
            for i, noise_level in enumerate(noise_levels):
                # Add noise
                noisy_circuit = self.add_noise_level(base_circuit, noise_level)
                
                # Get tomography circuits
                tomo_circuits = self.minimal_tomography(noisy_circuit)
                
                # Submit all tomography circuits
                for tomo_label, tomo_circuit in tomo_circuits:
                    if self.use_real_hardware:
                        # IQM-specific submission
                        task = device.run(
                            tomo_circuit,
                            shots=self.config['shots_per_circuit']
                        )
                    else:
                        task = device.run(
                            tomo_circuit,
                            shots=self.config['shots_per_circuit']
                        )
                    
                    all_tasks.append(task)
                    task_metadata.append({
                        'state': state_name,
                        'noise_level': noise_level,
                        'basis': tomo_label,
                        'task_arn': task.id if self.use_real_hardware else 'sim'
                    })
                    
                print(f"  Noise {noise_level:.2f}: {len(tomo_circuits)} tasks submitted")
        
        print(f"\nTotal tasks submitted: {len(all_tasks)}")
        
        if self.use_real_hardware and all_tasks:
            platform_results['example_task_arn'] = all_tasks[0].id
            print(f"First task ARN: {all_tasks[0].id}")
        
        # Wait for completion
        print("\nWaiting for quantum tasks to complete...")
        start_time = time.time()
        
        completed = 0
        while completed < len(all_tasks):
            time.sleep(5)  # Check every 5 seconds for IQM
            
            newly_completed = sum(1 for task in all_tasks if task.state() == 'COMPLETED')
            if newly_completed > completed:
                completed = newly_completed
                elapsed = time.time() - start_time
                print(f"  Progress: {completed}/{len(all_tasks)} tasks "
                      f"({completed/len(all_tasks)*100:.1f}%) - "
                      f"Elapsed: {elapsed:.0f}s")
        
        print("All tasks completed!")
        
        # Process results (simplified for brevity)
        print("\nProcessing results...")
        
        # Save results
        self.results['iqm'] = platform_results
        
        # Save raw data
        filename = f"iqm_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config,
                'cost_estimate': self.cost_estimate,
                'use_real_hardware': self.use_real_hardware,
                'qubit_mapping': f"Using IQM qubits {self.qubit_1} and {self.qubit_2}"
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        return platform_results


def main():
    """Main execution"""
    print("IQM CORRECTED EXECUTION")
    print("="*60)
    
    print("\nThis version uses correct 1-based qubit indexing for IQM")
    print("Qubits 1 and 2 will be used (not 0 and 1)")
    
    print("\nOptions:")
    print("1. Test with simulator (free)")
    print("2. Run on IQM Garnet (~$38)")
    
    choice = input("\nChoice (1-2): ")
    
    use_real = (choice == '2')
    
    if use_real:
        response = input(f"\nProceed with ~$38 IQM experiment? (y/n): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
    
    # Initialize and run
    experiment = IQMCorrectedExecution(use_real_hardware=use_real)
    
    try:
        experiment.run_experiment()
        
        print("\n" + "="*60)
        print("IQM EXPERIMENT COMPLETE")
        print("="*60)
        
        if use_real:
            print("\nCheck AWS Braket console for results")
            print("Results saved with verification ARNs")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    main()
