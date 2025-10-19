"""
#!/usr/bin/env python3
"""
Quantum Circuit Auto-Optimizer v2.0 with Balanced Optimization
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

with structure preservation and functionality validation!

Key improvements:
1. Preserves algorithm structure
2. Validates functionality after each change
3. Multi-objective optimization (σ_c + performance)
4. Safe transformations only

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
from braket.circuits import Circuit
from braket.circuits.gates import H, CZ, CNot, X, Y, Z, Rx, Ry, Rz, S, T
from braket.devices import LocalSimulator
from braket.circuits.noises import Depolarizing
import copy
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Store detailed optimization results"""
    circuit: Circuit
    sigma_c: float
    performance: float
    gate_count: int
    composite_score: float
    is_functional: bool
    chi_squared: float
    strategy: str


class BalancedQuantumOptimizer:
    """
    Balanced optimizer that preserves functionality while improving resilience
    """
    
    def __init__(self, target_algorithm: Circuit, 
                 algorithm_type: str = "grover",
                 performance_weight: float = 0.5,
                 resilience_weight: float = 0.5):
        """
        Initialize with balance parameters
        
        Args:
            target_algorithm: The circuit to optimize
            algorithm_type: Type of algorithm (grover, vqe, qaoa, etc.)
            performance_weight: Weight for algorithm performance (0-1)
            resilience_weight: Weight for noise resilience (0-1)
        """
        self.original_circuit = target_algorithm
        self.algorithm_type = algorithm_type
        self.performance_weight = performance_weight
        self.resilience_weight = resilience_weight
        self.device = LocalSimulator("braket_dm")
        
        # Algorithm-specific constraints
        self.constraints = {
            'grover': {
                'preserve_order': ['oracle', 'diffusion'],
                'min_performance': 0.5,  # Must be >50% better than random
                'critical_gates': ['cz', 'h-x-cz-x-h']  # Pattern to preserve
            },
            'vqe': {
                'preserve_order': ['ansatz', 'measurement'],
                'min_performance': 0.3,
                'critical_gates': ['ry', 'cnot']
            },
            'generic': {
                'preserve_order': [],
                'min_performance': 0.25,
                'critical_gates': []
            }
        }
        
        # Safe optimization strategies
        self.safe_strategies = {
            'add_echo': self.add_echo_pulses,
            'virtual_z': self.virtualize_z_gates,
            'decompose': self.decompose_complex_gates,
            'dd_insert': self.insert_dynamical_decoupling,
            'symmetrize': self.careful_symmetrization,
            'error_mitigation': self.add_error_mitigation,
            'phase_optimization': self.optimize_phases,
            'gate_cancellation': self.cancel_redundant_gates
        }
        
        # Tracking
        self.optimization_history = []
        self.best_result = None
        
    def identify_circuit_structure(self, circuit: Circuit) -> Dict:
        """
        Identify the structure of the circuit (oracle, diffusion, etc.)
        """
        structure = {
            'sections': [],
            'gate_patterns': [],
            'critical_indices': []
        }
        
        if self.algorithm_type == 'grover':
            # Identify Grover components
            instructions = list(circuit.instructions)
            
            # Find oracle (usually CZ in middle)
            for i, inst in enumerate(instructions):
                if type(inst).__name__ == 'CZ' or 'CZ' in str(inst):
                    # First CZ is likely oracle
                    structure['sections'].append(('oracle', i, i))
                    structure['critical_indices'].append(i)
                    break
            
            # Find diffusion operator (H-X-CZ-X-H pattern)
            # This is simplified - real implementation would be more sophisticated
            
        return structure
    
    def measure_functionality(self, circuit: Circuit, noise_level: float = 0.01) -> Dict:
        """
        Comprehensive functionality measurement
        """
        # Add small noise to test robustness
        noisy_circuit = Circuit()
        for inst in circuit.instructions:
            noisy_circuit.add(inst)
            if noise_level > 0:
                for q in inst.target:
                    noisy_circuit.depolarizing(q, noise_level)
        
        # Run circuit
        result = self.device.run(noisy_circuit, shots=1000).result()
        measurements = result.measurements
        
        if len(measurements) == 0:
            return {'performance': 0, 'chi_squared': 0, 'is_functional': False}
        
        # Algorithm-specific performance metrics
        if self.algorithm_type == 'grover':
            # For Grover: success = finding marked state
            # Assuming 2 qubits, marked state is |11⟩
            success_count = sum(1 for m in measurements if all(m))
            performance = success_count / len(measurements)
            
            # Check if significantly better than random
            expected_random = 0.25
            chi_squared = ((success_count - 250)**2 / 250 + 
                          ((1000 - success_count) - 750)**2 / 750)
            
            is_functional = (performance > self.constraints.get(
                self.algorithm_type, self.constraints['generic'])['min_performance'])
            
        else:
            # Generic performance metric
            # Measure deviation from uniform distribution
            outcomes = {}
            for m in measurements:
                key = tuple(m)
                outcomes[key] = outcomes.get(key, 0) + 1
            
            n_outcomes = len(outcomes)
            expected = len(measurements) / (2 ** len(measurements[0]))
            
            chi_squared = sum((count - expected)**2 / expected 
                            for count in outcomes.values())
            
            performance = 1.0 - (1.0 / (1.0 + chi_squared / 100))
            is_functional = chi_squared > 7.815  # 95% confidence
        
        return {
            'performance': performance,
            'chi_squared': chi_squared,
            'is_functional': is_functional
        }
    
    def measure_sigma_c_accurate(self, circuit: Circuit) -> float:
        """
        Accurate σ_c measurement with multiple noise levels
        """
        noise_levels = np.linspace(0, 0.3, 20)
        performances = []
        
        for noise in noise_levels:
            # Create noisy circuit
            noisy_circuit = Circuit()
            for inst in circuit.instructions:
                noisy_circuit.add(inst)
                if noise > 0:
                    for q in inst.target:
                        noisy_circuit.depolarizing(q, noise)
            
            # Measure performance
            result = self.device.run(noisy_circuit, shots=100).result()
            measurements = result.measurements
            
            # Calculate success metric
            if self.algorithm_type == 'grover' and len(measurements) > 0:
                success = sum(1 for m in measurements if all(m)) / len(measurements)
            else:
                # Generic: information preservation
                success = self.calculate_information_functional(measurements)
            
            performances.append(success)
        
        # Find critical point (50% performance drop)
        performances = np.array(performances)
        if performances[0] > 0:
            half_performance = performances[0] / 2
            idx = np.where(performances < half_performance)[0]
            if len(idx) > 0:
                sigma_c = noise_levels[idx[0]]
            else:
                sigma_c = noise_levels[-1]
        else:
            sigma_c = 0.01
        
        return sigma_c
    
    def calculate_information_functional(self, measurements) -> float:
        """Calculate information preservation"""
        if measurements is None or len(measurements) == 0:
            return 0
        
        measurements = np.array(measurements)
        
        # Shannon entropy
        outcomes = {}
        for m in measurements:
            key = tuple(map(int, m))
            outcomes[key] = outcomes.get(key, 0) + 1
        
        total = len(measurements)
        entropy = 0
        for count in outcomes.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize
        n_qubits = measurements.shape[1] if measurements.ndim > 1 else 1
        max_entropy = n_qubits
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def calculate_composite_score(self, performance: float, sigma_c: float, 
                                gate_count: int, original_gates: int) -> float:
        """
        Calculate balanced composite score
        """
        # Normalize metrics
        perf_normalized = performance  # Already 0-1
        sigma_c_normalized = min(sigma_c / 0.3, 1.0)  # Normalize to 0.3 max
        efficiency_normalized = 1.0 - (gate_count / (original_gates * 2))  # Prefer fewer gates
        
        # Apply weights
        score = (self.performance_weight * perf_normalized + 
                self.resilience_weight * sigma_c_normalized + 
                0.2 * efficiency_normalized)  # Small bonus for efficiency
        
        return score
    
    def add_echo_pulses(self, circuit: Circuit) -> Circuit:
        """
        Safe strategy: Add echo pulses around critical operations
        """
        new_circuit = Circuit()
        structure = self.identify_circuit_structure(circuit)
        
        for i, inst in enumerate(circuit.instructions):
            # Add echo before critical gates
            if i in structure['critical_indices'] and len(inst.target) > 1:
                # Simple XY echo
                for q in inst.target:
                    new_circuit.x(q)
                    new_circuit.y(q)
                    new_circuit.y(q)
                    new_circuit.x(q)
            
            # Add original instruction
            new_circuit.add(inst)
        
        return new_circuit
    
    def virtualize_z_gates(self, circuit: Circuit) -> Circuit:
        """
        Convert Z rotations to virtual (frame tracking)
        """
        new_circuit = Circuit()
        phase_tracking = {}
        
        for inst in circuit.instructions:
            gate_name = type(inst).__name__
            
            if 'Rz' in gate_name:
                # Track phase instead of applying
                for q in inst.target:
                    q_int = int(q)
                    # In real implementation, extract angle parameter
                    phase_tracking[q_int] = phase_tracking.get(q_int, 0) + np.pi/4
            else:
                # Apply accumulated phases before non-commuting gates
                if any(g in gate_name for g in ['X', 'Y', 'H']):
                    for q in inst.target:
                        q_int = int(q)
                        if q_int in phase_tracking and phase_tracking[q_int] != 0:
                            new_circuit.rz(q_int, phase_tracking[q_int])
                            phase_tracking[q_int] = 0
                
                new_circuit.add(inst)
        
        # Apply remaining phases
        for q, phase in phase_tracking.items():
            if phase != 0:
                new_circuit.rz(q, phase)
        
        return new_circuit
    
    def decompose_complex_gates(self, circuit: Circuit) -> Circuit:
        """
        Decompose complex gates into simpler, more resilient ones
        """
        # For now, just return original - would implement decomposition rules
        return circuit
    
    def insert_dynamical_decoupling(self, circuit: Circuit) -> Circuit:
        """
        Insert DD sequences in idle periods
        """
        new_circuit = Circuit()
        
        # Track qubit usage
        last_use = {}
        
        for i, inst in enumerate(circuit.instructions):
            # Check for idle qubits
            current_qubits = set(int(q) for q in inst.target)
            
            # Add DD for qubits that have been idle
            for q in last_use:
                if q not in current_qubits and i - last_use[q] > 2:
                    # Insert simple DD
                    new_circuit.x(q)
                    new_circuit.x(q)
            
            # Add instruction
            new_circuit.add(inst)
            
            # Update last use
            for q in current_qubits:
                last_use[q] = i
        
        return new_circuit
    
    def careful_symmetrization(self, circuit: Circuit) -> Circuit:
        """
        Add symmetry while preserving structure
        """
        new_circuit = Circuit()
        
        # Count operations per qubit
        qubit_ops = {}
        max_qubit = 0
        
        for inst in circuit.instructions:
            for q in inst.target:
                q_int = int(q)
                qubit_ops[q_int] = qubit_ops.get(q_int, 0) + 1
                max_qubit = max(max_qubit, q_int)
        
        # First, add original circuit
        for inst in circuit.instructions:
            new_circuit.add(inst)
        
        # Then add balancing operations at the end
        if qubit_ops:
            max_ops = max(qubit_ops.values())
            for q in range(max_qubit + 1):
                ops = qubit_ops.get(q, 0)
                if ops < max_ops:
                    # Add identity operations
                    for _ in range((max_ops - ops) // 2):
                        new_circuit.s(q)
                        new_circuit.si(q)  # S†S = I
        
        return new_circuit
    
    def add_error_mitigation(self, circuit: Circuit) -> Circuit:
        """
        Add simple error mitigation techniques
        """
        new_circuit = Circuit()
        
        # Pre-rotation for coherent errors
        n_qubits = max(int(q) for inst in circuit.instructions for q in inst.target) + 1
        
        for q in range(n_qubits):
            new_circuit.rz(q, np.pi/32)  # Small rotation
        
        # Original circuit
        for inst in circuit.instructions:
            new_circuit.add(inst)
        
        # Post-rotation to cancel
        for q in range(n_qubits):
            new_circuit.rz(q, -np.pi/32)
        
        return new_circuit
    
    def optimize_phases(self, circuit: Circuit) -> Circuit:
        """
        Optimize phase gates for noise resilience
        """
        # Simplified - would implement phase merging
        return self.virtualize_z_gates(circuit)
    
    def cancel_redundant_gates(self, circuit: Circuit) -> Circuit:
        """
        Cancel redundant gate pairs (XX=I, HH=I, etc.)
        """
        new_circuit = Circuit()
        instructions = list(circuit.instructions)
        skip_next = False
        
        for i in range(len(instructions)):
            if skip_next:
                skip_next = False
                continue
            
            current = instructions[i]
            
            # Check for cancellation
            if i + 1 < len(instructions):
                next_inst = instructions[i + 1]
                
                # Same gate on same qubit?
                if (type(current).__name__ == type(next_inst).__name__ and
                    current.target == next_inst.target and
                    type(current).__name__ in ['X', 'Y', 'Z', 'H']):
                    # Skip both (they cancel)
                    skip_next = True
                    continue
            
            new_circuit.add(current)
        
        return new_circuit
    
    def optimize_circuit_balanced(self, max_iterations: int = 10,
                                 target_sigma_c: float = 0.2,
                                 min_performance: float = 0.5,
                                 verbose: bool = True) -> Circuit:
        """
        Main optimization loop with balance
        """
        # Measure original circuit
        original_perf = self.measure_functionality(self.original_circuit)
        original_sigma_c = self.measure_sigma_c_accurate(self.original_circuit)
        original_gates = len(self.original_circuit.instructions)
        
        original_score = self.calculate_composite_score(
            original_perf['performance'],
            original_sigma_c,
            original_gates,
            original_gates
        )
        
        # Initialize best result
        self.best_result = OptimizationResult(
            circuit=self.original_circuit,
            sigma_c=original_sigma_c,
            performance=original_perf['performance'],
            gate_count=original_gates,
            composite_score=original_score,
            is_functional=original_perf['is_functional'],
            chi_squared=original_perf['chi_squared'],
            strategy='original'
        )
        
        if verbose:
            print("="*60)
            print("BALANCED QUANTUM OPTIMIZATION v2.0")
            print("="*60)
            print(f"Algorithm type: {self.algorithm_type}")
            print(f"Performance weight: {self.performance_weight}")
            print(f"Resilience weight: {self.resilience_weight}")
            print(f"\nOriginal circuit:")
            print(f"  Performance: {original_perf['performance']:.3f}")
            print(f"  σ_c: {original_sigma_c:.3f}")
            print(f"  Gates: {original_gates}")
            print(f"  Composite score: {original_score:.3f}")
            print(f"  Functional: {original_perf['is_functional']}")
            print("-"*60)
        
        current_circuit = copy.deepcopy(self.original_circuit)
        strategies_applied = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}:")
            
            iteration_results = []
            
            # Try each safe strategy
            for strategy_name, strategy_func in self.safe_strategies.items():
                try:
                    # Apply strategy
                    candidate_circuit = strategy_func(current_circuit)
                    
                    # Measure performance
                    perf_result = self.measure_functionality(candidate_circuit)
                    
                    # Skip if not functional
                    if not perf_result['is_functional']:
                        if verbose:
                            print(f"  {strategy_name}: Not functional (χ²={perf_result['chi_squared']:.1f})")
                        continue
                    
                    # Measure σ_c
                    sigma_c = self.measure_sigma_c_accurate(candidate_circuit)
                    gate_count = len(candidate_circuit.instructions)
                    
                    # Calculate composite score
                    composite_score = self.calculate_composite_score(
                        perf_result['performance'],
                        sigma_c,
                        gate_count,
                        original_gates
                    )
                    
                    result = OptimizationResult(
                        circuit=candidate_circuit,
                        sigma_c=sigma_c,
                        performance=perf_result['performance'],
                        gate_count=gate_count,
                        composite_score=composite_score,
                        is_functional=perf_result['is_functional'],
                        chi_squared=perf_result['chi_squared'],
                        strategy=strategy_name
                    )
                    
                    iteration_results.append(result)
                    
                    if verbose:
                        print(f"  {strategy_name}: perf={perf_result['performance']:.3f}, "
                              f"σ_c={sigma_c:.3f}, score={composite_score:.3f}")
                    
                except Exception as e:
                    if verbose:
                        print(f"  {strategy_name}: Failed - {str(e)}")
            
            # Select best result from this iteration
            if iteration_results:
                best_iteration = max(iteration_results, key=lambda x: x.composite_score)
                
                # Only accept if better than current best
                if best_iteration.composite_score > self.best_result.composite_score:
                    self.best_result = best_iteration
                    current_circuit = best_iteration.circuit
                    strategies_applied.append(best_iteration.strategy)
                    
                    if verbose:
                        print(f"\n✓ Applied: {best_iteration.strategy}")
                        print(f"  New best score: {best_iteration.composite_score:.3f}")
                        print(f"  Performance: {best_iteration.performance:.3f}")
                        print(f"  σ_c: {best_iteration.sigma_c:.3f}")
                    
                    # Record history
                    self.optimization_history.append(best_iteration)
                    
                    # Check if targets met
                    if (best_iteration.sigma_c >= target_sigma_c and 
                        best_iteration.performance >= min_performance):
                        if verbose:
                            print("\n🎉 Targets achieved!")
                        break
                else:
                    if verbose:
                        print("\n  No improvement found this iteration")
            else:
                if verbose:
                    print("\n  No functional candidates this iteration")
                    
            # Early stopping if no progress
            if len(self.optimization_history) > 3:
                recent_scores = [r.composite_score for r in self.optimization_history[-3:]]
                if max(recent_scores) - min(recent_scores) < 0.01:
                    if verbose:
                        print("\n  Converged (no recent improvement)")
                    break
        
        # Final report
        if verbose:
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Strategies applied: {strategies_applied}")
            print(f"\nOriginal:")
            print(f"  Performance: {original_perf['performance']:.3f}")
            print(f"  σ_c: {original_sigma_c:.3f}")
            print(f"  Score: {original_score:.3f}")
            print(f"\nOptimized:")
            print(f"  Performance: {self.best_result.performance:.3f}")
            print(f"  σ_c: {self.best_result.sigma_c:.3f}")
            print(f"  Score: {self.best_result.composite_score:.3f}")
            print(f"  Gates: {original_gates} → {self.best_result.gate_count}")
            
            # Calculate improvements
            perf_change = (self.best_result.performance / original_perf['performance'] - 1) * 100
            sigma_change = (self.best_result.sigma_c / original_sigma_c - 1) * 100
            
            print(f"\nImprovements:")
            print(f"  Performance: {perf_change:+.1f}%")
            print(f"  σ_c: {sigma_change:+.1f}%")
            print(f"  Maintained functionality: {self.best_result.is_functional}")
        
        return self.best_result.circuit
    
    def plot_optimization_journey(self):
        """
        Visualize the optimization process
        """
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = range(len(self.optimization_history))
        
        # Performance evolution
        ax = axes[0, 0]
        performances = [r.performance for r in self.optimization_history]
        ax.plot(iterations, performances, 'b-o', linewidth=2, markersize=8)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Min functional')
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Performance')
        ax.set_title('Algorithm Performance Evolution')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # σ_c evolution
        ax = axes[0, 1]
        sigma_cs = [r.sigma_c for r in self.optimization_history]
        ax.plot(iterations, sigma_cs, 'g-s', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('σ_c')
        ax.set_title('Noise Resilience Evolution')
        ax.grid(True, alpha=0.3)
        
        # Composite score
        ax = axes[1, 0]
        scores = [r.composite_score for r in self.optimization_history]
        ax.plot(iterations, scores, 'r-^', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Composite Score')
        ax.set_title('Overall Optimization Progress')
        ax.grid(True, alpha=0.3)
        
        # Strategy timeline
        ax = axes[1, 1]
        strategies = [r.strategy for r in self.optimization_history]
        unique_strategies = list(set(strategies))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
        
        y_positions = {s: i for i, s in enumerate(unique_strategies)}
        
        for i, (result, strategy) in enumerate(zip(self.optimization_history, strategies)):
            y = y_positions[strategy]
            color = colors[unique_strategies.index(strategy)]
            
            ax.scatter(i, y, c=[color], s=200, alpha=0.8)
            
            # Add performance indicator
            if result.performance > 0.7:
                ax.scatter(i, y, c='none', s=300, edgecolors='green', linewidths=2)
        
        ax.set_yticks(range(len(unique_strategies)))
        ax.set_yticklabels(unique_strategies)
        ax.set_xlabel('Optimization Step')
        ax.set_title('Applied Strategies')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f'Balanced Optimization Journey - {self.algorithm_type.upper()}', fontsize=14)
        plt.tight_layout()
        
        filename = f'balanced_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
        
        plt.show()


def demonstrate_balanced_optimization():
    """
    Demonstrate balanced optimization on Grover
    """
    print("BALANCED QUANTUM OPTIMIZATION DEMONSTRATION")
    print("==========================================\n")
    
    # Create Grover circuit
    grover = Circuit()
    
    # Superposition
    grover.h(0)
    grover.h(1)
    
    # Oracle
    grover.cz(0, 1)
    
    # Diffusion
    grover.h(0)
    grover.h(1)
    grover.x(0)
    grover.x(1)
    grover.cz(0, 1)
    grover.x(0)
    grover.x(1)
    grover.h(0)
    grover.h(1)
    
    print("Testing different weight configurations...\n")
    
    # Test different balance configurations
    configs = [
        (0.8, 0.2, "Performance-focused"),
        (0.5, 0.5, "Balanced"),
        (0.2, 0.8, "Resilience-focused")
    ]
    
    results = []
    
    for perf_weight, res_weight, name in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {name}")
        print(f"Performance weight: {perf_weight}, Resilience weight: {res_weight}")
        print('='*60)
        
        optimizer = BalancedQuantumOptimizer(
            grover,
            algorithm_type='grover',
            performance_weight=perf_weight,
            resilience_weight=res_weight
        )
        
        optimized = optimizer.optimize_circuit_balanced(
            max_iterations=5,
            target_sigma_c=0.15,
            min_performance=0.6,
            verbose=True
        )
        
        results.append((name, optimizer))
        
        # Plot journey
        optimizer.plot_optimization_journey()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON OF CONFIGURATIONS")
    print("="*60)
    
    for name, optimizer in results:
        best = optimizer.best_result
        print(f"\n{name}:")
        print(f"  Final performance: {best.performance:.3f}")
        print(f"  Final σ_c: {best.sigma_c:.3f}")
        print(f"  Final score: {best.composite_score:.3f}")
        print(f"  Functional: {best.is_functional}")
        print(f"  Strategy: {best.strategy}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. Balance is crucial - pure σ_c optimization kills functionality")
    print("2. Different weights lead to different optimal solutions")
    print("3. Structure-preserving transformations maintain functionality")
    print("4. Multi-objective optimization finds practical improvements")
    print("\n🎯 The sweet spot is often around 50/50 balance!")


if __name__ == "__main__":

    demonstrate_balanced_optimization()
