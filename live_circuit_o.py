
#!/usr/bin/env python3
"""
HAWK: Hardware-Aware Quantum Workflow Kit
=============================================
Copyright (c) 2025 ForgottenForge.xyz

Live optimization on real quantum hardware using σ_c metrics.
Features:
- Automatic platform detection (Simulator/IQM Garnet)
- Hardware-adapted σ_c measurement
- Live strategy testing and selection
- Multi-algorithm support (Grover, VQE, QAOA)
- HEB integration for larger circuits
- Real-time cost tracking
- Comprehensive reporting

Dual Licensed under:
- Creative Commons Attribution 4.0 International (CC BY 4.0)
- Elastic License 2.0 (ELv2)

Commercial licensing available. Contact: nfo@forgottenforge.xyz
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# AWS Braket imports
from braket.circuits import Circuit
from braket.circuits.gates import CNot, H, Rx, Ry, Rz, S, T, X, Y, Z, CZ, CCNot
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.circuits.noises import Depolarizing


class Platform(Enum):
    """Available platforms"""
    SIMULATOR = "simulator"
    IQM_GARNET = "iqm_garnet"


@dataclass
class OptimizationResult:
    """Results from optimization round"""
    circuit: Circuit
    sigma_c: float
    success_rate: float
    gate_count: int
    strategy: str
    cost: float
    shots_used: int
    
    @property
    def efficiency_score(self) -> float:
        """Combined metric: performance per gate"""
        if self.gate_count == 0:
            return 0
        return (self.sigma_c * self.success_rate) / np.sqrt(self.gate_count)


class LiveHardwareOptimizer:
    """
    Live optimization on quantum hardware with adaptive strategies
    """
    
    def __init__(self, platform: Platform = Platform.SIMULATOR):
        self.platform = platform
        
        # Initialize device
        if platform == Platform.SIMULATOR:
            self.device = LocalSimulator("braket_dm")
            self.device_name = "AWS Simulator"
            self.cost_per_task = 0
            self.cost_per_shot = 0
        else:  # IQM Garnet
            self.device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")
            self.device_name = "IQM Garnet"
            self.cost_per_task = 0.30
            self.cost_per_shot = 0.00035  # IQM pricing
            
        # Optimization strategies from auto_opti2.py
        self.strategies = {
            'baseline': lambda c: c,  # No change
            'echo': self.add_echo_sequences,
            'dd': self.add_dynamical_decoupling,
            'virtual_z': self.virtualize_z_rotations,
            'decompose': self.decompose_complex_gates,
            'symmetrize': self.add_symmetrization,
            'cancel': self.cancel_redundant_gates,
            'heb_2': lambda c: self.apply_heb_structure(c, block_size=2),
            'heb_4': lambda c: self.apply_heb_structure(c, block_size=4),
        }
        
        # Tracking
        self.total_cost = 0
        self.total_shots = 0
        self.optimization_history = []
        
        print(f"Initialized {self.device_name}")
        print(f"Cost: ${self.cost_per_task}/task + ${self.cost_per_shot}/shot")
        print("="*60)
    
    def measure_hardware_sigma_c(self, circuit: Circuit, shots: int = 500) -> Tuple[float, float]:
        """
        Measure σ_c on hardware using circuit depth repetition
        Returns: (sigma_c, success_rate)
        """
        performances = []
        depths = [1, 2, 3, 4] if self.platform == Platform.IQM_GARNET else [1, 2, 4, 8]
        
        print(f"  Measuring σ_c with {len(depths)} depth points...")
        
        for depth in depths:
            # Create deeper circuit by repetition
            if depth == 1:
                test_circuit = circuit
            else:
                test_circuit = self.repeat_circuit_blocks(circuit, depth)
            
            # Run on device
            start_time = time.time()
            task = self.device.run(test_circuit, shots=shots)
            
            if self.platform == Platform.IQM_GARNET:
                # Wait for quantum task
                while task.state() not in ["COMPLETED", "FAILED", "CANCELLED"]:
                    time.sleep(2)
                    
            result = task.result()
            elapsed = time.time() - start_time
            
            # Calculate success metric
            success = self.calculate_circuit_success(result, circuit)
            performances.append(success)
            
            # Update costs
            self.total_shots += shots
            self.total_cost += self.cost_per_task + shots * self.cost_per_shot
            
            print(f"    Depth {depth}: success={success:.3f} (took {elapsed:.1f}s)")
        
        # Calculate σ_c from performance decay
        sigma_c = self.extract_sigma_c(depths, performances)
        baseline_success = performances[0]  # Success at depth 1
        
        return sigma_c, baseline_success
    
    def calculate_circuit_success(self, result, original_circuit) -> float:
        """Calculate success rate based on circuit type"""
        measurements = result.measurements
        if len(measurements) == 0:
            return 0
            
        counts = {}
        for m in measurements:
            key = tuple(m.tolist())
            counts[key] = counts.get(key, 0) + 1
            
        total = len(measurements)
        
        # Detect circuit type and calculate appropriate success metric
        num_qubits = len(measurements[0])
        
        # For Grover-like circuits (has H and CZ gates)
        has_h = any('H' in str(inst) for inst in original_circuit.instructions)
        has_cz = any('CZ' in str(inst) for inst in original_circuit.instructions)
        
        if has_h and has_cz:
            # Grover: expect |11...1⟩ state
            target = tuple([1] * num_qubits)
            success = counts.get(target, 0) / total
        else:
            # Generic: use entropy-based metric
            entropy = 0
            for count in counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            max_entropy = min(num_qubits, np.log2(total))
            success = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
        return success
    
    def extract_sigma_c(self, depths: List[int], performances: List[float]) -> float:
        """Extract critical noise threshold from depth-performance curve"""
        if len(performances) < 2 or performances[0] == 0:
            return 0.01
            
        # Normalize performances
        normalized = [p / performances[0] for p in performances]
        
        # Find where performance drops below 0.5
        for i, (d, p) in enumerate(zip(depths, normalized)):
            if p < 0.5:
                # Interpolate
                if i > 0:
                    d_prev = depths[i-1]
                    p_prev = normalized[i-1]
                    # Linear interpolation
                    slope = (p - p_prev) / (d - d_prev)
                    critical_depth = d_prev + (0.5 - p_prev) / slope
                    # Convert depth to effective noise parameter
                    return 1.0 / critical_depth
                else:
                    return 1.0 / d
                    
        # If no 50% crossing, estimate from slope
        if len(depths) > 1:
            slope = (normalized[-1] - normalized[0]) / (depths[-1] - depths[0])
            if slope < 0:
                return -normalized[0] / (slope * depths[0])
                
        return 0.3  # Default high resilience
    
    def repeat_circuit_blocks(self, circuit: Circuit, repetitions: int) -> Circuit:
        """Repeat circuit blocks to simulate increased noise"""
        repeated = Circuit()
        
        # For simple repetition
        if repetitions <= 2:
            for _ in range(repetitions):
                for inst in circuit.instructions:
                    repeated.add(inst)
        else:
            # For higher repetitions, add some variety
            for rep in range(repetitions):
                for inst in circuit.instructions:
                    repeated.add(inst)
                # Add small rotation between repetitions
                if rep < repetitions - 1:
                    qubits = self.get_circuit_qubits(circuit)
                    if qubits:
                        repeated.rz(qubits[0], 0.01)
                        
        return repeated
    
    def get_circuit_qubits(self, circuit: Circuit) -> List[int]:
        """Get all qubits used in circuit"""
        qubits = set()
        for inst in circuit.instructions:
            qubits.update(inst.target)
        return sorted(list(qubits))
    
    # Optimization strategies
    def add_echo_sequences(self, circuit: Circuit) -> Circuit:
        """Add echo sequences for error suppression"""
        new_circuit = Circuit()
        
        # Find two-qubit gates
        two_qubit_indices = []
        for i, inst in enumerate(circuit.instructions):
            if len(inst.target) == 2:
                two_qubit_indices.append(i)
                
        # Add original gates with echo around two-qubit gates
        for i, inst in enumerate(circuit.instructions):
            if i in two_qubit_indices:
                # Add echo before
                for q in inst.target:
                    new_circuit.x(q)
                    new_circuit.y(q)
                    new_circuit.y(q)
                    new_circuit.x(q)
                    
            new_circuit.add(inst)
            
        return new_circuit
    
    def add_dynamical_decoupling(self, circuit: Circuit) -> Circuit:
        """Insert DD sequences in idle periods"""
        new_circuit = Circuit()
        last_use = {}
        
        for i, inst in enumerate(circuit.instructions):
            current_qubits = set(inst.target)
            
            # Check for idle qubits
            for q in last_use:
                if q not in current_qubits and i - last_use[q] > 3:
                    # Insert simple DD
                    new_circuit.x(q)
                    new_circuit.x(q)
                    
            new_circuit.add(inst)
            
            # Update last use
            for q in current_qubits:
                last_use[q] = i
                
        return new_circuit
    
    def virtualize_z_rotations(self, circuit: Circuit) -> Circuit:
        """Convert Z rotations to virtual (frame updates)"""
        new_circuit = Circuit()
        phase_tracking = {}
        
        for inst in circuit.instructions:
            if 'RZ' in str(type(inst)):
                # Track phase instead
                for q in inst.target:
                    phase_tracking[q] = phase_tracking.get(q, 0) + 0.1  # Simplified
            else:
                # Apply accumulated phases before non-commuting gates
                if any(g in str(type(inst)) for g in ['X', 'Y', 'H']):
                    for q in inst.target:
                        if q in phase_tracking and phase_tracking[q] != 0:
                            new_circuit.rz(q, phase_tracking[q])
                            phase_tracking[q] = 0
                            
                new_circuit.add(inst)
                
        # Apply remaining phases
        for q, phase in phase_tracking.items():
            if phase != 0:
                new_circuit.rz(q, phase)
                
        return new_circuit
    
    def decompose_complex_gates(self, circuit: Circuit) -> Circuit:
        """Decompose multi-qubit gates into simpler ones"""
        new_circuit = Circuit()
        
        for inst in circuit.instructions:
            if 'CCNot' in str(type(inst)) or len(inst.target) > 2:
                # Decompose Toffoli
                if len(inst.target) == 3:
                    a, b, c = inst.target
                    # Standard Toffoli decomposition
                    new_circuit.h(c)
                    new_circuit.cnot(b, c)
                    new_circuit.t(c).adjoint()
                    new_circuit.cnot(a, c)
                    new_circuit.t(c)
                    new_circuit.cnot(b, c)
                    new_circuit.t(c).adjoint()
                    new_circuit.cnot(a, c)
                    new_circuit.t(b)
                    new_circuit.t(c)
                    new_circuit.h(c)
                    new_circuit.cnot(a, b)
                    new_circuit.t(a)
                    new_circuit.t(b).adjoint()
                    new_circuit.cnot(a, b)
                else:
                    new_circuit.add(inst)
            else:
                new_circuit.add(inst)
                
        return new_circuit
    
    def add_symmetrization(self, circuit: Circuit) -> Circuit:
        """Add symmetrization for noise resilience"""
        new_circuit = Circuit()
        
        # Add original circuit
        for inst in circuit.instructions:
            new_circuit.add(inst)
            
        # Count operations per qubit
        qubit_ops = {}
        for inst in circuit.instructions:
            for q in inst.target:
                qubit_ops[q] = qubit_ops.get(q, 0) + 1
                
        # Balance operations
        if qubit_ops:
            max_ops = max(qubit_ops.values())
            for q, ops in qubit_ops.items():
                if ops < max_ops:
                    # Add identity-like operations
                    for _ in range((max_ops - ops) // 2):
                        new_circuit.s(q)
                        new_circuit.s(q).adjoint()
                        
        return new_circuit
    
    def cancel_redundant_gates(self, circuit: Circuit) -> Circuit:
        """Remove redundant gate sequences"""
        instructions = list(circuit.instructions)
        new_circuit = Circuit()
        skip_next = False
        
        for i in range(len(instructions)):
            if skip_next:
                skip_next = False
                continue
                
            current = instructions[i]
            
            # Check for cancellations
            if i + 1 < len(instructions):
                next_inst = instructions[i + 1]
                
                # Same gate on same qubit cancels for X, Y, Z, H
                if (str(type(current)) == str(type(next_inst)) and 
                    current.target == next_inst.target and
                    any(g in str(type(current)) for g in ['X', 'Y', 'Z', 'H'])):
                    skip_next = True
                    continue
                    
            new_circuit.add(current)
            
        return new_circuit
    
    def apply_heb_structure(self, circuit: Circuit, block_size: int = 4) -> Circuit:
        """Apply HEB structuring to circuit"""
        qubits = self.get_circuit_qubits(circuit)
        n_qubits = len(qubits)
        
        if n_qubits <= block_size:
            return circuit  # Too small for HEB
            
        # Create new circuit with HEB structure
        heb_circuit = Circuit()
        n_blocks = (n_qubits + block_size - 1) // block_size
        
        # Initialize all qubits in superposition (if Grover-like)
        has_h_start = any(i < 5 and 'H' in str(inst) for i, inst in enumerate(circuit.instructions))
        if has_h_start:
            for q in qubits:
                heb_circuit.h(q)
                
        # Main operation: process in blocks
        # This is simplified - real HEB would analyze circuit structure
        block_groups = []
        for b in range(n_blocks):
            start_q = b * block_size
            end_q = min(start_q + block_size, n_qubits)
            block_groups.append(list(range(start_q, end_q)))
            
        # Apply operations within blocks
        for group in block_groups:
            # Apply subset of original operations
            for inst in circuit.instructions[len(qubits):]:  # Skip initial H gates
                # Check if instruction applies to this block
                if all(q in group for q in inst.target):
                    heb_circuit.add(inst)
                    
        # Add weak inter-block coupling
        if n_blocks > 1:
            for b in range(n_blocks - 1):
                last_q = block_groups[b][-1]
                first_q_next = block_groups[b + 1][0]
                
                # Weak coupling
                heb_circuit.ry(last_q, 0.1 * np.pi)
                heb_circuit.cnot(last_q, first_q_next)
                heb_circuit.ry(last_q, -0.1 * np.pi)
                
        return heb_circuit
    
    def optimize_circuit(self, circuit: Circuit, algorithm_name: str = "Unknown",
                        max_rounds: int = 5, shots_per_test: int = 500) -> OptimizationResult:
        """
        Main optimization loop
        """
        print(f"\nOPTIMIZING: {algorithm_name}")
        print("="*60)
        
        # Measure baseline
        print("\nMeasuring baseline performance...")
        baseline_sigma_c, baseline_success = self.measure_hardware_sigma_c(circuit, shots_per_test)
        baseline_gates = len(circuit.instructions)
        
        best_result = OptimizationResult(
            circuit=circuit,
            sigma_c=baseline_sigma_c,
            success_rate=baseline_success,
            gate_count=baseline_gates,
            strategy="baseline",
            cost=self.total_cost,
            shots_used=self.total_shots
        )
        
        print(f"\nBaseline: σ_c={baseline_sigma_c:.3f}, success={baseline_success:.3f}, gates={baseline_gates}")
        print(f"Current cost: ${self.total_cost:.2f}")
        
        # Optimization rounds
        for round_num in range(max_rounds):
            print(f"\n--- Round {round_num + 1}/{max_rounds} ---")
            
            round_results = []
            
            # Test each strategy
            for strategy_name, strategy_func in self.strategies.items():
                if strategy_name == 'baseline':
                    continue
                    
                print(f"\nTesting strategy: {strategy_name}")
                
                try:
                    # Apply strategy
                    modified_circuit = strategy_func(circuit)
                    
                    # Quick validation
                    if len(modified_circuit.instructions) == 0:
                        print("  Strategy produced empty circuit, skipping")
                        continue
                        
                    # Measure performance
                    shots = shots_per_test // 2 if round_num > 0 else shots_per_test
                    sigma_c, success_rate = self.measure_hardware_sigma_c(modified_circuit, shots)
                    
                    result = OptimizationResult(
                        circuit=modified_circuit,
                        sigma_c=sigma_c,
                        success_rate=success_rate,
                        gate_count=len(modified_circuit.instructions),
                        strategy=strategy_name,
                        cost=self.total_cost,
                        shots_used=self.total_shots
                    )
                    
                    round_results.append(result)
                    
                    print(f"  Result: σ_c={sigma_c:.3f}, success={success_rate:.3f}, "
                          f"gates={result.gate_count}, efficiency={result.efficiency_score:.3f}")
                    
                except Exception as e:
                    print(f"  Strategy failed: {str(e)}")
                    continue
                    
            # Select best result from round
            if round_results:
                # Sort by efficiency score
                round_results.sort(key=lambda r: r.efficiency_score, reverse=True)
                best_round = round_results[0]
                
                # Check if it's better than current best
                if best_round.efficiency_score > best_result.efficiency_score * 1.05:  # 5% improvement threshold
                    print(f"\n✓ Improvement found: {best_round.strategy}")
                    print(f"  Efficiency: {best_result.efficiency_score:.3f} → {best_round.efficiency_score:.3f}")
                    best_result = best_round
                    circuit = best_round.circuit  # Use for next round
                else:
                    print(f"\nNo significant improvement this round")
                    if round_num > 1:  # Early stopping
                        print("Early stopping - no recent improvements")
                        break
            
            print(f"\nCumulative cost: ${self.total_cost:.2f}")
            
        # Final summary
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best strategy: {best_result.strategy}")
        print(f"Final σ_c: {baseline_sigma_c:.3f} → {best_result.sigma_c:.3f} "
              f"({(best_result.sigma_c/baseline_sigma_c - 1)*100:+.1f}%)")
        print(f"Final success rate: {baseline_success:.3f} → {best_result.success_rate:.3f} "
              f"({(best_result.success_rate/baseline_success - 1)*100:+.1f}%)")
        print(f"Gate count: {baseline_gates} → {best_result.gate_count} "
              f"({(1 - best_result.gate_count/baseline_gates)*100:.1f}% reduction)")
        print(f"Efficiency score: {best_result.efficiency_score:.3f}")
        print(f"\nTotal cost: ${self.total_cost:.2f}")
        print(f"Total shots: {self.total_shots}")
        
        self.optimization_history.append(best_result)
        
        return best_result
    
    def create_test_circuits(self) -> Dict[str, Circuit]:
        """Create test circuits for optimization"""
        circuits = {}
        
        # 1. Simple 2-qubit Grover
        grover_2q = Circuit()
        grover_2q.h(0)
        grover_2q.h(1)
        grover_2q.cz(0, 1)
        grover_2q.h(0)
        grover_2q.h(1)
        grover_2q.x(0)
        grover_2q.x(1)
        grover_2q.cz(0, 1)
        grover_2q.x(0)
        grover_2q.x(1)
        grover_2q.h(0)
        grover_2q.h(1)
        circuits["Grover-2Q"] = grover_2q
        
        # 2. 3-qubit Grover (more complex)
        grover_3q = Circuit()
        for i in range(3):
            grover_3q.h(i)
        # Oracle for |111>
        grover_3q.ccnot(0, 1, 2)
        grover_3q.cz(0, 2)
        # Diffusion
        for i in range(3):
            grover_3q.h(i)
            grover_3q.x(i)
        grover_3q.ccnot(0, 1, 2)
        for i in range(3):
            grover_3q.x(i)
            grover_3q.h(i)
        circuits["Grover-3Q"] = grover_3q
        
        # 3. VQE-style circuit
        vqe = Circuit()
        vqe.ry(0, np.pi/4)
        vqe.ry(1, np.pi/3)
        vqe.cnot(0, 1)
        vqe.rz(1, np.pi/2)
        vqe.ry(0, np.pi/6)
        vqe.ry(1, np.pi/5)
        circuits["VQE-2Q"] = vqe
        
        # 4. QAOA-style circuit
        qaoa = Circuit()
        qaoa.h(0)
        qaoa.h(1)
        qaoa.cnot(0, 1)
        qaoa.rz(1, 0.5)
        qaoa.cnot(0, 1)
        qaoa.rx(0, 0.7)
        qaoa.rx(1, 0.7)
        circuits["QAOA-2Q"] = qaoa
        
        return circuits
    
    def run_full_optimization_suite(self, custom_circuits: Optional[Dict[str, Circuit]] = None):
        """Run optimization on multiple circuits"""
        print(f"\nFULL OPTIMIZATION SUITE - {self.device_name}")
        print("="*70)
        
        # Use custom circuits or defaults
        if custom_circuits:
            circuits = custom_circuits
        else:
            circuits = self.create_test_circuits()
            
        print(f"Testing {len(circuits)} circuits")
        print(f"Platform: {self.platform.value}")
        
        # Adjust parameters based on platform
        if self.platform == Platform.IQM_GARNET:
            max_rounds = 3
            shots_per_test = 200
            print(f"Hardware mode: {max_rounds} rounds, {shots_per_test} shots/test")
        else:
            max_rounds = 5
            shots_per_test = 500
            print(f"Simulator mode: {max_rounds} rounds, {shots_per_test} shots/test")
            
        # Estimate cost
        if self.platform == Platform.IQM_GARNET:
            strategies_to_test = len(self.strategies) - 1  # Exclude baseline
            estimated_tasks = len(circuits) * max_rounds * strategies_to_test
            estimated_shots = estimated_tasks * shots_per_test
            estimated_cost = (estimated_tasks * self.cost_per_task + 
                            estimated_shots * self.cost_per_shot)
            
            print(f"\nEstimated cost: ${estimated_cost:.2f}")
            confirm = input("Proceed? (y/n): ")
            if confirm.lower() != 'y':
                print("Aborted")
                return
                
        results = {}
        
        # Optimize each circuit
        for name, circuit in circuits.items():
            result = self.optimize_circuit(
                circuit, 
                algorithm_name=name,
                max_rounds=max_rounds,
                shots_per_test=shots_per_test
            )
            results[name] = result
            
        # Create comprehensive report
        self.create_optimization_report(results)
        
        return results
    
    def create_optimization_report(self, results: Dict[str, OptimizationResult]):
        """Create detailed optimization report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_data = {
            'platform': self.platform.value,
            'device': self.device_name,
            'timestamp': timestamp,
            'total_cost': self.total_cost,
            'total_shots': self.total_shots,
            'results': {}
        }
        
        for name, result in results.items():
            json_data['results'][name] = {
                'sigma_c': result.sigma_c,
                'success_rate': result.success_rate,
                'gate_count': result.gate_count,
                'strategy': result.strategy,
                'efficiency_score': result.efficiency_score,
                'improvement': {
                    'sigma_c_percent': 0,  # Will calculate
                    'success_rate_percent': 0,
                    'gate_reduction_percent': 0
                }
            }
            
        filename = f"optimization_results_{self.platform.value}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to {filename}")
        
        # Create visualization
        self.plot_optimization_results(results)
        
        # Print summary table
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"{'Circuit':<15} {'Strategy':<12} {'σ_c':<8} {'Success':<8} {'Gates':<8} {'Efficiency':<10}")
        print("-"*80)
        
        for name, result in results.items():
            print(f"{name:<15} {result.strategy:<12} {result.sigma_c:<8.3f} "
                  f"{result.success_rate:<8.3f} {result.gate_count:<8} {result.efficiency_score:<10.3f}")
                  
        print("-"*80)
        print(f"Total cost: ${self.total_cost:.2f}")
        
    def plot_optimization_results(self, results: Dict[str, OptimizationResult]):
        """Visualize optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Live Hardware Optimization Results - {self.device_name}', fontsize=16)
        
        names = list(results.keys())
        
        # 1. σ_c comparison
        ax1 = axes[0, 0]
        sigma_c_values = [results[n].sigma_c for n in names]
        bars1 = ax1.bar(names, sigma_c_values, color='blue', alpha=0.7)
        ax1.set_ylabel('Critical Noise Threshold (σ_c)')
        ax1.set_title('Noise Resilience After Optimization')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add strategy labels
        for i, (name, bar) in enumerate(zip(names, bars1)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{results[name].strategy}',
                    ha='center', va='bottom', fontsize=8)
        
        # 2. Success rate comparison
        ax2 = axes[0, 1]
        success_values = [results[n].success_rate for n in names]
        ax2.bar(names, success_values, color='green', alpha=0.7)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Algorithm Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # 3. Gate count reduction
        ax3 = axes[1, 0]
        gate_counts = [results[n].gate_count for n in names]
        ax3.bar(names, gate_counts, color='red', alpha=0.7)
        ax3.set_ylabel('Gate Count')
        ax3.set_title('Circuit Complexity')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Efficiency scores
        ax4 = axes[1, 1]
        efficiency_scores = [results[n].efficiency_score for n in names]
        ax4.bar(names, efficiency_scores, color='purple', alpha=0.7)
        ax4.set_ylabel('Efficiency Score')
        ax4.set_title('Overall Optimization Efficiency')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_plot_{self.platform.value}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.show()


def main():
    """Main execution function"""
    print("HAWK: Hardware-Aware Quantum Workflow Kit")
    print("============================================\n")
    
    print("Select platform:")
    print("1. AWS Simulator (free, fast)")
    print("2. IQM Garnet (real quantum hardware, ~$15-30)")
    
    choice = input("\nChoice (1-2): ")
    
    if choice == "1":
        platform = Platform.SIMULATOR
    elif choice == "2":
        platform = Platform.IQM_GARNET
        print("\n⚠️  WARNING: Real hardware will cost money!")
        print("Estimated cost: $15-30 depending on circuit complexity")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted")
            return
    else:
        print("Invalid choice")
        return
    
    # Initialize optimizer
    optimizer = LiveHardwareOptimizer(platform)
    
    # Option to use custom circuits
    print("\nCircuit selection:")
    print("1. Use default test circuits (Grover, VQE, QAOA)")
    print("2. Test only Grover variants")
    print("3. Add custom circuit")
    
    circuit_choice = input("\nChoice (1-3): ")
    
    if circuit_choice == "2":
        # Grover variants only
        circuits = {
            "Grover-2Q": optimizer.create_test_circuits()["Grover-2Q"],
            "Grover-3Q": optimizer.create_test_circuits()["Grover-3Q"]
        }
        optimizer.run_full_optimization_suite(circuits)
    elif circuit_choice == "3":
        # Add custom circuit
        print("\nCustom circuit not implemented in this demo")
        print("Using default circuits instead")
        optimizer.run_full_optimization_suite()
    else:
        # Default: all circuits
        optimizer.run_full_optimization_suite()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print(f"Total cost: ${optimizer.total_cost:.2f}")
    print(f"Total shots: {optimizer.total_shots}")
    print("="*60)


if __name__ == "__main__":
    main()