#!/usr/bin/env python3
"""
Quantum Solver
=============================================
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
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import sqrtm

try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    from braket.aws import AwsDevice
    BRAKET_AVAILABLE = True
except ImportError:
    print("⚠️  AWS Braket not installed")
    BRAKET_AVAILABLE = False


# ==================== COMPLETE STRATEGY LIBRARY ====================

class OptimizationStrategy(Enum):
    """ALL optimization strategies from your research"""
    # Basic strategies
    GATE_COMPRESSION = "gate_compression"
    VIRTUAL_Z_GATES = "virtual_z"
    
    # DD Sequences
    DD_XX = "dd_xx"
    DD_XY4 = "dd_xy4"
    DD_CPMG = "dd_cpmg"
    DD_UHRIG = "dd_uhrig"
    DD_CUSTOM = "dd_custom"
    
    # Echo sequences
    ECHO_SPIN = "echo_spin"
    ECHO_CARR_PURCELL = "echo_carr_purcell"
    
    # Advanced strategies
    HEB_ENCODING = "heb_encoding"
    CIRCUIT_REORDERING = "circuit_reordering"
    SYMMETRIZATION = "symmetrization"
    PULSE_OPTIMIZATION = "pulse_optimization"
    NOISE_ADAPTIVE = "noise_adaptive"
    TOPOLOGY_MAPPING = "topology_mapping"
    
    # The special ones
    VERMICULAR = "vermicular"
    MULTI_STAGE_PROTECTION = "multi_stage_protection"


@dataclass
class HardwareProfile:
    """Complete hardware characterization"""
    name: str
    
    # Basic specs
    n_qubits: int
    connectivity: Dict[int, List[int]]
    
    # Noise parameters
    t1_times: Dict[int, float]  # Per-qubit T1
    t2_times: Dict[int, float]  # Per-qubit T2
    gate_errors_1q: Dict[str, float]  # Per gate type
    gate_errors_2q: Dict[Tuple[int, int], float]  # Per qubit pair
    readout_errors: Dict[int, float]  # Per qubit
    
    # Gate times
    gate_times: Dict[str, float]
    
    # Special features
    native_gates: List[str]
    supports_mid_circuit_measurement: bool = False
    supports_parametric_gates: bool = True
    
    # Optimal strategies for this hardware
    recommended_strategies: List[OptimizationStrategy] = field(default_factory=list)


# ==================== CRITICAL THRESHOLD ANALYZER V2 ====================

class AdvancedCriticalThresholdAnalyzer:
    """
    Enhanced σ_c analyzer that implements the FULL methodology from your research
    """
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_circuit_complete(self, circuit: Circuit, 
                               hardware: Optional[HardwareProfile] = None) -> Dict[str, Any]:
        """
        Complete σ_c analysis including:
        - Gate-by-gate fragility
        - Idle period detection
        - Entanglement structure
        - Multi-stage vulnerability
        - Hardware-specific adjustments
        """
        n_qubits = circuit.qubit_count
        instructions = circuit.instructions
        
        # Initialize analysis
        analysis = {
            'n_qubits': n_qubits,
            'gate_count': len(instructions),
            'depth': circuit.depth,
            'sigma_c': 0.0,
            'critical_gates': [],
            'idle_periods': {},
            'entanglement_profile': [],
            'multi_stage_risk': 0.0,
            'hardware_specific': {},
            'recommendations': []
        }
        
        # 1. Gate-by-gate analysis
        gate_fragilities = []
        qubit_activity = [[] for _ in range(n_qubits)]
        entanglement_depth = 0
        
        for i, inst in enumerate(instructions):
            # Base fragility
            if inst.operator.name in ['cnot', 'cz', 'swap', 'ccnot']:
                fragility = 0.8
                entanglement_depth += 1
            elif inst.operator.name in ['rx', 'ry', 'rz']:
                fragility = 0.4
            elif inst.operator.name in ['h', 'x', 'y', 'z', 's', 't']:
                fragility = 0.2
            else:
                fragility = 0.5
            
            # Position-based adjustment (errors accumulate)
            position_factor = 1 + 0.3 * (i / len(instructions))
            
            # Entanglement-based adjustment
            if entanglement_depth > 0:
                fragility *= (1 + 0.1 * np.log(1 + entanglement_depth))
            
            # Hardware-specific adjustment
            if hardware:
                # Check gate errors
                if inst.operator.name in hardware.gate_errors_1q:
                    fragility *= (1 + hardware.gate_errors_1q[inst.operator.name] * 100)
                
                # Check qubit quality
                for q in inst.target:
                    if q in hardware.t2_times:
                        coherence_factor = min(1.0, 50e-6 / hardware.t2_times[q])
                        fragility *= (1 + coherence_factor)
            
            gate_fragilities.append(fragility)
            
            # Track qubit activity
            for q in inst.target:
                qubit_activity[q].append(i)
            
            # Mark critical gates
            if fragility > 0.7:
                analysis['critical_gates'].append(i)
        
        # 2. Idle period detection
        for q, activity in enumerate(qubit_activity):
            if len(activity) > 1:
                idle_periods = []
                for i in range(len(activity) - 1):
                    gap = activity[i+1] - activity[i] - 1
                    if gap > 2:  # Significant idle period
                        idle_periods.append((activity[i], activity[i+1], gap))
                
                if idle_periods:
                    analysis['idle_periods'][q] = idle_periods
        
        # 3. Multi-stage vulnerability analysis
        # Look for repeated patterns (like in multi-stage Grover)
        pattern_length = min(20, len(instructions) // 3)
        if len(instructions) > pattern_length * 2:
            # Check for repeating structures
            patterns_found = 0
            for start in range(len(instructions) - pattern_length * 2):
                pattern1 = [inst.operator.name for inst in instructions[start:start+pattern_length]]
                pattern2 = [inst.operator.name for inst in instructions[start+pattern_length:start+pattern_length*2]]
                
                similarity = sum(1 for a, b in zip(pattern1, pattern2) if a == b) / pattern_length
                if similarity > 0.8:
                    patterns_found += 1
            
            if patterns_found > 0:
                analysis['multi_stage_risk'] = min(1.0, patterns_found * 0.3)
                analysis['recommendations'].append("Multi-stage structure detected - use VERMICULAR optimization")
        
        # 4. Calculate overall σ_c using complete model
        base_sigma_c = 0.2  # Higher base than simple version
        
        # Gate-based penalties
        gate_penalty = sum(gate_fragilities) * 0.001
        depth_penalty = circuit.depth * 0.002
        
        # Idle period penalty
        idle_penalty = sum(len(periods) * 0.01 for periods in analysis['idle_periods'].values())
        
        # Multi-stage penalty
        stage_penalty = analysis['multi_stage_risk'] * 0.05
        
        # Hardware penalty
        hw_penalty = 0
        if hardware:
            avg_gate_error = np.mean(list(hardware.gate_errors_1q.values()) + 
                                    list(hardware.gate_errors_2q.values()))
            hw_penalty = avg_gate_error * 10
        
        # Final σ_c calculation
        sigma_c = base_sigma_c - gate_penalty - depth_penalty - idle_penalty - stage_penalty - hw_penalty
        analysis['sigma_c'] = max(0.001, sigma_c)
        
        # 5. Generate specific recommendations
        if analysis['idle_periods']:
            total_idle = sum(sum(gap for _, _, gap in periods) 
                           for periods in analysis['idle_periods'].values())
            if total_idle > len(instructions) * 0.2:
                analysis['recommendations'].append("Significant idle time - apply DD sequences")
        
        if len(analysis['critical_gates']) > len(instructions) * 0.3:
            analysis['recommendations'].append("Many critical gates - consider HEB encoding")
        
        if circuit.depth > 50:
            analysis['recommendations'].append("Deep circuit - apply circuit compression")
        
        if entanglement_depth > n_qubits * 3:
            analysis['recommendations'].append("High entanglement - use echo sequences")
        
        return analysis
    
    def find_optimal_dd_positions(self, circuit: Circuit, analysis: Dict[str, Any]) -> List[int]:
        """Find optimal positions for DD sequence insertion"""
        positions = []
        
        # Be SELECTIVE - only add DD at the most critical positions
        # This was the key insight from VERMICULAR!
        
        # Only use top 3 critical gates, not all
        critical = sorted(analysis['critical_gates'][:3])
        for gate_idx in critical:
            if gate_idx < len(circuit.instructions) - 1:
                positions.append(gate_idx + 1)
        
        # For multi-stage, add at stage boundaries ONLY
        if analysis['multi_stage_risk'] > 0.5:
            stage_size = len(circuit.instructions) // 3
            positions.extend([stage_size, stage_size * 2])
        
        # Maximum 5 DD positions total!
        return sorted(set(positions))[:5]


# ==================== BRUTE FORCE OPTIMIZER ====================

class BruteForceOptimizer:
    """
    Implements the brute-force optimization approach that found VERMICULAR
    Tests ALL combinations systematically!
    """
    
    def __init__(self, simulator=None):
        self.simulator = simulator or LocalSimulator()
        self.strategy_combinations = []
        self.test_results = []
        
    def generate_strategy_combinations(self, 
                                     available_strategies: List[OptimizationStrategy],
                                     max_combinations: int = 3) -> List[List[OptimizationStrategy]]:
        """Generate all possible strategy combinations"""
        combinations = []
        
        # Single strategies
        for strategy in available_strategies:
            combinations.append([strategy])
        
        # Combinations of 2
        if max_combinations >= 2:
            for combo in itertools.combinations(available_strategies, 2):
                combinations.append(list(combo))
        
        # Combinations of 3
        if max_combinations >= 3:
            for combo in itertools.combinations(available_strategies, 3):
                combinations.append(list(combo))
        
        return combinations
    
    def test_strategy_combination(self, 
                                circuit: Circuit,
                                strategies: List[OptimizationStrategy],
                                optimizer: 'CompleteQuantumOptimizer',
                                target_metric: str = 'success_rate') -> Dict[str, Any]:
        """Test a specific combination of strategies"""
        # Apply strategies
        optimized = circuit
        for strategy in strategies:
            optimized = optimizer.apply_single_strategy(optimized, strategy)
        
        # Test on simulator
        result = self.simulator.run(optimized, shots=100).result()
        
        # Calculate metrics
        measurements = result.measurements
        
        # Simple success metric (can be customized)
        success_rate = self._calculate_success_metric(measurements, target_metric)
        
        # Calculate improvement in σ_c
        analyzer = AdvancedCriticalThresholdAnalyzer()
        original_analysis = analyzer.analyze_circuit_complete(circuit)
        optimized_analysis = analyzer.analyze_circuit_complete(optimized)
        
        sigma_c_improvement = optimized_analysis['sigma_c'] / original_analysis['sigma_c']
        
        return {
            'strategies': strategies,
            'success_rate': success_rate,
            'sigma_c_improvement': sigma_c_improvement,
            'gate_count': len(optimized.instructions),
            'depth': optimized.depth,
            'score': success_rate * sigma_c_improvement  # Combined score
        }
    
    def find_best_optimization(self, 
                             circuit: Circuit,
                             available_strategies: List[OptimizationStrategy],
                             optimizer: 'CompleteQuantumOptimizer',
                             max_time: float = 60.0) -> Dict[str, Any]:
        """
        Dingeliding for best optimization strategy
        """
        print("\n🔨 Starting dingeliding...")
        
        # Generate combinations
        combinations = self.generate_strategy_combinations(available_strategies)
        print(f"   Testing {len(combinations)} strategy combinations")
        
        best_result = None
        best_score = 0
        start_time = time.time()
        
        for i, strategies in enumerate(combinations):
            if time.time() - start_time > max_time:
                print(f"   Time limit reached after {i} tests")
                break
            
            # Test this combination
            result = self.test_strategy_combination(circuit, strategies, optimizer)
            
            # Update best if better
            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                print(f"   New best: {[s.value for s in strategies]} (score: {best_score:.3f})")
        
        print(f"\n✅ Best combination found: {[s.value for s in best_result['strategies']]}")
        print(f"   Success rate: {best_result['success_rate']:.1%}")
        print(f"   σ_c improvement: {best_result['sigma_c_improvement']:.2f}x")
        
        return best_result
    
    def _calculate_success_metric(self, measurements, metric_type: str) -> float:
        """Calculate success metric from measurements"""
        if metric_type == 'success_rate':
            # For Grover-like: check if we found the marked item
            # Simplified: most common outcome
            counts = {}
            for m in measurements:
                key = tuple(m)
                counts[key] = counts.get(key, 0) + 1
            
            if counts:
                max_count = max(counts.values())
                return max_count / len(measurements)
            return 0.0
        
        elif metric_type == 'energy':
            # For VQE-like: prefer low energy (more |0⟩s)
            zeros = sum(1 for m in measurements for bit in m if bit == 0)
            return zeros / (len(measurements) * len(measurements[0]))
        
        return 0.5


# ==================== OPTIMIZER ====================

class CompleteQuantumOptimizer:
    """
    OPTIMIZER v.0.1
    """
    
    def __init__(self):
        self.analyzer = AdvancedCriticalThresholdAnalyzer()
        self.hardware_profiles = self._load_hardware_profiles()
        
    def apply_single_strategy(self, circuit: Circuit, 
                            strategy: OptimizationStrategy) -> Circuit:
        """Apply a single optimization strategy"""
        if strategy == OptimizationStrategy.DD_XX:
            return self._apply_dd_xx(circuit)
        elif strategy == OptimizationStrategy.DD_XY4:
            return self._apply_dd_xy4(circuit)
        elif strategy == OptimizationStrategy.DD_CPMG:
            return self._apply_dd_cpmg(circuit)
        elif strategy == OptimizationStrategy.HEB_ENCODING:
            return self._apply_heb_encoding(circuit)
        elif strategy == OptimizationStrategy.ECHO_SPIN:
            return self._apply_spin_echo(circuit)
        elif strategy == OptimizationStrategy.CIRCUIT_REORDERING:
            return self._reorder_circuit(circuit)
        elif strategy == OptimizationStrategy.SYMMETRIZATION:
            return self._apply_symmetrization(circuit)
        elif strategy == OptimizationStrategy.VERMICULAR:
            return self._apply_vermicular(circuit)
        elif strategy == OptimizationStrategy.MULTI_STAGE_PROTECTION:
            return self._apply_multi_stage_protection(circuit)
        elif strategy == OptimizationStrategy.GATE_COMPRESSION:
            return self._compress_gates(circuit)
        elif strategy == OptimizationStrategy.VIRTUAL_Z_GATES:
            return self._virtualize_z_gates(circuit)
        else:
            return circuit
    
    def _apply_dd_xx(self, circuit: Circuit) -> Circuit:
        """Apply XX dynamical decoupling"""
        analysis = self.analyzer.analyze_circuit_complete(circuit)
        positions = self.analyzer.find_optimal_dd_positions(circuit, analysis)
        
        new_circuit = Circuit()
        
        for i, inst in enumerate(circuit.instructions):
            new_circuit.add_instruction(inst)
            
            # Add DD only at selected positions, not everywhere
            if i in positions and i < len(circuit.instructions) - 1:
                # Get active qubits from current and next instruction
                current_qubits = set(inst.target)
                next_inst = circuit.instructions[i + 1] if i + 1 < len(circuit.instructions) else None
                next_qubits = set(next_inst.target) if next_inst else set()
                
                # Only add DD to truly idle qubits
                idle_qubits = set(range(circuit.qubit_count)) - current_qubits - next_qubits
                for q in idle_qubits:
                    new_circuit.x(q).x(q)
        
        return new_circuit

    def _apply_dd_xy4(self, circuit: Circuit) -> Circuit:
        """Apply XY4 dynamical decoupling (more robust than XX)"""
        analysis = self.analyzer.analyze_circuit_complete(circuit)
        positions = self.analyzer.find_optimal_dd_positions(circuit, analysis)
        
        new_circuit = Circuit()
        
        for i, inst in enumerate(circuit.instructions):
            new_circuit.add_instruction(inst)
            
            if i in positions and i < len(circuit.instructions) - 1:
                current_qubits = set(inst.target)
                next_inst = circuit.instructions[i + 1] if i + 1 < len(circuit.instructions) else None
                next_qubits = set(next_inst.target) if next_inst else set()
                
                idle_qubits = set(range(circuit.qubit_count)) - current_qubits - next_qubits
                for q in idle_qubits:
                    new_circuit.x(q).y(q).x(q).y(q)
        
        return new_circuit

    def _apply_dd_cpmg(self, circuit: Circuit) -> Circuit:
        """Apply Carr-Purcell-Meiboom-Gill sequence"""
        analysis = self.analyzer.analyze_circuit_complete(circuit)
        positions = self.analyzer.find_optimal_dd_positions(circuit, analysis)
        
        new_circuit = Circuit()
        
        for i, inst in enumerate(circuit.instructions):
            new_circuit.add_instruction(inst)
            
            if i in positions and i < len(circuit.instructions) - 1:
                current_qubits = set(inst.target)
                next_inst = circuit.instructions[i + 1] if i + 1 < len(circuit.instructions) else None
                next_qubits = set(next_inst.target) if next_inst else set()
                
                idle_qubits = set(range(circuit.qubit_count)) - current_qubits - next_qubits
                for q in idle_qubits:
                    new_circuit.y(q).x(q).y(q).x(q)
        
        return new_circuit
    
    def _apply_heb_encoding(self, circuit: Circuit) -> Circuit:
        """Apply Hierarchical Entangling Block structure (2-4-2 or similar)"""
        n_qubits = circuit.qubit_count
        
        if n_qubits < 4:
            return circuit  # Too small for HEB
        
        # Determine block structure
        if n_qubits <= 8:
            structure = [2, n_qubits-4, 2]  # 2-middle-2
        else:
            # Larger: 3-middle-3 or custom
            structure = [3, n_qubits-6, 3]
        
        print(f"    Applying HEB structure: {structure}")
        
        # This is complex - simplified version
        # Real implementation reorganizes the entire circuit
        # into hierarchical blocks with controlled entanglement
        
        return circuit  # Placeholder for now
    
    def _apply_spin_echo(self, circuit: Circuit) -> Circuit:
        """Apply spin echo sequences for error correction"""
        new_circuit = Circuit()
        
        # Find good echo points (middle of circuit segments)
        echo_points = [len(circuit.instructions) // 4,
                      len(circuit.instructions) // 2,
                      3 * len(circuit.instructions) // 4]
        
        for i, inst in enumerate(circuit.instructions):
            new_circuit.add_instruction(inst)
            
            if i in echo_points:
                # Add π pulse (X rotation) on all qubits
                for q in range(circuit.qubit_count):
                    new_circuit.rx(q, np.pi)
        
        return new_circuit
    
    def _reorder_circuit(self, circuit: Circuit) -> Circuit:
        """Reorder gates to minimize error accumulation"""
        # Strategy: Move single-qubit gates earlier, 2-qubit gates later
        single_qubit = []
        two_qubit = []
        
        for inst in circuit.instructions:
            if len(inst.target) == 1:
                single_qubit.append(inst)
            else:
                two_qubit.append(inst)
        
        # Rebuild circuit
        new_circuit = Circuit()
        
        # Add single-qubit gates first (where possible)
        for inst in single_qubit[:len(single_qubit)//2]:
            new_circuit.add_instruction(inst)
        
        # Then two-qubit gates
        for inst in two_qubit:
            new_circuit.add_instruction(inst)
        
        # Remaining single-qubit gates
        for inst in single_qubit[len(single_qubit)//2:]:
            new_circuit.add_instruction(inst)
        
        return new_circuit
    
    def _apply_symmetrization(self, circuit: Circuit) -> Circuit:
        """Exploit problem symmetries to reduce circuit complexity"""
        # This is problem-specific - simplified version
        # Real implementation would analyze the problem structure
        # For now: look for repeated subcircuits and share them
        return circuit
    
    def _apply_vermicular(self, circuit: Circuit) -> Circuit:
        """The VERMICULAR optimization - specialized for multi-stage Grover"""
        new_circuit = Circuit()
        
        # Detect Grover structure
        h_gates = sum(1 for inst in circuit.instructions if inst.operator.name == 'h')
        has_oracle = any(inst.operator.name in ['cz', 'ccnot'] for inst in circuit.instructions)
        
        if h_gates >= circuit.qubit_count * 2 and has_oracle:
            print("    VERMICULAR mode activated!")
            
            # Add initial superposition with protection
            h_count = 0
            for inst in circuit.instructions:
                if inst.operator.name == 'h' and h_count < circuit.qubit_count:
                    new_circuit.add_instruction(inst)
                    h_count += 1
                    if h_count == circuit.qubit_count:
                        # Add pre-oracle DD
                        for q in range(circuit.qubit_count):
                            new_circuit.x(q).x(q)
                        break
            
            # Add rest of circuit
            oracle_seen = False
            for i in range(h_count, len(circuit.instructions)):
                inst = circuit.instructions[i]
                new_circuit.add_instruction(inst)
                
                # Detect oracle
                if not oracle_seen and inst.operator.name in ['cz', 'ccnot']:
                    oracle_seen = True
            
            # Add final protection
            if oracle_seen:
                for q in range(circuit.qubit_count):
                    new_circuit.x(q).x(q)
            
            return new_circuit
        else:
            return circuit
    
    def _apply_multi_stage_protection(self, circuit: Circuit) -> Circuit:
        """Special protection for multi-stage algorithms"""
        analysis = self.analyzer.analyze_circuit_complete(circuit)
        
        if analysis['multi_stage_risk'] > 0.3:
            print("    Multi-stage protection engaged")
            
            # Estimate stage boundaries
            stage_size = len(circuit.instructions) // 3
            
            new_circuit = Circuit()
            
            for i, inst in enumerate(circuit.instructions):
                new_circuit.add_instruction(inst)
                
                # Add protection at stage boundaries
                if i == stage_size or i == stage_size * 2:
                    # Strong DD sequence
                    for q in range(circuit.qubit_count):
                        new_circuit.x(q).y(q).x(q).y(q)
                    
                    # Echo pulse
                    for q in range(circuit.qubit_count):
                        new_circuit.rx(q, np.pi)
            
            return new_circuit
        
        return circuit
    
    def _compress_gates(self, circuit: Circuit) -> Circuit:
        """Standard gate compression"""
        new_circuit = Circuit()
        skip_next = False
        
        for i in range(len(circuit.instructions)):
            if skip_next:
                skip_next = False
                continue
            
            inst = circuit.instructions[i]
            
            # Check for cancellations
            if i < len(circuit.instructions) - 1:
                next_inst = circuit.instructions[i + 1]
                
                if (inst.operator.name == next_inst.operator.name and
                    inst.operator.name in ['x', 'y', 'z', 'h'] and
                    inst.target == next_inst.target):
                    skip_next = True
                    continue
            
            new_circuit.add_instruction(inst)
        
        return new_circuit
    
    def _virtualize_z_gates(self, circuit: Circuit) -> Circuit:
        """Convert Z rotations to virtual (phase tracking)"""
        # For Braket, mostly conceptual
        return circuit
    
    def _load_hardware_profiles(self) -> Dict[str, HardwareProfile]:
        """Load hardware profiles with optimal strategies"""
        profiles = {}
        
        # IQM Garnet - Best for VERMICULAR
        profiles['iqm_garnet'] = HardwareProfile(
            name='IQM Garnet',
            n_qubits=20,
            connectivity={i: [(i-1)%20, (i+1)%20] for i in range(20)},
            t1_times={i: 100e-6 for i in range(20)},
            t2_times={i: 100e-6 for i in range(20)},
            gate_errors_1q={'x': 0.001, 'y': 0.001, 'z': 0.0001},
            gate_errors_2q={(i, j): 0.005 for i in range(20) for j in range(20) if abs(i-j) == 1},
            readout_errors={i: 0.03 for i in range(20)},
            gate_times={'x': 20e-9, 'cnot': 60e-9},
            native_gates=['rx', 'ry', 'cz'],
            recommended_strategies=[
                OptimizationStrategy.DD_XX,
                OptimizationStrategy.VERMICULAR
            ]
        )
        
        # Rigetti - Needs more protection
        profiles['rigetti'] = HardwareProfile(
            name='Rigetti Ankaa-3',
            n_qubits=84,
            connectivity={},  # Complex topology
            t1_times={i: 30e-6 for i in range(84)},
            t2_times={i: 40e-6 for i in range(84)},
            gate_errors_1q={'x': 0.002, 'y': 0.002, 'z': 0.0002},
            gate_errors_2q={(i, j): 0.01 for i in range(84) for j in range(84)},
            readout_errors={i: 0.05 for i in range(84)},
            gate_times={'x': 30e-9, 'cnot': 180e-9},
            native_gates=['rx', 'ry', 'cz', 'fsim'],
            recommended_strategies=[
                OptimizationStrategy.DD_XY4,
                OptimizationStrategy.ECHO_SPIN,
                OptimizationStrategy.CIRCUIT_REORDERING
            ]
        )
        
        return profiles


# ==================== QUANTUM SOLVER ====================

class QuantumSolverUltimate:
    """
    quantum solver v.0.1 
    ForgottenForge.xyz
    """
    
    def __init__(self):
        self.analyzer = AdvancedCriticalThresholdAnalyzer()
        self.optimizer = CompleteQuantumOptimizer()
        self.brute_forcer = BruteForceOptimizer()
        self.simulator = LocalSimulator()
        
        print("\n🚀 Quantum Solver initialized")
        print("   Implementing")
    
    def solve_with_full_optimization(self, 
                                    problem_description: str,
                                    target_hardware: str = 'auto',
                                    use_brute_force: bool = True,
                                    max_optimization_time: float = 60.0):
        """
        Solve quantum problem using the optimization toolkit
        
        Args:
            problem_description: Natural language problem
            target_hardware: 'iqm_garnet', 'rigetti', or 'auto'
            use_brute_force: Use brute-force strategy search
            max_optimization_time: Max time for optimization
        """
        print("\n" + "="*80)
        print("QUANTUM SOLVER v.0.1")
        print("="*80)
        
        # Step 1: Create initial circuit (simplified for demo)
        print(f"\n📊 Analyzing: '{problem_description}'")
        circuit = self._create_circuit_from_description(problem_description)
        print(f"   Circuit: {circuit.qubit_count} qubits, {len(circuit.instructions)} gates")
        
        # Step 2: Complete σ_c analysis
        print(f"\n🔬 Advanced Critical Threshold Analysis...")
        
        # Select hardware
        if target_hardware == 'auto':
            if 'multi' in problem_description.lower():
                target_hardware = 'iqm_garnet'  # Best for VERMICULAR
            else:
                target_hardware = 'rigetti'
        
        hardware = self.optimizer.hardware_profiles.get(target_hardware)
        analysis = self.analyzer.analyze_circuit_complete(circuit, hardware)
        
        print(f"   σ_c = {analysis['sigma_c']:.6f}")
        print(f"   Critical gates: {len(analysis['critical_gates'])}")
        print(f"   Idle periods: {sum(len(p) for p in analysis['idle_periods'].values())}")
        print(f"   Multi-stage risk: {analysis['multi_stage_risk']:.1%}")
        
        print(f"\n   Recommendations:")
        for rec in analysis['recommendations']:
            print(f"     • {rec}")
        
        # Step 3: Determine optimization strategies
        if use_brute_force:
            print(f"\n🔨 Brute-force optimization search...")
            
            # Select strategies based on analysis
            candidate_strategies = []
            
            # Always consider these
            candidate_strategies.extend([
                OptimizationStrategy.GATE_COMPRESSION,
                OptimizationStrategy.DD_XX
            ])
            
            # Add based on analysis
            if analysis['idle_periods']:
                candidate_strategies.extend([
                    OptimizationStrategy.DD_XY4,
                    OptimizationStrategy.DD_CPMG
                ])
            
            if analysis['multi_stage_risk'] > 0.3:
                candidate_strategies.extend([
                    OptimizationStrategy.VERMICULAR,
                    OptimizationStrategy.MULTI_STAGE_PROTECTION
                ])
            
            if len(analysis['critical_gates']) > len(circuit.instructions) * 0.3:
                candidate_strategies.extend([
                    OptimizationStrategy.HEB_ENCODING,
                    OptimizationStrategy.ECHO_SPIN
                ])
            
            # Remove duplicates
            candidate_strategies = list(set(candidate_strategies))
            
            # Find best combination
            best_result = self.brute_forcer.find_best_optimization(
                circuit,
                candidate_strategies,
                self.optimizer,
                max_time=max_optimization_time
            )
            
            best_strategies = best_result['strategies']
        else:
            # Use hardware-recommended strategies
            best_strategies = hardware.recommended_strategies if hardware else [
                OptimizationStrategy.DD_XX,
                OptimizationStrategy.GATE_COMPRESSION
            ]
        
        # Step 4: Apply optimizations
        print(f"\n✨ Applying optimizations: {[s.value for s in best_strategies]}")
        
        optimized_circuit = circuit
        for strategy in best_strategies:
            print(f"   Applying {strategy.value}...")
            optimized_circuit = self.optimizer.apply_single_strategy(optimized_circuit, strategy)
        
        # Step 5: Re-analyze
        print(f"\n📊 Final analysis...")
        final_analysis = self.analyzer.analyze_circuit_complete(optimized_circuit, hardware)
        
        print(f"   Original: {len(circuit.instructions)} gates, σ_c = {analysis['sigma_c']:.6f}")
        print(f"   Optimized: {len(optimized_circuit.instructions)} gates, σ_c = {final_analysis['sigma_c']:.6f}")
        print(f"   σ_c improvement: {final_analysis['sigma_c']/analysis['sigma_c']:.1f}x")
        
        # Step 6: Test on simulator
        print(f"\n🧪 Testing on simulator...")
        
        # Original
        orig_result = self.simulator.run(circuit, shots=1000).result()
        orig_success = self._calculate_success_rate(orig_result.measurements)
        
        # Optimized
        opt_result = self.simulator.run(optimized_circuit, shots=1000).result()
        opt_success = self._calculate_success_rate(opt_result.measurements)
        
        improvement = opt_success / orig_success if orig_success > 0 else 1.0
        
        print(f"\n📈 RESULTS:")
        print(f"   Original success: {orig_success:.1%}")
        print(f"   Optimized success: {opt_success:.1%}")
        print(f"   Improvement: {improvement:.1f}x")
        
        if improvement > 10:
            print(f"\n🎉 VERMICULAR-CLASS OPTIMIZATION ACHIEVED! {improvement:.0f}x improvement!")
        elif improvement > 5:
            print(f"\n🎊 Excellent optimization! {improvement:.1f}x improvement!")
        elif improvement > 2:
            print(f"\n✅ Good optimization: {improvement:.1f}x improvement")
        
        # Step 7: QPU recommendations
        print(f"\n🎯 QPU Readiness:")
        if final_analysis['sigma_c'] > 0.01 and opt_success > 0.5:
            print(f"   ✅ Circuit is QPU-ready!")
            print(f"   Recommended: {hardware.name if hardware else 'IQM Garnet'}")
        else:
            print(f"   ⚠️  Needs more optimization for QPU")
        
        return {
            'original_circuit': circuit,
            'optimized_circuit': optimized_circuit,
            'strategies_applied': best_strategies,
            'improvement': improvement,
            'analysis': final_analysis
        }
    
    def _create_circuit_from_description(self, description: str) -> Circuit:
        """Create circuit from problem description (simplified)"""
        desc_lower = description.lower()
        
        # Detect problem type and parameters
        if 'search' in desc_lower or 'find' in desc_lower:
            # Grover
            n_qubits = 4  # Default
            
            # Extract number
            import re
            numbers = re.findall(r'\d+', description)
            if numbers:
                for num in numbers:
                    if int(num) > 10:
                        n_qubits = int(np.ceil(np.log2(int(num))))
                        break
            
            return self._create_grover_circuit(min(n_qubits, 5))
        
        else:
            # Generic circuit
            return self._create_generic_circuit(3)
    
    def _create_grover_circuit(self, n_qubits: int) -> Circuit:
        """Create Grover circuit"""
        circuit = Circuit()
        
        # Superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # Oracle (simplified)
        for i in range(n_qubits-1):
            circuit.x(i)
        
        # Multi-controlled Z
        if n_qubits == 2:
            circuit.cz(0, 1)
        elif n_qubits == 3:
            circuit.h(2)
            circuit.ccnot(0, 1, 2)
            circuit.h(2)
        else:
            for i in range(n_qubits-1):
                circuit.cz(i, i+1)
        
        for i in range(n_qubits-1):
            circuit.x(i)
        
        # Diffusion
        for i in range(n_qubits):
            circuit.h(i)
            circuit.x(i)
        
        if n_qubits == 2:
            circuit.cz(0, 1)
        elif n_qubits == 3:
            circuit.h(2)
            circuit.ccnot(0, 1, 2)
            circuit.h(2)
        else:
            for i in range(n_qubits-1):
                circuit.cz(i, i+1)
        
        for i in range(n_qubits):
            circuit.x(i)
            circuit.h(i)
        
        return circuit
    
    def _create_generic_circuit(self, n_qubits: int) -> Circuit:
        """Create generic test circuit"""
        circuit = Circuit()
        
        for i in range(n_qubits):
            circuit.h(i)
        
        for i in range(n_qubits-1):
            circuit.cnot(i, i+1)
        
        for i in range(n_qubits):
            circuit.rx(i, np.pi/4)
        
        return circuit
    
    def _calculate_success_rate(self, measurements) -> float:
        """Calculate success rate from measurements"""
        # Fix: Properly check if measurements is empty
        if measurements is None or len(measurements) == 0:
            return 0.0
        
        # Count most common outcome
        counts = {}
        for m in measurements:
            key = tuple(m)
            counts[key] = counts.get(key, 0) + 1
        
        if not counts:  # This is fine because counts is a dict
            return 0.0
            
        max_count = max(counts.values())
        return max_count / len(measurements)


# ==================== DEMO ====================

def run_complete_demo():
    """Run demo showing ALL optimization techniques"""
    solver = QuantumSolverUltimate()
    
    print("\n" + "="*80)
    print("QUANTUM SOLVER ULTIMATE DEMO")
    print("Showcasing ALL optimization techniques from VERMICULAR research")
    print("="*80)
    
    # Test problems
    problems = [
        "Search for item 5 in 32-element database",
        "Multi-stage search for 3 items in quantum database",
        "Find minimum energy of 4-qubit molecule",
    ]
    
    for problem in problems:
        print(f"\n{'='*80}")
        print(f"PROBLEM: {problem}")
        print('='*80)
        
        # Use brute force to find best strategy
        result = solver.solve_with_full_optimization(
            problem,
            target_hardware='auto',
            use_brute_force=True,
            max_optimization_time=30.0
        )
        
        input("\nPress Enter for next problem...")



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            run_complete_demo()
        else:
            # Solve specific problem from command line
            solver = QuantumSolverUltimate()
            problem = " ".join(sys.argv[1:])
            print(f"\nSolving: {problem}")
            solver.solve_with_full_optimization(problem)
    else:
        # Interactive mode 
        solver = QuantumSolverUltimate()
        
        print("\nEnter your quantum problem (or 'demo' for full demo):")
        problem = input("> ")
        
        if problem.lower() == 'demo':
            run_complete_demo()
        elif problem.lower() == 'exit':
            print("Goodbye!")
        else:

            solver.solve_with_full_optimization(problem)
