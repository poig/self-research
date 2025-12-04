"""
nisq_v2.py: Riemannian Coherent QLTO (The Unified Architecture) - MPS Fix

This implementation merges the "Sandwich" Quantum Walk from 'nisq.py' with the 
rigorous Commuting-Block Geometry. It has been patched to support:
1. Matrix Product State (MPS) simulation for larger systems/entanglement.
2. Robust V2 Primitive Wrappers that correctly handle parameter binding.
3. Fixed interaction with geometry engines.

Author: QLTO Synthesis Team
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional

# Qiskit Imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, CXGate, PauliEvolutionGate, PhaseGate
    from qiskit.synthesis import LieTrotter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    # USE V2 PRIMITIVES
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator, SamplerV2 as AerSampler
    
    QISKIT_AVAILABLE = True
except ImportError:
    print("WARNING: Qiskit not found. Code will not execute.")
    QISKIT_AVAILABLE = False

# Import Efficient Engines (The Glue)
try:
    from commute_fim import CommutingBlockFIM
    from commute_gradient import CommutingBlockGradient
    ENGINES_AVAILABLE = True
except ImportError:
    print("WARNING: 'commute_fim.py' or 'commute_gradient.py' not found. Falling back to heuristic mode.")
    ENGINES_AVAILABLE = False

# Import Visualization (Optional)
try:
    from visualize_ancilla import AncillaVisualizer
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# --- Primitive Wrappers (Fixed for V2) ---
class BaseEstimator: 
    def __init__(self, backend=None): 
        # Configure options for AerEstimatorV2
        method = 'automatic'
        if backend and hasattr(backend, 'options'):
            method = getattr(backend.options, 'method', 'automatic')
            
        # V2 takes 'options' dict
        # We set backend_options inside it
        self._estimator = AerEstimator(options={'backend_options': {'method': method}})
        
    def run(self, pubs, **kwargs): 
        """
        Robust run method that handles (Circuit, Observable, [Params]) tuples.
        """
        return self._estimator.run(pubs, **kwargs)

class BaseSampler: 
    def __init__(self, backend=None): 
        self._sampler = AerSampler()
        
    def run(self, pubs, **kwargs): 
        """
        Robust run method that handles (Circuit, [Params]) tuples.
        """
        return self._sampler.run(pubs, **kwargs)

# --- Core Logic ---

class RiemannianQLTO:
    def __init__(
        self, 
        ansatz: QuantumCircuit, 
        hamiltonian: SparsePauliOp, 
        bits_per_param: int = 1,
        shot_budget: int = 4096,
        backend=None,
        fim_full: bool = False,
    ):
        if not QISKIT_AVAILABLE: raise RuntimeError("Qiskit required.")
        
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.bits_per_param = bits_per_param
        self.shot_budget = shot_budget
        
        # MPS Backend Setup
        if backend is None:
            print("[Init] Configuring AerSimulator with method='matrix_product_state'")
            self.backend = AerSimulator(method='matrix_product_state')
        else:
            self.backend = backend
        
        # Initialize Engines
        self.estimator = BaseEstimator(self.backend)
        self.sampler = BaseSampler(self.backend)
        
        if ENGINES_AVAILABLE:
            # Paper 6: Efficient Metric Sensing
            self.fim_engine = CommutingBlockFIM(ansatz, full=fim_full)
            # Paper 2: Efficient Gradient Sensing
            self.grad_engine = CommutingBlockGradient(ansatz, hamiltonian)
            self.has_engines = True
            print(f"[Init] Riemannian Engines Online. Detected {len(self.fim_engine.layers)} commuting layers.")
        else:
            self.has_engines = False
            print("[Init] Engines missing. Reverting to blind heuristic walk.")

        # State tracking
        self.metric_diag = None
        self.gradient_vec = None
        self.nefv = 0 # Function evaluations counter
        
        # Circuit depth tracking (for benchmarking vs QAOA)
        self.last_circuit_depth = 0
        self.max_circuit_depth = 0
        
        # Sensing diagnostics (Paper 1: "monitor if the layer is trained well")
        self.layer_diagnostics = {}  # Stores activation rates, entropy per layer
        self.last_activation_rate = 0.0
        self.last_entropy = 0.0

    def build_w_gate(
        self, 
        param_reg, 
        sys_reg, 
        center_params: np.ndarray,
        search_radius: float,
        active_indices: Optional[List[int]] = None
    ) -> QuantumCircuit:
        qc = QuantumCircuit(param_reg, sys_reg, name="W_Gate")
        decomp = self.ansatz.decompose()
        param_order = list(self.ansatz.parameters)

        # === OPTIMIZATION: FAST PATH (Global Mode) ===
        if active_indices is None:
            # Original fast loop (No dictionary lookups, no 'if' checks)
            for instr in decomp.data:
                op = instr.operation
                if len(op.params) == 1 and isinstance(op.params[0], Parameter):
                    p_obj = op.params[0]
                    try:
                        p_idx = param_order.index(p_obj)
                    except ValueError: continue
                    
                    sys_q_idx = decomp.find_bit(instr.qubits[0]).index
                    target_qubit = sys_reg[sys_q_idx]
                    
                    # Direct Quantum Encoding
                    start_bit = p_idx * self.bits_per_param
                    min_val = center_params[p_idx] - search_radius
                    self._apply_gate(qc, op, min_val, target_qubit) 
                    
                    full_range = 2 * search_radius
                    for b in range(self.bits_per_param):
                        ctrl_qubit = param_reg[start_bit + b]
                        angle = full_range / (2**self.bits_per_param) * (2**b)
                        self._apply_controlled_gate(qc, op, angle, ctrl_qubit, target_qubit)
                        
                elif isinstance(op, CXGate):
                    q1 = decomp.find_bit(instr.qubits[0]).index
                    q2 = decomp.find_bit(instr.qubits[1]).index
                    qc.cx(sys_reg[q1], sys_reg[q2])
            return qc

        # Map global param index -> local register index
        active_map = {global_idx: i for i, global_idx in enumerate(active_indices)}
        
        qc = QuantumCircuit(param_reg, sys_reg, name="W_Gate")
        param_order = list(self.ansatz.parameters)
        decomp = self.ansatz.decompose()
        
        for instr in decomp.data:
            op = instr.operation
            
            # Check if instruction is parameterized
            if len(op.params) == 1 and isinstance(op.params[0], Parameter):
                p_obj = op.params[0]
                try:
                    p_idx = param_order.index(p_obj)
                except ValueError: continue

                # Identify target qubit in system register
                sys_q_idx = decomp.find_bit(instr.qubits[0]).index
                target_qubit = sys_reg[sys_q_idx]
                
                # --- BRANCH: IS PARAMETER ACTIVE? ---
                if p_idx in active_indices:
                    # === ACTIVE (QUANTUM SUPERPOSITION) ===
                    # Use the quantum parameter register
                    local_idx = active_map[p_idx]
                    start_bit = local_idx * self.bits_per_param
                    
                    # 1. Apply Base Rotation (Center - Radius)
                    min_val = center_params[p_idx] - search_radius
                    self._apply_gate(qc, op, min_val, target_qubit) 
                    
                    # 2. Apply Controlled Increments
                    full_range = 2 * search_radius
                    for b in range(self.bits_per_param):
                        ctrl_qubit = param_reg[start_bit + b]
                        angle = full_range / (2**self.bits_per_param) * (2**b)
                        self._apply_controlled_gate(qc, op, angle, ctrl_qubit, target_qubit)

                else:
                    # === FROZEN (CLASSICAL CONSTANT) ===
                    # Just apply the gate with its current fixed value
                    # No parameter qubits required!
                    fixed_val = center_params[p_idx]
                    self._apply_gate(qc, op, fixed_val, target_qubit)
            
            elif isinstance(op, CXGate):
                # Standard Entanglement (Unchanged)
                q1 = decomp.find_bit(instr.qubits[0]).index
                q2 = decomp.find_bit(instr.qubits[1]).index
                qc.cx(sys_reg[q1], sys_reg[q2])
                
        return qc

    # Helper for gate application to keep code clean
    def _apply_gate(self, qc, op, angle, target):
        if isinstance(op, RYGate): qc.ry(angle, target)
        elif isinstance(op, RZGate): qc.rz(angle, target)
        elif isinstance(op, RXGate): qc.rx(angle, target)
        elif isinstance(op, PhaseGate): qc.p(angle, target)

    def _apply_controlled_gate(self, qc, op, angle, ctrl, target):
        if isinstance(op, RYGate): qc.append(RYGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RZGate): qc.append(RZGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RXGate): qc.append(RXGate(angle).control(1), [ctrl, target])
        elif isinstance(op, PhaseGate): qc.append(PhaseGate(angle).control(1), [ctrl, target])

    def run_walk(
        self, 
        center_params: np.ndarray, 
        k_steps: int = 2, 
        delta_t: float = 0.5, 
        search_radius: float = 0.5,
        layer: bool = False,
        gradient_reuse: bool = False,
        coherence: bool = False,
    ):
        """
        Execute Riemannian quantum walk for parameter optimization.
        
        After calling this method, diagnostics are available via:
        - self.get_sensing_diagnostics() - Full layer-by-layer info
        - self.last_activation_rate - Quick check of last layer's activation
        - self.last_entropy - Quick check of last layer's entropy
        """
        # Clear diagnostics from previous run
        self.layer_diagnostics = {}
        
        # Precompute expensive terms ONCE per epoch
        if gradient_reuse:
            # 1. Compute Global Gradient ONCE (Cost: 2N or 2L circuits)
            global_grad = self.grad_engine.compute_gradient(self.estimator, center_params)
            self.nefv += self.grad_engine.get_nefv_cost()
        else:
            global_grad = None
        
        # 2. Compute Global FIM ONCE (Cost: L circuits) - REUSE across layers!
        global_fim = self.fim_engine.compute_fim(self.estimator, center_params)
        self.nefv += self.fim_engine.get_nefv_cost()
        
        if layer and self.has_engines:
            
            current_params = center_params.copy()
            
            # 2. Loop Layers (Cost: Cheap! FIM and Grad are reused)
            for i, layer_info in enumerate(self.fim_engine.layers):
                active_indices = layer_info['params']
                if not active_indices: continue
                
                # Pass the precomputed grad AND fim to the worker
                current_params = self._execute_walk(
                    current_params, 
                    k_steps, delta_t, search_radius, 
                    active_indices=active_indices,
                    precomputed_grad=global_grad,
                    precomputed_fim=global_fim,
                    coherence=coherence
                )
            return current_params
        else:
            # Global mode: Just run normally
            return self._execute_walk(
                center_params, 
                k_steps, delta_t, search_radius, 
                active_indices=None,
                precomputed_grad=global_grad,
                precomputed_fim=global_fim,
                coherence=coherence
            )
    
    def get_sensing_diagnostics(self) -> Dict[str, Any]:
        """
        Get sensing diagnostics from the last run_walk call.
        
        Returns a summary of the ancilla sensing performance:
        - Per-layer activation rates (P(ancilla=|1⟩))
        - Per-layer entropy (solution diversity)
        - Aggregate statistics
        
        Paper 1 Interpretation:
        - High activation (>50%): Oracle is sensing low-energy states well
        - Low activation (<20%): Oracle may be stuck or sensing poorly
        - High entropy: Exploring many parameter configurations
        - Low entropy: Converging to specific configuration
        """
        if not self.layer_diagnostics:
            return {'status': 'no_data', 'message': 'No walk executed yet'}
        
        activation_rates = [d['activation_rate'] for d in self.layer_diagnostics.values()]
        entropies = [d['normalized_entropy'] for d in self.layer_diagnostics.values()]
        
        return {
            'layers': self.layer_diagnostics,
            'mean_activation': np.mean(activation_rates) if activation_rates else 0.0,
            'mean_entropy': np.mean(entropies) if entropies else 0.0,
            'min_activation': np.min(activation_rates) if activation_rates else 0.0,
            'max_activation': np.max(activation_rates) if activation_rates else 0.0,
            'n_layers': len(self.layer_diagnostics),
            'training_quality': self._assess_training_quality(activation_rates, entropies)
        }
    
    def _assess_training_quality(self, activation_rates, entropies) -> str:
        """
        Assess training quality based on sensing diagnostics.
        
        Paper 1 Interpretation:
        - High activation (>40%): Oracle sensing low-energy states → "excellent"
        - Moderate activation (20-40%): Learning in progress → "good"
        - Low activation (<20%) + High entropy: Stuck/exploring → "exploring"
        - Low activation (<20%) + Low entropy: Converging to minimum → "converging"
        
        Returns a qualitative assessment string.
        """
        if not activation_rates:
            return "unknown"
        
        mean_act = np.mean(activation_rates)
        mean_ent = np.mean(entropies)
        
        if mean_act > 0.4:
            return "excellent"
        elif mean_act > 0.2:
            return "good"
        elif mean_ent < 0.5:
            # Low activation + low entropy = converging to solution
            return "converging"
        else:
            # Low activation + high entropy = stuck or random walk
            return "exploring"
    
    def _execute_walk(self, center_params, k_steps, delta_t, radius, active_indices=None, precomputed_grad=None, precomputed_fim=None, coherence=False):
        """
        Internal worker that builds and runs the circuit for a specific subset.
        
        Implements FULL SENSING PROTOCOL (Paper 1):
        1. Sensing: Ancilla in |+⟩, controlled-U(τ) accumulates phase
        2. Correlation: Hadamard converts phase → Z-population  
        3. Feedback: Controlled mixer based on ancilla state
        4. Measurement: Both ancilla (activation) and params (new values)
        
        Returns:
            new_params: Updated parameter vector
            Also updates self.layer_diagnostics with sensing info
        """
        
        # 1. Determine Dimensions
        if active_indices is None:
            active_indices = list(range(len(center_params)))
            
        n_active = len(active_indices)
        n_p_qubits = n_active * self.bits_per_param
        layer_id = hash(tuple(active_indices)) % 1000  # Simple layer identifier
        
        # --- OPTIMIZATION: FIM AND GRADIENT REUSE ---
        # 1. Metric: Use precomputed if available (CHEAP when reused!)
        if precomputed_fim is not None:
            metric_matrix = precomputed_fim
        else:
            metric_matrix = self.fim_engine.compute_fim(self.estimator, center_params)
            self.nefv += self.fim_engine.get_nefv_cost()
        metric_diag = np.diag(metric_matrix)
        metric_local = metric_diag[active_indices]
        
        # 2. Gradient: Use precomputed if available, otherwise compute (Expensive!)
        if precomputed_grad is not None:
            grad_full = precomputed_grad
        else:
            grad_full = self.grad_engine.compute_gradient(self.estimator, center_params)
            self.nefv += self.grad_engine.get_nefv_cost()
            
        grad_local = grad_full[active_indices]
        
        # 3. Build Registers
        # Notice: param register is sized ONLY for active parameters!
        # FULL SENSING: Add ancilla measurement register
        qr_anc = AncillaRegister(1, 'anc')
        qr_param = QuantumRegister(n_p_qubits, 'param')
        qr_sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cr_param = ClassicalRegister(n_p_qubits, 'cr_param')
        cr_anc = ClassicalRegister(1, 'cr_anc')  # Paper 1: Measure ancilla for sensing
        
        qc = QuantumCircuit(qr_anc, qr_param, qr_sys, cr_param, cr_anc)
        
        # 4. Initialization (Paper 1 Phase 1: Prepare |+⟩_A)
        qc.h(qr_anc)  # Ancilla in superposition for sensing
        qc.h(qr_param)  # Parameter register in superposition
        
        # 5. Hybrid W-Gate (Mixes Quantum Active + Classical Frozen)
        # MUST come BEFORE sensing so that e^{-iHτ} sees the encoded parameters!
        w_gate = self.build_w_gate(qr_param, qr_sys, center_params, radius, active_indices)
        qc.append(w_gate, list(qr_param) + list(qr_sys))
        
        # =====================================================
        # PHASE 1: SENSING (Paper 1) - Done ONCE after W-gate
        # "The sensing phase entangles the ancilla state with the 
        # system energy" - must happen AFTER parameter encoding
        # so we sense E(θ) not just E(initial state).
        # =====================================================
        trotter = LieTrotter(reps=1)
        sensing_time = delta_t * np.pi  # Total sensing duration
        evo_sense = PauliEvolutionGate(self.hamiltonian, time=sensing_time, synthesis=trotter)
        
        if coherence:
            # Controlled sensing: ancilla accumulates phase ⟨cos(E(θ)τ)⟩
            qc.append(evo_sense.control(1), [qr_anc[0]] + list(qr_sys))
        else:
            # Non-coherent: just evolve system (less sensing power)
            qc.append(evo_sense, qr_sys)
        
        # 6. Walk Loop - Parameter space exploration
        # NOTE: No more Hamiltonian evolution here! 
        # The sensing is done, now we just diffuse in parameter space
        for k in range(k_steps):
            # Annealing schedule within walk
            s = (k + 0.5) / k_steps
            res_scale = 1.0 / np.sqrt(self.bits_per_param)
            gamma = s * np.pi * delta_t * res_scale  # Phase accumulation rate
            beta = (1 - s) * np.pi * delta_t         # Mixing strength
            
            # --- Drift (Phase Kickback based on gradient) ---
            # This encodes "which direction is downhill" into phases
            # OPTIMIZATION: Use individual CRZ gates instead of grouped control
            # This is shallower than drift_qc.control(1) which creates multi-controlled block
            # Each walker is independent, so individual control is mathematically equivalent
            for i in range(n_active):
                g_ii = metric_local[i]
                grad_i = grad_local[i]
                ng_scale = np.clip(1.0 / (np.sqrt(g_ii) + 1e-6), 0.1, 5.0)
                
                for b in range(self.bits_per_param):
                    q_idx = i * self.bits_per_param + b
                    drift_weight = (2.0 ** b) / (2.0 ** (self.bits_per_param - 1))
                    angle = gamma * grad_i * ng_scale * drift_weight
                    
                    if coherence:
                        # Individual controlled-RZ: only 2-qubit depth
                        qc.crz(angle, qr_anc[0], qr_param[q_idx])
                    else:
                        qc.rz(angle, qr_param[q_idx])

            # --- Mixer (Diffusion) ---
            # OPTIMIZATION: Use individual CRX gates instead of grouped control
            for i in range(n_active):
                g_ii = metric_local[i]
                scale = np.clip(1.0 / (np.sqrt(g_ii) + 1e-6), 0.1, 5.0)
                
                for b in range(self.bits_per_param):
                    q_idx = i * self.bits_per_param + b
                    mix_weight = 1.0 / (2.0 ** b)
                    angle = beta * scale * mix_weight
                    
                    if coherence:
                        # Individual controlled-RX: only 2-qubit depth
                        qc.crx(angle, qr_anc[0], qr_param[q_idx])
                    else:
                        qc.rx(angle, qr_param[q_idx])
            
            # NOTE: True coherent integration means NO reset between steps.
            # The ancilla remains entangled, accumulating phase information
            # across all K steps. This matches paper claim:
            # "coherent integration (no intermediate reset)"
        
        # =====================================================
        # PHASE 2: CORRELATION (Paper 1)
        # Convert ancilla phase information to Z-basis population
        # This is the Hadamard test: ⟨cos(Eτ)⟩ becomes measurable
        # =====================================================
        qc.h(qr_anc)  # Phase → Population conversion
                    
        # 7. Measurement & Decode
        qc.append(w_gate.inverse(), list(qr_param) + list(qr_sys))
        
        # FULL SENSING: Measure both ancilla AND parameters
        qc.measure(qr_param, cr_param)
        qc.measure(qr_anc, cr_anc)  # Paper 1: "ancilla sensing"

        # print(qc.draw())
        
        t_qc = transpile(qc.decompose(reps=4), self.backend)
        
        # Track circuit depth for benchmarking
        self.last_circuit_depth = t_qc.depth()
        self.max_circuit_depth = max(self.max_circuit_depth, self.last_circuit_depth)
        
        result = self.backend.run(t_qc, shots=self.shot_budget).result()
        counts = result.get_counts()
        self.nefv += 1
        
        # DEBUG: Print first few bitstrings to understand format
        # if len(counts) > 0 and n_active <= 4:
        #     sample_keys = list(counts.keys())[:3]
        #     print(f"  [DEBUG] Sample bitstrings: {sample_keys}, n_p_qubits={n_p_qubits}")
        
        # =====================================================
        # SENSING DIAGNOSTICS (Paper 1: "monitor if layer is trained well")
        # =====================================================
        total_shots = sum(counts.values())
        ancilla_one_count = 0  # Count |1⟩_anc outcomes (low energy sensing)
        
        # Parse counts to separate ancilla and param outcomes
        param_counts_move = {}   # Counts when ancilla = |1⟩ (MOVE)
        param_counts_stay = {}   # Counts when ancilla = |0⟩ (STAY)
        
        for bitstr, count in counts.items():
            # Qiskit format with multiple classical registers: "cr_anc cr_param"
            # cr_anc is measured AFTER cr_param, so it appears FIRST in the string
            parts = bitstr.split(' ')
            
            if len(parts) == 2:
                # Format: "anc_bits param_bits"
                anc_bit = parts[0][-1]  # Last char of ancilla register (LSB)
                param_bits = parts[1]
            elif len(parts) == 1:
                # No space separator - ancilla is at MSB position
                clean = parts[0]
                anc_bit = clean[0]  # MSB is first char
                param_bits = clean[1:]
            else:
                # Unexpected format
                anc_bit = '0'
                param_bits = bitstr.replace(" ", "")
            
            if anc_bit == '1':
                ancilla_one_count += count
                param_counts_move[param_bits] = param_counts_move.get(param_bits, 0) + count
            else:
                param_counts_stay[param_bits] = param_counts_stay.get(param_bits, 0) + count
        
        # Activation Rate: P(ancilla = |1⟩) = "probability of sensing low energy"
        activation_rate = ancilla_one_count / total_shots if total_shots > 0 else 0.0
        
        # Shannon Entropy of parameter distribution (diversity of solutions)
        all_probs = np.array(list(counts.values())) / total_shots
        entropy = -np.sum(all_probs * np.log2(all_probs + 1e-12))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Store diagnostics
        self.layer_diagnostics[layer_id] = {
            'activation_rate': activation_rate,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'move_count': ancilla_one_count,
            'stay_count': total_shots - ancilla_one_count,
            'n_params': n_active,
        }
        self.last_activation_rate = activation_rate
        self.last_entropy = normalized_entropy
        
        # =====================================================
        # ACTIVATION-WEIGHTED DECODE (Paper 1: Trap-Diffusion)
        # Only use "MOVE" outcomes (ancilla=|1⟩) for parameter update
        # This implements: "low-energy configurations bias toward states 
        # that receive more mixing" - Paper 1, Section 2
        # =====================================================
        if len(param_counts_move) > 0 and activation_rate > 0.05:
            # Decode from MOVE outcomes only (these are the low-energy samples)
            decoded_block = self._decode_result_with_ancilla(
                param_counts_move, center_params[active_indices], radius, n_active
            )
        else:
            # Fallback: If activation too low, use all samples but stay conservative
            # This prevents getting stuck when oracle isn't sensing well
            decoded_block = self._decode_result_with_ancilla(
                {k[1:] if len(k) > n_p_qubits else k: v for k, v in counts.items()},
                center_params[active_indices], radius, n_active
            )
            # Scale down the update when activation is low (conservative)
            decoded_block = center_params[active_indices] + 0.3 * (decoded_block - center_params[active_indices])
        
        # Update the full vector
        new_params = center_params.copy()
        new_params[active_indices] = decoded_block
        
        return new_params
    
    def _decode_result_with_ancilla(self, param_counts, center_params, radius, n_params):
        """
        Decode parameter values from measurement counts.
        Uses weighted centroid over all outcomes.
        
        Args:
            param_counts: Dict of {bitstring: count} for parameter register only
            center_params: Current parameter values for this layer
            radius: Search radius
            n_params: Number of parameters in this layer
        """
        accumulated_params = np.zeros(n_params)
        total_weight = 0.0
        max_int = (2**self.bits_per_param) - 1
        
        for bitstr, count in param_counts.items():
            weight = count
            clean_str = bitstr.replace(" ", "")
            current_val = np.zeros(n_params)
            
            # Handle case where bitstring might be shorter than expected
            expected_len = n_params * self.bits_per_param
            if len(clean_str) < expected_len:
                clean_str = clean_str.zfill(expected_len)
            elif len(clean_str) > expected_len:
                clean_str = clean_str[-expected_len:]  # Take rightmost bits
            
            # Decode each parameter
            rev_str = clean_str[::-1]  # Reverse for LSB-first
            for i in range(n_params):
                start = i * self.bits_per_param
                end = start + self.bits_per_param
                p_bits = rev_str[start:end] if end <= len(rev_str) else '0' * self.bits_per_param
                
                val_int = 0
                for b_idx, bit in enumerate(p_bits):
                    if bit == '1':
                        val_int += 2**b_idx
                
                # Linear mapping: [0, max_int] -> [center-R, center+R]
                norm = val_int / max_int if max_int > 0 else 0.5
                p_min = center_params[i] - radius
                p_max = center_params[i] + radius
                real_val = p_min + (norm * (p_max - p_min))
                current_val[i] = real_val
            
            accumulated_params += current_val * weight
            total_weight += weight
        
        if total_weight == 0:
            return center_params
        
        return accumulated_params / total_weight

    # Update signature to accept a 'mode'
    def _decode_result(self, counts, center_params, radius, mode='linear'):
        """
        Decodes bitstrings to parameter values.
        mode='linear': Standard search [center-R, center+R]
        mode='log':    Exponential search [center/R, center*R] (requires radius > 1)
        """
        n_params = len(center_params)
        accumulated_params = np.zeros(n_params)
        total_weight = 0.0
        
        # Filter top results to reduce noise
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_n = min(len(sorted_counts), 16)
        
        max_int = (2**self.bits_per_param) - 1
        
        for bitstr, count in sorted_counts[:top_n]:
            weight = count
            clean_str = bitstr.replace(" ", "")
            current_val = np.zeros(n_params)
            
            for i in range(n_params):
                # ... [Same Bit Parsing Logic as before] ...
                # Reverse bit logic to match register order
                rev_str = clean_str[::-1]
                start = i * self.bits_per_param
                end = start + self.bits_per_param
                p_bits = rev_str[start:end]
                
                val_int = 0
                for b_idx, bit in enumerate(p_bits):
                    if bit == '1': val_int += 2**b_idx
                
                # --- IMPROVEMENT 2: LOGIC SWITCH ---
                if mode == 'linear':
                    # Standard: center ± radius
                    norm = val_int / max_int if max_int > 0 else 0.5
                    p_min = center_params[i] - radius
                    p_max = center_params[i] + radius
                    real_val = p_min + (norm * (p_max - p_min))
                    
                elif mode == 'log':
                    # Geometric: center * (radius^offset)
                    # Maps int [0..max] to range [-1, 1]
                    # radius becomes the "Multiplicative Factor" (e.g. 10x)
                    if max_int == 0: norm = 0
                    else: norm = (val_int / max_int) * 2 - 1  # -1.0 to 1.0
                    
                    # If radius is 10, this scans [center/10 ... center*10]
                    real_val = center_params[i] * (radius ** norm)

                current_val[i] = real_val
                
            accumulated_params += current_val * weight
            total_weight += weight
            
        if total_weight == 0: return center_params
        return accumulated_params / total_weight

# --- Helper: Heisenberg Hamiltonian ---
def force_heisenberg_hamiltonian(n_qubits):
    ops = []
    for i in range(n_qubits - 1):
        for pauli in ['X', 'Y', 'Z']:
            op_str = ["I"] * n_qubits
            op_str[i] = pauli
            op_str[i+1] = pauli
            ops.append(("".join(op_str), 1.0))
    return SparsePauliOp.from_list(ops)

def generate_frustrated_hamiltonian(n_qubits, seed=42):
    """
    Generates a Random Transverse-Field Ising Model (Spin Glass).
    H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i
    
    - J_ij: Random couplings in [-1, 1]. Creates frustration (competing constraints).
    - h_i:  Random transverse fields. Creates quantum fluctuations (non-commuting).
    
    This landscape is RUGGED. Simple gradient descent often fails.
    """
    np.random.seed(seed)
    ops = []
    
    # 1. Interaction Terms (Z_i Z_j)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = np.random.uniform(-1.0, 1.0)
            op_str = ["I"] * n_qubits
            op_str[i] = "Z"
            op_str[j] = "Z"
            ops.append(("".join(op_str), J))
            
    # 2. Transverse Field (X_i)
    for i in range(n_qubits):
        h = np.random.uniform(-1.0, 1.0)
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), h))
        
    op = SparsePauliOp.from_list(ops)
    if op is None:
        raise ValueError("Failed to generate Hamiltonian! (Result was None)")
    return op

if __name__ == "__main__":
    print("=== NISQ V2: Riemannian Coherent QLTO (Full Sensing Protocol) ===")
    print("[Info] Paper 1 Implementation: Ancilla sensing + activation monitoring")
    
    # Visualization flag (set True to generate fractal visualization)
    VISUALIZATION = True
    
    # 1. Setup Problem
    N = 4
    H = generate_frustrated_hamiltonian(N, seed=42)
    # EfficientSU2 is naturally structured into commuting blocks (Rotation layers)
    # Decompose to ensure it's compatible with Aer primitives
    ansatz = EfficientSU2(N, reps=1, entanglement='linear').decompose()
    print(f"Ansatz Ops: {ansatz.count_ops()}")
    
    print(f"Problem: {N} Qubits, {ansatz.num_parameters} Parameters.")
    
    # 2. Initialize Optimizer (Using 1 bit per param to keep simulation fast)
    qlto = RiemannianQLTO(ansatz, H, bits_per_param=1, shot_budget=8192)
    print(f"{qlto.bits_per_param} bits per param")
    
    # 3. Run Loop
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    search_radius = 0.6
    
    # Reference estimator for printing energy (Explicitly Statevector for Ground Truth)
    # Use V2 for reference too
    ref_est = AerEstimator(options={'backend_options': {'method': 'statevector'}})
    
    # Initialize visualizer if enabled
    visualizer = None
    if VISUALIZATION and VIZ_AVAILABLE:
        visualizer = AncillaVisualizer(output_dir="./figures")
        print("[Viz] Ancilla visualizer enabled. Will generate fractal at end.")
    
    print("\nStarting Optimization with Full Sensing Protocol...")
    print("=" * 70)
    start_time = time.time()
    
    for epoch in range(30):
        # Eval current energy
        # V2: run([ (circuit, observables, params) ])
        pub = (ansatz, H, params)
        try:
            job = ref_est.run([pub])
            result = job.result()
            E = float(result[0].data.evs)
        except Exception as e:
            print(f"Energy Eval Failed: {e}")
            E = 0.0
        
        # Step
        r = max(search_radius * (0.8 ** epoch), 1e-4)
        dt = max(0.5 * (0.85 ** epoch), 0.01)
        
        params = qlto.run_walk(params, k_steps=2, delta_t=dt, search_radius=r, layer=False, gradient_reuse=True, coherence=True)
        
        # Get sensing diagnostics (Paper 1: "monitor if layer is trained well")
        diag = qlto.get_sensing_diagnostics()
        
        # Record for visualization
        if visualizer:
            visualizer.record(
                epoch=epoch,
                activation_rate=diag['mean_activation'],
                energy=E,
                entropy=diag['mean_entropy']
            )
        
        print(f"Epoch {epoch+1:02d} | E: {E:+.4f} | Act: {diag['mean_activation']:.1%} | "
              f"H: {diag['mean_entropy']:.2f} | Quality: {diag['training_quality']:10s} | NEFV: {qlto.nefv}")
        
        # if E < -5.5:
        #     print(">>> Converged!")
        #     break
    
    print("=" * 70)
    print(f"Total Time: {time.time() - start_time:.2f}s")
    
    # Print final diagnostics summary
    print("\nFinal Sensing Diagnostics:")
    final_diag = qlto.get_sensing_diagnostics()
    print(f"  Mean Activation Rate: {final_diag['mean_activation']:.1%}")
    print(f"  Activation Range: [{final_diag['min_activation']:.1%}, {final_diag['max_activation']:.1%}]")
    print(f"  Mean Entropy: {final_diag['mean_entropy']:.3f}")
    print(f"  Training Quality: {final_diag['training_quality']}")
    
    # Generate visualizations
    if visualizer:
        print("\nGenerating visualizations...")
        visualizer.generate_2d_summary(filename="qlto_sensing_summary.png")
        visualizer.generate_3d_fractal(filename="qlto_fractal_3d.html")