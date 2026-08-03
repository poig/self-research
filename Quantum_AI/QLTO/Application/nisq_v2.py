"""
nisq_v2.py: Riemannian Coherent QLTO (The Unified Architecture) - MPS Fix

This implementation merges the "Sandwich" Quantum Walk from 'nisq.py' with the 
rigorous Commuting-Block Geometry. It has been patched to support:
1. Matrix Product State (MPS) simulation for larger systems/entanglement.
2. Robust V2 Primitive Wrappers that correctly handle parameter binding.
3. Fixed interaction with geometry engines.

Author: Tan Jun Liang
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional

# Qiskit Imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
    from qiskit.circuit import Parameter, ParameterExpression
    from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, RGate, CXGate, PauliEvolutionGate, PhaseGate, QFT
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
    def __init__(self, backend=None, precision=0.0):
        # Configure options for AerEstimatorV2
        method = 'automatic'
        if backend and hasattr(backend, 'options'):
            method = getattr(backend.options, 'method', 'automatic')

        # FAIRNESS FIX. The comment below used to say that passing 1/sqrt(shots)
        # "makes the gradient cost precision the way hardware would". It does not.
        # Aer's EstimatorV2 with default_precision=p returns the EXACT expectation
        # plus Gaussian noise of standard deviation p - measured std/p = 1.00,
        # 0.94, 1.06 over p spanning 18x while Var(H) = 12.03, i.e. completely
        # blind to Var(H) and to the number of measurement settings.
        # On Heisenberg N=4 that gave V2 gradients with std 0.011 where honest
        # 8192-shot sampling gives 0.066 - 6x the precision, ~36x the effective
        # shots - while V3 sampled for real. Every V2 row in the benchmark was
        # inflated by that, and V2-vs-V3 is the comparison the notes discuss most.
        # BackendEstimatorV2 on AerSimulator actually samples, allocating
        # 1/precision^2 shots per qubit-wise-commuting group.
        opts = {'backend_options': {'method': method}}
        if precision:
            from qiskit.primitives import BackendEstimatorV2
            from qiskit_aer import AerSimulator as _AerSim
            self._estimator = BackendEstimatorV2(
                backend=_AerSim(method=method if method != 'automatic'
                                else 'statevector'),
                options={'default_precision': float(precision)})
        else:
            self._estimator = AerEstimator(options=opts)
        
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

# --- Criticality Sensor (The Stomach) ---
class CriticalitySensor:
    def __init__(self, target_dim=1.5, reaction_rate=0.05):
        self.target = target_dim
        self.rate = reaction_rate
        self.history = []

    def measure_dimension(self, diag):
        """
        Maps Normalized Entropy (0..1) to Fractal Dimension (1..3).
        S=0 (Crystal) -> D=1
        S=0.5 (Edge)  -> D=2
        S=1 (Gas)     -> D=3
        """
        if not diag or 'mean_entropy' not in diag:
            return 1.5 # Neutral expectation
            
        S = diag['mean_entropy'] # Normalized Entropy [0,1]
        D = 1.0 + 2.0 * S 
        return D

    def update(self, current_temp, diag):
        D = self.measure_dimension(diag)
        self.history.append(D)
        
        # Homeostasis Logic
        # If D < Target (Too Ordered/Ice) -> Heat Up (Increase T)
        # If D > Target (Too Chaotic/Gas) -> Cool Down (Decrease T)
        error = self.target - D
        
        # Feedback Scaling
        factor = 1.0 + self.rate * error
        
        # Safety Clamps to prevent explosion/collapse
        factor = np.clip(factor, 0.8, 1.2) 
        
        new_temp = current_temp * factor
        new_temp = np.clip(new_temp, 1e-4, 2.0)
        
        status = "COOLING" if factor < 1.0 else "HEATING"
        print(f"  [Stomach] {status}: D={D:.2f} (Err {error:.2f}). Temp: {current_temp:.4f} -> {new_temp:.4f}")
        
        return new_temp

# --- Core Logic ---

class RiemannianQLTO:
    def __init__(self, ansatz, hamiltonian, bits_per_param=1, shot_budget=8192, use_fim=False, num_ancillas=2, backend=None, fim_full = False, use_qpe_sensing=False, precision=0.0, sv_max_qubits=26):
        if not QISKIT_AVAILABLE: raise RuntimeError("Qiskit required.")
        
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.bits_per_param = bits_per_param
        self.requested_num_ancillas = num_ancillas
        self.use_qpe_sensing = use_qpe_sensing
        self.num_ancillas = max(1, num_ancillas) if use_qpe_sensing else 1
        self.shot_budget = shot_budget
        
        # Engines first: the backend choice below needs the block sizes.
        if ENGINES_AVAILABLE:
            # Paper 6: Efficient Metric Sensing
            self.use_fim = use_fim
            if use_fim:
                self.fim_engine = CommutingBlockFIM(ansatz, full=fim_full)
            else:
                self.fim_engine = None
            # Paper 2: Efficient Gradient Sensing
            self.grad_engine = CommutingBlockGradient(ansatz, hamiltonian)
            self.has_engines = True
            layers_info = len(self.fim_engine.layers) if self.fim_engine else len(self.grad_engine.layers)
            print(f"[Init] Riemannian Engines Online. Detected {layers_info} commuting layers. FIM={'ON' if use_fim else 'OFF'}")
        else:
            self.has_engines = False
            self.use_fim = False
            self.fim_engine = None
            self.grad_engine = None
            print("[Init] Engines missing. Reverting to blind heuristic walk.")

        # Backend by circuit width, not by system size.
        #
        # The walk circuit is 1 + param_qubits + N wide and maximally entangled
        # across the param<->sys cut, which is the worst case for MPS. Measured
        # on the equivalent V3 circuit at 13 qubits: 82s under
        # matrix_product_state against 0.26s under statevector, a 316x
        # difference. Defaulting to MPS "because the system is big" was costing
        # V2 hundreds of seconds per problem (Heisenberg N=6: 418s; MaxCut N=6:
        # 493s) for circuits a statevector handles in memory.
        #
        # Statevector cost is 2^n * 16 bytes: 21q = 34 MB, 26q = 1.1 GB.
        if backend is not None:
            self.backend = backend
        else:
            blocks = (self.fim_engine.layers if self.fim_engine
                      else (self.grad_engine.layers if self.grad_engine else []))
            max_block = max((len(l['params']) for l in blocks), default=ansatz.num_parameters)
            width = self.num_ancillas + max_block * bits_per_param + ansatz.num_qubits
            method = 'statevector' if width <= sv_max_qubits else 'matrix_product_state'
            self.backend = AerSimulator(method=method)
            print(f"[Init] widest walk circuit {width}q -> AerSimulator({method})")
        self.sim_width = (self.num_ancillas
                          + max((len(l['params']) for l in
                                 (self.fim_engine.layers if self.fim_engine
                                  else (self.grad_engine.layers if self.grad_engine else []))),
                                default=ansatz.num_parameters) * bits_per_param
                          + ansatz.num_qubits)

        self.estimator = BaseEstimator(self.backend, precision=precision)
        self.sampler = BaseSampler(self.backend)

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
        
        # QPE energy estimate (read directly from ancilla, no separate circuit needed)
        self.last_qpe_energy = None
        self._last_sensing_time = None  # Store for phase→energy conversion

        # Sensing must use the TRACELESS Hamiltonian. The evolution is
        # controlled, so an identity term c*I becomes a relative phase
        # e^{-i c tau} between the ancilla branches, attenuating the signal by
        # cos(c tau) and mixing in Re<U>; at c*tau = pi/2 it vanishes entirely.
        # Molecular Hamiltonians carry large constants (LiH: c = -7.883), and
        # this is why V2's final energy drifted 0.15-0.21 below its own best on
        # H2 and LiH while staying stable on the spin models.
        _ident = 0.0
        _p, _c = [], []
        for _pauli, _coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if set(_pauli.to_label()) == {"I"}:
                _ident += complex(_coeff).real
            else:
                _p.append(_pauli.to_label()); _c.append(_coeff)
        self.h_offset = _ident
        self.H_sense = (SparsePauliOp(_p, _c).simplify() if _p
                        else SparsePauliOp("I" * hamiltonian.num_qubits, [0.0]))

        H_mat = hamiltonian.to_matrix()
        self.H_norm = float(np.linalg.norm(H_mat, ord=2))   # spectral norm
        msb_scale  = 2 ** max(num_ancillas - 1, 0)
        self.tau_0 = np.pi / (msb_scale * self.H_norm + 1e-12)
        if self.use_qpe_sensing:
            print(f"[Init] ‖H‖_2 = {self.H_norm:.4f}  →  τ₀ = {self.tau_0:.4f} rad "
                  f"(alias-free range ±{np.pi/self.tau_0:.2f})")
        else:
            print(f"[Init] Legacy single-ancilla sensing active. ‖H‖_2 = {self.H_norm:.4f}")

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

        # Standard Linear Mapping: [Center-R, Center+R]
        # range = 2*R
        # step = range / (2^bits - 1)
        # val = min_val + integer * step
        full_range = 2 * search_radius
        max_int = (2**self.bits_per_param) - 1
        if max_int == 0: step_size = 0
        else: step_size = full_range / max_int

        # === OPTIMIZATION: FAST PATH (Global Mode) ===
        if active_indices is None:
            # Original fast loop (No dictionary lookups, no 'if' checks)
            for instr in decomp.data:
                op = instr.operation
                p_idx = self._parameterised_index(op, param_order)
                if p_idx is not None:
                    sys_q_idx = decomp.find_bit(instr.qubits[0]).index
                    target_qubit = sys_reg[sys_q_idx]
                    
                    # 1. Apply Base (Min Value)
                    min_val = center_params[p_idx] - search_radius
                    self._apply_gate(qc, op, min_val, target_qubit) 
                    
                    # 2. Apply Increments (Bit-Weighted)
                    start_bit = p_idx * self.bits_per_param
                    for b in range(self.bits_per_param):
                        ctrl_qubit = param_reg[start_bit + b]
                        angle = step_size * (2**b)
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
            p_idx = self._parameterised_index(op, param_order)
            if p_idx is not None:
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
                    for b in range(self.bits_per_param):
                        ctrl_qubit = param_reg[start_bit + b]
                        angle = step_size * (2**b)
                        self._apply_controlled_gate(qc, op, angle, ctrl_qubit, target_qubit)

                else:
                    # === FROZEN (CLASSICAL CONSTANT) ===
                    fixed_val = center_params[p_idx]
                    self._apply_gate(qc, op, fixed_val, target_qubit)
            
            elif isinstance(op, CXGate):
                q1 = decomp.find_bit(instr.qubits[0]).index
                q2 = decomp.find_bit(instr.qubits[1]).index
                qc.cx(sys_reg[q1], sys_reg[q2])
                
        return qc

    # Helper for gate application to keep code clean
    @staticmethod
    def _parameterised_index(op, param_order):
        """Index of the ansatz parameter this gate rotates, or None.

        Must not test `len(op.params) == 1`: `efficient_su2().decompose()` emits
        RGate(theta, phi), whose second parameter is a plain float. That test
        silently dropped every RY-derived rotation from the W-gate, so the walk
        searched a circuit missing half the ansatz.
        """
        if not op.params:
            return None
        first = op.params[0]
        if isinstance(first, Parameter):
            target = first
        elif isinstance(first, ParameterExpression) and first.parameters:
            free = list(first.parameters)
            if len(free) != 1:
                return None
            target = free[0]
        else:
            return None
        try:
            return param_order.index(target)
        except ValueError:
            return None

    def _apply_gate(self, qc, op, angle, target):
        if isinstance(op, RYGate): qc.ry(angle, target)
        elif isinstance(op, RZGate): qc.rz(angle, target)
        elif isinstance(op, RXGate): qc.rx(angle, target)
        elif isinstance(op, PhaseGate): qc.p(angle, target)
        elif isinstance(op, RGate): qc.r(angle, float(op.params[1]), target)
        else: raise TypeError(f"W-gate cannot encode parameterised gate '{op.name}'")

    def _apply_controlled_gate(self, qc, op, angle, ctrl, target):
        if isinstance(op, RYGate): qc.append(RYGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RZGate): qc.append(RZGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RXGate): qc.append(RXGate(angle).control(1), [ctrl, target])
        elif isinstance(op, PhaseGate): qc.append(PhaseGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RGate): qc.append(RGate(angle, float(op.params[1])).control(1), [ctrl, target])
        else: raise TypeError(f"W-gate cannot encode parameterised gate '{op.name}'")

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
        self.last_qpe_energy = None
        self._last_sensing_time = None
        
        # Precompute expensive terms ONCE per epoch
        if gradient_reuse:
            # 1. Compute Global Gradient ONCE (Cost: 2M-N circuits)
            global_grad = self.grad_engine.compute_gradient(self.estimator, center_params)
            self.nefv += self.grad_engine.get_nefv_cost()['actual_with_cnot']
        else:
            global_grad = None
        
        # 2. Compute Global FIM ONCE (Cost: L circuits) - REUSE across layers!
        if self.use_fim and self.fim_engine is not None:
            global_fim = self.fim_engine.compute_fim(self.estimator, center_params)
            self.nefv += self.fim_engine.get_nefv_cost()
        else:
            global_fim = None  # Skip FIM - use identity metric
        
        if layer and self.has_engines:
            
            current_params = center_params.copy()
            last_energy = float('nan')
            
            layers_to_use = self.fim_engine.layers if self.fim_engine else self.grad_engine.layers
            for i, layer_info in enumerate(layers_to_use):
                active_indices = layer_info['params']
                if not active_indices: continue

                current_params, last_energy = self._execute_walk(
                    current_params,
                    k_steps, delta_t, search_radius,
                    active_indices=active_indices,
                    precomputed_grad=self._layer_gradient(
                        global_grad, current_params, search_radius, active_indices),
                    precomputed_fim=global_fim,
                    coherence=coherence,
                    drift_gain=1.0/np.sqrt(search_radius)
                )
            return current_params, last_energy
        else:
            return self._execute_walk(
                center_params,
                k_steps, delta_t, search_radius,
                active_indices=None,
                precomputed_grad=self._layer_gradient(
                    global_grad, center_params, search_radius, None),
                precomputed_fim=global_fim,
                coherence=coherence,
                drift_gain=1.0/np.sqrt(search_radius)
            )
    
    def _layer_gradient(self, global_grad, center_params, search_radius, active_indices):
        """Gradient handed to one walk. V2 just forwards whatever run_walk
        precomputed; this is the single seam subclasses override to supply the
        gradient another way (see nisq_v3.RiemannianQLTOv3)."""
        return global_grad

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
            'training_quality': self._assess_training_quality(activation_rates, entropies),
            # landscape_avg_energy: QPE-weighted average over the 2^(n*k) search
            # superposition — NOT the variational energy at the decoded point.
            # Use ref_est.run([(ansatz, H, decoded_params)]) for true energy.
            'landscape_avg_energy': self.last_qpe_energy,
        }
    
    def _assess_training_quality(self, activation_rates, entropies) -> str:
        """
        Assess training quality based on sensing diagnostics.
        
        Paper 1 Interpretation:
        - Activation ~ 50%: Maximum entropy sensing (Best) → "excellent"
        - Activation > 60% or < 40%: Biased → "good"
        - Activation < 20% + High Entropy: Stuck → "exploring"
        
        Returns a qualitative assessment string.
        """
        if not activation_rates:
            return "unknown"
        
        mean_act = np.mean(activation_rates)
        mean_ent = np.mean(entropies)
        
        if 0.4 <= mean_act <= 0.6:
            return "excellent"
        elif mean_act > 0.2:
            return "good"
        elif mean_ent < 0.5:
            # Low activation + low entropy = converging to solution
            return "converging"
        else:
            return "exploring"
    
    def _execute_walk(self, center_params, k_steps, delta_t, radius, active_indices=None, precomputed_grad=None, precomputed_fim=None, coherence=False, drift_gain=1.5):
        """
        Internal worker that builds and runs the circuit for a specific subset.
        """
        self.last_qpe_energy = None
        self._last_sensing_time = None
        
        # 1. Determine Dimensions
        if active_indices is None:
            active_indices = list(range(len(center_params)))
            
        n_active = len(active_indices)
        n_p_qubits = n_active * self.bits_per_param
        layer_id = hash(tuple(active_indices)) % 1000  # Simple layer identifier
        qpe_mode = self.use_qpe_sensing
        sensing_ancillas = self.num_ancillas
        
        # --- OPTIMIZATION: FIM AND GRADIENT REUSE ---
        # 1. Metric: Use precomputed if available (CHEAP when reused!)
        if precomputed_fim is not None:
            metric_matrix = precomputed_fim
            metric_diag = np.diag(metric_matrix)
            metric_local = metric_diag[active_indices]
        else:
            # No FIM - use identity metric (uniform weights)
            metric_local = np.ones(n_active)
        
        # 2. Gradient: Use precomputed if available, otherwise compute (Expensive!)
        if precomputed_grad is not None:
            grad_full = precomputed_grad
        else:
            # Reached only when no caller supplied a gradient. Announce it: this
            # submits 2M-N circuits, and a silent fallback here is how a walk
            # that was supposed to be gradient-free ends up paying for one.
            print("  [QLTO] no precomputed gradient - falling back to the "
                  "CommutingBlockGradient engine (2M-N circuits).")
            grad_full = self.grad_engine.compute_gradient(self.estimator, center_params)
            self.nefv += self.grad_engine.get_nefv_cost()['actual_with_cnot']
            
        grad_local = grad_full[active_indices]
        
        # 3. Build Registers
        # Notice: param register is sized ONLY for active parameters!
        # FULL SENSING: Add ancilla measurement register
        qr_anc = AncillaRegister(sensing_ancillas, 'anc')
        qr_param = QuantumRegister(n_p_qubits, 'param')
        qr_sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cr_param = ClassicalRegister(n_p_qubits, 'cr_param')
        cr_anc = ClassicalRegister(sensing_ancillas, 'cr_anc')
        
        qc = QuantumCircuit(qr_anc, qr_param, qr_sys, cr_param, cr_anc)
        
        # 4. Initialization (Paper 1 Phase 1: Prepare |+⟩_A)
        qc.h(qr_anc)  # All ancillas in superposition
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
        if qpe_mode:
            # QPE mode: alias-free multi-ancilla sensing.
            base_sensing_time = min(self.tau_0, delta_t * np.pi)
            self._last_sensing_time = base_sensing_time

            for a in range(sensing_ancillas):
                # Scale time exponentially for QPE: 2^a
                # This provides the necessary phase wraps for Inverse QFT
                time_scale = 2 ** a
                t_a = base_sensing_time * time_scale

                # CRITICAL FIX: Trotter error scales as O(t^2/r).
                # Because time_scale is exponential, a single Trotter step (reps=1)
                # for the MSB destroys the unitary. We must scale reps with time!
                reps = max(1, int(time_scale * 2))
                trotter = LieTrotter(reps=reps)

                evo_sense = PauliEvolutionGate(self.H_sense, time=t_a, synthesis=trotter)

                if coherence:
                    # Controlled sensing: ancilla a accumulates phase ⟨cos(E(θ) * t_a)⟩
                    qc.append(evo_sense.control(1), [qr_anc[a]] + list(qr_sys))
                else:
                    # Non-coherent: just evolve system (less sensing power)
                    qc.append(evo_sense, qr_sys)

            # =====================================================
            # PHASE 1.5: DECODING (Inverse QFT)
            # Convert phase to binary representation of energy
            # =====================================================
            qft_inv = QFT(num_qubits=sensing_ancillas, inverse=True, do_swaps=True)
            qc.append(qft_inv, qr_anc)
        else:
            # Legacy mode: single-ancilla Hadamard-test sensing.
            sensing_time = delta_t * np.pi
            self._last_sensing_time = sensing_time
            trotter = LieTrotter(reps=1)
            evo_sense = PauliEvolutionGate(self.H_sense, time=sensing_time, synthesis=trotter)

            if coherence:
                qc.append(evo_sense.control(1), [qr_anc[0]] + list(qr_sys))
            else:
                qc.append(evo_sense, qr_sys)

        # 6. Walk Loop - Parameter space exploration
        # NOTE: No more Hamiltonian evolution here! 
        # The sensing is done, now we just diffuse in parameter space
        for step in range(k_steps):
            # Annealing schedule within walk
            s = (step + 0.5) / k_steps
            res_scale = 1.0 / np.sqrt(self.bits_per_param)
            gamma = s * np.pi * delta_t * res_scale  # Phase accumulation rate
            beta = (1 - s) * np.pi * delta_t         # Mixing strength
            
            # --- Drift (Linear/Bit-Weighted) ---
            # We want to increase the probability of '1' states if Gradient is positive (Phase Kickback)
            # Since High Bits have High Impact, they get High Kick.
            # Angle ~ Gradient * Weight * Gamma
            for i in range(n_active):
                g_ii = metric_local[i]
                grad_i = grad_local[i]
                # max(g_ii, 0) belt-and-braces: a sampled FIM diagonal can go
                # slightly negative and sqrt() would return NaN.
                ng_scale = np.clip(1.0 / (np.sqrt(max(g_ii, 0.0)) + 1e-6), 0.1, 5.0)
                scaled_grad = grad_i * ng_scale
                
                for b in range(self.bits_per_param):
                    q_idx = i * self.bits_per_param + b
                    
                    # Weight proportional to bit significance (2^b)
                    # We normalize so max kick is reasonable
                    # DYNAMIC GAIN: Passed from run_walk
                    weight = (2.0 ** b) / (2.0 ** self.bits_per_param)
                    direction_angle = scaled_grad * gamma * weight * np.pi * drift_gain
                    
                    if coherence:
                        if qpe_mode:
                            # QPE Coherent Control: Weight kicks by binary significance
                            # qr_anc[0] is the MSB, so it gets the highest weight
                            anc_norm = 2 ** sensing_ancillas
                            for a in range(sensing_ancillas):
                                bit_weight = (2 ** (sensing_ancillas - 1 - a)) / anc_norm
                                qc.crz(direction_angle * bit_weight, qr_anc[a], qr_param[q_idx])
                        else:
                            qc.crz(direction_angle, qr_anc[0], qr_param[q_idx])
                    else:
                        qc.rz(direction_angle, qr_param[q_idx])

            # --- Mixer (Diffusion) ---
            # OPTIMIZATION: Use individual CRX gates instead of grouped control
            for i in range(n_active):
                g_ii = metric_local[i]
                scale = np.clip(1.0 / (np.sqrt(max(g_ii, 0.0)) + 1e-6), 0.1, 5.0)
                
                for b in range(self.bits_per_param):
                    q_idx = i * self.bits_per_param + b
                    mix_weight = 1.0 / (2.0 ** b)
                    angle = beta * scale * mix_weight
                    
                    if coherence:
                        if qpe_mode:
                            # QPE Coherent Control: Weight mixer by binary significance
                            anc_norm = 2 ** sensing_ancillas
                            for a in range(sensing_ancillas):
                                bit_weight = (2 ** (sensing_ancillas - 1 - a)) / anc_norm
                                qc.crx(angle * bit_weight, qr_anc[a], qr_param[q_idx])
                        else:
                            qc.crx(angle, qr_anc[0], qr_param[q_idx])
                    else:
                        qc.rx(angle, qr_param[q_idx])
            
            # NOTE: True coherent integration means NO reset between steps.
            # The ancilla remains entangled, accumulating phase information
            # across all K steps. This matches paper claim:
            # "coherent integration (no intermediate reset)"
        
        if not qpe_mode:
            qc.h(qr_anc)  # Phase -> Population conversion
        
        # =====================================================
        # PHASE 2: INVERSE-W
        # The phase info has already been converted by QFT_inv
        # =====================================================
                    
        # 7. Measurement & Decode
        qc.append(w_gate.inverse(), list(qr_param) + list(qr_sys))
        
        # FULL SENSING: Measure both ancilla AND parameters
        qc.measure(qr_param, cr_param)
        qc.measure(qr_anc, cr_anc)  # Paper 1: "ancilla sensing"

        # print(qc.draw())
        
        t_qc = transpile(qc, self.backend, optimization_level=1, basis_gates=['u3', 'cx', 'id', 'rz', 'rx', 'h'])
        
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

        decoded_block, diagnostics = self._postprocess_walk_counts(
            counts,
            center_params,
            active_indices,
            radius,
            n_active,
            n_p_qubits,
        )
        self.layer_diagnostics[layer_id] = diagnostics
        self.last_activation_rate = diagnostics['activation_rate']
        self.last_entropy = diagnostics['normalized_entropy']
        self.last_qpe_energy = diagnostics.get('landscape_avg_energy')
        
        # Update the full vector
        new_params = center_params.copy()
        new_params[active_indices] = decoded_block

        # ── INLINE POINT ENERGY ──────────────────────────────────────────────
        # ⟨ψ(θ_decoded)|H|ψ(θ_decoded)⟩ computed on the decoded parameter point
        # using the statevector estimator already in self.estimator.
        # This replaces the separate ref_est call in main() and gives the true
        # variational energy without a second circuit submission.
        try:
            _pub = (self.ansatz, self.hamiltonian, new_params)
            point_energy = float(self.estimator.run([_pub]).result()[0].data.evs)
            self.nefv += 1   # a submitted circuit like any other
        except Exception as _e:
            # Graceful fallback: keep landscape average if statevector fails
            point_energy = self.last_qpe_energy if self.last_qpe_energy is not None else float('nan')

        return new_params, point_energy

    def _postprocess_walk_counts(self, counts, center_params, active_indices, radius, n_active, n_p_qubits):
        """
        Convert raw measurement counts into a parameter update and diagnostics.

        Legacy mode uses the original single-ancilla Trap-Diffusion decode.
        QPE mode uses the newer multi-ancilla weighted decode.
        """
        total_shots = sum(counts.values())
        ancilla_one_count = 0

        if self.use_qpe_sensing:
            param_counts_weighted = {}

            for bitstr, count in counts.items():
                parts = bitstr.split(' ')

                if len(parts) == 2:
                    anc_bits_str = parts[0]
                    param_bits = parts[1]

                    try:
                        val = int(anc_bits_str[::-1], 2)
                    except ValueError:
                        val = 0

                    norm_factor = 2 ** self.num_ancillas
                    activation = 0.5 if norm_factor == 0 else val / norm_factor

                    ancilla_one_count += activation * count
                    param_counts_weighted[param_bits] = param_counts_weighted.get(param_bits, 0) + (count * activation)

            activation_rate = ancilla_one_count / total_shots if total_shots > 0 else 0.0

            qpe_energy_accum = 0.0
            qpe_weight = 0.0
            for bitstr, count in counts.items():
                parts = bitstr.split(' ')
                if len(parts) == 2:
                    anc_str = parts[0]
                    try:
                        anc_val = int(anc_str[::-1], 2)
                    except ValueError:
                        continue

                    n_anc = len(anc_str)
                    phase_fraction = anc_val / (2 ** n_anc)
                    if phase_fraction > 0.5:
                        phase_fraction -= 1.0

                    if self._last_sensing_time and self._last_sensing_time > 1e-10:
                        energy_est = 2 * np.pi * phase_fraction / self._last_sensing_time
                    else:
                        energy_est = phase_fraction

                    qpe_energy_accum += energy_est * count
                    qpe_weight += count

            landscape_avg_energy = qpe_energy_accum / qpe_weight if qpe_weight > 0 else None

            all_probs = np.array(list(counts.values())) / total_shots if total_shots > 0 else np.array([])
            entropy = -np.sum(all_probs * np.log2(all_probs + 1e-12)) if total_shots > 0 else 0.0
            max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            diagnostics = {
                'activation_rate': activation_rate,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'move_count': ancilla_one_count,
                'stay_count': total_shots - ancilla_one_count,
                'n_params': n_active,
                # NOTE: This is NOT the variational energy at θ_current.
                # It is the QPE-weighted landscape average over the full 2^(n*k)
                # parameter superposition in the search window. Kept for research
                # purposes only — do not use for convergence tracking.
                'landscape_avg_energy': landscape_avg_energy,
            }

            if len(param_counts_weighted) > 0 and activation_rate > 0.05:
                decoded_block = self._decode_result_with_ancilla(
                    param_counts_weighted, center_params[active_indices], radius, n_active
                )
            else:
                decoded_block = self._decode_result_with_ancilla(
                    {k[1:] if len(k) > n_p_qubits else k: v for k, v in counts.items()},
                    center_params[active_indices], radius, n_active
                )
                decoded_block = center_params[active_indices] + 0.3 * (decoded_block - center_params[active_indices])

            return decoded_block, diagnostics

        param_counts_move = {}
        param_counts_stay = {}

        for bitstr, count in counts.items():
            parts = bitstr.split(' ')

            if len(parts) == 2:
                anc_bit = parts[0][-1]
                param_bits = parts[1]
            elif len(parts) == 1:
                clean = parts[0]
                anc_bit = clean[0]
                param_bits = clean[1:]
            else:
                anc_bit = '0'
                param_bits = bitstr.replace(' ', '')

            if anc_bit == '1':
                ancilla_one_count += count
                param_counts_move[param_bits] = param_counts_move.get(param_bits, 0) + count
            else:
                param_counts_stay[param_bits] = param_counts_stay.get(param_bits, 0) + count

        activation_rate = ancilla_one_count / total_shots if total_shots > 0 else 0.0
        all_probs = np.array(list(counts.values())) / total_shots if total_shots > 0 else np.array([])
        entropy = -np.sum(all_probs * np.log2(all_probs + 1e-12)) if total_shots > 0 else 0.0
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        diagnostics = {
            'activation_rate': activation_rate,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'move_count': ancilla_one_count,
            'stay_count': total_shots - ancilla_one_count,
            'n_params': n_active,
            'landscape_avg_energy': None,
        }

        if len(param_counts_move) > 0 and activation_rate > 0.05:
            decoded_block = self._decode_result_with_ancilla(
                param_counts_move, center_params[active_indices], radius, n_active
            )
        else:
            decoded_block = self._decode_result_with_ancilla(
                {k[1:] if len(k) > n_p_qubits else k: v for k, v in counts.items()},
                center_params[active_indices], radius, n_active
            )
            decoded_block = center_params[active_indices] + 0.3 * (decoded_block - center_params[active_indices])

        return decoded_block, diagnostics
    
    def _decode_result_with_ancilla(self, param_counts, center_params, radius, n_params):
        """
        Decode parameter values from measurement counts using Linear Mapping.
        """
        accumulated_params = np.zeros(n_params)
        total_weight = 0.0
        max_int = (2**self.bits_per_param) - 1
        
        for bitstr, count in param_counts.items():
            weight = count
            clean_str = bitstr.replace(" ", "")
            current_val = np.zeros(n_params)
            
            expected_len = n_params * self.bits_per_param
            if len(clean_str) < expected_len:
                clean_str = clean_str.zfill(expected_len)
            elif len(clean_str) > expected_len:
                clean_str = clean_str[-expected_len:]
            
            # Linear Decode (Matches build_w_gate)
            # Qiskit Bitstring is Little Endian: [qn ... q0]
            # rev_str gives [q0, q1 ...] so p_bits matches range(bits) loop
            rev_str = clean_str[::-1]
            
            for i in range(n_params):
                start = i * self.bits_per_param
                end = start + self.bits_per_param
                p_bits = rev_str[start:end] if end <= len(rev_str) else '0' * self.bits_per_param
                
                val_int = 0
                for b_idx, bit in enumerate(p_bits):
                    if bit == '1': val_int += 2**b_idx
                    
                # Map [0, MaxInt] -> [Center-R, Center+R]
                if max_int > 0: norm = val_int / max_int
                else: norm = 0.5
                
                p_min = center_params[i] - radius
                p_max = center_params[i] + radius
                real_val = p_min + norm * (p_max - p_min)
                
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
    # H = force_heisenberg_hamiltonian(N)
    # EfficientSU2 is naturally structured into commuting blocks (Rotation layers)
    # Decompose to ensure it's compatible with Aer primitives
    ansatz = EfficientSU2(N, reps=2, entanglement='linear', su2_gates=['u3']).decompose()
    print(f"Ansatz Ops: {ansatz.count_ops()}")
    
    print(f"Problem: {N} Qubits, {ansatz.num_parameters} Parameters.")
    # Exact ground state energy for reference (computed classically)
    H_mat = H.to_matrix()
    exact_gs = float(np.min(np.linalg.eigvalsh(H_mat)))
    print(f"Exact GS energy: {exact_gs:.6f}")
    
    # 2. Initialize Optimizer
    # bits_per_param=1 is the quantum advantage regime:
    #   - QPE coherence feedback evaluates 2^n = 2^36 configurations per circuit
    #   - Each epoch: one deep coherent circuit replaces 2^36 classical evaluations
    #   - Increasing bits_per_param shifts toward landscape integration (less quantum speedup)
    qlto = RiemannianQLTO(ansatz, H, bits_per_param=1, shot_budget=8192, use_fim=False, num_ancillas=4)
    print(f"{qlto.bits_per_param} bits per param | {qlto.num_ancillas} Sensing Ancillas")
    
    # 3. No separate ref_est needed — run_walk now returns (params, E) directly.
    # ⟨ψ(θ)|H|ψ(θ)⟩ is computed inline after each decode step using the
    # calibrated statevector estimator, aligned with the alias-free τ₀.
    
    # 4. Run Loop
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    E = float('nan')  # will be overwritten each epoch
    
    # Initialize visualizer if enabled
    visualizer = None
    if VISUALIZATION and VIZ_AVAILABLE:
        visualizer = AncillaVisualizer(output_dir="./figures")
        print("[Viz] Ancilla visualizer enabled. Will generate fractal at end.")
    
    # Initialize Criticality Sensor
    sensor = CriticalitySensor(target_dim=1.5)
    current_radius = 0.95
    last_diag = None

    print("\nStarting Optimization with Homeostatic Loop...")
    print("=" * 70)
    start_time = time.time()
    
    for epoch in range(30):
        # Homeostatic update from previous epoch's sensing diagnostics
        if last_diag:
            current_radius = sensor.update(current_radius, last_diag)
            
        dt = max(current_radius * 1.5, 0.05)
        r = current_radius
        
        # QPE coherent walk: updates params via quantum coherence feedback.
        # Internally, the QPE sensing circuit evaluates 2^(n * bits_per_param)
        # configurations simultaneously via superposition, then the walk
        # amplifies the lower-energy vertex — genuine quantum parallel search.
        params, E = qlto.run_walk(
            params, k_steps=2, delta_t=dt, search_radius=r,
            layer=False, gradient_reuse=True, coherence=True
        )
        
        diag = qlto.get_sensing_diagnostics()
        last_diag = diag
        landscape_E = diag.get('landscape_avg_energy')
        qpe_str = f"{landscape_E:+.4f}" if landscape_E is not None else "  n/a  "
        
        # Record for visualization
        if visualizer:
            visualizer.record(
                epoch=epoch,
                activation_rate=diag['mean_activation'],
                energy=E,
                entropy=diag['mean_entropy']
            )
        
        print(f"Epoch {epoch+1:02d} | E_var: {E:+.6f} | E_qpe: {qpe_str} | "
              f"Act: {diag['mean_activation']:.1%} | "
              f"H: {diag['mean_entropy']:.2f} | Quality: {diag['training_quality']:10s} | NEFV: {qlto.nefv}")
    
    print("=" * 70)
    print(f"Total Time: {time.time() - start_time:.2f}s")
    
    # Print final diagnostics summary
    print("\nFinal Sensing Diagnostics:")
    final_diag = qlto.get_sensing_diagnostics()
    print(f"  Mean Activation Rate: {final_diag['mean_activation']:.1%}")
    print(f"  Activation Range: [{final_diag['min_activation']:.1%}, {final_diag['max_activation']:.1%}]")
    print(f"  Mean Entropy: {final_diag['mean_entropy']:.3f}")
    print(f"  Training Quality: {final_diag['training_quality']}")
    print(f"  ‖H‖_2 (spectral norm): {qlto.H_norm:.4f}")
    print(f"  τ₀ (alias-free sensing time): {qlto.tau_0:.4f} rad")
    print(f"  Final E_var  (decoded point ⟨ψ(θ)|H|ψ(θ)⟩): {E:+.6f}")
    lqpe = final_diag.get('landscape_avg_energy')
    print(f"  Final E_qpe  (QPE landscape average):        {lqpe:+.4f}" if lqpe else "  Final E_qpe: n/a")
    
    # Generate visualizations
    if visualizer:
        print("\nGenerating visualizations...")
        visualizer.generate_2d_summary(filename="qlto_sensing_summary.png")
        visualizer.generate_3d_fractal(filename="qlto_fractal_3d.html")