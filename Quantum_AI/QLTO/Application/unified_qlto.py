"""
unified_qlto.py: Unified Gradient-Walk Circuit

Key Innovation:
- Gradient sensing via commutator [G,H] evolution is built INTO the circuit
- No separate gradient computation needed
- Single circuit execution per epoch → O(1) instead of O(L)

The Idea:
∂E/∂θ = ⟨ψ|i[G,H]|ψ⟩

Since i[G,H] is Hermitian, we can evolve under it:
  U_sense = exp(-it · i[G,H])

When controlled by ancilla, this encodes gradient as phase!
Then we use this phase directly for the walk without measuring.

Author: QLTO Research Team
"""

import numpy as np
from typing import Optional, List

# Qiskit Imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import EfficientSU2, RXGate, RYGate, RZGate, CXGate, PauliEvolutionGate
    from qiskit.synthesis import LieTrotter
    from qiskit.quantum_info import SparsePauliOp, Operator
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    QISKIT_AVAILABLE = True
except ImportError:
    print("WARNING: Qiskit not found.")
    QISKIT_AVAILABLE = False


class BaseEstimator:
    """V2 Estimator wrapper."""
    def __init__(self, backend=None):
        method = 'automatic'
        if backend and hasattr(backend, 'options'):
            method = getattr(backend.options, 'method', 'automatic')
        self._estimator = AerEstimator(options={'backend_options': {'method': method}})
        
    def run(self, pubs, **kwargs):
        return self._estimator.run(pubs, **kwargs)


class CommutatorEvolution:
    """
    Builds evolution under commutator [G, H].
    
    For generator G and Hamiltonian H:
      C = i[G, H] = i(GH - HG)    ← Hermitian!
      
    Evolution U = exp(-it·C) encodes gradient ⟨C⟩ as phase.
    """
    
    @staticmethod
    def compute_commutator(generator: SparsePauliOp, hamiltonian: SparsePauliOp) -> SparsePauliOp:
        """
        Compute i[G, H] = i(GH - HG).
        
        This is Hermitian (can verify: (i[G,H])† = -i[H†,G†] = -i[H,G] = i[G,H]).
        """
        # GH - HG
        commutator = 1j * (generator @ hamiltonian - hamiltonian @ generator)
        commutator = commutator.simplify()
        
        # Ensure real coefficients (should be, since i[G,H] is Hermitian)
        real_coeffs = np.real(commutator.coeffs)
        return SparsePauliOp(commutator.paulis, real_coeffs)
    
    @staticmethod
    def build_commutator_evolution(
        commutator: SparsePauliOp, 
        time: float,
        trotter_reps: int = 1
    ) -> PauliEvolutionGate:
        """
        Build evolution gate exp(-it·[G,H]).
        """
        trotter = LieTrotter(reps=trotter_reps)
        return PauliEvolutionGate(commutator, time=time, synthesis=trotter)


class UnifiedQLTO:
    """
    Unified Gradient-Walk Quantum Optimizer.
    
    Instead of:
      1. Compute gradient (L circuits)
      2. Execute walk (1 circuit)
      
    We do:
      1. Single circuit with gradient sensing + walk combined
      
    The gradient is sensed via controlled evolution under [G,H],
    then used directly to bias the walk via controlled rotations.
    """
    
    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        n_iterations: int = 5,
        bits_per_param: int = 1,
        shot_budget: int = 4096,
        backend=None,
    ):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required.")
        
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.n_iterations = n_iterations
        self.bits_per_param = bits_per_param
        self.shot_budget = shot_budget
        
        if backend is None:
            self.backend = AerSimulator(method='matrix_product_state')
        else:
            self.backend = backend
            
        self.estimator = BaseEstimator(self.backend)
        
        # Precompute generators and commutators
        self._setup_generators()
        
        self.nefv = 0
        self.last_circuit_depth = 0
        self.max_circuit_depth = 0
    
    def _setup_generators(self):
        """Extract generators from ansatz and compute commutators."""
        self.generators = []
        self.commutators = []
        
        # Use original ansatz (not decomposed) to get parameters
        param_list = list(self.ansatz.parameters)
        n_qubits = self.ansatz.num_qubits
        
        # For each parameter, find its generator
        for p_idx, param in enumerate(param_list):
            # Check each instruction for this parameter
            for instr in self.ansatz.data:
                op = instr.operation
                if not hasattr(op, 'params') or len(op.params) != 1:
                    continue
                    
                # Check if this is the instruction for our parameter
                if op.params[0] is not param:
                    continue
                
                # Get qubit index
                qubit_idx = self.ansatz.find_bit(instr.qubits[0]).index
                
                # Determine generator based on gate name
                gate_name = op.name.lower()
                pauli_str = ['I'] * n_qubits
                
                if 'ry' in gate_name or gate_name == 'ry':
                    pauli_str[qubit_idx] = 'Y'
                elif 'rz' in gate_name or gate_name == 'rz':
                    pauli_str[qubit_idx] = 'Z'
                elif 'rx' in gate_name or gate_name == 'rx':
                    pauli_str[qubit_idx] = 'X'
                else:
                    continue
                
                # Generator = 0.5 * Pauli (for Rx(θ) = exp(-iθ/2 X))
                # Reverse for Qiskit's qubit ordering
                pauli_qiskit = ''.join(reversed(pauli_str))
                gen = SparsePauliOp(pauli_qiskit, 0.5)
                self.generators.append((p_idx, gen))
                
                # Compute commutator [G, H]
                comm = CommutatorEvolution.compute_commutator(gen, self.hamiltonian)
                if not np.allclose(comm.coeffs, 0):
                    self.commutators.append((p_idx, comm))
                break  # Found this parameter
    
    def _apply_gate(self, qc, op, angle, target):
        """Apply parameterized gate based on type."""
        if isinstance(op, RYGate):
            qc.ry(angle, target)
        elif isinstance(op, RZGate):
            qc.rz(angle, target)
        elif isinstance(op, RXGate):
            qc.rx(angle, target)
            
    def _apply_controlled_gate(self, qc, op, angle, ctrl, target):
        """Apply controlled parameterized gate."""
        if isinstance(op, RYGate):
            qc.append(RYGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RZGate):
            qc.append(RZGate(angle).control(1), [ctrl, target])
        elif isinstance(op, RXGate):
            qc.append(RXGate(angle).control(1), [ctrl, target])
    
    def build_w_gate(self, param_reg, sys_reg, center_params, search_radius):
        """Build W-gate for parameter encoding."""
        qc = QuantumCircuit(param_reg, sys_reg, name="W_Gate")
        decomp = self.ansatz.decompose()
        param_order = list(self.ansatz.parameters)
        
        for instr in decomp.data:
            op = instr.operation
            if len(op.params) == 1 and isinstance(op.params[0], Parameter):
                p_obj = op.params[0]
                try:
                    p_idx = param_order.index(p_obj)
                except ValueError:
                    continue
                
                sys_q_idx = decomp.find_bit(instr.qubits[0]).index
                target_qubit = sys_reg[sys_q_idx]
                
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
    
    def run_unified(
        self,
        center_params: np.ndarray,
        search_radius: float = 0.5,
        delta_t: float = 0.3,
    ) -> np.ndarray:
        """
        Execute unified gradient-walk circuit.
        
        Single circuit with:
        1. Parameter superposition via W-gate
        2. Gradient sensing via commutator evolution (per iteration)
        3. Walk biased by gradient phase (per iteration)
        4. Single measurement at end
        
        Returns:
            new_params: Updated parameter values
        """
        n_params = len(center_params)
        n_p_qubits = n_params * self.bits_per_param
        
        # Registers
        qr_anc = AncillaRegister(1, 'anc')
        qr_param = QuantumRegister(n_p_qubits, 'param')
        qr_sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cr_param = ClassicalRegister(n_p_qubits, 'cr_param')
        cr_anc = ClassicalRegister(1, 'cr_anc')
        
        qc = QuantumCircuit(qr_anc, qr_param, qr_sys, cr_param, cr_anc)
        
        # Initialize
        qc.h(qr_anc)
        qc.h(qr_param)
        
        # W-gate: encode parameters
        w_gate = self.build_w_gate(qr_param, qr_sys, center_params, search_radius)
        qc.append(w_gate, list(qr_param) + list(qr_sys))
        
        # Trotter for evolution
        trotter = LieTrotter(reps=1)
        
        # === UNIFIED SENSE + WALK ITERATIONS ===
        for iteration in range(self.n_iterations):
            iter_progress = (iteration + 0.5) / self.n_iterations
            sensing_time = delta_t * (1.0 - 0.3 * iter_progress)
            
            # --- GRADIENT SENSING via Commutator Evolution ---
            # For each parameter, evolve under controlled-[G,H]
            # This accumulates phase ∝ gradient on the ancilla
            
            for p_idx, comm in self.commutators:
                if np.allclose(comm.coeffs, 0):
                    continue  # Skip zero commutators
                
                # Controlled evolution under [G, H]
                # Phase kickback: |0⟩|ψ⟩ + exp(i⟨[G,H]⟩τ)|1⟩|ψ⟩
                comm_evo = PauliEvolutionGate(comm, time=sensing_time, synthesis=trotter)
                qc.append(comm_evo.control(1), [qr_anc[0]] + list(qr_sys))
            
            # --- WALK PHASE: Use accumulated gradient phase ---
            # The ancilla now has phase ∝ Σ∂E/∂θᵢ
            # Use this to bias the parameter walk
            
            res_scale = 1.0 / np.sqrt(self.bits_per_param)
            gamma = iter_progress * np.pi * delta_t * res_scale
            beta = (1 - iter_progress) * np.pi * delta_t
            
            for i in range(n_params):
                for b in range(self.bits_per_param):
                    q_idx = i * self.bits_per_param + b
                    
                    # Drift: ancilla phase controls direction
                    drift_weight = (2.0 ** b) / (2.0 ** (self.bits_per_param - 1))
                    qc.crz(gamma * drift_weight, qr_anc[0], qr_param[q_idx])
                    
                    # Mixer: exploration
                    mix_weight = 1.0 / (2.0 ** b)
                    qc.crx(beta * mix_weight, qr_anc[0], qr_param[q_idx])
            
            # Partial Hadamard for correlation (not full collapse)
            if iteration < self.n_iterations - 1:
                qc.ry(np.pi / 4, qr_anc[0])
        
        # Final correlation
        qc.h(qr_anc[0])
        
        # Inverse W-gate
        qc.append(w_gate.inverse(), list(qr_param) + list(qr_sys))
        
        # Measure
        qc.measure(qr_param, cr_param)
        qc.measure(qr_anc, cr_anc)
        
        # Execute - SINGLE CIRCUIT!
        t_qc = transpile(qc.decompose(reps=4), self.backend)
        self.last_circuit_depth = t_qc.depth()
        self.max_circuit_depth = max(self.max_circuit_depth, self.last_circuit_depth)
        
        result = self.backend.run(t_qc, shots=self.shot_budget).result()
        counts = result.get_counts()
        self.nefv += 1  # O(1) per epoch!
        
        return self._decode_result(counts, center_params, search_radius, n_params)
    
    def _decode_result(self, counts, center_params, radius, n_params):
        """Decode measurement results to parameter values."""
        accumulated_params = np.zeros(n_params)
        total_weight = 0.0
        max_int = (2**self.bits_per_param) - 1
        
        # Filter for ancilla=1 outcomes (low energy)
        param_counts = {}
        for bitstr, count in counts.items():
            parts = bitstr.split(' ')
            if len(parts) == 2:
                anc_bit = parts[0][-1]
                param_bits = parts[1]
            else:
                anc_bit = parts[0][0]
                param_bits = parts[0][1:]
            
            if anc_bit == '1':
                param_counts[param_bits] = param_counts.get(param_bits, 0) + count
        
        if not param_counts:
            param_counts = {k.replace(' ', ''): v for k, v in counts.items()}
        
        for bitstr, count in param_counts.items():
            weight = count
            clean_str = bitstr.replace(" ", "")
            current_val = np.zeros(n_params)
            
            expected_len = n_params * self.bits_per_param
            if len(clean_str) < expected_len:
                clean_str = clean_str.zfill(expected_len)
            elif len(clean_str) > expected_len:
                clean_str = clean_str[-expected_len:]
            
            rev_str = clean_str[::-1]
            for i in range(n_params):
                start = i * self.bits_per_param
                end = start + self.bits_per_param
                p_bits = rev_str[start:end] if end <= len(rev_str) else '0' * self.bits_per_param
                
                val_int = 0
                for b_idx, bit in enumerate(p_bits):
                    if bit == '1':
                        val_int += 2**b_idx
                
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


# --- Benchmark Wrapper ---

class UnifiedQLTO_Wrapper:
    """Wrapper for use in benchmark.py"""
    
    def __init__(
        self,
        ansatz,
        hamiltonian,
        n_iterations: int = 5,
        bits_per_param: int = 1,
        shot_budget: int = 4096,
        backend=None,
    ):
        self.optimizer = UnifiedQLTO(
            ansatz, hamiltonian,
            n_iterations=n_iterations,
            bits_per_param=bits_per_param,
            shot_budget=shot_budget,
            backend=backend,
        )
        self.estimator = self.optimizer.estimator
        self.epoch = 0
        
    @property
    def nefv(self):
        return self.optimizer.nefv
    
    @property
    def circuit_depth(self):
        return self.optimizer.last_circuit_depth
    
    @property
    def max_circuit_depth(self):
        return self.optimizer.max_circuit_depth
    
    def step(self, params):
        """Execute one epoch of unified optimization."""
        self.epoch += 1
        r = max(0.6 * (0.9 ** (self.epoch - 1)), 1e-4)
        dt = max(0.4 * (0.95 ** self.epoch), 0.01)
        return self.optimizer.run_unified(params, search_radius=r, delta_t=dt)


# --- Test ---

if __name__ == "__main__":
    print("=" * 60)
    print("  Unified QLTO - Gradient Sensing in Circuit")
    print("  O(1) per epoch via commutator evolution")
    print("=" * 60)
    
    from qiskit.circuit.library import EfficientSU2
    
    # Test problem - DON'T decompose! We need Parameter objects
    N = 4
    ops = [("ZIZI", 1.0), ("IZIZ", 1.0), ("XXII", 0.5), ("IIXX", 0.5)]
    H = SparsePauliOp.from_list(ops)
    ansatz = EfficientSU2(N, reps=1)  # Keep Parameter objects!
    
    print(f"Problem: {N} Qubits, {ansatz.num_parameters} Parameters")
    print(f"Commutator evolution replaces {len(ansatz.parameters)} gradient circuits")
    
    optimizer = UnifiedQLTO(ansatz, H, n_iterations=5, bits_per_param=1)
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    
    # Check commutators were computed
    print(f"Found {len(optimizer.commutators)} non-trivial commutators")
    
    from qiskit.primitives import StatevectorEstimator
    ref_est = StatevectorEstimator()
    
    print("\nStarting optimization...")
    for epoch in range(10):
        job = ref_est.run([(ansatz.assign_parameters(params), H)])
        E = float(job.result()[0].data.evs)
        
        print(f"Epoch {epoch+1:02d} | E: {E:+.4f} | NEFV: {optimizer.nefv} | Depth: {optimizer.last_circuit_depth}")
        
        params = optimizer.run_unified(params, search_radius=0.5, delta_t=0.3)
    
    print(f"\n=== Summary ===")
    print(f"Total NEFV: {optimizer.nefv} (O(1) per epoch!)")
    print(f"Max Circuit Depth: {optimizer.max_circuit_depth}")
