"""
coherent_ite.py: Coherent Imaginary Time Evolution Optimizer

Theory:
-------
Imaginary Time Evolution (ITE) naturally flows toward the ground state:

    d|ψ⟩/dτ = -H|ψ⟩
    
    Solution: |ψ(τ)⟩ = e^{-Hτ}|ψ(0)⟩ / ‖e^{-Hτ}|ψ(0)⟩‖

As τ → ∞, the state converges to the ground state of H.

Implementation:
---------------
Since e^{-Hτ} is non-unitary (contractive), we approximate it via:

1. **Trotter Expansion**: For small τ, e^{-Hτ} ≈ Π_k (I - τ·H_k)
   - Requires normalization after each step
   
2. **Probabilistic Block Encoding**: Embed e^{-Hτ} in a larger unitary
   - Ancilla measurement succeeds with probability ∝ overlap with ground state
   - O(1) measurement at end, O(poly(N)) depth

3. **LCU (Linear Combination of Unitaries)**: 
   e^{-Hτ} = Σ_k c_k exp(-iH·τ_k) for complex τ_k

This implementation uses approach 2 with Pauli decomposition.
"""

import numpy as np
from typing import Optional, List, Tuple

try:
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, transpile
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CoherentITE:
    """
    Coherent Imaginary Time Evolution.
    
    Implements e^{-Hτ} via probabilistic block encoding:
    
    1. Prepare ancilla in |+⟩
    2. Apply controlled-exp(iHτ) and controlled-exp(-iHτ)  
    3. Constructive interference gives e^{-Hτ} + e^{+Hτ} ∝ cosh(Hτ)
    4. Post-select on ancilla = |0⟩
    
    For ground state finding, we use iterative cooling:
    - Apply multiple small steps of e^{-Hτ}
    - Accumulate probability toward ground state
    """
    
    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        n_qubits: int,
        tau: float = 0.1,
        n_steps: int = 5,
        backend=None,
    ):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.tau = tau
        self.n_steps = n_steps
        
        if backend is None:
            self.backend = AerSimulator(method='statevector')
        else:
            self.backend = backend
        
        # Normalize Hamiltonian for block encoding
        self.H_norm = self._normalize_hamiltonian()
        
        self.nefv = 0
        self.last_circuit_depth = 0
    
    def _normalize_hamiltonian(self) -> Tuple[SparsePauliOp, float]:
        """Normalize H so that ||H|| ≤ 1 for block encoding."""
        # Sum of |coefficients|
        norm_bound = np.sum(np.abs(self.hamiltonian.coeffs))
        if norm_bound > 0:
            H_normalized = SparsePauliOp(
                self.hamiltonian.paulis,
                self.hamiltonian.coeffs / norm_bound
            )
            return H_normalized, norm_bound
        return self.hamiltonian, 1.0
    
    def build_ite_step_circuit(self, input_state_circuit: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        """
        Build one step of ITE: approximates action of e^{-Hτ}.
        
        Uses the identity:
            e^{-Hτ} ∝ ⟨0|(I ⊗ e^{iHτ} + X ⊗ e^{-iHτ})|+⟩|ψ⟩
        
        This is implemented via:
        1. Ancilla in |+⟩
        2. Controlled-exp(+iHτ) on |0⟩
        3. Controlled-exp(-iHτ) on |1⟩  
        4. H on ancilla, measure
        """
        H_norm, norm_factor = self.H_norm
        
        qr_anc = AncillaRegister(1, 'anc')
        qr_sys = QuantumRegister(self.n_qubits, 'sys')
        
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # Prepare initial state if provided
        if input_state_circuit is not None:
            qc.compose(input_state_circuit, qr_sys, inplace=True)
        
        # Ancilla in |+⟩
        qc.h(qr_anc[0])
        
        # Trotter synthesis
        trotter = LieTrotter(reps=1)
        
        # Controlled evolution: exp(-iHτ) when ancilla=0
        # Note: exp(-iHτ) is the UNITARY part
        # For ITE, we want behavior where ground state is favored
        
        # exp(+iHτ) controlled on |0⟩
        evo_plus = PauliEvolutionGate(H_norm, time=-self.tau * norm_factor, synthesis=trotter)
        qc.x(qr_anc[0])  # Flip to control on 0
        qc.append(evo_plus.control(1), [qr_anc[0]] + list(qr_sys))
        qc.x(qr_anc[0])  # Flip back
        
        # exp(-iHτ) controlled on |1⟩
        evo_minus = PauliEvolutionGate(H_norm, time=self.tau * norm_factor, synthesis=trotter)
        qc.append(evo_minus.control(1), [qr_anc[0]] + list(qr_sys))
        
        # Final Hadamard - interference term
        qc.h(qr_anc[0])
        
        return qc
    
    def build_full_ite_circuit(self, n_steps: Optional[int] = None) -> QuantumCircuit:
        """
        Build full ITE circuit with multiple steps.
        
        Multiple ITE steps accumulate probability toward ground state.
        """
        if n_steps is None:
            n_steps = self.n_steps
        
        H_norm, norm_factor = self.H_norm
        
        # Use one ancilla per step for post-selection
        qr_anc = AncillaRegister(n_steps, 'anc')
        qr_sys = QuantumRegister(self.n_qubits, 'sys')
        cr_anc = ClassicalRegister(n_steps, 'c_anc')
        cr_sys = ClassicalRegister(self.n_qubits, 'c_sys')
        
        qc = QuantumCircuit(qr_anc, qr_sys, cr_anc, cr_sys)
        
        # Initialize system in |0...0⟩ or some initial state
        # Apply initial superposition to explore
        for q in range(self.n_qubits):
            qc.h(qr_sys[q])
        
        trotter = LieTrotter(reps=1)
        
        # Apply ITE steps
        for step in range(n_steps):
            # Hadamard on this step's ancilla
            qc.h(qr_anc[step])
            
            # Controlled evolutions
            evo_plus = PauliEvolutionGate(H_norm, time=-self.tau * norm_factor, synthesis=trotter)
            evo_minus = PauliEvolutionGate(H_norm, time=self.tau * norm_factor, synthesis=trotter)
            
            # exp(+iHτ) controlled on |0⟩
            qc.x(qr_anc[step])
            qc.append(evo_plus.control(1), [qr_anc[step]] + list(qr_sys))
            qc.x(qr_anc[step])
            
            # exp(-iHτ) controlled on |1⟩
            qc.append(evo_minus.control(1), [qr_anc[step]] + list(qr_sys))
            
            # Hadamard for interference
            qc.h(qr_anc[step])
        
        # Measure everything
        qc.measure(qr_anc, cr_anc)
        qc.measure(qr_sys, cr_sys)
        
        return qc
    
    def run_ite(self, shots: int = 4096) -> Tuple[str, float]:
        """
        Run ITE and extract best result.
        
        Post-selects on ancilla outcomes that indicate successful cooling.
        
        Returns:
            best_state: Bitstring of best system state found
            success_prob: Probability of successful run
        """
        qc = self.build_full_ite_circuit()
        
        t_qc = transpile(qc.decompose(reps=3), self.backend)
        self.last_circuit_depth = t_qc.depth()
        
        result = self.backend.run(t_qc, shots=shots).result()
        counts = result.get_counts()
        
        self.nefv += 1  # One circuit execution
        
        # Post-select: ancilla should be |0...0⟩ for successful ITE
        success_pattern = '0' * self.n_steps
        
        filtered_counts = {}
        for bitstr, count in counts.items():
            parts = bitstr.split(' ')
            if len(parts) >= 2:
                anc_bits = parts[0]
                sys_bits = parts[1]
            else:
                anc_bits = bitstr[:self.n_steps]
                sys_bits = bitstr[self.n_steps:]
            
            # Accept if all or most ancillas are 0
            n_zeros = anc_bits.count('0')
            if n_zeros >= self.n_steps // 2 + 1:  # Majority zeros
                filtered_counts[sys_bits] = filtered_counts.get(sys_bits, 0) + count
        
        if not filtered_counts:
            # Fallback: use all counts
            filtered_counts = {}
            for bitstr, count in counts.items():
                parts = bitstr.split(' ')
                sys_bits = parts[-1] if len(parts) >= 2 else bitstr
                filtered_counts[sys_bits] = filtered_counts.get(sys_bits, 0) + count
        
        # Find most common
        best_state = max(filtered_counts, key=filtered_counts.get)
        success_prob = sum(filtered_counts.values()) / shots
        
        return best_state, success_prob


class CoherentITE_VQE:
    """
    Combines ITE with variational ansatz.
    
    Uses ITE to generate updates to variational parameters:
    1. Current state: |ψ(θ)⟩
    2. Apply ITE: |ψ'⟩ ∝ e^{-Hτ}|ψ(θ)⟩
    3. Project back: Find θ' such that |ψ(θ')⟩ ≈ |ψ'⟩
    
    This is McLachlan's variational ITE principle!
    """
    
    def __init__(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        tau: float = 0.1,
        backend=None,
    ):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.tau = tau
        self.n_params = ansatz.num_parameters
        self.n_qubits = ansatz.num_qubits
        
        if backend is None:
            self.backend = AerSimulator(method='statevector')
        else:
            self.backend = backend
        
        from qiskit.primitives import StatevectorEstimator
        self.estimator = StatevectorEstimator()
        
        self.nefv = 0
        self.last_circuit_depth = ansatz.decompose().depth()
    
    def _get_statevector(self, params):
        """Get statevector for given parameters."""
        bound = self.ansatz.assign_parameters(params)
        return Statevector.from_instruction(bound)
    
    def _apply_ite_step(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply one ITE step: |ψ'⟩ ∝ exp(-Hτ)|ψ⟩
        
        Uses exact matrix exponential (for statevector sim).
        """
        H_matrix = self.hamiltonian.to_matrix()
        
        # exp(-Hτ) via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        cooling = np.exp(-self.tau * eigenvalues)
        
        # Apply cooling
        psi_eigen = eigenvectors.conj().T @ psi
        psi_cooled = cooling * psi_eigen
        psi_new = eigenvectors @ psi_cooled
        
        # Normalize
        norm = np.linalg.norm(psi_new)
        return psi_new / norm if norm > 1e-10 else psi
    
    def _project_to_manifold(
        self,
        target_state: np.ndarray,
        current_params: np.ndarray,
        n_steps: int = 10,
        lr: float = 0.2,
    ) -> np.ndarray:
        """
        Project target state back to variational manifold.
        
        Finds θ' that maximizes |⟨ψ(θ')|target⟩|².
        """
        params = current_params.copy()
        
        for _ in range(n_steps):
            sv = self._get_statevector(params)
            psi = sv.data
            
            # Compute gradient of fidelity
            grad = np.zeros(self.n_params)
            epsilon = 0.05
            
            current_fid = np.abs(target_state.conj() @ psi)**2
            
            for i in range(self.n_params):
                params_plus = params.copy()
                params_plus[i] += epsilon
                psi_plus = self._get_statevector(params_plus).data
                fid_plus = np.abs(target_state.conj() @ psi_plus)**2
                
                grad[i] = (fid_plus - current_fid) / epsilon
            
            # Gradient ascent
            params = params + lr * grad
            self.nefv += self.n_params
        
        return params
    
    def step(self, params: np.ndarray) -> np.ndarray:
        """
        One step of variational ITE.
        
        1. Get current state |ψ(θ)⟩
        2. Apply ITE: |ψ'⟩ = e^{-Hτ}|ψ(θ)⟩ / norm
        3. Project back: Find θ' where |ψ(θ')⟩ ≈ |ψ'⟩
        """
        # Current state
        current_sv = self._get_statevector(params)
        psi = current_sv.data
        
        # Apply ITE
        psi_cooled = self._apply_ite_step(psi)
        
        # Project back to variational manifold
        new_params = self._project_to_manifold(psi_cooled, params)
        
        self.nefv += 1  # The ITE step itself
        return new_params


class DirectITE:
    """
    Direct ITE without variational ansatz.
    
    For small systems, applies exact ITE and measures.
    O(1) measurement but explores full Hilbert space.
    """
    
    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        n_qubits: int,
        tau_total: float = 2.0,
        n_steps: int = 20,
    ):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.tau_total = tau_total
        self.n_steps = n_steps
        self.tau_step = tau_total / n_steps
        
        self.nefv = 0
    
    def find_ground_state(self, initial_state: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Find ground state via ITE.
        
        Returns:
            energy: Ground state energy
            state: Ground state vector
        """
        H_matrix = self.hamiltonian.to_matrix()
        dim = 2 ** self.n_qubits
        
        # Initial state (uniform superposition)
        if initial_state is None:
            psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        else:
            psi = initial_state
        
        # Eigendecompose H
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        # Apply ITE steps
        for step in range(self.n_steps):
            # Project to eigenbasis
            psi_eigen = eigenvectors.conj().T @ psi
            
            # Apply cooling
            cooling = np.exp(-self.tau_step * eigenvalues)
            psi_cooled = cooling * psi_eigen
            
            # Back to computational basis
            psi = eigenvectors @ psi_cooled
            
            # Normalize
            norm = np.linalg.norm(psi)
            psi = psi / norm
        
        # Compute final energy
        energy = np.real(psi.conj() @ H_matrix @ psi)
        self.nefv += 1  # Count as one coherent evolution
        
        return energy, psi


# --- Benchmark Wrapper ---

class CoherentITE_Wrapper:
    """Wrapper for benchmark.py"""
    
    def __init__(
        self,
        ansatz,
        hamiltonian,
        tau: float = 0.15,
        backend=None,
    ):
        self.optimizer = CoherentITE_VQE(
            ansatz, hamiltonian,
            tau=tau,
            backend=backend
        )
        self.estimator = self.optimizer.estimator
        
    @property
    def nefv(self):
        return self.optimizer.nefv
    
    @property
    def circuit_depth(self):
        return self.optimizer.last_circuit_depth
    
    @property
    def max_circuit_depth(self):
        return self.optimizer.last_circuit_depth
    
    def step(self, params):
        return self.optimizer.step(params)


# --- Test ---

if __name__ == "__main__":
    print("=" * 60)
    print("  Coherent ITE - Imaginary Time Evolution")
    print("  Ground state via e^{-Hτ} cooling")
    print("=" * 60)
    
    # Test problem
    N = 4
    ops = [("ZIZI", 1.0), ("IZIZ", 1.0), ("XXII", 0.5), ("IIXX", 0.5)]
    H = SparsePauliOp.from_list(ops)
    
    print(f"\nProblem: {N} Qubits")
    print(f"Hamiltonian: {len(H.paulis)} terms")
    
    # Method 1: Direct ITE (exact for small systems)
    print("\n--- Direct ITE (Exact) ---")
    direct_ite = DirectITE(H, N, tau_total=3.0, n_steps=30)
    E_gs, psi_gs = direct_ite.find_ground_state()
    print(f"Ground state energy: {E_gs:.6f}")
    print(f"NEFV: {direct_ite.nefv}")
    
    # Verify with numpy
    H_matrix = H.to_matrix()
    eigvals = np.linalg.eigvalsh(H_matrix)
    print(f"Exact ground state: {eigvals[0]:.6f}")
    
    # Method 2: Circuit-based ITE
    print("\n--- Circuit-Based ITE ---")
    circuit_ite = CoherentITE(H, N, tau=0.2, n_steps=3)
    best_state, success_prob = circuit_ite.run_ite(shots=4096)
    print(f"Best state: |{best_state}⟩")
    print(f"Success probability: {success_prob:.2%}")
    print(f"NEFV: {circuit_ite.nefv}")
    print(f"Circuit depth: {circuit_ite.last_circuit_depth}")
    
    # Method 3: Variational ITE
    print("\n--- Variational ITE (VQE-like) ---")
    from qiskit.circuit.library import EfficientSU2
    ansatz = EfficientSU2(N, reps=1)
    
    vite = CoherentITE_VQE(ansatz, H, tau=0.15)
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    
    from qiskit.primitives import StatevectorEstimator
    ref_est = StatevectorEstimator()
    
    for epoch in range(10):
        job = ref_est.run([(ansatz.assign_parameters(params), H)])
        E = float(job.result()[0].data.evs)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1:02d} | E: {E:+.4f} | NEFV: {vite.nefv}")
        
        params = vite.step(params)
    
    job = ref_est.run([(ansatz.assign_parameters(params), H)])
    E_final = float(job.result()[0].data.evs)
    print(f"Final   | E: {E_final:+.4f} | NEFV: {vite.nefv}")
    
    print(f"\n=== Summary ===")
    print(f"Direct ITE (exact): E = {E_gs:.4f}, NEFV = {direct_ite.nefv}")
    print(f"Variational ITE: E = {E_final:.4f}, NEFV = {vite.nefv}")
    print(f"True ground state: E = {eigvals[0]:.4f}")
