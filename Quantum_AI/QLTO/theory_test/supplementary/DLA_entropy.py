"""
DLA_entropy.py: The Thermo-Algebraic Link
=========================================
Testing the "Grand Unification" Hypothesis:
    "The thermodynamic efficiency (eta) of a quantum optimizer is 
     inversely proportional to the dimension of its Dynamical Lie Algebra (DLA)."

Reference: arXiv:2407.12587 (DLA dimensions for QAOA)
    - Cycle Graph (C_n): DLA dim = 3n - 1 (Linear scaling)
    - Complete Graph (K_n): DLA dim ~ n^3 / 12 (Cubic scaling)

Prediction:
    We expect eta_cycle >> eta_complete.
    The "Information Friction" should be higher for the Complete Graph.

Protocol:
    Run the "Coherent Feedback" experiment (Phase 3) on both topologies
    and compare the slope of Work vs. Mutual Information.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
    from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, DensityMatrix, partial_trace, entropy
    from qiskit_aer import AerSimulator
    from qiskit.synthesis import LieTrotter
except ImportError:
    print("CRITICAL: Qiskit missing.")
    exit(1)

# ==============================================================================
# HAMILTONIAN FACTORY
# ==============================================================================

def build_cycle_hamiltonian(n_qubits):
    """
    Constructs Hamiltonian for Cycle Graph C_n.
    H = sum(Z_i Z_{i+1}) + sum(X_i)
    DLA Dimension: 3n - 1
    """
    ops = []
    # Interaction Terms (Periodic Boundary)
    for i in range(n_qubits):
        op_str = ["I"] * n_qubits
        op_str[i] = "Z"
        op_str[(i + 1) % n_qubits] = "Z"
        ops.append(("".join(op_str), 1.0))
    
    # Transverse Field (Driver)
    for i in range(n_qubits):
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), 0.5))
        
    return SparsePauliOp.from_list(ops)

def build_complete_hamiltonian(n_qubits):
    """
    Constructs Hamiltonian for Complete Graph K_n.
    H = sum_{i<j}(Z_i Z_j) + sum(X_i)
    DLA Dimension: ~ n^3 / 12
    """
    ops = []
    # All-to-All Interaction
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            op_str = ["I"] * n_qubits
            op_str[i] = "Z"
            op_str[j] = "Z"
            ops.append(("".join(op_str), 1.0))
            
    # Transverse Field
    for i in range(n_qubits):
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), 0.5))
        
    return SparsePauliOp.from_list(ops)

# ==============================================================================
# THERMODYNAMIC ENGINE
# ==============================================================================

class DLA_Efficiency_Experiment:
    def __init__(self, n_qubits=4, graph_type='cycle'):
        self.n_qubits = n_qubits
        self.graph_type = graph_type
        self.backend = AerSimulator(method='matrix_product_state')
        
        # 1. Select Hamiltonian
        if graph_type == 'cycle':
            self.hamiltonian = build_cycle_hamiltonian(n_qubits)
            self.dla_dim = 3 * n_qubits - 1
        elif graph_type == 'complete':
            self.hamiltonian = build_complete_hamiltonian(n_qubits)
            # Formula from Theorem 57 of the paper (for even n)
            self.dla_dim = (n_qubits**3 + 6*n_qubits**2 + 2*n_qubits + 12) // 12
        
        self.ansatz = EfficientSU2(n_qubits, reps=1, entanglement='linear').decompose()
        self.param_count = self.ansatz.num_parameters
        
        print(f"[Init] Graph: {graph_type.upper()} | Qubits: {n_qubits} | Theoretical DLA Dim: {self.dla_dim}")

    def run_step(self, params, dt_sense, kick_gain):
        """
        Runs the Coherent Feedback protocol.
        """
        qr_anc = AncillaRegister(1, 'demon')
        qr_sys = QuantumRegister(self.n_qubits, 'sys')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # State Prep
        qc.append(self.ansatz.assign_parameters(params), qr_sys)
        
        # Initial Energy Calculation (God Mode)
        rho_init = self._get_density_matrix(qc)
        rho_sys_init = partial_trace(rho_init, [0])
        E_init = np.real(np.trace(rho_sys_init.data @ self.hamiltonian.to_matrix()))

        # --- SENSING ---
        qc.h(qr_anc)
        trotter = LieTrotter(reps=1)
        evo = PauliEvolutionGate(self.hamiltonian, time=dt_sense, synthesis=trotter)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        qc.h(qr_anc)
        
        # --- MEASURE MUTUAL INFO ---
        rho_sensed = self._get_density_matrix(qc)
        rho_A = partial_trace(rho_sensed, list(range(1, self.n_qubits + 1)))
        rho_S = partial_trace(rho_sensed, [0])
        mi = entropy(rho_A, base=2) + entropy(rho_S, base=2) - entropy(rho_sensed, base=2)

        # --- COHERENT KICK ---
        # We use a small kick to stay in the linear response regime
        for i in range(self.n_qubits):
            qc.crx(kick_gain, qr_anc[0], qr_sys[i])

        # --- FINAL ENERGY ---
        rho_final = self._get_density_matrix(qc)
        rho_sys_final = partial_trace(rho_final, [0])
        E_final = np.real(np.trace(rho_sys_final.data @ self.hamiltonian.to_matrix()))
        
        return mi, E_init - E_final

    def _get_density_matrix(self, qc):
        qc_sim = qc.copy()
        qc_sim.save_statevector()
        res = self.backend.run(transpile(qc_sim, self.backend), shots=1).result()
        return DensityMatrix(res.get_statevector())

# ==============================================================================
# MAIN SWEEP
# ==============================================================================

def run_comparative_study():
    print("\n=== DLA EFFICIENCY SWEEP ===")
    print("Comparing Cycle Graph vs. Complete Graph")
    print("Hypothesis: Efficiency (slope) drops as DLA dimension increases.\n")
    
    N_QUBITS = 4
    DT_RANGE = np.linspace(0.1, 1.2, 12) # Keep dt short to stay in linear regime
    KICK_GAIN = 0.2 # Small kick to ensure we measure 'friction', not overheating
    
    # --- Run Cycle ---
    exp_c = DLA_Efficiency_Experiment(N_QUBITS, 'cycle')
    mi_c, work_c = [], []
    params_c = np.random.uniform(-0.5, 0.5, exp_c.param_count)
    
    for dt in DT_RANGE:
        m, w = exp_c.run_step(params_c, dt, KICK_GAIN)
        mi_c.append(m); work_c.append(w)
        
    # --- Run Complete ---
    exp_k = DLA_Efficiency_Experiment(N_QUBITS, 'complete')
    mi_k, work_k = [], []
    # Use same random seed for params to ensure fair starting ground energy-wise
    np.random.seed(42) 
    params_k = np.random.uniform(-0.5, 0.5, exp_k.param_count)
    
    for dt in DT_RANGE:
        m, w = exp_k.run_step(params_k, dt, KICK_GAIN)
        mi_k.append(m); work_k.append(w)

    # --- Analysis ---
    slope_c, _, r_c, _, _ = linregress(mi_c, work_c)
    slope_k, _, r_k, _, _ = linregress(mi_k, work_k)
    
    print("\n=== RESULTS ===")
    print(f"Cycle Graph (DLA={exp_c.dla_dim}):    eta = {slope_c:.4f} (R^2={r_c**2:.3f})")
    print(f"Complete Graph (DLA={exp_k.dla_dim}): eta = {slope_k:.4f} (R^2={r_k**2:.3f})")
    
    ratio = slope_c / slope_k
    print(f"\n>>> EFFICIENCY RATIO: {ratio:.2f}x")
    print(f">>> DLA DIM RATIO:    {exp_k.dla_dim / exp_c.dla_dim:.2f}x")
    
    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(mi_c, work_c, color='blue', s=80, label=f'Cycle (DLA={exp_c.dla_dim})')
    plt.plot(mi_c, slope_c*np.array(mi_c), 'b--', alpha=0.5)
    
    plt.scatter(mi_k, work_k, color='red', s=80, label=f'Complete (DLA={exp_k.dla_dim})')
    plt.plot(mi_k, slope_k*np.array(mi_k), 'r--', alpha=0.5)
    
    plt.title(f"Thermodynamic Efficiency vs. DLA Complexity (N={N_QUBITS})", fontsize=14)
    plt.xlabel("Mutual Information I(S:Demon) (Bits)", fontsize=12)
    plt.ylabel("Extracted Work -dE", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotation
    plt.text(0.6, 0.2, 
             f"$\\eta_{{cycle}} = {slope_c:.3f}$\n$\\eta_{{complete}} = {slope_k:.3f}$", 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', edgecolor='black'))
    
    plt.savefig("dla_efficiency_result.png", dpi=300)
    print("\n[Output] Saved result to 'dla_efficiency_result.png'")

if __name__ == "__main__":
    run_comparative_study()