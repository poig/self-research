"""
landauer_limit_test.py

EXPERIMENT 1b: THE LANDAUER LIMIT
---------------------------------
Critique: "Is the extracted work just borrowing energy that must be paid back 
           to erase the ancilla?"

Theory:
- Extracted Work W_ext = eta * I(S:A)
- Erasure Cost W_cost >= k_B T * ln(2) * S(A)  [when S(A) is in bits]
- Net Work = W_ext - W_cost

Quantum Advantage via I(S:A) = 2*S(A):
For pure bipartite entanglement, I(S:A) = 2*S(A).
This means the demon gets TWICE the mutual information per unit of ancilla entropy.
The factor-of-2 HALVES the information cost per unit work extracted,
making quantum feedback more efficient than classical feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import AerSimulator
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 4
KICK_STRENGTH = 0.2  # θ_gain = 0.2 rad (TABLE I). Note: Fig 1 uses 0.5
TAU_STEPS = 20
MAX_TAU = 1.5

# Landauer cost scale (k_B T) in the same energy units as the Hamiltonian.
# NOTE: In these scripts the Hamiltonian couplings are dimensionless random numbers,
# so choosing k_B T is a modeling assumption rather than a derived physical temperature.
KBT = 1.0

# Time evolution is synthesized (product formula) after transpilation.
# Default PauliEvolutionGate synthesis is LieTrotter(reps=1). Make it explicit
# so the approximation can be tightened when needed.
EVOLUTION_REPS = 1

# FIX: reps=1 is far from converged (trotter_r2_check.py: ordered R^2 moves
# 0.244 -> 0.651 going reps=1 -> exact at N=7), and every number in the paper
# was computed at reps=1. The controlled evolution is now built exactly from the
# eigendecomposition of H; the Trotter path is kept only to reproduce the old
# figure for comparison.
EXACT_EVOLUTION = True


class LandauerExperiment:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.backend = AerSimulator(method='statevector')
        
        # 1. Initialize Same Hamiltonian as Exp 1
        np.random.seed(42)
        ops = []
        # Interactions (Z_i Z_j)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                J = np.random.uniform(-1, 1)
                label = ["I"] * n_qubits
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), J))
        # Fields (X_i)
        for i in range(n_qubits):
            h = np.random.uniform(-0.5, 0.5)
            label = ["I"] * n_qubits
            label[i] = "X"
            ops.append(("".join(label[::-1]), h))
            
        self.H = SparsePauliOp.from_list(ops)
        self._spec = np.linalg.eigh(self.H.to_matrix())
        print(f"[Init] System N={n_qubits}. Landauer Test Ready. "
              f"Evolution: {'EXACT' if EXACT_EVOLUTION else f'Trotter r={EVOLUTION_REPS}'}")

    def _controlled_evo(self, tau):
        """|0><0|_anc (x) I + |1><1|_anc (x) e^{-iH tau}, ancilla as gate qubit 0."""
        evals, evecs = self._spec
        u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
        p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        return np.kron(np.eye(2 ** self.n), p0) + np.kron(u_sys, p1)

    def get_energy(self, state):
        return state.expectation_value(self.H).real

    def run_cycle_analysis(self, tau):
        """Returns (MutualInfo, AncillaEntropy, WorkExtracted, JointEntropy)"""
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # Init |+>
        qc.h(qr_sys)
        qc.h(qr_anc)
        
        # 1. Sensing (Entangle)
        if EXACT_EVOLUTION:
            qc.append(UnitaryGate(self._controlled_evo(tau)),
                      [qr_anc[0]] + list(qr_sys))
        else:
            evo = PauliEvolutionGate(self.H, time=tau,
                                     synthesis=LieTrotter(reps=EVOLUTION_REPS))
            qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # 2. Locking (Measurement Basis)
        qc.h(qr_anc)
        qc.save_statevector(label="post_sensing")
        
        # 3. Actuation
        for i in range(self.n):
             qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
        
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # --- ANALYSIS ---
        
        # A. Information & Cost
        sv_sensing = result.data(0)["post_sensing"]
        rho_sensing = DensityMatrix(sv_sensing)
        
        # Entropies (Base 2 -> Bits)
        S_SA = entropy(rho_sensing, base=2)
        rho_S = partial_trace(rho_sensing, [0]) 
        rho_A = partial_trace(rho_sensing, range(1, self.n+1)) 
        S_S = entropy(rho_S, base=2)
        S_A = entropy(rho_A, base=2)
        
        # Mutual Info
        mutual_info = S_S + S_A - S_SA
        
        # B. Work
        E_before = self.get_energy(rho_S)
        sv_final = result.data(0)["final"]
        rho_final_full = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final_full, [0])
        E_after = self.get_energy(rho_S_final)
        extracted_work = E_before - E_after
        
        return mutual_info, S_A, extracted_work, S_SA

    def run_experiment(self):
        print("="*70)
        print("EXPERIMENT 1b: LANDAUER COST ANALYSIS")
        print("Checking if Quantum Correlations allow Positive Net Work.")
        print("="*70)
        
        data_tau = []
        data_mi = []
        data_sa = []
        data_work = []
        data_s_sa = []
        
        print(f"{'Tau':<6} | {'MI (I)':<10} | {'S(A) (Cost)':<12} | {'Ratio I/S':<10} | {'Work':<10}")
        print("-" * 70)
        
        for tau in np.linspace(0.0, MAX_TAU, TAU_STEPS):
            mi, sa, work, s_sa = self.run_cycle_analysis(tau)
            
            ratio = mi / sa if sa > 1e-6 else 0.0
            
            data_tau.append(tau)
            data_mi.append(mi)
            data_sa.append(sa)
            data_work.append(work)
            data_s_sa.append(s_sa)
            
            print(f"{tau:<6.2f} | {mi:<10.4f} | {sa:<12.4f} | {ratio:<10.2f} | {work:<10.4f}")

        # --- THERMODYNAMIC ANALYSIS ---
        # 1. Determine Effective Temperature from the Constitutive Law (Work = eta * I)
        slope, _, _, _, _ = linregress(data_mi, data_work)
        eta_eff = slope  # Units: Energy per Bit (this is η, not T)
        
        print("-" * 70)
        print(f"Algorithmic Efficiency η = {eta_eff:.4f} Energy/Bit")
        
        # 2. Calculate Landauer Cost
        # Cost = k_B T ln(2) * S(A)  [with S(A) in bits]
        # Since we work in normalized units, we use ln(2) ≈ 0.693 and treat k_B T as a modeling parameter.
        LN2 = np.log(2)  # ≈ 0.693
        landauer_costs = [KBT * LN2 * sa for sa in data_sa]
        net_works = [w - c for w, c in zip(data_work, landauer_costs)]
        
        max_net_work = max(net_works)
        avg_ratio = np.mean([m/s for m, s in zip(data_mi, data_sa) if s > 0.1])
        
        print(f"Landauer Cost (kBT·ln2·S_A):   {np.mean(landauer_costs):.4f} (avg)")
        print(f"Assumed k_B T:                {KBT:.3f}")
        print(f"Avg Quantum Ratio (I / S_A):   {avg_ratio:.2f}")

        avg_joint_entropy = float(np.mean(data_s_sa))
        if avg_joint_entropy < 1e-6:
            print("\nNote: S(SA) ≈ 0 in statevector simulation, so I(S:A) = 2 S(A) holds as an identity.")
            print("      Treat I/S(A) ≈ 2 as a regime check (pure joint state), not an empirical advantage claim.")
        
        if avg_ratio > 1.5 and avg_joint_entropy >= 1e-6:
            print("\n✓ Entanglement-like regime detected (I/S(A) near 2).")
        elif avg_ratio <= 1.5:
            print("\n? BELOW ENTANGLEMENT THRESHOLD.")
            print(f"  I(S:A)/S(A) = {avg_ratio:.2f} (expected ~2.0 for pure bipartite entanglement)")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(data_tau, data_work, 'bo-', label='Extracted Work $W = \\eta \\cdot I(S:A)$')
        plt.plot(data_tau, landauer_costs, 'r--', label='Landauer Cost $k_BT\\ln 2 \\cdot S(A)$')
        
        plt.xlabel('Sensing Time $\\tau$')
        plt.ylabel('Energy')
        plt.title(
            f'Landauer Cost Comparison (I(S:A)/S(A) = {avg_ratio:.2f})\n'
            f'Note: I/S(A) \u2248 2 when S(SA) \u2248 0 (pure joint state)'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('thermo_landauer_check.png', dpi=150)
        print("Saved plot to 'thermo_landauer_check.png'")

if __name__ == "__main__":
    exp = LandauerExperiment(N_QUBITS)
    exp.run_experiment()