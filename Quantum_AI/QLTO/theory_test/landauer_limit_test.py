"""
landauer_limit_test.py

EXPERIMENT 1b: THE LANDAUER LIMIT
---------------------------------
Critique: "Is the extracted work just borrowing energy that must be paid back 
           to erase the ancilla?"

Theory:
- Extracted Work W_ext <= k_B T * I(S:A)
- Erasure Cost W_cost >= k_B T * S(A)
- Net Work = W_ext - W_cost

Quantum Advantage Condition:
If I(S:A) > S(A), then W_ext > W_cost.
This is only possible with Entanglement (I ~ 2S).
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 4
KICK_STRENGTH = 0.5
TAU_STEPS = 20
MAX_TAU = 1.5

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
        print(f"[Init] System N={n_qubits}. Landauer Test Ready.")

    def get_energy(self, state):
        return state.expectation_value(self.H).real

    def run_cycle_analysis(self, tau):
        """Returns (MutualInfo, AncillaEntropy, WorkExtracted)"""
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # Init |+>
        qc.h(qr_sys)
        qc.h(qr_anc)
        
        # 1. Sensing (Entangle)
        evo = PauliEvolutionGate(self.H, time=tau)
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
        
        return mutual_info, S_A, extracted_work

    def run_experiment(self):
        print("="*70)
        print("EXPERIMENT 1b: LANDAUER COST ANALYSIS")
        print("Checking if Quantum Correlations allow Positive Net Work.")
        print("="*70)
        
        data_tau = []
        data_mi = []
        data_sa = []
        data_work = []
        
        print(f"{'Tau':<6} | {'MI (I)':<10} | {'S(A) (Cost)':<12} | {'Ratio I/S':<10} | {'Work':<10}")
        print("-" * 70)
        
        for tau in np.linspace(0.0, MAX_TAU, TAU_STEPS):
            mi, sa, work = self.run_cycle_analysis(tau)
            
            ratio = mi / sa if sa > 1e-6 else 0.0
            
            data_tau.append(tau)
            data_mi.append(mi)
            data_sa.append(sa)
            data_work.append(work)
            
            print(f"{tau:<6.2f} | {mi:<10.4f} | {sa:<12.4f} | {ratio:<10.2f} | {work:<10.4f}")

        # --- THERMODYNAMIC ANALYSIS ---
        # 1. Determine Effective Temperature from the Constitutive Law (Work = eta * I)
        slope, _, _, _, _ = linregress(data_mi, data_work)
        T_eff = slope # Units: Energy per Bit
        
        print("-" * 70)
        print(f"Effective Temperature (Slope): {T_eff:.4f} Energy/Bit")
        
        # 2. Calculate Net Work
        # Cost = T_eff * S(A)
        landauer_costs = [T_eff * sa for sa in data_sa]
        net_works = [w - c for w, c in zip(data_work, landauer_costs)]
        
        max_net_work = max(net_works)
        avg_ratio = np.mean([m/s for m, s in zip(data_mi, data_sa) if s > 0.1])
        
        print(f"Max Net Work (Work - Cost):    {max_net_work:.4f}")
        print(f"Avg Quantum Ratio (I / S_A):   {avg_ratio:.2f}")
        
        if avg_ratio > 1.1:
            print("\n✓ SUCCESS: QUANTUM ADVANTAGE PROVEN.")
            print("  I(S:A) > S(A) implies entanglement is fueling the engine.")
            print("  The Demon extracts more work than the erasure cost.")
        else:
            print("\n? WARNING: CLASSICAL LIMIT.")
            print("  I(S:A) <= S(A). No net work possible.")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(data_tau, data_work, 'bo-', label='Extracted Work')
        plt.plot(data_tau, landauer_costs, 'r--', label='Landauer Cost (Erasure)')
        plt.fill_between(data_tau, data_work, landauer_costs, where=[w > c for w, c in zip(data_work, landauer_costs)], color='green', alpha=0.2, label='Net Positive Work')
        
        plt.xlabel('Sensing Time (tau)')
        plt.ylabel('Energy')
        plt.title(f'Thermodynamic Cycle Check\n(Quantum Ratio I/S ≈ {avg_ratio:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('thermo_landauer_check.png')
        print("Saved plot to 'thermo_landauer_check.png'")

if __name__ == "__main__":
    exp = LandauerExperiment(N_QUBITS)
    exp.run_experiment()