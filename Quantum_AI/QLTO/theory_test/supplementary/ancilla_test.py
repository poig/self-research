"""
linearity_ablation.py

SOURCE OF NON-LINEARITY TEST (FULL QLTO CIRCUIT)
------------------------------------------------
Hypothesis: 
Non-Linearity requires the FULL Cycle:
Superposition (W) -> Interaction (U) -> Disentanglement (W_dag).
Without W_dag, the system acts as noise. With W_dag, we see the Phase.

Test Protocol:
1. Initialize 2 Parameter Qubits (Superposition).
2. W-Gate: P0->S0, P1->S1 (Controlled Rotations).
3. Evolve: Apply Hamiltonian to System.
4. INVERSE W-Gate: Uncompute the entanglement.
5. Measure MI of Parameters.

Prediction:
- Independent H (Z0, Z1): MI ≈ 0 (Separable Phase)
- Interacting H (Z0Z1):   MI > 0 (Entangled Phase -> Non-Linearity)
"""

import numpy as np
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import partial_trace, entropy
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("CRITICAL: Qiskit not installed.")
    exit(1)

# ==============================================================================
# TEST ENGINE
# ==============================================================================

class FullCircuitTest:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits 
        self.backend = AerSimulator(method='statevector')

    def measure_mutual_information(self, interaction_type="independent"):
        """
        interaction_type: "independent" (Z+Z) or "interacting" (ZZ)
        """
        qr_param = QuantumRegister(2, 'param')
        qr_sys = QuantumRegister(self.n_qubits, 'sys')
        qc = QuantumCircuit(qr_param, qr_sys)
        
        # --- 1. INITIALIZATION ---
        qc.h(qr_param)
        
        # --- 2. W-GATE (Encode) ---
        # Param 0 shifts Sys 0, Param 1 shifts Sys 1
        # We use a large shift (pi) to make the states very distinct
        delta = np.pi 
        
        # Apply Controlled-RY
        # If P=0 -> Ry(0). If P=1 -> Ry(pi) = Flip to |1>
        qc.cry(delta, qr_param[0], qr_sys[0])
        qc.cry(delta, qr_param[1], qr_sys[1])
        
        # --- 3. HAMILTONIAN EVOLUTION ---
        # Time chosen to create significant phase
        t = 1.0
        
        if interaction_type == "independent":
            # H = Z0 + Z1. 
            # Energy depends on Z0 (controlled by P0) + Z1 (controlled by P1).
            # Phase = exp(-i(E0 + E1)) = exp(-iE0) * exp(-iE1). Separable.
            qc.rz(2*t, qr_sys[0])
            qc.rz(2*t, qr_sys[1])
            
        elif interaction_type == "interacting":
            # H = Z0 * Z1.
            # Energy depends on Z0 * Z1.
            # Phase = exp(-i(E0 * E1)). Coupled.
            qc.rzz(2*t, qr_sys[0], qr_sys[1])
        
        # --- 4. INVERSE W-GATE (Decode) ---
        # This is CRITICAL. It removes the entanglement with the System.
        # Ideally, it returns the system to |00>, leaving the phase on Params.
        qc.cry(-delta, qr_param[0], qr_sys[0])
        qc.cry(-delta, qr_param[1], qr_sys[1])
        
        # --- 5. MEASURE ---
        qc.save_statevector()
        
        t_qc = transpile(qc, self.backend)
        res = self.backend.run(t_qc).result().get_statevector()
        
        # Trace out System (indices 2, 3)
        rho_params = partial_trace(res, [2, 3])
        
        # Calculate Mutual Information of the Parameter Register
        S_AB = entropy(rho_params)
        rho_A = partial_trace(rho_params, [1])
        rho_B = partial_trace(rho_params, [0])
        S_A = entropy(rho_A)
        S_B = entropy(rho_B)
        
        MI = S_A + S_B - S_AB
        return max(0.0, MI)

    def run_experiment(self):
        print("="*60)
        print("FULL QLTO CIRCUIT TEST (W -> U -> W_dag)")
        print("Hypothesis: W_dag is required to see the correlation.")
        print("="*60)
        
        # --- TEST A: INDEPENDENT ---
        print("\n[Case A] Independent H (Z0 + Z1)")
        mi_indep = self.measure_mutual_information("independent")
        print(f"  Mutual Information: {mi_indep:.6f}")
        
        # --- TEST B: INTERACTING ---
        print("\n[Case B] Interacting H (Z0 * Z1)")
        mi_int = self.measure_mutual_information("interacting")
        print(f"  Mutual Information: {mi_int:.6f}")
        
        # --- ANALYSIS ---
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        
        if mi_int > mi_indep + 0.01:
            print("✓ SUCCESS: NON-LINEARITY PROVEN.")
            print(f"  Independent MI: {mi_indep:.4f} (Linear/Separable)")
            print(f"  Interacting MI: {mi_int:.4f} (Non-Linear/Entangled)")
            print("\n  The math holds:")
            print("  - exp(-i(A+B)) factors into independent rotations.")
            print("  - exp(-i(A*B)) creates entanglement.")
            print("  - W_dag successfully transferred this info to the parameters.")
        else:
            print("? FAILURE: Still no correlation. Check phases.")

if __name__ == "__main__":
    test = FullCircuitTest(n_qubits=2)
    test.run_experiment()