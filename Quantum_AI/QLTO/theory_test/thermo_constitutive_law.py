"""
thermo_constitutive_law.py

EXPERIMENT 1: THE EQUATION OF STATE
-----------------------------------
Validates the claim: "Work Extraction is proportional to Mutual Information."
(Fig. 1 Top of Manuscript)

Protocol:
1. Initialize System in generic state (|+>).
2. SENSING (Variable):
   - Interact Ancilla with System for time 'tau'.
   - This encodes H-gradient into Ancilla phase.
   - Vary 'tau' to vary Information gathered.
3. LOCKING:
   - Apply Hadamard to Ancilla (Phase -> Population).
   - MEASURE Mutual Information I(S:A) here.
4. ACTUATION (Fixed):
   - Apply fixed Coherent Feedback (CRx) of strength 'theta_kick'.
   - This isolates Info as the only variable.
5. WORK:
   - Calculate Energy Drop (-Delta H).

Prediction:
- Linear correlation between Work and Mutual Information.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 4
KICK_STRENGTH = 0.5  # Fixed mechanical actuation
TAU_STEPS = 20       # Number of sensing durations to test
MAX_TAU = 1.5        # Max sensing time

# ==============================================================================
# THERMODYNAMIC ENGINE
# ==============================================================================

class MaxwellDemonExperiment:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.backend = AerSimulator(method='statevector')
        
        # 1. Define Hamiltonian (Random Transverse Ising)
        # H = sum(J_ij Zi Zj) + sum(h_i Xi)
        np.random.seed(42) # Fixed seed for reproducibility
        ops = []
        # Interactions
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                J = np.random.uniform(-1, 1)
                label = ["I"] * n_qubits
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), J))
        # Fields
        for i in range(n_qubits):
            h = np.random.uniform(-0.5, 0.5)
            label = ["I"] * n_qubits
            label[i] = "X"
            ops.append(("".join(label[::-1]), h))
            
        self.H = SparsePauliOp.from_list(ops)
        print(f"[Init] System N={n_qubits}. Hamiltonian Terms={len(ops)}")

    def get_energy(self, state):
        """Computes <H> for a given density matrix/statevector."""
        # E = Tr(rho * H)
        # For statevector: <psi|H|psi>
        if isinstance(state, DensityMatrix):
            return state.expectation_value(self.H).real
        elif isinstance(state, Statevector):
            return state.expectation_value(self.H).real
        return 0.0
    
    def run_control_cycle(self, tau):
        """
        Run the same cycle but with NON-INTERACTING Hamiltonian
        (H_control = sum Zi). No entanglement should mean No Work.
        """
        qr_sys = QuantumRegister(2, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # --- 0. INITIALIZATION ---
        # Prepare System in |+> state (Generic high-energy state)
        qc.h(qr_sys)
        # Prepare Ancilla in |+> (Ready to sense phase)
        qc.h(qr_anc)
        
        # Save Initial Energy (Theoretical)
        # We need the state *before* feedback to know starting energy? 
        # Actually, standard definition is E_initial - E_final.
        # E_initial is fixed for all runs.
        # Let's calculate it once separately or re-calc here.
        # Simpler: Just run the circuit up to here to get rho_init.
        
        # --- 1. SENSING (The Variable) ---
        # Controlled-Evolution: |0> -> I, |1> -> e^{-i H tau}
        # This maps Energy -> Phase
        # Replace H_sys with H_non_interacting (just Z fields)
        evo = PauliEvolutionGate(SparsePauliOp(["ZI", "IZ"]), time=tau)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # --- 2. LOCKING (Information Capture) ---
        # Hadamard converts Phase (Energy Info) -> Bit (Population)
        qc.h(qr_anc)
        
        # === MEASURE INFORMATION HERE ===
        # We need the state *before* the kick to measure what the Demon knows.
        qc.save_statevector(label="post_sensing")
        
        # --- 3. ACTUATION (The Constant) ---
        # Coherent Feedback: Rotate System based on Ancilla
        # "If Ancilla is 1 (Energy was high/low), Rotate parameters"
        # Since we don't have variational parameters in this raw test, 
        # we apply a physical rotation exp(-i X theta) to 'cool' the spins.
        # We use a global X rotation as a generic cooling force against Z-dominant H.
        
        # Controlled-RX on all system qubits
        # Fixed strength KICK_STRENGTH
        for i in range(2):
             qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
             
        # --- 4. EXHAUST ---
        # Save final state
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # --- ANALYSIS ---
        
        # A. Mutual Information I(S:A) after Locking
        sv_sensing = result.data(0)["post_sensing"]
        rho_sensing = DensityMatrix(sv_sensing)
        
        # Entropy Calculation
        # I(S:A) = S(S) + S(A) - S(SA)
        # Note: qiskit.quantum_info.entropy returns Von Neumann entropy (base e or 2?)
        # Default is base 2.
        
        S_SA = entropy(rho_sensing)
        
        # Partial Traces
        rho_S = partial_trace(rho_sensing, [0]) # Trace out Ancilla (index 0)
        rho_A = partial_trace(rho_sensing, range(1, 3)) # Trace out System
        
        S_S = entropy(rho_S)
        S_A = entropy(rho_A)
        
        mutual_info = S_S + S_A - S_SA
        
        # B. Work Extraction
        # E_init (Energy of System before Feedback)
        # Note: rho_S calculated above IS the state of the system *before* feedback
        # (because the ancilla control hasn't acted on it yet, effectively).
        # Wait, the Controlled-Evolution *entangled* them.
        # The Reduced State rho_S is the correct "System State" at that moment.
        E_before = self.get_energy(rho_S)
        
        # E_final (Energy of System after Feedback)
        sv_final = result.data(0)["final"]
        rho_final_full = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final_full, [0]) # Trace out Ancilla (reset)
        
        E_after = self.get_energy(rho_S_final)
        
        extracted_work = E_before - E_after
        
        return mutual_info, extracted_work
        

    def run_cycle(self, tau):
        """
        Runs one thermodynamic cycle with sensing duration 'tau'.
        Returns (MutualInformation, ExtractedWork).
        """
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # --- 0. INITIALIZATION ---
        # Prepare System in |+> state (Generic high-energy state)
        qc.h(qr_sys)
        # Prepare Ancilla in |+> (Ready to sense phase)
        qc.h(qr_anc)
        
        # Save Initial Energy (Theoretical)
        # We need the state *before* feedback to know starting energy? 
        # Actually, standard definition is E_initial - E_final.
        # E_initial is fixed for all runs.
        # Let's calculate it once separately or re-calc here.
        # Simpler: Just run the circuit up to here to get rho_init.
        
        # --- 1. SENSING (The Variable) ---
        # Controlled-Evolution: |0> -> I, |1> -> e^{-i H tau}
        # This maps Energy -> Phase
        evo = PauliEvolutionGate(self.H, time=tau)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # --- 2. LOCKING (Information Capture) ---
        # Hadamard converts Phase (Energy Info) -> Bit (Population)
        qc.h(qr_anc)
        
        # === MEASURE INFORMATION HERE ===
        # We need the state *before* the kick to measure what the Demon knows.
        qc.save_statevector(label="post_sensing")
        
        # --- 3. ACTUATION (The Constant) ---
        # Coherent Feedback: Rotate System based on Ancilla
        # "If Ancilla is 1 (Energy was high/low), Rotate parameters"
        # Since we don't have variational parameters in this raw test, 
        # we apply a physical rotation exp(-i X theta) to 'cool' the spins.
        # We use a global X rotation as a generic cooling force against Z-dominant H.
        
        # Controlled-RX on all system qubits
        # Fixed strength KICK_STRENGTH
        for i in range(self.n):
             qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
             
        # --- 4. EXHAUST ---
        # Save final state
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # --- ANALYSIS ---
        
        # A. Mutual Information I(S:A) after Locking
        sv_sensing = result.data(0)["post_sensing"]
        rho_sensing = DensityMatrix(sv_sensing)
        
        # Entropy Calculation
        # I(S:A) = S(S) + S(A) - S(SA)
        # Note: qiskit.quantum_info.entropy returns Von Neumann entropy (base e or 2?)
        # Default is base 2.
        
        S_SA = entropy(rho_sensing)
        
        # Partial Traces
        rho_S = partial_trace(rho_sensing, [0]) # Trace out Ancilla (index 0)
        rho_A = partial_trace(rho_sensing, range(1, self.n+1)) # Trace out System
        
        S_S = entropy(rho_S)
        S_A = entropy(rho_A)
        
        mutual_info = S_S + S_A - S_SA
        
        # B. Work Extraction
        # E_init (Energy of System before Feedback)
        # Note: rho_S calculated above IS the state of the system *before* feedback
        # (because the ancilla control hasn't acted on it yet, effectively).
        # Wait, the Controlled-Evolution *entangled* them.
        # The Reduced State rho_S is the correct "System State" at that moment.
        E_before = self.get_energy(rho_S)
        
        # E_final (Energy of System after Feedback)
        sv_final = result.data(0)["final"]
        rho_final_full = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final_full, [0]) # Trace out Ancilla (reset)
        
        E_after = self.get_energy(rho_S_final)
        
        extracted_work = E_before - E_after
        
        return mutual_info, extracted_work

    def run_control_experiment(self):
        print("="*60)
        print("EXPERIMENT 0: NON-INTERACTING EQUATION OF STATE (Work vs Info)")
        print(f"Fixed Kick Strength: {KICK_STRENGTH}")
        print("="*60)
        
        taus = np.linspace(0.0, MAX_TAU, TAU_STEPS)
        info_data = []
        work_data = []
        
        print(f"{'Tau':<10} | {'Mutual Info (bits)':<20} | {'Work Extracted':<20}")
        print("-" * 60)
        
        for tau in taus:
            mi, work = self.run_control_cycle(tau)
            info_data.append(mi)
            work_data.append(work)
            print(f"{tau:<10.4f} | {mi:<20.6f} | {work:<20.6f}")
            
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(info_data, work_data)
        
        print("-" * 60)
        print(f"Fit Results: Work = {slope:.4f} * I + {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(info_data, work_data, color='blue', label='Experimental Data')
        
        # Plot Fit Line
        x_fit = np.linspace(min(info_data), max(info_data), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fit (R²={r_value**2:.2f})')
        
        plt.xlabel('Mutual Information I(S:A) [bits]')
        plt.ylabel('Extracted Work -Δ<H>')
        plt.title('Thermodynamic Constitutive Law\n(Decoupled Sensing/Actuation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('thermo_control_equation_of_state.png')
        print("Saved plot to 'thermo_control_equation_of_state.png'")

    def run_experiment(self):
        print("="*60)
        print("EXPERIMENT 1: EQUATION OF STATE (Work vs Info)")
        print(f"Fixed Kick Strength: {KICK_STRENGTH}")
        print("="*60)
        
        taus = np.linspace(0.0, MAX_TAU, TAU_STEPS)
        info_data = []
        work_data = []
        
        print(f"{'Tau':<10} | {'Mutual Info (bits)':<20} | {'Work Extracted':<20}")
        print("-" * 60)
        
        for tau in taus:
            mi, work = self.run_cycle(tau)
            info_data.append(mi)
            work_data.append(work)
            print(f"{tau:<10.4f} | {mi:<20.6f} | {work:<20.6f}")
            
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(info_data, work_data)
        
        print("-" * 60)
        print(f"Fit Results: Work = {slope:.4f} * I + {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(info_data, work_data, color='blue', label='Experimental Data')
        
        # Plot Fit Line
        x_fit = np.linspace(min(info_data), max(info_data), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Fit (R²={r_value**2:.2f})')
        
        plt.xlabel('Mutual Information I(S:A) [bits]')
        plt.ylabel('Extracted Work -Δ<H>')
        plt.title('Thermodynamic Constitutive Law\n(Decoupled Sensing/Actuation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('thermo_equation_of_state.png')
        print("Saved plot to 'thermo_equation_of_state.png'")

if __name__ == "__main__":
    exp = MaxwellDemonExperiment(N_QUBITS)
    exp.run_control_experiment()
    exp.run_experiment()