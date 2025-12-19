"""
multi_cycle_reset_test.py

EXPERIMENT: COHERENT VS MEASURED FEEDBACK (v4)
==============================================

TRUE DISTINCTION:
-----------------
A) COHERENT: Keep the FULL statevector (system+ancilla) across iterations
   → Preserves quantum correlations
   → But ancilla entropy accumulates

B) MEASURED: Project ancilla, reset to |0⟩, restart with system state
   → Classical feedback loop
   → Pays Landauer cost but fresh ancilla each time

This shows whether sustained entanglement helps or hurts.

Author: Theory Test Suite
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import (
    SparsePauliOp, partial_trace, entropy, 
    DensityMatrix, Statevector
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 3  # Smaller for full statevector tracking
N_CYCLES = 8
TAU = 0.4
KICK_STRENGTH = 0.4
LN2 = np.log(2)

# ==============================================================================
# EXPERIMENT
# ==============================================================================

class CoherentVsMeasuredExperiment:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.backend = AerSimulator(method='statevector')
        self.H = self._build_hamiltonian()
        print(f"[Init] System N={n_qubits}.")
    
    def _build_hamiltonian(self):
        """Complete graph with transverse field."""
        ops = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                label = ["I"] * self.n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), 1.0))
        for i in range(self.n):
            label = ["I"] * self.n
            label[i] = "X"
            ops.append(("".join(label[::-1]), 0.5))
        return SparsePauliOp.from_list(ops)
    
    def get_energy(self, state):
        """Get energy from statevector or density matrix."""
        if isinstance(state, np.ndarray):
            sv = Statevector(np.ascontiguousarray(state))
            return sv.expectation_value(self.H).real
        elif isinstance(state, DensityMatrix):
            return state.expectation_value(self.H).real
        return 0.0
    
    def run_coherent_mode(self, n_steps):
        """
        COHERENT MODE: Keep full statevector across all iterations.
        
        Circuit grows: we chain multiple sensing+feedback without measurement.
        """
        # Initial: system in |+⟩^n, ancilla in |0⟩
        dim_sys = 2**self.n
        psi_sys = np.ones(dim_sys, dtype=complex) / np.sqrt(dim_sys)
        psi_anc = np.array([1, 0], dtype=complex)  # |0⟩
        
        # Full state: ancilla ⊗ system
        psi_full = np.kron(psi_anc, psi_sys)
        
        results = {
            'energy': [],
            'work': [],
            'info': [],
            'ancilla_entropy': []
        }
        
        for step in range(n_steps):
            # Build circuit that operates on full state
            qr_anc = QuantumRegister(1, 'anc')
            qr_sys = QuantumRegister(self.n, 'sys')
            qc = QuantumCircuit(qr_anc, qr_sys)
            
            qc.initialize(psi_full, list(qr_anc) + list(qr_sys))
            
            # Hadamard on ancilla (creates superposition for sensing)
            qc.h(qr_anc)
            
            # Controlled evolution
            evo = PauliEvolutionGate(self.H, time=TAU)
            qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
            
            qc.h(qr_anc)  # Phase → population
            
            # Feedback
            for i in range(self.n):
                qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
            
            qc.save_statevector()
            
            t_qc = transpile(qc, self.backend)
            res = self.backend.run(t_qc).result()
            psi_full = res.get_statevector().data
            
            # Analysis
            rho_full = DensityMatrix(psi_full)
            rho_sys = partial_trace(rho_full, [0])
            rho_anc = partial_trace(rho_full, range(1, self.n + 1))
            
            E_sys = rho_sys.expectation_value(self.H).real
            S_anc = entropy(rho_anc)
            S_joint = entropy(rho_full)
            S_sys = entropy(rho_sys)
            MI = S_sys + S_anc - S_joint
            
            results['energy'].append(E_sys)
            results['ancilla_entropy'].append(S_anc)
            results['info'].append(MI)
            
            if step == 0:
                results['work'].append(0)
            else:
                results['work'].append(results['energy'][-2] - E_sys)
        
        return results
    
    def run_measured_mode(self, n_steps):
        """
        MEASURED MODE: Measure ancilla and reset each iteration.
        
        After each cycle:
        1. Measure ancilla (collapse)
        2. Reset ancilla to |0⟩
        3. Keep system state and continue
        """
        dim_sys = 2**self.n
        psi_sys = np.ones(dim_sys, dtype=complex) / np.sqrt(dim_sys)
        
        results = {
            'energy': [],
            'work': [],
            'info': [],
            'ancilla_entropy': [],
            'erasure_cost': []
        }
        
        current_sys = psi_sys
        
        for step in range(n_steps):
            qr_anc = QuantumRegister(1, 'anc')
            qr_sys = QuantumRegister(self.n, 'sys')
            qc = QuantumCircuit(qr_anc, qr_sys)
            
            # Initialize system with current state, ancilla fresh |0⟩
            qc.initialize(current_sys, qr_sys)
            # Ancilla starts in |0⟩ by default
            
            qc.h(qr_anc)
            evo = PauliEvolutionGate(self.H, time=TAU)
            qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
            qc.h(qr_anc)
            
            for i in range(self.n):
                qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
            
            qc.save_statevector()
            
            t_qc = transpile(qc, self.backend)
            res = self.backend.run(t_qc).result()
            psi_full = res.get_statevector().data
            
            # Analysis
            rho_full = DensityMatrix(psi_full)
            rho_sys = partial_trace(rho_full, [0])
            rho_anc = partial_trace(rho_full, range(1, self.n + 1))
            
            E_sys = rho_sys.expectation_value(self.H).real
            S_anc = entropy(rho_anc)
            S_joint = entropy(rho_full)
            S_sys = entropy(rho_sys)
            MI = S_sys + S_anc - S_joint
            
            results['energy'].append(E_sys)
            results['ancilla_entropy'].append(S_anc)
            results['info'].append(MI)
            results['erasure_cost'].append(LN2 * S_anc)
            
            if step == 0:
                results['work'].append(0)
            else:
                results['work'].append(results['energy'][-2] - E_sys)
            
            # MEASUREMENT: Extract system state (trace out ancilla)
            # Then reset ancilla to |0⟩ for next iteration
            eigenvalues, eigenvectors = np.linalg.eigh(rho_sys.data)
            current_sys = np.ascontiguousarray(eigenvectors[:, -1])
        
        return results
    
    def run_experiment(self):
        """Run both modes and compare."""
        print("\n" + "="*70)
        print("COHERENT vs MEASURED FEEDBACK")
        print("="*70)
        
        print("\n[Coherent] Running...")
        coherent = self.run_coherent_mode(N_CYCLES)
        
        print("[Measured] Running...")
        measured = self.run_measured_mode(N_CYCLES)
        
        # Results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        init_E = self.get_energy(np.ones(2**self.n) / np.sqrt(2**self.n))
        
        print(f"\n[Coherent Mode]")
        print(f"  Initial → Final Energy: {init_E:.3f} → {coherent['energy'][-1]:.3f}")
        print(f"  Avg Mutual Info: {np.mean(coherent['info']):.4f}")
        print(f"  Final Ancilla Entropy: {coherent['ancilla_entropy'][-1]:.4f}")
        
        print(f"\n[Measured Mode]")
        print(f"  Initial → Final Energy: {init_E:.3f} → {measured['energy'][-1]:.3f}")
        print(f"  Avg Mutual Info: {np.mean(measured['info']):.4f}")
        print(f"  Total Erasure Cost: {sum(measured['erasure_cost']):.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        steps = range(1, N_CYCLES + 1)
        
        # Energy
        ax = axes[0]
        ax.plot(steps, coherent['energy'], 'bo-', label='Coherent', markersize=8, linewidth=2)
        ax.plot(steps, measured['energy'], 'rs--', label='Measured', markersize=8, linewidth=2)
        ax.axhline(init_E, color='gray', linestyle=':', alpha=0.5, label='Initial')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('System Energy ⟨H⟩', fontsize=12)
        ax.set_title('Energy Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mutual Information
        ax = axes[1]
        ax.plot(steps, coherent['info'], 'bo-', label='Coherent', markersize=8, linewidth=2)
        ax.plot(steps, measured['info'], 'rs--', label='Measured', markersize=8, linewidth=2)
        ax.axhline(2.0, color='green', linestyle='--', alpha=0.5, label='Max I(S:A)=2')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Mutual Information', fontsize=12)
        ax.set_title('Information per Step', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ancilla Entropy
        ax = axes[2]
        ax.plot(steps, coherent['ancilla_entropy'], 'bo-', label='Coherent', markersize=8, linewidth=2)
        ax.plot(steps, measured['ancilla_entropy'], 'rs--', label='Measured', markersize=8, linewidth=2)
        ax.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Max S(A)=1')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Ancilla Entropy S(A)', fontsize=12)
        ax.set_title('Ancilla Saturation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_cycle_reset_test.png', dpi=150, bbox_inches='tight')
        print("\n[Saved] multi_cycle_reset_test.png")
        plt.show()
        
        # Conclusion
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        
        coh_drop = init_E - coherent['energy'][-1]
        meas_drop = init_E - measured['energy'][-1]
        
        print(f"\n  Coherent dropped: {coh_drop:.4f}")
        print(f"  Measured dropped: {meas_drop:.4f}")
        
        if coherent['ancilla_entropy'][-1] > 0.8:
            print("\n  ⚠ Coherent mode: Ancilla SATURATED (S ≈ 1)")
        
        if coh_drop > meas_drop * 1.1:
            print("\n✓ COHERENT mode cools more efficiently")
            print("  → Sustained entanglement helps despite ancilla saturation")
        elif meas_drop > coh_drop * 1.1:
            print("\n✓ MEASURED mode cools more efficiently")
            print("  → Fresh ancilla each cycle overcomes erasure cost")
        else:
            print("\n≈ Both modes perform similarly")
        
        return coherent, measured


if __name__ == "__main__":
    exp = CoherentVsMeasuredExperiment(N_QUBITS)
    results = exp.run_experiment()
