"""
topology_vs_disorder_test.py

FIX 1: SEPARATING TOPOLOGY VS DISORDER EFFECTS
==============================================

The original experiment conflated two effects:
1. Graph topology (complete vs chain)
2. Coupling disorder (uniform vs random)

This test separates them with 3 conditions:
- Uniform Complete Graph: J = -1.0, all-to-all (original "ordered")
- Random Complete Graph: J = random, all-to-all (isolates disorder effect)
- Uniform Chain: J = -1.0, nearest-neighbor (isolates topology effect)

Expected Results:
- If DISORDER drives DLA explosion: Random Complete >> Uniform Complete
- If TOPOLOGY drives efficiency: Complete >> Chain (even with same couplings)

Author: Critical Analysis Fix
Date: 2025
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
N_RANGE = range(3, 9)  # System sizes
TRIALS = 10
TAU_STEPS = 8
MAX_TAU = 1.0
KICK_STRENGTH = 0.3

# ==============================================================================
# EXPERIMENT
# ==============================================================================

class TopologyDisorderExperiment:
    def __init__(self):
        self.backend = AerSimulator(method='statevector')
    
    def build_hamiltonian(self, n, model_type, seed=42):
        """
        Build Hamiltonian for 3 different conditions:
        - 'uniform_complete': J = -1.0, all-to-all ZZ + random X fields
        - 'random_complete': J = random, all-to-all ZZ + random X fields
        - 'uniform_chain': J = -1.0, nearest-neighbor ZZ + random X fields
        """
        np.random.seed(seed)
        ops = []
        
        if model_type == 'uniform_complete':
            # All-to-all with uniform couplings
            for i in range(n):
                for j in range(i+1, n):
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    ops.append(("".join(label[::-1]), -1.0))
                    
        elif model_type == 'random_complete':
            # All-to-all with random couplings (spin glass)
            for i in range(n):
                for j in range(i+1, n):
                    J = np.random.uniform(-1.0, 1.0)
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    ops.append(("".join(label[::-1]), J))
                    
        elif model_type == 'uniform_chain':
            # Nearest-neighbor with uniform couplings
            for i in range(n - 1):
                label = ["I"] * n
                label[i] = "Z"
                label[i+1] = "Z"
                ops.append(("".join(label[::-1]), -1.0))
        
        # All models get random transverse fields
        for i in range(n):
            h = np.random.uniform(-0.5, 0.5)
            label = ["I"] * n
            label[i] = "X"
            ops.append(("".join(label[::-1]), h))
        
        return SparsePauliOp.from_list(ops)
    
    def get_energy(self, state, H):
        return state.expectation_value(H).real
    
    def run_efficiency_sweep(self, n, model_type, seed):
        """Sweep τ and compute dW/dI (efficiency)."""
        H = self.build_hamiltonian(n, model_type, seed)
        
        info_data = []
        work_data = []
        
        for tau in np.linspace(0.1, MAX_TAU, TAU_STEPS):
            # Build circuit
            qr_sys = QuantumRegister(n, 'sys')
            qr_anc = QuantumRegister(1, 'anc')
            qc = QuantumCircuit(qr_anc, qr_sys)
            
            qc.h(qr_sys)
            qc.h(qr_anc)
            
            # Sensing
            evo = PauliEvolutionGate(H, time=tau)
            qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
            qc.h(qr_anc)
            qc.save_statevector(label="post_sensing")
            
            # Feedback
            for i in range(n):
                qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])
            qc.save_statevector(label="final")
            
            # Execute
            t_qc = transpile(qc, self.backend)
            result = self.backend.run(t_qc).result()
            
            # Analysis
            sv_sensing = result.data(0)["post_sensing"]
            rho_sensing = DensityMatrix(sv_sensing)
            
            S_SA = entropy(rho_sensing, base=2)
            rho_S = partial_trace(rho_sensing, [0])
            rho_A = partial_trace(rho_sensing, range(1, n+1))
            S_S = entropy(rho_S, base=2)
            S_A = entropy(rho_A, base=2)
            mutual_info = S_S + S_A - S_SA
            
            E_before = self.get_energy(rho_S, H)
            sv_final = result.data(0)["final"]
            rho_final = DensityMatrix(sv_final)
            rho_S_final = partial_trace(rho_final, [0])
            E_after = self.get_energy(rho_S_final, H)
            
            work = E_before - E_after
            
            if mutual_info > 0.01:
                info_data.append(mutual_info)
                work_data.append(work)
        
        if len(info_data) > 2:
            slope, _, r, _, _ = linregress(info_data, work_data)
            return slope, r**2
        return 0.0, 0.0
    
    def run_experiment(self):
        print("=" * 70)
        print("TOPOLOGY vs DISORDER: SEPARATING EFFECTS")
        print("=" * 70)
        
        models = ['uniform_complete', 'random_complete', 'uniform_chain']
        labels = ['Uniform Complete (K_n)', 'Random Complete (Spin Glass)', 'Uniform Chain']
        colors = ['blue', 'red', 'green']
        
        results = {m: {'n': [], 'eta': [], 'eta_std': []} for m in models}
        
        for n in N_RANGE:
            print(f"\n--- N = {n} ---")
            
            for model in models:
                etas = []
                for trial in range(TRIALS):
                    eta, r2 = self.run_efficiency_sweep(n, model, seed=trial*100)
                    etas.append(eta)
                
                mean_eta = np.mean(etas)
                std_eta = np.std(etas)
                
                results[model]['n'].append(n)
                results[model]['eta'].append(mean_eta)
                results[model]['eta_std'].append(std_eta)
                
                print(f"  {model:20s}: η = {mean_eta:.4f} ± {std_eta:.4f}")
        
        # Analysis
        print("\n" + "=" * 70)
        print("ANALYSIS: SEPARATING EFFECTS")
        print("=" * 70)
        
        # Compare at largest N
        n_max = max(N_RANGE)
        idx = -1
        
        uniform_complete = results['uniform_complete']['eta'][idx]
        random_complete = results['random_complete']['eta'][idx]
        uniform_chain = results['uniform_chain']['eta'][idx]
        
        disorder_effect = uniform_complete - random_complete
        topology_effect = uniform_complete - uniform_chain
        
        print(f"\nAt N = {n_max}:")
        print(f"  Uniform Complete η: {uniform_complete:.4f}")
        print(f"  Random Complete η:  {random_complete:.4f}")
        print(f"  Uniform Chain η:    {uniform_chain:.4f}")
        print(f"\n  DISORDER effect (Uniform - Random Complete): {disorder_effect:.4f}")
        print(f"  TOPOLOGY effect (Complete - Chain):          {topology_effect:.4f}")
        
        if abs(disorder_effect) > abs(topology_effect):
            print("\n✓ DISORDER is the PRIMARY driver of efficiency difference")
        else:
            print("\n✓ TOPOLOGY is the PRIMARY driver of efficiency difference")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model in enumerate(models):
            ns = results[model]['n']
            etas = results[model]['eta']
            stds = results[model]['eta_std']
            ax.errorbar(ns, etas, yerr=stds, marker='o', markersize=8, 
                       linewidth=2, color=colors[i], label=labels[i], capsize=4)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('System Size N', fontsize=12)
        ax.set_ylabel('Efficiency η = dW/dI', fontsize=12)
        ax.set_title('Separating Topology vs Disorder Effects', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('topology_vs_disorder_result.png', dpi=150, bbox_inches='tight')
        print("\n[Saved] topology_vs_disorder_result.png")
        plt.show()
        
        return results


if __name__ == "__main__":
    exp = TopologyDisorderExperiment()
    results = exp.run_experiment()
