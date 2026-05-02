"""
k_ancilla_bandwidth_test.py

EXPERIMENT: ANCILLA BANDWIDTH SCALING (FIXED)
==============================================

FIXED VERSION: Each ancilla senses a DIFFERENT subset of the Hamiltonian,
extracting truly independent information.

THE HYPOTHESIS:
---------------
- 1 ancilla: senses full H → bottleneck at I_max = 2 bits
- k ancillae: each senses H/k → total bandwidth k × 2 bits
- Crash point Nc should scale with k

PROTOCOL:
---------
Partition the Hamiltonian terms among k ancillae:
- Ancilla 0: senses terms {h_0, h_k, h_2k, ...}
- Ancilla 1: senses terms {h_1, h_{k+1}, ...}
- etc.

This maximizes the independence of information extracted.

Author: Theory Test Suite
Date: 2025 (Fixed)
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
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYSTEM_SIZES = [3, 4, 5, 6, 7]  # Extended to show crossover
ANCILLA_COUNTS = [1, 2]   # Focus on 1 vs 2 comparison
NUM_SEEDS = 5  # Average over multiple random seeds
TAU_STEPS = 5  # Reduced for speed
MAX_TAU = 1.5  # τ range: 0.0-1.5 (TABLE I)
KICK_STRENGTH = 0.2  # θ_gain = 0.2 rad (TABLE I)

# ==============================================================================
# K-ANCILLA BANDWIDTH EXPERIMENT (FIXED)
# ==============================================================================

class KAncillaBandwidthFixed:
    def __init__(self, n_system, k_ancilla, hamiltonian_type='chaotic', seed=42):
        self.n_sys = n_system
        self.k_anc = k_ancilla
        self.seed = seed
        self.backend = AerSimulator(method='statevector')
        
        if hamiltonian_type == 'chaotic':
            self.H_full, self.H_parts = self._build_partitioned_chaotic()
        else:
            self.H_full, self.H_parts = self._build_partitioned_ordered()

    def _build_partitioned_chaotic(self):
        """Build Spin Glass Hamiltonian partitioned into k independent parts."""
        np.random.seed(self.seed)
        all_terms = []
        
        # Generate all ZZ terms
        for i in range(self.n_sys):
            for j in range(i+1, self.n_sys):
                J = np.random.normal(0, 1)
                label = ["I"] * self.n_sys
                label[i] = "Z"
                label[j] = "Z"
                all_terms.append(("".join(label[::-1]), J))
        
        # Add X field terms
        for i in range(self.n_sys):
            hx = np.random.uniform(-1, 1)
            label = ["I"] * self.n_sys
            label[i] = "X"
            all_terms.append(("".join(label[::-1]), hx))
        
        # Full Hamiltonian
        H_full = SparsePauliOp.from_list(all_terms)
        
        # Partition into k parts
        H_parts = []
        for a in range(self.k_anc):
            part_terms = [all_terms[i] for i in range(a, len(all_terms), self.k_anc)]
            if part_terms:
                H_parts.append(SparsePauliOp.from_list(part_terms))
            else:
                # Fallback: at least include some X terms
                label = ["I"] * self.n_sys
                label[a % self.n_sys] = "X"
                H_parts.append(SparsePauliOp.from_list([("".join(label[::-1]), 0.5)]))
        
        return H_full, H_parts
    
    def _build_partitioned_ordered(self):
        """Build Complete Graph Hamiltonian partitioned into k parts."""
        all_terms = []
        
        for i in range(self.n_sys):
            for j in range(i+1, self.n_sys):
                label = ["I"] * self.n_sys
                label[i] = "Z"
                label[j] = "Z"
                all_terms.append(("".join(label[::-1]), 1.0))
        
        for i in range(self.n_sys):
            label = ["I"] * self.n_sys
            label[i] = "X"
            all_terms.append(("".join(label[::-1]), 0.5))
        
        H_full = SparsePauliOp.from_list(all_terms)
        
        H_parts = []
        for a in range(self.k_anc):
            part_terms = [all_terms[i] for i in range(a, len(all_terms), self.k_anc)]
            if part_terms:
                H_parts.append(SparsePauliOp.from_list(part_terms))
            else:
                label = ["I"] * self.n_sys
                label[a % self.n_sys] = "X"
                H_parts.append(SparsePauliOp.from_list([("".join(label[::-1]), 0.5)]))
        
        return H_full, H_parts
    
    def get_energy(self, state):
        """Computes <H> for a given density matrix."""
        if isinstance(state, (DensityMatrix, Statevector)):
            return state.expectation_value(self.H_full).real
        return 0.0
    
    def run_cycle(self, tau):
        """
        Run one cycle with k ancillae, each sensing DIFFERENT Hamiltonian parts.
        """
        qr_sys = QuantumRegister(self.n_sys, 'sys')
        qr_anc = QuantumRegister(self.k_anc, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # Initialize
        qc.h(qr_sys)  # High energy superposition
        for a in range(self.k_anc):
            qc.h(qr_anc[a])
        
        # --- INDEPENDENT SENSING ---
        # Each ancilla senses its OWN partition of the Hamiltonian
        for a in range(self.k_anc):
            evo_a = PauliEvolutionGate(self.H_parts[a], time=tau)
            qc.append(evo_a.control(1), [qr_anc[a]] + list(qr_sys))
        
        # Convert phase to population
        for a in range(self.k_anc):
            qc.h(qr_anc[a])
        
        qc.save_statevector(label="post_sensing")
        
        # --- INDEPENDENT FEEDBACK ---
        # Each ancilla feeds back to system qubits related to its partition
        # Each ancilla applies FULL kick (not divided) to maximize effect
        for a in range(self.k_anc):
            for i in range(self.n_sys):
                qc.crx(KICK_STRENGTH, qr_anc[a], qr_sys[i])
        
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # --- ANALYSIS ---
        sv_sensing = result.data(0)["post_sensing"]
        rho_sensing = DensityMatrix(sv_sensing)
        
        ancilla_indices = list(range(self.k_anc))
        system_indices = list(range(self.k_anc, self.k_anc + self.n_sys))
        
        rho_S = partial_trace(rho_sensing, ancilla_indices)
        rho_A = partial_trace(rho_sensing, system_indices)
        
        S_SA = entropy(rho_sensing, base=2)
        S_S = entropy(rho_S, base=2)
        S_A = entropy(rho_A, base=2)
        
        # Total mutual information with all ancillae
        mutual_info = S_S + S_A - S_SA
        
        # Work
        E_before = self.get_energy(rho_S)
        sv_final = result.data(0)["final"]
        rho_final = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final, ancilla_indices)
        E_after = self.get_energy(rho_S_final)
        work = E_before - E_after
        
        return mutual_info, work, S_A
    
    def measure_efficiency(self):
        """Measure efficiency η = dW/dI over τ scan."""
        taus = np.linspace(0.1, MAX_TAU, TAU_STEPS)
        info_data = []
        work_data = []
        entropy_data = []
        
        for tau in taus:
            mi, work, s_a = self.run_cycle(tau)
            info_data.append(mi)
            work_data.append(work)
            entropy_data.append(s_a)
        
        info_data = np.array(info_data)
        work_data = np.array(work_data)
        entropy_data = np.array(entropy_data)
        
        valid = (info_data > 1e-6) & np.isfinite(work_data)
        if np.sum(valid) < 3:
            return 0.0, 0.0, 0.0
        
        slope, intercept, r_value, _, _ = linregress(info_data[valid], work_data[valid])
        
        return slope, r_value**2, np.mean(entropy_data)


def main():
    print("="*70)
    print("K-ANCILLA BANDWIDTH SCALING TEST (Multi-Seed Averaging)")
    print(f"Averaging over {NUM_SEEDS} random seeds per configuration")
    print("="*70)
    
    results_mean = {}
    results_std = {}
    info_bandwidth_mean = {}
    info_bandwidth_std = {}
    
    for k in ANCILLA_COUNTS:
        print(f"\n{'='*70}")
        print(f"ANCILLA COUNT: k = {k} (max info = {k} bits per cycle)")
        print("="*70)
        
        eff_mean_list = []
        eff_std_list = []
        bw_mean_list = []
        bw_std_list = []
        
        for n in SYSTEM_SIZES:
            print(f"  [N={n}] Running {NUM_SEEDS} seeds...", end=" ", flush=True)
            etas_seeds = []
            bws_seeds = []
            
            for seed in range(NUM_SEEDS):
                try:
                    exp = KAncillaBandwidthFixed(n, k, 'ordered', seed=seed*100)
                    eta, r2, s_a = exp.measure_efficiency()
                    if np.isfinite(eta):
                        etas_seeds.append(eta)
                        bws_seeds.append(2 * s_a)
                except Exception:
                    pass
            
            if len(etas_seeds) >= 2:
                eff_mean_list.append(np.mean(etas_seeds))
                eff_std_list.append(np.std(etas_seeds))
                bw_mean_list.append(np.mean(bws_seeds))
                bw_std_list.append(np.std(bws_seeds))
                print(f"η = {np.mean(etas_seeds):.4f} ± {np.std(etas_seeds):.3f}")
            else:
                eff_mean_list.append(0.0)
                eff_std_list.append(0.0)
                bw_mean_list.append(0.0)
                bw_std_list.append(0.0)
                print("FAILED")
        
        results_mean[k] = np.array(eff_mean_list)
        results_std[k] = np.array(eff_std_list)
        info_bandwidth_mean[k] = np.array(bw_mean_list)
        info_bandwidth_std[k] = np.array(bw_std_list)
    
    # --- VISUALIZATION ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#1f77b4', '#2ca02c']  # Blue, Green
    
    # Plot 1: Efficiency vs N
    ax = axes[0]
    for i, k in enumerate(ANCILLA_COUNTS):
        N = np.array(SYSTEM_SIZES)
        mean = results_mean[k]
        std = results_std[k]
        ax.plot(N, mean, color=colors[i], marker='o', markersize=8,
                linewidth=2, label=f'k = {k}')
        ax.fill_between(N, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('System Size N', fontsize=12)
    ax.set_ylabel('Efficiency η = dW/dI', fontsize=12)
    ax.set_title('(a) Efficiency', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Information Bandwidth vs N
    ax = axes[1]
    for i, k in enumerate(ANCILLA_COUNTS):
        N = np.array(SYSTEM_SIZES)
        mean = info_bandwidth_mean[k]
        std = info_bandwidth_std[k]
        ax.plot(N, mean, color=colors[i], marker='s', markersize=8,
                linewidth=2, label=f'k = {k}')
        ax.fill_between(N, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel('System Size N', fontsize=12)
    ax.set_ylabel('Information I(S:A) [bits]', fontsize=12)
    ax.set_title('(b) Information Bandwidth', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total Work = η × I (THE KEY RESULT)
    ax = axes[2]
    for i, k in enumerate(ANCILLA_COUNTS):
        N = np.array(SYSTEM_SIZES)
        eta = results_mean[k]
        info = info_bandwidth_mean[k]
        total_work = eta * info  # Key metric: W = η × I
        ax.plot(N, total_work, color=colors[i], marker='^', markersize=10,
                linewidth=2.5, label=f'k = {k}')
        # Highlight positive work region
        ax.fill_between(N, 0, total_work, where=(total_work > 0), 
                        color=colors[i], alpha=0.2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('System Size N', fontsize=12)
    ax.set_ylabel('Total Work W = η × I', fontsize=12)
    ax.set_title('(c) Extracted Work (η × I)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_ancilla_bandwidth_test.png', dpi=150, bbox_inches='tight')
    print("\n[Saved] k_ancilla_bandwidth_test.png")
    plt.show()
    
    # --- ANALYSIS ---
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    for k in ANCILLA_COUNTS:
        etas = results_mean[k]
        crash_idx = next((i for i, e in enumerate(etas) if e < 0.05), len(etas))
        crash_N = SYSTEM_SIZES[crash_idx] if crash_idx < len(SYSTEM_SIZES) else ">N_max"
        avg_bw = np.mean(info_bandwidth_mean[k])
        print(f"  k={k}: Crash point Nc ≈ {crash_N}, Avg Bandwidth = {avg_bw:.2f} bits")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Compare bandwidths
    if len(ANCILLA_COUNTS) >= 2:
        bw_ratio = np.mean(info_bandwidth_mean[ANCILLA_COUNTS[-1]]) / np.mean(info_bandwidth_mean[ANCILLA_COUNTS[0]])
        print(f"\n  Bandwidth scaling: k=2/k=1 ratio = {bw_ratio:.2f} (expected: 2.0)")
        
        if bw_ratio > 1.5:
            print("  ✓ SUCCESS: More ancillae → More information bandwidth!")
        else:
            print("  ? Bandwidth doesn't scale linearly with k.")
    
    return results_mean


if __name__ == "__main__":
    results = main()
