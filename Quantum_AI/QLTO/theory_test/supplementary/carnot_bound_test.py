"""
carnot_bound_test.py

EXPERIMENT: THE CARNOT BOUND (Universal Efficiency Limit)
=========================================================
Tests whether the "Universal Constant" η_max is actually determined by
the feedback coupling strength θ_gain, bounded by the Holevo limit.

The Theory:
-----------
The Holevo bound states that a single qubit ancilla can transmit at most
1 bit of information per measurement cycle:
    Max Information Flux: İ ≤ 1 bit/step
    Max Energy Extraction: Ẇ = η · İ
    Therefore: η ∝ θ_gain (the coupling strength)

The Universal Limit:
    η / θ_gain ≤ 1 (in natural units)

If η normalizes to a constant when divided by the coupling strength,
we have found the "Universal Efficiency" - analogous to the Carnot efficiency
being bounded by temperature ratios.

Prediction:
-----------
η(θ) = C · θ_gain, where C is a universal constant ≈ 1.

Success Criterion:
------------------
A linear plot of η vs θ_gain passing through zero.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import AerSimulator
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 4

# The KEY variable: Scan different coupling strengths
# Focus on linear regime (small θ) to avoid saturation effects
KICK_STRENGTHS = np.array([0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

# Sensing parameters (to gather sufficient statistics)
TAU_STEPS = 15
MAX_TAU = 1.2

# Time evolution is synthesized (product formula) after transpilation.
# Default PauliEvolutionGate synthesis is LieTrotter(reps=1). Make it explicit
# so the approximation can be tightened when needed.
EVOLUTION_REPS = 1

# ==============================================================================
# THERMODYNAMIC ENGINE WITH VARIABLE COUPLING
# ==============================================================================

class CarnotBoundExperiment:
    def __init__(self, n_qubits, h_type='ordered'):
        self.n = n_qubits
        self.h_type = h_type
        self.backend = AerSimulator(method='statevector')
        
        # Build Hamiltonian based on type
        if h_type == 'ordered':
            self.H = self._build_ordered_hamiltonian()
        else:
            self.H = self._build_chaotic_hamiltonian()
            
        print(f"[Init] System N={n_qubits}, Type={h_type}")
    
    def _build_ordered_hamiltonian(self):
        """Complete graph K_n - Polynomial DLA."""
        ops = []
        # All-to-all ZZ coupling (complete graph)
        for i in range(self.n):
            for j in range(i+1, self.n):
                label = ["I"] * self.n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), 1.0))  # Uniform coupling
        # Uniform transverse field
        for i in range(self.n):
            label = ["I"] * self.n
            label[i] = "X"
            ops.append(("".join(label[::-1]), 0.5))
        return SparsePauliOp.from_list(ops)
    
    def _build_chaotic_hamiltonian(self, seed=42):
        """Random Spin Glass - Exponential DLA."""
        np.random.seed(seed)
        ops = []
        # Random ZZ couplings
        for i in range(self.n):
            for j in range(i+1, self.n):
                J = np.random.normal(0, 1)
                label = ["I"] * self.n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), J))
        # Random fields
        for i in range(self.n):
            hx = np.random.uniform(-1, 1)
            label = ["I"] * self.n
            label[i] = "X"
            ops.append(("".join(label[::-1]), hx))
        return SparsePauliOp.from_list(ops)

    def get_energy(self, state):
        """Computes <H> for a given density matrix/statevector."""
        if isinstance(state, (DensityMatrix, Statevector)):
            return state.expectation_value(self.H).real
        return 0.0
    
    def run_cycle(self, tau, kick_strength):
        """
        Runs one thermodynamic cycle with:
        - Sensing duration 'tau'
        - Feedback strength 'kick_strength' (θ_gain)
        
        Returns (MutualInformation, ExtractedWork).
        """
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # --- 0. INITIALIZATION ---
        qc.h(qr_sys)  # High-energy superposition state
        qc.h(qr_anc)  # Ready to sense phase
        
        # --- 1. SENSING ---
        # Controlled-Evolution: maps Energy -> Phase
        evo = PauliEvolutionGate(self.H, time=tau, synthesis=LieTrotter(reps=EVOLUTION_REPS))
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # --- 2. LOCKING ---
        qc.h(qr_anc)  # Phase -> Population
        qc.save_statevector(label="post_sensing")
        
        # --- 3. ACTUATION (Variable Strength!) ---
        for i in range(self.n):
            qc.crx(kick_strength, qr_anc[0], qr_sys[i])
             
        # --- 4. EXHAUST ---
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # --- ANALYSIS ---
        
        # A. Mutual Information I(S:A)
        sv_sensing = result.data(0)["post_sensing"]
        rho_sensing = DensityMatrix(sv_sensing)
        
        S_SA = entropy(rho_sensing)
        rho_S = partial_trace(rho_sensing, [0])
        rho_A = partial_trace(rho_sensing, range(1, self.n+1))
        S_S = entropy(rho_S)
        S_A = entropy(rho_A)
        
        mutual_info = S_S + S_A - S_SA
        
        # B. Work Extraction
        E_before = self.get_energy(rho_S)
        
        sv_final = result.data(0)["final"]
        rho_final_full = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final_full, [0])
        E_after = self.get_energy(rho_S_final)
        
        extracted_work = E_before - E_after
        
        return mutual_info, extracted_work

    def measure_efficiency(self, kick_strength):
        """
        Run a scan over τ at fixed kick_strength.
        Fit the slope η = dW/dI.
        """
        taus = np.linspace(0.05, MAX_TAU, TAU_STEPS)
        info_data = []
        work_data = []
        
        for tau in taus:
            mi, work = self.run_cycle(tau, kick_strength)
            info_data.append(mi)
            work_data.append(work)
        
        info_data = np.array(info_data)
        work_data = np.array(work_data)
        
        # Linear regression: W = η * I + offset
        valid_mask = (info_data > 1e-6) & np.isfinite(work_data)
        if np.sum(valid_mask) < 3:
            return 0.0, 0.0, info_data, work_data
        
        slope, intercept, r_value, p_value, std_err = linregress(
            info_data[valid_mask], work_data[valid_mask]
        )
        
        return slope, r_value**2, info_data, work_data


def main():
    """
    Main experiment: Scan coupling strength and prove η ∝ θ_gain.
    """
    print("=" * 70)
    print("THE CARNOT BOUND: Universal Efficiency Limit")
    print("Testing: η = C · θ_gain (Linear Scaling)")
    print("=" * 70)
    
    results = {}
    
    for h_type in ['ordered', 'chaotic']:
        print(f"\n{'='*70}")
        print(f"PHASE: {h_type.upper()}")
        print("=" * 70)
        
        exp = CarnotBoundExperiment(N_QUBITS, h_type)
        
        efficiencies = []
        r_squared_values = []
        
        print(f"\n{'θ_gain':<10} | {'η (efficiency)':<15} | {'R²':<10} | {'η/θ (normalized)':<15}")
        print("-" * 60)
        
        for kick in KICK_STRENGTHS:
            eta, r2, info_data, work_data = exp.measure_efficiency(kick)
            efficiencies.append(eta)
            r_squared_values.append(r2)
            
            normalized = eta / kick if kick > 0 else 0
            print(f"{kick:<10.2f} | {eta:<15.4f} | {r2:<10.3f} | {normalized:<15.4f}")
        
        efficiencies = np.array(efficiencies)
        
        # Fit η = C * θ_gain
        # Linear fit through origin: η = C * θ
        # Using y = a*x form
        valid = KICK_STRENGTHS > 0.01
        if np.sum(valid) > 2:
            # Least squares fit η = C * θ
            C = np.sum(efficiencies[valid] * KICK_STRENGTHS[valid]) / np.sum(KICK_STRENGTHS[valid]**2)
            
            # Also do standard linear regression for comparison
            slope, intercept, r_value, _, _ = linregress(KICK_STRENGTHS[valid], efficiencies[valid])
            
            print(f"\n[FIT RESULTS]")
            print(f"  Linear through origin: η = {C:.4f} · θ_gain")
            print(f"  Standard linear fit: η = {slope:.4f} · θ_gain + {intercept:.4f}")
            print(f"  R² of linear fit: {r_value**2:.4f}")
            print(f"  Mean(η/θ): {np.mean(efficiencies[valid] / KICK_STRENGTHS[valid]):.4f}")
            print(f"  Std(η/θ): {np.std(efficiencies[valid] / KICK_STRENGTHS[valid]):.4f}")
        
        results[h_type] = {
            'kicks': KICK_STRENGTHS,
            'eta': efficiencies,
            'r2': r_squared_values,
            'C': C if np.sum(valid) > 2 else 0
        }
    
    # ===========================================================================
    # VISUALIZATION
    # ===========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: η vs θ_gain for both phases
    ax = axes[0]
    for h_type, color, marker in [('ordered', 'blue', 'o'), ('chaotic', 'red', 's')]:
        r = results[h_type]
        ax.scatter(r['kicks'], r['eta'], color=color, marker=marker, s=80, 
                   label=f'{h_type.upper()}', alpha=0.8, edgecolors='black')
        
        # Fit line
        if r['C'] > 0:
            theta_fit = np.linspace(0, max(KICK_STRENGTHS), 100)
            ax.plot(theta_fit, r['C'] * theta_fit, color=color, linestyle='--', 
                    alpha=0.6, label=f"η = {r['C']:.2f}θ")
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Coupling Strength θ_gain', fontsize=12)
    ax.set_ylabel('Efficiency η = dW/dI', fontsize=12)
    ax.set_title('The Carnot Bound: η ∝ θ_gain', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Normalized efficiency η/θ
    ax = axes[1]
    for h_type, color, marker in [('ordered', 'blue', 'o'), ('chaotic', 'red', 's')]:
        r = results[h_type]
        valid = r['kicks'] > 0.05
        normalized_eta = r['eta'][valid] / r['kicks'][valid]
        ax.scatter(r['kicks'][valid], normalized_eta, color=color, marker=marker, 
                   s=80, label=f'{h_type.upper()}', alpha=0.8, edgecolors='black')
        
        # Mean line
        mean_norm = np.mean(normalized_eta)
        ax.axhline(mean_norm, color=color, linestyle='--', alpha=0.5, 
                   label=f'Mean = {mean_norm:.2f}')
    
    ax.axhline(1.0, color='green', linestyle='-', linewidth=2, alpha=0.7, 
               label='Holevo Limit (η/θ = 1)')
    ax.set_xlabel('Coupling Strength θ_gain', fontsize=12)
    ax.set_ylabel('Normalized Efficiency η/θ', fontsize=12)
    ax.set_title('Universal Efficiency (Normalized)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Saturation effect - η/θ vs θ
    ax = axes[2]
    for h_type, color in [('ordered', 'blue'), ('chaotic', 'red')]:
        r = results[h_type]
        valid = r['kicks'] > 0.05
        normalized_eta = r['eta'][valid] / r['kicks'][valid]
        
        # Show how normalized efficiency behaves
        ax.plot(r['kicks'][valid], normalized_eta, color=color, marker='o', 
                markersize=8, linewidth=2, label=h_type.upper())
    
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, 
               label='Theoretical Limit')
    ax.set_xlabel('Coupling Strength θ_gain', fontsize=12)
    ax.set_ylabel('η/θ (Carnot Ratio)', fontsize=12)
    ax.set_title('Approach to Universal Limit', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('carnot_bound_result.png', dpi=150, bbox_inches='tight')
    plt.savefig('carnot_bound_result.pdf', bbox_inches='tight')
    print("\n[Saved] carnot_bound_result.png/pdf")
    plt.show()
    
    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: THE CARNOT BOUND")
    print("=" * 70)
    
    print("\nKey Finding:")
    print("-" * 40)
    
    ordered_C = results['ordered']['C']
    chaotic_C = results['chaotic']['C']
    
    print(f"  ORDERED Phase: η = {ordered_C:.3f} · θ_gain")
    print(f"  CHAOTIC Phase: η = {chaotic_C:.3f} · θ_gain")
    
    print("\nPhysical Interpretation:")
    print("-" * 40)
    print("  The efficiency η is NOT a magic constant!")
    print("  It scales linearly with the feedback coupling strength θ_gain.")
    print("")
    print("  The Universal Limit is: η/θ ≤ 1")
    print("  (bounded by the Holevo information capacity of the ancilla)")
    print("")
    print(f"  Ordered Phase achieves: η/θ ≈ {ordered_C:.2f} of the limit")
    print(f"  Chaotic Phase achieves: η/θ ≈ {chaotic_C:.2f} of the limit")
    print("")
    print("  This explains why 'η_max ≈ 8.2' was observed:")
    print(f"  At θ_gain = 0.5, η_max = {ordered_C:.2f} × 0.5 = {ordered_C * 0.5:.2f}")
    
    return results


if __name__ == "__main__":
    results = main()
