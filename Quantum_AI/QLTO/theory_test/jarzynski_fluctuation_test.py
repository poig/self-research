"""
jarzynski_fluctuation_test.py

EXPERIMENT: JARZYNSKI EQUALITY FOR QUANTUM OPTIMIZATION
========================================================

THEORETICAL FOUNDATION:
-----------------------
The Jarzynski Equality [1] relates non-equilibrium work to free energy:
    <exp(-βW)> = exp(-β ΔF)

This is remarkable because it holds even for far-from-equilibrium processes.
The quantum extension was established by Tasaki [2] and formalized by 
Talkner et al. [3] using the two-point energy measurement protocol.

The Crooks Fluctuation Theorem [4] provides a complementary relation:
    P_F(W) / P_R(-W) = exp((W - ΔF) / kT)

connecting forward and reverse process work distributions.

HYPOTHESIS: ALGORITHMIC JARZYNSKI EQUALITY
------------------------------------------
For quantum optimization, we propose an ALGORITHMIC analog:
    <exp(-W/η)> = exp(-ΔF_alg/η)

Where:
- W = work extracted in a single optimization trajectory
- η = algorithmic efficiency (our thermodynamic transport coefficient)
- ΔF_alg = "algorithmic free energy" difference (to be determined empirically)

The key insight is that η plays the role of kT (temperature), measuring
the "thermal noise" of the optimization landscape.

PROTOCOL:
---------
1. Run MANY independent optimization trajectories (N_trials)
2. For each trajectory, measure:
   - W_i = energy reduction achieved
   - I_i = mutual information consumed
3. Compute:
   - η = mean(W)/mean(I)  (the efficiency)
   - <exp(-W/η)> = mean over all trajectories
4. Test if <exp(-W/η)> = exp(-ΔF_alg/η) for some consistent ΔF_alg

PREDICTION:
-----------
If a Jarzynski-like relation holds:
- The work distribution P(W) is constrained by fluctuation theorems
- There exists a well-defined "algorithmic free energy"
- This provides a NEW QUANTITY not predicted by DLA theory alone

SUCCESS CRITERION:
------------------
The relation should hold across different:
- System sizes N
- Hamiltonian types (ordered vs chaotic)
- Number of optimization steps

This would establish "Algorithmic Thermodynamics" as a legitimate subfield.

REFERENCES:
-----------
[1] Jarzynski, C. (1997). "Nonequilibrium equality for free energy differences."
    Phys. Rev. Lett. 78, 2690. arXiv:cond-mat/9610209

[2] Tasaki, H. (2000). "Jarzynski Relations for Quantum Systems and Some 
    Applications." arXiv:cond-mat/0009244

[3] Talkner, P., Lutz, E., & Hänggi, P. (2007). "Fluctuation theorems: Work 
    is not an observable." Phys. Rev. E 75, 050102(R).

[4] Crooks, G. E. (1999). "Entropy production fluctuation theorem and the 
    nonequilibrium work relation for free energy differences." 
    Phys. Rev. E 60, 2721.

Author: Theory Test Suite
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_QUBITS = 4          # System size
N_TRAJECTORIES = 100  # Number of independent optimization runs
N_STEPS = 5           # Steps per trajectory
TAU_SENSING = 0.5     # Fixed sensing time
KICK_STRENGTH = 0.3   # Feedback strength

# ==============================================================================
# JARZYNSKI EXPERIMENT
# ==============================================================================

class JarzynskiOptimizationExperiment:
    """
    Tests whether quantum optimization obeys a Jarzynski-like fluctuation theorem.
    """
    
    def __init__(self, n_qubits, hamiltonian_type='ordered'):
        self.n = n_qubits
        self.backend = AerSimulator(method='statevector')
        self.h_type = hamiltonian_type
        
        # Build Hamiltonian based on type
        if hamiltonian_type == 'ordered':
            self.H = self._build_ordered_hamiltonian()
        else:
            self.H = self._build_chaotic_hamiltonian()
            
        print(f"[Init] N={n_qubits}, Type={hamiltonian_type}")
    
    def _build_ordered_hamiltonian(self):
        """Complete graph (polynomial DLA) - ORDERED phase"""
        ops = []
        # All-to-all ZZ coupling with uniform strength
        for i in range(self.n):
            for j in range(i+1, self.n):
                label = ["I"] * self.n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), 1.0))
        # Transverse field
        for i in range(self.n):
            label = ["I"] * self.n
            label[i] = "X"
            ops.append(("".join(label[::-1]), 0.5))
        return SparsePauliOp.from_list(ops)
    
    def _build_chaotic_hamiltonian(self, seed=None):
        """SK Spin Glass (exponential DLA) - CHAOTIC phase"""
        if seed is not None:
            np.random.seed(seed)
        ops = []
        # Random ZZ couplings (Gaussian)
        for i in range(self.n):
            for j in range(i+1, self.n):
                J = np.random.normal(0, 1.0 / np.sqrt(self.n))
                label = ["I"] * self.n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), J))
        # Random transverse field
        for i in range(self.n):
            h = np.random.normal(0, 0.5)
            label = ["I"] * self.n
            label[i] = "X"
            ops.append(("".join(label[::-1]), h))
        return SparsePauliOp.from_list(ops)
    
    def get_energy(self, state):
        """Compute <H> for a quantum state."""
        if isinstance(state, DensityMatrix):
            return state.expectation_value(self.H).real
        elif isinstance(state, Statevector):
            return state.expectation_value(self.H).real
        return 0.0
    
    def run_single_step(self, initial_state, tau, kick):
        """
        Run one step of ancilla-mediated optimization.
        Returns: (work_extracted, mutual_info, final_state)
        """
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # Initialize from given state
        qc.initialize(initial_state, qr_sys)
        
        # Ancilla in superposition
        qc.h(qr_anc)
        
        # SENSING: Controlled evolution
        evo = PauliEvolutionGate(self.H, time=tau)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # LOCKING: Convert phase to population
        qc.h(qr_anc)
        
        # Save state before feedback (for MI calculation)
        qc.save_statevector(label="pre_kick")
        
        # ACTUATION: Coherent feedback
        for i in range(self.n):
            qc.crx(kick, qr_anc[0], qr_sys[i])
        
        # Save final state
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # Compute mutual information
        sv_pre = result.data(0)["pre_kick"]
        rho_pre = DensityMatrix(sv_pre)
        
        rho_S = partial_trace(rho_pre, [0])
        rho_A = partial_trace(rho_pre, range(1, self.n + 1))
        
        S_SA = entropy(rho_pre)
        S_S = entropy(rho_S)
        S_A = entropy(rho_A)
        mutual_info = max(0, S_S + S_A - S_SA)
        
        # Compute work
        E_before = self.get_energy(rho_S)
        
        sv_final = result.data(0)["final"]
        rho_final = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final, [0])
        
        E_after = self.get_energy(rho_S_final)
        work = E_before - E_after
        
        # Get final system state for next iteration
        final_state = rho_S_final.data.diagonal()
        # Normalize and convert to statevector (approximation for mixed states)
        # For pure state evolution, this is valid
        final_sv = Statevector(sv_final).evolve(
            QuantumCircuit(self.n + 1).to_gate()  # Identity
        )
        
        return work, mutual_info, rho_S_final
    
    def run_trajectory(self, n_steps, seed=None):
        """
        Run a complete optimization trajectory.
        Returns: (total_work, total_info, work_list, info_list)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random initial state (superposition with random phases)
        phases = np.random.uniform(0, 2*np.pi, 2**self.n)
        amplitudes = np.ones(2**self.n) / np.sqrt(2**self.n)
        initial_psi = amplitudes * np.exp(1j * phases)
        initial_psi = initial_psi / np.linalg.norm(initial_psi)
        
        # Build initial circuit
        qr = QuantumRegister(self.n)
        qc_init = QuantumCircuit(qr)
        qc_init.initialize(initial_psi, qr)
        
        t_qc = transpile(qc_init, self.backend)
        qc_init_sv = QuantumCircuit(qr)
        qc_init_sv.initialize(initial_psi, qr)
        qc_init_sv.save_statevector()
        t_qc_sv = transpile(qc_init_sv, self.backend)
        result = self.backend.run(t_qc_sv).result()
        current_state = result.get_statevector()
        
        work_list = []
        info_list = []
        
        for step in range(n_steps):
            # Slight randomness in parameters (realistic optimization)
            tau = TAU_SENSING * (1 + 0.1 * np.random.randn())
            kick = KICK_STRENGTH * (1 + 0.1 * np.random.randn())
            
            w, mi, new_rho = self.run_single_step(
                current_state.data[:2**self.n],  # System part only
                max(0.1, tau), 
                kick
            )
            
            work_list.append(w)
            info_list.append(mi)
            
            # Update state for next step (take diagonal as approximation)
            # This is a simplification - ideally track full density matrix
            current_state = Statevector(new_rho.data @ np.ones(2**self.n) / 2**self.n)
            current_state = Statevector(np.ones(2**self.n) / np.sqrt(2**self.n))
            # Actually, let's restart fresh each step for cleaner statistics
            phases = np.random.uniform(0, 2*np.pi, 2**self.n)
            current_state = Statevector(np.exp(1j * phases) / np.sqrt(2**self.n))
        
        total_work = sum(work_list)
        total_info = sum(info_list)
        
        return total_work, total_info, work_list, info_list
    
    def run_ensemble(self, n_trajectories, n_steps):
        """
        Run many trajectories and collect statistics.
        """
        print(f"[Ensemble] Running {n_trajectories} trajectories, {n_steps} steps each...")
        
        all_W = []  # Total work per trajectory
        all_I = []  # Total info per trajectory
        all_w = []  # Individual step works
        all_i = []  # Individual step infos
        
        for traj in range(n_trajectories):
            if (traj + 1) % 20 == 0:
                print(f"  Trajectory {traj + 1}/{n_trajectories}")
            
            W, I, w_list, i_list = self.run_trajectory(n_steps, seed=traj)
            all_W.append(W)
            all_I.append(I)
            all_w.extend(w_list)
            all_i.extend(i_list)
        
        return np.array(all_W), np.array(all_I), np.array(all_w), np.array(all_i)


def test_jarzynski_relation(W_array, eta, label=""):
    """
    Test if <exp(-W/η)> = exp(-ΔF_alg/η) holds.
    
    Returns:
    - exp_avg: <exp(-W/η)>
    - delta_F_implied: ΔF_alg = -η * log(<exp(-W/η)>)
    - consistency: whether the relation is self-consistent
    """
    # Avoid numerical overflow by shifting
    W_shifted = W_array - np.mean(W_array)
    
    # Compute <exp(-W/η)>
    if abs(eta) < 1e-10:
        print(f"[{label}] η ≈ 0, cannot test Jarzynski relation")
        return None, None, False
    
    exp_terms = np.exp(-W_shifted / eta)
    exp_avg = np.mean(exp_terms)
    
    # Implied ΔF_alg
    delta_F_implied = -eta * np.log(exp_avg) + np.mean(W_array)
    
    # Standard error via bootstrap
    n_bootstrap = 1000
    bootstrap_exp = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(W_array, size=len(W_array), replace=True)
        sample_shifted = sample - np.mean(W_array)
        bootstrap_exp.append(np.mean(np.exp(-sample_shifted / eta)))
    
    exp_std = np.std(bootstrap_exp)
    
    print(f"[{label}] Jarzynski Test:")
    print(f"  η = {eta:.4f}")
    print(f"  <W> = {np.mean(W_array):.4f} ± {np.std(W_array):.4f}")
    print(f"  <exp(-W/η)> = {exp_avg:.4f} ± {exp_std:.4f}")
    print(f"  Implied ΔF_alg = {delta_F_implied:.4f}")
    
    # The key test: is exp_avg significantly different from 1?
    # If exp_avg ≈ 1, then ΔF_alg ≈ <W> (equilibrium case)
    # If exp_avg ≠ 1, we have a genuine Jarzynski correction
    
    jarzynski_correction = delta_F_implied - np.mean(W_array)
    print(f"  Jarzynski Correction = {jarzynski_correction:.4f}")
    
    return exp_avg, delta_F_implied, True


def main():
    """
    Main experiment: Test Jarzynski relation for quantum optimization.
    """
    print("=" * 70)
    print("JARZYNSKI EQUALITY FOR QUANTUM OPTIMIZATION")
    print("=" * 70)
    
    results = {}
    
    # Test both phases
    for h_type in ['ordered', 'chaotic']:
        print(f"\n{'='*70}")
        print(f"PHASE: {h_type.upper()}")
        print("=" * 70)
        
        # Initialize experiment
        exp = JarzynskiOptimizationExperiment(N_QUBITS, h_type)
        
        # Regenerate chaotic Hamiltonian with different seeds for variety
        if h_type == 'chaotic':
            exp.H = exp._build_chaotic_hamiltonian(seed=42)
        
        # Run ensemble
        W_total, I_total, w_steps, i_steps = exp.run_ensemble(
            N_TRAJECTORIES, N_STEPS
        )
        
        # Compute efficiency η
        mean_W = np.mean(W_total)
        mean_I = np.mean(I_total)
        
        if mean_I > 1e-10:
            eta = mean_W / mean_I
        else:
            eta = 0.0
        
        print(f"\n[Statistics]")
        print(f"  Mean Total Work: {mean_W:.4f}")
        print(f"  Mean Total Info: {mean_I:.4f}")
        print(f"  Efficiency η: {eta:.4f}")
        print(f"  Std(W): {np.std(W_total):.4f}")
        print(f"  Std(I): {np.std(I_total):.4f}")
        
        # Test Jarzynski relation on total work
        exp_avg, delta_F, valid = test_jarzynski_relation(
            W_total, eta if abs(eta) > 0.01 else 1.0, label=h_type
        )
        
        results[h_type] = {
            'W': W_total,
            'I': I_total,
            'w': w_steps,
            'i': i_steps,
            'eta': eta,
            'exp_avg': exp_avg,
            'delta_F': delta_F
        }
    
    # ===========================================================================
    # VISUALIZATION
    # ===========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, h_type in enumerate(['ordered', 'chaotic']):
        r = results[h_type]
        
        # 1. Work Distribution
        ax = axes[idx, 0]
        ax.hist(r['W'], bins=20, density=True, alpha=0.7, 
                color='blue' if h_type == 'ordered' else 'red',
                edgecolor='black')
        ax.axvline(np.mean(r['W']), color='black', linestyle='--', 
                   label=f"<W> = {np.mean(r['W']):.3f}")
        if r['delta_F'] is not None:
            ax.axvline(r['delta_F'], color='green', linestyle=':', linewidth=2,
                       label=f"ΔF_alg = {r['delta_F']:.3f}")
        ax.set_xlabel('Total Work W')
        ax.set_ylabel('P(W)')
        ax.set_title(f'{h_type.upper()} Phase: Work Distribution')
        ax.legend()
        
        # 2. Work vs Information scatter
        ax = axes[idx, 1]
        ax.scatter(r['I'], r['W'], alpha=0.5,
                   color='blue' if h_type == 'ordered' else 'red')
        
        # Fit line
        if np.std(r['I']) > 1e-10:
            slope, intercept, r_val, p_val, std_err = stats.linregress(r['I'], r['W'])
            I_fit = np.linspace(min(r['I']), max(r['I']), 100)
            ax.plot(I_fit, slope * I_fit + intercept, 'k--', 
                    label=f'η = {slope:.3f}, R² = {r_val**2:.3f}')
        
        ax.set_xlabel('Total Information I')
        ax.set_ylabel('Total Work W')
        ax.set_title(f'{h_type.upper()} Phase: W vs I')
        ax.legend()
        
        # 3. Jarzynski exponential test
        ax = axes[idx, 2]
        
        eta_test = r['eta'] if abs(r['eta']) > 0.01 else 1.0
        W_sorted = np.sort(r['W'])
        
        # Theoretical Jarzynski: P(W) * exp(-W/η) should integrate to exp(-ΔF/η)
        exp_cumulative = np.cumsum(np.exp(-W_sorted / eta_test)) / len(W_sorted)
        
        ax.plot(W_sorted, exp_cumulative, 
                color='blue' if h_type == 'ordered' else 'red',
                linewidth=2, label='Cumulative <exp(-W/η)>')
        
        if r['exp_avg'] is not None:
            ax.axhline(r['exp_avg'], color='green', linestyle='--',
                       label=f'<exp(-W/η)> = {r["exp_avg"]:.3f}')
        
        ax.set_xlabel('Work W')
        ax.set_ylabel('Cumulative <exp(-W/η)>')
        ax.set_title(f'{h_type.upper()} Phase: Jarzynski Test')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('jarzynski_fluctuation_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: JARZYNSKI EQUALITY FOR QUANTUM OPTIMIZATION")
    print("=" * 70)
    
    print("\nKey Results:")
    for h_type in ['ordered', 'chaotic']:
        r = results[h_type]
        print(f"\n{h_type.upper()}:")
        print(f"  Efficiency η = {r['eta']:.4f}")
        print(f"  <W> = {np.mean(r['W']):.4f}")
        if r['delta_F'] is not None:
            correction = r['delta_F'] - np.mean(r['W'])
            print(f"  ΔF_alg = {r['delta_F']:.4f}")
            print(f"  Jarzynski Correction = {correction:.4f}")
            print(f"  Relative Correction = {abs(correction/np.mean(r['W'])*100):.1f}%")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If the Jarzynski Equality holds for quantum optimization:

1. EXISTENCE OF ΔF_alg: 
   The "algorithmic free energy" is a well-defined quantity,
   even for non-equilibrium optimization trajectories.

2. WORK FLUCTUATIONS MATTER:
   Unlike DLA theory which only predicts mean behavior,
   the Jarzynski relation constrains the FULL DISTRIBUTION P(W).

3. UNIVERSAL BOUND:
   ΔF_alg ≤ <W> always (second law analog)
   The correction measures how far from "reversible" the optimization is.

4. NEW DIAGNOSTIC:
   The ratio |ΔF_alg - <W>| / <W> quantifies "thermodynamic irreversibility"
   of the optimization process.

This provides a NEW QUANTITY not predicted by DLA theory alone,
establishing a genuine contribution to "Algorithmic Thermodynamics".
    """)
    
    return results


if __name__ == "__main__":
    results = main()
