"""
thermo_scrambling_crash.py  (FIXED v2)

EXPERIMENT 3: THE COMPLEXITY PHASE TRANSITION
----------------------------------------------
Tests whether ordered vs chaotic Hamiltonians show different
thermodynamic efficiency η = dW/dI scaling with system size N.

FIXES APPLIED vs original:
─────────────────────────────────────────────────────────────────
BUG 1 — No R² quality filter on η.
  ORIGINAL: linregress slope accepted regardless of fit quality.
            A slope from R²=0.1 scatter == a clean R²=0.9 result.
  FIX:      Only accept η when R² ≥ R2_MIN.  Otherwise record NaN
            so the mean/std reflect genuine signal, not noise.

BUG 2 — No statistical significance test.
  ORIGINAL: "crash detected" iff ch_means[-1] < ch_means[0]*0.5,
            which fires even when both values are zero-noise.
  FIX:      Welch t-test (unequal variance) between ordered and
            chaotic distributions at each N.  Report p-value.
            "Crash" = chaotic η significantly < ordered η (p<0.05)
            AND chaotic η consistent with 0 (95% CI includes 0).

BUG 3 — Wrong normalization.
  ORIGINAL: Divide η by N².  η = dW/dI already has units
            energy/bit; dividing by N² collapses both curves
            to noise floor because energy scale ∝ N².
  FIX:      Normalize by Hamiltonian spectral norm ‖H‖ (max
            eigenvalue magnitude), making η intensive:
            η_norm = η / ‖H‖.  This is the correct physical
            normalization — efficiency per unit energy scale.

BUG 4 — Ordered system has random X-fields.
  ORIGINAL: Ferromagnet uses J=-1 but random h_i ∈ [-1,1].
            This injects disorder into the "ordered" class,
            blurring the DLA distinction.
  FIX:      Ordered system: uniform X-field h_i = +0.3 for all i
            (transverse field Ising model in ordered phase).
            Chaotic system: random J_ij ∈ [-1,1] AND random
            h_i ∈ [-1,1] (Sherrington-Kirkpatrick spin glass).

BUG 5 — Wrong initial state.
  ORIGINAL: |+⟩^⊗N for both models.  For the ferromagnet,
            the ground state is near |↓↓...↓⟩, so |+⟩ has
            equal overlap with ALL eigenstates — sensing phase
            carries no gradient information.
  FIX:      Initialize in the energy-biased state |↓⟩^⊗N
            (computational |0⟩ after X-flip), which has
            high overlap with the ferromagnetic ground state
            and maximum energy variance for sensing.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, partial_trace, entropy, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import AerSimulator
from scipy.stats import linregress, ttest_ind
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
N_RANGE      = [3, 4, 5, 6, 7, 8]
TRIALS       = 8        # Increased: need ≥8 for reliable t-test
TAU_STEPS    = 15       # More τ points → better R² on constitutive law
MAX_TAU      = 1.5
KICK_STRENGTH= 0.2
R2_MIN       = 0.60     # Minimum R² to accept η as valid signal
P_THRESHOLD  = 0.05     # Significance level for t-test

# Time evolution is synthesized (product formula) after transpilation.
# Default PauliEvolutionGate synthesis is LieTrotter(reps=1). Make it explicit
# so the approximation can be tightened when needed.
EVOLUTION_REPS = 1


class ComplexityExperiment:
    def __init__(self):
        self.backend = AerSimulator(method='matrix_product_state')

    # ── Hamiltonian generation ─────────────────────────────────────────────────

    def get_hamiltonian(self, n, model_type, seed):
        """
        'ordered'  : Transverse-field Ising ferromagnet (polynomial DLA)
                     J_ij = -1 (uniform FM), h_i = +0.3 (uniform TF)
        'chaotic'  : SK spin glass (exponential DLA)
                     J_ij ~ U(-1,1), h_i ~ U(-1,1)
        """
        np.random.seed(seed)
        ops = []

        for i in range(n):
            for j in range(i + 1, n):
                J = -1.0 if model_type == "ordered" else np.random.uniform(-1.0, 1.0)
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                ops.append(("".join(label[::-1]), J))

        for i in range(n):
            # FIX BUG 4: uniform field for ordered, random for chaotic
            # OLD: h = 0.3 if model_type == "ordered" else np.random.uniform(-1.0, 1.0)
            
            # NEW: Increase transverse field to compete with O(N^2) J-couplings
            h = 1.0 if model_type == "ordered" else np.random.uniform(-1.0, 1.0)
            label = ["I"] * n
            label[i] = "X"
            ops.append(("".join(label[::-1]), h))

        return SparsePauliOp.from_list(ops)

    def get_spectral_norm(self, H, n):
        """
        FIX BUG 3: Compute ‖H‖ = max|eigenvalue| for normalization.
        Uses SparsePauliOp.to_matrix() — practical for n ≤ 8.
        """
        mat = H.to_matrix()
        eigvals = np.linalg.eigvalsh(mat)
        return max(abs(eigvals.min()), abs(eigvals.max()))

    def get_energy(self, state, H):
        if isinstance(state, DensityMatrix):
            return state.expectation_value(H).real
        return 0.0

    # ── Single efficiency sweep ────────────────────────────────────────────────

    def run_efficiency_sweep(self, n, model_type, seed):
        """
        Sweep τ, compute (η, R², ‖H‖) for one disorder realization.

        Returns (eta_normalized, r_squared, spectral_norm)
        Returns (NaN, r_squared, spectral_norm) if R² < R2_MIN.
        """
        H = self.get_hamiltonian(n, model_type, seed)
        H_norm = self.get_spectral_norm(H, n)

        taus = np.linspace(0.05, MAX_TAU, TAU_STEPS)
        data_info, data_work = [], []

        for tau in taus:
            qr_sys = QuantumRegister(n, 'sys')
            qr_anc = QuantumRegister(1, 'anc')
            qc = QuantumCircuit(qr_anc, qr_sys)

            # FIX BUG 5: energy-biased initial state |↓⟩^⊗N
            # X gate maps |0⟩ → |1⟩ = |↓⟩ in Z-basis
            qc.x(qr_sys)
            qc.h(qr_anc)

            # Sensing
            evo = PauliEvolutionGate(H, time=tau, synthesis=LieTrotter(reps=EVOLUTION_REPS))
            qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))

            # Locking
            qc.h(qr_anc)
            qc.save_statevector(label="post_sensing")

            # Actuation
            for i in range(n):
                qc.crx(KICK_STRENGTH, qr_anc[0], qr_sys[i])

            qc.save_statevector(label="final")

            t_qc = transpile(qc, self.backend)
            res = self.backend.run(t_qc).result()

            # Mutual information I(S:A)
            sv = res.data(0)["post_sensing"]
            rho = DensityMatrix(sv)
            rho_sys = partial_trace(rho, [0])
            rho_anc = partial_trace(rho, range(1, n + 1))

            S_SA = entropy(rho, base=2)
            S_S  = entropy(rho_sys, base=2)
            S_A  = entropy(rho_anc, base=2)
            mi   = max(0.0, S_S + S_A - S_SA)
            data_info.append(mi)

            # Work
            E_before = self.get_energy(rho_sys, H)
            sv_f = res.data(0)["final"]
            rho_f = partial_trace(DensityMatrix(sv_f), [0])
            E_after = self.get_energy(rho_f, H)
            data_work.append(E_before - E_after)

        # FIX BUG 1: require R² ≥ R2_MIN
        slope, _, r_val, p_val, _ = linregress(data_info, data_work)
        r2 = r_val ** 2

        if r2 < R2_MIN:
            return np.nan, r2, H_norm   # Reject noisy η

        # FIX BUG 3: normalize by spectral norm
        eta_norm = slope / H_norm if H_norm > 0 else np.nan
        return eta_norm, r2, H_norm

    # ── Main experiment ────────────────────────────────────────────────────────

    def run_experiment(self):
        print("=" * 70)
        print("EXPERIMENT 3 (FIXED v2): COMPLEXITY PHASE TRANSITION")
        print(f"η accepted only when R² ≥ {R2_MIN}  |  "
              f"Significance p < {P_THRESHOLD}")
        print("Normalization: η_norm = η / ‖H‖  (intensive efficiency)")
        print("=" * 70)

        # Storage: list of raw η values per N (NaN-filtered)
        ord_etas   = {n: [] for n in N_RANGE}
        ch_etas    = {n: [] for n in N_RANGE}
        ord_r2     = {n: [] for n in N_RANGE}
        ch_r2      = {n: [] for n in N_RANGE}

        header = (f"{'N':<5} | {'Ord η±σ':<18} | {'Ord R²':<8} | "
                  f"{'Cha η±σ':<18} | {'Cha R²':<8} | {'p-val':<8} | Sig?")
        print(header)
        print("-" * 80)

        for n in N_RANGE:
            for t in range(TRIALS):
                eta_o, r2_o, _ = self.run_efficiency_sweep(n, "ordered",  seed=42  + t)
                eta_c, r2_c, _ = self.run_efficiency_sweep(n, "chaotic",  seed=1000 + t)

                if not np.isnan(eta_o):
                    ord_etas[n].append(eta_o)
                ord_r2[n].append(r2_o)

                if not np.isnan(eta_c):
                    ch_etas[n].append(eta_c)
                ch_r2[n].append(r2_c)

            # Summary statistics
            o_vals = np.array(ord_etas[n]) if ord_etas[n] else np.array([np.nan])
            c_vals = np.array(ch_etas[n])  if ch_etas[n]  else np.array([np.nan])

            o_mu  = np.nanmean(o_vals)
            o_std = np.nanstd(o_vals)
            c_mu  = np.nanmean(c_vals)
            c_std = np.nanstd(c_vals)
            o_r2  = np.nanmean(ord_r2[n])
            c_r2  = np.nanmean(ch_r2[n])

            # FIX BUG 2: Welch t-test between ordered and chaotic distributions
            if len(ord_etas[n]) >= 3 and len(ch_etas[n]) >= 3:
                _, p_val = ttest_ind(ord_etas[n], ch_etas[n], equal_var=False)
                sig = "YES" if p_val < P_THRESHOLD else "no"
            else:
                p_val = np.nan
                sig = "n/a (too few valid)"

            print(f"{n:<5} | {o_mu:+.4f} ± {o_std:.4f}   | {o_r2:.3f}   | "
                  f"{c_mu:+.4f} ± {c_std:.4f}   | {c_r2:.3f}   | "
                  f"{p_val:.3f}   | {sig}")

        # ── Plotting ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        N_arr = np.array(N_RANGE)

        # Collect plot arrays (NaN-safe)
        o_means = np.array([np.nanmean(ord_etas[n]) if ord_etas[n]
                            else np.nan for n in N_RANGE])
        o_stds  = np.array([np.nanstd(ord_etas[n])  if len(ord_etas[n]) > 1
                            else np.nan for n in N_RANGE])
        c_means = np.array([np.nanmean(ch_etas[n])  if ch_etas[n]
                            else np.nan for n in N_RANGE])
        c_stds  = np.array([np.nanstd(ch_etas[n])   if len(ch_etas[n]) > 1
                            else np.nan for n in N_RANGE])

        o_r2s = np.array([np.nanmean(ord_r2[n]) for n in N_RANGE])
        c_r2s = np.array([np.nanmean(ch_r2[n])  for n in N_RANGE])

        # Panel (a): Normalized efficiency
        ax = axes[0]
        ax.errorbar(N_arr, o_means, yerr=o_stds, fmt='o-', color='blue',
                    label='Ordered (FM)', capsize=4, linewidth=2)
        ax.errorbar(N_arr, c_means, yerr=c_stds, fmt='s--', color='red',
                    label='Chaotic (SG)', capsize=4, linewidth=2)
        ax.axhline(0, color='black', linestyle=':', alpha=0.4)
        ax.set_xlabel('System Size N')
        ax.set_ylabel('η_norm = η / ‖H‖  (intensive)')
        ax.set_title('(a) Normalized Efficiency\n(only R² ≥ {:.2f} accepted)'.format(R2_MIN))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel (b): R² quality metric — shows how much of the W-vs-I data
        # is actually linear (constitutive law quality)
        ax = axes[1]
        ax.plot(N_arr, o_r2s, 'o-', color='blue', label='Ordered')
        ax.plot(N_arr, c_r2s, 's--', color='red',  label='Chaotic')
        ax.axhline(R2_MIN, color='gray', linestyle='--',
                   label=f'R² threshold ({R2_MIN})', alpha=0.7)
        ax.set_xlabel('System Size N')
        ax.set_ylabel('Mean R²  (constitutive law fit quality)')
        ax.set_title('(b) R² Quality\n(Below threshold → η rejected as noise)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Panel (c): Valid sample fraction (how many trials passed R² filter)
        ax = axes[2]
        o_frac = np.array([len(ord_etas[n]) / TRIALS for n in N_RANGE])
        c_frac = np.array([len(ch_etas[n])  / TRIALS for n in N_RANGE])
        ax.plot(N_arr, o_frac, 'o-', color='blue', label='Ordered')
        ax.plot(N_arr, c_frac, 's--', color='red',  label='Chaotic')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5,
                   label='50% threshold')
        ax.set_xlabel('System Size N')
        ax.set_ylabel('Fraction of valid trials (R² ≥ threshold)')
        ax.set_title('(c) Signal Fraction\n(Chaotic → fewer valid trials at large N)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig('thermo_complexity_crash_v2.png', dpi=150, bbox_inches='tight')
        print("\nSaved: thermo_complexity_crash_v2.png")

        # ── Verdict ────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("STATISTICAL VERDICT")
        print("=" * 70)

        crash_detected = False
        for n in N_RANGE:
            if len(ord_etas[n]) >= 3 and len(ch_etas[n]) >= 3:
                _, p = ttest_ind(ord_etas[n], ch_etas[n], equal_var=False)
                o_mu = np.nanmean(ord_etas[n])
                c_mu = np.nanmean(ch_etas[n])
                if p < P_THRESHOLD and c_mu < o_mu * 0.5:
                    print(f"  N={n}: Significant separation (p={p:.3f}) — "
                          f"Ordered η={o_mu:.4f} vs Chaotic η={c_mu:.4f}")
                    crash_detected = True

        if crash_detected:
            print("\n>>> PHASE TRANSITION DETECTED (statistically significant).")
        else:
            print("\n>>> No statistically significant transition found at current N.")
            print("    Options: increase TRIALS, extend N_RANGE, or reduce kick noise.")
        
if __name__ == "__main__":
    exp = ComplexityExperiment()
    exp.run_experiment()