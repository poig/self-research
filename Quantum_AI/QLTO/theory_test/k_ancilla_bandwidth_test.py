"""
k_ancilla_bandwidth_test.py

EXPERIMENT: ANCILLA BANDWIDTH SCALING (FIXED)
==============================================

FIXED VERSION: Each ancilla senses a DIFFERENT subset of the Hamiltonian,
extracting (more nearly) independent information.

THE HYPOTHESIS:
---------------
- 1 ancilla qubit: limited information bandwidth per cycle.
- k ancilla qubits: more total information bandwidth.

NOTE:
-----
This script measures *mutual information* I(S:A), which for a pure global
state has the upper bound I(S:A) \le 2k (bits) when A is k qubits.

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
    DensityMatrix, Statevector,
)
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.synthesis.evolution import LieTrotter
from qiskit_aer import AerSimulator
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYSTEM_SIZES = [3, 4, 5, 6, 7]  # Extended to show crossover
ANCILLA_COUNTS = [1, 2]         # Focus on 1 vs 2 comparison
NUM_SEEDS = 5                   # Average over multiple random seeds
MAX_TAU = 1.5                   # τ range: 0.0-1.5 (TABLE I)
KICK_STRENGTH = 0.2             # θ_gain = 0.2 rad (TABLE I)

# FIX 1: τ resolution. TAU_STEPS was 5, i.e. every η and R² in the published
# figure came from a 5-point regression. Raised to 20 to match harmonized_sweep.
TAU_STEPS = 20

# FIX 2: Trotter error. EVOLUTION_REPS = 1 is far from converged — see
# trotter_r2_check.py, where ordered R² moves 0.244 → 0.651 going reps=1 → exact
# at N=7. The controlled evolution is now built exactly from the eigen-
# decomposition of each Hamiltonian part; the Trotter path is retained only for
# reproducing the old figure.
EXACT_EVOLUTION = True

# FIX 3: both arms. main() previously instantiated 'ordered' only, so the
# chaotic branch of _build_partitioned_chaotic() was never exercised and the
# 5-seed loop averaged a deterministic Hamiltonian against itself (hence the
# ±0.000 error bars). Both arms are now run; seeds are meaningful for 'chaotic'
# and are reported as deterministic for 'ordered'.
HAMILTONIAN_TYPES = ['ordered', 'chaotic']

# FIX 4: actuation confound. Each ancilla applied a full-strength kick to every
# system qubit, so k=2 delivered twice the total rotation of k=1. Any work
# difference then conflates "more bandwidth" with "more actuation". With
# MATCH_ACTUATION the per-ancilla kick is KICK_STRENGTH/k, holding total
# actuation fixed so the comparison isolates bandwidth. Both modes are run.
MATCH_ACTUATION = True

# Time evolution is synthesized (product formula) after transpilation.
# Default PauliEvolutionGate synthesis is LieTrotter(reps=1). Make it explicit
# so the approximation can be tightened when needed.
EVOLUTION_REPS = 1


# ==============================================================================
# K-ANCILLA BANDWIDTH EXPERIMENT (FIXED)
# ==============================================================================

class KAncillaBandwidthFixed:
    def __init__(self, n_system, k_ancilla, hamiltonian_type='chaotic', seed=42,
                 match_actuation=None):
        self.n_sys = n_system
        self.k_anc = k_ancilla
        self.seed = seed
        self.match_actuation = (MATCH_ACTUATION if match_actuation is None
                                else match_actuation)
        self.backend = AerSimulator(method='statevector')

        if hamiltonian_type == 'chaotic':
            self.H_full, self.H_parts = self._build_partitioned_chaotic()
        else:
            self.H_full, self.H_parts = self._build_partitioned_ordered()

        # Eigendecomposition per part, for exact controlled evolution (FIX 2).
        self._spec = [np.linalg.eigh(Hp.to_matrix()) for Hp in self.H_parts]

    def _controlled_evo(self, a, tau):
        """|0><0|_anc (x) I  +  |1><1|_anc (x) e^{-i H_a tau}, on 1 + n_sys qubits.

        Appended to [anc[a]] + sys, so the ancilla is the gate's qubit 0 (least
        significant) and Qiskit's kron places the system factor on the left.
        """
        evals, evecs = self._spec[a]
        u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
        p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        return np.kron(np.eye(2 ** self.n_sys), p0) + np.kron(u_sys, p1)

    def _build_partitioned_chaotic(self):
        """Build Spin Glass Hamiltonian partitioned into k independent parts."""
        np.random.seed(self.seed)
        all_terms = []

        # Generate all ZZ terms
        for i in range(self.n_sys):
            for j in range(i + 1, self.n_sys):
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
            for j in range(i + 1, self.n_sys):
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
        """Run one cycle with k ancillae, each sensing different Hamiltonian parts."""
        qr_sys = QuantumRegister(self.n_sys, 'sys')
        qr_anc = QuantumRegister(self.k_anc, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)

        # Initialize
        qc.h(qr_sys)  # High energy superposition
        for a in range(self.k_anc):
            qc.h(qr_anc[a])

        # --- INDEPENDENT SENSING ---
        for a in range(self.k_anc):
            if EXACT_EVOLUTION:
                qc.append(UnitaryGate(self._controlled_evo(a, tau)),
                          [qr_anc[a]] + list(qr_sys))
            else:
                evo_a = PauliEvolutionGate(
                    self.H_parts[a],
                    time=tau,
                    synthesis=LieTrotter(reps=EVOLUTION_REPS),
                )
                qc.append(evo_a.control(1), [qr_anc[a]] + list(qr_sys))

        # Convert phase to population
        for a in range(self.k_anc):
            qc.h(qr_anc[a])

        qc.save_statevector(label="post_sensing")

        # --- INDEPENDENT FEEDBACK ---
        # Each ancilla applies the same kick to the whole system.
        # (If η becomes negative, this is evidence the chosen feedback is
        # not aligned with the information extracted.)
        kick = (KICK_STRENGTH / self.k_anc if self.match_actuation
                else KICK_STRENGTH)
        for a in range(self.k_anc):
            for i in range(self.n_sys):
                qc.crx(kick, qr_anc[a], qr_sys[i])

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
            return 0.0, 0.0, 0.0, 0.0, 0.0

        slope, intercept, r_value, _, _ = linregress(info_data[valid], work_data[valid])

        # FIX 5: report measured work directly. Panel (c) of the published figure
        # plotted the derived proxy eta x 2S(A) -- a product of two fitted
        # quantities -- when W is measured on every cycle and available here.
        return (slope, r_value ** 2, float(np.mean(entropy_data)),
                float(np.max(work_data)), float(np.min(work_data)))


def _run_config(n, k, ham_type, match_actuation):
    """One (N, k, Hamiltonian, actuation-mode) cell.

    'ordered' is deterministic by construction, so it is run once and reported
    as such rather than averaged against itself over seeds (FIX 3).
    """
    seeds = range(NUM_SEEDS) if ham_type == 'chaotic' else [0]
    etas, r2s, bws, wmaxs, wmins = [], [], [], [], []
    for seed in seeds:
        try:
            exp = KAncillaBandwidthFixed(n, k, ham_type, seed=seed * 100,
                                         match_actuation=match_actuation)
            eta, r2, s_a, wmax, wmin = exp.measure_efficiency()
            if np.isfinite(eta):
                etas.append(eta)
                r2s.append(r2)
                bws.append(2 * s_a)
                wmaxs.append(wmax)
                wmins.append(wmin)
        except Exception as exc:
            # Do not swallow silently: a bare `pass` here turned a missing
            # return statement into 20 rows of "FAILED" with no diagnosis.
            print(f"\n    [error] N={n} k={k} {ham_type} seed={seed}: "
                  f"{type(exc).__name__}: {exc}")
    if not etas:
        return None
    return dict(
        eta=float(np.mean(etas)), eta_std=float(np.std(etas)),
        r2=float(np.mean(r2s)),
        bw=float(np.mean(bws)), bw_std=float(np.std(bws)),
        wmax=float(np.mean(wmaxs)), wmin=float(np.mean(wmins)),
        n_ok=len(etas), deterministic=(ham_type != 'chaotic'),
    )


def main():
    print("=" * 88)
    print("K-ANCILLA BANDWIDTH SCALING TEST  (regenerated)")
    print(f"  evolution: {'EXACT' if EXACT_EVOLUTION else f'LieTrotter(reps={EVOLUTION_REPS})'}"
          f"   tau: {TAU_STEPS} points in [0.1, {MAX_TAU}]   theta_gain = {KICK_STRENGTH}")
    print(f"  chaotic arm: {NUM_SEEDS} seeds;  ordered arm: deterministic, 1 run")
    print("=" * 88)

    results = {}
    for ham in HAMILTONIAN_TYPES:
        for matched in [True, False]:
            mode = 'matched' if matched else 'unmatched'
            print(f"\n{'-' * 88}")
            print(f"{ham.upper()}   actuation: {mode}"
                  f"   (per-ancilla kick = {KICK_STRENGTH}"
                  f"{'/k' if matched else ''})")
            print(f"{'-' * 88}")
            print(f"  {'N':>3}" + "".join(
                f"{'k=' + str(k) + ' eta':>13}{'k=' + str(k) + ' maxW':>13}"
                for k in ANCILLA_COUNTS))
            for n in SYSTEM_SIZES:
                cells = []
                for k in ANCILLA_COUNTS:
                    res = _run_config(n, k, ham, matched)
                    results[(ham, mode, k, n)] = res
                    cells.append(res)
                row = f"  {n:>3}"
                for res in cells:
                    if res is None:
                        row += f"{'FAILED':>13}{'':>13}"
                    else:
                        row += f"{res['eta']:>+13.4f}{res['wmax']:>+13.4f}"
                print(row)

    # ---------------------------------------------------------------- figure
    n_rows = len(HAMILTONIAN_TYPES) * 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows),
                             squeeze=False)
    colors = ['#1f77b4', '#2ca02c']
    N = np.array(SYSTEM_SIZES)

    r = 0
    for ham in HAMILTONIAN_TYPES:
        for matched in [True, False]:
            mode = 'matched' if matched else 'unmatched'
            get = lambda k, key: np.array(
                [results[(ham, mode, k, n)][key]
                 if results[(ham, mode, k, n)] else np.nan
                 for n in SYSTEM_SIZES])

            ax = axes[r][0]
            for i, k in enumerate(ANCILLA_COUNTS):
                mean, std = get(k, 'eta'), get(k, 'eta_std')
                ax.plot(N, mean, color=colors[i], marker='o', markersize=7,
                        linewidth=2, label=f'k = {k}')
                if not results[(ham, mode, k, SYSTEM_SIZES[0])]['deterministic']:
                    ax.fill_between(N, mean - std, mean + std,
                                    color=colors[i], alpha=0.2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
            ax.set_ylabel('slope dW/dI', fontsize=11)
            ax.set_title(f'({chr(97 + r)}1) {ham}, {mode} actuation\n'
                         f'fitted slope', fontsize=11)
            ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[r][1]
            for i, k in enumerate(ANCILLA_COUNTS):
                mean, std = get(k, 'bw'), get(k, 'bw_std')
                ax.plot(N, mean, color=colors[i], marker='s', markersize=7,
                        linewidth=2, label=f'k = {k}')
                if not results[(ham, mode, k, SYSTEM_SIZES[0])]['deterministic']:
                    ax.fill_between(N, mean - std, mean + std,
                                    color=colors[i], alpha=0.2)
            ax.set_ylabel('2·S(A) [bits]', fontsize=11)
            ax.set_title(f'({chr(97 + r)}2) information bandwidth proxy',
                         fontsize=11)
            ax.legend(); ax.grid(True, alpha=0.3)

            # FIX 5: measured work, not the derived product eta x 2S(A).
            ax = axes[r][2]
            for i, k in enumerate(ANCILLA_COUNTS):
                ax.plot(N, get(k, 'wmax'), color=colors[i], marker='^',
                        markersize=8, linewidth=2.2, label=f'k = {k} (max)')
                ax.plot(N, get(k, 'wmin'), color=colors[i], marker='v',
                        markersize=6, linewidth=1.2, linestyle=':',
                        label=f'k = {k} (min)')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
            ax.set_xlabel('System Size N', fontsize=11)
            ax.set_ylabel('measured W', fontsize=11)
            ax.set_title(f'({chr(97 + r)}3) measured work over the τ scan',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            r += 1

    plt.tight_layout()
    plt.savefig('k_ancilla_bandwidth_test.png', dpi=150, bbox_inches='tight')
    print("\n[Saved] k_ancilla_bandwidth_test.png")

    # ------------------------------------------------------- sign stability
    print("\n" + "=" * 88)
    print("SIGN-STABILITY READING")
    print("  claim under test: k=1 turns negative at some N while k=2 stays positive")
    print("=" * 88)
    for ham in HAMILTONIAN_TYPES:
        for mode in ['matched', 'unmatched']:
            print(f"\n  {ham}, {mode} actuation:")
            for k in ANCILLA_COUNTS:
                etas = [results[(ham, mode, k, n)]['eta']
                        if results[(ham, mode, k, n)] else np.nan
                        for n in SYSTEM_SIZES]
                neg = [n for n, e in zip(SYSTEM_SIZES, etas) if e < 0]
                wmins = [results[(ham, mode, k, n)]['wmin']
                         if results[(ham, mode, k, n)] else np.nan
                         for n in SYSTEM_SIZES]
                print(f"    k={k}: slope " +
                      " ".join(f"{e:+.4f}" for e in etas) +
                      f"   negative at N={neg if neg else 'none'}"
                      f"   min measured W = {np.nanmin(wmins):+.4f}")
            e1 = np.array([results[(ham, mode, ANCILLA_COUNTS[0], n)]['eta']
                           for n in SYSTEM_SIZES])
            e2 = np.array([results[(ham, mode, ANCILLA_COUNTS[-1], n)]['eta']
                           for n in SYSTEM_SIZES])
            n1, n2 = int((e1 < 0).sum()), int((e2 < 0).sum())
            if n1 > n2:
                verdict = "HOLDS: k=2 negative at fewer sizes than k=1"
            elif n1 == n2:
                verdict = "NEUTRAL: same number of negative sizes"
            else:
                verdict = "FAILS: k=2 negative at MORE sizes than k=1"
            print(f"    -> {verdict}  ({n1} vs {n2} of {len(SYSTEM_SIZES)})")

    return results


if __name__ == "__main__":
    results = main()
