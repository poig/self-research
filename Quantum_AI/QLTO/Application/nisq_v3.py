"""QLTO V3 - one-circuit gradient sensing plus a quantum walk, no gradient engine.

A W-gate places 2^n parameter configurations in superposition, each entangled
with its own ansatz state; QPE reads an energy per configuration; the per-bit
measurement marginals give the gradient for every coordinate at once. One epoch
costs 2 circuits per commuting block plus one energy readout - 180 circuits for
20 epochs on a 4-block ansatz - independent of the parameter count.

    opt = QLTOv3(ansatz, hamiltonian)
    params, energy = opt.minimize()

Defaults are the measured optima; see RESEARCH_NOTES.md, which holds the full
research record - derivations, benchmark results, the fairness audit, and the
negative results that shaped the design.

Author: Tan Jun Liang
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import (QFT, RXGate, RYGate, RZGate, RGate, PhaseGate,
                                    CXGate, PauliEvolutionGate)
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


# ─────────────────────────────────────────────────────────────────────────────
# Ansatz structure
# ─────────────────────────────────────────────────────────────────────────────

_AXIS_RANK = {'X': 0, 'Y': 1, 'Z': 2}


def rotation_axis(op) -> Optional[str]:
    """Rotation axis of a parameterised single-qubit gate, or None.

    Matching on the gate name alone is not enough: efficient_su2().decompose()
    lowers RY/RZ to r(theta, phi) and p(theta), so a name test for 'ry'/'rz'
    labels every rotation 'Z' and collapses the block structure.
    """
    name = op.name.lower()
    if name == 'rx':
        return 'X'
    if name == 'ry':
        return 'Y'
    if name in ('rz', 'p', 'u1', 'phase'):
        return 'Z'          # P(theta) = e^{i theta/2} RZ(theta)
    if name == 'r':
        try:
            phi = float(op.params[1])   # R(theta, phi): axis cos(phi)X + sin(phi)Y
        except (TypeError, ValueError):
            return None
        if abs(np.sin(phi)) < 1e-9:
            return 'X'
        if abs(np.cos(phi)) < 1e-9:
            return 'Y'
        return f'R{phi:.6f}'
    return None


def parameterised_index(op, param_order) -> Optional[int]:
    """Index of the ansatz parameter this gate rotates, or None.

    Must not test len(op.params) == 1: RGate carries (theta, phi) with phi a
    plain float, and that test silently drops every RY-derived rotation.
    """
    if not op.params:
        return None
    first = op.params[0]
    if isinstance(first, Parameter):
        target = first
    elif isinstance(first, ParameterExpression) and first.parameters:
        free = list(first.parameters)
        if len(free) != 1:
            return None
        target = free[0]
    else:
        return None
    try:
        return param_order.index(target)
    except ValueError:
        return None


def detect_layers(ansatz) -> List[Dict[str, Any]]:
    """Partition parameters into commuting blocks.

    Qiskit emits the rotation layer interleaved per qubit (RY(q0), RZ(q0),
    RY(q1), ...), so a contiguous scan sees the axis alternate on every gate
    and yields singleton blocks. Rotations on different qubits commute, so
    regrouping by axis within each entangler-free segment is exact.

    V3 needs only the parameter partition - which parameters share a walk
    circuit - not generators or instruction indices, so this returns just that.
    """
    decomposed = ansatz.decompose()
    param_order = list(ansatz.parameters)

    layers, segment = [], []

    def flush():
        if not segment:
            return
        by_axis: Dict[str, List[int]] = {}
        for p_idx, axis in segment:
            by_axis.setdefault(axis, []).append(p_idx)
        for axis in sorted(by_axis, key=lambda a: _AXIS_RANK.get(a, 3)):
            layers.append({'params': by_axis[axis], 'axis': axis})
        segment.clear()

    for instr in decomposed.data:
        p_idx = parameterised_index(instr.operation, param_order)
        axis = rotation_axis(instr.operation) if p_idx is not None else None
        if p_idx is not None and axis is not None and len(instr.qubits) == 1:
            segment.append((p_idx, axis))
        else:
            flush()          # an entangler ends the block
    flush()
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser
# ─────────────────────────────────────────────────────────────────────────────

class QLTOv3:
    """QLTO whose gradient comes from the sensing circuit.

    Args:
        ansatz:        parameterised circuit; must decompose to single-qubit
                       rotations plus CX.
        hamiltonian:   SparsePauliOp cost operator.
        shot_budget:   shots per circuit.
        tau_scale:     sensing time tau = tau_scale / ||H||_2.
        backend:       Aer backend; defaults to MPS.
    """

    def __init__(self, ansatz, hamiltonian, shot_budget=8192, tau_scale=1.0,
                 backend=None, sim_method='auto', sv_max_qubits=26,
                 num_ancillas=3, qpe_margin=2.0, uncompute_w=False,
                 merged_walk=True, skip_dead_blocks=True, sort_terms=True,
                 sim_seed=None, free_energy_log=False):
        # num_ancillas=3, LOWERED FROM 4 (v26_fix_validation.log). Depth and gate
        # count both scale with Sigma_a r_a, which is 1+1+2+4 = 8 at kappa=4 and
        # 1+1+2 = 4 at kappa=3, so this halves BOTH. Measured across four
        # problems, 6 seeds, against the kappa=4 reference:
        #
        #     problem          dE      sigma   cx      survival @ p=5e-3
        #     H2              -0.0090   0.4   0.48x    0.198 -> 0.462
        #     MaxCut N=4      -0.0035   0.2   0.54x    0.451 -> 0.650
        #     Heisenberg N=4  +0.0497   1.2   0.50x    0.061 -> 0.246
        #     Heisenberg N=6  +0.0490   1.1   0.49x    0.009 -> 0.098
        #
        # Every arm inside the harness's own noise floor, at half the gates and
        # 2-10x the unmitigated survival. This is the change that moves V3 from
        # "returns noise" toward "runs" on real hardware, and it costs nothing
        # measurable. Consistent with anomaly_e, which swept k=3..7, moved the
        # QPE bin width 16x straddling the signal, and found the gradient error
        # unmoved at every block: RESOLUTION IS NOT WHAT KAPPA BUYS.
        #
        # kappa=2 was also tested and is TEMPTING BUT NOT TAKEN: 0.22-0.31x the
        # gates and survival up to 0.704, but Heisenberg N=6 regressed +0.2901.
        # That reads as only 1.3 sigma, and it is still probably real - the same
        # run's null arm (see below) puts this harness's spurious-difference
        # scale at 0.03-0.09, and 0.29 is 3-10x that. Revisit with more seeds.
        #
        # A NOTE ON SIGMA IN THAT TABLE, because it changes how every A/B in
        # these notes should be read. v26 also ran base against TERM-SORTED at
        # kappa=4, which v27 then proved are the SAME UNITARY to 0.000e+00 - a
        # null experiment. It returned 0.2, 2.2, 3.3 and 1.9 sigma across the
        # four problems. TWO OF FOUR EXCEEDED 2 SIGMA WITH NOTHING TO DETECT,
        # because "paired seeds" pin only the initial parameters while the
        # sampling stays unseeded, so the pairing never removes the dominant
        # variance. Treat 2-3 sigma at six seeds here as consistent with zero.
        #
        # DEFAULTS ARE THE MEASURED OPTIMA. num_ancillas was 1 (Hadamard test),
        # which is the wrong default: the k=1 path costs 1.91x MORE shots than
        # fairly-charged parameter-shift because of its 1/tau^2 variance, and it
        # carries a sin() bias no shot budget or product formula removes (see
        # THE GRADIENT SCALE BIAS). k=4 QPE reads a sampled eigenvalue directly,
        # is asymptotically unbiased, and is what every benchmark number in
        # RESULT was produced with. Anyone calling QLTOv3(ansatz, H) and taking
        # the default was silently getting the deprecated path.
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.shot_budget = shot_budget
        self.tau_scale = tau_scale

        # SIM_SEED makes the SHOT NOISE reproducible, which is a different thing
        # from seeding the initial parameters and is the one that was missing.
        #
        # WHY IT MATTERS, measured rather than argued. v26 compared the shipping
        # configuration against a term-sorted one, and v27 then proved those are
        # THE SAME UNITARY to 0.000e+00. That arm was therefore a null
        # experiment - a difference that must be exactly zero - and across four
        # problems at six "paired" seeds it reported 0.2, 2.2, 3.3 and 1.9 sigma.
        # TWO OF FOUR EXCEEDED 2 SIGMA WITH NOTHING TO DETECT.
        #
        # The reason is that the seed argument elsewhere in this project pins the
        # INITIAL PARAMETERS only. Sampling stayed unseeded, so two "paired" arms
        # were still independent draws and the pairing removed the smaller
        # variance while leaving the dominant one. Three results in these notes
        # were withdrawn to this mechanism: top-4, the merged walk, and v22's
        # MaxCut 3.0 sigma.
        #
        # With sim_seed set and reset_shot_stream() called between arms, two arms
        # issuing circuits in the same order draw IDENTICAL shot noise, so a null
        # comparison returns exactly 0.0 and a real effect is not competing with
        # a ~3 sigma noise floor.
        # Default None = previous behaviour, unseeded.
        self.sim_seed = sim_seed
        self._shot_index = 0


        # FREE ENERGY LOG: report the degree-0 Walsh coefficient of the sensing
        # shots instead of running a separate expectation value. Costs nothing,
        # tracks convergence at r = 0.978-0.996, and carries a state-dependent
        # absolute offset up to +1.09 that no shot budget removes. Monitors, does
        # not report - full reasoning at the return site in run_walk.
        self.free_energy_log = bool(free_energy_log)
        self._last_degree0 = None

        # SORT_TERMS: reorder H_sense into layers of mutually disjoint support
        # before it reaches PauliEvolutionGate. This is a SCHEDULING fix and it
        # is EXACT - only terms with disjoint support are ever transposed, and
        # disjoint Paulis commute, so the emitted product is literally the same
        # unitary (measured 0.00e+00 spectral difference, v24_term_ordering.log).
        #
        # WHY IT MATTERS. get_heisenberg_problem and most Hamiltonian builders
        # emit bonds in chain order (0,1), (1,2), (2,3)... and EVERY CONSECUTIVE
        # PAIR SHARES A QUBIT, so the whole evolution is one serial dependency
        # chain that the transpiler may not reorder. The parallelism is real -
        # all even bonds are mutually disjoint, as are all odd - but the term
        # order hides it. Measured on a single Trotter stage: depth exponent
        # N^1.25 -> N^0.00, flat at 225 from N=4 to N=12. On the full sensing
        # circuit: N^1.22 -> N^0.64, 1.4-2.7x, the residual being the ancilla
        # critical path and the ansatz's own CX chain (v24b_sorted_sensing.log).
        #
        # SCOPE. Depth is what IBM bills through circuit_length and what
        # coherence limits. GATE COUNT IS UNCHANGED - sorting reschedules, it
        # does not remove gates - so this does NOT improve fidelity. V3's
        # survival problem is the cx count and is untouched by this flag.
        # Set False to reproduce pre-fix circuits.
        self.sort_terms = bool(sort_terms)
        # W is block-diagonal in the param computational basis, W = sum_x
        # |x><x| (x) V_x, so for the measured (param, anc) marginals the
        # uncompute is EXACTLY invariant: Tr_sys[V_x^dag rho_x V_x] =
        # Tr_sys[rho_x] by cyclicity. Measured TVD 0.0103-0.0108 against a
        # 0.0457 shot-noise floor confirms it. Off by default; it buys 6-19%
        # of gate count and, contrary to the earlier note here, 0% of DEPTH -
        # the ancilla owns the critical path (2*n*k gates against 2k per param
        # qubit), so W^dag fits inside slack that already existed.
        #
        # SCOPE, and it matters: "removable" applies to the COMPUTATIONAL-BASIS
        # MARGINAL readout only. W^dag is exactly what UNCOMPUTES the system back
        # to |0..0>, so any interference-based readout - post-selecting sys=|0..0>
        # to obtain the coherent phase function sum_x <psi_x|e^{-iHt}|psi_x> |x>
        # on the param register - REQUIRES it. Without W^dag the branches are
        # |x>|psi_x> and |x> e^{-iHt}|psi_x>, and post-selection picks up
        # <0|psi_x> rather than the Loschmidt amplitude. Invisible to incoherent
        # marginals, essential to coherent ones. Do not read the TVD result as
        # "W^dag is useless".
        self.uncompute_w = bool(uncompute_w)

        # MERGED WALK: replace CRZ-then-CRX with one tilted-axis controlled
        # rotation, -37% walk depth (162->102 at N=4, 246->156 at N=6). The two
        # are NOT equivalent - at the angles actually used they differ by 0.813 in
        # operator norm, so this is different dynamics at lower depth.
        # Validated PAIRED at 12 seeds, both arms from identical initial
        # parameters: -0.0032 +- 0.0101, 0.3 sigma, better on 7/12
        # (supplement/results/v10_merge_paired.log). An earlier UNPAIRED test
        # claimed +2.5 sigma; that did not replicate and was cross-run drift.
        # RISK AXIS CHECKED, unlike the Suzuki rule that preceded it. BCH error
        # scales as alpha*beta with alpha = g*gamma*0.5pi/sqrt(R), and the
        # measured max alpha across the suite is H2 0.78, Heisenberg N=8 3.11,
        # MaxCut N=6 2.68, Heisenberg N=4 6.53 - so validation happened at the
        # LARGEST alpha in the suite and every other problem is 2-8x safer.
        #
        # THOSE ALPHA NUMBERS MEAN SOMETHING ELSE TOO, missed when they were
        # first recorded. alpha is the PER-STEP drift angle, and a single step at
        # 6.53 is already past pi - so the k-step product wraps. The walk unitary
        # is a product of rotations about a shared ancilla, so the angle ADDS and
        # the decoded step is PERIODIC in the gradient. Derived and validated to
        # 0.00241 against this circuit: see RESEARCH_NOTES / WHAT THE WALK
        # COMPUTES and supplement/v37b, v37c, v37d, v37e. Read as a BCH risk
        # axis, this table understated the problem; the risk was not truncation
        # error, it was aliasing.
        self.merged_walk = bool(merged_walk)

        # SKIP DEAD BLOCKS: a block whose gradient is identically zero cannot
        # help, and the walk does not merely waste circuits on it - with
        # grad_local = 0 every CRZ angle is zero, only the CRX mixer runs, and
        # _decode_walk returns a shot-noise-limited estimate of the hypercube
        # centre. So those parameters take a RANDOM WALK every epoch, jittering
        # the landscape the live blocks are optimising against.
        # Measured (supplement/results/v12_deadblock.log): MaxCut N=4 blk3
        # |g| = 3.3e-16 and MaxCut N=6 blk3 = 1.0e-15 - machine-precision zero -
        # while Heisenberg has none. This is the DIAGONAL-HAMILTONIAN RULE: a
        # final RZ block commutes with a diagonal H, which covers the whole
        # combinatorial class (MaxCut, Ising, QUBO). Skipping saves 25% of V3's
        # circuits there and removes the jitter.
        self.skip_dead_blocks = bool(skip_dead_blocks)
        self._dead_blocks = None      # detected lazily on the first run_walk
        # 1 -> Hadamard-test sensing: each shot is one +-1 bit, and the estimate
        #      of <H> has variance ~ 1/(tau^2 S). tau = tau_scale/range(H)
        #      shrinks as O(1/N), so this variance grows as O(N^2/S).
        # k>1 -> QPE sensing: each shot returns a sampled EIGENVALUE, so the
        #      variance is Var(H)/S with no tau penalty at all - O(N/S) for an
        #      extensive H. The tau^2 factor is exactly what forces the 16x shot
        #      budget the single-ancilla version needs to match V2.
        self.num_ancillas = max(1, int(num_ancillas))

        self.layers = detect_layers(ansatz)

        # Simulator choice matters enormously here and the sensible default for
        # the rest of the suite is the wrong one for V3. Its circuits are narrow
        # but maximally entangled across the param<->sys cut, which is the worst
        # case for MPS: measured at Heisenberg N=6 (13 qubits), one sensing
        # circuit takes 82s under matrix_product_state and 0.26s under
        # statevector - a 316x difference. Trotter reps barely register (82 vs
        # 73).
        #
        # Chosen per circuit rather than once, because layered and global mode
        # have very different widths: layered needs 1 + max_block + N qubits,
        # global needs 1 + M + N. At Heisenberg N=6 that is 13 vs 31 - 34 MB
        # against 34 GB - so a single up-front choice would be wrong for one of
        # them.
        self.sv_max_qubits = sv_max_qubits
        self._forced_backend = backend
        self._sim_method = sim_method
        self._sv = self._mps = None
        self._warned_mps = False

        self.width_layered = 1 + max((len(l['params']) for l in self.layers),
                                     default=0) + ansatz.num_qubits
        self.width_global = 1 + ansatz.num_parameters + ansatz.num_qubits

        self.backend = self._backend_for(self.width_layered)
        self.estimator = AerEstimator(
            options={'backend_options': {'method': getattr(
                getattr(self.backend, 'options', None), 'method', 'automatic')}})
        self.H_sense, self.h_offset, self.H_range = self._sensing_hamiltonian(hamiltonian)
        if self.sort_terms:
            self.H_sense = self._layer_sort(self.H_sense)
        self.tau = tau_scale / (self.H_range + 1e-12)

        # QPE base time. The aliasing constraint applies to the BASE unitary
        # U = exp(-i H tau0): its phase phi = -E tau0 / 2pi must stay inside one
        # turn, so |E| tau0 <= pi. The 2^a ancilla evolution times resolve that
        # single turn into k bits - they do NOT relax the constraint, so tau0 is
        # independent of k. (nisq_v2 divides by 2^(k-1) here, which shrinks the
        # used phase window by that factor and makes the decoded energy scale
        # wrong by 2^k - verified: decoded energy doubled per added ancilla.)
        self.H0_norm = (float(np.linalg.norm(self.H_sense.to_matrix(), ord=2))
                        if self.H_sense.num_qubits <= 14
                        else float(np.sum(np.abs(self.H_sense.coeffs))))
        # qpe_margin > 1 keeps the spectrum away from the +-0.5 wrap boundary.
        # At margin=1 the extreme eigenvalues sit exactly on it, so any state
        # with weight near the spectrum edges has samples wrap around and the
        # decoded MEAN is corrupted - measured as a 2.99 error on a state whose
        # true energy was -3.00. The cost is resolution: 2*margin*||H0||/2^k.
        self.qpe_margin = float(qpe_margin)
        self.tau0 = np.pi / (self.qpe_margin * self.H0_norm + 1e-12)

        self.nefv = 0
        self.last_circuit_depth = 0
        self.max_circuit_depth = 0

        method = getattr(getattr(self.backend, 'options', None), 'method', '?')
        fits = "fits" if self.width_global <= self.sv_max_qubits else "too wide"
        print(f"[V3] {len(self.layers)} commuting blocks "
              f"{[len(l['params']) for l in self.layers]} | range(H)="
              f"{self.H_range:.4f} identity={self.h_offset:+.4f} "
              f"tau={self.tau:.4f} | layered {self.width_layered}q {method}, "
              f"global {self.width_global}q ({fits}) | no gradient engine")

    def _backend_for(self, n_qubits):
        """Statevector while it fits in memory, MPS beyond.

        Statevector cost is 2^n * 16 bytes: 21q = 34 MB, 26q = 1.1 GB,
        28q = 4.3 GB, 31q = 34 GB. sv_max_qubits is that budget, not a
        statement about what the algorithm can do.
        """
        if self._forced_backend is not None:
            return self._forced_backend
        if self._sim_method != 'auto':
            if self._sv is None:
                self._sv = AerSimulator(method=self._sim_method)
            return self._sv
        if n_qubits <= self.sv_max_qubits:
            if self._sv is None:
                self._sv = AerSimulator(method='statevector')
            return self._sv
        if not self._warned_mps:
            gb = (2 ** n_qubits) * 16 / 1e9
            print(f"[V3] {n_qubits} qubits needs {gb:.1f} GB as a statevector "
                  f"(limit {self.sv_max_qubits}q); falling back to MPS, which is "
                  f"~300x slower for these circuits.")
            self._warned_mps = True
        if self._mps is None:
            self._mps = AerSimulator(method='matrix_product_state')
        return self._mps

    def _sense_layers(self):
        """H_sense's terms grouped into layers of mutually disjoint support."""
        layers = []
        for pauli, coeff in zip(self.H_sense.paulis, self.H_sense.coeffs):
            label = str(pauli)[::-1]
            sup = {i for i, ch in enumerate(label) if ch != 'I'}
            for lay in layers:
                if not (sup & lay['used']):
                    lay['terms'].append((label, float(np.real(coeff))))
                    lay['used'] |= sup
                    break
            else:
                layers.append({'terms': [(label, float(np.real(coeff)))],
                               'used': set(sup)})
        return [lay['terms'] for lay in layers]

    @staticmethod
    def _layer_sort(op):
        """Reorder Pauli terms into layers of mutually disjoint support.

        Greedy first-fit: each term joins the first layer whose qubits it does
        not touch, otherwise it opens a new one. Terms are then emitted layer by
        layer, so terms that can execute simultaneously are adjacent in the
        sequence and the transpiler can schedule them in parallel.

        EXACT, not approximate. The only transpositions performed are between
        terms with disjoint support, and disjoint Paulis commute, so the product
        exp(-i h_1 t) exp(-i h_2 t) ... is unchanged. Verified by direct spectral
        comparison at N=4,6,8: 0.00e+00 (v24_term_ordering.log). This is NOT a
        different product formula and carries no extra Trotter error.
        """
        layers = []
        for pauli, coeff in zip(op.paulis, op.coeffs):
            label = str(pauli)[::-1]
            sup = {i for i, ch in enumerate(label) if ch != 'I'}
            for lay in layers:
                if not (sup & lay['used']):
                    lay['terms'].append((pauli, coeff)); lay['used'] |= sup
                    break
            else:
                layers.append({'terms': [(pauli, coeff)], 'used': set(sup)})
        ordered = [t for lay in layers for t in lay['terms']]
        return SparsePauliOp.from_list([(str(p), c) for p, c in ordered])

    @staticmethod
    def _sensing_hamiltonian(H):
        """Traceless H, its identity coefficient, and its spectral range.

        A constant term in H is unobservable under ordinary evolution, but the
        sensing evolution is CONTROLLED, so exp(-i c tau) becomes a *relative*
        phase between the ancilla branches. Writing H = H0 + c*I:

            Im<e^{-iH tau}> = cos(c tau) * Im<e^{-iH0 tau}>
                            - sin(c tau) * Re<e^{-iH0 tau}>

        The wanted term Im<e^{-iH0 tau}> ~ -tau<H0> is attenuated by cos(c tau)
        and contaminated by Re<e^{-iH0 tau}> ~ 1. At c tau = pi/2 the signal
        vanishes outright and the ancilla reads the wrong operator entirely.

        Separately, tau must scale with the spectral RANGE, not the spectral
        norm: only the variation of H across the search window carries gradient
        information, and an identity term inflates ||H|| without contributing
        any. Measured on the benchmark set, LiH has c = -7.883 against a range
        of 1.783, so ||H|| = 8.950 gave tau five times too small; combined with
        cos(c tau) = 0.637 that is a ~8x loss of signal. Heisenberg and MaxCut
        have c = 0 and are unaffected.
        """
        ident = 0.0
        keep_p, keep_c = [], []
        for pauli, coeff in zip(H.paulis, H.coeffs):
            if set(pauli.to_label()) == {"I"}:
                ident += complex(coeff).real
            else:
                keep_p.append(pauli.to_label())
                keep_c.append(coeff)

        H0 = (SparsePauliOp(keep_p, keep_c).simplify() if keep_p
              else SparsePauliOp("I" * H.num_qubits, [0.0]))

        if H0.num_qubits <= 14:
            ev = np.linalg.eigvalsh(H0.to_matrix())
            rng = float(ev[-1] - ev[0])
        else:
            # 2 * sum|coeff| bounds the range without building the matrix
            rng = 2.0 * float(np.sum(np.abs(H0.coeffs)))
        return H0, ident, max(rng, 1e-12)

    # ── W-gate ───────────────────────────────────────────────────────────────

    def _apply(self, qc, op, angle, target):
        if isinstance(op, RYGate): qc.ry(angle, target)
        elif isinstance(op, RZGate): qc.rz(angle, target)
        elif isinstance(op, RXGate): qc.rx(angle, target)
        elif isinstance(op, PhaseGate): qc.p(angle, target)
        elif isinstance(op, RGate): qc.r(angle, float(op.params[1]), target)
        else: raise TypeError(f"W-gate cannot encode '{op.name}'")

    def _apply_ctrl(self, qc, op, angle, ctrl, target):
        if isinstance(op, RYGate): g = RYGate(angle)
        elif isinstance(op, RZGate): g = RZGate(angle)
        elif isinstance(op, RXGate): g = RXGate(angle)
        elif isinstance(op, PhaseGate): g = PhaseGate(angle)
        elif isinstance(op, RGate): g = RGate(angle, float(op.params[1]))
        else: raise TypeError(f"W-gate cannot encode '{op.name}'")
        qc.append(g.control(1), [ctrl, target])

    def build_w_gate(self, param_reg, sys_reg, center_params, search_radius,
                     active_indices):
        """|x>_param |0>_sys  ->  |x>_param |psi(theta_x)>_sys.

        Active parameters get a base rotation at c_i - R plus a controlled
        rotation of 2R, so |0> maps to c_i - R and |1> to c_i + R. Frozen
        parameters are applied as constants.
        """
        qc = QuantumCircuit(param_reg, sys_reg, name="W")
        decomp = self.ansatz.decompose()
        param_order = list(self.ansatz.parameters)
        active_map = {g: i for i, g in enumerate(active_indices)}

        for instr in decomp.data:
            op = instr.operation
            p_idx = parameterised_index(op, param_order)

            if p_idx is not None:
                target = sys_reg[decomp.find_bit(instr.qubits[0]).index]
                if p_idx in active_map:
                    self._apply(qc, op, center_params[p_idx] - search_radius, target)
                    self._apply_ctrl(qc, op, 2.0 * search_radius,
                                     param_reg[active_map[p_idx]], target)
                else:
                    self._apply(qc, op, center_params[p_idx], target)
            elif isinstance(op, CXGate):
                q1 = decomp.find_bit(instr.qubits[0]).index
                q2 = decomp.find_bit(instr.qubits[1]).index
                qc.cx(sys_reg[q1], sys_reg[q2])
        return qc

    # ── gradient from the sensing circuit ────────────────────────────────────

    def sense_gradient(self, center_params, search_radius, active_indices):
        """Gradient from one circuit's measurement marginals. No gradient engine.

        Estimates the R-smeared gradient, not the analytic one. Since the walk
        searches exactly that hypercube it is the matched signal, but it is not
        grad E and should not be reported as such.
        """
        if self.num_ancillas > 1:
            return self._sense_gradient_qpe(center_params, search_radius,
                                            active_indices)

        n_active = len(active_indices)
        tau = self.tau

        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n_active, 'param')
        sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
        c_param = ClassicalRegister(n_active, 'c_param')
        c_anc = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sys, c_param, c_anc)

        qc.h(anc)
        qc.h(param)
        qc.append(self.build_w_gate(param, sys, center_params, search_radius,
                                    active_indices), list(param) + list(sys))
        qc.append(PauliEvolutionGate(self.H_sense, time=tau,
                                     synthesis=LieTrotter(reps=2)).control(1),
                  [anc[0]] + list(sys))
        qc.sdg(anc)    # Y basis -> Im<U> ~ -tau<H>; a plain H would read
        qc.h(anc)      # Re<U> ~ 1 - tau^2<H^2>/2, the wrong observable
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)

        counts = self._run(qc)
        return self._decode_gradient(counts, center_params, active_indices,
                                     search_radius, tau)

    def _build_qpe_sensing_circuit(self, center_params, search_radius,
                                   active_indices):
        """The QPE sensing circuit, factored out so every decode shares one build.

        Both _sense_gradient_qpe and sense_moment_gradients read the SAME circuit
        and differ only in the classical arithmetic applied to its shots - which
        is the point of T2, so they must not drift apart.
        """
        n_active = len(active_indices)
        k = self.num_ancillas

        anc = AncillaRegister(k, 'anc')
        param = QuantumRegister(n_active, 'param')
        sysr = QuantumRegister(self.ansatz.num_qubits, 'sys')
        c_param = ClassicalRegister(n_active, 'c_param')
        c_anc = ClassicalRegister(k, 'c_anc')

        qc = QuantumCircuit(anc, param, sysr, c_param, c_anc)

        qc.h(anc)
        qc.h(param)
        qc.append(self.build_w_gate(param, sysr, center_params, search_radius,
                                    active_indices), list(param) + list(sysr))

        for a in range(k):
            t = (2 ** a) * self.tau0
            # Trotter error grows with evolution time; reps must track it or
            # the most significant ancilla decodes garbage.
            #
            # SECOND-ORDER, HALF THE REPS. Suzuki-2 costs ~2x the gates of
            # Lie-Trotter per rep but its error is O(t^3/r^2) against O(t^2/r),
            # so reps=2^a/2 buys a higher order cancellation for the same rep
            # budget. Measured on Heisenberg N=4 (supplement/results/v4_frontier.log),
            # gradient bias against the exact R-smeared target:
            #
            #   lie   reps=2^a    SHIPPED    bias 0.189   depth 484
            #   suz2  reps=2^a/2  NOW        bias 0.067   depth 536
            #   suz2  reps=2^a               bias 0.042   depth 991
            #
            # 2.8x less bias for 11% more depth, and slightly lower noise. The
            # worst block goes from 2.159x the true gradient to 1.145x. Suzuki-4
            # is dominated - suz4 reps=2^a/8 needs depth 1316 to reach the same
            # 0.067, because its per-rep gate overhead outweighs the extra order
            # at these evolution times. Richardson extrapolation over reps was
            # also tested and rejected: same bias as suz2 at equal depth but 2x
            # the circuits and 2x the noise, since extrapolating across two
            # independent estimates amplifies variance by ~sqrt(5) while a
            # product formula cancels the same order coherently for free.
            #
            # AND A STEP FLOOR, because reps=2^a/2 alone was over-generalised.
            # That rule fixes the REP COUNT and lets the Trotter STEP float:
            # step = 2^a tau0 / (2^a/2) = 2 tau0, and tau0 = pi/(margin ||H0||)
            # is LARGE exactly when ||H0|| is small. H2 (||H0||=0.827) therefore
            # got step 3.8 and a top-ancilla evolution of t=15.2, far outside any
            # product formula's asymptotic regime, while Heisenberg
            # (||H0||=6.46) got a comfortable 0.49.
            # Measured gradient bias (supplement/results/v13_repschedule.log):
            #
            #                        H2      Heis N=4   MaxCut N=4
            #   lie  2^a          0.6627      0.0673      0.0388
            #   suz2 2^a/2        0.3436      0.0437      0.0336
            #   suz2 2^a          0.0541      0.0357      0.0366
            #
            # suz2 2^a/2 beats the old lie 2^a everywhere - the change was right -
            # but it left H2 at 8x Heisenberg's bias. Taking the max of the two
            # criteria gives H2 reps 1,2,4,8 (full) while Heisenberg and MaxCut
            # keep 1,1,2,4 (half), so the depth is spent only where the evolution
            # is actually long.
            # NOTE the operator-norm error disagrees with this and prefers the old
            # rule; it is the wrong metric. The gradient uses DIFFERENCES of
            # energies across vertices, so Trotter error that is uniform over the
            # hypercube cancels and never reaches the estimator.
            reps = int(max(1, (2 ** a) // 2, np.ceil(t / 2.0)))
            qc.append(PauliEvolutionGate(
                self.H_sense, time=t,
                synthesis=SuzukiTrotter(order=2, reps=reps)).control(1),
                [anc[a]] + list(sysr))

        qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)
        return qc

    def _sense_gradient_qpe(self, center_params, search_radius, active_indices):
        """Gradient from QPE sensing: each shot returns a sampled eigenvalue.

        The single-ancilla Hadamard test returns one +-1 bit per shot, so the
        <H> estimate carries variance ~1/(tau^2 S) and tau shrinks as 1/range.
        QPE instead decodes an energy directly, giving Var(H)/S with no tau
        factor - the difference between O(N^2/S) and O(N/S) for an extensive H.

        No sdg here: the phase is read by the inverse QFT, not by a basis
        rotation, so the Y-basis trick of the k=1 path does not apply.
        """
        qc = self._build_qpe_sensing_circuit(center_params, search_radius,
                                             active_indices)
        counts = self._run(qc)
        return self._decode_gradient_qpe(counts, center_params, active_indices,
                                         search_radius)

    def _decode_gradient_qpe(self, counts, center_params, active_indices,
                             search_radius):
        """Per-bit conditional mean of the DECODED ENERGY, not of a +-1 bit.

        U = exp(-i H tau0) has eigenvalue exp(2 pi i phi) with phi = -E tau0/2pi,
        so E = -2 pi phi / tau0. phi is wrapped into [-1/2, 1/2) because the
        spectrum is signed (H_sense is traceless).
        """
        n_active = len(active_indices)
        k = self.num_ancillas
        num = np.zeros((2, n_active))
        den = np.zeros((2, n_active))
        e_tot, e_n = 0.0, 0          # degree-0 Walsh coefficient, for the free log

        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            # cr_anc registered last -> printed first. Read as-is: measured
            # against exact <H_sense> over four random points, the unreversed
            # order with E = -2 pi phi / tau0 recovers the energy to within the
            # QPE resolution (err 0.807 vs resolution 0.808 at k=4); every other
            # sign/order combination is off by 1.2-2.9x that.
            m = int(parts[0], 2)
            phi = m / (2 ** k)
            if phi >= 0.5:
                phi -= 1.0
            energy = -2.0 * np.pi * phi / (self.tau0 + 1e-12)

            xbits = parts[1][::-1]
            e_tot += energy * cnt
            e_n += cnt
            for i in range(n_active):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += energy * cnt
                den[b, i] += cnt

        # Degree-0 is the SAME empirical mean the gradient is built from, so it
        # is free. Kept for the optional free energy log; see run_walk.
        self._last_degree0 = e_tot / e_n if e_n else None

        mean1 = np.divide(num[1], den[1], out=np.zeros(n_active), where=den[1] > 0)
        mean0 = np.divide(num[0], den[0], out=np.zeros(n_active), where=den[0] > 0)
        # Energies are decoded directly, so no 1/tau rescaling and no sign flip.
        grad = np.zeros(len(center_params))
        grad[active_indices] = (mean1 - mean0) / (2.0 * search_radius + 1e-12)
        return grad

    def sense_moment_gradients(self, center_params, search_radius,
                               active_indices, powers=(1, 2)):
        """Gradients of <H^p> for several p, from ONE QPE sensing circuit.

        QPE samples eigenvalues with probability |<E_k|psi>|^2, so over shots
        E[e^p] = <H^p> for every p, and e^p is a PER-SHOT quantity - which makes
        its degree-1 Walsh coefficient an empirical mean, hence LINEAR and
        unbiased at any shots-per-vertex exactly like the first moment (T2).
        So every moment is already sitting in the shot record and costs nothing
        beyond different classical arithmetic on it.

        Verified (supplement/results/v5_moments.log): the degree-1 coefficients
        of e^2 come back at cos 0.99507 against exact, norm ratio 0.9847 -
        BETTER norm fidelity than the first moment's 0.9473.

        This is what makes folded-spectrum objectives cheap. Minimising
        <(H-omega)^2> = <H^2> - 2 omega <H> + omega^2 needs both moments, and its
        gradient is just grad<H^2> - 2 omega grad<H> - a linear combination of
        what this returns, so INTERIOR excited states cost the same as ground
        states. Also gives Var(H) per vertex for a diagonal preconditioner.

        TWO CAVEATS from that log, both about the second moment specifically:
          * accuracy peaks at k=4-5 (-1.4%, -0.7% on <H^2>) and DEGRADES at k=6-7
            (-4.2%, -7.3%), because higher k means longer evolutions and more
            accumulated Trotter error in the high-order bits, distorting the
            tails that the second moment weights hardest;
          * far more qpe_margin-sensitive than the first moment - margin 1.2 gives
            -17.4% from wrap, 4.0 gives +15.4% from lost resolution. The default
            margin=2.0 was chosen for <H> alone. RETUNE IT for second-moment work.

        Returns {p: gradient_vector} with one entry per requested power.
        Requires the QPE path (num_ancillas > 1); the Hadamard readout returns a
        +-1 bit, not an energy, so it has no moments to give.
        """
        if self.num_ancillas <= 1:
            raise ValueError(
                "sense_moment_gradients needs QPE sensing (num_ancillas > 1). "
                "The k=1 Hadamard path measures a +-1 bit, not a sampled "
                "eigenvalue, so e^p is not available.")

        n_active = len(active_indices)
        k = self.num_ancillas
        qc = self._build_qpe_sensing_circuit(center_params, search_radius,
                                             active_indices)
        counts = self._run(qc)

        num = {p: np.zeros((2, n_active)) for p in powers}
        den = np.zeros((2, n_active))
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            m = int(parts[0], 2)
            phi = m / (2 ** k)
            if phi >= 0.5:
                phi -= 1.0
            e = -2.0 * np.pi * phi / (self.tau0 + 1e-12)
            xbits = parts[1][::-1]
            for i in range(n_active):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                den[b, i] += cnt
                for p in powers:
                    num[p][b, i] += (e ** p) * cnt

        out = {}
        for p in powers:
            m1 = np.divide(num[p][1], den[1], out=np.zeros(n_active),
                           where=den[1] > 0)
            m0 = np.divide(num[p][0], den[0], out=np.zeros(n_active),
                           where=den[0] > 0)
            g = np.zeros(len(center_params))
            g[active_indices] = (m1 - m0) / (2.0 * search_radius + 1e-12)
            out[p] = g
        return out

    def folded_spectrum_gradient(self, center_params, search_radius,
                                 active_indices, omega):
        """Gradient of <(H-omega)^2>, for INTERIOR excited states near omega.

        = grad<H^2> - 2*omega*grad<H>, both from the same single circuit.
        Note H_sense is traceless, so omega is measured on the SHIFTED spectrum;
        subtract self.h_offset from a target energy in the original units.
        For the two EXTREMAL states no folding is needed at all - pass -H to the
        constructor and the walk climbs instead of descending.
        """
        mom = self.sense_moment_gradients(center_params, search_radius,
                                          active_indices, powers=(1, 2))
        return mom[2] - 2.0 * float(omega) * mom[1]

    # THE DRIFT COEFFICIENT IS NOT THE OPTIMAL PHASE, and this is the largest
    # known gap in the walk. sense_gradient returns g_i = E_hat({i})/R, the
    # degree-1 Walsh TRUNCATION OF THE ENERGY, and _execute_walk writes it
    # straight into the drift. But the phase that maximises concentration on the
    # good corners is a different object. Measured against the exact walk model
    # (supplement/results/v50_design_on_true_model.log), an optimised degree-1
    # phase beats the shipped truncation by 2.42x at m=1, 1.95x at m=2, 1.32x at
    # m=4, on the same circuit with the same schedule.
    #
    # NOT ACTED ON, because computing that optimum needs |psi_x> for all 2^n
    # vertices and the imprint unitary - classical simulation of the ansatz,
    # which is what the circuit exists to avoid. Free at N=4, impossible at N=30.
    # The open question is whether the optimum is ESTIMABLE from the measured
    # coefficients; if it is, this is a 2.4x design win, and if not it is a
    # ceiling. Degree-2 terms add only 4-6% on top of the optimised degree-1
    # phase, which is why T7 stays closed.
    def _decode_gradient(self, counts, center_params, active_indices,
                         search_radius, tau):
        """g_i ~ <signal | x_i=1> - <signal | x_i=0>, signal = +-1 from the ancilla."""
        n_active = len(active_indices)
        num = np.zeros((2, n_active))
        den = np.zeros((2, n_active))

        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            sign = 1.0 if parts[0][-1] == '0' else -1.0
            xbits = parts[1][::-1]        # little-endian -> param index order
            for i in range(n_active):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += sign * cnt
                den[b, i] += cnt

        mean1 = np.divide(num[1], den[1], out=np.zeros(n_active), where=den[1] > 0)
        mean0 = np.divide(num[0], den[0], out=np.zeros(n_active), where=den[0] > 0)
        # signal ~ -tau*E, vertices 2R apart  =>  dE ~ -(m1 - m0) / (2R tau)
        grad = np.zeros(len(center_params))
        grad[active_indices] = -(mean1 - mean0) / (2.0 * search_radius * tau + 1e-12)
        return grad

    # ── walk ─────────────────────────────────────────────────────────────────

    def _execute_walk(self, center_params, k_steps, delta_t, radius,
                      active_indices, grad):
        n_active = len(active_indices)
        grad_local = grad[active_indices]
        drift_gain = 1.0 / np.sqrt(max(radius, 1e-9))

        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n_active, 'param')
        sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
        c_param = ClassicalRegister(n_active, 'c_param')
        c_anc = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sys, c_param, c_anc)

        qc.h(anc)
        qc.h(param)
        w = self.build_w_gate(param, sys, center_params, radius, active_indices)
        qc.append(w, list(param) + list(sys))

        # Traceless too: the walk's ancilla readout suffers the same relative
        # phase from an identity term as the sensing readout does.
        qc.append(PauliEvolutionGate(self.H_sense, time=delta_t * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1),
                  [anc[0]] + list(sys))

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t              # phase accumulation
            beta = (1.0 - s) * np.pi * delta_t       # mixing strength
            if self.merged_walk:
                # ONE controlled gate per qubit per step instead of two.
                # RY(phi) Z RY(phi)^dag = Z cos phi + X sin phi, so with
                # theta = sqrt(alpha^2+beta^2) and phi = atan2(beta, alpha),
                #     RY(-phi); CRZ(theta); RY(phi)  ==  controlled-exp(-i(aZ+bX)/2)
                # exactly (verified to 4e-16), and the RY conjugation is
                # UNCONTROLLED because controlled-(V W V^dag) = V CW V^dag.
                for i in range(n_active):
                    al = grad_local[i] * gamma * 0.5 * np.pi * drift_gain
                    th = float(np.hypot(al, beta))
                    ph = float(np.arctan2(beta, al))
                    qc.ry(-ph, param[i])
                    qc.crz(th, anc[0], param[i])
                    qc.ry(ph, param[i])
            else:
                for i in range(n_active):
                    # identity metric: no QFIM rescaling in V3
                    qc.crz(grad_local[i] * gamma * 0.5 * np.pi * drift_gain,
                           anc[0], param[i])
                for i in range(n_active):
                    qc.crx(beta, anc[0], param[i])

        # NO sdg HERE, UNLIKE THE SENSING PATH, and that is not an oversight -
        # it is tested. _build_sensing_circuit uses sdg;h to read the Y basis,
        # Im<U> ~ -tau<H>, because a plain h reads Re<U> ~ 1 - tau^2<H^2>/2, the
        # wrong observable. The walk has only h, so its ancilla reports <H^2>.
        # Adding the sdg was measured (supplement/results/v41c_walk_quadrature.log)
        # and moves the oracle quality corr(P,-E) from 0.0673 to 0.0690 - nothing.
        # The inconsistency is real and inert; do not "fix" it expecting a gain.
        #
        # ALSO NOTE uncompute_w CANNOT AFFECT THE PARAM MARGINAL, provably: W is
        # controlled ON param, hence block-diagonal in the param basis, so W^dag
        # cannot move param populations. v41b measured the two identical to four
        # decimals. Its only value is the gate count, as the OPEN section says.
        qc.h(anc)                                    # phase -> population
        if self.uncompute_w:
            qc.append(w.inverse(), list(param) + list(sys))
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)

        counts = self._run(qc)
        block = self._decode_walk(counts, center_params, active_indices, radius)
        new_params = center_params.copy()
        new_params[active_indices] = block
        return new_params

    def _decode_walk(self, counts, center_params, active_indices, radius):
        """Weighted mean of the sampled vertices, restricted to anc=1 when it fires."""
        n_active = len(active_indices)
        total = sum(counts.values())
        move, allc, anc_ones = {}, {}, 0

        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            a, x = parts[0][-1], parts[1]
            allc[x] = allc.get(x, 0) + cnt
            if a == '1':
                anc_ones += cnt
                move[x] = move.get(x, 0) + cnt

        # activation_rate is still needed HERE, as the branch condition below -
        # it decides whether the anc=1 marginal is usable. It is no longer
        # STORED: as a diagnostic it was worthless, sitting near 50% for every k
        # and every mode, and normalized_entropy alongside it measured
        # concentration rather than correctness. Both were written every walk and
        # read by nothing.
        activation = anc_ones / total if total else 0.0

        centre = center_params[active_indices]
        if move and activation > 0.05:
            return self._weighted_vertices(move, centre, radius, n_active)
        # ancilla never fired: damp toward the unconditioned mean
        return centre + 0.3 * (self._weighted_vertices(allc, centre, radius,
                                                       n_active) - centre)

    @staticmethod
    def _weighted_vertices(param_counts, centre, radius, n_active):
        acc = np.zeros(n_active)
        wsum = 0.0
        for bitstr, cnt in param_counts.items():
            bits = bitstr.replace(" ", "").zfill(n_active)[-n_active:][::-1]
            vals = np.array([centre[i] + (radius if bits[i] == '1' else -radius)
                             for i in range(n_active)])
            acc += vals * cnt
            wsum += cnt
        return acc / wsum if wsum else centre

    # ── driver ───────────────────────────────────────────────────────────────

    def _run(self, qc):
        backend = self._backend_for(qc.num_qubits)
        t_qc = transpile(qc, backend, optimization_level=1)
        self.last_circuit_depth = t_qc.depth()
        self.max_circuit_depth = max(self.max_circuit_depth, self.last_circuit_depth)
        self.nefv += 1
        kwargs = {}
        if self.sim_seed is not None:
            # A DISTINCT seed per circuit, from a REPRODUCIBLE sequence. Reusing
            # one seed for every circuit would correlate the shot noise across
            # circuits within a run, which is worse than not seeding at all.
            kwargs['seed_simulator'] = int(self.sim_seed) + self._shot_index
            self._shot_index += 1
        return backend.run(t_qc, shots=self.shot_budget,
                           **kwargs).result().get_counts()

    def reset_shot_stream(self):
        """Restart the shot-noise sequence, so a second run replays the first.

        Two arms of an A/B that issue circuits in the same order will then draw
        the SAME shot noise, which is what makes a paired comparison actually
        paired. Call between arms.
        """
        self._shot_index = 0

    def grad_step(self, center_params, search_radius, active_indices, grad,
                  alpha=0.9):
        """Bounded classical step on the sensed gradient. 1 circuit per block.

        p_i <- p_i - alpha * R * g_i / max_j|g_j|, so the largest coordinate moves
        alpha*R and the step lives in the SAME +-R box the walk moves in. Nothing
        is tuned: alpha is fixed and the scale comes from the schedule's R.

        WHY THIS EXISTS. The walk costs 2 circuits per block-epoch against this
        one, and it has never been measured ahead of it. At Heisenberg N=4 /
        256 shots, over four independent runs spanning two implementations and
        both merged_walk settings:
            walk      -5.79 (stale log) .. -4.54 .. -5.05 .. -5.15
            gradstep  -5.81 (stale log) .. -5.39 .. -5.58
        The walk is behind in every one. See supplement/v20, v53, v53b, v53c.

        AND IT IS IMMUNE TO THE GRADIENT'S WORST DEFECT. anomaly_c measured a
        converged per-block SCALE bias up to 2.4x, 80 sigma from unity. Because
        this step normalises by max|g| WITHIN the block, any per-block scale error
        cancels exactly. A step that used raw magnitudes would not be so lucky.

        WHAT IT KEEPS. Everything that makes QLTO cheap lives in the SENSING, not
        the walk: T1/T2 (all M components from one circuit, unbiased at any
        shots-per-vertex), T10 (~1.5 circuits/gradient, constant in M), the
        difference-cancellation of Trotter error, and the R-annealed bias-variance
        trade. This decoder keeps all of it and drops the second circuit.

        NOT THE DEFAULT, deliberately. The walk is unbeaten only in regimes nobody
        has measured: wide R, where v9_globalgrid puts the box's multi-modal onset
        at R=pi/2 and where a stochastic bounded step is supposed to pay, and
        N>=6 against this arm. The headline benchmark was also run with the walk,
        so flipping the default without re-running it would leave the results and
        the code inconsistent. Flip it when those two gaps are closed.
        """
        p = np.asarray(center_params, dtype=float).copy()
        g = np.asarray(grad)[active_indices]
        mx = float(np.max(np.abs(g)))
        if mx < 1e-12:
            return p
        p[active_indices] = p[active_indices] - alpha * search_radius * g / mx
        return p

    def boltzmann_step(self, center_params, search_radius, active_indices,
                       t_frac=0.1, min_per_vertex=8):
        """Update a block from the sensing shots ALONE - no walk circuit.

        Boltzmann-weighted average over the sampled vertices,
        w_x = exp(-(E_x - E_min)/T) with T = t_frac * (energy spread), which is
        argmin as t_frac -> 0 and the hypercube centre as t_frac -> inf.
        Measured (supplement/results/v4_softmin.log, 6 seeds, 3 problems) to TIE
        the sense+walk path at HALF the circuits: -1.7916/-6.0340/-9.0375 against
        the walk's -1.7653/-6.0064/-9.0390. It never beat the walk; it matched it
        for half the cost, which is why this exists as an option and not a default.

        *** SHARPER LIMIT THAN THE GUARD SUGGESTS. *** The guard trips at
        2^n > shots/min_per_vertex, i.e. n <~ 10 at 8192 shots. But T10 puts the
        COST-OPTIMAL block width at n* ~ 0.65 M, so this decoder is usable at the
        optimal width only when 0.65 M <= 10, i.e. M <= 15. At N=4 (M=16) it is
        already at the boundary and beyond that you must narrow the blocks, which
        costs more circuits than the decoder saves. So it is not "works until the
        shots run out" - it is INCOMPATIBLE WITH THE COST-OPTIMAL CONFIGURATION
        past N=4. The guard alone would let you think n=8 at M=32 is a win when
        you have given up more elsewhere.

        *** IT DOES NOT SCALE, AND THAT IS WHY IT IS GUARDED. *** Unlike the
        marginal gradient, this decode is NONLINEAR: it must resolve each vertex's
        energy before weighting it, so it needs shots >~ 2^n. The marginal is
        unbiased at any shots-per-vertex (T2); this is not. It works at n<=6 with
        8192 shots and degrades toward n ~ log2(shots) ~ 13, which is exactly the
        wide-block regime where T10's cost advantage lives. Shipping it as a
        default would look free at benchmark sizes and break where it matters.

        t_frac=0.1 is the measured optimum: 0.3 was mixed and 1.0 catastrophic
        (+3.98 at N=6, 13 sigma), because a high temperature washes out to the
        hypercube centre. Fixed-m rules were worse still - top4 degenerates to a
        no-op whenever m >= 2^n, since averaging every corner of a symmetric box
        returns the centre exactly.
        """
        n = len(active_indices)
        if 2 ** n > self.shot_budget / max(min_per_vertex, 1):
            raise ValueError(
                f"boltzmann_step needs shots >~ {min_per_vertex} per vertex, but "
                f"2^{n} = {2**n} vertices against {self.shot_budget} shots. This "
                f"decode is nonlinear and cannot be used at this block width - "
                f"use the marginal gradient path (sense_gradient + _execute_walk), "
                f"which is unbiased at any shots-per-vertex.")
        if self.num_ancillas <= 1:
            raise ValueError("boltzmann_step needs QPE sensing (num_ancillas > 1) "
                             "to obtain a per-vertex energy.")

        k = self.num_ancillas
        counts = self._run(self._build_qpe_sensing_circuit(
            center_params, search_radius, active_indices))
        num, den = {}, {}
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            m = int(parts[0], 2)
            phi = m / (2 ** k)
            if phi >= 0.5:
                phi -= 1.0
            e = -2.0 * np.pi * phi / (self.tau0 + 1e-12)
            xb = parts[1][::-1]
            key = tuple(1 if (i < len(xb) and xb[i] == '1') else 0
                        for i in range(n))
            num[key] = num.get(key, 0.0) + e * cnt
            den[key] = den.get(key, 0) + cnt

        verts = [v for v in num if den[v] >= min_per_vertex]
        out = np.asarray(center_params, dtype=float).copy()
        if not verts:
            return out
        E = np.array([num[v] / den[v] for v in verts])
        spread = float(E.max() - E.min())
        T = max(t_frac * spread, 1e-9)
        w = np.exp(-(E - E.min()) / T)
        if w.sum() <= 0:
            return out
        for i, idx in enumerate(active_indices):
            vals = np.array([center_params[idx]
                             + (search_radius if v[i] else -search_radius)
                             for v in verts])
            out[idx] = float(np.average(vals, weights=w))
        return out

    def probe_linearity(self, center_params, search_radius, active_indices):
        """Is search_radius still inside the linear regime? Measurable in-situ.

        THE POINT: cos(g_measured, grad E) is what you actually care about and it
        is NOT computable in a real run, because grad E is unknown. This IS
        computable, from sensing circuits, and it tracks that quantity closely.

        Sense at R and at R/2 and compare. Both estimate the same d_iE, so in the
        linear regime the two gradients agree in direction and magnitude; where
        the cubic term bites they diverge. Returns (cosine, magnitude_ratio) with
        (1.0, 1.0) meaning fully linear.

        Calibration from the exact 2-bit study (supplement/results/v9b_multiscale.log),
        which measured the same effect via per-bit Walsh coefficients within one
        circuit - there the coarse/fine ratio should be 2.0 exactly and reads:

            R       ratio     cos(g, grad E)
            0.20    2.0203        0.99981
            0.40    2.0857        0.99700
            0.60    2.2116        0.98509   <- default R
            1.00    2.8508        0.88203
            1.50   15.1368        0.36572

        So the diagnostic degrades smoothly and blows up exactly where the
        direction collapses. At the default R=0.6 the landscape is already mildly
        nonlinear.

        COST AND CAVEAT: two sensing circuits instead of one, but NO extra qubits -
        the alternative, a 2-bit param encoding, doubles the param register, and by
        T10 circuits are the cheap axis while width is the scarce one. The two
        circuits carry independent shot noise, so this ratio is noisier than the
        single-circuit 2-bit version; average it over a few epochs before acting.

        NOT WIRED INTO THE SCHEDULE. R = 0.6*0.9^epoch remains the default because
        an adaptive schedule driven by this has NOT been A/B tested end to end.
        The intended policy is "shrink R until the ratio approaches 1", and that is
        the next experiment, not a shipped behaviour.
        """
        g_full = self.sense_gradient(center_params, search_radius, active_indices)
        g_half = self.sense_gradient(center_params, search_radius / 2.0,
                                     active_indices)
        a = g_full[active_indices]
        b = g_half[active_indices]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0, 0.0
        return float(a @ b / (na * nb)), na / nb

    def adaptive_probe(self, center_params, search_radius, active_indices):
        """One R/2 probe serving BOTH the R schedule and the energy log.

        The two jobs need the same pair of circuits, so they cost one probe
        between them rather than one each.

        RICHARDSON, and this is algebra rather than a fit. The degree-0 Walsh
        coefficient obeys

            Ehat(0)(R) = E(theta_c) + (R^2/2) * sum_i d2E/dtheta_i^2 + b

        with b the QPE binning bias, which is R-INDEPENDENT - measured at R=0
        where the smearing term vanishes identically. Two radii therefore
        eliminate the curvature exactly:

            E + b = (R1^2 Ehat_2 - R2^2 Ehat_1) / (R1^2 - R2^2)

        and at R2 = R1/2 that is (4*Ehat(R/2) - Ehat(R))/3.

        WHY R/2 AND NOT CONSECUTIVE EPOCHS. Epochs run at R and 0.9R, giving
        (Ehat_2 - 0.81 Ehat_1)/0.19 - a ~7x noise amplification that swamps the
        correction. R/2 gives (4 Ehat_2 - Ehat_1)/3, amplification ~1.4x. The
        near pair is free and useless; the far pair costs a circuit and works.

        WHAT IT DOES NOT FIX: b itself. Richardson removes the R-dependent term
        and leaves the binning bias, which needs ONE real energy evaluation per
        run to calibrate - not one per epoch.

        RATIO, for the R schedule. Both gradients estimate the same d_iE, since
        the sensed value is Ehat({i})/R and that is R-independent to leading
        order. So their magnitude ratio is 1 in the linear regime and grows as
        the cubic term bites. Calibrated against cos(g, grad E) via
        v9b_multiscale, whose per-bit ratio is twice this one:

            R      this ratio    cos(g, grad E)
            0.20      1.010          0.9998
            0.40      1.043          0.9970
            0.60      1.106          0.9851   <- the fixed schedule sits here
            1.00      1.425          0.8820
            1.50      7.568          0.3657

        Returns (ratio, energy) with energy already offset by h_offset, so it is
        comparable to <H> rather than to <H_sense>.
        """
        g_full = self.sense_gradient(center_params, search_radius, active_indices)
        e_full = self._last_degree0
        g_half = self.sense_gradient(center_params, search_radius / 2.0,
                                     active_indices)
        e_half = self._last_degree0

        a = g_full[active_indices]
        b_ = g_half[active_indices]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b_))
        ratio = na / nb if nb > 1e-12 else 1.0
        energy = float('nan')
        if e_full is not None and e_half is not None:
            energy = (4.0 * e_half - e_full) / 3.0 + self.h_offset
        return ratio, energy

    @staticmethod
    def sensing_radius_cap(probe_cos, threshold=0.99, floor=0.7):
        """Shrink factor for the SENSING radius from the probe cosine.

        NOT A SCHEDULE, and the distinction is the whole reason this is not
        wired in. See the overloading note below.

        USE THE COSINE, NOT THE MAGNITUDE RATIO. My first version used
        ||g(R)||/||g(R/2)||, mapping v9b's per-bit table onto it. That was wrong:
        v9b compares two bit-levels INSIDE ONE CIRCUIT, where both smear over the
        same coordinates so T10's attenuation exp(-c R^2 n) cancels and only
        nonlinearity remains. Two SEPARATE circuits at R and R/2 smear by
        different amounts, so their magnitude ratio reads mostly attenuation -
        measured 0.911-0.916 at R=0.6 where the mapping predicted 1.106, with
        exp(-c R^2 n * 3/4) = 0.862 accounting for the difference.
        Attenuation is a UNIFORM SCALING, so it cancels exactly in a cosine.
        Measured (v34_probe_cosine.log), probe cos against the unobservable truth:

            R      H2 probe/true   MaxCut probe/true   Heis probe/true
            0.2     0.996 / 0.998    0.998 / 0.999      0.959 / 0.995
            0.6     0.997 / 0.980    0.990 / 0.989      0.992 / 0.998
            1.0     0.971 / 0.916    0.947 / 0.906      0.993 / 0.995
            1.5     0.858 / 0.682    0.948 / 0.823      0.972 / 0.978

        The cosine tracks the truth where nonlinearity bites. It also carries a
        SHOT-NOISE FLOOR at small R - Heisenberg reads 0.959 at R=0.2 where the
        truth is 0.995, because the signal is small there - so this may only be
        trusted to trigger SHRINKING, never to hold or grow.

        *** WHY THIS IS NOT WIRED INTO THE R SCHEDULE ***
        R IS OVERLOADED. It is the sensing radius AND the walk's step bound -
        _decode_walk returns a weighted mean of +-R corners, so the move is
        bounded by R and shrinking R is what converges the optimiser. The table
        above shows Heisenberg stays linear to R=1.5, so the LINEARITY answer is
        "hold R at 0.6 or more", while the CONVERGENCE answer is the decay to
        0.073 that the fixed schedule performs. Both are right about different
        things and one number cannot serve both.
        T9c's proposal ("shrink R until the ratio approaches 2") silently assumes
        R is a sensing parameter alone. Acting on it as written would hold the
        step size fixed and stop the optimiser converging.
        DOING THIS PROPERLY needs the two roles SEPARATED - sense at R_sense set
        by this diagnostic, step at R_step set by the convergence schedule -
        which means build_w_gate and _execute_walk take different radii. That is
        a circuit change, not a schedule tweak, and it is not built.
        """
        if not np.isfinite(probe_cos) or probe_cos >= threshold:
            return 1.0
        deficit = (threshold - probe_cos) / max(threshold, 1e-9)
        return float(np.clip(1.0 - 4.0 * deficit, floor, 1.0))

    def _structurally_dead_blocks(self, blocks):
        """Blocks whose gradient is identically zero, found symbolically and free.

        For the LAST block nothing follows it, so the observable it sees is H
        itself and d<H>/d theta_i = 0 exactly when its generator commutes with H:
        [G,H]=0 implies e^{iθG} H e^{-iθG} = H, so <H> cannot depend on θ. This is
        the DIAGONAL-HAMILTONIAN RULE, and it covers the whole combinatorial class
        (MaxCut, Ising, QUBO), where a final RZ layer commutes with a diagonal H.

        WHY SYMBOLIC RATHER THAN MEASURED. Three statistical versions were tried
        and all produced false positives, which are expensive - killing a live
        block cost MaxCut N=4 a 27x energy loss. A magnitude threshold cannot
        work at all, because a dead block's SENSED gradient sits at the shot-noise
        floor, not at its exact 1e-15. A two-sample noise test killed MaxCut's
        blk1 (|g| = 0.061, small but real); tightening it killed a live H2 block
        on one seed and not another, H2's blocks holding only n=2 components. And
        adding a relative-magnitude criterion still could not separate MaxCut's
        blk3 (exactly 0.0) from LiH's blk3 (0.016-0.025, alive but BELOW the shot
        noise) - both look identical to any measurement at this budget.
        The deeper problem is that detection runs once and the skip is permanent,
        which is sound only for a STRUCTURAL zero. Commutation is permanent;
        being weak at epoch 1 is not, and a block whose gradient grows later would
        never be reconsidered. So only exact commutation qualifies.
        Costs zero circuits, has no threshold, and cannot false-positive.

        Only the last block is tested. Earlier blocks see H conjugated by
        everything after them, which spreads support (see T8), so their gradients
        are generically nonzero even when their own generators commute with H.
        """
        dead = set()
        if not blocks:
            return dead
        last = len(blocks) - 1
        active = blocks[last]
        if not active:
            return dead
        axis = None
        for l in self.layers:
            if l['params'] == active:
                axis = l.get('axis')
                break
        if axis not in ('X', 'Y', 'Z'):
            return dead
        n_q = self.ansatz.num_qubits
        param_order = list(self.ansatz.parameters)
        decomp = self.ansatz.decompose()
        qubits = set()
        for instr in decomp.data:
            p_idx = parameterised_index(instr.operation, param_order)
            if p_idx in active:
                qubits.add(decomp.find_bit(instr.qubits[0]).index)
        if not qubits:
            return dead
        # Every generator commutes with every term of H?
        for q in qubits:
            lbl = ['I'] * n_q
            lbl[n_q - 1 - q] = axis            # Qiskit label order is reversed
            gen = SparsePauliOp.from_list([("".join(lbl), 1.0)])
            if not bool(np.all(self.hamiltonian.paulis.commutes(gen.paulis[0]))):
                return dead
        dead.add(last)
        return dead

    def minimize(self, initial_params=None, epochs=20, k_steps=15,
                 r0=0.6, r_decay=0.9, dt0=0.5, dt_decay=0.95, seed=None):
        """Run the optimiser. No tuning required - defaults are the measured ones.

        Every number here is what produced the 8-problem benchmark in RESULT,
        unchanged across problems: the same k_steps and the same schedules ran
        H2 through Heisenberg N=8, spanning ||H0|| from 0.83 to 21.2 and M from
        8 to 32. That is the claim worth making about this optimiser - not an
        accuracy number, but that one setting works across the suite.

            opt = QLTOv3(ansatz, hamiltonian)
            params, energy = opt.minimize()

        Returns (params, energy). Pass keywords only if you want to deviate.
        """
        if initial_params is None:
            rng = np.random.RandomState(seed)
            initial_params = rng.uniform(-np.pi, np.pi,
                                         self.ansatz.num_parameters)
        params = np.asarray(initial_params, dtype=float).copy()
        energy = float('nan')
        for ep in range(epochs):
            R = max(r0 * (r_decay ** ep), 1e-4)
            dt = max(dt0 * (dt_decay ** (ep + 1)), 0.01)
            params, energy = self.run_walk(params, k_steps=k_steps,
                                           delta_t=dt, search_radius=R)
        return params, energy

    def run_walk(self, center_params, k_steps=15, delta_t=0.5,
                 search_radius=0.5, layer=True, decoder='walk'):
        """One epoch. Returns (params, energy).

        Per layer: one sensing circuit for the gradient, one walk circuit, one
        energy readout. No gradient-engine circuits.

        decoder='walk'      sense + quantum walk, 2 circuits per block. Default,
                            and the only path that scales - its marginal gradient
                            is linear, hence unbiased at any shots-per-vertex.
        decoder='boltzmann' sense only, 1 circuit per block. Ties the walk at half
                            the cost on small blocks and RAISES on wide ones; see
                            boltzmann_step for why that guard is not optional.
        decoder='gradstep'  sense only, 1 circuit per block. Bounded classical step
                            on the same marginal, so it is LINEAR and scales where
                            boltzmann cannot. Never measured behind the walk, and
                            immune to the per-block scale bias. See grad_step.
        """
        blocks = ([l['params'] for l in self.layers] if layer
                  else [list(range(len(center_params)))])

        params = np.asarray(center_params, dtype=float).copy()

        # Detect dead blocks ONCE, on the first epoch, where the gradients are
        # large because the parameters are still random. A relative threshold is
        # used rather than an absolute one so it adapts to ||H||; doing this at
        # convergence instead would wrongly mark everything dead.
        if self.skip_dead_blocks and self._dead_blocks is None:
            self._dead_blocks = self._structurally_dead_blocks(blocks)
            if self._dead_blocks:
                print(f"[V3] dead blocks {sorted(self._dead_blocks)} "
                      f"(generators commute with H - gradient identically zero) "
                      f"- skipping, saves "
                      f"{2*len(self._dead_blocks)} circuits/epoch")

        for bi, active in enumerate(blocks):
            if not active:
                continue
            if self.skip_dead_blocks and self._dead_blocks and \
                    bi in self._dead_blocks:
                continue
            if decoder == 'boltzmann':
                params = self.boltzmann_step(params, search_radius, active)
                continue

            grad = self.sense_gradient(params, search_radius, active)
            if decoder == 'gradstep':
                params = self.grad_step(params, search_radius, active, grad)
                continue
            params = self._execute_walk(params, k_steps, delta_t, search_radius,
                                        active, grad)

        if self.free_energy_log and self._last_degree0 is not None:
            # FREE LOG. The energy is already in the sensing shots: the DEGREE-0
            # Walsh coefficient is the plain mean of the decoded per-vertex
            # energies, and by T2 it costs nothing extra - the same shot record
            # that gives the gradient gives every other Walsh order.
            #
            # SEQUENCING MAKES IT EXACT IN THE RIGHT SENSE. Sensing measures the
            # hypercube around the CURRENT centre, before the walk moves it, so
            # this reports the parameters as they were at the start of the epoch
            # - a one-epoch reporting lag, and no circuit at all.
            #
            # IT IS BIASED, AND THE BIAS DOES NOT VANISH WITH R
            # (v31_free_energy_log.log). Expanding over the hypercube,
            #     Ehat(empty) = E(theta_c) + (R^2/2) sum_i d^2E/dtheta_i^2 + O(R^4)
            # and that curvature term does shrink - MaxCut tracks the prediction
            # to -0.0015 at R=0.073. But Heisenberg FLOORS at +0.35 independent
            # of R, because the QPE decode carries the sensing evolution's
            # Trotter error as an ABSOLUTE energy offset. THE GRADIENT ESCAPES
            # THAT ERROR AND THIS DOES NOT, for a structural reason: degree-1 is
            # a DIFFERENCE across vertices, so a uniform distortion cancels
            # before reaching the estimator; degree-0 is an absolute value and
            # nothing cancels.
            #
            # SO IT MONITORS, IT DOES NOT REPORT. Over a 20-epoch trajectory
            # (v32_log_from_sensing.log):
            #
            #   problem          offset mean  std    Pearson r  exact improved
            #   H2                  +0.0420  0.0237    0.9947       +0.2319
            #   MaxCut N=4          +0.0323  0.0820    0.9956       +2.3268
            #   Heisenberg N=4      +1.0900  0.2543    0.9784       +4.4097
            #
            # Offset NOISE is 3.5-10% of the improvement being tracked and the
            # curves correlate at 0.978-0.996 with rank preserved, so "has it
            # converged" is answered correctly while "what is the energy" is not.
            # Use it on hardware to replace a per-epoch expectation value - which
            # costs G CIRCUITS, not the 1 that nefv charges - and take ONE
            # accurate evaluation at the end. That is 20 epochs of logging
            # reduced to a single measurement.
            # Default OFF: the benchmark compares E_final against exact and
            # against other optimisers, which a +1.09 offset would corrupt.
            #
            # h_offset IS NOT OPTIONAL. The sensing evolution runs on the
            # TRACELESS H_sense - the identity term is stripped because under a
            # CONTROLLED evolution it becomes a relative phase between the
            # ancilla branches rather than an unobservable constant. So the
            # decoded energies are <H_sense>, and <H> = <H_sense> + h_offset.
            # Omitting it reported -1.3585 where the true energy was +0.0812 on
            # MaxCut N=4, an error the size of the identity coefficient.
            # The value reported is the LAST block's degree-0, so it lags the
            # returned parameters by one block's update, not by a whole epoch.
            return params, float(self._last_degree0) + self.h_offset

        # logging only - and NOT the one circuit these notes used to call it:
        # this is a full expectation value, so on hardware it is G circuits.
        self.nefv += 1
        energy = float(self.estimator.run(
            [(self.ansatz, self.hamiltonian, params)]).result()[0].data.evs)
        return params, energy


def frustrated_hamiltonian(n_qubits, seed=42):
    """Random transverse-field Ising model (spin glass): rugged landscape."""
    rng = np.random.RandomState(seed)
    ops = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            s = ["I"] * n_qubits
            s[i] = s[j] = "Z"
            ops.append(("".join(s), rng.uniform(-1.0, 1.0)))
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "X"
        ops.append(("".join(s), rng.uniform(-1.0, 1.0)))
    return SparsePauliOp.from_list(ops)


if __name__ == "__main__":
    from qiskit.circuit.library import efficient_su2

    N = 4
    H = frustrated_hamiltonian(N, seed=42)
    ansatz = efficient_su2(N, reps=1)
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))

    print("=== NISQ V3: walk-readout gradient (standalone) ===")
    print(f"{N} qubits, {ansatz.num_parameters} params | exact GS = {exact:.6f}")

    qlto = QLTOv3(ansatz, H, shot_budget=8192)
    np.random.seed(42)
    params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    t0 = time.time()
    best = float('inf')
    for epoch in range(20):
        r = max(0.6 * (0.9 ** epoch), 1e-4)
        dt = max(0.5 * (0.95 ** (epoch + 1)), 0.01)
        params, E = qlto.run_walk(params, k_steps=20, delta_t=dt, search_radius=r)
        best = min(best, E)
        print(f"Epoch {epoch + 1:02d} | E = {E:+.6f} | circuits = {qlto.nefv}")

    print(f"\nTotal {time.time() - t0:.1f}s | {qlto.nefv} circuits | "
          f"E_final {E:+.6f} | E_best {best:+.6f} | gap {E - exact:+.4f}")
