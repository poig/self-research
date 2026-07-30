"""
nisq_v3.py: QLTO with the gradient read out of the sensing circuit itself.

Standalone - numpy and qiskit only. No nisq_v2, no commute_gradient. V2 spends
2M-N circuits per epoch inside CommutingBlockGradient; V3 spends one sensing
circuit per block and reads the gradient off its measurement marginals.

═══ MECHANISM ═══

The W-gate prepares a uniform superposition over the 2^n vertices of the
hypercube {c_i +- R}, each entangled with its own ansatz state. An ancilla-
controlled e^{-i H0 tau} encodes each vertex's energy in the ancilla.
Conditioned on the measured vertex x the ancilla reports E(theta_x), and a
gradient is a marginal, not a per-vertex quantity:

    g_i  ~  <signal | x_i=1> - <signal | x_i=0>  =  2R d_iE + O(R^3)

the O(R^2) cross terms cancelling under the symmetric +-R perturbation of the
other coordinates. Every shot carries a value for every bit, so all components
come from the same shots: ONE circuit, not 2M.

num_ancillas=1   Hadamard test. One +-1 bit per shot, Var(<H>) ~ 1/(tau^2 S).
                 tau = tau_scale/range(H) shrinks as O(1/N), so this is
                 O(N^2/S) - the reason it needed a 16x shot budget to match V2.
num_ancillas=k   QPE. Each shot decodes a sampled EIGENVALUE: Var(H)/S, no tau
                 penalty, O(N/S). Measured at Heisenberg N=6 on identical shots
                 and circuits: 0.94 Hartree better, 10x lower variance, 3.1x
                 depth. Coherent time buys precision at the Heisenberg limit
                 (1/T) where shots only manage the standard quantum limit
                 (1/sqrt S); measured exchange rate 3.1x depth <-> 16x shots.

Resolution is 2*margin*||H0||/2^k and must clear the signal 2R*d_iE. At k=4, H2
sits at resolution 0.204 against signal ~0.24 - and H2 is where V3-QPE places
last. k should scale with ||H0||/(R d_iE) rather than being fixed. UNIMPLEMENTED.

═══ RESULT: 8 problems, 5 trials, every method tuned to an interior optimum ═══

No method dominates. Four win two problems each.

    problem              winner              V3 QPE     circuits win/V3
    Frustrated Ising  V2      -1.4088        3rd          720 / 180
    MaxCut N=4        V3 QPE   0.0065        best         180
    MaxCut N=6        QNG     -0.0137        3rd         1040 / 180
    H2                AdamW   -1.8575        last         320 / 180
    LiH               AdamW   -8.9229        2nd          640 / 180
    Heisenberg N=4    V2      -6.0550        3rd          720 / 180
    Heisenberg N=6    V3 QPE  -9.0550        best         180
    Heisenberg N=8    QNG    -12.1692        3rd         1360 / 180

V3 QPE vs V2: one significant V2 win (3.1 sigma), one marginal, six ties - V3
nominally ahead on three of those. V3 QPE vs tuned AdamW: two significant V3
wins, three AdamW, three ties.

DEFENSIBLE CLAIM: competitive with the best classical and quantum-gradient
optimisers - top group on accuracy, 2 of 8 outright - at 180 circuits on EVERY
problem against 320-1360, and about half V2's depth (136-1060 vs 572-2320).
NOT "V3 wins on accuracy". It does not.

The cost figure is the durable part: 180 is flat in M while every baseline scales
with it, so the ratio widens as problems grow (7x at N=8).

Read E_final, not E_best. E_best is min-over-epochs of noisy evaluations and is
biased low by ~0.02 at 20 epochs - QNG's -0.0137 on MaxCut N=6 is below that
problem's exact 0.0 purely from this.

═══ WHAT THE BENCHMARK NEEDED FIRST ═══

Seven asymmetries, all closed; every one had favoured QLTO.
  * NEFV was a hardcoded formula (2*len(layers)), not a count of what ran
  * baselines received EXACT statevector gradients (reproducible to 5.6e-17)
  * V3 chose statevector while V2 was forced onto MPS (316x at 13 qubits)
  * V2 sensed with the identity term included; V3 traceless
  * QAOA silently dropped every non-Z Pauli - two thirds of Heisenberg
  * PennyLane QNG was scored on a different circuit than it optimised
  * QLTO was tuned; the baselines sat at lr=0.1 from the original file

The last mattered most because it was invisible - it produced plausible numbers.
Tuning to interior optima moved AdamW 0.0463 -> 0.0306 (lr 0.1 -> 0.5) and QNG
to lr=0.3, after which both WON two problems each having been middling in every
earlier run. QLTO's own optimum barely moved (k=15, already the default). Earlier
"QLTO beats the baselines" results were substantially a tuning artifact.

Remaining asymmetry, deliberately not chased: every method has its PRIMARY knob
tuned and its secondary knobs at defaults - AdamW's betas/weight_decay, SPSA's
alpha/gamma/c, QNG's FIM regularisation, and V3's num_ancillas / tau_scale /
qpe_margin / R0 / decay schedules. V3 has more untuned knobs than the baselines,
so the comparison is if anything now conservative for V3 rather than generous.
r0/r_decay/dt0/dt_decay/tau_scale/qpe_margin are exposed on QLTO_Wrapper for
anyone who wants to close that gap; defaults reproduce the schedule these
results were measured with.

═══ PRIOR ART ═══

Jordan, PRL 95 050501 (2005), is NOT the citation. It requires a reversible
arithmetic oracle that coherently evaluates f into a register, giving an exact
phase. <psi(theta)|H|psi(theta)> is an expectation value - no such circuit
exists, which is why VQE needs repeated measurement at all.

Gilyen, Arunachalam & Wiebe, arXiv:1711.00465, IS the citation and covers VQE by
name: LCU probability->phase oracle conversion at O(log 1/eps), Jordan-style
gradient on top, O~(sqrt(d)/eps) queries.

V3 differs in kind, not only in simplicity: no oracle conversion, no LCU, no
coherent QFT readout. The Hamiltonian evolution is native to the problem and the
gradient comes from classical marginals. Scaling trade: theirs O~(sqrt(d)/eps),
V3 O(1/eps^2) and INDEPENDENT of d. V3 is cheaper whenever eps > 1/sqrt(d) - at
d=48 that is eps > 0.14, and cosine 0.95 was measured sufficient to reach V2
parity. Better in eps for them, better in d for V3, and d-dependence is what
hurts VQE.

UNCHECKED: whether this specific shallow instantiation - parameter superposition
+ Hamiltonian-native phase kickback + classical marginal readout, no oracle
conversion - is published. The concept space is mapped; this corner may not be.

═══ IMPLEMENTATION TRAPS (each cost a measurement to find) ═══

tau0 = pi/(margin*||H0||), NOT pi/(2^(k-1)*||H0||). The aliasing constraint binds
    the BASE unitary; the 2^a ancilla times resolve that turn rather than
    relaxing it. Tell: decoded energy doubled per added ancilla. nisq_v2's
    use_qpe_sensing path still carries this error - never enabled, never shown.
ancilla bit order: read the printed register UNREVERSED, E = -2 pi phi / tau0.
    Verified against exact <H_sense> across all four sign/order combinations;
    the others are 1.2-2.9x worse.
qpe_margin > 1 is required. At margin=1 the extreme eigenvalues sit on the +-0.5
    wrap boundary; measured 2.99 error on a state whose true energy was -3.00.
identity term must be stripped from the SENSING Hamiltonian. Under a CONTROLLED
    evolution c*I becomes a relative phase: signal attenuated by cos(c tau),
    contaminated by Re<U>, gone entirely at c tau = pi/2. LiH (c=-7.883) lost 8x.
W-gate must not test len(op.params)==1. efficient_su2 decomposes to
    RGate(theta,phi) and that test silently drops every RY-derived rotation -
    the walk then searches a circuit missing half the ansatz.
simulator by circuit WIDTH, not system size.

═══ OPEN, RANKED ═══

adaptive k      Formula derived, unused. H2 and Heisenberg N=8 - V3's two worst
                results - are both under-resolved at k=4. Cheapest gain
                available. Caveat: MaxCut N=4 is also nominally under-resolved
                and V3 won it, so the rule is incomplete.
ansatz          LARGEST gain, and it favours V3. reps=1 caps at -6.1231 while
                reps=3 reaches exact at N=4; every method in the suite is
                fighting over the last 1-2% beneath that ceiling. Raising reps
                multiplies M and V3's cost is flat in M. HVA underperforms as
                implemented (p=4 -> -5.146) but its gradients used an invalid
                shift rule for multi-term generators - a loose bound only.
global mode     2 circuits/epoch against 2B+1, independent of M and B. Matches
                layered accuracy where measured (H2, Heisenberg N=4), so ~60
                circuits rather than 180. Blocked by simulator memory
                (1+M+N qubits; 31q = 34 GB), not by the algorithm.
||g|| magnitude Direction cosine 0.999 but norm ratio 0.55 and 2.08 across two
                blocks of one circuit. UNEXPLAINED.
free savings    point-energy is 1 circuit/epoch of logging the optimiser never
                reads. W-dagger is block-diagonal in the param basis and cannot
                change the measured marginals - removable, halving the walk's W
                contribution to depth. Both untested.
schedule        gamma and beta both scale with k, entangling k with step size.
                Likely cause of the isolated dips (H2 layered k=10; Heisenberg
                layered AND global k=20). Normalising total accumulated angle
                would decouple them.
drift/mixer     Untested and high-impact per ablation: zeroing the gradient
                costs 4.32 Hartree and RANDOM drift is worse than none, so it is
                the direction that matters. CRZ is diagonal in both registers
                and moves no populations - it only writes phases CRX later
                converts. Nobody has varied that mechanism.
diagnostics     activation_rate is useless (~50% for every k and every mode).
                normalized_entropy measures concentration, not correctness - it
                falls monotonically with k while energy peaks then declines, so
                "walk until concentrated" overshoots. Run-to-run VARIANCE
                tracked quality perfectly but needs repeated runs.

═══ EXTENSIONS WORTH BUILDING ═══

The reusable primitive is more general than ground-state search: encode a
parameter configuration into a state, measure a Hamiltonian-derived property over
a superposition of configurations, extract coordinate-wise structure from the
marginals. Ranked by how much machinery already exists.

COMPARATOR on the energy register - highest leverage, one circuit element,
    unlocks three things at once. QPE now yields a k-bit ENERGY per vertex, so a
    threshold test (E < t) becomes available for the first time:
      * Grover / Durr-Hoyer minimum finding. Replace the CRX mixer with a
        reflection about the low-energy subspace. The current walk is a weak
        biased diffusion, NOT Grover - with an energy register it can be proper
        amplitude amplification: ~sqrt(2^N) ~ 64 iterations per block at N=12 to
        find the EXACT best vertex, against k=15 steps returning a weighted mean.
      * Quantum counting -> adaptive radius. Count the fraction of vertices below
        the current energy: many good ones means R is too small, almost none
        means too large. Replaces R = 0.6*0.9^epoch, an arbitrary schedule
        inherited from nisq_v2's __main__, with a measured quantity.
      * Threshold-conditioned drift, instead of a linear gradient term.
X-BASIS SECOND MOMENT - Re<U> ~ 1 - tau^2<H^2>/2 sits in circuits already being
    run and discarded. Gives Var(H) per vertex free: a diagonal preconditioner
    (what Adam's v term and the diagonal Fisher both estimate expensively), and
    the second moment that folded-spectrum objectives need.
OVERLAP ESTIMATION - the W-gate IS a controlled state preparation, so a Hadamard
    test between two parameter configurations gives <psi(theta_a)|psi(theta_b)>
    with no new machinery. Enables deflation and fidelity objectives.

APPLICATION PIVOTS

excited states     Cheapest and cleanest fit. Folded spectrum minimises
    (grounded)     <(H-omega)^2> = <H^2> - 2 omega <H> + omega^2, and BOTH moments
                   come from the same circuit - Y basis for <H>, X basis for
                   <H^2>. Excited-state search then costs the same as
                   ground-state search, where most VQE variants pay substantially
                   for the second moment. With overlap estimation, deflation gives
                   a spectrum-walking method.
Hamiltonian        Swap what the W-gate encodes: candidate HAMILTONIAN parameters
  learning         rather than ansatz parameters, evolve a known state, compare
    (grounded)     against measured data. The marginal estimator then returns
                   d(loss)/d(H coefficients). Plausibly a BETTER target than
                   chemistry: moderate parameter counts, LOOSE precision - which
                   is exactly the regime where V3 beats Gilyen et al., eps >
                   1/sqrt(d) - and every quantum device needs calibrating.
metrology          Maximise Fisher information over a parameterised probe. QFI
  (speculative)    relates to variance, so the X-basis moment feeds it directly,
                   and V2 already carries an unused QFIM engine.
reservoir          Freeze the walk, inject data through the param register, train
  computing        only a linear readout on ancilla statistics. The encoder and
  (speculative)    nonlinear map already exist. Different goal - classification
                   tolerates far looser precision than chemistry - so it is a
                   spin-off, not a QLTO improvement.

HONEST STEER: chemistry is where V3 measured WORST (H2 last of six), where
classical methods are strongest, and where precision demands are harshest.
Loose-precision, moderate-dimension problems are where the eps > 1/sqrt(d)
scaling actually favours this method. If only one thing gets built, build the
comparator - it unlocks three of the four components above, and it only became
possible once QPE produced an energy register.

═══ NON-CLAIMS ═══

Barren plateaus: NOT addressed. V3 is a cost-function-difference estimator, the
    class Arrasmith et al. (Quantum 5, 558) show is exponentially suppressed on a
    plateau. Smoothing helps rugged landscapes; a plateau has nothing to smooth
    toward.
State preparation / Hilbert-space overlap: SIDESTEPPED, not solved. QPE here
    estimates <H> by averaging sampled eigenvalues, so no ground-state overlap is
    needed - but the difficulty reappears as the ansatz ceiling, where every
    optimiser plateaus. analysis.md's "dissolves the state preparation
    bottleneck" is not supported by this data.
Classical computing: not eliminated. Each epoch still decodes bitstrings, bins
    them, forms the update and sets the next radius classically. What moves into
    the circuit is the OPTIMISER - no Adam moments, no Fisher inversion - not the
    control loop.

DIAGONAL-HAMILTONIAN RULE: a final RZ block commutes with a diagonal H, so its
gradient is identically zero - measured ||g_exact|| = 0.00000 on MaxCut N=4's
last block. Half of efficient_su2's blocks are Z, so a quarter of its parameters
do nothing on those problems. Match the last block's axis to H.

═══ WHY NO QFIM ═══

Every run in this project used use_fim=False, and V2 still placed best overall.
The natural-gradient metric appears to be redundant here, and there is a
mechanism for it rather than just an absence of benefit.

A QFIM preconditioner exists to fix parameter-space conditioning: it rescales
steps so that equal parameter changes produce equal STATE changes. But the walk
never works in parameter space - it evaluates real states at the hypercube
vertices and measures their energies. A direction that barely moves the state
produces vertices with nearly equal energy, so the marginal difference is ~0 and
the walk does not step that way. That is precisely what the metric would have
prescribed, obtained for free. The walk takes many steps of its own (k_steps)
over measured states rather than following a preconditioned route, so it does not
need to be told which way is downhill in a rescaled coordinate system.

Skipping it is also a real saving: the QFIM costs L circuits per epoch (measured
count, not the old formula).

CAVEATS - and the mechanism above is NOT the reason V2's FIM did nothing.

Tested empirically: enabling use_fim in V2 does not help. The cause is V2's USE
of the metric, not the protocol. commute_fim.py implements the block-diagonal
QFIM correctly:

    F_ij = Re<G_i G_j> - <G_i><G_j>,    F_ii = 1 - <G_i>^2   (G^2 = I for Paulis)

and needs no conjugation by the future circuit, because <d_i psi|d_j psi> =
(1/4)<phi|G_i^dag W^dag W G_j|phi> and W^dag W = I cancels - which is exactly why
the protocol is O(L). (Physics verified here; arXiv:2505.09818's exact protocol
not read.)

V2 then departs from natural gradient three ways:
  * DIAGONAL ONLY. commute_fim computes every within-block off-diagonal entry
    from the SAME measurement at zero extra circuit cost, and _execute_walk calls
    np.diag() and discards all of them.
  * 1/sqrt(F) instead of F^-1. Natural gradient is F^-1 g; even a diagonal
    approximation is g_i/F_ii, not g_i/sqrt(F_ii). The square root makes this
    RMSProp-like, not natural-gradient-like.
  * clipped to [0.1, 5.0], capping whatever effect survives.

So V2 never implemented natural gradient, and "QFIM does not help" is not
established - only that THIS usage does not. Direct evidence the proper version
works: benchmark.py's CorrectQNG does pinv(F_block) @ g_block and WON two of
eight problems in the fair suite, including Heisenberg N=8 at -12.1692, the best
result any method reached there.

Also note commute_fim.py's generator detection was broken until this session -
the gate-name test labelled every generator 'Z', so each qubit's two parameters
got identical Pauli strings. Any FIM test predating that fix used a metric built
from duplicated operators.

Two things worth doing before accepting the redundancy argument: use the full
block (drop np.diag) with a proper block solve, and A/B use_fim on the same
problems now that the generators are correct.

Scope: bits_per_param=1, identity metric (no QFIM), single sensing ancilla for
the walk. V2 retains the QPE multi-ancilla walk mode, the QFIM path and the
criticality sensor.

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
from qiskit.synthesis import LieTrotter
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
                 num_ancillas=1, qpe_margin=2.0):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.shot_budget = shot_budget
        self.tau_scale = tau_scale
        self.bits_per_param = 1   # one +-R vertex per parameter; see module docstring
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
        self.layer_diagnostics: Dict[int, Any] = {}
        self.last_activation_rate = 0.0

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

    def _sense_gradient_qpe(self, center_params, search_radius, active_indices):
        """Gradient from QPE sensing: each shot returns a sampled eigenvalue.

        The single-ancilla Hadamard test returns one +-1 bit per shot, so the
        <H> estimate carries variance ~1/(tau^2 S) and tau shrinks as 1/range.
        QPE instead decodes an energy directly, giving Var(H)/S with no tau
        factor - the difference between O(N^2/S) and O(N/S) for an extensive H.

        No sdg here: the phase is read by the inverse QFT, not by a basis
        rotation, so the Y-basis trick of the k=1 path does not apply.
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
            reps = max(1, 2 ** a)
            qc.append(PauliEvolutionGate(self.H_sense, time=t,
                                         synthesis=LieTrotter(reps=reps)).control(1),
                      [anc[a]] + list(sysr))

        qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)

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
            for i in range(n_active):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += energy * cnt
                den[b, i] += cnt

        mean1 = np.divide(num[1], den[1], out=np.zeros(n_active), where=den[1] > 0)
        mean0 = np.divide(num[0], den[0], out=np.zeros(n_active), where=den[0] > 0)
        # Energies are decoded directly, so no 1/tau rescaling and no sign flip.
        grad = np.zeros(len(center_params))
        grad[active_indices] = (mean1 - mean0) / (2.0 * search_radius + 1e-12)
        return grad

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
            for i in range(n_active):
                # identity metric: no QFIM rescaling in V3
                qc.crz(grad_local[i] * gamma * 0.5 * np.pi * drift_gain,
                       anc[0], param[i])
            for i in range(n_active):
                qc.crx(beta, anc[0], param[i])

        qc.h(anc)                                    # phase -> population
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

        activation = anc_ones / total if total else 0.0
        self.last_activation_rate = activation
        probs = np.array(list(counts.values())) / total if total else np.array([])
        ent = -np.sum(probs * np.log2(probs + 1e-12)) if total else 0.0
        self.layer_diagnostics[tuple(active_indices)] = {
            'activation_rate': activation,
            'normalized_entropy': ent / (np.log2(len(counts)) if len(counts) > 1 else 1.0),
        }

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
        return backend.run(t_qc, shots=self.shot_budget).result().get_counts()

    def run_walk(self, center_params, k_steps=20, delta_t=0.5, search_radius=0.5,
                 layer=True):
        """One epoch. Returns (params, energy).

        Per layer: one sensing circuit for the gradient, one walk circuit, one
        energy readout. No gradient-engine circuits.
        """
        self.layer_diagnostics = {}
        blocks = ([l['params'] for l in self.layers] if layer
                  else [list(range(len(center_params)))])

        params = np.asarray(center_params, dtype=float).copy()
        for active in blocks:
            if not active:
                continue
            grad = self.sense_gradient(params, search_radius, active)
            params = self._execute_walk(params, k_steps, delta_t, search_radius,
                                        active, grad)

        # logging only - one circuit, and the one place V3 evaluates H directly
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
