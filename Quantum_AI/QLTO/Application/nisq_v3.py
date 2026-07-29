"""
nisq_v3.py: QLTO with the gradient read out of the sensing circuit itself.

Standalone. Imports numpy and qiskit and nothing else - no nisq_v2, no
commute_gradient, no commute_fim. That is the point of the file: V2 spends
2M-N circuits per epoch inside CommutingBlockGradient, and V3 reaches
comparable accuracy without any gradient engine in the import graph at all.

─────────────────────────────────────────────────────────────────────────────
THE MECHANISM
─────────────────────────────────────────────────────────────────────────────

The W-gate prepares a uniform superposition over the 2^n vertices of the
hypercube {c_i +- R}, each entangled with its own ansatz state. An ancilla-
controlled e^{-iH*tau} plus a Y-basis readout gives, conditioned on the
measured vertex x,

    <Z_anc | x>  =  Im <psi(theta_x)| e^{-iH tau} |psi(theta_x)>  ~=  -tau E(theta_x)

Per-vertex energies are unrecoverable - 2^n vertices, finitely many shots. But
a gradient is a marginal, not a per-vertex quantity:

    g_i  ~  <signal | x_i=1> - <signal | x_i=0>   =   2R d_iE + O(R^3)

the O(R^2) cross terms cancelling under the symmetric +-R perturbation of the
other coordinates. Every shot carries a value for every bit, so all n
components come from the same shots.

Stated honestly this is not exponential parallelism. It is:

    one circuit yields S simultaneous-perturbation gradient samples,
    where classical SPSA needs 2 circuits per sample.

That converts circuit cost into shot cost.

Estimator checked against an exact gradient, N=4, one centre: cosine 0.974 at
200k shots, 0.902 at 8192. The X-basis readout V2 uses for its walk feedback
scores 0.803 at matched settings - a plain H reads Re<U> ~ 1 - tau^2<H^2>/2, a
variance-like signal, while the energy sits in the imaginary part and needs the
Sdg. V3 uses the Y basis for sensing and keeps the X basis for the walk.

─────────────────────────────────────────────────────────────────────────────
MEASURED - Heisenberg N=4, efficient_su2(reps=1), 20 epochs, ONE seed
─────────────────────────────────────────────────────────────────────────────

    optimizer          E_final   E_best   circuits   per epoch
    QLTO walk-grad     -6.020    -6.071        180        9.0
    QLTO engine-grad   -6.082    -6.082       1287       64.3
    AdamW              -5.773    -5.773        640       32.0
    SPSA               -5.890    -5.890        200        2.0
    (exact GS -6.4641)

    at a shared 180-circuit budget:
    walk-grad -6.071 | SPSA -5.882 | AdamW -4.903 | engine-grad -4.497

Within 0.010 Hartree of the engine gradient at 7.2x fewer circuits, and ahead
of SPSA at matched budget. E_best still differs from E_final by 0.05, so the
run is not fully settled - the step size is still slightly too large for the
noise in the sensed direction.

ONE seed, N=4. The earlier draft of this file, which wrapped V2 rather than
standing alone, scored -5.938 best at 284 circuits; the gain here is partly the
cleaner decode (V2's low-activation fallback mangled its bitstring keys) and
partly seed noise, and one run cannot separate the two.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS IS NOT
─────────────────────────────────────────────────────────────────────────────

Not validated beyond N=4 on one seed. Not a simulation result - both QLTO
variants plateau ~0.5 Hartree above the true ground state because
efficient_su2(reps=1) cannot represent it. The claim is about this optimiser's
measurement cost and nothing wider. Against SPSA the margin is 0.05 Hartree on
one seed, which is not evidence of anything yet.

Scope, deliberately narrower than V2: bits_per_param=1, identity metric (no
QFIM), single sensing ancilla. V2 keeps the QPE multi-ancilla mode, the QFIM
path and the criticality sensor.

─────────────────────────────────────────────────────────────────────────────
QPE SENSING (num_ancillas > 1): 4x FEWER SHOTS FOR THE SAME GRADIENT
─────────────────────────────────────────────────────────────────────────────

The single-ancilla Hadamard test returns one +-1 bit per shot, so the <H>
estimate carries variance ~1/(tau^2 S). tau = tau_scale/range(H) shrinks as
O(1/N), making that variance O(N^2/S) - which is why V3 needed a 16x shot
budget to match V2 at Heisenberg N=6.

QPE decodes an eigenvalue sample per shot instead, giving Var(H)/S with no tau
factor at all: O(N/S) for an extensive H.

Measured, Heisenberg N=6, gradient vs the exact analytic gradient over 3 centres:

    sensing        shots   cosine   rel err   depth
    Hadamard k=1    8192   0.7874     0.689     123
    Hadamard k=1   32768   0.9092     0.439     123
    QPE k=4         8192   0.9492     0.347     772
    QPE k=5         8192   0.9535     0.347    1653
    QPE k=6         8192   0.9480     0.340    3446

QPE k=4 at 8192 shots beats the Hadamard test at 32768 on both metrics: a 4x
shot saving. k=4/5/6 are indistinguishable, so the residual error is NOT
resolution-limited - it is the Var(H) sampling floor plus the O(R^3) smearing
bias, both irreducible. Use the smallest k that clears the resolution
requirement; k=6 costs 4.5x the depth of k=4 for nothing.

The price is depth: 772 vs 123, 6.3x. On shots x depth QPE is ~1.6x WORSE, so
this is a win only when the bottleneck is shot count or circuit submissions
rather than coherence time.

END TO END, Heisenberg N=6, 3 seeds, 20 epochs, k_steps=15:

    sensing         shots   E_final     std   circ  depth  total shots
    Hadamard k=1     8192   -8.0812   0.811    180    248        1.5M
    Hadamard k=1    32768   -8.4861   0.632    180    248        5.9M
    QPE k=4          8192   -9.0233   0.081    180    772        1.5M
    QPE k=4         32768   -9.0765   0.058    180    772        5.9M
    V2 (reference)   8192   -9.0909   0.072   1000    658        8.2M

QPE k=4 at 8192 shots ties V2 (1.1 sigma) on 5.6x fewer circuits AND 5.5x fewer
total shots, at comparable depth. At identical shots and circuits, swapping the
Hadamard test for QPE bought 0.94 Hartree and a 10x variance reduction.

PRIOR ART - the concept is established, the instantiation is not obviously so.

  Jordan, PRL 95 050501 (2005), is NOT the right citation. It needs a reversible
  arithmetic blackbox that coherently evaluates f and writes it into a register,
  giving an exact deterministic phase. <psi(theta)|H|psi(theta)> is an
  EXPECTATION VALUE - no such circuit exists, which is why VQE needs repeated
  measurement at all. Jordan's single query depends on that exactness.

  Gilyen, Arunachalam & Wiebe, arXiv:1711.00465, IS the right citation and
  covers this setting explicitly: LCU-based probability->phase oracle conversion
  at O(log 1/eps) overhead, Jordan-style gradient on top, applied to VQE by name
  ("the expected energy ... is mapped to the probability of some measurement
  outcome"), with a figure for converting ground-state energy to a probability.
  Complexity O~(sqrt(d)/eps) probability-oracle queries.

  Where V3 differs, and why it is not merely a worse version:
    * no oracle conversion, no LCU, no coherent arithmetic. The Hamiltonian
      evolution is native to the problem; the gradient comes from classical
      marginal statistics over parameter bits.
    * SCALING TRADE: theirs is O~(sqrt(d)/eps); V3 is O(1/eps^2) shots and
      INDEPENDENT of d, because every component is read from the same shots.
      V3 is cheaper whenever eps > 1/sqrt(d) - at d=48 that is eps > 0.14, and
      cosine 0.95 was measured sufficient to reach V2 parity. Descent needs a
      direction, not a derivative. Their algorithm is better in eps; V3 is
      better in d, and d-dependence is what hurts VQE.
    * depth 772 vs a construction needing coherent QFT arithmetic on the
      parameter register.

  STILL UNCHECKED: whether this specific shallow instantiation - parameter
  superposition + Hamiltonian-native phase kickback + classical marginal
  readout, no oracle conversion - is already published. The concept space is
  mapped; this corner of it may not be.

THREE IMPLEMENTATION BUGS found here, all caught by measurement:
  * tau0 must be pi/(margin * ||H0||), NOT pi/(2^(k-1) * ||H0||). The aliasing
    constraint binds the base unitary; the 2^a ancilla times resolve that turn
    rather than relaxing it. Tell: decoded energy doubled per added ancilla.
    NOTE nisq_v2's use_qpe_sensing path has this same error - it has never been
    enabled, so it has never shown, but it would be wrong if switched on.
  * ancilla bit order: read the printed register unreversed with
    E = -2 pi phi / tau0. Verified against exact <H_sense> across all four
    sign/order combinations; the others are 1.2-2.9x worse.
  * qpe_margin > 1 is required. At margin=1 the extreme eigenvalues sit exactly
    on the +-0.5 wrap boundary and states with weight near the spectrum edges
    decode to a corrupted mean - measured 2.99 error on a state whose true
    energy was -3.00.

UNEXPLAINED: gradient direction reaches cosine 0.999 while the norm ratio comes
out 0.55 and 2.08 across two blocks of the same circuit. Direction is what the
drift consumes so it may be harmless, but the scale error is unaccounted for.

─────────────────────────────────────────────────────────────────────────────
WHAT SIMULATION HIDES: GLOBAL MODE AND ADAPTIVE DEPTH
─────────────────────────────────────────────────────────────────────────────

Two of V3's most useful configurations are cheap on hardware and painful to
simulate, so a simulator benchmark systematically understates them.

GLOBAL MODE (layer=False) puts all M parameters in one register: 2 circuits per
epoch - sensing + walk - plus one optional energy readout, independent of M and
B, against 2B+1 layered. It needs 1+M+N qubits, routine on hardware and
exponential in memory to simulate: 21q = 34 MB at Heisenberg N=4, 31q = 34 GB at
N=6. Measured where it fits, it matches layered accuracy at a third of the
circuits (H2 -1.8495 @ 60 circuits vs -1.8488 @ 180; Heisenberg N=4 -5.834 @
3/epoch vs -5.871 @ 9/epoch). Its walk depth grows as O(M^2) against layered's
B circuits of O(N^2), so it wins on circuit count and loses on gate volume.

ADAPTIVE DEPTH via dynamic circuits - measure and reset the ancilla each walk
step and branch on the outcome, either walking again or stopping to read the
parameters out - is native on hardware (mid-circuit measurement and feed-forward
are standard) and forces a simulator to give up final-state sampling for
shot-by-shot trajectory simulation, turning one simulation into `shots` of them.
It is the natural answer to the k_steps problem below, where the right k is
problem-dependent and must currently be guessed in advance. NOT IMPLEMENTED, and
with a real caveat: the obvious stopping signals do not work. See the axis map -
entropy falls monotonically while energy peaks then declines, and
activation_rate is pinned at ~50%. What a per-step ancilla measurement should
key on is an open question, not a solved one.

─────────────────────────────────────────────────────────────────────────────
AXIS MAP - what is settled, what is open
─────────────────────────────────────────────────────────────────────────────

SETTLED (measured):
  tau           tau_scale / spectral_range(traceless H). The identity term must
                be stripped: under a CONTROLLED evolution it becomes a relative
                phase that attenuates the signal by cos(c*tau) and contaminates
                it with Re<U>. LiH lost ~8x to this; at c*tau = pi/2 the signal
                would vanish outright.
  simulator     Choose by circuit width, not system size. These circuits are
                narrow but maximally entangled across param<->sys, the worst
                case for MPS: 82s vs 0.26s at 13 qubits, 316x.
  shots         Every optimiser pays the same budget. The baselines had been
                running on exact statevector expectations, reproducible to
                5.6e-17, which no hardware can supply.
  both circuits Neither is redundant. Zeroing the sensed gradient costs 4.32
                Hartree, dropping the walk circuit costs 4.71, and random drift
                is WORSE than no drift - so it is the direction that matters.

OPEN, ranked by expected value:
  ansatz        DOMINANT and barely explored. efficient_su2(reps=1) caps at
                -6.1231 on Heisenberg N=4 against exact -6.4641; reps=3 reaches
                exact. Both optimisers plateau within 0.05 of the ceiling, so
                accuracy work IS ansatz work. HVA underperforms as implemented
                (p=4 reaches only -5.146) but its gradients used an invalid shift
                rule for multi-term generators - treat as a loose lower bound.
  drift/mixer   Untested, and the ablation says high-impact. CRZ is diagonal in
                both registers so it moves NO populations; it only writes phases
                that CRX later converts into movement. The entire update rests on
                a phase-then-interfere mechanism nobody has varied. CRY drift,
                reordering, amplitude amplification in place of the mixer: all
                untouched.
  schedule      gamma = s*pi*delta_t grows with step index, so total drift and
                total mixer angle both scale with k. That entangles k with
                effective step size and is the likely cause of the isolated dips
                (H2 layered at k=10; Heisenberg layered AND global at k=20).
                Normalising so total angle is fixed - k controlling granularity
                only - is the change that would make k behave.
  k_steps       Problem-dependent, NOT dimension-dependent. A rule of
                k ~ 3*params_per_walk was fitted and then refuted: global
                searches 16 parameters and still peaks at k=10, same as layered's
                4. It depends on the landscape, the starting point and the path.
                Default 15; k=10 looked marginally better on Heisenberg in both
                modes.
  encoding      bits_per_param=1 only. Linear vs log vs Gray untested (v2 has a
                dead mode='log' path).
  decode rule   Weighted mean over anc=1 shots. Softmax weighting or best-vertex
                untested.
  diagnostics   activation_rate is USELESS: ~50% for every k and every mode,
                because the ancilla starts in equal superposition and the energy
                bias is first order in tau. normalized_entropy measures
                concentration but not correctness - it falls monotonically with k
                while energy peaks and declines, so "walk until concentrated"
                would overshoot. Run-to-run VARIANCE tracked quality perfectly in
                both sweeps (minimum std coincided with best energy every time)
                but needs repeated runs, so it cannot drive a within-run rule.
  W-dagger      Applied before measurement for faithfulness to V2, but it is
                block-diagonal in the param basis and cannot change the measured
                (param, anc) distribution. Should be removable, halving the
                walk's W contribution. Untested.
  point-energy  One circuit per epoch, logging only - the optimiser never reads
                it. Droppable, or run every k-th epoch.
  scale         N=8, N=12, multiple seeds. The variance argument predicts the
                estimator is dimension-independent. That is a prediction.

PROBLEM-DEPENDENT RULE worth remembering: for a DIAGONAL Hamiltonian (MaxCut,
Ising) a final RZ block commutes with H and its gradient is identically zero -
measured ||g_exact|| = 0.00000 on MaxCut N=4's last block. Half of
efficient_su2's blocks are Z, so a quarter of its parameters do nothing on those
problems. Match the last block's axis to the Hamiltonian's structure.

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
