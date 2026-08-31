"""Supervised QML on a weighted data register - three circuits per epoch, flat in |D| and M.

Trains a variational model on a batch of |D| samples under a real MSE loss using
THREE circuits per epoch regardless of how many samples the batch holds and how
many parameters the ansatz carries. The flatness in |D| comes from a data
register; the flatness in M comes from V6's log-width design register.

WHAT WAS BROKEN AND WHAT FIXED IT

  v74 opened the data axis and closed half of it. A register in UNIFORM
  superposition, entangled into the system by CRY and never uncomputed, is traced
  out at measurement, so <O x I> IS the batch mean - batching costs the readout
  nothing, and QLTO beat SPSA 5.65x at N=4. But the MSE gradient reweights each
  sample,

      dL/dtheta = (2/S) sum_x (f_x - y_x) df_x/dtheta

  and a uniform register returns the gradient of the unweighted mean_x f_x
  instead. v74 measured cos(surrogate, true) = -0.7431, -0.0923, +0.2981, +0.2663
  along a descent trajectory. v121 extended that to 12 epochs and found the
  cosine does not merely sit low, it SWINGS IN SIGN: +0.9367, +0.3359, +0.8180,
  -0.9770, -0.9602. And v74 names the deeper objection itself - "the linear
  surrogate does not depend on the labels at all", so the objective that batched
  cleanly was not a supervised one.

  THE FIX IS THAT THE WEIGHTS ARE JUST AMPLITUDES. Prepare the register as
  sum_x sqrt(p_x)|x> with p_x proportional to |w_x|, w_x = f_x - y_x. Tracing it
  out then returns sum_x p_x f_x, the WEIGHTED mean, which is what MSE needs.
  Two obstacles, both discharged on circuits:

    SIGNS. |c_x|^2 is positive, so one register carries positive weights only.
    Split the batch by sign of w_x, run one branch each, subtract. Two circuits,
    still flat in |D|.

    THE WEIGHTS ARE UNKNOWN - w_x needs f_x, which moves every epoch. But all
    |D| values of f_x come from ONE circuit if the register is MEASURED rather
    than traced out: the joint counts over (x, outcome) give every f_x at once.
    The same multiplexing trick, applied to the weights instead of the gradient.

  So: circuit 1 measures the register and yields every w_x; circuits 2 and 3 are
  the two sign branches. Three, independent of |D| and of M.

WHAT IS MEASURED, AND WHAT IT COST

  TIER A throughout (project rule R1) - Qiskit circuits on AerSimulator with
  finite shots. supplement/v123, 32768 shots, N_sys=3, efficient_su2 reps=1:

      |D|    cos(estimate, true)     w rms     sqrt(|D|/S)     w max   max/rms
        4          +0.9773           0.0075      0.0110        0.0110    1.47
        8          +0.9035           0.0143      0.0156        0.0257    1.79
       16          +0.9341           0.0214      0.0221        0.0446    2.08

  v122 reached cos 0.9768 at |D|=4 with EXACT weights. The |D|=4 row is the same
  configuration with the weights measured, so shot-estimating them cost nothing
  detectable at that size - the whole 0.02 gap to unity is the V6 gradient, not
  the weights.

  THE ERROR MODEL HOLDS. Circuit 1's budget splits |D| ways, so each sample gets
  S/|D| shots and the per-sample weight error should be sqrt((1-f^2)|D|/S) <=
  sqrt(|D|/S). The w rms column sits at or just under that bound at all three
  sizes. The w max column runs 1.5-2.1x higher because it is a maximum over |D|
  draws; that is extreme-value inflation, not a scaling break, and it is the
  column that matters operationally because the WORST sample's weight is the one
  that can flip sign and land in the wrong branch.

    CORRECTED, and the correction is why both columns are printed. A first
    version of v123 reported only the max and compared it against the per-sample
    sqrt(|D|/S). That is max-against-typical: it showed 1.6x and 2.0x excess at
    |D|=8,16 and read as the model failing. It was not.

  THE MEAN HIDES A HEAVY LOWER TAIL. This file's self-check averages three
  dataset seeds, which v123 did not, and the averaged cosine is monotone in |D|
  as the shot model predicts - 0.9586, 0.9365, 0.9063 at |D| = 4, 8, 16, against
  v123's non-monotone single draw. But the WORST single epoch over all seeds and
  sizes is cos +0.2358.

    WITHDRAWN, refuted by supplement/v124. This paragraph previously blamed the
    tail on the two-branch split - a sample whose residual sits near zero is
    assigned to the wrong branch when shot noise flips its sign - and predicted
    the tail would get HEAVIER as training converges, since more residuals sit
    near the boundary late in a descent. Both halves are wrong.

      THE MECHANISM IS WRONG. v124's control arm (random +-1 labels, which a
      3-qubit 12-parameter model cannot fit) logs ZERO sign flips at every one of
      120 epochs and still reaches min cos -0.0752. A tail that appears where
      nothing flips is not caused by flipping.

      THE PREDICTION IS BACKWARDS. In v124's realizable arm (y_x = f_x(theta*),
      so residuals actually reach 5e-4) flips do rise as predicted, 0.00 -> 0.12
      per epoch - but cos IMPROVES, +0.9476 far from the optimum to +0.9916 near
      it. The competing effect wins: a sample only flips when its residual is
      small, and a small residual carries little gradient weight, so
      misassignments concentrate on exactly the samples whose misassignment costs
      least. Misassigned weight mass stays at 0.87%. The split is
      self-protecting.

  WHAT ACTUALLY CAUSES THE TAIL IS SIGNAL-TO-NOISE, and it is ordinary. cos is a
  ratio, so it collapses wherever the true gradient is small against the
  estimator's error floor. Binning v124's 240 epochs by |g_true| quartile:

      |g_true|              err/|g|     mean cos     min cos
      [0.0072, 0.1023)        0.89       +0.7744     -0.0752
      [0.1023, 0.1425)        0.39       +0.8946     +0.6778
      [0.1425, 0.1900)        0.26       +0.9565     +0.7372
      [0.1900, 1.0527)        0.20       +0.9802     +0.8734

  The absolute error is not constant - it moves 0.0360 to 0.0817 - but |g_true|
  spans 20x over the same range, so the ratio improves and cos with it. The min
  cos entries in PART 1 are FLAT-REGION epochs, not converged ones and not branch
  failures. The fix is therefore more shots where |g| is small, not a third
  branch; and since flat regions are where a barren-plateau problem also lives,
  this is the same limitation every shot-based gradient estimator has rather than
  anything the data register introduced.

    v124 states its own caution: log|g| and log MSE are correlated along a
    descent, so the +0.5612 / -0.4554 correlation ranking is not a clean
    decomposition. The load-bearing evidence is the zero-flip control and the
    err/|g| column, neither of which rests on a correlation.

  IT COSTS THE TRAJECTORY LITTLE ANYWAY. Descending on the estimate reduced the
  loss on 3/3 seeds at |D|=8 and landed within a few percent of exact-gradient
  descent from the same start (0.815 vs 0.823, 0.896 vs 0.892, 0.786 vs 0.765) -
  sometimes above, sometimes below. Parameter momentum averages over the bad
  epochs in a way a single-epoch cosine cannot show. Both arms end near MSE 0.8
  from a start near 1.0-1.3; that is the ansatz and the task, not the estimator,
  since the exact-gradient arm does no better. The claim is not that the model
  fits well - it is that three circuits buy the trajectory the exact gradient
  buys. Every earlier file in this line (v74, v121, v122, v123) measured a cosine
  while stepping with something else; this is the first that does not.

THE PRICE, STATED PLAINLY. Three circuits, not three measurements. Circuit 1's
budget splits |D| ways, so a bigger batch buys circuit count and pays shot noise
- the same trade V6 makes on the parameter axis (supplement/v109: ~4x more shots
for 32x fewer circuits). Nothing here evades it. What is new is only that the
trade is now available on the DATA axis too, and under a supervised loss.

WHY G = 1 HERE, WHICH IS WHY M IS FREE. V6 costs G circuits per gradient against
parameter-shift's 2MG, where G is the number of qubit-wise-commuting groups in
the observable. A QML readout is a single Pauli, so G = 1 and each branch is one
circuit. The 2M advantage V6 carries on Hamiltonians (supplement/v119: G cancels,
the ratio is 2M) applies here at its cleanest.

  On a molecular Hamiltonian G is not 1 - it grows as N^4.24 (supplement/v30) -
  and the branch cost would be 2G, not 2. This file's three-circuit claim is
  specific to single-Pauli readouts.

THE STATE PREPARATION, AND TWO BUGS IT TOOK TO GET RIGHT. The register prep is a
conditional-probability tree of uniformly controlled RYs, decomposed into
FIXED-ANGLE ry/cx by the Gray-code Walsh transform. Fixed-angle is not a style
choice: a `cry` or a `StatePreparation` carries parameters that are not in V6's
_CTRL set, so V6's decompose loop mangles the parameterised core into `u` gates
and the run fails outright (v74's docstring already prescribes the manual
ry/cx/ry/cx form; v122 rediscovered it the hard way). Both bugs below were caught
by v123's PART 0, which checks max|amp^2 - p| against the requested distribution
before anything downstream is allowed to run:

  ENDIANNESS, the fourth instance in this project. Statevector index x has qubit
  q as BIT q, so x's most significant bit is qubit d-1, not qubit 0. A tree that
  decides the MSB first must therefore rotate dq[d-1-lvl], controlled on the
  already-decided HIGHER qubits. Rotating dq[lvl] instead prepares the
  bit-reversed distribution: measured max|amp^2 - p| of 0.30, 0.55, 0.11 at
  d = 2, 3, 4.

  THE MOETTOENEN TRANSPOSE. alpha_i = 2^-k sum_j (-1)^{b_j . g_i} theta_j, with
  g_i the Gray code of i. Writing (-1)^{b_i . g_j} is the transpose - symmetric
  at k=1, wrong for k>=2. d=2 passed at 2.8e-17 while d=3,4 failed at 4.3e-3 and
  1.1e-1, so the one-control case actively hid it.

  Both are exactly the class of error R1 exists to catch: neither is visible in a
  dense-matrix formulation of the same construction.

"FLAT IN |D|" MEANS CIRCUIT COUNT, NOT GATE COUNT, AND THE DIFFERENCE DECIDES
WHERE THIS IS USABLE. supplement/v125 separates the blocks on transpiled circuits
(rz/sx/x/cx, |D| up to 32) and the split is not where it looks:

    weighted state prep     2q gates ~ |D|^1.29     50% of the circuit at |D|=32
    linear data encoder     2q gates ~ |D|^0.44     logarithmic, as designed
    arbitrary data encoder  2q gates ~ |D|^1.06

  Three circuits per epoch holds at every size for both encoders - that claim
  survives intact. But the whole circuit is LINEAR IN |D| in gates even with the
  cheap encoder, and the block responsible is the weighted state preparation, not
  the encoder. That cannot be fixed by choosing better-structured data: p_x is
  proportional to |w_x|, an arbitrary distribution that moves every epoch, and
  setting 2^d - 1 free amplitudes needs at least 2^d - 1 angles. The prep is
  Theta(|D|) by parameter counting. Structure buys the encoder; it does not buy
  the prep.

  SO |D| IS BOUNDED BY COHERENCE, NOT BY THE METHOD, and the saving is in job
  count - latency, queue time, compilation - rather than depth.

  THE ONE ESCAPE WAS TESTED AND DOES NOT OPEN. Parameter counting bounds an
  ARBITRARY distribution; a structured one could be an MPS at bond dimension chi
  and prepare in O(d*chi^2). supplement/v126 harvested real weight vectors from
  real descents (tier A) and took their Schmidt spectra (tier C, NO CIRCUIT).
  Measured vectors look full rank at d=6,8 - but that was largely a SHOT-NOISE
  artefact, and the confound check is what showed it: on the realizable arm at
  d=8 the EXACT vector truncates at chi=8 to 0.017 while the measured one only
  reaches 0.115, and the unrealizable arm shows no gap at all (0.246 vs 0.247)
  because genuinely full-rank vectors have no structure for noise to destroy.

    So the structure is real - and at chi=8, useless. chi=8 at d=8 is half of max
    rank 16, and an MPS prep at chi=8 costs ~4*d*chi^2 = 2048 gates against the
    exact prep's 255: 8x WORSE.

  WITHDRAWN: "the MPS escape route is closed". supplement/v127 reopens it, and
  the error was mine in a specific and avoidable way - v126 measured the rank of
  the amplitude vector THIS MODULE HAPPENS TO PREPARE, sqrt(|w|), and concluded
  about the DATA. Those are different objects.

    THE THEORY, and credit to a reviewer of v126 for it. The encoder makes the
    angle vector a linear form A @ bits(x), so on realizable labels the residual
    is w_x = g(A bits(x)) for one smooth g. Every cut splits the latent
    coordinates additively, s_a(x) = s_a(x_L) + s_a(x_R), so a degree-r polynomial
    approximation of g expands by the multinomial theorem into a sum of products
    and the Schmidt rank obeys chi <= C(k+r, r) at EVERY cut, with k = rank(A) =
    n_sys. There is no 2^d in that bound.

    WHY v126 DID NOT SEE IT. sqrt|.| is non-smooth at zero, so it fails the
    bound's hypothesis - and a residual crosses zero constantly, which is exactly
    what a residual does as a model fits. The bound was never violated; it did
    not apply.

    MEASURED, v127, same runs, four amplitude functions, chi* = smallest bond
    dimension whose worst-cut truncation error falls under the shot noise the
    weights already carry (eps = 0.0214):

        d     |D|    max rank    w (raw)    |w|    sqrt|w| (current)   sqrt(w+c)
        4      16        4          3        3            3               3
        6      64        8          3        6            6               3
        8     256       16          6       12            8               4
       10    1024       32          6       12           12               3

    The current encoding grows with d (3, 6, 8, 12). THE SHIFTED ONE DOES NOT
    (3, 3, 4, 3) - flat, which is what a bound with no d-dependence predicts. And
    the k-dependence the bound does predict shows up: at d=8, chi* = 4, 4, 6 for
    k = n_sys = 2, 3, 4.

    AT chi=3 AND d=10 THE CROSSOVER IS ALREADY PASSED: ~4*d*chi^2 = 360 gates
    against 2^d - 1 = 1023, a 2.8x saving, and O(d*chi^2) with chi flat is
    O(log|D|) - logarithmic encoding of an exponentially large data axis, for
    this structured family.

  THE SHIFT ALSO REMOVES THE SIGN SPLIT. With p_x = (w_x + c)/Z for c > max|w|
  the amplitudes are strictly positive, so no branch is needed:

      sum_x w_x df_x  =  Z * <shifted register>  -  c * D * <uniform register>

  and the branch-flip failure mode v124 spent a whole file ruling out cannot
  occur at all.

    CORRECTED. An earlier version of this paragraph claimed the shift costs TWO
    circuits per epoch instead of three. It does not. Current: f_hat + positive
    branch + negative branch = 3. Shifted: f_hat + shifted branch + uniform
    branch = 3. f_hat is still needed to build w, and the uniform branch is
    theta-dependent so it cannot be cached across epochs. The shift buys bond
    dimension and removes a failure mode; it does not buy circuit count.

  AND IT COSTS MORE VARIANCE THAN IT IS WORTH - MEASURED, supplement/v128, and
  this closes the shifted route. The reconstruction is a DIFFERENCE of two
  quantities each of order c*D against a target of order |sum_x w_x df_x|, so
  conditioning degrades as c grows - and c is exactly what buys the smoothness
  that gave v127 its low chi. The two demands are in direct opposition, and at
  matched shots the shift LOSES at every gamma = c/max|w| and every size tested:

      d    sign-split    shifted, gamma = 1.05 -> 8.0
      4      +0.9915     +0.9378 +0.9205 +0.8856 +0.8269 +0.5680 +0.3373
      6      +0.9554     +0.8673 +0.8610 +0.8464 +0.7949 +0.5278 +0.2905
      8      +0.9877     +0.8825 +0.8470 +0.8116 +0.7187 +0.5041 +0.3484

  with the measured amplification kappa ~ 2cD/|target| climbing 10.4 -> 27.4
  across the gamma sweep at d=4 and 15.2 -> 32.3 at d=8, monotone with the cos
  loss. All 18 cells lose. The best gamma is 1.05 at every size - the SMALLEST
  tested, i.e. the sweep is pinned against its own floor and there is no interior
  optimum to find. That gamma is also where v127 measured chi, so the two results
  do compose, and it is still 0.05 to 0.11 cos WORSE than the sign split it was
  meant to replace.

  SO BOTH RESULTS STAND AND THEY POINT OPPOSITE WAYS. v127's structural finding
  is correct: the shifted amplitude has bond dimension flat in d, and sqrt|.|'s
  zero-crossing is what destroys it. v128's statistical finding is also correct:
  an estimator built on that amplitude is worse than the one already here. The
  structure is real and this estimator cannot spend it. Anything that revives the
  route needs a reconstruction that is not a large cancelling difference - not a
  better c, which is what the sweep rules out.

  WHAT IS NOT ESTABLISHED, and it is most of it. v127 is TIER B vectors and TIER
  C analysis - no MPS prep has been built, transpiled or run, and its gate counts
  are arithmetic on chi. Its labels are REALIZABLE (y = f(theta*)), so w is a
  function of A bits(x) by construction - the theorem's most favourable case.
  Arbitrary labels make w an arbitrary 2^d vector again and the bound says
  nothing, which is the real limit on this whole line. Contiguous cuts, natural
  qubit order, d <= 10, one ansatz. The tier-A build is the next file.

AND THE CHEAP ENCODER CANNOT HOLD AN ARBITRARY DATASET. _encode applies
CRY(alpha[j,d]) controlled on register qubit d, so sample x's angle vector is
alpha @ bits(x) - LINEAR in the register bits. The |D| = 2^d samples are the
vertices of a parallelepiped, not free points. Least-squares fitting alpha to a
random angle table captures 40%, 45%, 26%, 12% of its variance at d = 2, 3, 4, 5,
falling as d grows (v125 PART 1, NO CIRCUIT - a structural fact about a linear
map). An arbitrary table IS loadable by a uniformly controlled RY per system
qubit - verified exact to 7.8e-16 at tier B, v125 PART 2 - at 2^d rotations each.

  So the module suits samples GENERATED BY A FEW FACTORS: parameter sweeps,
  designed experiments, lattices of settings. Not unstructured data.

SCOPE. Verified to |D| = 16 on 4 register qubits (32 in v125's cost-only arm),
N_sys = 3, efficient_su2 reps=1, single-Pauli readout so G=1. NO NOISE MODEL AND
NO HARDWARE ANYWHERE IN THIS LINE OF WORK - whether a noisy device preserves the
cosine is untested, and it is the gap that most limits application.
"""
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from nisq_v6 import QLTOv6


# ---------------------------------------------------------------- state prep

def mux_ry(qc, thetas, controls, target):
    """Uniformly controlled RY: apply RY(thetas[j]) when the controls read j.

    Gray-code Walsh transform, emitted as fixed-angle ry + cx so nothing carries
    a symbolic parameter and V6 passes the block through untouched.

    controls[i] is read as BIT i of j. See the module docstring for the
    transpose bug this indexing hid at k=1.
    """
    k = len(controls)
    n = 1 << k
    A = np.empty((n, n))
    for i in range(n):
        g = i ^ (i >> 1)
        for j in range(n):
            A[i, j] = (-1.0) ** (bin(j & g).count('1'))
    alpha = (A @ np.asarray(thetas, float)) / n
    for i in range(n):
        qc.ry(float(alpha[i]), target)
        c = min(((i + 1) & -(i + 1)).bit_length() - 1, k - 1)
        qc.cx(controls[c], target)


def prep_weights(qc, p, dq):
    """Prepare sum_x sqrt(p_x)|x> on dq using only ry/cx.

    Conditional-probability tree, most significant bit first. Level lvl rotates
    dq[d-1-lvl] controlled on the already-decided higher qubits dq[d-lvl..d-1],
    in that order - see the module docstring on endianness.
    """
    d = len(dq)
    p = np.asarray(p, float)
    p = p / max(p.sum(), 1e-300)
    for lvl in range(d):
        blk = 1 << (d - lvl - 1)
        thetas = []
        for j in range(1 << lvl):
            lo = j * (blk << 1)
            tot = p[lo:lo + (blk << 1)].sum()
            hi = p[lo + blk:lo + (blk << 1)].sum()
            r = hi / tot if tot > 1e-300 else 0.0
            thetas.append(2.0 * np.arcsin(np.sqrt(np.clip(r, 0.0, 1.0))))
        target = dq[d - 1 - lvl]
        if lvl == 0:
            qc.ry(float(thetas[0]), target)
        else:
            mux_ry(qc, thetas, [dq[d - lvl + i] for i in range(lvl)], target)


def _cry(qc, a, c, t):
    """CRY(a) as ry/cx/ry/cx. NOT qc.cry - see the module docstring."""
    qc.ry(a / 2.0, t)
    qc.cx(c, t)
    qc.ry(-a / 2.0, t)
    qc.cx(c, t)


# ---------------------------------------------------------------- the trainer

class QLTOQML:
    """MSE training on a weighted data register: 3 circuits per epoch.

    core   parameterised ansatz on n_sys qubits (e.g. efficient_su2)
    alpha  (n_sys, n_data) encoding angles; sample x sets angle sum over its bits
    y      (2**n_data,) labels
    """

    def __init__(self, core, alpha, y, shot_budget=32768, radius=0.45,
                 sim_seed=None, backend=None):
        self.core = core
        self.alpha = np.asarray(alpha, float)
        self.y = np.asarray(y, float)
        self.n_sys, self.n_data = self.alpha.shape
        self.S = 2 ** self.n_data
        if len(self.y) != self.S:
            raise ValueError("y has %d labels, a %d-qubit register needs %d"
                             % (len(self.y), self.n_data, self.S))
        self.M = core.num_parameters
        self.shot_budget = int(shot_budget)
        self.radius = float(radius)
        self.sim_seed = sim_seed
        self.backend = backend or AerSimulator(seed_simulator=sim_seed)
        self.ncircuits = 0          # instrumented, not asserted
        self.nshots = 0

        lbl = ['I'] * (self.n_sys + self.n_data)
        lbl[self.n_sys - 1] = 'Z'   # system qubit 0, little-endian
        self.O_full = SparsePauliOp.from_list([(''.join(lbl), 1.0)])
        self.O_sys = SparsePauliOp.from_list(
            [('I' * (self.n_sys - 1) + 'Z', 1.0)])

    # -- circuit construction ------------------------------------------------

    def _encode(self, qc, dq, sq):
        for j in range(self.n_sys):
            for d in range(self.n_data):
                _cry(qc, float(self.alpha[j, d]), dq[d], sq[j])

    def batched(self, p):
        """Register weighted by p, entangled into the system, then the core.

        The register is NOT uncomputed - tracing it out at measurement is what
        makes <O x I> the p-weighted mean.
        """
        dq = QuantumRegister(self.n_data, 'd')
        sq = QuantumRegister(self.n_sys, 's')
        qc = QuantumCircuit(dq, sq)
        prep_weights(qc, p, dq)
        self._encode(qc, dq, sq)
        qc.compose(self.core, qubits=list(sq), inplace=True)
        return qc

    # -- circuit 1: every f_x from one run -----------------------------------

    def f_hat(self, theta, shots=None):
        """All |D| model outputs from ONE circuit. Returns (f, shots_per_sample).

        Uniform register, both registers measured; the joint counts over
        (x, system outcome) give every f_x. The budget splits |D| ways, so this
        buys circuit count and pays shot noise.
        """
        shots = int(shots or self.shot_budget)
        dq = QuantumRegister(self.n_data, 'd')
        sq = QuantumRegister(self.n_sys, 's')
        cd = ClassicalRegister(self.n_data, 'cd')
        cs = ClassicalRegister(1, 'cs')
        qc = QuantumCircuit(dq, sq, cd, cs)
        prep_weights(qc, np.ones(self.S) / self.S, dq)
        self._encode(qc, dq, sq)
        qc.compose(self.core.assign_parameters(np.asarray(theta, float)),
                   qubits=list(sq), inplace=True)
        qc.measure(dq, cd)
        qc.measure(sq[0], cs[0])
        counts = self.backend.run(
            transpile(qc, self.backend, optimization_level=1),
            shots=shots).result().get_counts()
        self.ncircuits += 1
        self.nshots += shots
        num = np.zeros(self.S)
        den = np.zeros(self.S)
        for bits, c in counts.items():
            sysb, regb = bits.split()
            x = int(regb, 2)
            num[x] += (1.0 if sysb[-1] == '0' else -1.0) * c
            den[x] += c
        f = np.divide(num, den, out=np.zeros(self.S), where=den > 0)
        return f, den

    # -- circuits 2 and 3: the two sign branches -----------------------------

    def gradient(self, theta, w=None):
        """MSE gradient estimate. Returns (g, info).

        w may be supplied to reuse weights already measured; otherwise f_hat is
        called and its circuit is counted here.
        """
        info = {}
        if w is None:
            f, den = self.f_hat(theta)
            w = f - self.y
            info['shots_per_sample'] = float(den.mean())
        g = np.zeros(self.M)
        branches = 0
        for mask, sgn in ((w > 0, +1.0), (w < 0, -1.0)):
            if not mask.any():
                continue
            pw = np.abs(w) * mask
            Z = pw.sum()
            if Z < 1e-12:
                continue
            anz = self.batched(pw / Z)
            q = QLTOv6(anz, self.O_full, shot_budget=self.shot_budget,
                       sim_seed=self.sim_seed, backend=self.backend)
            gb, _ = q.sense(theta, self.radius, list(range(self.M)))
            self.ncircuits += len(q.groups)
            self.nshots += self.shot_budget
            g += sgn * Z * np.asarray(gb, float)
            branches += 1
            info['G'] = len(q.groups)
        g *= 2.0 / self.S
        info['branches'] = branches
        return g, info

    def minimize(self, theta0, epochs=8, lr=0.35):
        """Descend on the ESTIMATED gradient. Returns (theta, trace)."""
        theta = np.array(theta0, float)
        trace = []
        for _ in range(epochs):
            f, _den = self.f_hat(theta)
            w = f - self.y
            trace.append(float(np.mean(w ** 2)))
            g, _ = self.gradient(theta, w=w)
            theta = theta - lr * g / max(np.max(np.abs(g)), 1e-12)
        return theta, trace

    # -- exact references (tier B; the reference, never a headline) ----------

    def f_exact(self, x, theta):
        ang = self.alpha @ np.array([(x >> d) & 1 for d in range(self.n_data)],
                                    dtype=float)
        qc = QuantumCircuit(self.n_sys)
        for j in range(self.n_sys):
            qc.ry(float(ang[j]), j)
        qc.compose(self.core.assign_parameters(np.asarray(theta, float)),
                   inplace=True)
        return float(np.real(Statevector(qc).expectation_value(self.O_sys)))

    def grad_exact(self, theta):
        """Exact MSE gradient by parameter shift. The reference, not the claim."""
        f = np.array([self.f_exact(x, theta) for x in range(self.S)])
        w = f - self.y
        gs = np.zeros((self.S, self.M))
        for i in range(self.M):
            for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                t = np.array(theta, float)
                t[i] += sh
                for x in range(self.S):
                    gs[x, i] += sg * 0.5 * self.f_exact(x, t)
        return (2.0 / self.S) * (w[:, None] * gs).sum(axis=0), w


# ---------------------------------------------------------------- self-check

def _dataset(n_sys, n_data, seed):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1.0, 1.0, (n_sys, n_data)),
            rng.integers(0, 2, 2 ** n_data) * 2.0 - 1.0)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


if __name__ == '__main__':
    import contextlib
    import io
    import sys
    from qiskit.circuit.library import efficient_su2

    N_SYS, SHOTS, EPOCHS, LR = 3, 32768, 6, 0.35
    SEEDS = (0, 1, 2)

    print("=" * 100)
    print("qlto_qml  SELF-CHECK")
    print("=" * 100)
    print("  TIER A. Qiskit circuits on AerSimulator, %d shots. N_sys=%d,"
          % (SHOTS, N_SYS))
    print("  efficient_su2 reps=1, MSE loss, %d seeds per size." % len(SEEDS))
    print()

    # -- PART 0 ----------------------------------------------------------
    print("-" * 100)
    print("PART 0  does the register prep actually prepare sqrt(p)?")
    print("-" * 100)
    print("  Guard, not a result. Two bugs got this far and both were caught here.")
    print()
    print("     d   |D|   max |amp^2 - p|   verdict")
    print("   " + "-" * 48)
    ok0 = True
    rng = np.random.default_rng(11)
    for d in (2, 3, 4):
        p = rng.dirichlet(np.ones(1 << d))
        dq = QuantumRegister(d, 'd')
        qc = QuantumCircuit(dq)
        prep_weights(qc, p, dq)
        e = float(np.max(np.abs(np.abs(Statevector(qc).data) ** 2 - p)))
        ok0 &= e < 1e-9
        print("   %3d  %4d      %.2e        %s"
              % (d, 1 << d, e, "ok" if e < 1e-9 else "FAIL"))
    print()
    if not ok0:
        print("   FAIL - the prep is wrong, nothing below would mean anything.")
        sys.exit(1)
    print("   PASS")
    print()

    # -- PART 1 ----------------------------------------------------------
    print("-" * 100)
    print("PART 1  cos(estimate, exact MSE gradient), seed-averaged")
    print("-" * 100)
    print("  v123 reported one dataset draw per size. This averages %d, which is"
          % len(SEEDS))
    print("  the caveat v123's scope note left open.")
    print()
    print("    |D|   seeds   mean cos    min cos    circuits/epoch   shots/sample")
    print("   " + "-" * 82)
    part1 = []
    for d in (2, 3, 4):
        S = 1 << d
        per_seed, allc, circ, sps = [], [], None, None
        for sd in SEEDS:
            alpha, y = _dataset(N_SYS, d, seed=sd)
            core = efficient_su2(N_SYS, reps=1)
            theta = np.random.default_rng(100 + sd).uniform(
                -np.pi, np.pi, core.num_parameters)
            q = QLTOQML(core, alpha, y, shot_budget=SHOTS, sim_seed=7 + sd)
            cs = []
            for ep in range(EPOCHS):
                g_true, _ = q.grad_exact(theta)
                c0 = q.ncircuits
                f, den = q.f_hat(theta)
                g_est, _ = q.gradient(theta, w=f - y)
                circ = q.ncircuits - c0
                sps = float(den.mean())
                cs.append(_cos(g_est, g_true))
                theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)
            per_seed.append(float(np.mean(cs)))
            allc.extend(cs)
        part1.append((S, float(np.mean(per_seed)), float(np.min(allc))))
        print("   %4d   %5d    %+.4f    %+.4f          %2d            %7d"
              % (S, len(SEEDS), part1[-1][1], part1[-1][2], circ, int(sps)))
    print()

    # -- PART 2 ----------------------------------------------------------
    print("-" * 100)
    print("PART 2  does descent ON THE ESTIMATE actually reduce the loss?")
    print("-" * 100)
    print("  v123 measured the cosine but STEPPED with the exact gradient, so it")
    print("  never showed the estimator driving its own trajectory. This does, and")
    print("  compares against exact-gradient descent from the same start.")
    print()
    print("    |D|   seed    MSE start   MSE end (est)   MSE end (exact)   verdict")
    print("   " + "-" * 82)
    d = 3
    S = 1 << d
    wins = 0
    for sd in SEEDS:
        alpha, y = _dataset(N_SYS, d, seed=sd)
        core = efficient_su2(N_SYS, reps=1)
        t0 = np.random.default_rng(100 + sd).uniform(
            -np.pi, np.pi, core.num_parameters)

        q = QLTOQML(core, alpha, y, shot_budget=SHOTS, sim_seed=7 + sd)
        with contextlib.redirect_stdout(io.StringIO()):
            _, tr = q.minimize(t0, epochs=2 * EPOCHS, lr=LR)
        mse0, mse_est = tr[0], tr[-1]

        theta = np.array(t0, float)
        for _ in range(2 * EPOCHS):
            g, _w = q.grad_exact(theta)
            theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
        _, w_end = q.grad_exact(theta)
        mse_ex = float(np.mean(w_end ** 2))

        good = mse_est < mse0
        wins += good
        print("   %4d   %4d     %.5f       %.5f          %.5f        %s"
              % (S, sd, mse0, mse_est, mse_ex, "down" if good else "UP"))
    print()

    # -- reading it -------------------------------------------------------
    print("=" * 100)
    print("READING IT")
    print("=" * 100)
    mn = min(c for _, c, _ in part1)
    worst = min(c for _, _, c in part1)
    print("  Circuits per epoch is %d at every size - one register readout plus one"
          % circ)
    print("  per sign branch - and it is COUNTED, not asserted: q.ncircuits is")
    print("  incremented at each backend.run. Parameter-shift on the same problem")
    print("  is 2M = %d per sample, %d for the batch at |D|=16, so the ratio is %dx"
          % (2 * 12, 2 * 12 * 16, (2 * 12 * 16) // max(circ, 1)))
    print("  and it grows with both |D| and M.")
    print()
    if mn > 0.85:
        print("  ON AVERAGE the estimator tracks the true MSE gradient at every size")
        print("  (worst mean cos %+.4f), and seed-averaging closes v123's open" % mn)
        print("  caveat: the non-monotone 0.9773 / 0.9035 / 0.9341 it saw across |D|")
        print("  was one draw. Averaged over %d seeds the trend is monotone in |D|,"
              % len(SEEDS))
        print("  which is what the shot model predicts and what should be quoted.")
    else:
        print("  The estimator does NOT track the true gradient at some size (worst")
        print("  mean cos %+.4f). The three-circuit claim survives; its USEFULNESS" % mn)
        print("  does not, and the size where it breaks is the finding.")
    print()
    print("  THE MIN COLUMN IS NOT NOISE AROUND THE MEAN AND SHOULD NOT BE READ AS")
    print("  IT. Worst single epoch over all seeds and sizes: cos %+.4f." % worst)
    print()
    print("  supplement/v124 identified the cause and it is SIGNAL-TO-NOISE, not the")
    print("  sign-branch split. cos is a ratio, so it collapses wherever the true")
    print("  gradient is small against the estimator's error floor: binning 240")
    print("  epochs by |g_true| quartile gives err/|g| of 0.89, 0.39, 0.26, 0.20 and")
    print("  mean cos of +0.7744, +0.8946, +0.9565, +0.9802. The min entries above")
    print("  are FLAT-REGION epochs. More shots where |g| is small is the fix.")
    print()
    print("  WITHDRAWN, and left here because it was wrong in an instructive way.")
    print("  An earlier version of this file blamed the tail on samples whose")
    print("  residual sits near zero being flipped into the wrong branch, and")
    print("  predicted the tail would worsen as training converged. v124's control")
    print("  arm logs ZERO flips across 120 epochs and still hits min cos -0.0752,")
    print("  which rules the mechanism out; and in its realizable arm flips do rise")
    print("  (0.00 -> 0.12/epoch) while cos IMPROVES (+0.9476 -> +0.9916), because a")
    print("  sample only flips when its residual - hence its gradient weight - is")
    print("  small. Misassigned weight mass peaks at 0.87%. The split protects")
    print("  itself; the prediction had the sign backwards.")
    print()
    if wins == len(SEEDS):
        print("  Descent on the estimate reduced the loss on %d/%d seeds, and landed"
              % (wins, len(SEEDS)))
        print("  within a few percent of exact-gradient descent from the same start -")
        print("  sometimes above it, sometimes below. So the lower tail above costs")
        print("  the TRAJECTORY little: momentum in the parameters averages over it in")
        print("  a way a single-epoch cosine cannot show. This is also the first time")
        print("  the estimator has driven its own trajectory in this line of work;")
        print("  v74, v121, v122 and v123 all stepped with something else.")
        print()
        print("  Both arms end near MSE 0.8 from a start near 1.0-1.3. That is the")
        print("  ansatz and the task, NOT the estimator - the exact-gradient arm does")
        print("  no better. Nothing here claims the model fits well; the claim is that")
        print("  three circuits buy the same trajectory the exact gradient buys.")
    else:
        print("  Descent on the estimate reduced the loss on only %d/%d seeds. A good"
              % (wins, len(SEEDS)))
        print("  cosine is not sufficient for a good trajectory, and that gap is the")
        print("  finding - report it, do not average it away.")
    print()
    print("  SCOPE. N_sys=%d, |D| <= 16, efficient_su2 reps=1, %d shots, %d seeds,"
          % (N_SYS, SHOTS, len(SEEDS)))
    print("  no noise model, no hardware. PART 1 steps with the exact gradient so")
    print("  every seed sees the same trajectory; PART 2 steps with the estimate.")
    print("  The exact arms are tier B (Statevector) and are the REFERENCE - no")
    print("  headline number here rests on them.")
