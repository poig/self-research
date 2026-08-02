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

That identity is exact for the quantity being differenced. What the ancilla
actually reports is not <H> but a distorted version of it, and the distortion is
state-dependent, so g_i comes back with the right direction and a per-block scale
error of up to 2x - see THE GRADIENT SCALE BIAS.

num_ancillas=1   Hadamard test. One +-1 bit per shot, Var(<H>) ~ 1/(tau^2 S).
                 tau = tau_scale/range(H) shrinks as O(1/N), so this is
                 O(N^2/S) - the reason it needed a 16x shot budget to match V2.
                 Also carries an irreducible sin() bias: the readout is
                 -<sin(H tau)>, not -tau<H>.
num_ancillas=k   QPE. Each shot decodes a sampled EIGENVALUE: Var(H)/S, no tau
                 penalty, O(N/S). No sin() term, so unlike the k=1 path it is
                 asymptotically unbiased. Measured at Heisenberg N=6 on identical shots
                 and circuits: 0.94 Hartree better, 10x lower variance, 3.1x
                 depth. Coherent time buys precision at the Heisenberg limit
                 (1/T) where shots only manage the standard quantum limit
                 (1/sqrt S); measured exchange rate 3.1x depth <-> 16x shots.

Resolution is 2*margin*||H0||/2^k and must clear the signal 2R*d_iE. At k=4, H2
sits at resolution 0.204 against signal ~0.24 - and H2 is where V3-QPE places
last. k should scale with ||H0||/(R d_iE) rather than being fixed. UNIMPLEMENTED,
AND THE ARGUMENT IS NOT SUPPORTED BY THE ONE DIRECT TEST OF IT: sweeping k from 3
to 7 moved the bin width 16x, straddling the signal, and the gradient error did
not budge at any block. See OPEN/adaptive k. The resolution story is intuitive and
may still matter for H2 specifically, but do not treat it as established.
Note also qpe_margin was tuned for <H> alone; the SECOND moment is far more
margin-sensitive - see the INTERIOR excited states pivot.

═══ RESULT: 8 problems, 5 trials, every method tuned to an interior optimum ═══

*** STALE. Measured under a benchmark harness with three fairness defects, all
*** three of which penalised V3. benchmark.py is now fixed; THE TABLE BELOW HAS
*** NOT BEEN RE-MEASURED. See BENCHMARK FAIRNESS AUDIT. Do not quote these
*** numbers as the fair comparison - the direction of every correction favours
*** V3, so treat them as a floor, not an estimate.

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
with it, so the ratio widens as problems grow (7x at N=8). NOTE the circuits
column above UNDERCOUNTS every baseline by their Pauli-group factor G - 3 on all
five Heisenberg/Ising problems, 1 on MaxCut, 2-3 on the molecules - so the true
ratios on Heisenberg are ~3x wider than shown. The 3.2x-fewer-SHOTS result is
separate and is measured in IS IT ACTUALLY CHEAPER.

Read E_final, not E_best. E_best is min-over-epochs of noisy evaluations and is
biased low by ~0.02 at 20 epochs - QNG's -0.0137 on MaxCut N=6 is below that
problem's exact 0.0 purely from this.

═══ A FIX VALIDATED ON ONE PROBLEM AND SHIPPED FOR ALL ═══

Worth recording as a process failure as much as a bug, because the bug was
invisible from the problem it was validated on.

The Suzuki-2 sensing change was measured on HEISENBERG N=4, found to cut gradient
bias 2.8x for 11% depth, and applied to every problem. It sets reps = 2^a/2, which
fixes the REP COUNT and lets the Trotter STEP float:

    step = 2^a tau0 / (2^a / 2) = 2 tau0,   tau0 = pi / (margin ||H0||)

so the step scales as 1/||H0||. Heisenberg (||H0||=6.46) gets a comfortable 0.49.
H2 (||H0||=0.827) gets 3.8, with the top ancilla evolving to t=15.2 - far outside
any product formula's asymptotic regime. Validating on Heisenberg could never have
exposed this, because Heisenberg sits at the safe end of exactly the axis the rule
fails on.

Measured gradient bias (supplement/results/v13_repschedule.log):

                        H2      Heis N=4   MaxCut N=4
    lie  2^a        0.6627      0.0673      0.0388     (pre-fix)
    suz2 2^a/2      0.3436      0.0437      0.0336     (shipped)
    suz2 2^a        0.0541      0.0357      0.0366

THE CHANGE ITSELF WAS RIGHT - suz2 2^a/2 beats the old rule on all three. But it
left H2 at 8x Heisenberg's bias, so part of "V3 places last on H2" was this, not
the algorithm. FIX: reps = max(1, 2^a/2, ceil(t/2)), taking the max of the rep-
count and step-bound criteria. By arithmetic that gives H2 reps 1,2,4,8 and leaves
Heisenberg and MaxCut at 1,1,2,4, so depth is spent only where the evolution is
long. Verified: H2 bias 0.3436 -> 0.0758 at depth 239, the other two unchanged at
identical depth.

TWO METRIC TRAPS FOUND WHILE CHASING THIS:
  * Operator(PauliEvolutionGate(...)) returns the gate's EXACT matrix exponential
    and ignores the synthesis entirely - it reported 0.000e+00 difference between
    every schedule. Transpile to a basis first to see what the circuit does.
  * Even with real synthesis, OPERATOR-NORM error is the wrong metric and prefers
    the pre-fix rule. The gradient consumes DIFFERENCES of energies across
    vertices, so Trotter error that is uniform over the hypercube cancels before
    reaching the estimator. Always measure gradient bias, not operator fidelity.

GENERAL RULE THIS IMPLIES: any hyperparameter tied to tau0 is implicitly tied to
1/||H0||, so it must be validated across a RANGE of ||H0||, not across a range of
qubit counts. The suite's ||H0|| spans 0.83 (H2) to 21.2 (Heisenberg N=8) - a 25x
range - and tuning on one end of it proves nothing about the other.

═══ THEORY: WHAT IS PROVABLE, AND WHAT IS NOT ═══

The claim this project can defend is about the ESTIMATOR, not about convergence.
That distinction matters because the strong version - "it finds the ground state
of any system" - is not merely unproven, it is provably FALSE for this estimator
class: Arrasmith et al. (Quantum 5, 558) show cost-function-difference estimators
are exponentially suppressed on barren plateaus. No VQE- or QAOA-family method has
a global convergence proof either. So the deliverable is a theorem about the
estimator plus a stated REGIME OF VALIDITY, and chasing universality is chasing
something that does not exist.

T1  IDENTITY. The per-bit marginal is exactly twice the degree-1 Walsh-Fourier
    coefficient of the energy restricted to the +-R hypercube:

        <E | x_i=1> - <E | x_i=0>  =  2 * Ehat({i}),
        Ehat({i}) = E_sigma[ E(theta_c + R*sigma) * sigma_i ],  sigma in {-1,+1}^n

    PROVEN, and verified numerically to 1.665e-16 - machine precision, an
    identity and not an approximation. So sense_gradient returns the best
    degree-1 L^2 fit to the energy over the hypercube. This single fact explains
    every measured property below.

T2  UNBIASEDNESS AT ANY SHOT COUNT. Ehat({i}) is an expectation, so its
    estimator is an empirical mean - LINEAR in the measured energies. Therefore
    E[ghat] = g regardless of how many shots land on each vertex, including
    fewer than one. PROVEN in one line, and it is the load-bearing fact of the
    whole design: it is exactly why the marginal survives at large block width
    where argmin, top-m and Boltzmann decoders fail, since those are NONLINEAR
    and must resolve each vertex's energy before applying the nonlinearity, at
    S/2^n shots per vertex. Also why one circuit yields all M components: every
    coefficient is an expectation over the SAME samples.

T3  APPROXIMATION. Ehat({i})/R = d_iE + O(R^2 * sup|d^3 E|). The O(R^2) cross
    terms cancel by the odd symmetry of sigma_i under the symmetric +-R
    perturbation. DERIVED; explicit constants NOT yet written. Measured at one
    point (Heisenberg N=4, R=0.6): cos(Ehat, grad E) = 0.9973 with
    ||Ehat/R|| / ||grad E|| = 0.795 - direction preserved, magnitude attenuated,
    which is the same signature seen throughout this project.

T4  VARIANCE. Var(g_i) = (1/S) * [a + b*(n-1)], with a the measurement term and
    b the cross-coordinate landscape term. DERIVED AND MEASURED. For the
    Hadamard path b = 0 structurally, because a shot is a bounded +-1 Bernoulli
    variable whose variance cannot exceed 1 however much the energy varies across
    the hypercube; measured fit b/a = -0.004. The analytic value
    a = 1/(S R^2 tau^2) = 0.0304 against a measured 0.0301, agreeing to 0.9%.
    For QPE b > 0, since a shot returns a decoded energy whose spread does
    include vertex-to-vertex variation.

T5  COST. All M components come from ceil(M/n) circuits at ONE measurement
    setting each, because the energy is read from the phase of exp(-iHt) rather
    than from measuring Pauli terms. Parameter-shift needs 2*M*G circuits for G
    qubit-wise-commuting groups. Counting, hence exact. Measured consequence at
    Heisenberg N=4: 3.2x fewer total shots and 48x fewer circuits at n=8.

T6  THE LOCAL LANDSCAPE IS QUADRATIC, AND THAT BOUNDS THE WHOLE APPROACH.
    Given T1, the natural question is what the REST of the Walsh spectrum holds.
    By Parseval, Var(E) over the hypercube = sum_{S != empty} Ehat(S)^2, so the
    decomposition is exact and classically computable. Measured over 3 centres x
    every block (supplement/results/v5_walsh.log):

        problem            R     deg1    deg2    deg3    deg4+
        Heisenberg N=4    0.6   0.757   0.239   0.004   0.000
        Heisenberg N=4    0.3   0.894   0.106   0.000   0.000
        Heisenberg N=6    0.6   0.856   0.142   0.001   0.000
        Heisenberg N=6    0.3   0.951   0.049   0.000   0.000
        H2                0.6   0.838   0.162       -       -

    DEGREE 1 + DEGREE 2 IS 99.6%+ OF THE LANDSCAPE, and degree-3 and above are
    numerically zero. The energy on the +-R hypercube is essentially EXACTLY
    quadratic in the +-1 bits. Two consequences:
      * sense_gradient reads degree-1 only, so it sees 76-86% of the local
        structure at R=0.6 and leaves 14-24% unused. Shrinking R linearises the
        problem (deg2 0.239 -> 0.106 at N=4) but attenuates the signal, which is
        the same trade as the block-width optimum.
      * A degree-1 + degree-2 model is a COMPLETE local model. No higher-order
        decoder can extract more, so this is the ceiling of the sensing primitive,
        not just the current implementation's limit. Useful to know where the top
        is.

    Per block the average understates it. At Heisenberg N=4, R=0.6:

        blk   |deg1|   |deg2|   deg2 SNR at 8192 shots
         0    1.5261   0.3586   3.24
         1    0.1505   0.0088   0.51
         2    0.2997   0.4145   4.34
         3    0.2098   0.5219   3.70

    On blocks 2 and 3 the DEGREE-2 WEIGHT EXCEEDS DEGREE-1 - the walk steers on
    the minority component there - and those are the same small-gradient blocks
    that carried the worst scale bias. Degree-2 coefficients cost zero extra
    circuits (T2: every Walsh coefficient is an expectation over the same shots)
    and measure at SNR 3-4. blk1's 0.51 is not a problem: its degree-2 is
    genuinely ~0, so there is nothing to miss.

T7  THE SENSING IS DEGREE-2 COMPLETE; THE WALK IS DEGREE-1 LIMITED. T6 says the
    degree-2 structure is there and measurable, so the obvious upgrade is to give
    the walk a quadratic drift: add controlled-RZZ(Ehat({i,j})*gamma) alongside
    the existing CRZ. Decomposes without a Toffoli, since
    controlled-(V W V^dag) = V * controlled-W * V^dag with V = CX, so it is
    CX ; CRZ ; CX with the CX gates uncontrolled.
    BUILT AND TESTED. IT DOES NOT HELP (supplement/results/v5_deg2walk.log), sweeping a gain
    with 0 reproducing V3 exactly:

        gain    N=4 E_final   vs 0        N=6 E_final   vs 0
        0.0        -5.8985      -            -8.5061      -
        0.25       -5.9044   -0.006 (0.1s)   -8.6540   -0.148 (0.4s)
        0.5        -5.7706   +0.128 (0.9s)   -8.5803   -0.074 (0.2s)
        1.0        -5.8076   +0.091 (1.0s)   -8.0340   +0.472 (1.0s)
        2.0        -5.6529   +0.246 (1.8s)   -6.6121   +1.894 (2.5s)

    Monotonic degradation past gain 0.25, and gain=1.0 - where the phase IS the
    correct quadratic energy model, both Walsh degrees being energies that enter
    with equal weight - hurts at both sizes. So this is not a normalisation miss.
    The std column is the tell: 0.0595 -> 0.278 at N=4, 0.588 -> 1.396 at N=6.
    Variance grows with gain. Degree-2 measures at SNR 3-4 against degree-1's 21,
    so feeding it in injects estimation noise for no signal - the SAME mechanism
    as the QFIM result, and the third instance in this project of shot-estimated
    extra structure degrading the update.
    TWO EXPLANATIONS, NOT SEPARATED by this run:
      * the mixer cannot use it. CRX is a PRODUCT of independent single-qubit
        rotations, so it structurally cannot convert PAIRWISE phase correlations
        into correlated population motion. If so, a degree-2 drift is unusable
        without a correlated mixer (controlled-XX/XY), and the two upgrades only
        work together. Predicts a correlated mixer fixes it.
      * pure noise injection. Predicts more shots fixes it.
    CAVEAT on that log: its depth column is uninformative. max_circuit_depth is
    the max over ALL circuits including the QPE sensing circuit, which dominates
    at 536/920, so the added CRZZ cost was never actually measured.

    THIS CLOSES A LOOP. The Boltzmann and top-m decoders work on RAW VERTEX
    ENERGIES, so they implicitly use degree-1 AND degree-2 - the whole landscape -
    while the walk uses degree-1 only. Yet they tie (IS THE WALK NECESSARY?). So
    the extra landscape structure the classical decode exploits is worth about as
    much as the coherent amplification the walk provides. Those two results were
    separately puzzling and are jointly consistent.

T8  LOCALITY BOUNDS THE WALSH DEGREE - so T6 is a theorem, not an accident, and
    the entangler is what creates the structure the estimator misses.
    A block is single-qubit rotations on DISTINCT qubits. A k-local Hamiltonian
    term is supported on k qubits, so its expectation depends on at most k of the
    block's sigma variables, and any function of k binary variables has Walsh
    degree <= k. Summing over terms: for a k-local effective observable the energy
    on the +-R hypercube has degree <= k EXACTLY. The effective observable for a
    block is H conjugated by everything AFTER it, so single-qubit gates preserve
    the bound and entanglers break it - a CX spreads 2-local to 4-local.
    Sharp prediction, and it is not "degree-3 is small": for efficient_su2(reps=1)
    decomposing to RY, RZ, CX, RY, RZ, blocks 2 and 3 have nothing entangling
    after them and must be degree <= 2 to MACHINE PRECISION, while blocks 0 and 1
    sit before the CX and may carry degree 3+.
    CONFIRMED (supplement/results/v5_locality.log), Heisenberg N=4 and N=6, two
    centres each - degree-3 weight:

        blk 0 (CX after)   4.9e-04   3.4e-03   3.7e-03   1.4e-03
        blk 1 (CX after)   1.1e-05   2.8e-08   2.1e-06   1.7e-07
        blk 2 (none after) 7.9e-32   6.5e-32   1.9e-32   2.5e-32
        blk 3 (none after) 8.9e-32   1.6e-32   5.5e-32   6.4e-32

    1e-32 is exact zero in double precision. Consequences:
      * How much landscape the degree-1 estimator misses is knowable A PRIORI
        from H's locality. For 2-local H it is exactly the degree-2 part and
        nothing else, so degree-1 + degree-2 is a COMPLETE model for the late
        blocks - not an empirical near-completeness.
      * WARNING FOR THE ANSATZ PLAN. OPEN/ansatz recommends raising reps as the
        largest available gain, on the grounds that it lifts the -6.1231 ceiling
        and that V3's cost is flat in M. T8 says more entangling layers put MORE
        weight at degree 3+, which the degree-1 marginal cannot see at all. So
        raising reps may improve the reachable minimum while degrading the
        estimator's coverage of the landscape. Those pull in opposite directions
        and the net is UNTESTED. Measure the degree spectrum at reps=2,3 before
        assuming the ceiling gain is free.
      * This is the honest home for the DLA question. Bounded dynamical Lie
        algebra is not a way to replace the ancilla: a polynomially-sized DLA is
        exactly the condition under which expectation values are CLASSICALLY
        computable by Lie-algebraic simulation, so the regime where the sensing
        primitive earns anything is the regime where the DLA is exponentially
        large. What the DLA/locality structure does provide is this bound - a
        limit-setting tool, telling you the ceiling of any decoder and when the
        problem is classically easy, not a cheaper readout.

T9  *** PARTLY RETRACTED - see T9b. The conclusion below was reached at LOCAL
    radii (R <= 1.2) using MINIMA COUNTS. At wide radius, with best-point-found as
    the metric, extra bits do help and this entry's verdict is too strong. ***

    bits_per_param > 1 DOES NOT BUY A MULTI-BASIN SEEDING PHASE. The proposal:
    encode each parameter in b qubits for 2^b levels, use the finer grid to find
    basins, then drop to b=1 for descent. Checked classically first
    (supplement/results/v7_bitsperparam.log) - no circuit needed to ask whether the
    structure exists.
    THE COUNTING ARGUMENT FIRST. More bits refine the GRID, not the SAMPLING. With
    b bits over n parameters there are 2^(b n) vertices and still only S shots, so
    the same measurements are spread over a finer lattice. By T2 the linear
    gradient read does not care - it is unbiased however few shots land per vertex
    - but basin detection is NONLINEAR, needs the vertex-energy distribution, and
    therefore gets strictly WORSE as the grid refines. n=4,b=2 gives 256 vertices
    at 32 shots each and works; n=8,b=2 gives 65536 at 0.125 each and does not.
    MEASURED, on a 4-level grid over the same range:

        Heisenberg N=4  R=0.6 blk0   minima b=1: 1   b=2: 1
                        R=1.2 blk0   minima b=1: 2   b=2: 1
        H2              R=0.6 blk0   minima b=1: 1   b=2: 2
                        R=1.2 blk1   minima b=1: 1   b=2: 2

    One to two minima either way and the direction is MIXED, so a block's local
    hypercube holds essentially no multi-basin structure for a finer grid to find.
    Note the Heisenberg R=1.2 row: the COARSE grid reports a minimum the fine grid
    does not, i.e. coarse basin counts are unreliable in both directions.
    THREE REASONS THE SEEDING IDEA FAILS, together:
      * the walk ALREADY produces interior updates - _decode_walk returns a
        weighted mean of sampled corners, which is an interior point - so finer
        positioning is largely redundant with the averaging decode;
      * real multi-basin structure lives in the GLOBAL landscape across all M
        parameters, not inside one block's slice, and the global hypercube is
        exactly where the shot wall makes b>1 unaffordable;
      * coarse grids invent spurious minima, so small-block basin detection is
        unreliable at b=1 too.
    CAVEAT ON THAT LOG: its "pred err %" column (35-175%) is ill-posed for this
    question. The corner model is exact ON the corners by construction, and the
    true energy is trigonometric in theta rather than multilinear, so failing to
    extrapolate inward is expected and says nothing - the walk only ever visits
    corners. Read the minima columns, not that one.

T10 THE PER-GRADIENT ADVANTAGE GROWS WITH M - the optimal block width scales
    with the problem, so circuits per gradient stay constant.
    The one quantity that decides large-M viability is how the smeared signal
    decays with block width, because Var is flat in n (T4, b/a = -0.004) so
    cost ~ ceil(M/n) * Var / ||g_sm(n)||^2 hinges entirely on the numerator.
    Computed classically, no circuits (supplement/results/v8_attenuation.log),
    ratio = ||g_smeared|| / ||g_exact|| over the same coordinates, R=0.6:

        N=4  (M=16)  n=1..16   ratio 0.941 -> 0.433
        N=6  (M=24)  n=1..24   ratio 0.941 -> 0.442
        N=8  (M=32)  n=1..32   ratio 0.941 -> 0.450

    EXPONENTIAL beats polynomial at all three sizes (residuals 0.088 vs 0.272,
    0.106 vs 0.219, 0.127 vs 0.364), so ratio ~ exp(-cR^2 n) and

        n* = 1/(2 c R^2)                 cR^2      n*    n*/M   M/n*
        N=4                             0.0494   10.1    0.63   1.58
        N=6                             0.0308   16.3    0.68   1.48
        N=8                             0.0239   20.9    0.65   1.53

    THE DECAY RATE FALLS AS 1/N: cR^2 * N = 0.198, 0.185, 0.191 across the three.
    That is what locality predicts - coordinate i is attenuated only by
    coordinates sharing Hamiltonian terms with it, a fixed-size neighbourhood that
    is a SHRINKING FRACTION of the system as N grows. Consequently

        n* ~ 0.65 M   ->   circuits per gradient = M/n* ~ 1.5, CONSTANT in M
        advantage over parameter-shift's 2MG circuits = 2 G n* ~ 1.3 G M, GROWING

    OUT-OF-SAMPLE CHECK, which is why this is believable: n* was predicted at
    0.65*32 = 20.8 for N=8 from the N=4 and N=6 points alone, before running it.
    Measured 20.9. And the formula 2*G*n* reproduces the independently measured
    48x circuit advantage exactly at n=8, G=3.

    A CORRECTION WORTH RECORDING. Fitting only N=4 gave a clean exponential
    (residual 0.088 vs 0.272) and the conclusion "n* is constant, the advantage is
    a fixed factor". That was wrong. Fitting a decay law at ONE size tells you the
    law, not how its PARAMETERS scale with size, and those are different questions.
    Three sizes were needed to see cR^2 ~ 1/N.

    DENSE-COUPLING STRESS TEST, and the mechanism above is WRONG
    (supplement/results/v8b_dense.log). The locality story predicted that an
    all-to-all Hamiltonian would have a neighbourhood spanning the whole system,
    so cR^2 would stay CONSTANT in N, n* would saturate and the advantage would
    collapse to a fixed factor. Tested against the suite's own transverse-field SK
    spin glass (sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i, every pair), same ansatz:

        family                      cR^2*N spread   cR^2 spread   n*/M
        Heisenberg  (degree 2)          2.2%          30.8%       0.63 0.67 0.65
        Frustrated  (all-to-all)        9.6%          20.1%       0.89 0.71 0.74

    cR^2 ~ 1/N holds for the DENSE Hamiltonian too. The prediction failed and the
    result is better than predicted: the growing advantage is not restricted to
    local Hamiltonians.
    THE CORRECTED MECHANISM. It was never coupling degree. Smearing a coordinate
    attenuates only those Hamiltonian terms whose PARTNER coordinate is also
    smeared, and the fraction of any coordinate's partners that are active is
    n/M - independent of how many partners it has. So cR^2 ~ 1/M follows for
    sparse and dense coupling alike, and "locality dividend" was the wrong frame.
    BUT THE DENSE FIT IS NOTICEABLY NOISIER: 9.6% against 2.2%, and n*/M drifts
    0.89 -> 0.71 -> 0.74 instead of sitting flat like Heisenberg's 0.63/0.67/0.65.
    That drift could be slow saturation invisible at three points. Read "grows
    with M" as SOLID for local Hamiltonians and SUGGESTIVE for dense ones.
    Also note the frustrated model is a rugged spin glass - exactly where a local
    optimiser with no global-search mechanism should struggle no matter how cheap
    its gradients are. Cheap gradients are not the binding constraint there.

    THREE REMAINING LIMITS:
      * The fits degrade with size (residual 0.088 -> 0.106 -> 0.127) and N=8 has
        a non-monotonic point at n=2. 400 smearing samples is getting thin.
      * The companion result - that cost at n* is CONSTANT in M - is
        metric-dependent. Cost is for equal RELATIVE precision on the gradient
        VECTOR, and ||g|| grows like sqrt(M), so that requirement weakens per
        component as M grows. Do not read it as per-coefficient precision being
        free at scale.
      * This is the cost of ONE gradient, not of the optimisation. How many steps
        are needed, and whether the gradient is resolvable at all at large M, is
        where barren plateaus live - and that remains proven AGAINST difference
        estimators. A growing per-gradient saving does not touch that wall.

T9b AT WIDE RADIUS, EXTRA BITS DO HELP - T9 was closed on the wrong metric at the
    wrong radius. Retested with best-point-found rather than minima counts, and
    out to R = pi/2, the widest useful radius (at R = pi the corners c+pi and c-pi
    coincide for a 2pi-periodic parameter and the encoding degenerates).
    Heisenberg N=4, one block of 4 params, 3 centres
    (supplement/results/v9_globalgrid.log):

        R        b=1 best   b=3 best   deficit   minima b=1 -> b=3
        0.600     -3.6502    -3.7940    0.1438      1.3 -> 1.3
        1.200     -3.0711    -4.2423    1.1713      1.7 -> 2.0
        1.571     -2.3099    -4.3121    2.0022      1.7 -> 3.3

    At R = pi/2 two levels reach -2.31 where eight reach -4.31, a 2.0 Hartree
    gap, and the box becomes genuinely multi-modal (1.7 -> 3.3 minima). Two
    levels cannot represent that. So the coarse-to-fine intuition is right:
    WIDE-RANGE SEARCH NEEDS RESOLUTION THAT +-R CORNERS DO NOT HAVE, and T9's
    dismissal came from testing only R <= 1.2 and counting minima instead of
    measuring the best point reached.
    WHAT STILL BOUNDS IT:
      * The shot wall applies to SEEDING specifically. Finding the best vertex is
        NONLINEAR, so it needs shots >~ vertices, unlike the gradient (T2). b=3 at
        n=4 is 4096 vertices against 8192 shots - workable. n=8 is 16.7M - dead.
        Multi-bit seeding is therefore confined to SMALL blocks, which does not
        compose with the wide blocks where T10's cost advantage lives.
      * The comparison is STATIC GRID vs STATIC GRID, not algorithm vs algorithm.
        The walk does not take an argmin over one grid; it takes a weighted-mean
        step and ITERATES with decaying R over ~20 epochs. That sequential
        refinement may already reach what a single fine grid reaches, spending the
        same shots differently. Nothing here rules that out.
      * Cascading refinement layers WITHOUT resetting the system register is not a
        new mechanism: rotations on one qubit add angles, so
        R(c-R) CR(2R,s1) CR(R,s2) = R(c-R+2R s1+R s2) - the layers compose into
        exactly the multi-bit W-gate. It becomes genuinely different only if the
        layers are separated by MID-CIRCUIT MEASUREMENT and feedforward, which
        buys one-circuit bisection at ~20x the depth, against V3's low-depth
        position, and needs dynamic-circuit noise this project has not studied.
    THE DECIDING EXPERIMENT, not yet run: end to end, multi-bit WIDE seeding
    followed by 1-bit descent, against the existing R-decay schedule, AT EQUAL
    TOTAL SHOTS. Neither T9 nor T9b performed it.
    AND A LIMIT NO ENCODING FIXES: at large M the binding constraint is signal
    MAGNITUDE, not grid resolution. Barren plateaus shrink the gradient
    exponentially, and T2 gives unbiasedness, not precision - if g ~ 2^-M then
    S ~ 2^2M shots are needed however finely the parameters are gridded. A finer
    grid on a flat landscape buys nothing.

T9c A FREE IN-SITU STEP-SIZE DIAGNOSTIC - the one actionable thing multi-bit
    encoding does buy. First, a register clarification the proposal it came from
    got wrong: ANCILLAS RESOLVE THE ENERGY (k QPE bits), PARAM QUBITS RESOLVE THE
    PARAMETERS. Adding ancillas cannot refine theta; they are separate axes.
    But with a 2-bit encoding theta_i = c_i + R s_i0 + (R/2) s_i1, BOTH per-bit
    degree-1 Walsh coefficients come from the SAME shot record by T2:

        Ehat({i0}) ~ R   * d_iE      Ehat({i1}) ~ (R/2) * d_iE

    so their ratio is exactly 2 where the landscape is linear in coordinate i.
    Exact enumeration of the 2^8 grid, Heisenberg N=4, 3 centres
    (supplement/results/v9b_multiscale.log):

        R      ratio w0/w1    cos(w,g)    |w1/(R/2)| / |g|
        0.20      2.0203       0.99981         0.9530
        0.40      2.0857       0.99700         0.8253
        0.60      2.2116       0.98509         0.6505   <- shipping R
        1.00      2.8508       0.88203         0.3027
        1.50     15.1368       0.36572         0.0312

    THE RATIO TRACKS DIRECTION QUALITY, AND IS MEASURABLE WHERE QUALITY IS NOT.
    cos(w, grad E) cannot be computed in a real run - grad E is unknown - but the
    ratio can, from shots already taken. It reads 2.02 when cos is 0.9998 and 15.1
    when cos has collapsed to 0.366. At the shipping R=0.6 it reads 2.21, so the
    current radius is mildly into the nonlinear regime.
    A PREDICTION THAT FAILED, with the reason: the fine bit does NOT give a better
    gradient DIRECTION. cos(w0,g) and cos(w1,g) are identical to five decimals at
    every R, because the two bit-levels stay parallel as vectors. The cubic term
    (R s0 + (R/2) s1)^3 contributes 1.75 R^3 s0 + 1.625 R^3 s1, a ratio of 1.077
    against the linear term's 2.0, and that correction has the same functional form
    for every coordinate - so nonlinearity rotates both estimates identically. The
    fine bit buys a SMALLER MAGNITUDE, not a better direction. There is no
    bias-variance dial here, only a diagnostic.
    CHEAPER THAN MULTI-BIT. The ratio does not require doubling the param register.
    Two single-bit sensing circuits at R and R/2 give the same number for 2x
    circuits and NO extra width - and by T10 circuits are cheap (~1.5 per gradient)
    while width is the scarce hardware resource. Cheaper still, consecutive epochs
    already run at 0.6*0.9^e, so comparing successive gradients gives similar
    information from circuits already being run, confounded only by the centre
    moving between them.
    THE PAYOFF: R = 0.6*0.9^epoch is flagged elsewhere in these docs as an
    arbitrary schedule inherited from nisq_v2's __main__, and EXTENSIONS proposes
    replacing it with quantum counting. This is cheaper and more direct - SHRINK R
    UNTIL THE RATIO APPROACHES 2. Untested end to end.

T11 THE WALK IS COHERENT LOCALLY AND NOT GLOBALLY - measured, and it closes the
    HSP question quantitatively rather than by argument.
    An earlier claim in these notes that QLTO "computes the Walsh transform by
    sampling" was too crude and is corrected here. The SENSING path does
    reconstruct Walsh coefficients classically from measured bitstrings, but the
    WALK is genuine quantum interference: CRX mixes different |x> within the anc=1
    branch and the final H interferes the branches.
    The limit is that after CRX moves |x> -> |x'>, the system register still holds
    |psi_x> tied to the ORIGINAL x, so tracing it out weights every cross term by
    the overlap:

        P(x') = sum_{x,x''} A*(x->x') A(x''->x') <psi_x|psi_x''>

    Coherence between two vertices survives only in proportion to
    |<psi_x|psi_x''>|. Measured Gram matrix, Heisenberg N=4, one block, 3 centres
    (supplement/results/v11_coherence.log):

        R        all pairs   adjacent   antipodal
        0.100      0.9894     0.9950      0.9802
        0.300      0.9079     0.9553      0.8330
        0.600      0.6734     0.8253      0.4640   <- default R
        1.000      0.3086     0.5403      0.0852
        1.571      0.0000     0.0000      0.0000

    At the default R the adjacent overlap is 0.83, so the walk's local mixing IS
    doing real interference - the first direct evidence of that. Antipodal overlap
    is already 0.46 and collapses to 0.085 by R=1.0. At R=pi/2 everything is
    exactly zero, because a +-pi/2 offset rotates each qubit to an orthogonal
    state.
    THE BIND THAT CLOSES HSP. A Simon-style extraction needs coherence between x
    and x XOR s, which for high-weight s is ANTIPODAL. Antipodal coherence needs
    small R (0.98 at R=0.1), but small R means the hypercube spans almost no
    parameter range - nothing is being searched. COHERENCE AND RANGE PULL IN
    OPPOSITE DIRECTIONS and cross well below where the algorithm operates.
    Designing an ansatz with psi_x = psi_{x XOR s} would fix the overlap by
    construction at any R, but requires knowing s, which is the answer. Circular,
    exactly like "learn a Hamiltonian with the right period".
    The non-circular remnant: an ansatz's OWN gauge redundancy creates natural
    collisions - distinct theta giving identical states, which is what QFIM
    rank-deficiency measures - and interference could detect those directions.
    Real, but modest: by T10 circuits per gradient are ~1.5 regardless of M, so
    shrinking M buys fewer optimisation STEPS, not fewer circuits.
    AN UNRESOLVED TENSION, stated because both halves are now measured: the walk's
    interference is real and substantial (0.83 adjacent), and yet IS THE WALK
    NECESSARY? found a purely classical Boltzmann decode that TIES it. The
    coherence exists; nothing has yet shown it paying. That is the sharpest open
    question about the walk.

REGIME OF VALIDITY - state this, do not claim more:
  * works where the landscape carries non-vanishing degree-1 Walsh weight at
    scale R. Fails on barren plateaus, BY PROOF, not by measurement.
  * the smearing attenuates signal as n grows (||g|| 2.88 -> 1.61 from n=1 to
    n=16) while noise stays flat, so there is an INTERIOR optimal block width -
    measured at n ~ M/2, not at global.
  * ANSATZ CLASS, and this is the limit most easily overlooked. The construction
    assumes every parameter sits on a SINGLE-QUBIT rotation, so |0> -> c-R and
    |1> -> c+R is one base rotation plus one controlled rotation. build_w_gate
    handles exactly that plus CXGate and silently ignores anything else. An
    ansatz with multi-qubit parameterised generators - HVA's exp(-i theta X_i X_j)
    - needs controlled multi-qubit rotations, changing both depth and the
    commuting-block detection. HVA underperforming here (p=4 -> -5.146) is that
    boundary showing up empirically. So the theory holds for single-qubit-rotation
    ansaetze with local H, and "all systems" is not established.

STILL OWED for a publishable theory section: T3's explicit constants; the Trotter
and sin() bias bounds (standard product-formula bounds plus the tau^2<H^3>/6 term
derived in THE GRADIENT SCALE BIAS); and larger-M empirics, since the flat-in-M
cost advantage only becomes visible where M is large.

═══ BENCHMARK FAIRNESS AUDIT ═══

Audited after the cost study turned up the same class of bug twice in one day
(supplement/results/audit_benchmark.log). Three defects, ALL THREE penalising V3, so every
correction moves the RESULT table in V3's favour.

1. THE SHOT BUDGET WAS NOT ENFORCED ON THE BASELINES. benchmark.py set
   PRECISION = 1/sqrt(8192) and passed it to
   StatevectorEstimator(default_precision=...), whose contract is: return the
   EXACT expectation plus Gaussian noise of standard deviation `precision`. It
   never samples, so it is blind to Var(H) AND to the number of measurement
   settings. Confirmed: std/precision = 0.95, 1.02, 1.02 across precisions
   spanning 18x while Var(H) = 12.03. The old comment - "precision is the
   standard error, so 1/sqrt(SHOTS) is the matching setting" - is true only when
   Var(H) = 1.
   A device reaching standard error sigma needs G*sum_g Var(H_g)/sigma^2 shots,
   so the effective budget handed to the baselines was:

     problem          G   Var(H)   eff shots   vs the 8192 claimed
     H2               2    0.185        3059   0.37x
     LiH              3    0.159        3995   0.49x
     MaxCut N=4       1    0.766        6278   0.77x
     MaxCut N=6       1    3.977       32580      4x
     Heisenberg N=4   3    8.315      191331     23x
     Heisenberg N=6   3   14.110      351902     43x
     Heisenberg N=8   3   21.241      464701     57x

   PROBLEM-DEPENDENT, because Var(H) is large for unit-coefficient spin sums and
   tiny for the molecules. Cross-referenced against who won what, the pattern is
   damning: V3's two Heisenberg losses are the 23x (to V2) and 57x (to QNG)
   problems, its clear win at N=6 happened DESPITE 43x, and AdamW's H2/LiH wins
   sit at 0.37x/0.49x so they are clean or understated. V3 was never subsidised -
   its sensing calls backend.run(qc, shots=...) for real. V2 WAS, through
   BaseEstimator(precision=PRECISION), so V2-vs-V3 is the comparison most at risk.
   FIXED: make_estimator now returns BackendEstimatorV2 on AerSimulator, which
   samples for real. Same nominal budget, measured std 0.03794 against the old
   0.01105 on Heisenberg N=4 - and 0.00384 on H2, where real sampling is BETTER
   than the fixed noise was.

2. BASELINE CIRCUIT COUNTS OMITTED THE MEASUREMENT SETTINGS. Every baseline
   billed one expectation value as one circuit (AdamW: 2*len(params); SPSA: +=2;
   QNG: += 2*n_params + n_layers), but a Pauli sum needs one circuit per
   qubit-wise-commuting group. V3 genuinely needs ONE whatever H is, because its
   energy comes from the phase of exp(-iHt) rather than from measuring Pauli
   terms. G=3 on every Heisenberg problem, so the headline cost gap was
   understated 3x there; G=1 on MaxCut, so those were already right.
   FIXED: circuit_multiplier stamped per optimizer, circuits = nefv * G.
   Measured effect at Heisenberg N=4, 3 epochs: AdamW 96 -> 288 against V3's 27.

3. ENERGIES WERE LOGGED THROUGH DIFFERENT ESTIMATORS. Reporting used
   `opt.estimator`: exact for V3, fixed-noise for V2, shot-noisy for the
   baselines once (1) was fixed. E_best is a min over epochs, so reporting noise
   biases it LOW - a free advantage on the exact column the table ranks by, handed
   to whichever method logged most noisily.
   FIXED: one exact REPORT_ESTIMATOR for every method. Logging is not part of any
   optimizer's cost, so it should be noiseless and identical.

WHAT THIS DOES NOT FIX. The ansatz ceiling still binds - reps=1 caps at -6.1231
against reps=3's -6.4641 at N=4, and every method clusters within 1-2% of it, so
the suite cannot resolve optimisers on accuracy no matter how fair the shots are.
Re-running will sharpen the COST claim, which was already the durable one, and is
unlikely to change the accuracy verdict.

COST OF RE-RUNNING: the new estimator genuinely simulates G circuits of 8192
shots per expectation value instead of adding a number to an exact result, so the
suite is substantially slower than before. Budget accordingly.

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
                results - are both under-resolved at k=4. Caveat: MaxCut N=4 is
                also nominally under-resolved and V3 won it, so the rule is
                incomplete. DEMOTED from "cheapest gain available": the k sweep
                run for the bias study (supplement/results/anomaly_e.log) took the bin
                width from 3.232 down to 0.202 - a factor of 16, straddling the
                signal - and the gradient error did not move at any block. So at
                Heisenberg N=4, k=3 is already sufficient and extra ancillas buy
                nothing. That is one problem, and H2's marginal case is untested,
                but the resolution argument is no longer supported by the only
                direct measurement of it.
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
||g|| magnitude EXPLAINED, and it is a real defect in the shipping settings. See
                THE GRADIENT SCALE BIAS below. Short version: two opposite-signed
                systematic errors in the sensing evolution, not shot noise. The
                DIRECTION survives both (cosine 0.93-1.00), which is why the walk
                works anyway and why this went unnoticed.
free savings    point-energy is 1 circuit/epoch of logging the optimiser never
                reads. W-dagger REMOVAL NOW TESTED (supplement/results/anomaly_wdag.log):
                correct but nearly worthless, and the depth claim was wrong.
                Removing it moves the measured distribution by TVD 0.0103-0.0108
                against a shot-noise floor of 0.0457 (same circuit, two seeds),
                so it is indeed invisible to the marginals, and the decoded
                block moves by ~0.006. But the saving is 0% of DEPTH at every
                k_steps from 1 to 30, and only 19%->6% of GATE COUNT as k grows.
                Depth is untouched because the ancilla is the critical path: it
                takes part in all 2*n*k controlled gates while each param qubit
                sees only 2k, so W-dagger runs entirely inside the slack that
                already existed. Worth taking for the gate count (hardware error
                budget is per gate) but it is not a depth win, and it matters
                least in exactly the long-walk regime V3 actually uses.
schedule        TESTED, and the coupling turns out to be load-bearing - do NOT
                normalise it without retuning. sum_step s = k/2 exactly, so the
                total accumulated angle is LINEAR in k: k_steps is a step-size
                multiplier wearing a resolution costume. Measured |move| grows
                0.306 -> 0.860 as k goes 1 -> 30 (supplement/results/v4_schedule.log).
                Scaling gamma and beta by 2/k makes the total angle
                k-independent and |move| goes flat at ~0.355, exactly as the
                algebra says.
                But comparing the two at fixed dt is the same step-size confound
                that made the raw natural gradient look bad until natural_norm -
                normalising simply steps less far. With dt tuned per schedule
                (supplement/results/v4_schedule2.log):
                    current     best -4.561 (k=15, dt=0.15)
                    normalised  best -4.415 (k=30, dt=1.5)
                The normalisation delivers what it promises - the winning dt is
                stable at 1.5 for every k>=5, and energy converges monotonically
                in k (-4.122, -4.316, -4.393, -4.400, -4.415) instead of the
                current schedule's non-monotonic wander - but it gives up 0.146.
                The accidental k-coupling acts as a free adaptive step size, and
                at the shipping k=15 that is worth more than clean tuning. Keep
                current; revisit only if k ever needs to be tuned per problem.
walk Trotter    TESTED, and the walk is INSENSITIVE to it - reps=1 is already
                right. The walk evolves at t = dt*pi = 0.942 with LieTrotter
                reps=1 against the sensing path's tau=0.106 with reps=2, which by
                t^2/r is ~158x more Trotter error, in the very step that imprints
                each vertex's energy as an ancilla phase. Raising it changes
                nothing: every variant lands within 0.0-0.2 sigma of shipping
                (supplement/results/v4_walk_trotter.log), lie r=8 at +0.013 for 2.4x the
                depth. The drift only needs the ORDERING of vertex energies, and
                a smooth systematic phase shift preserves ordering. Same theme as
                the gradient scale bias: direction survives, magnitude does not
                matter. Do not spend depth here.
drift/mixer     PARTLY TESTED NOW, and the drift half is closed. Ablation
                background: zeroing the gradient costs 4.32 Hartree and RANDOM
                drift is worse than none, so it is the direction that matters.
                CRZ is diagonal in both registers and moves no populations - it
                only writes phases CRX later converts.
                DRIFT: upgrading it from a degree-1 to a degree-2 phase model
                does not help - see T7. The information exists (T6) and is
                measurable, but feeding it in only adds variance.
                MIXER: PROBED, verdict NEITHER OPENED NOR CLOSED
                (supplement/results/v7_mixer.log). Cheap probe before building
                energy-conditioning, same move as checking the degree-2 Walsh
                weight before building CRZZ: shape beta per coordinate using the
                gradient already measured,
                beta_i = beta * (1 + lambda*(1 - |g_i|/max|g|)), lambda=0 being
                exactly the current walk.
                    lambda   N=4 E_final   vs uniform      N=6 E_final   vs
                    -0.5        -6.0281   -0.041 (2.2s)       -9.0191   -0.045 (0.5s)
                     0.0        -5.9870        -              -8.9743        -
                    +0.5        -6.0120   -0.025 (2.0s)       -8.9797   -0.005 (0.0s)
                    +1.0        -6.0275   -0.041 (2.0s)       -9.0293   -0.055 (0.6s)
                    +2.0        -6.0185   -0.032 (1.5s)       -8.9919   -0.018 (0.2s)
                All EIGHT nonzero-lambda cells beat uniform, which is hard to get
                by chance. Three reasons not to believe it yet:
                  * BOTH SIGNS HELP EQUALLY. lambda=-0.5 (less mixing, on the
                    high-gradient coordinates) and lambda=+1.0 (more mixing, on
                    the flat ones) both give -0.041. If the mechanism were
                    "explore the flat directions" the sign would matter. It does
                    not, so whatever is happening is not that - more likely the
                    uniform beta is simply mistuned and any perturbation of the
                    AVERAGE mixing amount helps.
                  * THE EFFECT EQUALS THE BASELINE'S OWN REPRODUCIBILITY.
                    v4_softmin measured this exact lambda=0 configuration at
                    -6.0064 +- 0.0143 on 6 seeds; this run gives
                    -5.9870 +- 0.0098 on 5. Two measurements of the SAME thing
                    differ by 0.019, about half the claimed effect.
                  * This project's 2-sigma-on-few-seeds results have a bad record:
                    top-4 at 1.1 sigma did not replicate, the merged walk at
                    2.5 sigma contradicted itself across sizes.
                TO SETTLE IT: ~20 seeds with the baseline INTERLEAVED in the same
                run rather than compared across runs, which removes exactly the
                cross-run drift confounding it now. Only then is energy-
                conditioned mixing worth building.
                The original note, still standing: the
                leading explanation for T7's failure is that a PRODUCT of
                independent single-qubit CRX rotations cannot convert pairwise
                phase correlations into correlated population motion, so drift
                and mixer may only be upgradable TOGETHER. A controlled-XX or
                XY mixer plus degree-2 drift is the test that separates "the
                mixer cannot use it" from "it is just noise"; more shots at fixed
                gain separates the same two the other way.
                MERGED ROTATION - tested, depth win real, energy effect NOT
                resolved (supplement/results/v5_merge.log). The CRZ-then-CRX product per
                qubit per step can be replaced by ONE tilted-axis controlled
                rotation: RY(-phi); CRZ(theta); RY(phi) with theta=sqrt(a^2+b^2)
                and phi=atan2(b,a) is EXACTLY exp(-i(aZ+bX)/2) - verified to
                4e-16 - and the RY conjugation is uncontrolled, so it is one
                controlled gate instead of two.
                This is NOT a small-angle approximation of the current walk. At
                the angles actually used (a=3.8, b=0.94) the merged operator and
                the original product differ by 0.813 in operator norm, so it is
                DIFFERENT DYNAMICS at lower cost, and a stronger test of how much
                the CRZ/CRX ordering matters than of the merge itself.
                MEASURED: depth -37% at both sizes (162->102 at N=4, 246->156 at
                N=6). Energy is contradictory - N=4 -0.0407 (2.5 sigma BETTER),
                N=6 +0.1760 (0.7 sigma WORSE) - so unresolved, and a 2.5 sigma on
                4 seeds is exactly the pattern that failed to replicate in
                IS THE WALK NECESSARY?. Merged also ran 3.6x noisier at N=6 (std
                0.4636 vs 0.1287). Replicate at more seeds before believing either
                sign.
                CAVEAT on that log's CX column: it reads 21/35 for BOTH arms
                because AerSimulator keeps crz/crx native, so transpile never
                decomposed them and the counter only saw the W-gate's CX. The
                controlled-gate halving is real by construction; the CX figure is
                an artefact. Depth is the trustworthy number.
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

COMPARATOR on the energy register. QPE yields a k-bit ENERGY per vertex, so a
    threshold test (E < t) is available. THE GROVER BRANCH IS NOW ASSESSED AND
    LARGELY CLOSED - see WHY GROVER IS NOT THE NEXT STEP below. What survives:
      * Quantum counting -> adaptive radius. Count the fraction of vertices below
        the current energy: many good ones means R is too small, almost none
        means too large. Replaces R = 0.6*0.9^epoch, an arbitrary schedule
        inherited from nisq_v2's __main__, with a measured quantity. Note the
        same counting is already available classically from the sensing shots at
        every benchmarked size, so build the classical version first.
      * Threshold-conditioned drift, instead of a linear gradient term.

═══ IS IT ACTUALLY CHEAPER? YES, BUT ONLY WITH QPE AND ONLY BATCHED ═══

Measured, not asserted (supplement/results/v4_cost.log, v4_cost2.log). Heisenberg N=4, M=16,
S=8192 per evaluation, 8 repeats, cost index = circuits * Var(g_i) / ||g||^2,
which is the shots needed for equal RELATIVE precision on the descent direction
(angle error is ||dg||/||g||, so each method is normalised by its OWN target -
QLTO estimates the R-smeared gradient, parameter-shift the exact one).

    method                circuits   cost/|g|^2   vs parameter-shift
    parameter-shift (fair)      96   4.7580e-03   1.00
    QPE k=4  n=1                16   7.1473e-03   1.50   worse
    QPE k=4  n=4  SHIPPING       4   3.5637e-03   0.75
    QPE k=4  n=8                 2   1.4744e-03   0.31   best measured
    Hadamard n=8                 2   9.1029e-03   1.91   worse

So at n=8: 3.2x FEWER TOTAL SHOTS and 48x fewer circuits. Not merely a circuit-
count saving - a real information advantage. Three ingredients, and the third
kills the k=1 path:

  LINEARITY. The marginal is linear in the measured energy, so every shot informs
    every coordinate and n components come out of one circuit. This IS the
    advantage: QPE at n=1 is WORSE than parameter-shift (1.50) and the win
    appears only on batching, improving monotonically with n. It is also why the
    marginal survives where argmin/top-m/Boltzmann cannot - a linear estimator is
    unbiased however few shots land on each vertex, while any nonlinear decode
    must first resolve each vertex's energy, costing S/2^n per vertex.
  NO PAULI GROUPING. QPE reads the energy from the phase of exp(-iHt) in ONE
    measurement setting. Parameter-shift needs G qubit-wise-commuting groups per
    energy, hence 2*M*G circuits. G=3 here - and NOTE it is 3 for Heisenberg at
    N=4, 6 and 8 alike, so this family gives a fixed 3x, not a growing one. The
    "hundreds of groups for molecular Hamiltonians" argument is real but is NOT
    demonstrated by this test; LiH is in the suite and would settle it.
  QPE, NOT HADAMARD. The k=1 path LOSES on shots (1.91x) from its 1/tau^2
    variance. This is the sharpest confirmation of the variance argument in
    MECHANISM, and it means the cost claim belongs to the QPE path only.

THE OPTIMUM BLOCK WIDTH IS INTERIOR, NOT GLOBAL. Raw shot cost does fall ~16x
from n=1 to n=M because Var(g_i) is FLAT in n - fitted b/a = -0.004 for Hadamard,
i.e. the landscape term is absent, because a Hadamard shot is a bounded +-1
Bernoulli variable whose variance cannot exceed 1 no matter how much energy
varies across the hypercube. But the SIGNAL attenuates: smearing over more
coordinates flattens the gradient, ||g|| falling 2.88 -> 1.61 from n=1 to n=16.
Noise flat, signal shrinking, so cost/||g||^2 has a minimum in between - measured
for Hadamard at n=8 (9.10e-3) beating n=16 (9.76e-3). Global mode is therefore a
convenience, not the cost endpoint, and the earlier expectation that it would be
~M times cheaper counted the shot saving while ignoring the attenuation.

BASELINE TRAP, worth more than the result. The first pass had parameter-shift 300x
cheaper and concluded QLTO only converts circuits into shots. That was wrong:
Aer's EstimatorV2 with precision=p returns the EXACT expectation plus Gaussian
noise of std p - it does not simulate shots. The tell was that 2*V_ps*S = 0.97 =
p^2 exactly, with no dependence on Var(H) (=12.38 here) and none on grouping,
a ~27x subsidy. Fixed by computing Var(<H>) = (G/S) * sum_g Var(H_g) exactly from
the statevector. Same class of bug as the exact-statevector baselines fixed
earlier in this project: a baseline that silently gets free precision.

═══ WHY HIDDEN-SUBGROUP STRUCTURE IS NOT REACHABLE FROM HERE ═══

The recurring proposal is: plant a Shor-like hidden period, extract it with a QFT
on the param register, and use Hamiltonian learning to construct whatever
operation is needed. Four obstructions, and the first is structural rather than
quantitative.

1. THE W-GATE'S FUNCTION CONTAINS NO HAMILTONIAN. It computes
   x -> psi(theta_x) = U(theta_x)|0>, in which H appears nowhere; H enters only in
   the readout <psi_x|H|psi_x>. Shor's mechanism needs periodicity in the FUNCTION
   WRITTEN INTO THE REGISTER, because collisions f(x) = f(x XOR s) are what erase
   which-path information and leave a clean coset for the QFT. So NO AMOUNT OF
   HAMILTONIAN LEARNING CAN CHANGE THE COLLISION STRUCTURE - it is a property of
   the ansatz alone.

2. THE COLLISIONS THAT DO EXIST ARE CLASSICALLY KNOWN. The only exact ones come
   from ansatz gauge redundancy - two RZ rotations on one qubit with nothing
   between them make only theta_i + theta_j matter, so x and x XOR (e_i + e_j)
   give literally the same state. Simon would find those. But you wrote the
   circuit, so you can read its gauge group off the diagram for free. Quantum
   detection of something visible in the circuit diagram is pointless.
   Note also that with Pauli-generated rotations (P^2 = I, period 2pi) no OTHER
   exact collision is constructible: flipping bit i shifts theta_i by 2R, and
   2R = 2pi means R = pi, at which point ALL 2^n vertices coincide and carry no
   information at all.

3. MAKING THE ANSATZ H-DEPENDENT CLOSES THE LOOP THE WRONG WAY. With an HVA-style
   ansatz the generators come from H, so collisions become relations among those
   generators - which is the DYNAMICAL LIE ALGEBRA. Rich collision structure means
   a SMALL DLA, and a small DLA means the whole thing is classically simulable by
   Lie-algebraic methods. THE CONDITION FOR EXPLOITABLE STRUCTURE AND THE
   CONDITION FOR CLASSICAL SIMULABILITY COINCIDE.

4. THE LEVEL-SET ROUTE FAILS ON RESOLUTION. One could instead ask for periodicity
   in the ENERGY, E_x = E_{x XOR s}, measure it, and collapse onto an iso-energy
   set. That is even constructible - it is 2^n LINEAR constraints on H's
   coefficients, so Hamiltonian learning could plant such a period. But planting
   it requires knowing s (circular), a natural H has no exact landscape degeneracy
   (non-generic), and decisively: QPE reads energies at FINITE RESOLUTION. Shor's
   f(x) is exact integers so the collapse lands on an exact coset; energies are
   continuous and bin-limited, so the collapse lands on a FUZZY level set, and a
   fuzzy coset does not give clean QFT peaks. Making the bins finer than the
   smallest landscape gap costs ancillas exponentially.

See also T11, which measures the coherence bound independently: antipodal overlap
- what a high-weight s would need - is 0.46 at the default R and 0.085 by R=1.0,
while coherence and search range pull in opposite directions.

═══ WHY GROVER IS NOT THE NEXT STEP ═══

THE WALK IS ALREADY A CONTINUOUS AMPLITUDE AMPLIFICATION. Grover alternates a
phase flip on marked states with a reflection about the mean. _execute_walk
alternates CRZ, which writes a gradient-weighted phase on the param register,
with CRX, which mixes it - the same phase-then-diffuse structure, standing to
Grover roughly as QAOA stands to adiabatic search. Two differences, both
deliberate: Grover's phase is a hard +-1 on marked states where the walk's is
continuous in energy, and Grover's diffusion is the global reflection
2|s><s| - I where the walk's is a product of local CRX rotations. So adding
Grover is not adding a missing primitive; it is replacing a soft gradient-
weighted mechanism with a hard thresholded one.

THAT TRADE LOOKS BAD ON THIS PROJECT'S OWN EVIDENCE. The ablation records that
zeroing the gradient costs 4.32 Hartree and that RANDOM drift is worse than no
drift, so the gradient WEIGHTING is what does the work. A hard E<t oracle keeps
only a binary good/bad label and discards that weighting.

AND THE SEARCH IT ACCELERATES IS NOT HARD YET. The QPE sensing circuit measures
the param register and the energy register in the same shot, so its counts are
already a table of (vertex -> energy). With S shots over 2^n vertices each vertex
is sampled S/2^n times: at n=4 and S=8192 that is 512 samples each, at n=6 it is
128, at n=8 it is 32. The argmin is FREE at every benchmarked size. Grover only
earns anything once 2^n >~ S, i.e. n >~ log2(S) ~ 13 params per block - past N=8,
the largest problem in the suite. And at that point its DEPTH is 2^(n/2) times
one oracle: at n=16 roughly 256 x 1000 = 2.6e5, which is fault-tolerant
territory, not NISQ. Raising bits_per_param does not rescue this - it grows the
same hypercube shots already exhaust, moving the crossover closer while running
into the identical depth wall on arrival.

MEASURED, NOT ARGUED (supplement/results/v4_argmin.log). Because the argmin is free, the
soft-vs-hard question could be settled directly: same circuit, same shots, three
decoders. Hard argmin LOSES at both sizes - E_final +0.137 at N=4 and +0.277 at
N=6 - and the failure mode is structural, not statistical: argmin always jumps to
a CORNER, moving every active parameter by exactly +-R, so it can never take a
small or zero step even at a minimum, and only the decaying R damps it. The
quantity Grover would accelerate is the one that performs worst.

═══ IS THE WALK NECESSARY? (asked properly, answered: yes) ═══

The argmin run also showed top-4 vertex averaging BEATING the marginal+walk path
by 0.136 at N=6 on half the circuits with 5.6x lower variance, which would have
meant the walk circuit - the novel part - was removable. IT DID NOT REPLICATE.
Recorded because the failure mode is more instructive than the result.

These runs are stochastic beyond the seed: `seed` fixes only the initial
parameters, the sampling is unseeded, so two experiments at the same settings are
independent draws. At N=6, E_final:

    decoder     4 seeds            6 seeds
    marginal    -8.8583 +- 0.2875  -9.0390 +- 0.0372
    top4        -8.9939 +- 0.0511  -8.9650 +- 0.0646

top4 was stable across both. The BASELINE swung 0.18 - one bad run in the first
set - and that bad draw was the entire effect. The variance advantage was the
same artefact and reverses on replication.

With 6 seeds on three problems (supplement/results/v4_softmin.log), NOTHING BEATS THE WALK
on accuracy. Every fixed-m and fixed-fraction rule is worse and fails
structurally at the edges: top4 is a no-op at H2, because averaging all four
corners of a symmetric +-R box returns the centre exactly, and boltz T=1.0
washes out to the same centre (+3.98 at N=6, 13 sigma).

ONE decoder ties it: a Boltzmann-weighted average over all sampled vertices,
w_x = exp(-(E_x - E_min)/T) with T = 0.1 * (energy spread), which reaches
-1.7916 / -6.0340 / -9.0375 against the walk's -1.7653 / -6.0064 / -9.0390 -
never worse, at HALF the circuits. So the honest statement is not that the walk
is redundant but that it roughly breaks even against a cheap classical decode of
the same shots. Whether the coherent step pays for itself is open; that it is
doing something real is not.

THE LESSON, which is the durable part: in variational optimisation the baseline's
run-to-run spread is routinely larger than the effect being claimed. A 1-sigma
result on 4 seeds is a coin flip. This project has now produced two of these -
this one, and diag_sqrt's 0.9 sigma, which would have become "the QFIM helps" if
all four metric variants had not been run. Replicate before believing, and quote
sigma with the seed count next to it.
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

PIVOTS RE-ASSESSED after the theory and cost work. Three change, one steer below
was wrong.

TOP of spectrum    (the HIGHEST eigenvalue - distinct from the next entry, which
  = highest        is about INTERIOR states. H -> -H reaches the two ENDS of the
  eigenvalue       spectrum and nothing between them.)
                   FREE, and it needs no code change at all. Anti-controlling the
                   sensing evolution (control on |0> instead of |1>) gives
                   X_anc CU X_anc, so on an eigenstate the branch amplitudes
                   become (e^{-iEt}|0> + |1>)/sqrt2 = e^{-iEt}(|0> + e^{+iEt}|1>)
                   /sqrt2 - up to global phase the relative phase flips sign,
                   which is identical to t -> -t, i.e. H -> -H. So the readout
                   returns -E and the walk climbs. Simply pass -H.
                   VERIFIED on Heisenberg N=4 (spectrum -6.4641 to +3.0000),
                   3 seeds: minimising H reaches -6.032, minimising -H reaches
                   +2.978.
                   CAVEAT on that apparent 99.3% accuracy: Heisenberg's MAXIMAL
                   state is the ferromagnetic product state, which efficient_su2
                   represents exactly, so this target is trivial. The mechanism is
                   what is validated, not a claim that excited states are easier.
                   SCOPE: this gives the two EXTREMAL states only. Interior
                   excited states still need folded spectrum or deflation - see
                   the next entry.
INTERIOR excited   (states BETWEEN the two ends - first excited, second excited.
  states           H -> -H cannot reach these, which is why this entry exists
                   separately from the one above.)
                   UPGRADED, and RE-ROUTED off the X basis. The description above
                   takes <H^2> from Re<U> ~ 1 - tau^2<H^2>/2, which is a
                   HADAMARD-path readout - and that path measured 1.91x WORSE than
                   parameter-shift on shots while carrying an irreducible sin()
                   bias, plus the signal is tau^2-suppressed so its relative
                   precision is poor. QPE makes it free instead: phase estimation
                   samples eigenvalues with probability |<E_k|psi>|^2, so over
                   shots E[e^m] = <H^m> for every m, and e^m is a PER-SHOT
                   quantity, hence its Walsh coefficients are empirical means -
                   LINEAR, so unbiased at any shots-per-vertex by T2, exactly like
                   the first moment.
                   VERIFIED (supplement/results/v5_moments.log): the degree-1
                   Walsh coefficients of e^2 come back at cos = 0.99507 with norm
                   ratio 0.9847 - BETTER norm fidelity than the first moment's
                   0.9473. So a folded-spectrum gradient needs no new circuit
                   element at all, only different classical post-processing of
                   shots already taken.
                   TWO CAVEATS the test found. Moment recovery is best at k=4-5
                   (-1.4%, -0.7% on <H^2>) and DEGRADES at k=6-7 (-4.2%, -7.3%),
                   because higher k means longer evolutions and more accumulated
                   Trotter error in the high-order bits, which distorts the tails,
                   which the second moment weights harder than the first. And it
                   is far more margin-sensitive: qpe_margin 1.2 gives -17.4% from
                   wrap, 4.0 gives +15.4% from lost resolution. margin=2.0 with
                   k=4-5 is the measured sweet spot. Tune the margin for any
                   second-moment work; the default was chosen for <H> alone.
Hamiltonian        STILL THE BEST TARGET, and the blocker is SMALLER than first
  learning         assessed. Loose precision and moderate parameter count are
                   where the cost win lives, and the no-Pauli-grouping advantage
                   is largest for Hamiltonians with many non-commuting terms -
                   which is this case.
                   THE BLOCKER, RESOLVED IN PRINCIPLE. Encoding H coefficients
                   means parameters stop sitting on single-qubit rotations, and
                   build_w_gate handles only those plus CXGate. But a controlled
                   multi-qubit Pauli rotation decomposes exactly the way the
                   degree-2 CRZZ did (already built and verified in T7):
                   e^{-i theta P} = V e^{-i theta Z_j} V^dag for the Clifford V
                   diagonalising P, so controlled-e^{-i theta P} =
                   V ; CRZ(2 theta) ; V^dag with V UNCONTROLLED - two Clifford
                   conjugations and one CRZ per term, fully general. Closing this
                   unblocks HVA ansaetze at the same time.
                   THREE DESIGN POINTS that make it fit better than chemistry:
                     * MAKE THE LOSS BE AN OBSERVABLE, not a function of several
                       observables. T2's linearity - the entire advantage - holds
                       for expectation values; a sum-of-squares over several
                       expectations is nonlinear and forfeits it. Return
                       probability or a local <Z_i> are single expectations and
                       keep it.
                     * THE RETURN-PROBABILITY VERSION NEEDS NO SENSING MACHINERY
                       AT ALL. Prepare |0..0>, apply the device's U_true, then
                       apply U_model(theta)^dag from the W-gate; at theta = c_true
                       the system returns to |0..0>. So measure the SYSTEM
                       register and test for all-zeros - no ancilla, no QPE, no
                       Trotterised sensing evolution. The per-shot readout is a
                       bounded bit, so T4's Bernoulli bound applies and the
                       cross-coordinate variance term b = 0 structurally, exactly
                       as measured for the Hadamard path.
                     * SHORT EVOLUTION TIME IS THE SWEET SPOT TWICE OVER. Return
                       probability decays exponentially in N and T (orthogonality
                       catastrophe), so short T preserves signal; and by T8 the
                       Walsh degree is bounded by the light-cone of the evolution,
                       so short T also keeps the degree LOW, which is where the
                       degree-1 estimator is nearly complete. Two independent
                       constraints pointing the same way.
                   COST at a realistic M=100 device terms with n=8: 13 circuits
                   per gradient against 200 for parameter-shift, and no Pauli
                   grouping on either the model evolution or the readout.
                   NOW BUILT AND TESTED (supplement/results/v6_hamlearn.log). N=4,
                   5 planted Z-type coefficients so Trotter is exact and cannot
                   confound, probe |+..+> because a diagonal H leaves |0..0> an
                   eigenstate and the return probability would be 1 everywhere.
                     * GRADIENT: cos(measured, smeared) = 0.99943 with norm ratio
                       0.9989, and cos against the TRUE gradient 0.99934 - the
                       direction is essentially exact while the magnitude sits at
                       49% of it (R=0.4 over 5 smeared coordinates). Same
                       direction-survives-magnitude-does-not signature as
                       everywhere else in this project, now in a different
                       application.
                     * RECOVERY: from ||theta-c|| = 0.311 to 0.047 in 30 epochs,
                       return probability 0.907 -> 0.998, worst coefficient off by
                       0.034. THIRTY circuits against 300 for parameter-shift, a
                       2M saving that grows linearly in the number of terms.
                     * The oscillation in that log (0.065 at ep5, 0.163 at ep10,
                       0.047 at ep30) is the crude step rule - a fixed magnitude
                       0.9*R*ghat keeps stepping at ~R near the optimum and only
                       the decaying R damps it. Not a method failure; a line
                       search would smooth it.
                   THIS IS THE FIRST PIVOT VALIDATED END TO END, and it needed no
                   ancilla, no QPE and no sensing evolution.
                   HONEST SCOPE: the test is deliberately classically easy
                   (commuting diagonal terms) because it validates the estimator
                   and the loop, not advantage. Non-commuting terms reintroduce
                   Trotter error in U_model, and the smearing attenuation worsens
                   as more coefficients share one hypercube - the same interior
                   block-width optimum found in the cost study will apply.

PARALLEL QLTO - two systems, one circuit. The general rule first, because it is
    what makes this work at all: ANY PER-SHOT LINEAR COMBINATION OF MEASURED
    QUANTITIES PRESERVES T2. Unbiasedness comes from the estimator being an
    empirical mean, so if each shot yields several numbers, any fixed linear
    combination of them is still an empirical mean and still unbiased at any
    shots-per-vertex. Composite objectives are therefore free, and the constraint
    is only that the combination be linear per shot - the same constraint that
    killed argmin, top-m and Boltzmann.
    Two uses follow, both grounded:
      * DEFLATION -> INTERIOR EXCITED STATES, without folded spectrum. System A
        holds |psi(theta)> under optimisation, system B holds a FIXED already-found
        eigenstate, and a controlled-SWAP between them yields a per-shot overlap
        bit s. Combine linearly per shot: e + beta*s. Its degree-1 Walsh
        coefficients are the gradient of <H> + beta|<psi_0|psi>|^2, so the
        orthogonality penalty costs no extra circuits. The W-gate is already a
        controlled state preparation, so system B needs no new machinery.
      * MULTI-PROBE HAMILTONIAN LEARNING - NOTE THE PARAM REGISTER IS SHARED, not
        duplicated. The coefficients being learned are the SAME in every
        experiment; what differs is the probe state and/or evolution time. So it
        is one hypercube over coefficients evaluated against several physical
        experiments, per-shot return bits combined linearly.
        VALIDATED STATISTICALLY FIRST, before building any parallel circuit
        (supplement/results/v6_multiprobe.log). Holding TOTAL shots per epoch
        fixed at 16384 so each of P probes gets 1/P of them:

            P   shots/probe   max |err|   rms |err|
            1        16384       0.0364      0.0296
            2         8192       0.0287      0.0285
            3         5461       0.0216      0.0183
            4         4096       0.0180      0.0170

        Error HALVES from P=1 to P=4 despite each probe getting a quarter of the
        shots, so PROBE DIVERSITY BEATS SHOT PRECISION - the diversity is real and
        not just extra measurement. That is exactly the condition under which the
        parallel register pays: it delivers P gradients in ONE circuit, so P=4
        accuracy at P=1 circuit cost, spending N extra qubits instead of P-1 extra
        circuits. (Sequential P probes = P circuits is mathematically identical for
        the gradient, so the parallel version is purely a circuit-count win.)
    WHAT WOULD NOT HELP: cross-driving the DRIFT - using anc_A to CRZ param_B.
    anc_A carries E_A, which depends on param_A, so unless the two systems
    physically interact that drift carries no information about B's own landscape
    and is not descent on anything; and if they DO interact it is just one larger
    QLTO with a partitioned param set, which sense_gradient already supports
    through active_indices. Structurally it is also the same object as the
    degree-2 CRZZ - a coupling term in the drift phase - so T7 applies unchanged.
    WHAT MIGHT: cross-driving the MIXER. CRX moves populations, so conditioning it
    on energy is an ADAPTIVE EXPLORATION RATE, and the walk currently has a fixed
    beta schedule with nothing adapting it - every ablation to date varied only the
    drift. This is the mixer half T7 flagged as the remaining open question.
    But it does NOT need a second QLTO: condition the mixer on the SAME instance's
    k-bit QPE energy register. Same mechanism, a fraction of the qubits, and still
    untested.
metrology          Strengthened by the same mechanism: QFI relates to variance,
  (speculative)    and Var(H) per vertex is now free and LINEAR from the QPE
                   sample second moment rather than needing the X basis. Still
                   speculative.
reservoir          Unaffected. Note the walk being degree-1 limited does not hurt
  computing        here, because the readout trains on raw ancilla statistics, not
  (speculative)    on the marginal - the full nonlinearity is in the measurement
                   record either way.

HONEST STEER: chemistry is where V3 measured WORST (H2 last of six), where
classical methods are strongest, and where precision demands are harshest.
Loose-precision, moderate-dimension problems are where the eps > 1/sqrt(d)
scaling actually favours this method.

THE OLD STEER - "if only one thing gets built, build the comparator" - IS NOW
WRONG. The Grover branch it was meant to unlock is closed (WHY GROVER IS NOT THE
NEXT STEP); quantum counting is available classically from the sensing shots at
every benchmarked size, and it feeds nonlinear decoders that do not survive past
n ~ log2(S); and threshold-conditioned drift is contraindicated twice over, by the
Grover analysis showing hard thresholds discard the gradient weighting that does
the work, and by T7 showing the drift cannot absorb extra structure anyway.
REPLACEMENT STEER: build nothing. The moments are already in the shot record as
linear functionals, so excited states, the variance preconditioner and the
metrology QFI are all reachable by changing the classical decode alone. That is
the cheapest unlock available and it needs no new circuit element.

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

═══ THE GRADIENT SCALE BIAS ═══

The sensed gradient has the right DIRECTION and the wrong LENGTH, per block, by
up to 2x. Found by comparing ||g_sensed|| against the R-SMEARED gradient - the
quantity sense_gradient actually targets - rather than against grad E, which was
the mislabelled baseline that made this look mysterious for so long.
(supplement/results/anomaly_wdag.log, anomaly_c.log, anomaly_e.log, anomaly_f.log.)

NOT shot noise. Sweeping the budget to 1e6 shots on Heisenberg N=4, the spread
falls as 1/sqrt(S) while the mean does not move at all:

    ratio = ||g_sens|| / ||g_smeared||    blk0 Y   blk1 Z   blk2 Y   blk3 Z
    Hadamard @ 1e6 shots                   0.949    1.237    0.831    1.136
    QPE k=4  @ 1e6 shots                   0.967    2.139    0.830    1.750

blk1 under QPE is 2.139 +- 0.014 - eighty sigma from 1.0. The pattern tracks the
block's ROTATION AXIS (both Y low, both Z high), not its position or its gradient
magnitude.

TWO MECHANISMS, OPPOSITE SIGNS.

1. TROTTER ERROR inflates. The sensed observable is not H but the effective
   Hamiltonian of the Trotterised evolution, whose error is O(t^2 [H_i,H_j]) and
   therefore STATE-dependent - so it lands differently on each block. Confirmed
   by sweeping reps at FIXED tau, which moves Trotter error and leaves everything
   else untouched:

     Hadamard, tau=0.1057    blk0 Y   blk1 Z   blk2 Y   blk3 Z
       lie reps=1             0.934    1.616    0.791    1.349
       lie reps=2  SHIPPING   0.949    1.220    0.876    1.132
       lie reps=16            0.942    1.009    0.876    0.967
       suzuki4 reps=4         0.939    0.803    0.877    0.986

     QPE k=4, tau0=0.2430
       reps x1     SHIPPING   0.969    2.121    0.826    1.728
       reps x4                0.990    1.294    0.922    1.200
       reps x8                0.991    1.119    0.968    1.071

2. THE sin() NONLINEARITY compresses, and only the Hadamard path has it. The
   Y-basis readout is -<sin(H tau)>, so dividing by tau estimates

       <sin(H tau)>/tau  =  <H> - (tau^2/6)<H^3> + O(tau^4)

   sin(x) < x, so this always reads LOW. It is why the Hadamard Y blocks sit at
   0.88-0.95 no matter how many reps are thrown at them, and why Suzuki-4 - which
   removes nearly all the Trotter error - exposes a floor where every block reads
   <= 1 (0.94, 0.80, 0.88, 0.99). Lie reps=16 landing at 1.009 on blk1 is NOT
   convergence; it is residual Trotter cancelling the sin() deficit.

QPE IS ASYMPTOTICALLY UNBIASED AND HADAMARD IS NOT. QPE decodes an energy
directly, so it has no sin() term, and at reps x8 every block including the Y
ones converges toward 1 (0.991, 1.119, 0.968, 1.071) where Hadamard's stay pinned
at 0.88. This is a second, independent reason to prefer QPE over the variance
argument already recorded - the k=1 path has a bias no budget and no product
formula can remove, only smaller tau, which costs 1/tau^2 in variance.

RULED OUT: QPE quantisation. Sixteen-fold finer bins (k=3 to k=7, bin width
3.232 -> 0.202 against a signal of ~0.25) leave the bias completely unmoved. The
resolution argument that motivates adaptive k does not explain this.

WHAT IT COSTS TO FIX, AND WHY THAT IS NOT OBVIOUSLY WORTH IT. Both shipping
configurations are under-resolved - Hadamard reps=2 carries a 22% scale error on
blk1, QPE reps x1 carries 112%. But reps x8 multiplies the sensing circuit's
evolution depth by eight, and low depth is half of what V3 is selling. A per-block
scale error acts like a per-block learning-rate error, and the tuned schedule
absorbs some of it, so it does not follow that fixing the gradient improves the
final energy. UNTESTED and worth testing: A/B the end-to-end benchmark at reps x1
against reps x4. If accuracy does not move, that is itself the interesting result
- it would say the walk is robust to gradient scale error, consistent with the
ablation showing direction is what matters and with the QFIM being redundant.

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

SETTLED (supplement/results/fim_test2.log). Both caveats above were closed by running four
metric variants against each other on the same seeds, gradient source, walk,
k_steps and schedule - only the preconditioning differs. Heisenberg, 4 seeds,
20 epochs, k=15, generators correct:

    metric        N=4 E_best   vs none      N=6 E_best   vs none
    none            -6.0486         -         -9.0540         -
    diag_sqrt       -6.0446   +0.0040 (0.2s)  -9.0986   -0.0447 (0.9s)
    natural         -6.0241   +0.0245 (0.6s)  -8.8777   +0.1763 (0.7s)
    natural_norm    -6.0175   +0.0312 (0.6s)  -8.8592   +0.1947 (0.8s)

`natural` is the proper block solve pinv(F_blk) @ g_blk with the identity metric
inside the walk - the full block, no sqrt, no clip. It does not help. Nor does
diag_sqrt. Nothing is significant and every proper-metric variant is nominally
WORSE. So the redundancy argument now rests on evidence, not on the absence of a
test.

`natural_norm` is the one that matters, because it removes the confound. F^-1
changes both the DIRECTION and the MAGNITUDE of the step (measured ||F^-1 g||=2.30
against ||g||=0.794 on one block), so a fixed schedule tuned for g would overshoot
and a loss could be a step-size artefact rather than the metric being useless.
natural_norm rescales F^-1 g back to ||g|| so only the direction differs. It is
still nominally worse at both sizes. The metric's DIRECTION carries nothing here.

One result was not predicted: at N=6 both natural variants have std 0.48-0.50
against none's 0.088. With the magnitude matched that extra variance cannot be
step size, so it comes from F itself being SHOT-ESTIMATED - F^-1 amplifies the
metric's own estimation noise while contributing no signal. Under sampling the
QFIM is not merely redundant here, it is harmful. A metric is only worth its
noise if it buys direction, and on the hypercube it does not.

This does not contradict CorrectQNG winning two problems: that is ordinary
gradient descent preconditioned by F, where the metric is the ONLY thing steering
the step. The walk already searches states rather than parameters, so it is the
walk that makes F redundant, not F that is wrong.

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
                 num_ancillas=4, qpe_margin=2.0, uncompute_w=False,
                 merged_walk=True, skip_dead_blocks=True):
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
        """
        self.layer_diagnostics = {}
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
