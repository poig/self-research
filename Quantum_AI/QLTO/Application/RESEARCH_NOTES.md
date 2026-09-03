# QLTO — research notes

A research record, not API documentation: every number cites the log that
produced it, and negative results are kept alongside positive ones because most
of what was learned here is where things do *not* work. **Withdrawn claims stay
in place, beside whatever refuted them** (project rule R2) — do not tidy one away.

## What is in this file

Parts III–VII: the current state of the project.

| part | what it is |
|---|---|
| **III — the QML axis map** | v121–v131. Per axis, which of *qubits / circuits / gates* stays small. Conflating those three is where most over-claims came from |
| **IV — where advantage can live** | design + literature, labelled `[measured]` / `[lit]` / `[design]` line by line |
| **V — the accounting rule** | substrate vs. invocation count. **Read before any cost comparison** |
| **VI — the walk step, derived** | the separability theorem; supersedes the walk verdict in I and III |
| **VII — the bridge** | simulability ladder, potential degree, the three complexity barriers |
| **VIII — the sensing register is a walk** | v135. The design register *is* the hypercube mixer, so its eigenbasis is the measurement grading — and the radius is a second, independent axis |
| **IX — the walk built, and the prototype** | v136–v141 + `qlto_walk.py`, `qlto_prototype.py`. The cycle register is a PARTICLE and the hypercube a SPIN; three-level sensing gives the Hessian; end-to-end training |

**Parts I and II are in [`ARCHIVE_V3_V6.md`](ARCHIVE_V3_V6.md)** — the V3-era
record and the V6/calibration line. Kept verbatim per R2; its header lists which
of its verdicts the parts here supersede.

## The mechanism, in brief

Parts III–VII assume this; the derivation is in the archive's *MECHANISM* section.

The W-gate prepares a uniform superposition over the `2ⁿ` vertices of a box
`θ_c ± R` and entangles each vertex with the ansatz state it labels. One
measurement circuit per commuting group then gives **every** parameter's gradient
component from the same shot record, decoded as a marginal:

    g_i = [ <E | σ_i = +1> − <E | σ_i = −1> ] / 2R

V5 spent one register qubit per parameter. **V6 carries all M on a
resolution-IV Hadamard design over `⌈log₂(M+1)⌉+1` qubits**, so one gradient costs
`G` circuits — the number of qubit-wise-commuting groups — and nothing in that
count depends on `M`. `readout='qpe'` replaces the basis measurement with a phase
ladder, making it one circuit whatever `G` is (v132).

## Navigation

**Part III — the QML axis map** (v121–v131)
- The map: qubits vs circuits vs gates, per axis
- Data axis · Parameter axis · Observable axis · Nonlinearity · Backprop
- The speedup question, answered plainly *(read its central claim narrowly — Part IV)*
- The cost model: when the circuit saving pays, and when it stops (v130)
- The walk axis, closed at the theorem level *(found landscapes only — Part IV)*
- Backprop: impossible, not merely expensive · DiffusionBlocks

**Part IV — where the advantage can actually live** *(design + literature)*
- The correction that reopens everything: not all structure is dequantizable
- The principle: advantage comes from the DATA-GENERATING PROCESS
- The natural instance: classical inputs, quantum-hard labels
- Tree encoding: what it buys · Noise: deferred, not dismissed
- Three flows: classification · imaging (QCNN) · LLM
- **The three axes are one system, not a menu** — encoding, gradient, step
- What is genuinely open · Correction to Part III's walk verdict

**Part V — the accounting rule** *(read before any cost comparison)*
- Substrate vs. invocation count — the rule, and why the substrate cancels
- Grouping helps both arms · a large G bounds the problem · phase sensing != precision QPE
- What the QFT sensing route gives: 1 invocation vs 2M
- Resource arithmetic at 100 qubits / 200k depth · status and the open ledger item

**Part VI — the walk step, derived** *(supersedes the walk verdict in I and III)*
- What the step is: CTQW on the hypercube in a potential = the measured gradient
- Why it works: the adiabatic endpoint IS signed gradient descent
- The limit as a theorem: the evolution factorises, zero entanglement, O(n) classical
- Why v99's PART 4 and PART 5 describe different constructions
- Where improvement is: non-factorising mixer (Wang's taxonomy) or degree-2 potential

**Part VII — the bridge to classical hardness** *(design + one tier-C measurement)*
- The simulability ladder: product · free fermion · Clifford · MPS · treewidth
- Measured: potential DEGREE is the binding constraint, not the mixer
- Circulant mixer + computational potential = split-operator Schrodinger on the landscape
- The defensible claim is v99 PART 4's exp(width) vs exp(height), not representation
- Why resolution-V design rows (`M^2`) exactly match the degree-2 coefficients
- The composed algorithm: a trust-region method with a quantum subproblem solver
- **Three complexity barriers**: permutation invariance · NP not in BQP · the DLA collapse
- The DLA collapse: trainable => simulable, the QML dilemma as complexity
- What it does NOT establish, which is most of it

**Part VIII — the sensing register is a walk** *(v135, tier B)*
- The identification: `h(param)` prepares the hypercube mixer's ground state,
  and Walsh degree `|S|` **is** its eigenvalue `n−2|S|`
- Therefore the mixer taxonomy is a table of MEASUREMENT structures, not only
  of ways to break separability
- What the current register measures, exactly: a low-pass filter in Walsh weight
- **The second axis**: the radius grades by DEGREE; `log₂d` qubits, QFT-separable
- The estimator as a cubature problem — support size vs. total variation
- Two theorems: coordinate-supported designs pay `ν ≥ M`; spreading forces `R ∝ 1/√M`
- The two prior-art poles: Lin et al. 2022 (adjoint) and He 2023 (circuit count)

**Part IX — the walk built, and the prototype** *(v136–v141, tier A/B)*
- The register split: cycle = particle `e^{-S0/h}`, hypercube = SPIN `e^{-n S~}`
  — so the `{±1}ⁿ` box register every variational algorithm uses degrades
  exponentially in the PARAMETER COUNT
- Corrections to Part VI (Laplacian sign, ground-state vs top-of-spectrum)
  and Part VII (the mixer matters, for a different reason than stated)
- Where the Fourier energy sits: weight-1 and -2 identically ZERO, bulk at M/2
- Three levels: the diagonal Hessian appears, `p0` becomes a knob, q≥4 is a hard stop
- `qlto_walk.py`: g, diag(H), offdiag(H) from one shot record
- The walk as a fast SAMPLER: 90.2%% of brute-force descent in 3.1 moves vs 97.1
- The exponent trade, and why the LATTICE is what makes thin barriers exist
- Why Liu-Su-Li's query model is already ours
- `qlto_prototype.py`: end to end, and the three things only a loop surfaces

Logs live in `supplement/results/`; the script that produced each one is named in
`supplement/README.md`.

---

# Part III — the QML axis map: what can be exponential, and what we established

Part II ended at the frontier comparison. This part answers a narrower and more
useful question: **for each axis along which a QML problem can be exponentially
large, which resource stays small, which does not, and which did we actually
measure?**

The distinction that organises everything below is between three resources that
are routinely conflated:

| resource | what it is | who pays |
|---|---|---|
| **qubits** | register width | hardware size |
| **circuits** | distinct jobs submitted | queue time, latency, compilation |
| **gates** | depth x width of one job | coherence |

A claim of "flat in |D|" is meaningless until it says *which of the three*.
Almost every over-claim in this line came from eliding that.

## The map

| axis | size | qubits | circuits | gates | established by |
|---|---|---|---|---|---|
| **data** | D = 2^d | **log D** yes | **3, flat** yes | **Theta(D)** NO | v74, v122-v125 |
| **parameters** | M | **log M** yes | **G, flat in M** yes | **Theta(M)** NO | V6, v119 |
| **observable** | G groups | — | **G** (=1 for QML) yes | — | v30, v119, v120 |
| **nonlinearity** | — | — | — | — | not addressed |
| **backprop** | L layers | — | no analogue | — | not addressed |

Read the NO column first. **On both axes that matter, the exponential is removed
from circuit count and NOT from gate count**, and the reason is the same in both
cases: QLTO compresses *interrogation*, not *representation*.

## Data axis — D = 2^d

**What works.** A register in superposition, entangled into the system and never
uncomputed, is traced out at measurement, so the expectation of O tensor I is the
batch mean (v74). Weighting it by p_x proportional to |w_x| gives the MSE
gradient (v121 tier B; v122 cos 0.9768 with exact weights; v123 cos 0.9773 with
weights measured from one circuit). Three circuits per epoch, counted not
asserted, at every size tested.

**What does not.** v125 separated the blocks on transpiled circuits:

    weighted state prep     2q gates ~ |D|^1.29     50% of the circuit at |D|=32
    linear data encoder     2q gates ~ |D|^0.44     logarithmic, as designed
    arbitrary data encoder  2q gates ~ |D|^1.06

The **state prep**, not the encoder, is the block that is linear in |D| — and it
is Theta(|D|) by parameter counting, since setting 2^d - 1 free amplitudes needs
at least 2^d - 1 angles. Structure buys the encoder; it does not buy the prep.

**And the cheap encoder cannot hold arbitrary data.** The angle vector
A @ bits(x) is linear in the register bits, so the D samples are parallelepiped
vertices. Least-squares fit to a random angle table captures 40%, 45%, 26%, 12%
of its variance at d = 2, 3, 4, 5 — falling as d grows.

**The one loophole, and its fate.** Theta(D) bounds an *arbitrary* distribution.
For w_x = g(A bits(x)) with rank(A) = k, every cut splits the latent coordinates
additively and a degree-r polynomial approximation gives

> **chi <= C(k+r, r) at every cut — no 2^d.**

Measured (v127), same runs, four amplitude functions, chi\* = smallest bond
dimension under the shot noise already present:

| d | \|D\| | max rank | w (raw) | \|w\| | sqrt\|w\| (current) | sqrt(w+c) (shifted) |
|---|---|---|---|---|---|---|
| 4 | 16 | 4 | 3 | 3 | 3 | **3** |
| 6 | 64 | 8 | 3 | 6 | 6 | **3** |
| 8 | 256 | 16 | 6 | 12 | 8 | **4** |
| 10 | 1024 | 32 | 6 | 12 | 12 | **3** |

The current encoding grows with d; the shifted one is **flat**, and the
k-dependence the bound predicts appears (chi\* = 4, 4, 6 for k = 2, 3, 4 at
d=8). sqrt of an absolute value is non-smooth at zero and a residual crosses zero
constantly — the bound was never violated, it did not apply.

**Then v128 closed it anyway, on variance.** The shifted reconstruction is a
difference of two quantities of order cD against a target of order the norm of
sum_x w_x df_x, and c is exactly what buys the smoothness. At matched shots the
shift loses at every gamma = c/max|w| and every size — all 18 cells — with kappa
climbing 15.2 to 32.3 at d=8 in lockstep with the cos loss. **Best gamma is 1.05
at all three sizes: the smallest tested, so the sweep is pinned against its own
floor and there is no interior optimum.**

> **The structure is real and this estimator cannot spend it.** Both results
> stand and point opposite ways. Reviving the route needs a reconstruction that
> is not a large cancelling difference — not a better c.

## Parameter axis — M

V6 costs **G circuits per gradient against parameter-shift's 2MG**; G cancels, so
the ratio is 2M and general commuting does *not* widen it (v119).

> **See Part V.** G is paid by parameter-shift 2M times over, so a large G bounds
> the APPLICATION — which problems are reachable — and not the choice of method
> among those that are.

**But the ansatz still contains M parameterised gates.** Nothing here compresses
that. The log-width design register makes the *measurement* logarithmic in M; the
*circuit* is still Theta(M). This is the exact dual of the data axis:
representation is untouched, interrogation is compressed.

Two hard limits, both established:

- **O(1) circuits and arbitrary precision are mutually exclusive.**
  Var = Var_settings/K + E[Var_shots]/S — driving K to O(1) leaves a floor no
  shot budget removes.
- **The circuit saving is bought with shots**, ~4x more for 32x fewer (v109).

## Observable axis — G

G = number of qubit-wise commuting groups. G ~ N^4.24 for molecules (v30), G = 3
Heisenberg, G = 1 MaxCut. **A QML readout is a single Pauli, so G = 1
structurally** — this is V6's ideal regime and the reason the QML branch cost is
2 and not 2G. Minimum-clique-cover grouping (v120) cuts term counts but, because
G cancels in the ratio, does not widen V6's advantage.

## Nonlinearity — not addressed

The trace-out identity returns sum_x p_x f_x, which is **linear in f**. That is
precisely why MSE needed the weighted-register trick at all: a uniform register
returns the gradient of the unweighted mean, and v121 measured its cosine
*swinging in sign* (+0.9367, +0.3359, +0.8180, -0.9770, -0.9602) along a descent.

Any nonlinearity must live in f itself (the ansatz) or in classical
post-processing. **Softmax, ReLU, attention: no result in this project.** v99
established that the quantum-walk advantage is real but that the walk does not
access it, which is the nearest thing to a nonlinearity result here and is
negative.

## Backprop — no analogue

Parameter-shift and V6 both give gradients. Neither gives backprop's defining
property: reverse-mode reuse making the gradient cost O(1) forward passes
regardless of depth. Nothing in this project addresses it.

## The speedup question, answered plainly

**No exponential quantum-vs-classical separation is established anywhere in this
line, and there is a specific structural reason to expect none on the data axis.**

The obstruction is self-inflicted and worth stating precisely: *the latent
structure that makes the quantum encoding cheap is the same structure that makes
the classical problem tractable.* If w_x = g(A bits(x)) with k latent
coordinates, then the function being learned is a function of k real variables,
and a classical tensor-network contraction of the same MPS costs O(d chi^3) —
polynomial. Grant the classical side the same structural access and it matches.
This is Tang's dequantization pattern recurring in our own setting.

> **READ THAT SENTENCE NARROWLY — see Part IV.** It holds for the structure v127
> measured (smooth, low-rank latent) and is FALSE in general. Group-theoretic and
> cryptographic structure is quantum-easy and classically-hard at once, and Liu,
> Arunachalam & Temme (Nature Physics 17, 1013, 2021) prove an exponential
> separation for supervised learning on CLASSICAL data on exactly that basis.
> "Structure" is not one thing, and only the smooth kind is dequantizable. The
> claim that survives is the narrower one: **this project's data axis is
> dequantizable**, because the structure it relies on is the smooth kind.

Every ratio in this line is **quantum-vs-quantum**: 2M against parameter-shift,
32x circuits against 4x shots. Those are real and measured. They are engineering
wins on job count — queue time, latency, compilation — which is the dominant
practical cost on cloud QPUs. They are not complexity-theoretic advantage.

**Where the input model is defensible is the calibration line.** twirl_cal and
qlto_hl take their oracle as exp(-iHt) supplied by hardware: there is no
amplitude to prepare and nothing for a dequantization argument to attack, because
the data *is* the device. That is the one side of the wall where an exponential
claim would not be resting on an assumed input model — and it is where this
project was already standing before the QML detour.

## What would have to be true for a QML speedup

All four, and we have none of them:

1. **efficient input** — a data-access model no classical algorithm gets for free
2. **efficient gradient** — QLTO supplies this, and it is the only box ticked
3. **a model class hard to simulate classically** — untested; our ansatz is 3
   qubits and trivially simulable
4. **a task where classical local optimisation is provably exponential** — the
   welded-trees construction shows such tasks exist; embedding one in a
   parameterised loss is not done

Box 2 alone is an optimiser, not an advantage.

## The cost model: when the circuit saving pays, and when it stops (v130)

Every advantage in this project is a circuit-count saving bought with shots. That
is a good trade or a bad one depending on `r_circ / r_shot`, which is a property
of the MACHINE and not of the algorithm:

    total cost  =  r_circ * circuits  +  r_shot * shots

Measured at matched accuracy (cos ~0.955), against parameter-shift **on the same
data register** — 1 + 4M circuits, not the naive per-sample 2M|D|, which
overstates the advantage by |D|/2:

| arm | circuits | total shots | cos |
|---|---|---|---|
| qlto_qml | **3** | 49,152 | +0.9541 |
| param-shift on register | 49 | **12,544** | +0.9556 |

So 3.9x more shots for 16x fewer circuits, consistent with v109's ~4x/32x.
Solving the cost model:

> **QLTO is the cheaper route to the same accuracy only when one circuit costs
> more than about 796 shots.**

**On cloud hardware this holds by orders of magnitude.** A shot is microseconds;
per-job overhead — queue, compile, waveform upload, recalibration — is seconds to
minutes. Every number in this project was measured in that regime.

**And it inverts on a local accelerator.** A photonic or NV-diamond chip sitting
inside the machine rather than behind a queue exists precisely to drive `r_circ`
toward the cost of reloading a pulse sequence, i.e. toward a few shot-equivalents.
Below 796, QLTO is paying ~4x the shots for a saving worth nothing.

> **The advantage is largest in the regime people want to escape and smallest in
> the one they are building toward.** Not a reason to stop — queues will exist for
> a long time — but the circuit-count framing has a shelf life, and any claim
> resting on it should name `r_circ / r_shot` as the assumption it depends on.

The threshold falls as M grows (parameter-shift costs 1+4M circuits, so more
parameters means more circuits saved), so QLTO holds up longer at large M as
`r_circ` drops. At M=12 it is 796.

### This is a quantum-vs-quantum result, and the classical question is worse

Part III already established both axes are Theta(D) in gates. A cost comparison
against classical therefore reduces to cost per operation, and three factors run
against quantum, none of which this project can fix:

- **Throughput.** A GPU does ~1e15 operations/sec; a quantum chip does perhaps
  1e5-1e6 gates/sec. At Theta(D) work on both sides that is a ~1e9 gap before
  energy is considered at all.
- **Energy.** A dilution refrigerator draws 10-25 kW continuously, so the
  wall-plug cost per gate is enormous. Room-temperature photonic and NV platforms
  remove exactly this term, which is why they are the right platforms to name —
  but removing it does not touch throughput.
- **Error correction.** 1e3-1e4 physical operations per logical one, multiplying
  whatever the per-operation cost turns out to be.

**Cheap-at-scale requires the classical side to be asymptotically worse, not
constant-factor worse.** On classical data with latent structure it is not: the
same structure that makes the quantum encoding cheap makes classical
tensor-network contraction cheap, at O(d chi^3).

The regime where the asymptotics do favour quantum is where the classical cost is
Theta(2^N) rather than Theta(D) — simulating quantum systems. That is the
calibration and Hamiltonian-learning line, where classical is exponential and
quantum is polynomial, and where any per-operation cost ratio is eventually
overwhelmed. It is the same conclusion Part III reached from the input-model
direction, arrived at independently from cost.

### CAVEAT ON THE MATCHING CRITERION — this section matches on cos, which is wrong

The 796 figure matches the two arms at equal **per-epoch cosine**. That is the
wrong criterion and it is unfavourable to QLTO in a way worth stating before the
number is quoted anywhere.

Cosine measures directional alignment on a single step. An optimiser does not
need high per-step alignment: SGD converges at cos ~0.1, and SPSA at ~1/sqrt(M)
by construction (v129). What matters is whether the iterate descends while the
trust radius shrinks — and V6 already shrinks it, via `_radius` and the
`radius_exponent` knob.

v130's own table shows how much this matters:

| arm | total shots | cos |
|---|---|---|
| qlto_qml | **3,072** | +0.6979 |
| param-shift | 12,544 | +0.9556 |

If cos ~0.70 is enough to converge, QLTO uses **4x fewer shots AND 16x fewer
circuits**, and the trade reverses from a 4x shot penalty to a 4x shot saving.
The 796 break-even is then meaningless.

The correct comparison is **shots-to-reach-a-target-loss**, not shots-to-match-a-
cosine. See v131.

## The walk axis, closed at the theorem level (Manouchehri & Wang 2014)

v99 measured the tunneling mechanism and found it real — classical annealing
costs ~exp(height), quantum tunneling ~exp(width), and classical success collapses
1.00 -> 0.00 over height 2 -> 20 while quantum transmission stays flat. v99's own
conclusion was that the open question is "whether an ansatz can be designed whose
landscape has tall THIN barriers", and v118b removed the cost objection by
measuring the arithmetic oracle flat at 1045 gates regardless of width.

**Reading the source closes it, and not for an engineering reason.**
Manouchehri & Wang, *Physical Implementation of Quantum Walks* (Springer 2014),
Sect. 2.1 confirms Childs et al.: a CTQW crosses the glued tree in O(d^2) where
classical is bounded below by 2^Omega(d). Three results on the same and
neighbouring pages remove it again for anything shaped like an optimiser:

- **Tregenna et al. (2003)**, Sect. 2.1 — the fast hitting time is "highly
  sensitive to the symmetry of the problem: for quantum walks starting at a node
  other than root A, the exit becomes exponentially harder to find and the
  quantum walk does no better than a classical algorithm."
- **Keating et al. (2007)**, Sect. 1.4 — on glued trees with imperfections "the
  propagation of quantum information is suppressed exponentially in the amount of
  imperfection, and is therefore unlikely to be useful for algorithmic purposes."
  Anderson localization.
- **Yin et al. (2008)**, Sect. 1.4 — under *static* disorder CTQWs "often perform
  worse than their classical counterparts due to Anderson localization"; under
  *dynamic* disorder "there is no benefit... the ballistic evolution crosses over
  to classical diffusion after some time."

**An optimisation landscape violates both preconditions by construction.** An
optimiser starts at a random theta_0, not at the distinguished symmetric root —
Tregenna alone removes the exponential. And a loss landscape *is* a disordered
medium: rugged, data-dependent, irregular. That is what makes it worth optimising,
and it is exactly what Yin says makes the walk worse than classical.

> So the barrier is not that we have not found the right embedding. The properties
> that make an optimisation problem interesting are the properties the theorem
> requires be absent. Recorded as BLOCKED, with the theorem that blocks it, rather
> than left as "open".

**SCOPE OF THAT VERDICT — see Part IV.** It applies to FOUND landscapes only. A
CONSTRUCTED one satisfies Tregenna by making your start the root, and admits no
disorder for Keating/Yin to attack. Constructed landscapes fail for a different
reason: the glued-trees separation is **informational, not geometric** — it needs
the graph hidden behind an oracle with random vertex names, and a structure you
build is one you can navigate. Part IV also records that the column reduction IS a
commutant projection, which merges this axis with the commutant/MBL one.

**SECOND SCOPE NOTE — see Part VIII.** Tregenna, Keating and Yin are all about
**glued-trees hitting time from a distinguished root**, and Anderson localization
is a statement about diffusive TRANSPORT on a lattice. Neither speaks to a walk
used as an ANNEALING MIXER over a designed potential, which is what Part VI's
step and QWOA both are. The verdict above closes walk-as-transport on found
landscapes; it does not close walk-as-mixer, and Part VIII gives a third use —
walk-as-measurement-basis — that it does not reach either.

Wang's companion lecture notes (CQCWS1_19335) sharpen the same edge from the
implementation side: efficient circuit implementation of a walk requires **high
symmetry, sparsity with efficiently computable entries, efficient
diagonalisability (circulant/Toeplitz/Hankel via QFT), or composite structure
(commuting graphs, Cartesian products)**. That list is close to a list of graphs
that are also classically tractable — and QLTO's product mixer sits on it as a
Cartesian product, which is precisely why v99 Part 5 found the inner loop
collapses to one controlled-SU(2) per qubit, exact to 3.5e-15.

## Backprop: impossible, not merely expensive

Part III's table lists backprop as "no analogue". The reason is stronger than
that entry suggests and it reframes what QLTO is for.

**Reverse-mode AD is impossible on a quantum computer.** It requires reading
intermediate activations; measuring an intermediate quantum state collapses it,
and no-cloning forbids stashing a copy. So every quantum gradient method is
necessarily forward-mode. There is no engineering choice being made.

That places QLTO in a family whose classical members are currently active:

| | passes | memory | recovers |
|---|---|---|---|
| backprop | 1 fwd + 1 bwd | O(depth x width) activations | all M, exact |
| MeZO / SPSA | 2 fwd | O(1), a seed | 1 random projection |
| **QLTO** | G circuits | log M ancillas | **all M** |

QLTO holds MeZO's position on the memory axis with backprop's completeness on the
gradient axis.

**The honest limit.** Stripped of quantum, the design register is a resolution-IV
Hadamard design recovering M directional derivatives from few evaluations —
classical design-of-experiments. Classically, backprop still wins on compute (one
backward pass beats any number of forward probes) and loses only on memory. So
this does not retire backprop where backprop exists. It is the right tool where
backprop *cannot* exist: quantum circuits, black-box simulators, non-differentiable
objectives.

### DiffusionBlocks (Shing, Koyama, Akiba; ICLR 2026, arXiv 2506.14202v4)

A classical result, and the most useful adjacent one found so far. Residual
connections z_l = z_{l-1} + f_l(z_{l-1}) are Euler discretisations of the reverse
probability-flow ODE, and score matching at each noise level is **provably
independent** of other levels. So L layers partition into B blocks, each assigned
a noise range and trained on its own score-matching objective, with gradients
needed for **one block at a time** and memory falling proportionally to B. Unlike
earlier block-wise training this local objective is derived rather than ad hoc,
which is why it holds up on transformers rather than only small classification.

Why it matters here, stated without overreach:

- It is a **principled decomposition that never needs a global backward pass** —
  the same constraint quantum hardware imposes absolutely. Classical ML is
  choosing, for memory, what quantum circuits have no choice about.
- Under such a decomposition a quantum gradient call sees **M_block, not M_total**,
  so the design register is ceil(log2(M_block+1))+1 qubits.
- Each block's local objective is **one scalar loss, hence G = 1** — V6's ideal
  regime, reached structurally rather than by assumption.

It does **not** supply a quantum advantage, and it does not touch data encoding.
It is classical and self-sufficient. What it supplies is a training architecture
whose shape is compatible with the no-backprop constraint — necessary, not
sufficient.

---

# Part IV — where the advantage can actually live, and the flows that reach it

Part III mapped what is measured. This part is **design reasoning and literature**,
not measurement, and every claim below is labelled as one of:

- **[measured]** — a tier-A/B/C result from this project, with its file
- **[lit]** — a published result, cited
- **[design]** — reasoning about architecture, not yet built or tested

Nothing here is a QLTO result. It exists because Part III's negatives kept
pointing at the same gap, and the gap turned out to be in my own framing rather
than in the physics.

## The correction that reopens everything

Part III states: *"the latent structure that makes the quantum encoding cheap is
the same structure that makes the classical problem tractable."*

**That is over-generalized and should be read narrowly.** It is true for the
structure v127 measured — smooth latent structure, `w_x = g(A·bits(x))`, where
classical tensor contraction matches at O(d·χ³) **[measured, v127]**. It is
**false in general**, because structure is not one thing:

| structure | quantum | classical | advantage |
|---|---|---|---|
| smooth latent / low-rank | poly | **poly** (contract the TN) | none |
| loopy / PEPS | hard | #P-hard | none — hard for both |
| **group-theoretic / cryptographic** | **poly** | **exponential** | **yes** |

A QFT over Z_N is polynomial; extracting the period classically is not. So
quantum-easy-and-classically-hard structure exists, and "structure" cannot be
treated as one category.

**The category is proven nonempty for CLASSICAL data.** Liu, Arunachalam & Temme,
*A rigorous and robust quantum speed-up in supervised machine learning*, Nature
Physics 17, 1013 (2021) **[lit]** — a supervised classification task on classical
inputs with a rigorous, noise-robust exponential separation: a quantum kernel
learns it, no classical learner can, under discrete-log hardness.

  Cryptographic hardness is natural hardness, and it has already been used for
  exactly this — the hiding a separation needs is not necessarily contrived.

## The principle this yields

> **The advantage cannot come from the optimiser, the encoder, or the model. It
> has to come from the DATA-GENERATING PROCESS being quantum-hard. Everything
> else is plumbing that must merely not be exponential.**

Stated that way the axes compose, and QLTO's role becomes precise — it is
plumbing, and good plumbing:

| axis | requirement | status |
|---|---|---|
| input encoding | polynomial | ✓ if descriptors are low-latent **[measured, v125]** |
| gradient | polynomial, forward-mode | ✓ QLTO: G circuits, log M register **[measured]** |
| loss | scalar | ✓ G = 1 structurally **[measured, v119]** |
| **label rule** | **quantum-hard to compute** | **the axis that decides everything** |

Note what this does to v127: **low latent dimension in the INPUTS stops being a
limitation and becomes a requirement.** Cheap-to-encode inputs are wanted. The
hardness was never supposed to live in the encoder.

## The natural instance: classical inputs, quantum-hard labels

Discrete log is a proof, not an application. The non-artificial version **[design]**:

    inputs   molecular / materials descriptors - classical, low-dimensional, cheap
    labels   ground-state energies, spectra, reaction barriers - Theta(2^N) classically

A classical learner is fitting a function it cannot efficiently evaluate. A
quantum learner can evaluate it natively. The hardness is physics, not
construction. And every thread in this project converges there:

- **encoding** cheap, because descriptors have few latent factors (v125, v127)
- **gradient** QLTO, forward-mode, the only mode quantum permits
- **loss** one scalar, G = 1
- **label rule** quantum-hard by physics
- **twirl_cal / qlto_hl** not a rival path but the CALIBRATION LAYER that makes
  the stack run on a real device, where `e^{−iHt}` is the oracle

## Tree encoding: what it buys and what it does not

**It does not give arbitrary data.** A TTN needs χ = 2^(d/2) at the middle cut to
represent an arbitrary vector, which hands back 2^d. The counting bound of v125
stands and no hierarchy evades it.

**It does not need to.** Real data is hierarchically low-rank — an empirical fact,
not a hope; it is why wavelet and JPEG compression work. Natural images have
decaying cross-scale correlations; language has hierarchical constituent
structure. Cost is O(d·χ²) with χ small for natural data. Stoudenmire & Schwab,
*Supervised Learning with Tensor Networks*, NeurIPS 2016 **[lit]** trained an MPS
directly on MNIST, so the encoder exists and works.

## Noise: deferred, not dismissed

Two halves, both honest:

- **If the label rule is quantum-hard, fault tolerance is required anyway.** Then
  noise is a timeline question and it is correct not to design around NISQ.
- **But nothing in v121–v131 carries a noise model**, and mitigation is not free —
  its sampling overhead grows with depth. `twirl_cal` is exactly that machinery,
  which is a useful symmetry: the calibration line is what makes the QML line run.

Do not let noise decide the architecture; do not claim a result before running
one.

## The three flows

### Classification — works today, advantage conditional

    x -> descriptors (low latent dim)
      -> linear encoder, log|D| qubits          [cheap, v125]
      -> U(theta)
      -> single Pauli readout -> f(x)           [G = 1]
      -> loss vs y
      -> QLTO gradient, G circuits, log M ancillas
      -> classical update

This is precisely what `qlto_qml` runs **[measured]**. The pipeline is not the
open question. Advantage lives in one place only: whether `y(x)` is quantum-hard.
MNIST — no. Molecular property prediction — possibly.

### Imaging — the trainability guarantee lives here, the encoding problem does too

    image -> [ENCODER: not supplied by QCNN]
          -> conv / pool, hierarchical, O(log N) parameters
          -> readout

That is a **QCNN** — Cong, Choi & Lukin, *Quantum Convolutional Neural Networks*,
Nature Physics 15, 1273 (2019) **[lit]** — and it carries a property almost
nothing else in QML has: **QCNNs provably avoid barren plateaus** — Pesah, Wang,
Cerezo, Sharma, Sone & Coles, PRX 11, 041011 (2021) **[lit]**.

  **CORRECTED. An earlier version of this section said "the hierarchy fixes it",
  meaning the encoding. It does not.** QCNN's tree is the PROCESSING structure;
  it acts on a state already in the register and supplies no encoder. Amplitude
  encoding still costs Theta(2^N) — the wall v125 measured, unchanged by what
  comes after it — and angle encoding still fits only N features on N qubits, so
  an 8-qubit QCNN sees 8 numbers, not an image. A TTN *can* be used for state
  preparation, but that is a separate construction bolted on the front, and it is
  the one v127 measured: cheap only at low latent dimension, and dequantizable
  exactly where it is cheap.

  **AND ITS NATIVE DOMAIN HAS NO ENCODING PROBLEM AT ALL.** QCNN was introduced
  for quantum phase recognition and error correction, where the input is a
  physical state already in the register. QCNN-on-images is a later adaptation,
  and the adaptation is where the wall gets imported — the same conclusion this
  Part reaches from the input-model side.

  **TENSION WORTH RECORDING [design].** QCNN's O(log N) parameters, which buy the
  trainability guarantee, are also what limit QLTO's 2M leverage. Trainability and
  large-M leverage pull against each other, so QCNN + QLTO is a SMALLER win than
  QLTO on a dense ansatz. The two ideas do not simply add.

### The three axes are one system, not a menu

Gradient, data encoding, and the optimisation step are not alternative places to
look for an advantage. **Every architecture above needs all three, and a gap in
any one of them is fatal on its own:**

| axis | what it costs if unsolved | QLTO component |
|---|---|---|
| **data encoding** | Theta(D) gates; arbitrary data unreachable | v125/v127 measured the wall; not solved |
| **gradient** | 2M circuits per step | the SENSING oracle — solved, v6/v132/v133 |
| **optimisation step** | descent stalls in traps, or has no gradient to follow | the WALK — mechanism real (v99), construction unresolved |

QCNN is the clean illustration: it has a trainability guarantee (step), needs a
gradient (v133 measured QLTO supplies one at G=1, one circuit), and has the
encoding problem in full. Three axes, one architecture, and only the middle one
is closed.

  This is why the sensing oracle and the walk must not be conflated. The sensing
  oracle's job is COST and v89 proves it cannot help a landscape; the walk's job
  is the LANDSCAPE and it says nothing about cost. Applying either result to the
  other's question is a category error, and Part V's substrate rule is the same
  discipline applied to cost comparisons.

### LLM — blocked on representation, and no measurement trick reaches it

QLTO's log M register handles a billion parameters in ~35 ancillas. That part is
genuinely fine. **But the circuit must physically contain M gates.** A
billion-gate circuit does not exist.

DiffusionBlocks (arXiv 2506.14202v4) helps structurally — one block at a time, so
M_block not M_total — but a block of a real model is still ~1e8 parameters. Still
fatal.

The realistic shape is **quantum as a subroutine, not as the model**: a quantum
kernel layer, a sampling primitive, or an attention-like inner product whose
FEATURE MAP is quantum-hard. The transformer stays classical; one operation
inside it does not.

## The pattern across all three

Encoding is solvable for all three via trees. Gradient is solvable for the first
two via QLTO. **Representation is what fails at scale, and it fails on the
PARAMETER axis** — the exact dual of what Part III measured failing on the data
axis. Both times: interrogation compresses, representation does not.

So the near-term target is where the axes align: **modest M, cheap hierarchical
encoding, and a label rule that is quantum-hard by physics.** Molecular and
materials property prediction sits in that intersection. Imaging sits adjacent
with a trainability guarantee but less QLTO leverage. LLMs do not sit in it.

## What is genuinely open

Not "is advantage possible" — Liu et al. settled that. The live questions:

1. **Is the natural instance separated, or only the artificial one?** Chemistry
   labels are quantum-hard in the WORST case. Real molecules may sit in an easy
   subclass — which is why classical ML for chemistry works as well as it does.
   No chemistry learning task has a proven separation.
2. **Does the separation survive LEARNING?** Hard to compute is not hard to learn;
   a hard function can have an easily-learnable approximation. This gap is where
   most QML advantage claims die.
3. **Does QLTO's gradient survive noise?** Zero noise models in v121–v131. This is
   the nearest gap and the only one of the three that is ours to close.

## Correction to Part III's walk verdict

Part III records the walk as blocked by Tregenna/Keating/Yin. That verdict applies
to **FOUND** landscapes — an optimiser starting at random theta_0 in a disordered
loss landscape. It does not apply to **CONSTRUCTED** ones, where you build the
tree and make your start its root, satisfying Tregenna's symmetry requirement by
construction and admitting no disorder for Keating/Yin to attack.

**Constructed landscapes fail for a different reason [design].** The glued-trees
separation is **informational, not geometric**: it needs the graph hidden behind
an oracle with random vertex names. A structure you construct is a structure you
know, and a structure you know is one you can navigate. Building it is what tells
you the answer.

  So stacking hierarchies cannot manufacture the separation — it moves you along
  the tree/MPS row (classically contractible, no advantage) and, pushed further,
  into the PEPS row (#P-hard, hard for everyone). The glued-trees row is not a
  geometry and cannot be reached by choosing one.

**And the grouping idea is still right, in the right place [design].** Grouping
interchangeable parameters constructs an automorphism group; the glued-trees
speedup IS a commutant phenomenon — the walk Hamiltonian commutes with the graph
automorphisms, so an exponential graph collapses to a (2d+2)-dimensional column
walk, and that projection is a commutant projection, the same move `twirl_cal`
makes with a Pauli group. **This merges the walk axis and the commutant axis into
one path.** Where it pays is where the hiding is free: an unknown device
Hamiltonian, whose symmetry group you can know without knowing H, and whose
commutant projection the device performs for you by time-averaging.

---

# Part V — the accounting rule: substrate vs. invocation count

A rule for reading every cost comparison in this file.

## The rule

> **A variational method's cost factors into the cost of the SUBSTRATE — the
> sensing primitive every competing method must also run — and the NUMBER OF
> INVOCATIONS of it. QLTO changes only the second. Compare on the second.**

    method                      invocations / epoch      cost per invocation
    QLTO                        G   (1 with QFT sensing)   C_substrate
    parameter-shift             2MG (2M with QFT sensing)  C_substrate

`C_substrate` appears on both lines and cancels. A large substrate cost bounds
**the application** — which problems are reachable at all — and says nothing
about which method to use on the problems that are reachable.

## Three consequences that follow directly

**Pauli grouping helps both arms identically.** C_PS/C_V6 = 2MG/G = **2M**, so G
cancels and better grouping — general commuting, minimum clique cover (v119,
v120) — does not widen V6's advantage. It lowers the substrate for everyone.

**A large G bounds the problem, not the method.** At N=100, extrapolating v30's
G ~ N^4.24 from its N=6 anchor (G=75) gives 75 x (100/6)^4.24 ~ **1.1e7 groups**.
That is fatal for VQE-style chemistry — and it is fatal for parameter-shift 2M
times over. The wall is measurement, not coherence, so more circuit depth does
not move it.

**Phase sensing and precision QPE are different substrates with different cost
models.** Published qubitization estimates (~lambda/eps, 1e9-1e10 Toffoli for
chemical accuracy) describe full ground-state energy estimation: a precision
ladder of repeated controlled evolutions. V3's QFT sensing is a single controlled
evolution at modest tau, read as a relative phase. The estimates do not transfer
between them, and in either case the number is substrate.

## What the QFT sensing route gives

Phase sensing uses controlled `e^{-iHt}` rather than a Pauli decomposition, so **G
does not appear at all**. This is a universal fix — it removes G for every method
— and it moves the whole cost into depth. With G gone the comparison reaches its
cleanest form:

| | invocations per epoch |
|---|---|
| **QLTO + QFT sensing** | **1** — all M gradient components and the energy record |
| parameter-shift + same sensing | 2M |

Nothing cancels the 2M. This is QLTO's strongest position, and it exists because
the substrate has been factored out of the comparison rather than charged to one
side of it.

## Resource arithmetic at 100 qubits / 200k gate depth

The question splits along the same line: does the **substrate** run (universal),
and what is the **invocation count** (what QLTO changes).

### QML classification — clears with two orders of headroom

    data register        10 qubits -> |D| = 1024,   prep ~1k gates
    system               78 qubits
    QLTO design register 12 qubits   (ceil(log2(M+1))+1, M ~ 2000)
    encoder              78 x 10 CRY  ~ 1.6k CX
    ansatz               efficient_su2(78, reps=10),  M ~ 1716, depth ~1e3

A few thousand gates of depth against 200k. G = 1 structurally, since a QML
readout is a single Pauli, so:

    QLTO             3 circuits/epoch
    parameter-shift  2M ~ 3400 circuits/epoch        ~1000x, growing with M

**[design]** — this allocation is arithmetic. The largest configuration actually
executed is N_sys = 3, |D| <= 32 (Part III).

### Chemistry — substrate-bound, universally

VQE-style: invocation count is G ~ 1e7 at N=100. Depth is not the binding
constraint, so 200k does not help.

QFT/QPE sensing: G vanishes, and the cost moves entirely into depth. **The N^4
does not disappear — it relocates from circuit COUNT to gate count per circuit.**
v30 measured the QPE-path depth as `sum_a r_a * T * 16` gates: **266,880 depth,
18,682 us at N=12**. That is already above a 200k budget at twelve qubits, so
chemistry via phase sensing is substrate-bound well before N=100, and the earlier
reading of 200k as sufficient here was wrong.

  The divide underneath it: **sampling can spend its total evolution time in
  pieces, each shorter than T2; QPE needs the same time contiguously.** T2 decides
  which is available. That one fact explains V5's QPE death (survival 0.098), why
  the twirl construction cannot use a phase readout (v106), and why amplitude
  estimation was demoted — three faces of one wall.

  Iterative QPE needs one ancilla, so qubit overhead is not the constraint;
  coherent depth is, and only that.

Precision enters linearly, and loose precision is where this line has always been
strongest — Part I: *"LOOSE precision, which is exactly the regime where V3 beats
Gilyen et al., eps > 1/sqrt(d)."* Screening, ranking and relative energies sit
there; chemical accuracy on large active spaces does not.

**And this bounds the whole line honestly.** QPE is Heisenberg-limited,
eps ~ 1/T_evolution against sampling's eps ~ 1/sqrt(shots) — a quadratically
better rate. **QLTO is a NISQ-regime construction and its value expires when
error correction arrives**, at which point QPE is simply used and none of this
machinery is needed. Worth stating plainly, because it fixes which claims are
worth defending: near-term cost reduction, not asymptotic advantage.

## Status

What this project has is a **working algorithm, measured on circuits**: V6's
log-width design register, `qlto_qml`'s three-circuit epoch and its self-driven
descent, `twirl_cal`'s calibration — all tier A. The foundation is optimizable
from here.

One open item in the evidence ledger: the QPE-plus-design-register composition —
one invocation per epoch for the whole gradient and energy record — is recorded in
Part I from the V5 era and has not been re-run at tier A the way v121-v131 were.
If the QFT sensing route is the direction, that re-measurement comes first.

---

# Part VI — the walk step, derived; and why the earlier walk verdict is misleading

Parts I–V treat "the walk" as one object with one verdict. It is not. It is one
point in a taxonomy of implementable walk graphs, and specifically the degenerate
one. This part derives what the step actually computes, states the limit as a
theorem rather than a measurement, and separates what was measured from what was
assumed.

## What the step is, exactly

The param register holds `n` qubits, `H^{⊗n}` puts them in uniform superposition
over the `2^n` vertices `x ∈ {−1,+1}^n` of the box `θ_c + R·x`. `_execute_walk`
then applies, per step, `crz(αᵢ)` and `crx(β)` on each param qubit, so the
generator is

    H_walk  =  β(t) Σᵢ Xᵢ  +  Σᵢ αᵢ(t) Zᵢ ,        αᵢ ∝ gᵢ

with the schedule `γ = s·π·δt` ramping **up** and `β = (1−s)·π·δt` ramping
**down**, `s = (step+½)/k_steps`.

**Two identifications make the whole thing legible.**

**The mixer is the hypercube.** For the `n`-cube, `Xᵢ` flips bit `i`, which *is*
the edge relation, so `A = Σᵢ Xᵢ`.

  **SIGN, corrected in Part IX.** This line read `L = A − nI` "differs by a
  global phase". The graph Laplacian is `L = nI − A`; `A − nI` is `−L`, and a
  sign flip is not a global phase. The Schrödinger kinetic term is
  `−h²∆ = +h²(D − A)`, so the mixer enters as `−A` up to a constant. Building
  the walk from this line gives the wrong dispersion, and it did — v136's
  first two rounds used `e^{−iAδt}` and measured a slope of −2 where the
  theory predicts −1.
`β Σᵢ Xᵢ` is a continuous-time quantum walk on the box vertices, in the sense of
the book's Eq. 1.49–1.52.

**The potential is the linearised objective.** `Σᵢ αᵢ Zᵢ` acting on `|x⟩` gives
`(Σᵢ αᵢ xᵢ)|x⟩`, and with `αᵢ ∝ gᵢ` that is `∝ R Σᵢ gᵢ xᵢ`, the first-order term
of `E(θ_c + Rx)`. So the diagonal field **is** the linearisation of the objective
over the box.

    the walk = CTQW on the hypercube, in a potential given by the measured gradient

## Why it works as an optimiser

At `s=0` the Hamiltonian is pure mixer and the prepared state is the uniform
superposition — every vertex equally weighted. At `s=1` it is pure potential and
the ground state is the vertex minimising `Σᵢ αᵢ xᵢ`, i.e.

    xᵢ = −sign(gᵢ)      hence      θ ← θ_c − R·sign(g)

**The adiabatic limit of the walk is signed gradient descent with step R.**

  **WHICH END OF THE SPECTRUM — see Part IX.** V3 tracks the TOP, not the
  ground state. `build_w_gate` maps `|0> -> c−R` and `|1> -> c+R`, so the
  displacement is `−R⟨Z⟩` and the descent move needs the MAXIMUM of
  `+(γ/2)Σg_iZ_i`; `h(param)` prepares all-`|+⟩`, which is the MAXIMUM of
  `+βΣX_i`. Two sign flips cancel, so the endpoint stated here is right and
  the description around it is inverted. A reimplementation that follows the
  description rather than the code gets the sign wrong. That
is the mathematical statement of why the step works at all: it is an annealing
interpolation whose endpoint is the descent move, so it inherits descent's
correctness and can only differ by being non-adiabatic. Everything the walk could
offer *above* descent comes from running at finite `T`, where the state does not
track the instantaneous ground state and can spread and interfere.

## The limit, as a theorem

Every term of `H_walk` acts on a single, distinct qubit:

    H_walk = Σᵢ ( β Xᵢ + αᵢ Zᵢ )

so the terms **mutually commute**, and

    exp(−i H_walk t)  =  ⊗ᵢ exp(−i(β Xᵢ + αᵢ Zᵢ) t)

> **THE WALK AS BUILT IS A TENSOR PRODUCT OF n INDEPENDENT SINGLE-QUBIT
> ROTATIONS. It generates no entanglement, and it is classically simulable in
> O(n).**

**Verified numerically** — `‖exp(−iH_walk t) − ⊗ᵢ exp(−i(βXᵢ+αᵢZᵢ)t)‖` and the
entanglement entropy across the middle cut, from a product input, over random
`α`, `β`:

    n     ||full - product||     entanglement entropy
    3         5.44e-16                0.00e+00
    4         1.84e-15                0.00e+00
    5         3.61e-15                0.00e+00
    6         4.63e-15                0.00e+00

Machine zero and exactly zero entropy at every size. This is not a conjecture,
and **v99 PART 5 already measured it** and read it as an
optimisation: the `k_steps` loop collapses to one controlled-SU(2) per qubit,
`‖U_loop − U_1‖ = 3.5e-15` at every `k`, 76.8× depth saving at `k=64`. That
collapse *is* the separability. The file recorded the consequence for gate count
and not the consequence for advantage.

**Two independent causes, and either alone is sufficient:**

- the **mixer** factorises — the hypercube is the Cartesian-product family, the
  one family in Wang's taxonomy that splits as `e^{−i(H₁⊕H₂)t} = e^{−iH₁t} ⊗
  e^{−iH₂t}`
- the **potential** is degree-1 — `Σᵢ αᵢ Zᵢ` has no `ZᵢZⱼ` terms, and minimising a
  linear function over the hypercube is separable per coordinate

So the construction is separable twice over.

## What that does to the earlier verdict

v99 measured the walk "roughly breaking even against a cheap classical decode of
the same shots" and left open whether the coherent step pays for itself.

**That was never open.** A product of single-qubit rotations cannot beat a
classical computation, because it *is* one. The measurement was correct and the
framing made it look like an empirical near-miss that better tuning might close.
It is a theorem, and no amount of tuning moves it.

The same reading corrects two more entries:

- the drift-phase **wrap** (`hypot`/`atan2` saturating) is a real effect and is a
  bounded nonlinearity, but it is a nonlinearity applied *per qubit* inside a
  product — it cannot create correlation between parameters
- v99's **PART 4** tunnelling result — classical success collapsing 1.00 → 0.00
  while quantum transmission stays flat — was measured on a SYNTHETIC spike
  potential `E(w) = w + h·exp(−(w−c)²/2σ²)`, which is a function of Hamming
  WEIGHT and hence **not** degree-1. That potential is not separable, which is
  precisely why tunnelling appeared there. **The mechanism was demonstrated on a
  potential the actual walk cannot represent.**

That last point is the crux: PART 4 and PART 5 of the same file describe
different constructions, and the advantage lives in the one that was not built.

## Where the improvement is

Separability has two causes, so there are two independent ways to break it, and
either is a drop-in at the same circuit position.

**Change the mixer.** Wang's taxonomy (CQCWS1_19335, slides 5→end) lists the
efficiently implementable families, and the hypercube is the only one that
factorises:

| graph | eigenstructure | cost on 2ⁿ vertices | factorises |
|---|---|---|---|
| hypercube *(current)* | `Σ Xᵢ` | n gates | **yes** |
| complete `K_N` | `Λ = {N,0,…,0}`, 2 distinct | `2log₂N + 1` ≈ 2n+1 | no — Grover diffusion |
| cycle / Möbius | `Λ = 2cos(2kπ/N)` | QFT, `O(log²N)` | no |
| Paley / SRG | ≤ 3 distinct eigenvalues | QFT-diagonalizable | no |
| glued trees | symmetry family | tree construction | no — the proven-separation one |

The complete graph is the cheapest escape: two distinct eigenvalues, ~2× the
hypercube's gate count, and it is the mixer QWOA already validates. Note the
notes' *"Why Grover is not the next step"* closes hard-threshold ORACLE SEARCH and
is itself flagged partly wrong — Grover-as-diffusion-mixer is a different use and
is not refuted ground.

**Or change the potential.** Add degree-2 terms `Σᵢⱼ αᵢⱼ ZᵢZⱼ`, which makes the
diagonal non-separable even on the hypercube. The machinery is partly present:
V6's design register at `design_resolution=4` returns degree-1 Walsh coefficients,
and resolution 5 (`_resv_cols`, v90) exists specifically to lift 3- and 4-term
confounding — the same construction that would supply pairwise coefficients.

  Either change alone makes `exp(−iH_walk t)` entangling. Both is strictly better,
  and the diagonal potential composes with any mixer through Wang's
  commuting-graph rule, so the two are independent knobs.

## What is measured and what is not

**Measured:** the separability (v99 PART 5, 3.5e-15); the tunnelling mechanism on
a weight-dependent potential (v99 PART 4); the thin-barrier oracle cost being flat
at 1045 gates regardless of width (v118b).

**Not measured, and now the actual open question:** whether a non-factorising
mixer, or a degree-2 potential, gives a step that beats classical descent on a
landscape where descent fails. Every prior walk-vs-classical number in this
project (v20, v53, v53b, v53c, v99) was taken on the separable construction and
therefore measured a classical computation against a classical computation.
**Those comparisons should not be quoted as evidence about quantum walks.**

---

# Part VII — the bridge: what would actually be classically unsimulable, and how QLTO reaches it

Part VI showed the walk as built is a tensor product and therefore classically
free. This part asks the converse question mathematically: **what is the cheapest
change that lands outside every known classical simulation class, and does QLTO's
existing machinery reach it?**

## The simulability ladder

A quantum evolution `exp(-iHt)` is classically tractable if it falls into any of
these. To be hard it must escape **all** of them.

| class | tractable because | what escapes it |
|---|---|---|
| **product / separable** | terms act on disjoint qubits | any coupling term at all |
| **free fermion (Gaussian, matchgate)** | JW maps to quadratic `c†ᵢcⱼ`; Valiant, Terhal–DiVincenzo | quartic terms — `ZᵢZⱼ` is `n̂ᵢn̂ⱼ`, interacting |
| **Clifford** | Gottesman–Knill | generic rotation angles (automatic here) |
| **low entanglement (MPS)** | area law, bond dimension poly | volume-law entanglement |
| **low treewidth** | tensor contraction poly in treewidth | dense coupling graph |

## Where each candidate walk sits

Writing `H_walk = (mixer) + (potential)`:

| mixer | potential | mid-cut entropy (n=8, max 4) | class |
|---|---|---|---|
| hypercube `Σ Xᵢ` | degree-1 | **0.0000** | product — **where QLTO is** |
| hypercube | **degree-2** | **1.6167** | escapes product |
| circulant (cycle) | degree-1 | 0.1000 | weak |
| circulant | **degree-2** | **2.9858** | near volume law |

**[measured]** — exact `expm`, random coefficients, tier C (operator identity, no
circuit).

  **AND THE MIXER DECIDES THE SEMICLASSICAL LAW — Part IX.** The conclusion
  below is right and its stated reason is not the load-bearing one. A
  degree-1 potential has exactly ONE minimum on any connected graph, so no
  mixer rescues it — that is the real argument. But the mixer choice decides
  something this table cannot see: v136 measured the cycle register under
  `ΔE = e^{−S₀/h}` (slope −0.998) and the hypercube under `ΔE = e^{−n·S̃}`
  (r = −0.99994), so the hypercube's tunnelling degrades exponentially in the
  PARAMETER COUNT and the cycle's does not. And on circuits a circulant mixer
  breaks separability with a degree-1 drift alone. Far from the wrong knob.

> **THE POTENTIAL DEGREE IS THE BINDING CONSTRAINT, NOT THE MIXER.** A degree-2
> potential entangles even on the hypercube; a non-factorising mixer with a
> degree-1 potential barely does. Part VI emphasised the mixer taxonomy; that was
> the wrong knob.

  **HOW FAR THAT CARRIES.** The table is tier C, and R1 admits tier C for scoping
  only — so this is a hypothesis about the knob, not a headline. Two specific
  reasons to hold it loosely: the coefficients are RANDOM, and random degree-2
  couplings maximise entanglement by construction where a sparse structured
  potential need not; and the rows are not comparable, the circulant being a cycle
  on `2ⁿ` vertices against a hypercube on `n` qubits. Part VIII gives the mixer a
  second job — fixing the measurement grading — on which it is not the wrong knob
  at all.

The reason is structural. `Σᵢ αᵢ Zᵢ` is a sum of commuting single-qubit terms, so
whatever the mixer, the potential contributes no correlation. `Σᵢⱼ αᵢⱼ ZᵢZⱼ`
under JW is `n̂ᵢn̂ⱼ` — **quartic in fermions**, so it escapes the Gaussian class
too, which no choice of mixer alone does.

## What the object then is

With a circulant mixer (diagonal in the Fourier basis) and a potential diagonal in
the computational basis, Trotterising gives

    exp(-iHt) ≈ ( F† e^{-iΛδ} F · e^{-iVδ} )^k

which is exactly the **split-operator method**: kinetic energy diagonal in
momentum, potential diagonal in position. So the walk is literally **a quantum
particle propagating in the loss landscape**, and the register is the position
grid.

  Representation cost: `d·log N` qubits against a classical grid's `N^d` points.
  That is an exponential separation **in representing the dynamics** — and it is
  the wrong claim to lean on, because no classical optimiser grids the landscape.
  Being exponentially better at simulating your own process is not an advantage at
  the task. This is the same trap Part V's substrate rule guards against.

Equivalently, in the computational basis with an all-to-all degree-2 potential,
`H_walk` is a transverse-field model with dense `ZᵢZⱼ` couplings — **quantum
annealing on a dense QUBO**, whose ground-state problem is NP-hard.

## The real advantage claim, and its precondition

The defensible claim is not representation. It is the one v99 PART 4 measured:

    classical annealing over a barrier   cost ~ exp(height)
    quantum tunnelling through it        cost ~ exp(width)

measured with classical success collapsing 1.00 → 0.00 over height 2 → 20 while
quantum transmission stayed flat. **And PART 4's potential was `E(w) = w +
h·exp(-(w-c)²/2σ²)`, a function of Hamming weight — which is degree-2 and above,
hence exactly the non-separable class this Part identifies.** Part VI records that
PART 4 and PART 5 describe different constructions; this is what the difference
was.

So the precondition is a landscape with **tall thin barriers**, and the
construction that can represent one requires a degree ≥ 2 potential.

## The tension, stated exactly

    cheap potential  (degree-1)  →  separable  →  no advantage possible
    rich potential   (degree-2)  →  escapes every class  →  must be MEASURED

Degree-2 means `M(M-1)/2` pairwise coefficients. That is the cost, and it is where
QLTO's existing machinery is relevant rather than incidental.

## Why the design register fits this exactly

`_design_spec` at `design_resolution=4` returns degree-1 main effects with
degree-2 **confounded** — which is precisely why the walk's potential is degree-1
today. Resolution V exists already (`_resv_cols`, motivated by v90's measurement
that the shipped Gray columns have `min|S| = 3` and the cosine falls to 0.714 at
M=16) and its two conditions are stated in the code:

    (a) no column equals the XOR of two others   -> no 3-term relation
    (b) all pairwise XORs are distinct           -> no 4-term relation

Condition (b) is exactly what makes pairwise effects separately identifiable. And
the widths line up by parameter counting:

    resolution V register width   m ~ 2 log2(M)        (v90 measured m_row 6, 8, 8)
    design rows available         2^m  =  M^2
    degree-2 coefficients needed  M(M-1)/2  <  M^2      ✓

> **The resolution-V design register has exactly enough rows to identify the
> degree-2 model, at double the register width — still logarithmic in M.**

  **COUNTING IS NECESSARY, STRENGTH IS SUFFICIENT.** Rows ≥ coefficients does not
  give identifiability; what does is that the design reproduce the product
  measure's marginal on any `d+1` coordinates, i.e. STRENGTH `d+1`. The counting
  above happens to survive — Rao's bound for strength 4 is `1 + M + C(M,2) = 667`
  at `M = 36`, under `M² = 1296` — but that is the argument that has to be made.
  Part VIII states it and gives the cost of the construction.

## The algorithm this composes into

    1. sense at resolution V   ->  local QUADRATIC model (gradient + pairwise), G circuits
    2. load it as the walk potential  ->  Sum_i a_i Z_i + Sum_ij a_ij Z_i Z_j
    3. anneal with a mixer            ->  quantum annealing on a box-constrained QUBO
    4. read a vertex                  ->  the step

This is a **trust-region method whose subproblem is solved quantumly**. Classical
trust-region builds the same local quadratic model and must approximate the
box-constrained subproblem, which is NP-hard. Here the subproblem is solved on the
same register that measured it.

  **[design]** — not built. Steps 1 and 4 exist in V6 and V3 respectively; step 2
  is a change of what the drift gates encode, from `crz(gᵢ)` to `crz(aᵢ)` plus
  `crzz(aᵢⱼ)`; step 3 is the existing annealing schedule.

## What this does NOT establish, and it is most of it

- **NP-hard subproblem ≠ quantum speedup.** Quantum annealing has no proven
  advantage on generic NP-hard instances. The `exp(width)` vs `exp(height)`
  separation is for *specific barrier shapes*, not all landscapes.
- **Approximate subproblem solutions may suffice.** Classical trust-region methods
  work well with cheap approximate steps, so beating the exact subproblem is not
  obviously worth anything.
- **The shot cost of `M²` coefficients is unmeasured.** The design register
  supplies the *circuits* in `G`; it says nothing about the shots needed to
  resolve `M²` numbers to useful precision, and that is the number that would
  decide the method.
- **Entanglement is necessary, not sufficient.** Every row above measures
  entanglement, which is a precondition for hardness and not a proof of it, and
  certainly not evidence of optimisation quality.


## Three complexity barriers, and the one survivor

The construction above is worth building only if some target could in principle
show an exponential separation. This section maps where that is possible, because
three separate barriers rule out most of the obvious targets — and they rule them
out by theorem rather than by difficulty.

### Barrier 1 — graph properties admit no exponential speedup

Ben-David, Childs, Gilyén, Kretschmer, Podder & Wang, *Symmetries, Graph
Properties, and Quantum Speedups* (SIAM J. Comput.) **[lit, CITATION UNVERIFIED —
recalled, not read; check before quoting]**: for any function invariant under
vertex permutations — i.e. any **graph property** — quantum gives at most a
polynomial speedup.

**Every NP-hard graph problem is a graph property.** Max-Cut, colouring, TSP,
clique are permutation-invariant by construction. So "exponential quantum-walk
speedup on an NP-hard graph problem" is closed by theorem.

  **And this is exactly why glued trees escapes.** Its hardness comes from RANDOM
  VERTEX LABELLING hiding the structure, which is precisely what makes it *not* a
  graph property. Hand over the graph explicitly and traversal is BFS. The
  separation lives in QUERY complexity, not time complexity on explicit inputs —
  the same observation Part IV makes about the hiding being informational rather
  than geometric.

### Barrier 2 — NP is not believed to be in BQP

An exponential speedup on an NP-hard problem gives NP ⊆ BQP, which is widely
disbelieved, and BBBV proved Grover's quadratic is optimal in the black-box
setting. Nothing about a walk changes either fact.

### Barrier 3 — the DLA collapse: trainable ⟹ simulable

This is the one that bites *inside* variational quantum computing, and it is the
sharpest statement of the QML dilemma this project keeps meeting.

Ragone, Bakalov, Sauvage, Kemper, Ortiz Marrero, Larocca & Cerezo, *A Lie
algebraic theory of barren plateaus for deep parameterized quantum circuits*,
Nature Communications 15 (2024) **[lit]** gives an **exact expression for the
variance of the loss in terms of the dimension of the circuit's dynamical Lie
algebra**, unifying every known source of barren plateaus — expressiveness, input
entanglement, observable locality, noise — under one framework, and resolving the
standing conjecture linking loss concentration to `dim(g)`.

Set beside Bridi, Lim, Pira, Santos, Marquezino & Adhikary **[lit]**, whose
Theorem 1 gives `dim(g_QWOA) ≤ m² + 1` and whose Theorem 3 concludes that any
NPO-PB problem outside BPPO requires QWOA overparameterization:

    polynomial DLA  ->  no barren plateau     ->  trainable
    polynomial DLA  ->  small reachable set   ->  classically simulable

> **Trainability and classical simulability are controlled by the same quantity.**
> Every construction with a *proven* trainability guarantee — QCNN, QWOA, shallow
> local-cost ansaetze — buys it by keeping the DLA polynomial, which is the same
> condition that makes it classically tractable. Exponential DLA gives the
> expressivity and takes the gradient away.

  **THE SECOND ARROW IS NOT A THEOREM.** `polynomial DLA → no barren plateau` is
  what Ragone et al. give. `polynomial DLA → classically simulable` is not: a
  low-dimensional reachable set is necessary for Lie-algebraic simulation
  (Somma–Ortiz–Knill) but not sufficient — the algebra must also be efficiently
  REPRESENTABLE, with the initial state and the observable inside it. Written as a
  bare implication the barrier is stronger than the cited results support.

That explains, in one statement, every specific finding elsewhere in these notes:
why QCNN is both trainable and simulable (Part IV), why Bridi's Theorem 3 demands
overparameterization, and why this project's own walk factorises (Part VI).

### The survivor

| route | barrier |
|---|---|
| NP-hard graph problem | permutation invariance (Barrier 1) |
| NP-hard generally | NP ⊄ BQP; BBBV (Barrier 2) |
| trainable variational model | DLA collapse (Barrier 3) |
| **simulating quantum dynamics** | **none — BQP-complete by definition** |

The only route no barrier touches is the one that is neither an oracle
construction nor a combinatorial problem. Which is where ground-state chemistry,
`twirl_cal` and the commutant thread already sit — reached here for the fourth
time, now from complexity rather than from cost, input models, or measurement.

**What this means for the trust-region construction above.** It may be a better
optimiser than gradstep, and that is worth measuring. It may not claim an
exponential separation on the QUBO subproblem: Barrier 2 forbids it generically,
and Barrier 3 says that whatever makes the step trainable is what makes it
simulable. **The falsifiable claim at the end of this Part is about beating
gradstep on tall-thin-barrier landscapes — a constant-or-polynomial-factor claim
— and it should never be restated as more than that.**

## The one testable claim to take from this

If a landscape's local quadratic model has tall thin barriers in the sense v99
PART 4 constructed, then a resolution-V QLTO step should beat gradstep by a margin
that grows with barrier height and is flat in barrier width. **That is falsifiable,
it uses only machinery that exists, and it is the experiment Part VI's correction
leaves open.**


---

# Part VIII — the sensing register is a walk, and the radius is a second axis

Parts VI and VII treat "the walk register" and "the design register" as two
objects. They are one register with one eigenbasis, and identifying them
reorganises both: the mixer Part VI wants to change for *separability* is the same
operator whose spectrum decides *what a single shot can measure*.

## The identification

`qc.h(param)` prepares the uniform superposition over `2ⁿ` box vertices — which is
the `s=0` ground state of `Σᵢ Xᵢ`, Part VI's hypercube mixer. The design rows are
its vertices. And the decode reads Walsh functions, which Part VI already notes
are its eigenvectors:

    Σᵢ Xᵢ |χ_S⟩ = (n − 2|S|) |χ_S⟩

> **The hypercube walk's energy IS the Walsh degree.** "Resolution IV vs V" and
> "the mixer's spectrum" are the same number written twice.

That is why degree is the natural grading of the sensing problem — not a modelling
choice but a consequence of which mixer prepares the register.

## Therefore the mixer taxonomy is a table of measurement structures

Part VI reads Wang's taxonomy as a list of ways to break separability. It is also,
and independently, a list of measurement bases:

> The mixer's eigenbasis is the measurement grading. Changing the walk graph
> changes which functionals of the landscape one shot can read.

| mixer | eigenbasis | one shot grades by | Hamming |
|---|---|---|---|
| hypercube `Σ Xᵢ` *(current)* | Walsh `χ_S` | degree `\|S\|` | local |
| complete `K_N` | uniform ⊥ rest | 1 bit | — |
| circulant on `Z_N` | Fourier `e^{2πikx/N}` | frequency `k` | **nonlocal** |
| Paley / SRG | ≤ 3 eigenvalues | 2 bits | — |

Part VII values the circulant for escaping the product class. The sharper reason
is that its eigenbasis grades by **momentum rather than Hamming weight**, so it is
not subject to the degree-spreading floor `p ≥ n/(2·deg)` that binds any
Hamming-local mixer — which is also QWOA's escape. Same fact; stated as a
measurement property it says what to build and why.

## What the current register measures, exactly

Every parameter enters through a generator with `P² = I`, so the landscape's
Fourier support is exactly the grid `{−1,0,1}^M`:

    E(θ) = Σ_{k ∈ {−1,0,1}^M} c_k e^{i k·θ}

Pushing the `±R` design through it, over the full factorial:

    α_j = sin R · Σ_k c_k (i k_j) e^{i k·θ} (cos R)^{|k|₀ − 1}

    α_j(R) / sin R  =  Σ_{d ≥ 1} (cos R)^{d−1} · D_j^{(d)}(θ)

with `D_j^{(d)}` the weight-`d` part of `∂_j E`, and `∂_j E` its value at
`cos R = 1`. Nothing is truncated: `sin(Rk_j) = k_j sin R` and
`cos(Rk_l) ∈ {1, cos R}` hold exactly on the grid.

**Read as a filter.** The estimator applies `A : c_k ↦ cos(R)^{|k|₀−1} c_k`, a
diagonal low-pass in Walsh weight. QLTO does not measure a biased gradient — it
measures the *exact* gradient of a low-pass-filtered landscape. `A⁻¹` multiplies by
`sec(R)^{|k|₀−1}`, unbounded in weight, which is why exactness costs and where the
cost lives.

## The second axis: the radius grades by degree

The vertex axis says *which parameters*. The radius says *which degree*, and the
two are independent. `sin(aR₀)·cos^{d−1}(aR₀)` has frequency content in `a` of
exactly `{1, 3, …, d}` — maximum frequency `d`. So a radius register of
`⌈log₂ 2d_max⌉` qubits — **3 qubits for degree 4** — carrying controlled rotations
`RY(2R₀·2^a·σ_j)` admits two readouts:

- **measured** in the computational basis: labelled multi-radius in ONE circuit
  rather than `k`, degree separation done classically
- **inverse-QFT'd**: the degrees separate as a phase, coherently

The ladder is structurally the one already built in `_qpe_template`.

    axis 1 — vertex / σ  →  which parameters  →  log₂M qubits
    axis 2 — radius / R  →  which degree      →  log₂d qubits

**[design]** — the radius register is not built. The identity it rests on is
measured below.

## Measured (v135)

**[measured, tier B]** — `Statevector`, exact amplitudes, no sampling. R1 admits
tier B for exactness identities; **no number in this section is an accuracy or
cost figure and none may be quoted as one.**
Log: `supplement/results/v135_support_and_degree_axis.log`.

Support claim — second-harmonic content of `E(θ_j)`, `efficient_su2` + Heisenberg,
9-point fit including `cos 2t`, `sin 2t`:

    ansatz            M   tied   |1st harm|   |2nd harm|
    N=4 reps=1       16      0      1.7137     8.25e-16
    N=4 reps=2       24      0      1.6118     6.66e-16
    N=6 reps=1       24      0      1.8623     9.04e-16
    N=6 reps=2       36      0      1.4531     5.55e-16

Machine zero. `supp(c) ⊆ {−1,0,1}^M` holds for the ansatz V6 actually runs, and no
ansatz in use ties a parameter across gates — a tie would admit frequency `±2` and
void every expansion above.

Degree separation — full `2⁶` factorial over 6 active parameters, all shifted
simultaneously at 6 radii, `α_j(R)/sin R` extrapolated in `cos R` to `cos R = 1`:

    param   exact ∂_jE      extrapolated       err
      0      +0.099129       +0.099129      2.8e-16
      1      +0.642080       +0.642080      2.1e-15
      3      −1.037113       −1.037113      4.4e-16

**Exact gradient from a fully multiplexed design** — every row shifts all six
parameters — agreeing with parameter-shift to `1e-15`. Residuals fall
geometrically with fit degree:

    radii   fit degree   median relative residual
      2          1              3.84e-02
      3          2              4.12e-03
      4          3              9.81e-05

`D` is formally `n−1 = 5` on this block; the coefficients decay regardless, so what
costs is `D_eff`, not `D`. Four radii puts the residual two orders under any
realistic shot floor.

  **NOT TESTED.** The full factorial, not a fractional design — whether a
  strength-4 fraction reproduces this is untouched. And there is no shot noise
  anywhere, so the variance question is untouched too.

## The estimator as a cubature problem

A design is a signed measure `μ` on shift-vectors with kernel
`W_μ(k) = ∫ e^{ik·u} dμ`. The estimator returns `Σ_k c_k e^{ikθ₀} W_μ(k)`, so

    unbiased for ∂_j E   ⟺   W_μ(k) = i k_j   on {−1,0,1}^M

Two costs, and they are different resources: `|supp μ|` → register width;
`‖μ‖_TV` → shot cost `ν ≥ ‖μ‖²_TV`. Every published gradient method is a choice of
`μ`, which is what makes them comparable at all.

**The obstruction, exactly.** Every available kernel is a product
`Πₗ (e^{−iuₗ}, 1, e^{iuₗ})` — middle entry pinned to 1. The target's `j`-factor is
`(−i, 0, i)` — middle entry 0. So `W(0) = 0` requires cancellation, and
cancellation costs total variation. That one line generates parameter-shift's two
points, the nonexistence of a two-point spectator rule, and the three-level fix.

**Coordinate-supported designs pay `ν ≥ M`.** Supports for different `j` are
disjoint, so under a shared shot allocation `f`,

    ν_j = Σ_d (w_d^{(j)})² / f_d  ≥  ‖w^{(j)}‖₁² / F_j  ≥  1/F_j ,   Σ_j F_j ≤ 1

and AM–HM gives `Σ_j 1/F_j ≥ M²`, hence `max_j ν_j ≥ M`. Parameter-shift and both
prior-art methods below pay `Θ(M)` in shots for a structural reason, not an
incidental one.

**Spreading costs total variation, and that forces the radius law.** The exact
per-coordinate rule leaving a spectator undisturbed with all shifts nonzero uses
three levels `(−s, +s, π)` with weights `(b, b, 1−2b)`, `b = 1/(1+cos s)` —
`W(k) = 1` for `k ∈ {−1,0,1}` at every `s`, and `1−2b = −tan²(s/2)`. Its total
variation is `1 + 2ε` with `ε = tan²(s/2)`, so over `M−1` spectators

    ‖μ‖_TV = (1+2ε)^{M−1} ≈ e^{M R²/2}      bounded iff  R ∝ 1/√M

> **V6's radius law is derivable from the measure.** `_radius` justifies
> `radius_exponent = 0.5` by state displacement — "a block of `n` parameters
> displaces the state by about `√n·R`" — and the code flags it shipped and
> unswept. The cubature side reaches the same exponent independently.

## Where the two prior-art poles sit

**[lit]** Lin, Li, Shao, Wang & Wu, *Implementing arbitrary quantum operations via
quantum walks on a cycle graph* (arXiv:2210.14450). A DTQW on a cycle is the
*ansatz*; `M = 4nT` coin parameters are fitted to a target unitary `V`, with the
adjoint identity (their Eq. 10)

    ∂L/∂α_j^(x,t) = Im(⟨Φ^(t)| Σ̂_j^(x,t) |Ψ^(t)⟩)

read by a Hadamard test — one circuit per parameter, `M` per gradient, needing
controlled `U^{T,0}` and controlled `V`. `Σ̂_j` is a position-controlled Pauli
reflection, hence unitary and insertable. Numerics are exact statevector; shots are
not discussed.

> **This is walk-plus-gradient prior art.** Parts III–VII should not be read as
> claiming the combination is unattempted. What is unattempted is steering a walk
> across an UNKNOWN landscape — theirs fits a KNOWN target, which is exactly why a
> Hadamard test applies to it and not to `⟨H⟩`.

Their loss is `1 − Re⟨b|V†U|b⟩`, one measurement setting for a basis input, so
`G = 1`. Their `n=2, T=5` example is `M = 40` circuits per gradient against the
design register's 1.

**[lit]** He, Guang Ping, *Computing the gradients with respect to all parameters
of a quantum neural network using a single circuit* (arXiv:2307.08167). Two
ancillas probabilistically activate one of `2n+1` shifted cost functions per shot,
on the same abelian-group identity V6 requires — his Eq. 4 `RY(s)RY(θ) = RY(θ+s)`
is V6's `_CTRL` condition `op(a)op(b) = op(a+b)`. One circuit, depth `O(n)` rather
than `O(n²)` stacked, and far fewer classical registers.

But each shot informs ONE parameter, so `s' = s(2n+1)` shots are needed, and the
paper says so directly.

> **He optimises circuit count at constant total shots; V6 optimises information
> per shot.** His `2n+1` branches are unary laid out in TIME; V5's register was
> unary laid out in SPACE; V6's move was unary → binary. The log-register move is
> structurally unavailable to him: his construction must activate exactly one block
> per shot, and a design row's whole point is activating all of them at once.

## What is open

Exactness *below the shot floor* with full multiplexing needs a design reproducing
the product measure's marginal on `d+1` coordinates — **strength `r_max + 2`**, not
merely enough rows to count coefficients. Splitting by level-pattern blocks (rows
with exactly `r` coordinates at level `π`, where the weight `(1+ε)^{M−r}(−ε)^r` is
constant) makes strength apply within each block, at

    Σ_{r ≤ r_max} C(M,r) · M^{(r_max+2)/2}   rows   ≈  21 register qubits
                                                       at M = 36, r_max = 2

The remaining step is a construction, not a concept: the minimal run size of a
**3-level orthogonal array of strength 4 on 36 factors**. Design theory; there is
no quantum content in the statement.

*Exactness to machine precision* is separate and open. The bounds known here are
`m ≥ 2M+1` — the conditions at `k = ±e_l` involve `v_l`, `v̄_l` and `v₀`, which
must be independent — against `3^M` achievable by the full product. The gap between
them at `ν = O(1)` is the whole question.

  **AND THE STANDING CAVEAT.** Everything in this Part is tier B or design. The
  variance of the multi-radius estimator under shots is unmeasured, and Part V's
  substrate rule applies: a second register is substrate, but the `k` radii it
  replaces were invocations, and only the second column counts.

---

# Part IX — the walk rebuilt as a circuit, and the first end-to-end prototype

Part VI derived the walk's separability and Part VII asked what would be
classically hard. Both reasoned about an object that had never been built. This
Part is what happened when it was — `supplement/v136`–`v141`, and two new
modules, `qlto_walk.py` and `qlto_prototype.py`.

## The result that reorganises Parts VI and VII

**[measured, v136 PART 5 and PART 7]** The two candidate registers are not the
same kind of physical system, and they obey different semiclassical laws:

    cycle register       DeltaE = e^{-S0/h}     slope -0.998 against Liu-Su-Li
                                                Eq. 7's predicted -1
    hypercube register   DeltaE = e^{-n S~}     ln DeltaE linear in n to
                                                r = -0.99994

`h` is the mixer/potential ratio — free, and **independent of M**. `n` is the
**parameter count**. So the hypercube register's tunnelling degrades
exponentially as the model grows, and the cycle's does not.

**Why.** `n·I − 2J_x` is *not* the Laplacian of the hypercube's reduced chain.
Its row sums are

    n − sqrt(w(n-w+1)) − sqrt((w+1)(n-w))   =  n − sqrt(n) at the ends, ~0 in the middle

leaving a built-in inverted potential of size `O(h²n)` that no Agmon metric on
`V` alone can see. The hypercube register is a **large-spin (Lipkin–Meshkov–Glick
shaped) system whose semiclassical parameter is 1/n**, not a discretised
particle. Liu–Su–Li's framework is a particle with `−h²∆` on `R^d`; sweeping `h`
at fixed `n` is simply not its semiclassical limit.

> **Every variational algorithm that uses a `{±1}ⁿ` box register — which is most
> of them, V3–V6 included — is under the spin law.** That is not stated in
> Liu–Su–Li, in the QAOA/QWOA literature, or in Wang's implementation notes.

## Which corrects Part VI and Part VII

- Part VI's `L = A − nI` is a **sign error**: the graph Laplacian is `nI − A`.
- Part VI describes the walk as tracking the mixer's **ground state**. V3's
  `build_w_gate` maps `|0> -> c−R` and `|1> -> c+R`, so displacement is
  `−R⟨Z⟩`, and the descent move needs the **maximum** of `+(γ/2)Σg_iZ_i`; `h(param)`
  prepares all-`|+⟩`, the maximum of `+βΣX_i`. V3 tracks the **top** of the
  spectrum throughout. Two sign flips cancel, so the endpoint is right and the
  description is inverted.
- Part VII's "the mixer is the wrong knob" reaches the right conclusion by the
  wrong route. A degree-1 potential has exactly one minimum on **any** connected
  graph — flipping bit `i` changes `Σα_ix_i` by `−2α_ix_i`, so `x` is a local
  minimum iff `x_i = −sign(α_i)` for every `i`. No mixer rescues that. But the
  mixer matters enormously for a different reason: **it decides which
  semiclassical law you are under.** And its tier-C entanglement table
  understated it — **[measured, v136 PART 1]** a circulant mixer breaks
  separability with a **degree-1** drift alone (Möbius 0.5928, complete 0.8780,
  cycle 0.2410 mid-cut entropy from a product input), against the hypercube's
  exactly 0.0000.

## Where the Fourier energy actually sits

**[measured, tier B, v137]** Sampling `E(θ)` on a `3^M` grid — three points per
axis is exactly Nyquist for support `{−1,0,1}` — and taking the M-dimensional
FFT gives every `A_k`. Binned by weight `|k|₀`, over six configurations:

    Heisenberg N=2 reps=1  M= 8   weights 3..8    peak 5
    Heisenberg N=2 reps=2  M=12   weights 5..12   peak 7
    Heisenberg N=3 reps=1  M=12   weights 3..10   peak 6
    TFIM       N=2 reps=1  M= 8   weights 3..6    peak 3
    TFIM       N=2 reps=2  M=12   weights 4..10   peak 5
    TFIM       N=3 reps=1  M=12   weights 3..8    peak 4

**Weight-1 and weight-2 energy is identically zero in all six**, minimum weight
is ≥3 everywhere and rises with depth, and the bulk sits near `M/2`.
`A_{e_j}` is the first harmonic in `θ_j` after averaging over the full period of
every other parameter, and that average projects onto the commutant far enough
that nothing survives; `A_0` vanishes because the full-torus average of `⟨H⟩` is
`Tr(H)/2ⁿ = 0`.

Two things follow, and the second corrects Part VIII.

  **A FULL-PERIOD DESIGN CANNOT GIVE A LOCAL GRADIENT AT LOG WIDTH.** Encoding
  the register as a linear phase `u_j(d) = (2π/N)c_j d` makes `E(θ+u(d))` a
  Fourier series in `d` with frequency `⟨k,c⟩`, so a DFT separates modes — but
  bin `c_j` returns `A_{e_j}`, which does not exist. Recovering `∂_jE = Σ_{k_j≠0}
  A_k(ik_j)` needs every bin containing `j`, and the label `⟨k,c⟩` does not carry
  `k_j`. Separating them needs a Sidon set of order M, hence `N ~ 2^M` and a
  register of M qubits. Built, measured at `cos = −0.0666`, and withdrawn.

  **V6's SMALL-R BOX IS THE MECHANISM, NOT A DEFECT.** Its marginal is a finite
  difference, so every weight lands in the same bin attenuated by
  `cos(R)^{|k|₀−1}` but never separated. That attenuation is exactly what lets
  one log-width register carry all weights at once. **`D_eff` is not small — the
  bulk is at `M/2` — so the design works *despite* high weight, not because of
  low weight.** Part VIII reads the geometric decay of v135's fit residuals as
  evidence of a low-weight landscape; it is not. It is evidence that a
  low-degree polynomial approximates the sum well *on the sampled radii*, which
  is a weaker and different claim. The practical conclusion — four radii give
  ~1e-4 relative error — is unaffected.

## Three levels, and why not four

**[measured, tier B, v138]** For a shift `u_j = R σ_j`,

    E(θ + Rσ) = Σ_k A_k Π_{j: k_j≠0} g_{k_j}(σ_j),
    g_κ(σ)    = 1 + σ²(cos R − 1) + i κ σ sin R

At `q=2`, `σ² ≡ 1`, so `g` collapses to `cos R + iκσ sin R` and the σ-free part
of every spectator factor is `cos R`. **V6's low-pass filter is an artifact of
two levels, not of finite radius.** Two consequences, both measured:

  **The diagonal curvature becomes visible.** `∂²E/∂θ_j² = −Σ_{k_j≠0}A_k` is
  degenerate with the constant when `σ_j²` is constant. At `q=3` it is recovered
  to `rel 0.0083` at R=0.15, and the `q=2` case returns NaN *by construction*
  because `σ² − ⟨σ²⟩` is identically zero.

  **The spectator attenuation becomes tunable.** `α = p₀ + 2p₁cos R`, and the
  measured bias tracks `0.37·(1 − α³)` to 5% across `p₀ = 0 … 0.9` — a **9.8×
  reduction at `p₀ = 0.9`**, where `q=2` forces `p₀ = 0` and has no such knob.

**`q ≥ 4` buys nothing.** Per coordinate the landscape supplies exactly three
functions of the shift, `σ ↦ e^{ik_jRσ}` for `k_j ∈ {−1,0,1}`, whose determinant
is nonzero for `cos R ≠ 1`. Three levels span the space; a fourth adds rows
without adding information. A hard stop, not a diminishing return.

## `qlto_walk.py` — the module

**[measured, tier A]** A 3-level design register returning gradient, diagonal
Hessian and off-diagonal Hessian from **one shot record per commuting group**:

    R      cos(g)      rel g      rel H_diag   rel H_off
    0.50   0.999815   8.08e-02   9.51e-02     1.71e-01
    0.35   0.999944   2.91e-02   8.52e-02     2.57e-01
    circuit: 12 qubits, depth 25, 2q 27

The third level is decoded from **two parities** of the row index,
`σ_j = (a_j + b_j)/2`, realised as `RY(θ_j + R)` then `cRY(−R)` from each — two
controlled rotations per parameter against V6's one.

Two design conditions are load-bearing and both were violated in the first
build. The XORs `c_j ⊕ e_j` must **differ per parameter**, or every `σ_j²` is the
same function of `d` and the design is degenerate. And the columns must satisfy
**resolution V** — no column equal to the XOR of two others, all pairwise XORs
distinct — which plain Gray columns violate immediately (`1⊕3 = 2`), aliasing the
`{j,l}` interaction onto parameter 3's main effect. The failure signature is
diagnostic: **an error that GROWS as `R→0` can only be an alias**, since bias
shrinks with R and at tier B there is no noise.

The Hessian is the fragile channel: signal `sin²R` against the gradient's
`sin R`, so per-entry SNR is `R` times worse and it degrades as R falls while the
gradient improves. `λ_max` is an aggregate over `~m²/2` entries and survives that;
the Newton solve does not. **The gradient and the Hessian want different radii** —
the gradient's is a TRUST REGION and must shrink as the optimiser converges, the
Hessian's is a MEASUREMENT setting and should sit wherever the SNR is best.

## What the walk step actually does

**[measured, tier A, v139/v140]** Sense → `h = κR²√λ_max` → cycle mixer with the
measured `H` as a degree-2 potential → read a vertex. The potential is `RZ + RZZ`,
verified against the direct quadratic form at `2.6e-16` — `O((dκ)²)` gates, not a
`DiagonalGate`'s `O(2^{dκ})`.

Against four controls on the same box, all clipped to the same infinity-norm:

    brute force  ΔE = −0.860   100%     exhaustive over the same 4096 vertices
    WALK         ΔE = −0.776   90.2%
    grad         ΔE = −0.706   82.1%
    Newton       ΔE = −0.446   51.9%
    random       ΔE = −0.028    3.2%

and on moves to reach a top-1% vertex:

    uniform      97.1 moves    (control, predicted 100)
    WALK          3.1 moves    31x

The two are not in tension: the walk is a **fast sampler of good vertices, not an
exact minimiser**. It captures 90% of the achievable descent in 3 moves where
brute force uses 4096 evaluations, and loses on the *quality* of the point found.

The time is **sharply resonant** — at `h=0.234`, `t=9.1 → 34.8 moves`,
`t=36.5 → 5.5`, `t=73.1 → 25.5`. That is Liu–Su–Li's `t = π/ΔE` appearing
directly: off-resonance the walk is barely better than uniform.

  **AND THE HYPERCUBE IS NEARLY AS GOOD AT THIS SIZE** — 3.5 moves at depth 518
  against the cycle's 3.1 at 1117, so 1.9× cheaper normalised by depth. That is
  not a contradiction of the spin/particle result: `e^{−n·S̃}` degrades with the
  **parameter count**, and at `n = 12` it has not bitten. The circulant's
  advantage is scaling, not a constant factor, and this box is too small to show
  it.

## The exponent trade, and where the separation actually lives

Liu–Su–Li give, for a barrier of height `H` and width `w`:

    classical SGD    T ~ e^{2H/s}
    quantum walk     T ~ e^{S0/h},   S0 = ∫√(f−E)dx ≈ c·w√H

so quantum wins iff `w√H/h < 2H/s` — and **everything turns on how `w` scales
with `H`**:

    smooth quadratic barrier, curvature λ:  H = λw²/8  ⟹  w ~ √H
                                            ⟹  S0 ~ H       LINEAR  ⟹  NO separation
    barrier of FIXED width as H grows:      ⟹  S0 ~ w√H    SQRT    ⟹  separation

On a **continuous** box a degree-2 model's barriers are smooth and there is no
separation. On a **discrete** box a QUBO's barriers are measured in **Hamming
distance** — set by the coupling structure, not the height — so `w` is fixed and
the separation is possible.

> **The lattice is not a discretisation error. It is what makes thin barriers
> exist at all.**

Which collides with the spin/particle result and produces the design's central
tension:

    κ = 1     Hamming-thin barriers, separation possible
              BUT the spin law kills tunnelling as the model grows
    κ large   particle law, h free and flat in M
              BUT the continuum widens barriers as √H and the separation goes

putting the operating point at **κ ≈ 3–5** — wide enough to be a particle by
v136's fit, narrow enough that a barrier is still a few lattice steps.

## Why their query model is ours

Liu–Su–Li's Theorem 1.2 is a **query-complexity** separation: the landscape sits
behind an oracle `U_f|x⟩|z⟩ = |x⟩|f(x)+z⟩`, and the hardness comes from
**locally non-informative regions** — measure concentration hides the corridor
direction `v`. Their Eq. 120 landscape is a narrow tube of height `H₁` through a
plateau of height `H₂ ≫ H₁`, with `v` drawn at random from the unit sphere.

Parts III–VII read that access restriction as a caveat: *the hardness is
informational, not geometric*. It is worth reading the other way.

> **QLTO never has `f` explicitly.** `sense()` returns `g, H` from shots; there
> is no closed form of `E(θ)` anywhere in the pipeline, and the design register
> **is** the oracle. The access restriction their lower bound assumes is not a
> device we would have to impose — it is the actual situation.

  **[BLOCKED, v141]** Building Eq. 120 on a lattice and running the walk from
  `Φ₋` gives `P(W+) = 0.0000` in every configuration against SGD's 0.84–1.00. The
  tell is that `Φ₋` is localised in `W₋` to 1.000 in every row: there is **no
  tunnelling doublet at all**. Their Assumption 2.5 demands *resonant* wells —
  energy difference `O(h^∞)` — because the mechanism is a two-level Rabi
  oscillation that a detuning `Δ` suppresses by `(ΔE/Δ)²`. On a lattice the two
  wells contain different numbers of grid points (`|W₋| = 12` against
  `|W₊| = 13`), so they are detuned by orders of magnitude more than the
  splitting. And the classical arm is equally out of regime: the concentration
  factor is `e^{−dw²/2R²}` and at `d = 2` there is none. **The construction needs
  exact well degeneracy on a lattice AND a dimension high enough to hide the
  corridor; the second is out of simulation reach.**

## `qlto_prototype.py` — end to end

**[measured, tier A]** Data register → 3-level sensing → step → repeat.

    NEWTON step   MSE 0.061 → 0.004      cos(g) 0.98–0.997
    WALK step     MSE 0.061 → 0.004      non-monotone, final exact 0.0137

**Three circuits per epoch for the gradient, five with a second radius for the
Hessian, flat in `|D|` and in `M`.** That is the claim the prototype supports.

Three things it surfaced that only appear in a loop:

  **THE WEIGHTED REGISTER GIVES THE RESIDUAL HESSIAN, NOT THE MSE HESSIAN.** For
  `L = (1/S)Σ(f_x−y_x)²` the Hessian splits into a Gauss-Newton term `Σ ∂f∂f` and
  a residual term `Σ(f−y)∂²f`. The register returns `Σ_x w_x ×` (derivative of
  `f_x`), so it gives the residual and misses Gauss-Newton — which near a good
  fit is the dominant one. Measured: `rel` to the residual Hessian 0.35–0.76,
  `rel` to the full MSE Hessian 1.0–1.34. The instrument is correct; the loss
  decomposition is the limit. `JᵀJ` would need per-sample Jacobians, which is
  exactly the `|D|`-dependence the data register exists to avoid.

  **A SAMPLED STEP EQUILIBRATES RATHER THAN CONVERGING.** `walk_step` returns a
  *sample* from the walk's output distribution, and with a fixed `h` the state
  stays delocalised for the whole evolution. V3's `_execute_walk` always had the
  schedule — `β = (1−s)πδt` down, `γ = sπδt` up — and at `s→1` the Hamiltonian is
  pure potential, whose ground state is the vertex wanted. Restoring it, plus
  scoring every measured outcome against the classical model rather than taking
  the mode, is what makes the walk a *proposal distribution* used correctly.

  **`cos` IS SCALE-INVARIANT AND CANNOT SEE A MAGNITUDE ERROR.** This project's
  primary gradient metric is `cos(estimate, true)` — in V6, v123, v133 and
  `qlto_walk`'s own self-check. A uniform factor is invisible to all of them, and
  a step rule that clips or normalises absorbs it, so the optimiser still
  descends. Only the Hessian reports a magnitude, and only the magnitude exposed
  it. **Any estimator that reports a magnitude needs a magnitude check beside the
  cosine.**

## The ledger at 200 qubits / 10⁶ gates

    data          log₂|D|      = 30      10⁹ samples, flat in |D|   MEASURED
    model         N_sys        = 100
    design, q=3   2log₂M + 1   = 35      M = 1.25×10⁵               MEASURED
    QPE           k = 10       =  5      G → 1, ε ≈ ‖H‖/512         v132
    radius/scratch             =  7
                               ─────
                                 177

QPE's ladder is `2^k · L · N`; for a **single-observable QML loss** (`L = 1`) that
is `~2×10⁵` and fits, giving 2.7× better precision than 32768 shots in one
circuit. For **chemistry** (`L ~ N⁴`) the ladder alone is `~10¹³` and the
G-removal is a wash on gate-work — what survives there is the operational win of
one job submission instead of `G ~ N^4.24` (v30).

    classical epoch   O(|D| · M)     touches every sample
    quantum epoch     O(S · M)       3–5 circuits, flat in |D|

so the saving grows with **dataset size**, reaches parity on **parameter count**
(depth is linear in M and does not compress), and against other **quantum**
gradient methods the ratio grows as `2M`.

## The one measurement that decides the rest

Whether real loss-landscape Hessians produce corner minima separated by **tall,
Hamming-thin** barriers. If `w` stays fixed as `H` grows the separation is live at
scale; if `w ~ √H` it is not, and no walk engineering changes that.

It needs no quantum hardware: sense `H`, enumerate the box's local minima
classically, and measure barrier heights against the Hamming distances between
them. **[design]** — not run.
