# QLTO V6 — M parameters sensed on log₂M qubits

The current line. A resolution-IV Hadamard design carries **all M parameters at
once** on `⌈log₂(M+1)⌉+1` register qubits, so one gradient costs `G` circuits —
the number of qubit-wise-commuting groups in `H` — and nothing in that count
depends on `M`.

```python
from nisq_v6 import QLTOv6

q = QLTOv6(ansatz, hamiltonian, shot_budget=8192)
theta, trace = q.minimize(theta0, epochs=20)
```

V5 spent **one register qubit per parameter**. At M=48 that is 48 ancillas —
undeployable. V6 needs **7**. That is the result; the circuit count follows from
it.

## The cost is three numbers, not one

| cost | V6 | parameter-shift | measured in |
|---|---|---|---|
| circuits per gradient | **`G`** | `2MG` | `v87` — exactly `G` on all 7 problems, 16–64× under |
| width | `N + ⌈log₂(M+1)⌉ + 4` | `N` | `v87` — 2.25×–5×, **additive**, ratio falls with N |
| classical per gradient | `O(M)` Walsh decode | `O(M)` | `v87` — no exponential term |

Quoting only the first invites the obvious rebuttal, so all three ship together.
`Θ(G)` is a **construction-level** fact — V6 issues `G` circuits whatever `M` is,
at any size, needing no measurement. That it still *converges* competitively at
that cost is a separate claim, measured to N=8.

## What is measured

| | result |
|---|---|
| 8-problem suite, 5 trials, one setting | **1st on 2, 2nd on 5, 3rd on 1**, cheapest on all 8 |
| sharpest row | Heisenberg N=8: **−11.6222 at 60 circuits** vs AdamW **−11.6133 at 3840** |
| total resources, not just circuits | 28× fewer **qubit·shots** at N=8, *charging V6 its full width penalty* |
| variance scaling, fixed-norm regime | per-component variance **flat** (2.9e-5 over a 16× range in M) while parameter-shift's grows 16×; exponents **1.006 vs 2.000** (`v82`) |
| crosstalk | *decays* with M, 0.0339 → 0.0133 (`v82`) |
| vs the strongest competitor | QN-SPSA (Gacon et al. 2021), swept and bracketed: **7–10× worse at 4.4× the circuits** (`v91`) |
| quantum-data task | Hamiltonian learning has `G = 1` **structurally**: 16 coefficients on a 6-qubit register, 1 circuit per gradient, 32× under parameter-shift (`v88`) |
| the estimator, exactly | `E_s[s_i·E(θ+Rs)] = sin(R)·Σ_{T∋i} cos^(\|T\|−1)(R)·∂_iE_T` — a low-pass filter on Fourier degree, verified to **3.8e-17** (`v89`) |

## What is *not* claimed

- **Not** backpropagation scaling. In the wide-ansatz regime V6's variance
  exponent is 1.94 against parameter-shift's 2.00 (`v82`) — there the advantage
  is circuits, not shots. This is why V6 does not contradict the lower bound of
  Abbas et al. (NeurIPS 2023), which forbids backprop scaling for single-copy
  measurement.
- **Not** a barren-plateau escape. Proven, not merely untested: every factor
  `cos^(d−1)(R)` lies in [0,1], so smoothing attenuates every Fourier component
  and amplifies none — `|∇E_R| ≤ |∇E|`, with `max ratio ≤ 1` on every
  configuration measured (`v89`).
- **Not** improved by preconditioning. `F⁻¹` on top of V6 hurts (κ(F)=349, the
  inverse amplifies noise along low-metric directions), and obtaining the metric
  cheaply fails too (`v91`). The `2M` saving and the preconditioning are
  mutually exclusive.
- **Not** improved by Richardson extrapolation: it loses at every budget tested,
  and its fitted exponent is worse than plain V6's (`v84`).
- **Not** a chaotic or period-doubling optimiser. The iteration is period-2
  everywhere from gain 0.005 to 0.95 — max-normalisation makes a fixed point
  structurally unreachable — with no cascade at any gain (`v96`).

## Knobs, and which way they were measured

| knob | default | finding |
|---|---|---|
| `design_resolution` | 4 | 5 removes 3- and 4-term column aliasing and recovers the Hamiltonian-learning cosine (0.714 → 0.927 at M=16, `v90`) — but on **VQE** it is three ties and one marginal loss at 20–52% more depth (`v97`). Keep 4. |
| `n_scratch` | 3 | 2 saves a qubit at neutral depth and tied-or-better energy; 1 saves two but costs 21–50% depth (`v98`). The genuinely free lever, and the one never swept until now. |
| `r0`, `r_decay` | 0.6, 0.95 | still fitted. The SNR-optimal radius derived from the degree law under-predicts (`v92`), and there is no dynamical instability to derive it from (`v96`). |

# QLTO V3 — a one-circuit local-landscape oracle

One circuit puts 2ⁿ parameter configurations in superposition, reads an energy for
each, and returns a gradient for **every coordinate at once** from the measurement
marginals. Cost per epoch is flat in the parameter count.

```python
from nisq_v3 import QLTOv3

opt = QLTOv3(ansatz, hamiltonian)     # defaults are the measured optima
params, energy = opt.minimize()       # 20 epochs, no tuning
```

No per-problem tuning: the same settings ran all 8 benchmark problems, spanning
‖H₀‖ from 0.83 to 21.2 and M from 8 to 32 parameters.

## How it works

```
per epoch:
  R  = 0.6 · 0.9^epoch              shrink the search box
  dt = 0.5 · 0.95^(epoch+1)         shrink the step
  for each commuting block:
      circuit 1  W-gate → 2ⁿ corners in superposition, each entangled with its
                 own ansatz state; QPE reads an energy per corner; per-bit
                 marginals give the gradient
      circuit 2  quantum walk drifts the param register along that gradient
  circuit 3      log the energy
```

**180 circuits for 20 epochs** on a 4-block ansatz (140 when a block is provably dead), regardless of how many parameters there are.

## What is measured

| | result |
|---|---|
| accuracy | **1st or 2nd on 6 of 8 problems**, one outright win (MaxCut N=4), at **140–180 circuits vs 400–4080** |
| cost vs parameter-shift | **8–36× fewer circuits**, and circuits are what the vendors bill |
| how that scales | circuit advantage **grows with M** — 10.2× at M=16, 15.3× at M=24, 20.4× at M=32 |
| priced on real tariffs | one Heisenberg N=6 gradient: **$243 parameter-shift vs $24–30 QLTO** on IBM; 18× on Braket |
| projected to large N | direct-readout cost per gradient is **flat in N** ($30.0 at N=8 → $30.7 at N=100); parameter-shift reaches **134×** |
| what the circuits cost | 19–141× the depth for QPE — a **coherence** constraint, largely invisible to billing since `rep_delay` (250 µs) dominates circuit duration |
| gradient quality | at **fixed R**, direction plateaus at cos ≈ 0.98; the floor is R being pinned, not the method — trading R against shots gives O(1/ε³) with no floor |
| classical complexity | **Θ(N·U) vs parameter-shift's Θ(N²·U)** per gradient — a factor of N, growing (wall-clock build time is a Qiskit artefact, not this) |
| the estimator | the marginal *is* the degree-1 Walsh coefficient of the energy on the ±R hypercube, exact to 1e-16 |
| why it scales | that estimator is **linear**, so it is unbiased at *any* shots-per-vertex — including fewer than one |
| landscape structure | locality bounds the Walsh degree; degree-1 + degree-2 is 99.6%+ of the local landscape |
| application | Hamiltonian learning validated end-to-end: 30 circuits vs 300 |

## What is *not* claimed

- **Not** accuracy dominance. Across 8 problems: 1 win, 2 seconds, 5 thirds.
  Competitive, not winning — and the reps=1 ansatz ceiling means the suite cannot
  resolve optimizers on accuracy anyway.
- **Not** a barren-plateau solution. This is a cost-function-difference estimator,
  the class Arrasmith et al. prove is *exponentially suppressed* on a plateau.
- **Not** an exponential speedup. Superposition here is a resource for computing
  averages cheaply, not for searching. See `RESEARCH_NOTES.md` for why Grover and
  hidden-subgroup structure are both closed.

## Layout

```
nisq_v6.py           the current optimizer — log-width design register
qnspsa.py            QN-SPSA (Gacon et al. 2021), the closest competitor
nisq_v3.py           predecessor: one-circuit QPE oracle
nisq_v2.py           predecessor: Riemannian / commuting-block QFIM variant
commute_*.py         V2's gradient and metric engines
benchmark.py         8-problem suite vs AdamW, SPSA, QNG, QN-SPSA, QAOA
RESEARCH_NOTES.md    the full research record — derivations, results, dead ends
supplement/          investigation scripts, one question each
supplement/results/  their logs; every number in the notes cites one
results/             benchmark output
```

## What binds at scale, and it is not width

V6 removed the `M` factor entirely, so the remaining quantum cost **is** `G`. For
molecular Hamiltonians `v30` measured `G ~ N^4.24`, and nothing in the V6 line
touches that. The register overhead is `N + log₂M + O(1)` and therefore
asymptotically negligible on its own; shaving qubits off it does not change the
exponent.

V5's QPE path was the attempt to remove `G` — reading the energy from a phase
rather than measuring each commuting group — and it died on depth, with measured
survival **0.098** at Heisenberg N=6. So the open scaling question is whether
`G`-independence is reachable without a Trotter ladder. Everything since has been
optimising the term that no longer dominates.

## On V2

`nisq_v2.py` is the earlier Riemannian variant: it estimates the Quantum Fisher
Information Matrix from commuting blocks and preconditions the step with it. It
is kept as the benchmark comparison, but the metric was measured **not to help** —
proper block natural gradient, magnitude-matched natural gradient, and the
diagonal square-root variant all fail to beat no metric at all, and estimating F
from shots adds variance without adding signal. See `RESEARCH_NOTES.md`.

## Reading the notes

`RESEARCH_NOTES.md` is a research record, not API documentation. Negative results
sit beside positive ones because most of what I learned is where things do
*not* work — five separate attempts to give the update more information (natural
gradient, nonlinear decoders, degree-2 drift, mixer shaping, multi-level encoding)
all failed, and that consistency is itself the finding.

Two traps documented there are worth knowing before trusting any benchmark:
`StatevectorEstimator(default_precision=p)` returns the *exact* expectation plus
fixed noise — it never samples, and it silently subsidised baselines by 23–57×;
and sub-2σ results on few seeds reversed twice under replication.
