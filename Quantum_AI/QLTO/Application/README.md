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
nisq_v3.py           the optimizer (this is the one to read)
nisq_v2.py           predecessor: Riemannian / commuting-block QFIM variant
commute_*.py         V2's gradient and metric engines
benchmark.py         8-problem suite vs AdamW, SPSA, QNG, QAOA, V2
RESEARCH_NOTES.md    the full research record — derivations, results, dead ends
supplement/          investigation scripts, one question each
supplement/results/  their logs; every number in the notes cites one
results/             benchmark output
```

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
