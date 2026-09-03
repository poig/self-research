# QLTO — M parameters sensed on log₂M qubits

A resolution-IV Hadamard design carries **all M parameters at once** on
`⌈log₂(M+1)⌉+1` register qubits, so one gradient costs `G` circuits — the number
of qubit-wise-commuting groups in `H` — and nothing in that count depends on `M`.

```python
from nisq_v6 import QLTOv6

q = QLTOv6(ansatz, hamiltonian, shot_budget=8192)
theta, trace = q.minimize(theta0, epochs=20)
```

V5 spent one register qubit per parameter — 48 ancillas at M=48, undeployable.
V6 needs **7**. That is the result; the circuit count follows from it.

---

## The cost is three numbers, not one

| cost | V6 | parameter-shift |
|---|---|---|
| circuits per gradient | **`G`** | `2MG` |
| width | `N + ⌈log₂(M+1)⌉ + 4` | `N` |
| classical per gradient | `O(M)` Walsh decode | `O(M)` |

Quoting only the first invites the obvious rebuttal, so all three ship together.
`G` cancels from the ratio: `2MG/G = 2M`. **A large `G` bounds which problems are
reachable, not which method to use on them.**

## What is *not* claimed

- **Not** backprop scaling, and **not a shot saving** — V6 trades more shots for
  fewer circuits.
- **Not** a barren-plateau escape by construction alone: every factor
  `cos^(d−1)(R)` lies in [0,1], so `|∇E_R| ≤ |∇E|`.
- **Not** improved by preconditioning — the `2M` saving and preconditioning are
  mutually exclusive.
- **Not** asymptotic advantage. QLTO is a NISQ-regime construction; its value
  expires when error correction arrives and QPE is simply used instead.

---

## Current work: where a separation can live

The active question is not "is QLTO cheap" — that is the constant-factor claim
above and it is settled. It is whether there is a task where QLTO is separated
from every classical algorithm by more than a constant, and what hardness
assumption that separation would rest on. Ten files, each labelled by how it was
obtained (see "How claims are tiered" below):

**`qlto_separation.py`** — TIER C, derivation. Checks five candidate routes to a
complexity separation and rules out four by argument: gradient estimation itself
is polynomial classically, so no estimator trick over it can be exponential. The
route that survives changes the *input* rather than the estimator — learning
from quantum data, where Huang et al. (Science 2022) prove an exponential
separation in experiment count between algorithms with and without quantum
memory. This is the file the other nine build on or check against.

**`qlto_quantum_data.py`** (TIER B) → **`qlto_qdata_loop.py`** (TIER A) — the
surviving route, derived then built: the gradient of a nonlinear functional of
quantum data from one design register, first as an exact-amplitude identity,
then as real circuits on `AerSimulator` trained end to end against a BFGS
reference.

**`qlto_gradnorm.py`**, **`qlto_certified_radius.py`**, **`qlto_fourier_sampling.py`**,
**`qlto_weight_spectrum.py`** — TIER B, exact-amplitude identities about what the
design register's amplitudes (not just its measured counts) can carry: the
gradient norm as one probability, a certified trust radius from derivative-tensor
norms, reading Walsh coefficients directly off amplitudes, and the landscape's
per-degree Fourier weight.

**`qlto_local_design.py`** — TIER C, `NO CIRCUIT`. The open combinatorial problem
of a locally-routable design register in 2-D — exact GF(2) construction, not yet
a circuit.

**`qlto_szegedy.py`**, **`qlto_training_time.py`** — TIER C, derivations that
check specific claims against a closed form (Szegedy's quadratic walk bound; the
data-prep-is-one-time argument, which fails because measurement destroys the
state every shot). Both explicitly `NO CIRCUIT` — scoping and argument only, no
accuracy or cost figure taken from them.

## The prior lines (`modules/`)

Working code the current files build on or reference, kept for reuse rather than
narration:

- **`twirl_cal.py`** — device calibration via twirl designs; a twirl IS a design
  row, full rank by construction. Measured 3.0% relative error at T=0.25 on real
  circuits — the circuit-vs-analytic gap that motivates building everything as a
  real circuit rather than trusting an analytic pass.
- **`qlto_walk.py`** — three-level design register, gradient and Hessian from one
  shot record.
- **`qlto_prototype.py`** — data register + sensing + walk step composed
  end to end; the file where the 256× branch-averaging bug was caught by a
  Hessian magnitude that a cosine similarity couldn't see.
- **`qlto_qml.py`** — supervised QML on a weighted data register, three circuits
  per epoch, flat in `|D|` and `M`.
- **`qlto_hl.py`** — QLTO applied to Hamiltonian learning.
- **`nisq_v2.py`**, **`nisq_v3.py`**, **`nisq_v5.py`**, **`nisq_v6a.py`** — earlier
  lines (Riemannian/QFIM, one-circuit walk oracle, QPE, an alternate V6).
- **`qnspsa.py`** — QN-SPSA (Gacon et al., Quantum 5, 567, 2021) as the measured
  competitor.
- **`twirl_stage2_coherent.py`** — coherent-target-basis Stage-2 prototype for a
  companion twirling scheme.
- **`commute_fim.py`**, **`commute_gradient.py`**, **`commute_gradient_paper.py`**,
  **`commute_gradient_paper_unitary.py`** — QFIM and commuting-block gradient
  estimators, the second two an exact implementation of Bowles et al. (2024)
  Theorem 3.
- **`check_gs.py`** — small ground-state check used by the above.

---

## How claims are tiered

Every result is labelled by how it was obtained, and the tier gates what it may
support:

| tier | what it is | may support |
|---|---|---|
| **A** | `QuantumCircuit` on `AerSimulator` with `shots=` | any claim, including headline |
| **B** | circuit built, read exactly via `Statevector`/`Operator` | mechanism and structure — **never** an accuracy or cost figure |
| **C** | no circuit — dense linear algebra or pure argument | scoping/derivation only, labelled `NO CIRCUIT` |
| **D** | no quantum object at all — a resource ledger | nothing; not an experiment |

The rule exists because it was measured twice: an earlier analytic pass reported
0.13% error where the real circuit (`twirl_cal.py`) gave 3.0% — a 23× gap that
also moved the operating point and surfaced two endianness bugs dense matrices
hide entirely.

## Layout

```
nisq_v6.py            gradient engine — the stable, current line
benchmark.py           harness for nisq_v6.py, 8-problem suite

qlto_separation.py     TIER C  where a complexity separation can live — start here
qlto_quantum_data.py   TIER B  the surviving route, exact amplitudes
qlto_qdata_loop.py     TIER A  the surviving route, real circuits + shots
qlto_gradnorm.py       TIER B  gradient norm as one probability
qlto_certified_radius  TIER B  certified trust radius from derivative-tensor norms
qlto_fourier_sampling  TIER B  Walsh coefficients read off amplitudes
qlto_weight_spectrum   TIER B  landscape's per-degree Fourier weight
qlto_local_design.py   TIER C  NO CIRCUIT — open 2-D routable design problem
qlto_szegedy.py        TIER C  NO CIRCUIT — Szegedy's quadratic bound, checked
qlto_training_time.py  TIER C  NO CIRCUIT — is data prep one-time? no

modules/                prior lines: twirl_cal, qlto_walk, qlto_prototype,
                         qlto_qml, qlto_hl, nisq_v2/v3/v5/v6a, qnspsa,
                         twirl_stage2_coherent, commute_*, check_gs
```
