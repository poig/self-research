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

| cost | V6 | parameter-shift | measured in |
|---|---|---|---|
| circuits per gradient | **`G`** | `2MG` | `v87` — exactly `G` on all 7 problems |
| width | `N + ⌈log₂(M+1)⌉ + 4` | `N` | `v87` — 2.25×–5×, **additive**, ratio falls with N |
| classical per gradient | `O(M)` Walsh decode | `O(M)` | `v87` — no exponential term |

Quoting only the first invites the obvious rebuttal, so all three ship together.
`G` cancels from the ratio: `2MG/G = 2M`. **A large `G` bounds which problems are
reachable, not which method to use on them** — see Part V of the notes.

## Sharpest measured results

| | result |
|---|---|
| 8-problem suite, 5 trials | 1st on 2, 2nd on 5, 3rd on 1 — **cheapest on all 8** |
| sharpest row | Heisenberg N=8: **−11.6222 at 60 circuits** vs AdamW −11.6133 at 3840 |
| total resources | 28× fewer **qubit·shots** at N=8, charging V6 its full width penalty |
| variance scaling | per-component variance **flat** while parameter-shift's grows 16×; exponents **1.006 vs 2.000** (`v82`) |
| vs strongest competitor | QN-SPSA (Gacon 2021), swept: **7–10× worse at 4.4× the circuits** (`v91`) |
| the estimator, exactly | a low-pass filter on Fourier degree, verified to **3.8e-17** (`v89`) |

## What is *not* claimed

- **Not** backprop scaling, and **not a shot saving** — V6 trades ~4× more shots
  for ~32× fewer circuits (`v109`). No contradiction with Abbas et al. (NeurIPS
  2023).
- **Not** a barren-plateau escape. Proven, not merely untested: every factor
  `cos^(d−1)(R)` lies in [0,1], so `|∇E_R| ≤ |∇E|` (`v89`).
- **Not** improved by preconditioning — the `2M` saving and preconditioning are
  mutually exclusive (`v91`).
- **Not** asymptotic advantage. QLTO is a NISQ-regime construction; its value
  expires when error correction arrives and QPE is simply used instead.

---

## The three application modules

**`nisq_v6.py`** — the gradient engine. Log-width design register, `G` circuits
per gradient, `radius_exponent` knob. The other two build on it.

**`twirl_cal.py`** — device calibration. A Pauli conjugation flips coefficient
signs, so **a twirl IS a design row** and the device supplies the evolution
exactly in every branch — no model, no Trotter bias. Full rank is a theorem: the
columns are Walsh characters of distinct symplectic vectors. Measured 3.0%
relative error at T=0.25 on real circuits.

**`qlto_qml.py`** — supervised QML on a weighted data register. Three circuits per
epoch, flat in |D| and M, counted not asserted; the estimator drives its own
descent, 3/3 seeds. `G = 1` structurally, since a QML readout is one Pauli.

Earlier lines: `nisq_v3.py` (the one-circuit walk oracle), `nisq_v5.py` (QPE),
`nisq_v2.py` (Riemannian/QFIM). Each carries its own docstring.

---

## Where to read what

This README is the summary. The detail is split in two:

[`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) — **the current state**, Parts III–IX:

| part | covers | read it when |
|---|---|---|
| **III — the QML axis map** | per axis: which of *qubits / circuits / gates* stays small (v121–v131) | you are asking "does it scale" |
| **IV — where advantage can live** | design + literature, labelled `[measured]/[lit]/[design]` | you are choosing a direction |
| **V — the accounting rule** | substrate vs. invocation count | **before any cost comparison** |
| **VI — the walk step, derived** | the separability theorem: the walk as built is a tensor product | you are touching the walk |
| **VII — the bridge** | simulability ladder, potential degree, the three complexity barriers | you are claiming an advantage |
| **VIII — the sensing register is a walk** | the design register IS the hypercube mixer, so its eigenbasis fixes the measurement grading; the radius is a second axis (v135) | you are touching `sense`, the radius, or comparing to a published gradient method |
| **IX — the walk built, and the prototype** | cycle register = particle, hypercube = SPIN degrading in the parameter count; three-level sensing gives the Hessian; end-to-end training (v136–v141) | you are touching the walk, the Hessian, or `qlto_prototype.py` |

[`ARCHIVE_V3_V6.md`](ARCHIVE_V3_V6.md) — **the historical record**, Parts I–II:
the V3 sensing oracle and walk with their theory, and the V6/calibration line.
Kept verbatim per R2; its header lists which of its verdicts Parts III–VII
supersede, so read those with the correction rather than on their own.

Other documents:

- [`TRACTABILITY_CERTIFICATES.md`](TRACTABILITY_CERTIFICATES.md) — when the
  construction is certified to apply
- [`supplement/`](supplement/) — every experiment, one script per claim, logs in
  `supplement/results/`
- [`../../../CLAUDE.md`](../../../CLAUDE.md) — **R1**, the circuits-not-matrices
  rule and its three tiers; **R2**, withdrawals stay in the record

## How claims are tiered (R1)

Every result is labelled by how it was obtained, and the tier gates what it may
support:

| tier | what it is | may support |
|---|---|---|
| **A** | `QuantumCircuit` on `AerSimulator` with `shots=` | any claim, including headline |
| **B** | circuit built, read exactly via `Statevector`/`Operator` | mechanism and structure — **never** an accuracy or cost figure |
| **C** | no circuit — dense linear algebra | scoping only, labelled `NO CIRCUIT` |

The rule exists because it was measured twice: `v101` reported 0.13% analytically
where the real circuit gave 3.0%, a 23× gap that also moved the operating point
and surfaced two endianness bugs dense matrices hide entirely.

## Layout

```
nisq_v6.py       gradient engine — the current line
twirl_cal.py     device calibration via twirl designs
qlto_qml.py      supervised QML on a weighted data register
nisq_v5.py       QPE line          nisq_v3.py   walk oracle
nisq_v2.py       Riemannian/QFIM   benchmark.py harness
commute_*.py     QFIM and commuting-block estimators
RESEARCH_NOTES.md  current state, Parts III–IX
ARCHIVE_V3_V6.md   historical record, Parts I–II
supplement/        one script per claim + results/
```
