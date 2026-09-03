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

## A complexity separation was searched for, and not found

Whether there is a task where QLTO is separated from every classical algorithm
by more than a constant was checked directly: five candidate routes, four ruled
out by argument (gradient estimation itself is polynomial classically, so no
estimator trick over it can be exponential), the fifth open only by importing
Huang et al.'s already-published result on quantum-memory-assisted state
learning rather than adding anything new. No separation survived, and the
gradient estimator below is measured elsewhere in this project to coincide with
SPSA under antithetic sampling — so what's left is a circuit-count constant, not
an advantage. The exploration is not kept in this directory; this paragraph is
the record of it, per the project's own rule that withdrawals stay noted beside
the claim rather than as a document of their own.

## Current work: device calibration (`modules/`)

The active line is device Hamiltonian calibration, in `twirl_cal.py`. The rest of
`modules/` is prior working code kept for reuse rather than narration:

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
nisq_v6.py    gradient engine — the stable line, equivalent to SPSA under
              antithetic sampling plus an ancilla-readout refinement
benchmark.py  harness for nisq_v6.py, 8-problem suite

modules/      twirl_cal.py — device calibration, the active line — plus
              prior code kept for reuse: qlto_walk, qlto_prototype, qlto_qml,
              qlto_hl, nisq_v2/v3/v5/v6a, qnspsa, twirl_stage2_coherent,
              commute_*, check_gs
```
