# Quantum AI Research

One project, two strands: a theory strand on single-ancilla coherent feedback,
and an application strand on the gradient estimator that came out of it.

**Read the withdrawals before the claims.** Several results here were retracted by
their own author after further testing, and the retractions are kept beside the
original claims rather than tidied away. This README is an index; every number
below is sourced one directory down.

## [QLTO](./QLTO/) — Quantum Landscape Tunneling Optimizer

### Theory — posted, and the central claims of v1–v2 are withdrawn

**[arXiv:2512.14701](https://arxiv.org/abs/2512.14701)** — *Commutator-Governed
Energy Exchange in Single-Ancilla Coherent Feedback*, Jun Liang Tan (quant-ph,
cs.ET). v1 2 Dec 2025, v2 20 Dec 2025, **v3 12 Aug 2026**, retitled from
*Information-Theoretic Constraints on Variational Quantum Optimization*.

v3 withdraws both of the results v1–v2 were built on:

| withdrawn in v1–v2 | why |
|---|---|
| a constitutive relation `ΔE ≤ η·I(S:A)` linking work extraction to system–ancilla mutual information | the sensing sweep behind the figure is not injective in `I(S:A)`, and an explicit one-parameter family holds `I(S:A)` and logarithmic negativity *exactly* invariant while the work varies continuously and changes sign — so no `W = f(I(S:A))` exists |
| an efficiency transition governed by Dynamical Lie Algebra dimension | does not survive normalisation by `‖H‖`; the linear fits defining `η` used a first-order Trotter approximation far from convergence |

**What replaces them** is exact rather than empirical: the energy change per cycle
is `W = ⟨Ψ₁|A − U†AU|Ψ₁⟩` with `A = I_A ⊗ H`, agreeing with circuit simulation to
3e-15 and independent of feedback strength. It vanishes identically whenever the
feedback generator commutes with `H`, and at fixed correlations the reachable work
is a closed-form interval **symmetric about zero** — which is the protocol's
defect, not its feature: an unfiltered conditional kick heats as readily as it
cools.

**The algebraic core is machine-checked.** `QLTO/theory_test/QuantumFeedback/` is a
Lean 4 + Mathlib development covering the identity, its vanishing under
commutation, the tensor structure carrying that to the controlled protocol, and
the symmetry of the reachable interval. `lake build` succeeds with no `sorry`, and
every exported theorem depends on exactly `[propext, Classical.choice, Quot.sound]`.
Two statements come out stronger than the paper needs — they hold in *any* finite
dimension, not only the N ≤ 7 simulated, and the identity is formalised with no
unitarity hypothesis and no expansion in θ. The closed *form* of the interval
endpoint is **not** formalised: it rests on von Neumann's trace inequality, which
is not in Mathlib.

Scope: `theory_test/` is exact statevector simulation at N ≤ 7. No sampling noise
anywhere, no hardware run. Claim-by-claim map in
[`QLTO/theory_test/README.md`](./QLTO/theory_test/README.md).

### Application — the surviving claim is circuit cost, not accuracy

QLTO V6 carries all `M` parameters on a `⌈log₂(M+1)⌉+1`-qubit resolution-IV
Hadamard design, so one gradient costs `G` circuits — the number of
qubit-wise-commuting groups in `H` — and nothing in that count depends on `M`.
Against parameter-shift's `2MG`, measured at exactly `G` on all 7 problems.

The cost is **three numbers**, and quoting only the first is the claim most open
to attack: circuits (`G` vs `2MG`), width (`N + ⌈log₂(M+1)⌉ + 4` vs `N`, additive,
ratio falling with `N`), and classical post-processing (`O(M)` decode, no
exponential term).

What is **not** claimed, each measured rather than merely untested:

- **Not** a barren-plateau escape. Every smoothing factor `cos^(d−1)(R)` lies in
  [0,1], so `|∇E_R| ≤ |∇E|` — it attenuates every Fourier component and amplifies
  none.
- **Not** backpropagation scaling, so it does not contradict Abbas et al.
  (NeurIPS 2023). In the wide-ansatz regime at matched *total shots* the variance
  exponent is 1.94 against parameter-shift's 2.00 — an estimator-level measurement
  on a synthetic landscape, which says nothing about depth or hardware.
- **Not** a tunneling optimizer, despite the name. The quantum walk is retired: it
  lost 0 of 7 to a plain classical decode of its own marginal at half the
  circuits. The *mechanism* is real, but that statement is **tier C — `NO
  CIRCUIT`**: it comes from a dense matrix exponential on a constructed 1D spike
  potential, not a VQE landscape, and the quantum transmission that "stays flat"
  as the barrier grows is flat at 0.3–0.7%, roughly 200 repetitions per success.
  A tier-C result like this supports feasibility and nothing more. The walk does
  not reach it either way: its ceiling is `sqrt(RW)` with `RW ~ 2ⁿ`, still
  exponential where descent is ~n/2.

### Current line — device Hamiltonian calibration

The newest work applies the design register to learning an unknown device
Hamiltonian. A Pauli conjugation flips coefficient signs, so a twirl *is* a design
row and the device supplies the evolution exactly in every branch — no synthesised
model, therefore no product formula and no Trotter bias. The Walsh column for each
term is handed over by its own symplectic vector.

Built as a real Qiskit circuit in `QLTO/Application/twirl_cal.py`:

- **8 circuits, flat in N and in M** — `2 × n_probes`. Two measurement bases
  suffice for *any* Pauli Hamiltonian, because a term invisible to all-Z lies in
  `{I,Z}^N`, invisible to all-X lies in `{I,X}^N`, and the intersection is the
  identity alone.
- **1.9% ± 0.15%** relative error at 524288 shots per circuit, 6.7% ± 1.0% at
  65536 — seed-averaged, on circuits, with a shot floor.
- **Noise costs scale, not shape.** Across two-qubit error rates from 0 to 1e-2
  the cosine to the truth moves 0.0045 while the global scale falls 0.983 → 0.630.
  That is the recoverable failure mode.

**Four claims made about this line have been withdrawn after testing**, and the
withdrawals are documented beside them in `Application/README.md`: it is not
`O(1)` as first stated (the shipped path was 8N until the readout was grouped by
basis); the 3.0% headline was a single unseeded draw; the "8× shots, no gain —
bias-limited" reading is refuted at the operating point; and the accuracy ratio
against the iterative path compared a lucky draw to a worst case across a 9.6×
shot gap and a cold start to a warm one.

Two limits are structural rather than fixable. **QPE cannot replace the readout**:
the twirl is a similarity transformation, so the evolved spectrum is invariant and
a phase carries literally no signal (1.4e-15 spectral deviation, 2.1e-16 phase
coefficient). And **the design is confounded at degree 2** — `σ_j σ_k` is itself a
Walsh character, `v_XX + v_YY = v_ZZ` on every bond, and `M > 2N` forces such
dependencies — which unlike QLTO's register cannot be repaired with more design
rows, because here the columns *are* the Paulis.

Against the field, the honest position is that this is **not** the frontier, and
it is now measured rather than asserted. On the literature's own axis — total
evolution time to precision ε — twirl_cal fits `ε ~ T_total^(−0.217)` against
SQL's −0.500. It also sits in the wrong row of the comparison: it requires the
term set named in advance, so it solves the *known-structure* problem where the
frontier is Heisenberg-limited `O(1/ε)` with no ancilla, while spending 2N.
[arXiv:2606.19486](https://arxiv.org/abs/2606.19486)'s Stage 2 already does the
same one-dataset-decodes-all-coefficients trick **ancilla-free**, which confirms
independently that the register buys circuit compression rather than information.

And the two cannot be reconciled by importing their kernel, for a reason that
turns out to be general: batching any such protocol to K settings leaves a
variance floor `Var_settings/K` that shots cannot touch, so **K itself must scale
as ε⁻²**. A fixed O(1) design therefore buys itself a precision floor.
*"O(1) circuits"* and *"matches the proven-optimal protocol"* are mutually
exclusive; this line chose the former.

The defensible claim is correspondingly narrow: a heuristic, fixed-design
estimator with O(1) circuit settings and a precision floor, useful at moderate ε
on per-task-billed hardware. Still untested and still decisive: T1/T2 idle decay
on the register during the device evolution.

That file is also the project's clearest illustration of its own house rule. The
same construction evaluated on classical amplitudes reported 0.13%; building it as
a circuit moved the operating point and exposed two endianness bugs that dense
matrices hide entirely.

## Layout

```
QLTO/
├── Application/    the optimizer — V3, V5, V6, benchmarks, research notes  ← start here
├── theory_test/    numerics behind the theory paper, mapped claim-by-claim
│   └── QuantumFeedback/  Lean 4 + Mathlib proofs of the algebraic core
├── paper/          manuscripts
└── Feigenbaum/     earlier exploratory work
```

## License

Self-research by the author (Tan Jun Liang). Reuse, redistribution, or
republishing requires explicit permission — see the repository root `README.md`.
