# QLTO — Quantum Landscape Tunneling Optimizer

Two lines of work on ancilla-mediated variational quantum optimization: a theory
strand on single-ancilla coherent feedback, and an application strand on the
optimizer itself. **The theory strand's original claims were withdrawn after
testing; the application strand is where the surviving results are.**

```
QLTO/
├── Application/    the optimizer — V3, benchmarks, research notes   ← start here
├── theory_test/    numerical experiments behind the theory paper
├── paper/          manuscripts
└── Feigenbaum/     earlier exploratory work
```

## Application — QLTO V3

A one-circuit local-landscape oracle. One circuit places 2ⁿ parameter
configurations in superposition, reads an energy for each, and returns a gradient
for every coordinate at once from the measurement marginals.

```python
opt = QLTOv3(ansatz, hamiltonian)
params, energy = opt.minimize()      # no tuning; same settings across 8 problems
```

Measured: **8–36× fewer circuits** than fairly-charged parameter-shift, and that
advantage **grows with parameter count** (10.2× at M=16 → 20.4× at M=32).

**Priced against how the vendors actually bill**, one Heisenberg N=6 gradient costs
$243 on IBM by parameter-shift against **$24–30** by QLTO. IBM's published formula
is `overhead + (rep_delay + circuit_length) × circuits × shots` with `rep_delay`
defaulting to 250 µs charged *per circuit per shot*, and AWS Braket charges per
task plus per shot with **no depth term at all** — so circuit count is the billed
quantity and depth is nearly free. Projected analytically (no simulator needed,
these are counting arguments), the direct-readout variant's cost per gradient is
**flat in N** — $30.0 at N=8, $30.7 at N=100 — while parameter-shift grows linearly
to **134× more expensive**.

On the audited harness, across 8 problems with one setting: **1st or 2nd on 6 of 8**,
one outright win, at **140–180 circuits against 400–4080**.

Each QPE circuit is 19–141× deeper, which matters for *coherence* but is largely
invisible to *billing*. Accuracy is competitive, not dominant. At fixed R the
gradient direction plateaus at cos ≈ 0.98 — a bias floor, and the analysis shows
it is a property of pinning R, not of the method: trading R against shots gives
O(1/ε³) with no floor. An earlier claim of *3.2× fewer shots* is **withdrawn**; it
came from normalising each estimator against its own target instead of a common one.

Full record in [`Application/RESEARCH_NOTES.md`](Application/RESEARCH_NOTES.md);
start with [`Application/README.md`](Application/README.md).

## Theory — status

**Paper:** *Information-Theoretic Constraints on Variational Quantum Optimization:
Efficiency Transitions and the Dynamical Lie Algebra* ([arXiv:2512.14701](https://arxiv.org/abs/2512.14701),
v1 Dec 2025, v2 Dec 2025).

**Versions 1–2 proposed a constitutive relation ΔE ≤ η·I(S:A) linking work
extraction to system–ancilla mutual information, and an efficiency transition
governed by Dynamical Lie Algebra dimension. Both are withdrawn.** The sensing
sweep behind the original figure is not injective in I(S:A), and an explicit
one-parameter family exists on which I(S:A) and the logarithmic negativity are
*exactly* invariant while the extracted work varies continuously and changes sign —
so no function W = f(I(S:A)) exists. The DLA efficiency transition does not survive
normalisation by ‖H‖, and the linear fits defining η used a first-order Trotter
approximation far from convergence.

**What replaces them** is an exact expression for the energy change in terms of the
feedback commutator, W = ⟨Ψ₁|A − U†AU|Ψ₁⟩ with A = I_A ⊗ H, verified against
circuit simulation to 3×10⁻¹⁵ and independent of feedback strength; it vanishes
identically whenever the feedback generator commutes with H; and at fixed
correlations the reachable work is a closed-form interval symmetric about zero.
That symmetry is the protocol's defect — an unfiltered conditional kick heats as
readily as it cools.

The corrected manuscript is drafted (`research_paper/paper/paper1`) and not yet
posted. Numerics in `theory_test/`.

### An independent check of the surviving claim

The commutator condition shows up in the optimizer too. A block of rotations whose
generators commute with H has an *identically zero* gradient — measured at
**3.3e-16** on MaxCut, where a final RZ layer commutes with a diagonal Hamiltonian,
against 1.36–1.79 for Heisenberg where it does not. Different protocol, same
identity. Skipping those blocks cut run-to-run variance 30× and 25% of circuits.

## Requirements

Python 3.9+, Qiskit ≥ 2.0, qiskit-aer, NumPy, SciPy.
