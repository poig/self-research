# QLTO — Quantum Landscape Tunneling Optimizer

Two lines of work on ancilla-mediated variational quantum optimization: a theory
strand on single-ancilla coherent feedback, and an application strand on the
optimizer itself. **The theory strand's original claims were withdrawn after
testing and replaced by an exact result that is now machine-checked in Lean; the
application strand's surviving claim is circuit cost, not accuracy.** Several
results in this repository were withdrawn by their own author after further
testing, and the withdrawals are documented alongside the claims — read the
`RESEARCH_NOTES.md` and the `README` in each subdirectory before citing anything.

```
QLTO/
├── Application/    the optimizer — V3, V5, benchmarks, research notes  ← start here
│   └── supplement/     ~70 numbered experiments, each with its log
├── theory_test/    numerics behind the theory paper, mapped claim-by-claim
│   └── QuantumFeedback/  Lean 4 + Mathlib proofs of the algebraic core
├── paper/          manuscripts
└── Feigenbaum/     earlier exploratory work
```

## Application — QLTO V3 / V5

A one-circuit local-landscape oracle. One circuit places 2ⁿ parameter
configurations in superposition, reads an energy for each, and returns a gradient
for every coordinate at once from the measurement marginals.

```python
opt = QLTOv3(ansatz, hamiltonian)
params, energy = opt.minimize()      # no tuning; same settings across 8 problems
```

The circuit-count law is now exact rather than fitted. For `efficient_su2` with
`r` reps on `N` qubits, block partitioning by disjoint qubit support gives
`L = 2(r+1)` blocks of `n = N` parameters, so one gradient costs `G·L = 2G(r+1)`
circuits against parameter-shift's `2MG`:

```
C_PS / C_QLTO  =  2M/L  =  2N        exactly, on all 12 (N, r) rows tested
```

**Constant in N** at fixed `G` and reps, **linear in reps**, and the advantage
over parameter-shift **grows linearly in qubit count**. `G` is the only route by
which `N` could re-enter and it does not for the families tested: `G = 3` for
Heisenberg and `G = 1` for MaxCut at N = 4, 6, 8, 10 alike.

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

**The depth penalty belongs to the QPE path, not to the default.** The 19–141×
figure was measured on the `2^a` Trotter ladder. On the direct readout that now
ships, a circuit costs about 30% more depth and 2–3× the entangling gates than
parameter-shift — but because QLTO runs `2N` fewer circuits, one *gradient* spends
**3.1–9.7× less total depth and 1.3–5.5× fewer two-qubit gates**. The per-circuit
and per-gradient ratios point opposite ways, which is what made the per-circuit
figure misleading on its own.

Local CPU is at parity too. An earlier measurement put QLTO 40–100× worse, all of
it circuit rebuilding; the sensing circuit is now built once as a parameterised
template and bound each epoch, so warm build is 2–16 ms against 535–753 ms before.

Accuracy is competitive, not dominant. **The bias floor turned out to be a
protocol artefact.** At fixed R the gradient direction plateaus at cos ≈ 0.98, and
that was twice read as a property of the estimator. It is not: bias goes like
`R²` and variance like `1/(R²S)`, so the optimal radius shrinks as the budget
grows. With R free, the error keeps falling at the predicted exponent —
`1−cos ∝ T^(−0.75)` measured against `−2/3` predicted, parameter-shift `−0.92`
against `−1` — and at Heisenberg N=6 **QLTO beats parameter-shift at matched total
shots** by ~1.5× in the low-to-mid budget range. The crossover budget grows
steeply with M. Parameter-shift still has the better exponent and wins eventually
at any fixed M, so this is a regime claim, not a reversal.

The walk is retired. The quantum walk on the parameter register — the "tunneling"
in the name — lost 0 of 7 on the full suite to a plain classical decode of its own
marginal at half the circuits. What survives is the sensing front-end: a gradient
estimator, not an optimizer, and not a tunneling one.

Full record in [`Application/RESEARCH_NOTES.md`](Application/RESEARCH_NOTES.md);
start with [`Application/README.md`](Application/README.md).

### Where this sits against the field

The benchmark set here — parameter-shift, AdamW, QNG, SPSA — is the right
comparison for a NISQ variational optimizer of about 2021, and against it the
circuit-count result holds. It is **not** the current frontier, and QLTO has not
been measured against the frontier:

- **Bowles, Wierichs & Park**, *Backpropagation scaling in parameterised quantum
  circuits* (Quantum 9, 1873, 2025) — achieves backpropagation-like scaling, i.e.
  better than `2N`, for circuits whose parameters feed into *commuting* gates.
  Better on its circuit class; does not apply to a general ansatz.
- **Chinzei et al.**, [arXiv:2406.18316](https://arxiv.org/abs/2406.18316) — proves
  a trade-off `ℱ_eff ≤ 𝒳_exp ≤ 4ⁿ/ℱ_eff − ℱ_eff` between the number of
  *simultaneously measurable* gradient components and dim(DLA), forcing
  `ℱ_eff = 1` for a hardware-efficient ansatz. QLTO reads N components per circuit
  on `efficient_su2`, which is consistent only because its components are not
  commuting observables and its estimand is the *R-smeared* gradient. That
  suggests the bias is the price of the batching rather than a defect — a reading
  of someone else's theorem, not a derivation.
- **Generalized Hadamard Test**, [arXiv:2408.05406](https://arxiv.org/abs/2408.05406)
  — improves the *constant* in front of `Θ(M)` (measured ~9× over naive
  parameter-shift) by choosing per parameter whether to group the generator or the
  observable. Complementary to, not competing with, a law that is constant in M.

SPSA deserves a specific note: QLTO's marginal under antithetic sampling *is* the
SPSA estimator — measured gap exactly 0.0000. The difference is entirely in the
readout, where a bounded ±1 ancilla bit removes the `|∇E|² − (∂ᵢE)²` cross-talk
that SPSA structurally cannot shed.

## Theory — status

**Posted:** *Information-Theoretic Constraints on Variational Quantum Optimization:
Efficiency Transitions and the Dynamical Lie Algebra*
([arXiv:2512.14701](https://arxiv.org/abs/2512.14701), v1–v2 Dec 2025).

**Prepared as v3, retitled:** *Commutator-Governed Energy Exchange in
Single-Ancilla Coherent Feedback*.

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

The corrected manuscript (`research_paper/paper/paper1`) is prepared as v3 and not
yet posted. Numerics and their logs are in `theory_test/`, mapped claim-by-claim
in [`theory_test/README.md`](theory_test/README.md).

**The algebraic core is machine-checked.** `theory_test/QuantumFeedback/` is a
Lean 4 + Mathlib development covering the identity, its vanishing under
commutation, the tensor structure carrying that to the controlled protocol, and
the **symmetry of the reachable interval** — the part that forbids directional
cooling. `lake build` succeeds and every exported theorem depends on exactly
`[propext, Classical.choice, Quot.sound]`, with no `sorry`. Two statements come
out stronger than the paper needs: they hold in *any* finite dimension rather
than the N ≤ 7 simulated, and the identity is formalised with no unitarity
hypothesis and no expansion in θ. The closed *form* of the interval endpoint is
not formalised — it rests on von Neumann's trace inequality, which is not in
Mathlib — and nothing in the development assumes it.

### An independent check of the surviving claim

The commutator condition shows up in the optimizer too. A block of rotations whose
generators commute with H has an *identically zero* gradient — measured at
**3.3e-16** on MaxCut, where a final RZ layer commutes with a diagonal Hamiltonian,
against 1.36–1.79 for Heisenberg where it does not. Different protocol, same
identity. Skipping those blocks cut run-to-run variance 30× and 25% of circuits.

## Requirements

Python 3.9+, Qiskit ≥ 2.0, qiskit-aer, NumPy, SciPy.
