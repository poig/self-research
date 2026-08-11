# Evidence for *Commutator-Governed Energy Exchange in Single-Ancilla Coherent Feedback*

This directory is the code and data behind the manuscript. Every claim-bearing
script writes a `.log` next to it, committed, so a reader can compare against the
paper without running anything. All results are exact statevector simulation at
N ≤ 7; there is no sampling noise anywhere, and no hardware run.

Run any script directly, no arguments:

```bash
python isospectral_family.py
```

## Claim → evidence

| Paper | Claim | Script | Output |
|---|---|---|---|
| Thm 1 (`thm:identity`) | `W = ⟨Ψ₁\|A − U†AU\|Ψ₁⟩` exactly, every θ and τ | `norm_check.py`, `ancilla_test.py` | agreement to 3e-15 |
| Cor 1 (`cor:zero`) | `W = 0` identically when `[H,Y] = 0` | `forbidden_benchmark_check.py` | `.log` |
| Cor 1, 2nd part | zero work reachable at unchanged correlation | `isospectral_family.py` | `.log`, `isospectral_family.png` — **Fig. 1** |
| Thm 2 (`thm:interval`) | reachable work is `[−W*, +W*]`, closed form | `achievable_work_theorem.py` | `.log` |
| Thm 2, saturation | explicit `V = U_X U_M†` attains the bound | `achievable_work_theorem.py` | err 1.6e-15, 400 Haar V, 0 violations |
| Thm 2, asymmetry | interval symmetric to machine precision | `exact_interval_asymmetry.py` | `.log` |
| Prop (`prop:general`) | `Δ⟨A⟩ = (θ/2)Tr(i[ρ,A]G) + O(θ²)`; work and VQE gradient are instances | `general_response_interval.py` | `.log` — gradient matches parameter-shift to 4e-16 |
| Prop (`prop:twoconditions`) | two independent sufficient causes of symmetry | `two_conditions.py`, `what_breaks_symmetry.py`, `generator_symmetry_rule.py`, `commutator_spectrum_symmetry.py` | `.log` |
| Sec. "Relation to Cooling" | breaking both conditions buys `\|D\| ≤ 0.0147` | `directional_fraction.py` | `.log` |
| Cor (`cor:purefixed`) | pure post-sensing branch forces symmetry | `purity_forbids_cooling.py` | `.log` |
| Prop (`prop:secondorder`) | second-order directed work, `𝒜(θ) = 2.137θ` | `second_order_directed_work.py` | `.log` |
| Eq. (`eq:landauerthreshold`) | `k_B T* = 0.406` at τ = 1.42 | `landauer_threshold_temperature.py`, `landauer_limit_test.py` | `.log`, `thermo_landauer_check.png` — **Fig. 2** |
| Sec. "A Constructive Consequence" | symbolically-predicted zero-gradient blocks | `phase_degree_bound.py`, `walk_symmetry_classification.py` | `.log` |

### Machine-checked proofs — `QuantumFeedback/`

Theorem 1, Corollary 1, the tensor structure carrying Corollary 1 to the
controlled protocol, and the **symmetry half of Theorem 2** are formalised in
Lean 4 + Mathlib in `QuantumFeedback/`. `lake build` succeeds and every exported
theorem depends on exactly `[propext, Classical.choice, Quot.sound]` — **no
`sorry`**. The formal statements hold in *any* finite dimension, not only the
N ≤ 7 simulated here, and Theorem 1 is formalised with no unitarity hypothesis
and no expansion in θ.

The closed form of the endpoint `W*` is **not** formalised — it rests on von
Neumann's trace inequality, which is not in Mathlib. Nothing in the development
assumes it, and the interval's symmetry does not depend on it. See
`QuantumFeedback/README.md`.

Supporting checks that do not appear as numbered results: `su2_check.py`,
`intercept_check.py`, `alignment_decomposition.py`, `asymmetric_generator.py`,
`asymmetry_diagnostic.py`, `break_time_reversal.py`, `trotter_r2_check.py`,
`harmonized_sweep.py`, `k_ancilla_bandwidth_test.py`, `two_conditions.py`.

## `superseded/` and `supplementary/` — read before citing anything in them

**Nothing in these two directories supports the current version.** They are kept
for provenance, because versions 1–2 of the preprint were built on them and a
reader auditing the correction needs to see what was withdrawn.

`superseded/` holds the constitutive-relation and efficiency-transition analysis:
`thermo_constitutive_law.py`, `thermo_scrambling_crash.py` and their figures.
Versions 1–2 proposed `ΔE ≤ η·I(S:A)` fitted across a sensing-time sweep, and an
efficiency transition governed by DLA dimension. Both are withdrawn.
`isospectral_family.py` in the root directory is the counterexample that rules out
the first: it constructs a family on which `I(S:A)` and the logarithmic negativity
are invariant to ~2e-15 while the work changes sign. The efficiency transition does
not survive normalisation by ‖H‖, and the original fits used a first-order Trotter
approximation far from convergence.

`supplementary/` holds the older DLA, Carnot-bound, Jarzynski and demon-optimisation
material from the same period, under the same caveat.

## Four numerical hazards, each of which produced plausible output rather than an error

Documented in the manuscript's Methods and repeated here because they cost real time:

1. Qiskit's `PauliEvolutionGate` defaults to `LieTrotter(reps=1)`, far from
   converged for these Hamiltonians — moving to exact evolution changed a reported
   R² by a factor of three at N = 7.
2. `DensityMatrix.expectation_value` does not check that observable and state have
   matching dimension, and returns a number when they do not.
3. An acceptance gate on regression quality admits fits to numerical noise: a case
   whose extracted work was 1e-31 cleared R² ≥ 0.60.
4. **Two scripts claimed the same seeded Hamiltonian in their docstrings and built
   different ones.** `landauer_limit_test.py` couples all pairs;
   `landauer_threshold_temperature.py` coupled nearest neighbours only. Same seed,
   but the extra J draws advance the stream, so the transverse fields differed too.
   The threshold was therefore computed on a chain and quoted against a
   complete-graph figure — `k_B T*` understated 3.5×, peak work 0.063 against the
   figure's 0.258. Fixed here; both now build all pairs. **A seeded random
   Hamiltonian looks reproducible exactly when it is not being compared.** Assert
   agreement of the constructed operator across scripts, not of the seed.

All current results use exact evolution, dimension-checked observables, and a
single shared Hamiltonian construction.

### Added in the reach pass

| Paper | Claim | Script | Output |
|---|---|---|---|
| `cor:tracenorm` | `W* = (θ/2)·‖Y‖∞·‖M₁₁‖₁` exactly on pure branches | `reach_monotone_and_size.py` | ratio 1.000 pure, 0.905–0.981 mixed |
| size-independence | `spec(ΣᵢXᵢ)` symmetric at every n, checked to n=11 | `reach_monotone_and_size.py` | defect ~1e-14 vs 1.15–1.85 for generic Hermitian |
| Scope of the obstruction | symmetric for k=1,2,3 ancillas, non-product `K`, all cycles | `class_extension.py` | all rows `0.0e+00` |
| Sec. "The filter works" | filtered `K` cools to 94.5%; unfiltered moves energy by exactly zero | `filter_price.py` | `.log` |
| Sec. "The price" | ~32 Hamiltonian evolutions per cycle; below that it **heats** | `filter_price.py` | `.log` |

Two conventions that cost real time and are worth stating, since both produced
plausible output rather than an error:

- **Ancilla pairing.** Cooling requires the lowering operator `K` paired with the
  ancilla *raising* operator `|1><0|`. Getting it backwards makes the protocol
  heat monotonically, which reads as a physics result rather than a sign error.
- **Filter discretisation.** An under-resolved time integral does not merely
  approximate the filter badly — it inverts the sign of the effect. At 8 and 16
  samples the protocol heats; at 32 it cools.


### Not a manuscript claim: the input-model obstruction

| Claim | Script | Output |
|---|---|---|
| Simon-structured `H` has Pauli support exactly on the annihilator subgroup — 32/64 strings, all orthogonal to `s`, none otherwise — while the gradient looks statistically ordinary | `shor_hamiltonian_signature.py` | `.log` |

This one supports no result in the paper. It is the sharpest form of why
hidden-subgroup structure is unreachable for a specified (as opposed to
oracle-given) Hamiltonian, and lives in the QLTO research notes rather than here.
