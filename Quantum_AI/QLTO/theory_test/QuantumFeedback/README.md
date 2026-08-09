# Machine-checked proofs

Lean 4 (v4.32.2) + Mathlib formalisation of the algebraic core of
*Commutator-Governed Energy Exchange in Single-Ancilla Coherent Feedback*.

```bash
lake exe cache get     # prebuilt Mathlib oleans, ~7 GB
lake build             # builds QuantumFeedback/Basic.lean
lake env lean Check.lean   # prints the axiom dependencies
```

`lake build` reports **Build completed successfully**, and `Check.lean` reports
that every theorem below depends on exactly `[propext, Classical.choice,
Quot.sound]` — the three standard Lean axioms. **No `sorryAx` appears**, so
nothing here is stubbed or assumed.

## What is proved

| Paper | Lean | Statement |
|---|---|---|
| Thm 1 | `work_eq_conj` | `⟨ψ,Aψ⟩ − ⟨Uψ,A Uψ⟩ = ⟨ψ,(A − UᴴAU)ψ⟩` |
| — | `conj_eq_self_of_commute` | `Uᴴ U = 1`, `Commute A U` ⟹ `Uᴴ A U = A` |
| Cor 1 | `work_eq_zero_of_commute` | the work vanishes identically under commutation |
| structure | `commute_kron_controlled` | `Commute H V` ⟹ `Commute (1 ⊗ₖ H) (P ⊗ₖ V)` |
| Thm 2 (symmetry) | `neg_mem_reachable` | the reachable set is closed under negation |
| Thm 2 (symmetry) | `sSup_eq_neg_sInf_of_neg_mem` | such a set has `sSup = −sInf` |

Two things are worth noting about the *strength* of these statements, both of
which are stronger than the manuscript needs.

**They hold in any finite dimension.** The index type is an arbitrary
`Fintype`, so nothing depends on the N ≤ 7 of the numerical work. The simulations
verify these results at particular sizes; the Lean proofs establish them at all
sizes.

**Theorem 1 assumes no unitarity and no small-angle expansion.** `work_eq_conj`
is stated for arbitrary `A` and `U`. That is the formal content of "exactly, for
every θ and every τ" — there is no order in the feedback strength at which it
could fail, because no expansion is taken.

## What is NOT proved, and why

**The closed form of the endpoint.** Theorem 2 gives
`W* = Σₖ λₖ↓(M₁₁)·λₖ↓(ΣX)`, which is von Neumann's trace inequality (1937). That
is not in Mathlib at the time of writing and is a substantial development on its
own. **Nothing here assumes it.**

This matters less than it might appear. The trace inequality supplies the
*numerical value* of the interval endpoint. What forbids directional cooling is
that the interval is *symmetric about zero* — and that is proved outright, by
`neg_mem_reachable`, from a hypothesis that is elementary to state: some unitary
`Q` carries `Y` to `−Y`. The proof is constructive and the witness is explicit —
replace the frame `V` by `Q·V` — so the negation of every reachable value is
reachable by a named unitary, not merely known to exist.

The remaining results of the paper — the isospectral construction, the two
sufficient conditions for symmetry, the purity corollary, the second-order
response, the Landauer threshold — are numerical and live in the parent
directory with their `.log` files. They are not formalised here.
