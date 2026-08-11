# Tractability certificates: formalising "extend the catalogue"

A formalisation of the programme discussed in `RESEARCH_NOTES.md` under
"ONE OBJECT". The informal claim was: because Hamiltonians are a universal
encoding language, a structural property that certifies tractability applies to
everything encodable in it — so progress means adding entries to a catalogue of
such properties rather than finding one universal analyser.

This file makes that precise, proves the "one universal analyser" version is
impossible for a reason simpler than the natural-proofs barrier, and reduces the
programme to a concrete open question that is testable at present scale.

---

## 1. Setting

Write `𝓗ₙ` for the set of n-qubit Hamiltonians presented **by their Pauli
decomposition**

```
    H = Σ_P c_P P ,      P ∈ {I,X,Y,Z}^{⊗n} ,   |supp(c)| ≤ poly(n)
```

The presentation matters and is the whole subject of obstruction 5 in
`RESEARCH_NOTES.md`: a Hamiltonian arrives as its coefficient list, not as an
oracle. Everything below is about what can be read off that list.

Fix a computational task `T` on `𝓗ₙ` — e.g. *"output a state with energy within
ε of the ground state"*, or *"find θ with E(θ) ≤ E_min + ε using poly(n) circuit
evaluations"*.

**Definition 1 (tractability certificate).**
A predicate `Π : 𝓗ₙ → {0,1}` is a *tractability certificate for T* if

- **(C1) checkable.** `Π(H)` is computable in time `poly(n)` from the Pauli list.
- **(C2) sound.** There exist an algorithm `A` and a polynomial `q` such that for
  every `H` with `Π(H) = 1`, `A` solves `T` on `H` in time `q(n)`.

(C1) is not decoration. Without it, `Π(H) = 1 iff T is easy on H` is a certificate
in name and useless in fact. The content of a certificate is that it is cheaper to
check than the task is to perform.

**Definition 2 (coverage).**
For a distribution `D` on `𝓗ₙ`, the *coverage* of `Π` is
`cov_D(Π) = Pr_{H∼D}[Π(H) = 1]`.

---

## 2. There is no universal certificate

**Theorem 1.** Let `Π` be a tractability certificate for the local Hamiltonian
ground-state problem with `Π ≡ 1` on all of `𝓗ₙ`. Then `NP ⊆ P`.

*Proof.* Lucas's Ising formulations encode every problem in a standard
NP-complete list as the ground state of a Hamiltonian with `poly(n)` Pauli terms,
computable in polynomial time from the instance. Given such an instance, encode it
as `H ∈ 𝓗ₙ`; by hypothesis `Π(H) = 1`; by (C2) algorithm `A` returns a ground
state in `q(n)` time, from which the instance's answer is read off. ∎

The proof needs no complexity-theoretic barrier — the universality of the encoding
does the work directly. **The same expressiveness that makes Hamiltonians a
universal language for problems makes a universal analyser of them equivalent to
solving NP.** That is the precise form of the obstruction to a single tool.

A finer statement is available for *broad but non-universal* `Π` via natural
proofs (Razborov–Rudich): an efficiently computable property that holds for a
non-negligible fraction of instances and certifies an efficient algorithm cannot
be extended to a general separation without contradicting the existence of strong
one-way functions. Theorem 1 is the crude version and is sufficient here.

**Corollary 1.** Every sound certificate is *partial*: `∃H` with `Π(H) = 0`.
Progress therefore consists of enlarging

```
    𝒞 = ⋃ᵢ { H : Πᵢ(H) = 1 }
```

the union of certified families, and is measured by `cov_D(⋁ᵢ Πᵢ)` on
distributions of interest — not by seeking a single `Π`.

---

## 3. The existing catalogue, as instances

| `Π` | check | certifies |
|---|---|---|
| `Π_stoq` — all off-diagonal `c_P ≤ 0` | `O(#terms)` | sign-problem-free QMC |
| `Π_free` — quadratic in fermion operators after Jordan–Wigner | `O(#terms)` | exact diagonalisation |
| `Π_tw,k` — interaction graph has treewidth `≤ k` | FPT in `k` | tensor-network contraction |
| `Π_DLA` — `dim 𝔤(H, ansatz) ≤ poly(n)` | see below | classical Lie-algebraic simulation |

**A caution that Definition 1 makes visible.** Two familiar "structures" are not
certificates in this sense without care:

- *1D gapped* certifies DMRG, but **checking the gap is not known to be easy** —
  it fails (C1) in general, even though *1D and local* is checkable.
- *`Π_DLA`* requires computing the dimension of a Lie closure, which can itself be
  exponential. It satisfies (C1) only on families where the closure is known
  analytically.

That distinction — between a property that *holds* and a property that can be
*checked from the specification* — is exactly obstruction 5 restated at the level
of the catalogue.

---

## 4. The open question, stated so it can be answered

Take `T` = trainability of a variational ansatz on `H`. The standard certificate
is `Π_DLA`, resting on `Var(∂E) ∼ 1/dim 𝔤` (Ragone et al.).

Define a second, *manifestly* (C1)-checkable candidate. Writing
`H = Σ_S ĥ(S) P_S` and

```
    p(S) = ĥ(S)² / Σ_T ĥ(T)²  ,      PR(H) = 1 / Σ_S p(S)²
```

`PR` is the participation ratio of the Pauli spectrum — the effective number of
terms carrying the weight. Set `Π_conc,k(H) = 1 iff PR(H) ≤ k`. This is computable
in `O(#terms)` directly from the coefficient list, so (C1) is immediate; whether
it is **sound** is the open part.

**Question.** Are `Π_conc` and `Π_DLA` comparable?

Three outcomes, with different consequences:

1. `Π_conc ⟹ Π_DLA` — concentration is a cheap sufficient condition for the known
   certificate, useful as a fast pre-check and nothing more.
2. `Π_DLA ⟹ Π_conc` — dimension counting is the stronger notion and concentration
   adds nothing.
3. **Incomparable** — there exist `H` concentrated with exponential DLA, and `H`
   with polynomial DLA and spread spectrum. Then `Π_conc ∨ Π_DLA` strictly
   enlarges `𝒞`, and by Corollary 1 that is precisely what progress means here.

Outcome 3 is the interesting one and is the concrete form of "extend the
catalogue". It is decidable by construction at present scale: exhibit one
Hamiltonian of each kind.

---

## 5. What this does and does not settle

It **does** convert an aspiration into a definition with a theorem attached, and
reduces a programme to a question with a finite answer.

It does **not** establish that `Π_conc` is sound. Soundness needs either a bound
of the form *"concentrated spectrum ⟹ gradient variance not exponentially small"*
or a counterexample. Nothing here supplies one, and the incomparability question
above is logically prior: if `Π_conc` turns out to be implied by `Π_DLA` there is
no reason to look for the bound.
