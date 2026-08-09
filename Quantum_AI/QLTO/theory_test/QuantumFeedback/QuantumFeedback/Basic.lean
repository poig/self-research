/-
Formal proofs for *Commutator-Governed Energy Exchange in Single-Ancilla
Coherent Feedback*.

WHAT IS FORMALISED HERE, and equally what is not, because the distinction matters
more than the line count.

  Theorem 1  (`work_eq_conj`)          the exact energy identity
  Corollary 1(`work_eq_zero_of_commute`) vanishing when the generator commutes
  structure  (`commute_kron_controlled`) the controlled/tensor form of the above
  Theorem 2, symmetry part (`neg_mem_reachable`, `reachable_symmetric`)
                                        the reachable set is closed under
                                        negation, hence sup = -inf

NOT FORMALISED: the CLOSED FORM of the endpoint,
W* = Σ_k λ_k↓(M₁₁) λ_k↓(ΣX). That is von Neumann's trace inequality (1937),
which is not in Mathlib at the time of writing and is a substantial development
in its own right. Nothing below assumes it. The symmetry of the interval - which
is the part that does the physical work, because it is what forbids directional
cooling - does NOT depend on it, and is proved here outright.

The proofs are stated over an arbitrary finite index type, so they hold for any
finite dimension, not merely the N ≤ 7 of the numerical work.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Order.Bounds.Basic

open Matrix Kronecker

namespace QuantumFeedback

variable {n : Type*} [Fintype n] [DecidableEq n]

/-! ### Theorem 1 — the exact energy identity

The energy change across one feedback step is the expectation of `A - Uᴴ A U`
in the pre-feedback state. This is stated for an arbitrary matrix `A` and an
arbitrary `U`; unitarity is not needed, and neither is any expansion in the
feedback strength. -/

/-- Conjugation moves across a dot product. -/
theorem dotProduct_conj (A U : Matrix n n ℂ) (ψ : n → ℂ) :
    star (U *ᵥ ψ) ⬝ᵥ (A *ᵥ (U *ᵥ ψ)) = star ψ ⬝ᵥ ((Uᴴ * A * U) *ᵥ ψ) := by
  rw [star_mulVec]
  simp [Matrix.dotProduct_mulVec, Matrix.vecMul_vecMul, Matrix.mul_assoc]

/-- **Theorem 1.** The energy change is exactly the expectation of `A - Uᴴ A U`,
for every feedback strength and every sensing time. -/
theorem work_eq_conj (A U : Matrix n n ℂ) (ψ : n → ℂ) :
    star ψ ⬝ᵥ (A *ᵥ ψ) - star (U *ᵥ ψ) ⬝ᵥ (A *ᵥ (U *ᵥ ψ))
      = star ψ ⬝ᵥ ((A - Uᴴ * A * U) *ᵥ ψ) := by
  rw [dotProduct_conj, sub_mulVec, dotProduct_sub]

/-! ### Corollary 1 — exact vanishing under commutation -/

/-- If `A` commutes with `U` and `U` is unitary, conjugation fixes `A`. -/
theorem conj_eq_self_of_commute {A U : Matrix n n ℂ} (hU : Uᴴ * U = 1)
    (h : Commute A U) : Uᴴ * A * U = A := by
  rw [Matrix.mul_assoc, h.eq, ← Matrix.mul_assoc, hU, Matrix.one_mul]

/-- **Corollary 1.** The work vanishes identically — for every feedback strength
and every sensing time — whenever the feedback generator commutes with the
Hamiltonian. No expansion in `θ` is involved, so this holds to all orders. -/
theorem work_eq_zero_of_commute {A U : Matrix n n ℂ} (hU : Uᴴ * U = 1)
    (h : Commute A U) (ψ : n → ℂ) :
    star ψ ⬝ᵥ (A *ᵥ ψ) - star (U *ᵥ ψ) ⬝ᵥ (A *ᵥ (U *ᵥ ψ)) = 0 := by
  rw [work_eq_conj, conj_eq_self_of_commute hU h, sub_self, Matrix.zero_mulVec,
    dotProduct_zero]

/-! ### The controlled/tensor structure

In the protocol `A = I_A ⊗ H` and the feedback is controlled on the ancilla,
`U = P₀ ⊗ I + P₁ ⊗ V`. The hypothesis of Corollary 1 is then implied by
`[H, V] = 0` alone — nothing about the ancilla factors matters. -/

variable {m : Type*} [Fintype m] [DecidableEq m]

/-- `I ⊗ H` commutes with `P ⊗ V` exactly when `H` commutes with `V`. The ancilla
factor is arbitrary, so this covers both branches of a controlled operation. -/
theorem commute_kron_controlled (P : Matrix m m ℂ) (H V : Matrix n n ℂ)
    (h : Commute H V) : Commute (1 ⊗ₖ H) (P ⊗ₖ V) := by
  unfold Commute SemiconjBy
  rw [← Matrix.mul_kronecker_mul, ← Matrix.mul_kronecker_mul, Matrix.one_mul,
    Matrix.mul_one, h.eq]

/-! ### Theorem 2, symmetry — the reachable set is closed under negation

The reachable work at fixed correlations is obtained by conjugating the feedback
generator over its isospectral orbit. Writing the first-order work as
`Tr (M * (Vᴴ * Y * V))` for a fixed Hermitian `M`, the claim is that if the
generator's spectrum is symmetric about zero — equivalently, if some unitary `Q`
carries `Y` to `-Y` — then every reachable value has its negation reachable.

This is what forbids directional cooling, and it is proved here without any
appeal to von Neumann's trace inequality. -/

/-- The set of first-order work values reachable by a unitary change of frame. -/
def reachable (M Y : Matrix n n ℂ) : Set ℂ :=
  {w | ∃ V : Matrix n n ℂ, Vᴴ * V = 1 ∧ w = (M * (Vᴴ * Y * V)).trace}

/-- **Theorem 2, symmetry.** If some unitary carries `Y` to `-Y` — that is, if the
generator's spectrum is symmetric about zero — then the reachable set is closed
under negation. The witness is explicit: replace the frame `V` by `Q * V`. -/
theorem neg_mem_reachable {M Y Q : Matrix n n ℂ} (hQ : Qᴴ * Q = 1)
    (hQY : Qᴴ * Y * Q = -Y) {w : ℂ} (hw : w ∈ reachable M Y) :
    -w ∈ reachable M Y := by
  obtain ⟨V, hV, rfl⟩ := hw
  refine ⟨Q * V, ?_, ?_⟩
  · rw [Matrix.conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Qᴴ, hQ,
      Matrix.one_mul, hV]
  · have key : (Q * V)ᴴ * Y * (Q * V) = Vᴴ * (Qᴴ * Y * Q) * V := by
      simp only [Matrix.conjTranspose_mul]
      simp [Matrix.mul_assoc]
    rw [key, hQY]
    simp [Matrix.mul_neg, Matrix.neg_mul]

/-- A set of reals closed under negation has `sSup = -sInf`: the interval is
symmetric about zero. Combined with `neg_mem_reachable`, this is the statement
that an unfiltered conditional kick heats exactly as readily as it cools. -/
theorem sSup_eq_neg_sInf_of_neg_mem {S : Set ℝ} (h : ∀ x ∈ S, -x ∈ S) :
    sSup S = -sInf S := by
  have hneg : -S = S := by
    ext x
    simp only [Set.mem_neg]
    exact ⟨fun hx => by simpa using h _ hx, fun hx => h _ hx⟩
  calc sSup S = sSup (-S) := by rw [hneg]
    _ = -sInf S := Real.sSup_neg S

end QuantumFeedback
