/-
  ClassNP.lean: Complexity Class Definitions via Bellantoni-Cook

  Defines P, NP, Karp reductions, NP-hardness, and NP-completeness
  using the SafeFunction characterization of polynomial time.

  These are machine-independent definitions that match the Mathlib style
  used for Primrec and Partrec.
-/

import BQP_NP.Complexity.SafeRecursion

set_option autoImplicit false

open Matrix

variable {α : Type*} [DecidableEq α]

/-! ## Language Definitions -/

/-- A language over alphabet α is a set of strings (lists) -/
abbrev Language (α : Type*) := Set (List α)

/-! ## Complexity Class P -/

/-- Class P: Languages decidable in polynomial time.

    A language L is in P if there exists a poly-time computable function f
    such that: x ∈ L ↔ f(x) = [] (empty list = accept convention) -/
def ClassP (α : Type*) [DecidableEq α] : Set (Language α) :=
  {L | ∃ f : (Fin 1 → List α) → List α,
    PolyTimeComputable f ∧ ∀ x, x ∈ L ↔ f ![x] = []}

/-! ## Complexity Class NP -/

/-- Class NP: Languages with polynomial-time verifiable certificates.

    A language L is in NP if there exists:
    - A poly-time verifier V(x, cert)
    - A polynomial bound k on certificate size
    Such that: x ∈ L ↔ ∃ cert, |cert| ≤ |x|^k and V(x, cert) = []

    Note: We use List.length for the polynomial bound. -/
def ClassNP (α : Type*) [DecidableEq α] : Set (Language α) :=
  {L | ∃ (V : (Fin 2 → List α) → List α) (k : ℕ),
    PolyTimeComputable V ∧
    ∀ x, x ∈ L ↔ ∃ cert : List α, cert.length ≤ x.length ^ k ∧ V ![x, cert] = []}

/-! ## Reductions -/

/-- Karp (many-one) reduction: A ≤ₚ B.

    Language A reduces to B if there exists a poly-time computable function f
    such that: x ∈ A ↔ f(x) ∈ B -/
def IsKarpReducible (A B : Language α) : Prop :=
  ∃ f : (Fin 1 → List α) → List α,
    PolyTimeComputable f ∧ ∀ x, x ∈ A ↔ f ![x] ∈ B

notation:50 A " ≤ₚ " B => IsKarpReducible A B

/-! ## NP-Hardness and NP-Completeness -/

/-- NP-hard: Every language in NP Karp-reduces to this language.

    This is the core definition that we need for the BQP ≠ NP proof. -/
def IsNPHard (L : Language α) : Prop :=
  ∀ B ∈ ClassNP α, B ≤ₚ L

/-- NP-complete: In NP and NP-hard (the hardest problems in NP). -/
def IsNPComplete (L : Language α) : Prop :=
  L ∈ ClassNP α ∧ IsNPHard L

/-! ## Basic Properties -/

/-- P ⊆ NP (trivial: use empty certificate) -/
theorem ClassP_subset_ClassNP : ClassP α ⊆ ClassNP α := by
  intro L hL
  obtain ⟨f, hf_poly, hf_correct⟩ := hL
  -- Use f as verifier with certificate ignored
  -- The verifier ignores the certificate and just runs f on the input
  refine ⟨fun w => f ![w 0], 0, ?_, ?_⟩
  · -- Need to show this is PolyTimeComputable
    -- Composition: f(proj(w))
    exact SafeFunction.comp_closed (SafeFunction.normal_proj 0) hf_poly
  · -- Correctness
    intro x
    constructor
    · intro hx
      use []  -- Empty certificate
      -- x.length ^ 0 = 1
      simp only [List.length_nil, Nat.zero_le, true_and]
      rw [← hf_correct]
      exact hx
    · intro ⟨_, _, hV⟩
      rw [hf_correct]
      exact hV

/-- Karp reduction is transitive -/
theorem IsKarpReducible.trans {A B C : Language α}
    (hAB : A ≤ₚ B) (hBC : B ≤ₚ C) : A ≤ₚ C := by
  obtain ⟨f, hf, hcorr_f⟩ := hAB
  obtain ⟨g, hg, hcorr_g⟩ := hBC
  -- Compose f and g
  refine ⟨fun v => g ![f v], ?_, ?_⟩
  · -- Show composition is poly-time
    exact SafeFunction.comp_closed hf hg
  · -- Correctness
    intro x
    rw [hcorr_f, hcorr_g]

/-- If L is NP-complete and L ≤ₚ L', then L' is NP-hard -/
theorem IsNPComplete.reduces_to_implies_hard
    {L L' : Language α} (hL : IsNPComplete L) (h : L ≤ₚ L') : IsNPHard L' := by
  intro B hB_in_NP
  have hBL := hL.2 B hB_in_NP
  exact IsKarpReducible.trans hBL h

/-! ## SAT Problem (Placeholder for Cook-Levin) -/

namespace SAT

/-- A literal: variable index with polarity -/
structure Literal (n : ℕ) where
  var : Fin n
  polarity : Bool

/-- Evaluate a literal under an assignment -/
def Literal.eval {n : ℕ} (l : Literal n) (assignment : Fin n → Bool) : Bool :=
  if l.polarity then assignment l.var else !assignment l.var

/-- A 3-clause: disjunction of exactly 3 literals -/
structure Clause3 (n : ℕ) where
  l1 : Literal n
  l2 : Literal n
  l3 : Literal n

/-- Evaluate a 3-clause under an assignment -/
def Clause3.eval {n : ℕ} (c : Clause3 n) (assignment : Fin n → Bool) : Bool :=
  c.l1.eval assignment || c.l2.eval assignment || c.l3.eval assignment

/-- A 3-SAT instance: conjunction of 3-clauses -/
structure Instance (n : ℕ) where
  clauses : List (Clause3 n)

/-- A SAT instance is satisfiable if some assignment satisfies all clauses -/
def Instance.isSatisfiable {n : ℕ} (φ : Instance n) : Prop :=
  ∃ assignment : Fin n → Bool,
    φ.clauses.all (fun c => c.eval assignment) = true

end SAT

/-- 3-SAT as a language (for a fixed encoding).
    We define this axiomatically - the encoding details don't affect the theory. -/
axiom SAT3Language : Language Bool

/-- Cook-Levin Theorem (AXIOMATIZED for now):
    3-SAT is NP-complete.

    Full proof would require ~10K lines following AFP/Isabelle formalization. -/
axiom cookLevin : IsNPComplete (α := Bool) SAT3Language
