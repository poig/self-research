/-
  PauliBasis.lean: Formalizing PauliStrings as a finite basis
-/
import BQP_NP.Basic.PauliOperators
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.Pi

/-- PauliString n is a finite type with 4^n elements -/
instance (n : ℕ) : Fintype (PauliString n) :=
  inferInstanceAs (Fintype (Fin n → Pauli))

theorem pauli_card : Fintype.card Pauli = 4 := by
  -- Derived Fintype for 4-constructor inductive should have card 4.
  show List.length ([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z]) = 4
  rfl

theorem pauliString_card (n : ℕ) : Fintype.card (PauliString n) = 4^n := by
  show Fintype.card (Fin n → Pauli) = 4^n
  rw [Fintype.card_fun, pauli_card, Fintype.card_fin]

/-- Decidable commutativity for Pauli strings -/
instance (n : ℕ) : DecidableRel (fun (P Q : PauliString n) =>
    ∃ i, (P i).mul (Q i) ≠ (Q i).mul (P i)) :=
  inferInstance

/-- Pauli strings are linearly independent (axiomatized for Year 2) -/
axiom pauliString_linearly_independent {n : ℕ} :
  LinearIndependent ℂ (PauliString.toMatrix (n := n))
