/-
  AdjointGraph.lean: Formalizing the connectivity of the Lie algebra.
-/
import BQP_NP.Basic.PauliBasis
import BQP_NP.Basic.LieAlgebra
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.AdjMatrix
import Mathlib.Analysis.Complex.Basic

open Matrix
open Classical

namespace BQP_NP.Year2

/--
  The non-commutativity relation on Pauli strings.
  Two operators are connected if they do not commute.
-/
def nonCommuting {n : ℕ} (P Q : PauliString n) : Prop :=
  matrixCommutator P.toMatrix Q.toMatrix ≠ 0

noncomputable instance {n : ℕ} (P Q : PauliString n) : Decidable (nonCommuting P Q) :=
  Classical.dec _

/--
  The Adjoint Graph of the full n-qubit operator space.
  Vertices: Pauli strings.
  Edges: Non-zero commutator.
-/
def adjointGraph (n : ℕ) : SimpleGraph (PauliString n) where
  Adj P Q := nonCommuting P Q ∧ P ≠ Q
  symm P Q h := by
    simp [nonCommuting, matrixCommutator] at h
    simp [nonCommuting, matrixCommutator]
    constructor
    · intro h_zero
      apply h.1
      rw [← neg_sub] at h_zero
      rw [neg_eq_zero] at h_zero
      exact h_zero
    · exact Ne.symm h.2
  loopless P := by
    simp [nonCommuting, matrixCommutator]

/--
  The Adjacency Matrix of the Adjoint Graph.
  This is the discrete version of the Killing Operator.
-/
noncomputable def adjointAdjacencyMatrix (n : ℕ) :
    Matrix (PauliString n) (PauliString n) ℂ :=
  (adjointGraph n).adjMatrix ℂ

/-! ## Localization Metrics -/

/--
  Inverse Participation Ratio (IPR) of a vector v.
  IPR(v) = Σ |v_i|^4 / (Σ |v_i|^2)^2
  Measures the degree of localization.
-/
noncomputable def IPR {ι : Type*} [Fintype ι] (v : ι → ℂ) : ℝ :=
  let norm4 := ∑ i, (Complex.normSq (v i))^2
  let norm2 := ∑ i, Complex.normSq (v i)
  norm4 / (norm2^2)

/--
  Strong localization: IPR is bounded below by a constant (or grows).
  Delocalization: IPR → 0 as dimension grows.
-/
def IsLocalized {ι : Type*} [Fintype ι] (v : ι → ℂ) : Prop :=
  IPR v ≥ 1 / 4 -- Heuristic threshold

end BQP_NP.Year2
