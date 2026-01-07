/-
  CurvatureExplosion.lean: Formalization of the Year 1 Curvature Conjecture.
-/
import BQP_NP.Geometry.MatrixGeometry
import BQP_NP.Basic.LieAlgebra
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Basic

open Matrix
open BQP_NP.Geometry
open BQP_NP.Basic.LieAlgebra

/--
  The Curvature Explosion Conjecture:
  For any NP-hard Hamiltonian family, there exists a direction p in the DLA
  such that the sectional curvature is exponentially negative.
-/
def curvature_explosion_conjecture : Prop :=
  ∀ (n : ℕ), ∃ (c : ℝ), c > 0 ∧
  ∀ (H H_mixer : Hamiltonian n),
    IsNPHardHamiltonian H →
    ∃ (X Y : Matrix (Fin (2 ^ n)) (Fin (2 ^ n)) ℂ),
      X ∈ DLA H H_mixer ∧ Y ∈ DLA H H_mixer ∧
      (sectionalCurvature X Y).re ≤ -Real.exp (c * n)
