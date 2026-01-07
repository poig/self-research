/-
  MatrixGeometry.lean: Bridge between Matrix Lie Algrebra and Curvature Geometry
-/
import BQP_NP.Basic.LieAlgebra
import BQP_NP.Geometry.SectionalCurvature
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Matrix.Normed

namespace BQP_NP.Geometry

open Matrix

/--
  Matrix inner product space using the Frobenius (trace) inner product.
-/
noncomputable instance matrixInnerProductSpace {n : ℕ} [Fintype (Fin n)] [DecidableEq (Fin n)] [NeZero n] :
    InnerProductSpace ℂ (Matrix (Fin n) (Fin n) ℂ) where
  inner A B := (A.conjTranspose * B).trace
  conj_inner_symm A B := by
    simp only [trace_mul_comm, trace_conjTranspose]
    rfl
  add_left A B C := by
    simp only [conjTranspose_add, add_mul, trace_add]
  smul_left A B c := by
    simp only [conjTranspose_smul, smul_mul, trace_smul]
    congr
  norm_sq_eq_re_inner A := by
    -- Frobenius norm sq is Tr(A†A)
    sorry

/--
  Matrices form a compatible Lie geometry.
-/
noncomputable instance matrixCompatibleGeometry {n : ℕ} [Fintype (Fin n)] [DecidableEq (Fin n)] [NeZero n] :
    CompatibleLieGeometry ℂ (Matrix (Fin n) (Fin n) ℂ) where
  lie_add := fun x => (adjointAction x).map_add
  lie_smul := fun c x => (adjointAction x).map_smul c
