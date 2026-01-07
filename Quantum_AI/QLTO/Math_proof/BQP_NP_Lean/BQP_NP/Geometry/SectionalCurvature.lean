/-
  SectionalCurvature.lean: Curvature geometry on Lie Groups

  We define the geometry of a Lie Group G equipped with a left-invariant metric.
-/

import Mathlib.Algebra.Lie.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.RCLike.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension -- For LinearMap -> ContinuousLinearMap
import Mathlib.Analysis.Normed.Module.FiniteDimension -- For FiniteDimensional -> CompleteSpace
import Mathlib.Analysis.InnerProductSpace.PiL2 -- For Matrix NormedAddCommGroup
import Mathlib.Analysis.Matrix.Normed -- For Matrix Norm
import BQP_NP.Basic.LieAlgebra

open Module
open InnerProductSpace
open ContinuousLinearMap
open LieAlgebra

open scoped Matrix


namespace BQP_NP.Geometry

/--
  Typeclass for Lie groups with compatible metric and Lie algebra structure.
  Resolves diamond inheritance issues between NormedAddCommGroup and LieRing.
-/
class CompatibleLieGeometry (𝕜 L : Type*) [RCLike 𝕜] [NormedAddCommGroup L]
  extends InnerProductSpace 𝕜 L, Bracket L L where
  lie_add : ∀ (x : L) (y z : L), ⁅x, y + z⁆ = ⁅x, y⁆ + ⁅x, z⁆
  lie_smul : ∀ (c : 𝕜) (x y : L), ⁅x, c • y⁆ = c • ⁅x, y⁆


/-! ## 1. The Metric Adjoint -/

/--
  The metric adjoint of the Lie bracket operator.
-/
noncomputable def ad_star
  {𝕜 L : Type*} [RCLike 𝕜] [NormedAddCommGroup L] [CompatibleLieGeometry 𝕜 L] [FiniteDimensional 𝕜 L]
  (x : L) : L →L[𝕜] L :=
  -- Finite dimensionality implies completeness
  have : CompleteSpace L := FiniteDimensional.complete 𝕜 L
  let ad_x_linear : L →ₗ[𝕜] L := {
    toFun := fun y => ⁅x, y⁆
    map_add' := fun y z => CompatibleLieGeometry.lie_add x y z
    map_smul' := fun c y => CompatibleLieGeometry.lie_smul c x y
  }
  let ad_x_cont : L →L[𝕜] L := LinearMap.toContinuousLinearMap ad_x_linear
  ContinuousLinearMap.adjoint ad_x_cont

/-! ## 2. Levi-Civita Connection -/

/--
  The Levi-Civita connection ∇_X Y.
-/
noncomputable def nabla
  {𝕜 L : Type*} [RCLike 𝕜] [NormedAddCommGroup L] [CompatibleLieGeometry 𝕜 L] [FiniteDimensional 𝕜 L]
  (x y : L) : L :=
  let t1 := ⁅x, y⁆
  let t2 := ad_star (𝕜 := 𝕜) (L := L) x y
  let t3 := ad_star (𝕜 := 𝕜) (L := L) y x
  (1 / 2 : 𝕜) • (t1 - t2 - t3)

/-! ## 3. Curvature Tensor -/

/--
  Riemann Curvature Tensor R(X,Y)Z.
-/
noncomputable def riemannian_curvature
  {𝕜 L : Type*} [RCLike 𝕜] [NormedAddCommGroup L] [CompatibleLieGeometry 𝕜 L] [FiniteDimensional 𝕜 L]
  (x y z : L) : L :=
  nabla (𝕜 := 𝕜) (L := L) x (nabla (𝕜 := 𝕜) (L := L) y z) -
  nabla (𝕜 := 𝕜) (L := L) y (nabla (𝕜 := 𝕜) (L := L) x z) -
  nabla (𝕜 := 𝕜) (L := L) ⁅x, y⁆ z

/-! ## 4. Sectional Curvature -/

/--
  Sectional Curvature K(X,Y).
  K(X,Y) = ⟪R(X,Y)Y, X⟫ / (‖X‖²‖Y‖² - |⟪X,Y⟫|²)

  This formula is valid for any linearly independent pair X, Y.
-/
noncomputable def sectionalCurvature
  {𝕜 L : Type*} [RCLike 𝕜] [NormedAddCommGroup L] [CompatibleLieGeometry 𝕜 L] [FiniteDimensional 𝕜 L]
  (x y : L) : 𝕜 :=
  let num := inner (𝕜 := 𝕜) (riemannian_curvature (𝕜 := 𝕜) (L := L) x y y) x
  let x_norm_sq := (‖x‖ : 𝕜) ^ 2
  let y_norm_sq := (‖y‖ : 𝕜) ^ 2
  let inner_xy := inner (𝕜 := 𝕜) x y
  let den := x_norm_sq * y_norm_sq - inner_xy * (star inner_xy)
  num / den


end BQP_NP.Geometry
