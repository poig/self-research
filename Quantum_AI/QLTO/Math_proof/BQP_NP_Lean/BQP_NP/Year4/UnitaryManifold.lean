/-
  UnitaryManifold.lean: Concrete geometry of the Unitary Group.

  This file replaces the "UnitaryManifold" axiom with the actual SU(2^n) group.
  We define:
  1. The manifold U(N) as `Matrix.unitaryGroup`.
  2. The tangent space at Identity as `skewAdjoint` matrices (Lie algebra u(N)).
  3. The Riemannian metric via the real part of the trace inner product.
-/

import Mathlib.LinearAlgebra.UnitaryGroup
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.InnerProductSpace.Basic

open Matrix
open Complex

namespace BQP_NP.Year4

variable {n : ℕ}

/--
  The Concrete Unitary Manifold U(2^n).
  We use the Mathlib definition `unitaryGroup`.
-/
def ConcreteUnitaryManifold (n : ℕ) := unitaryGroup (Fin (2^n)) ℂ

/--
  The Tangent Space at Identity (Lie Algebra u(N)).
  T_I U(N) = { H | H + H† = 0 } (Skew-Hermitian matrices).

  Mathlib doesn't have a pre-packaged "tangent space" def for matrix groups,
  so we define it as the subspace of matrices satisfying the condition.
-/
def TangentSpaceAtIdentity (n : ℕ) := { M : Matrix (Fin (2^n)) (Fin (2^n)) ℂ // M.IsSkewHermitian }

/--
  The Trace Inner Product / Metric on the Tangent Space.
  g(A, B) = Re(Tr(A† B)) = -Re(Tr(A B)) for skew-Hermitian A, B.

  This is the standard bi-invariant metric on U(N) used by Nielsen.
-/
noncomputable def traceMetric {n : ℕ} (A B : TangentSpaceAtIdentity n) : ℝ :=
  (matrixInnerProduct A.val B.val).re

/--
  Connection to Phase 8 (Complexity).
  We assert that this concrete manifold satisfies the axiomatic properties
  used in `ComplexityGeometry.lean`.
-/
instance : MetricSpace (ConcreteUnitaryManifold n) :=
  -- Mathlib groups inherit metric from embedding in Matrix space
  Subtype.metricSpace

end BQP_NP.Year4
