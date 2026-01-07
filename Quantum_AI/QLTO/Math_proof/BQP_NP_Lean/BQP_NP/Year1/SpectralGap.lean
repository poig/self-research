/-
  Year 1 Theorem: Spectral Gap Collapse

  Conjecture: For NP-hard Hamiltonians, the spectral gap of the Killing form
  decays exponentially with system size n.
-/
import BQP_NP.Basic.LieAlgebra
import Mathlib.Analysis.SpecialFunctions.Exp  -- For Real.exp

open Real

namespace BQP_NP.Year1

/-- A function f(n) is exponentially small if f(n) ≤ C * exp(-α n) for some α > 0. -/
def IsExponentiallySmall (f : ℕ → ℝ) : Prop :=
  ∃ (C α : ℝ), C > 0 ∧ α > 0 ∧ ∀ n, f n ≤ C * exp (-α * n)

/--
  The Spectral Gap Conjecture (Mathematical Version).

  For any sequence of Hamiltonians {H_n} that encodes an NP-complete problem,
  the spectral gap of the DLA's Killing form is exponentially small.
-/
axiom spectral_gap_conjecture :
  ∀ (H_seq : (n : ℕ) → Hamiltonian n) (Mixer_seq : (n : ℕ) → Hamiltonian n),
  (∀ n, IsNPHardHamiltonian (H_seq n)) →
  IsExponentiallySmall (fun n => spectralGapDLA (H_seq n) (Mixer_seq n))

/--
  The converse: BQP implies polynomial spectral gap.
  (Or at least sub-exponential).
-/
axiom bqp_poly_gap :
  ∀ (H_seq : (n : ℕ) → Hamiltonian n) (Mixer_seq : (n : ℕ) → Hamiltonian n) (A : (n:ℕ) → Ansatz n 1),
  (∀ n, PolytimeConvergence (A n) (H_seq n)) →
  ¬ IsExponentiallySmall (fun n => spectralGapDLA (H_seq n) (Mixer_seq n))

end BQP_NP.Year1
