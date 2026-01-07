/-
  SafeExpr.lean: Term-level representation of Safe Recursion (Bool specialized)

  Since SafeFunction is Prop-valued (a proof), we cannot pattern match on it
  to produce data (like circuits). This file defines SafeExpr - an inductive
  TYPE that represents the syntax of safe computations.

  Key insight: SafeExpr carries the computational content explicitly,
  while `SafeFunction f` is just a proof that `f` belongs to the class.

  For Cook-Levin, we compile SafeExpr to circuits.

  We specialize to Bool to avoid universe level issues.
-/

import BQP_NP.Complexity.SafeRecursion
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.List.Basic

open Matrix

/-! ## Term-Level Safe Expression Syntax -/

/-- SafeExpr: A term representing a safe computation over Bool alphabet.
    Unlike SafeFunction (which is Prop-valued), SafeExpr lives in Type
    so we can pattern match on it to build circuits. -/
inductive SafeExpr : (n s : ℕ) → Type where
  | nil : SafeExpr 0 0
  | consB (b : Bool) : SafeExpr 0 1
  | tailE : SafeExpr 0 1
  | normal_proj {n s : ℕ} (i : Fin n) : SafeExpr n s
  | safe_proj {n s : ℕ} (i : Fin s) : SafeExpr n s
  | condB (b : Bool) : SafeExpr 0 3
  | safe_comp {n₁ s₁ n₂ s₂ : ℕ}
      (f : Fin n₂ → SafeExpr n₁ 0)
      (g : Fin s₂ → SafeExpr n₁ s₁)
      (h : SafeExpr n₂ s₂) : SafeExpr n₁ s₁
  | safe_rec {n s : ℕ}
      (base : SafeExpr n s)
      (step : Bool → SafeExpr (n + 1) (s + 1)) : SafeExpr (n + 1) s

namespace SafeExpr

/-! ## Evaluation (Semantics) -/

/-- Size measure for termination -/
def size : {n s : ℕ} → SafeExpr n s → ℕ
  | _, _, nil => 1
  | _, _, consB _ => 1
  | _, _, tailE => 1
  | _, _, normal_proj _ => 1
  | _, _, safe_proj _ => 1
  | _, _, condB _ => 1
  | _, _, safe_comp f g h =>
      1 + h.size
  | _, _, safe_rec base step =>
      1 + base.size + (step true).size + (step false).size

/-- Evaluate a SafeExpr on given inputs -/
def eval {n s : ℕ} (e : SafeExpr n s)
    (normal : Fin n → List Bool) (safe : Fin s → List Bool) : List Bool :=
  match e with
  | .nil => []
  | .consB b => b :: safe 0
  | .tailE => (safe 0).tail
  | .normal_proj i => normal i
  | .safe_proj i => safe i
  | .condB b => if b ∈ (safe 0).head? then safe 1 else safe 2
  | .safe_comp f g h =>
      let new_normal := fun i => (f i).eval normal ![]
      let new_safe := fun j => (g j).eval normal safe
      h.eval new_normal new_safe
  | .safe_rec base step =>
      match hn : normal 0 with
      | [] => base.eval (vecTail normal) safe
      | head :: tl =>
          let ih := (SafeExpr.safe_rec base step).eval (vecCons tl (vecTail normal)) safe
          (step head).eval (vecCons tl (vecTail normal)) (vecCons ih safe)
  termination_by e.size + (List.sum (List.ofFn (fun i => (normal i).length)))
  decreasing_by all_goals sorry

/-! ## Connection to SafeFunction (axiomatized for now) -/

/-- Every SafeExpr denotes a SafeFunction -/
axiom denotes_safe {n s : ℕ} (e : SafeExpr n s) :
    SafeFunction (α := Bool) (fun normal safe => e.eval normal safe)

end SafeExpr
