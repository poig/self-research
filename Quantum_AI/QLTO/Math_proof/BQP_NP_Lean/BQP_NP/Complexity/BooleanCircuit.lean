/-
  BooleanCircuit.lean: Boolean Circuits for Cook-Levin Proof

  Defines:
  - Gate types (AND, OR, NOT, INPUT)
  - Circuit structure (DAG of gates)
  - Evaluation semantics
  - Conversion to CNF (Tseitin transformation)
-/

import Mathlib.Data.List.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Logic.Basic
import BQP_NP.Complexity.ClassNP

namespace BooleanCircuit

/-- Types of gates allowed in the circuit -/
inductive GateType
| CONST (b : Bool)
| INPUT (idx : ℕ)     -- Input bit at distinct index
| NOT                 -- Takes 1 input
| AND                 -- Takes 2 inputs
| OR                  -- Takes 2 inputs

/-- A gate in the circuit -/
structure Gate where
  type : GateType
  inputs : List ℕ     -- Indices of input gates (must be < index of this gate)

/-- A boolean circuit is a list of gates.
    Implicitly, gate `i` is the `i`-th element of the list.
    We enforce acyclicity by requiring gates to only reference indices `j < i`. -/
structure Circuit where
  gates : List Gate
  output : ℕ          -- Index of the output gate
  valid_dag : ∀ (i : ℕ) (g : Gate),
    gates[i]? = some g → ∀ InputIdx ∈ g.inputs, InputIdx < i
  valid_output : output < gates.length

/-- Get list element with default -/
def List.getD' (l : List α) (n : ℕ) (default : α) : α :=
  match l[n]? with
  | some x => x
  | none => default

/-- Evaluate the circuit on a given input assignment -/
def eval (c : Circuit) (assignment : ℕ → Bool) : Option Bool :=
  let values : List Bool := c.gates.foldl (fun acc gate =>
    let val := match gate.type with
      | GateType.CONST b => b
      | GateType.INPUT idx => assignment idx
      | GateType.NOT =>
          match gate.inputs with
          | [i] => !(List.getD' acc i false)
          | _ => false
      | GateType.AND =>
          match gate.inputs with
          | [i, j] => (List.getD' acc i false) && (List.getD' acc j false)
          | _ => false
      | GateType.OR =>
          match gate.inputs with
          | [i, j] => (List.getD' acc i false) || (List.getD' acc j false)
          | _ => false
    acc ++ [val]
  ) []
  values[c.output]?

/-- Size of the circuit (number of gates) -/
def size (c : Circuit) : ℕ := c.gates.length

/-! ## Tseitin Transformation -/

open SAT

/-- Helper to create a literal from a gate index -/
def mkLit {n : ℕ} (idx : Fin n) (pol : Bool) : Literal n :=
  { var := idx, polarity := pol }

/-- Helper to pad clauses to 3 literals by repeating the last one -/
def mkClause3 {n : ℕ} (l1 l2 : Literal n) (l3 : Option (Literal n) := none) : Clause3 n :=
  let L3 := l3.getD l2
  { l1 := l1, l2 := l2, l3 := L3 }

/-- Generate clauses for a specific gate -/
def gateClauses {n : ℕ} (gateIdx : Fin n) (g : Gate) : List (Clause3 n) :=
  let out := mkLit gateIdx true
  let not_out := mkLit gateIdx false
  match g.type with
  | GateType.CONST b =>
      let l := mkLit gateIdx b
      [mkClause3 l l]
  | GateType.INPUT _ => []
  | GateType.NOT =>
      match g.inputs with
      | [i] =>
          if h : i < n then
             let in1 := mkLit ⟨i, h⟩ true
             let not_in1 := mkLit ⟨i, h⟩ false
             [mkClause3 not_out not_in1, mkClause3 out in1]
          else []
      | _ => []
  | GateType.AND =>
      match g.inputs with
      | [i, j] =>
          if h : i < n ∧ j < n then
             let in1 := mkLit ⟨i, h.1⟩ true
             let in2 := mkLit ⟨j, h.2⟩ true
             let not_in1 := mkLit ⟨i, h.1⟩ false
             let not_in2 := mkLit ⟨j, h.2⟩ false
             [mkClause3 not_out in1, mkClause3 not_out in2, mkClause3 not_in1 not_in2 (some out)]
          else []
      | _ => []
  | GateType.OR =>
      match g.inputs with
      | [i, j] =>
           if h : i < n ∧ j < n then
             let in1 := mkLit ⟨i, h.1⟩ true
             let in2 := mkLit ⟨j, h.2⟩ true
             let not_in1 := mkLit ⟨i, h.1⟩ false
             let not_in2 := mkLit ⟨j, h.2⟩ false
             [mkClause3 not_in1 out, mkClause3 not_in2 out, mkClause3 not_out in1 (some in2)]
           else []
      | _ => []

/-- Convert Circuit to SAT Instance using Tseitin -/
def toSAT (c : Circuit) : SAT.Instance c.gates.length :=
  let n := c.gates.length
  let clauses := (List.range n).flatMap (fun i =>
    if h : i < n then
       gateClauses ⟨i, h⟩ (c.gates[i]'h)
    else [])
  -- Add constraint that output gate must be true
  let output_lt_n : c.output < n := c.valid_output
  let outClause := mkClause3 (mkLit ⟨c.output, output_lt_n⟩ true) (mkLit ⟨c.output, output_lt_n⟩ true)
  { clauses := clauses ++ [outClause] }

/-! ## Correctness Theorem -/

/-- Tseitin Transformation Correctness:
    The circuit evaluates to true on some input iff the SAT instance is satisfiable. -/
theorem tseitin_correct (c : Circuit) :
    (∃ input : ℕ → Bool, eval c input = some true) ↔ (toSAT c).isSatisfiable := by
  sorry

end BooleanCircuit
