# Mathematical Roadmap for Formalizing BQP ≠ NP via Dynamical Lie Algebras

**Author:** Jun Liang Tan (University of Queensland)
**Date:** December 2025 (Updated January 2026)
**Status:** Active Development

---

## Executive Summary

This document provides a **rigorous mathematical roadmap** for proving BQP ≠ NP using Dynamical Lie Algebras (DLAs). It includes:
1. The mathematical foundations and key conjectures
2. A detailed **Lean formalization plan** based on Mathlib capabilities
3. Clear distinction between **provable** vs. **axiomatized** components

---

## 0. Foundational Principle: Non-Commutativity as Complexity

> **Master Axiom:** Computational complexity is determined by commutator structure.

### The Core Insight

```
IF [subproblems] = 0 (commute):
  → Shared eigenbasis exists
  → Problems solve independently  
  → POLYNOMIAL TIME (P)

IF [subproblems] ≠ 0 (don't commute):
  → No shared eigenbasis
  → Cannot decompose
  → EXPONENTIAL TIME (NP-hard)

NP-COMPLETENESS = MAXIMAL NON-COMMUTATIVITY
  Every variable couples to others
  No separable substructure exists
  This is IRREDUCIBLE — not a limitation of algorithms
```

### Connection to Physics

| Physics | Computation | Mathematical Object |
|:---|:---|:---|
| `[x, p] = iℏ` → Quantum mechanics | `[subproblems] ≠ 0` → NP-hard | Lie bracket |
| `[∇_μ, ∇_ν] ≠ 0` → Curvature | Curved solution landscape | Sectional curvature |
| Spectral gap → Phase transitions | Spectral gap → Complexity | Killing form eigenvalues |

---

## 1. Mathematical Definitions (Rigorously Formalizable)

### 1.1 Pauli Operators and Strings

**Definition 1.1 (Pauli Operators):**
The single-qubit Pauli operators are:
$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad
Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Lean File:** `BQP_NP/Basic/PauliOperators.lean`
**Mathlib Status:** ✅ Can define using `Matrix (Fin 2) (Fin 2) ℂ`

**Definition 1.2 (Pauli String):**
An n-qubit Pauli string is $P = P_1 ⊗ P_2 ⊗ ... ⊗ P_n$ where each $P_i ∈ \{I, X, Y, Z\}$.

**Lean File:** `BQP_NP/Basic/PauliOperators.lean`
**Mathlib Status:** ✅ Define as `Fin n → Pauli`

**Theorem 1.1 (Pauli Orthogonality):**
$$\text{Tr}(P^\dagger Q) = 2^n \delta_{PQ}$$

**Lean File:** `BQP_NP/Basic/PauliBasis.lean`
**Mathlib Status:** ✅ **PROVABLE** with `Matrix.trace` and Kronecker product properties.

**Proof Sketch:**
1. `Tr(σ_i† σ_j) = 2 δ_{ij}` for single-qubit Paulis (direct calculation)
2. `Tr(A ⊗ B) = Tr(A) · Tr(B)` (Mathlib: needs formalization)
3. Combine by induction on n

---

### 1.2 Dynamical Lie Algebra (DLA)

**Definition 1.3 (Hamiltonian):**
A Hamiltonian $H$ is a Hermitian matrix: $H = H^\dagger$.

**Lean File:** `BQP_NP/Basic/LieAlgebra.lean`
**Mathlib Status:** ✅ Use `Matrix.IsHermitian`

**Definition 1.4 (DLA):**
Given generators $\{H, H_{\text{mixer}}\}$, the DLA is the smallest Lie subalgebra containing them:
$$\mathfrak{g} = \text{span}\{H, H_{\text{mixer}}, [H, H_{\text{mixer}}], [H, [H, H_{\text{mixer}}]], ...\}$$

**Lean File:** `BQP_NP/Basic/LieAlgebra.lean`
**Mathlib Status:** ✅ Use `LieSubalgebra.lieSpan` from Mathlib

**Definition 1.5 (Killing Form):**
$$K(X, Y) = \text{Tr}(\text{ad}_X \circ \text{ad}_Y)$$
where $\text{ad}_X(Y) = [X, Y]$.

**Lean File:** `BQP_NP/Basic/LieAlgebra.lean`
**Mathlib Status:** ✅ **AVAILABLE** as `LieAlgebra.killingForm` in `Mathlib.Algebra.Lie.Killing`

**Definition 1.6 (Spectral Gap):**
$$\lambda_{\min}(\mathfrak{g}) = \min\{\lambda : \lambda \text{ is eigenvalue of } K, \lambda \neq 0\}$$

**Lean File:** `BQP_NP/Basic/LieAlgebra.lean`
**Mathlib Status:** ⚠️ Eigenvalues exist in Mathlib, but specific spectral gap properties need formalization.

---

### 1.3 Adjoint Graph

**Definition 1.7 (Adjoint Graph):**
- Vertices: Pauli strings in the DLA basis
- Edges: $(P, Q) \in E \iff [P, Q] \neq 0$

**Lean File:** `BQP_NP/Year2/AdjointGraph.lean`
**Mathlib Status:** ✅ Use `SimpleGraph` from `Mathlib.Combinatorics.SimpleGraph.Basic`

**Definition 1.8 (Cheeger Constant):**
$$h(G) = \min_{S \subset V, |S| \leq |V|/2} \frac{|∂S|}{|S|}$$

**Lean File:** `BQP_NP/Year2/AdjointGraph.lean`
**Mathlib Status:** ❌ **NOT IN MATHLIB** - Must define ourselves

---

## 2. Key Theorems (The Proof Structure)

### 2.1 Theorem 1: Spectral Gap Collapse

**Statement:**
For any Hamiltonian family $\{H_N\}$ encoding an NP-complete problem:
$$\lambda_{\min}(\mathfrak{g}_N) \leq e^{-cN}$$
for some constant $c > 0$.

**Proof Strategy:**
1. Show NP-hardness implies maximal non-commutativity
2. Maximal non-commutativity → structure constants follow GOE statistics
3. GOE statistics → exponentially small eigenvalues (Wigner's theorem)

**Mathlib Status:**
- Step 1: ⚠️ Requires reduction formalization
- Step 2: ❌ GOE statistics NOT IN MATHLIB
- Step 3: ❌ Wigner's theorem NOT IN MATHLIB

**Lean Approach:** Axiomatize the GOE → spectral gap connection, prove reduction rigorously.

---

### 2.2 Theorem 2: Curvature Explosion

**Statement:**
For NP-hard DLAs, the sectional curvature satisfies:
$$\kappa(X, Y) \leq -e^{\alpha N}$$
for some directions $X, Y$ and constant $\alpha > 0$.

**Proof Strategy:**
1. Use O'Neill formula for Lie group curvature:
   $$\kappa(X, Y) = \frac{1}{4}\|[X, Y]\|^2 - K(\text{ad}_Z^2)$$
   where $K$ is the Killing form.
2. Show small spectral gap → large $\|[X, Y]\|^2$ relative to metric.

**Mathlib Status:**
- O'Neill formula: ❌ **NOT IN MATHLIB**
- Sectional curvature: ❌ **NOT IN MATHLIB**
- Riemannian metric on Lie groups: ⚠️ Partial (RiemannianBundle exists)

**Lean Approach:** 
Define sectional curvature via the O'Neill formula directly for matrix Lie groups. This requires ~500 lines of new formalization.

---

### 2.3 Theorem 3: Localization Bottleneck

**Statement:**
The Inverse Participation Ratio (IPR) of adjoint eigenvectors satisfies:
$$\xi_N \geq e^{\beta N} \text{ (NP-hard)}, \quad \xi_N = O(1) \text{ (BQP)}$$

**Proof Strategy:**
1. Small Cheeger constant → eigenvector localization (Cheeger-Buser inequality)
2. Localization → exponential IPR

**Mathlib Status:**
- Cheeger constant: ❌ **NOT IN MATHLIB**
- Cheeger-Buser inequality: ❌ **NOT IN MATHLIB**
- IPR definition: ✅ Can define

**Lean Approach:** Define Cheeger constant and IPR rigorously. State Cheeger-Buser as axiom.

---

### 2.4 Main Theorem: BQP ≠ NP

**Statement:**
$$\text{BQP} \neq \text{NP}$$

**Proof (by contradiction):**
1. Assume BQP ⊇ NP-complete.
2. Then ∃ poly-depth quantum circuit solving 3-SAT.
3. 3-SAT encodes to Hamiltonian $H$ with NP-hard DLA (Reduction).
4. By Theorem 1: $\lambda_{\min} \leq e^{-cN}$
5. By Theorem 2: $\kappa \leq -e^{\alpha N}$
6. By Nielsen Geometry: Circuit depth $\geq$ geodesic distance
7. By Toponogov Comparison: Geodesic distance $\geq e^{N}$ (from negative curvature)
8. **CONTRADICTION:** Poly-depth circuit cannot traverse exponential distance.

**Mathlib Status:**
- Step 6 (Nielsen bound): ❌ **NOT IN MATHLIB**
- Step 7 (Toponogov): ❌ **NOT IN MATHLIB**

**Lean File:** `BQP_NP/MainTheorem/BQP_ne_NP.lean`

---

## 3. Lean Formalization Plan (Detailed)

### Phase 1: Rigorous Foundations (FULLY PROVABLE)

| File | Content | Mathlib Support | Status |
|:---|:---|:---|:---|
| `Basic/PauliOperators.lean` | Pauli matrices, strings, weight | ✅ Full | 🔧 Needs cleanup |
| `Basic/PauliBasis.lean` | Orthogonality, linear independence | ✅ Full | 🔧 Needs proofs |
| `Basic/LieAlgebra.lean` | DLA, commutator, using Mathlib | ✅ Use `LieSubalgebra` | 🔧 Refactor needed |

**Specific Theorems to Prove:**

```lean
-- In PauliBasis.lean
theorem trace_pauli_I : trace I = 2 := by native_decide
theorem trace_pauli_X : trace X = 0 := by native_decide  
theorem trace_pauli_Y : trace Y = 0 := by native_decide
theorem trace_pauli_Z : trace Z = 0 := by native_decide

theorem pauli_orthogonality (P Q : PauliString n) :
    trace (P.toMatrix.conjTranspose * Q.toMatrix) = 
    if P = Q then 2^n else 0 := by
  -- Proof by induction on n using trace_kronecker
  sorry -- TO BE PROVEN

-- In LieAlgebra.lean (using Mathlib)
instance : LieRing (Hamiltonian n) where
  bracket := matrixCommutator
  add_lie := by sorry -- TO BE PROVEN
  lie_add := by sorry -- TO BE PROVEN
  lie_self := by sorry -- TO BE PROVEN  
  leibniz_lie := by sorry -- TO BE PROVEN
```

---

### Phase 2: Killing Form (MOSTLY PROVABLE)

| File | Content | Mathlib Support | Status |
|:---|:---|:---|:---|
| `Year2/KillingEvaluation.lean` | Killing form for Paulis | ✅ Use `killingForm` | 🔧 Connect to Mathlib |

**Key Connection:**

```lean
-- Use Mathlib's Killing form
import Mathlib.Algebra.Lie.Killing

-- Our DLA is a Lie algebra over ℂ
-- The Killing form is available as `LieAlgebra.killingForm`

-- Goal: Prove our explicit formula matches Mathlib's
theorem killing_form_pauli_diagonal (P Q : PauliString n) :
    killingForm ℂ (PauliLieAlgebra n) P Q = 
    if P = Q then (structureConstantSum P) else 0 := by
  sorry -- TO BE PROVEN
```

---

### Phase 3: Geometry (REQUIRES NEW FORMALIZATION)

| File | Content | Mathlib Support | Status |
|:---|:---|:---|:---|
| `Geometry/SectionalCurvature.lean` | O'Neill formula | ❌ None | ❌ Must create |
| `Geometry/MatrixGeometry.lean` | Matrix manifold | ⚠️ Partial | ❌ Must create |
| `Year4/UnitaryManifold.lean` | $SU(2^n)$ as manifold | ⚠️ Partial | ❌ Must create |
| `Year4/ComplexityGeometry.lean` | Nielsen complexity | ❌ None | ❌ Must create |

**O'Neill Curvature Formula (to formalize):**

For a bi-invariant metric on a Lie group $G$:
$$\kappa(X, Y) = \frac{1}{4} \frac{\|[X, Y]\|^2}{\|X\|^2 \|Y\|^2 - \langle X, Y \rangle^2}$$

This is computable from commutator structure!

```lean
-- In SectionalCurvature.lean
def sectionalCurvatureONeill {n : ℕ} (X Y : Matrix (Fin (2^n)) (Fin (2^n)) ℂ) : ℝ :=
  let bracket := matrixCommutator X Y
  let bracketNormSq := (frobeniusNorm bracket)^2
  let xNormSq := (frobeniusNorm X)^2
  let yNormSq := (frobeniusNorm Y)^2
  let innerXY := (matrixInnerProduct X Y).re
  let denominator := xNormSq * yNormSq - innerXY^2
  if denominator = 0 then 0 else (1/4) * bracketNormSq / denominator
```

---

### Phase 4: Spectral Graph Theory (REQUIRES AXIOMS)

| File | Content | Mathlib Support | Status |
|:---|:---|:---|:---|
| `Year2/AdjointGraph.lean` | Adjoint graph | ✅ `SimpleGraph` | ✅ Done |
| `Year2/Localization.lean` | IPR, Cheeger | ❌ None | ⚠️ Axiomatized |
| `Year2/SpectralBridge.lean` | Cheeger-Killing | ❌ None | ⚠️ Axiomatized |

**What we CAN do:**
- Define Cheeger constant explicitly
- Define IPR explicitly
- State the Cheeger-Buser inequality as an axiom

**What requires axioms:**
- The actual Cheeger-Buser inequality proof (this is a major theorem in spectral graph theory)

---

### Phase 5: Reduction (PROVABLE WITH EFFORT)

| File | Content | Mathlib Support | Status |
|:---|:---|:---|:---|
| `Year3/Reduction.lean` | 3-SAT → Hamiltonian | ⚠️ Partial | 🔧 Refactor |
| `Year3/Gadget.lean` | Clause Hamiltonian | ✅ Matrices | 🔧 Refactor |

**Key Theorem:**

```lean
-- The clause Hamiltonian has ground energy 0 iff clause is satisfied
theorem clause_hamiltonian_spectrum (C : Clause n) (assignment : Fin n → Bool) :
    -- If clause is satisfied, ∃ ground state with E = 0
    (C.isSatisfied assignment = true) ↔ 
    (∃ ψ, ClauseMatrix C * ψ = 0 ∧ ψ ≠ 0) := by
  sorry -- TO BE PROVEN
```

---

### Phase 6: Main Theorem (DEPENDS ON PHASES 1-5)

| File | Content | Dependencies |
|:---|:---|:---|
| `MainTheorem/BQP_ne_NP.lean` | Final theorem | All previous |

**Structure:**

```lean
-- The main theorem, with explicitly stated axioms
theorem BQP_ne_NP 
    (ax_spectral_gap : SpectralGapCollapseAxiom)
    (ax_curvature : CurvatureExplosionAxiom)
    (ax_nielsen : NielsenComplexityLowerBound)
    (ax_toponogov : ToponogovComparison) :
    ¬(∀ L, IsNPComplete L → InBQP L) := by
  -- Proof using the axioms
  sorry
```

---

## 4. Mathlib Contribution Plan

To make this proof fully rigorous (no axioms), we would need to contribute:

### Priority 1: Sectional Curvature for Lie Groups
- **Estimated effort:** 1000-2000 lines
- **Prerequisites:** Riemannian manifold basics, connection theory
- **Mathlib PR:** Would be significant contribution

### Priority 2: Cheeger Constant and Inequality
- **Estimated effort:** 500-1000 lines  
- **Prerequisites:** Spectral graph theory
- **Mathlib PR:** Useful for many applications

### Priority 3: Comparison Theorems
- **Estimated effort:** 2000+ lines
- **Prerequisites:** Geodesics, Jacobi fields
- **Mathlib PR:** Major project, possibly separate library

---

## 5. Immediate Action Items

### Today:
1. [ ] Prove `trace_pauli_X = 0` rigorously (no sorry)
2. [ ] Prove `pauli_orthogonality` for n = 1
3. [ ] Connect our Hamiltonian type to Mathlib's LieRing

### This Week:
1. [ ] Formalize O'Neill curvature formula
2. [ ] Define Cheeger constant rigorously
3. [ ] Prove clause Hamiltonian spectrum (for simple cases)

### This Month:
1. [ ] Complete Phase 1 and 2 with no `sorry`
2. [ ] Document axioms precisely
3. [ ] Begin Mathlib contribution for sectional curvature

---

## 7. Rigor Audit: Issues That Others May Argue About

**Last Updated:** January 2026  
**Audit Status:** 🔴 40+ axioms, 20+ sorry blocks to address

### 7.1 Critical Issues: Axioms (40+)

These are unproven assumptions. Each must eventually be proven or documented as a conjecture.

#### Category A: Core Mathematical Conjectures (Hard to Prove)

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `spectral_gap_conjecture` | SpectralGap.lean | ❌ Unproven | Core conjecture: NP-hard → exponential gap |
| `bqp_poly_gap` | SpectralGap.lean | ❌ Unproven | BQP has polynomial spectral gap |
| `curvature_explosion_rate` | CurvatureExplosion.lean | ❌ Unproven | Negative curvature grows exponentially |
| `geodesic_lower_bound` | CurvatureExplosion.lean | ❌ Unproven | Geodesic distance exponential |

#### Category B: Spectral Graph Theory (Should Be Provable)

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `graphCheegerConstant` | SpectralBridge.lean | ❌ Define | Should define, not axiomatize |
| `killing_is_laplacian_like` | SpectralBridge.lean | ❌ Unproven | Killing form ~ graph Laplacian |
| `spectral_gap_bridge` | SpectralBridge.lean | ❌ Unproven | Cheeger inequality analog |
| `cheeger_implies_localization` | SpectralBridge.lean | ❌ Unproven | Low Cheeger → eigenvector localization |
| `complexity_implies_localization` | SpectralBridge.lean | ❌ Unproven | High complexity → localization |
| `localization_forces_depth` | Localization.lean | ❌ Unproven | Localization → circuit depth |

#### Category C: Algebraic (Should Be Provable with Mathlib)

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `pauliString_linearly_independent` | PauliBasis.lean | 🔧 Provable | Standard result, use Mathlib |
| `killing_form_pauli_diagonal` | KillingEvaluation.lean | 🔧 Provable | Use trace_pauli_string_prod |
| `operatorSparsity_exists` | PauliOperators.lean | 🔧 Provable | Pauli decomposition exists |
| `tfim_dla_dimension_bound` | TFIM.lean | ⚠️ Hard | TFIM DLA dimension |

#### Category D: Complexity Theory (Standard Results)

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `SAT3Language` | ClassNP.lean | 🔧 Define | Should define 3-SAT language |
| `cookLevin` | ClassNP.lean | ⚠️ Provable | Cook-Levin theorem (major effort) |
| `SAT_is_NP_complete` | BQP_ne_NP.lean | ⚠️ Provable | Use Cook-Levin |
| `InBQP` | BQP_ne_NP.lean | 🔧 Define | Should define BQP properly |
| `InNP` | BQP_ne_NP.lean | 🔧 Define | Should define NP properly |
| `IsNPComplete` | BQP_ne_NP.lean | 🔧 Define | Should define NP-completeness |

#### Category E: Reduction and Gadgets

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `clause_hamiltonian_exists` | Reduction.lean | 🔧 Provable | Clause → Hamiltonian |
| `reduction_rigor_lemma` | Reduction.lean | ⚠️ Hard | Main reduction correctness |
| `commutator_structure_preserved` | Reduction.lean | ⚠️ Hard | Structure constants preserved |
| `commutator_avalanche_lower_bound` | CommutatorAvalanche.lean | ❌ Unproven | Avalanche grows exponentially |
| `full_avalanche_theorem` | CommutatorAvalanche.lean | ❌ Unproven | Complete avalanche theorem |

#### Category F: Differential Geometry (Requires New Formalization)

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `UnitaryManifold` | ComplexityGeometry.lean | ❌ Define | Should define as SU(2^n) |
| `geodesicDistance` | ComplexityGeometry.lean | ❌ Define | Riemannian distance |
| `CircuitComplexity` | ComplexityGeometry.lean | ❌ Define | Nielsen complexity |
| `nielsen_complexity_bound` | ComplexityGeometry.lean | ❌ Unproven | Geodesic ≤ complexity |
| `curvature_volume_growth` | ComplexityGeometry.lean | ❌ Unproven | Volume growth from curvature |
| `no_wormhole_lemma` | ComplexityGeometry.lean | ❌ Unproven | No shortcuts |
| `matrix_ips_exists` | MatrixGeometry.lean | 🔧 Provable | Inner product space |
| `killing_trace_correspondence` | MatrixGeometry.lean | 🔧 Provable | Trace inner product |

#### Category G: Safe Recursion

| Axiom | File | Status | Notes |
|:------|:-----|:-------|:------|
| `safeHead` | SafeRecursion.lean | 🔧 Provable | Safe head function |
| `safeAppend` | SafeRecursion.lean | 🔧 Provable | Safe append function |
| `SafeFunction.comp_closed` | SafeRecursion.lean | 🔧 Provable | Composition closure |
| `polyTime_const` | SafeRecursion.lean | 🔧 Provable | Constants polytime |
| `denotes_safe` | SafeExpr.lean | 🔧 Provable | Expression safety |

---

### 7.2 Critical Issues: Sorry Blocks (20+)

| File | Count | Description |
|:-----|:------|:------------|
| `CompileCorrect.lean` | 8 | Circuit compilation correctness |
| `CookLevin.lean` | 4 | Cook-Levin reduction proofs |
| `BQP_ne_NP.lean` | 1 | Main theorem proof |
| `LieAlgebra.lean` | 1 | Lie ring instance proofs |
| `TFIM.lean` | 1 | TFIM model lemmas |
| `KillingEvaluation.lean` | 2 | Killing form calculations |
| `Gadget.lean` | 1 | Gadget construction |
| `CommutatorAvalanche.lean` | 1 | Avalanche lemmas |
| `BooleanCircuit.lean` | 1 | Circuit evaluation |
| `SafeExpr.lean` | 1 | Termination proof |
| `SafeToCircuit.lean` | 1 | Compilation termination |

---

### 7.3 Style Issues (May Be Controversial)

#### `simp` Usage (43 occurrences)

Some reviewers prefer explicit tactics. Files with `simp`:
- `PauliOperators.lean` - ✅ Cleaned in trace theorem
- `LieAlgebra.lean` - 🔧 Review needed
- `KillingEvaluation.lean` - 🔧 Review needed
- `AdjointGraph.lean` - 🔧 Review needed
- `Gadget.lean` - 🔧 Review needed
- Others (10+ files)

#### `open Classical` Usage (4 occurrences)

Constructivists may object. Files:
- `PauliOperators.lean` - ✅ Local scope (one theorem)
- `KillingEvaluation.lean` - 🔧 Review needed
- `AdjointGraph.lean` - 🔧 Review needed
- `CommutatorAvalanche.lean` - 🔧 Review needed

#### `norm_num` / `norm_cast` Usage

Generally acceptable but some prefer explicit calculation.

---

### 7.4 Priority Action Plan

#### Immediate (Can Be Done Now)

1. **Prove `pauliString_linearly_independent`** - Uses orthogonality
2. **Prove `killing_form_pauli_diagonal`** - Uses trace theorem
3. **Define `InBQP`, `InNP`, `IsNPComplete`** properly
4. **Remove `simp` from critical proofs**

#### Short Term (This Month)

5. **Prove safe recursion lemmas** - Standard proofs
6. **Prove circuit compilation** - Replace sorry in CompileCorrect.lean
7. **Clean up `open Classical`** - Use local scope

#### Medium Term (Next 3 Months)

8. **Formalize Cook-Levin** - Major effort, ~2000 lines
9. **Define differential geometry types** - Remove UnitaryManifold axiom
10. **Prove Cheeger inequality** - Spectral graph theory

#### Long Term (Research Level)

11. **Prove spectral gap conjecture** - Core mathematical result
12. **Prove curvature explosion** - Differential geometry
13. **Prove Nielsen complexity bound** - Geometric analysis

---

### 7.5 Axiom Classification Summary

| Status | Count | Category |
|:-------|:------|:---------|
| ❌ Hard Conjecture | 12 | Core theorems requiring new math |
| ⚠️ Provable with Effort | 10 | Standard but non-trivial |
| 🔧 Should Be Easy | 18 | Definitions or simple lemmas |
| **Total** | **40** | |

---

## 8. Bibliography

1. M. A. Nielsen et al., *Quantum Computation as Geometry*, Science **311**, 1133 (2006).
2. J. M. Aizenman, S. Molchanov, *Localization at Large Disorder*, Commun. Math. Phys. **157** (1993).
3. O'Neill, *The Fundamental Equations of a Submersion*, Michigan Math. J. **13** (1966).
4. Cheeger, *A Lower Bound for the Smallest Eigenvalue of the Laplacian*, Problems in Analysis (1970).
5. Mathlib Community, *Mathlib4*, https://github.com/leanprover-community/mathlib4

---

*End of Roadmap*

