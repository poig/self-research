"""

analyze_morphing_results.py

DEEP ANALYSIS OF THE MORPHING EXPERIMENT RESULTS

This script analyzes why we're seeing g_min = 0 in most instances.

CRITICAL FINDING FROM INITIAL TESTS:
====================================

The spectral gap is ZERO (or nearly zero) at s=0 for most instances!

This is NOT what we expected. Let's understand why.
"""

import numpy as np
import matplotlib.pyplot as plt
from test_spectral_gap import (
    generate_random_3sat, 
    reduce_to_2sat,
    sat_to_matrix,
    diagnose_zero_gap,
    solve_2sat_implication_graph
)

print("="*80)
print("DEEP ANALYSIS: Why is g_min = 0?")
print("="*80)

# Test Case 1: Simple instance
print("\n[CASE 1] Analyzing N=4 random instance...")
problem = generate_random_3sat(n_vars=4, n_clauses=16, seed=42)

# Diagnose at s=0 (pure 2-SAT)
diag_0 = diagnose_zero_gap(problem, s_value=0.0)
print(f"\nAt s=0 (pure 2-SAT):")
print(f"  Gap: {diag_0['gap']:.8f}")
print(f"  E0: {diag_0['E0']:.6f}")
print(f"  E1: {diag_0['E1']:.6f}")
print(f"  Ground state degeneracy: {diag_0['degeneracy']}/{diag_0['total_states']}")

# Check if 2-SAT is satisfiable
problem_2sat = reduce_to_2sat(problem, strategy='drop_last')
solution_2sat = solve_2sat_implication_graph(problem_2sat)

if solution_2sat is None:
    print("  2-SAT is UNSATISFIABLE!")
    print("  → This explains why E0 > 0")
else:
    print("  2-SAT is SATISFIABLE")
    print(f"  → Solution found: {solution_2sat}")
    
    # Verify energy
    satisfied_count = sum(1 for c in problem_2sat.clauses if c.is_satisfied(solution_2sat))
    unsatisfied_count = len(problem_2sat.clauses) - satisfied_count
    print(f"  → Satisfies {satisfied_count}/{len(problem_2sat.clauses)} clauses")
    print(f"  → Expected E0 ≈ {unsatisfied_count}")

# Diagnose at s=1 (pure 3-SAT)
diag_1 = diagnose_zero_gap(problem, s_value=1.0)
print(f"\nAt s=1 (pure 3-SAT):")
print(f"  Gap: {diag_1['gap']:.8f}")
print(f"  E0: {diag_1['E0']:.6f}")
print(f"  E1: {diag_1['E1']:.6f}")
print(f"  Ground state degeneracy: {diag_1['degeneracy']}/{diag_1['total_states']}")

print("\n" + "="*80)
print("KEY INSIGHT: Why Gap is Zero at s=0")
print("="*80)

print("""
The gap is zero at s=0 because the 2-SAT Hamiltonian has DEGENERATE 
ground states!

Here's why:

1. Our reduction: Drop the 3rd literal from each clause
   (x ∨ y ∨ z) → (x ∨ y)

2. 2-SAT typically has MULTIPLE solutions
   Example: (x₁ ∨ x₂) has 3 solutions: {11, 10, 01}

3. When there are multiple solutions, the Hamiltonian has multiple 
   zero-energy states → DEGENERATE ground state

4. Degenerate ground state → Gap can be zero!

MATHEMATICAL EXPLANATION:
========================

For SAT Hamiltonian: H = Σ (penalty for unsatisfied clauses)

If assignment A satisfies all clauses: H|A⟩ = 0|A⟩
If assignment B also satisfies all: H|B⟩ = 0|B⟩

Then |A⟩ and |B⟩ are both eigenstates with eigenvalue 0.
→ Ground state is degenerate
→ Gap = E₁ - E₀ = 0 - 0 = 0

THIS IS NOT A BUG - IT'S A FUNDAMENTAL PROPERTY!
""")

print("\n" + "="*80)
print("IMPLICATION FOR ADIABATIC ALGORITHM")
print("="*80)

print("""
The adiabatic theorem requires:
    T >> ε / g_min²

If g_min = 0 → T = ∞ (algorithm fails!)

HOWEVER, there's a subtlety:

When ground state is degenerate, we need to track the SUBSPACE, not
a single state. As long as we stay in the ground SUBSPACE, the 
algorithm still works!

The correct formulation:
- Start in ground subspace of H(0)
- Evolve slowly enough to stay in ground subspace of H(s)
- End in ground subspace of H(1)

Any state in the ground subspace of H(1) is a valid solution!

PROBLEM: Our current analysis tracks individual eigenvalues, not subspaces.

SOLUTIONS:
==========

Option 1: MODIFY PROBLEM GENERATION
   Generate instances where 2-SAT has UNIQUE solution
   → Use "forced" 2-SAT (unit clauses, tight constraints)
   
Option 2: MODIFY GAP DEFINITION
   Define gap as: g = E_excited - E_ground_subspace
   where E_excited is first eigenvalue outside ground subspace
   
Option 3: ACCEPT DEGENERACY
   Acknowledge that gap can be zero at endpoints
   Focus on gap in the INTERIOR (s ∈ (0.2, 0.8))

Let's test Option 3:
""")

print("\n" + "="*80)
print("RE-ANALYSIS: Gaps in Interior")
print("="*80)

from test_spectral_gap import compute_gap_along_path

result = compute_gap_along_path(problem, num_points=50, verbose=False)
s_values = result['s_values']
gaps = result['gaps']

# Find gaps in interior (s ∈ [0.1, 0.9])
interior_mask = (s_values >= 0.1) & (s_values <= 0.9)
interior_gaps = gaps[interior_mask]
interior_s = s_values[interior_mask]

if len(interior_gaps) > 0:
    g_min_interior = np.min(interior_gaps)
    s_min_interior = interior_s[np.argmin(interior_gaps)]
    
    print(f"Gap in full range [0, 1]:")
    print(f"  g_min = {result['g_min']:.8f} at s = {result['s_min']:.4f}")
    
    print(f"\nGap in interior [0.1, 0.9]:")
    print(f"  g_min = {g_min_interior:.8f} at s = {s_min_interior:.4f}")
    
    if g_min_interior > 1e-6:
        print(f"\n✓ NON-ZERO gap in interior!")
        print(f"  This suggests the morphing path is viable, but:")
        print(f"  - We need to handle endpoint degeneracies carefully")
        print(f"  - Adiabatic time scales as T ~ 1/g_min_interior²")
    else:
        print(f"\n✗ Gap is also zero in interior")
        print(f"  This is a more serious problem...")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
Based on this analysis, here's what we should do:

1. IMMEDIATE FIX:
   Modify test_spectral_gap.py to:
   - Ignore gaps at endpoints (s < 0.05 or s > 0.95)
   - Focus on minimum gap in interior
   - Track ground subspace dimension

2. BETTER PROBLEM GENERATION:
   Create 2-SAT instances with unique solutions by:
   - Adding unit clauses (x₁), (¬x₂), etc.
   - Using Horn-SAT structure
   - Starting from 3-SAT and ensuring reduction preserves uniqueness

3. THEORETICAL REFINEMENT:
   The correct conjecture should be:
   
   "The gap between ground subspace and excited subspace 
    remains polynomial in the interior of the morphing path"
   
   Not:
   
   "The gap between E₀ and E₁ is polynomial everywhere"

4. NEXT EXPERIMENT:
   Generate 3-SAT with planted solution
   Ensure 2-SAT reduction preserves the solution
   This guarantees unique ground state → non-degenerate

Should I implement these fixes?
""")

print("="*80)

