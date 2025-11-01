"""

debug_sat_matrices.py

CRITICAL DEBUGGING: Why are all gap profiles identical?

This manually constructs and verifies SAT Hamiltonians
to find the bug causing the arithmetic sequence gaps.
"""

import numpy as np
from qlto_sat_scaffolding import SATProblem, SATClause


def manual_sat_matrix(problem: SATProblem) -> np.ndarray:
    """
    Manually construct SAT Hamiltonian matrix.
    Double-check against sat_to_matrix from test file.
    """
    N = problem.n_vars
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=float)
    
    print(f"\nManual construction for N={N}, M={len(problem.clauses)} clauses")
    print(f"Matrix dimension: {dim}x{dim}")
    
    # For each basis state
    for x in range(dim):
        # Convert to assignment
        assignment = {}
        bits = []
        for i in range(N):
            bit = bool((x >> i) & 1)
            assignment[i+1] = bit
            bits.append('1' if bit else '0')
        
        # Count violated clauses
        violated = 0
        satisfied_list = []
        for clause in problem.clauses:
            is_sat = clause.is_satisfied(assignment)
            satisfied_list.append('✓' if is_sat else '✗')
            if not is_sat:
                violated += 1
        
        # Diagonal entry is penalty
        H[x, x] = float(violated)
        
        if x < 8:  # Print first few states
            print(f"  State {''.join(bits)}: {violated} violated " + 
                  f"{''.join(satisfied_list)}")
    
    # Analyze eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\nEigenvalue spectrum:")
    print(f"  Min: {eigenvalues[0]:.6f}")
    print(f"  Max: {eigenvalues[-1]:.6f}")
    print(f"  First 10: {eigenvalues[:10]}")
    
    # Count ground states
    E_min = eigenvalues[0]
    ground_count = np.sum(np.abs(eigenvalues - E_min) < 1e-6)
    print(f"  Ground degeneracy: {ground_count}")
    
    return H


def test_single_clause():
    """Test the simplest case: one clause"""
    print("\n" + "="*80)
    print("TEST 1: Single Clause (x1 OR x2 OR x3)")
    print("="*80)
    
    clause = SATClause((1, 2, 3))
    problem = SATProblem(3, [clause])
    
    H = manual_sat_matrix(problem)
    
    print("\nExpected:")
    print("  - 7 states with energy 0 (all except 000)")
    print("  - 1 state with energy 1 (the 000 state)")
    print("  - Eigenvalues: [0,0,0,0,0,0,0,1]")


def test_two_clauses():
    """Test two clauses"""
    print("\n" + "="*80)
    print("TEST 2: Two Clauses (x1 OR x2 OR x3) AND (x1 OR NOT x2 OR x3)")
    print("="*80)
    
    clause1 = SATClause((1, 2, 3))
    clause2 = SATClause((1, -2, 3))
    problem = SATProblem(3, [clause1, clause2])
    
    H = manual_sat_matrix(problem)
    
    print("\nExpected:")
    print("  - Some states with energy 0 (satisfy both)")
    print("  - Some states with energy 1 (violate one)")
    print("  - Some states with energy 2 (violate both)")


def test_scaffolding_construction():
    """Test how scaffolding builds H(s) = H_seed + s*H_rest"""
    print("\n" + "="*80)
    print("TEST 3: Scaffolding H(s) Construction")
    print("="*80)
    
    # Create a small problem
    clauses = [
        SATClause((1, 2, 3)),      # Seed
        SATClause((1, -2, 3)),     # Rest 1
        SATClause((-1, 2, -3)),    # Rest 2
    ]
    problem = SATProblem(3, clauses)
    
    # H_seed: just first clause
    problem_seed = SATProblem(3, [clauses[0]])
    H_seed = manual_sat_matrix(problem_seed)
    
    # H_rest: other clauses
    problem_rest = SATProblem(3, clauses[1:])
    H_rest = manual_sat_matrix(problem_rest)
    
    # H_full: all clauses
    H_full = manual_sat_matrix(problem)
    
    # Check if H_full = H_seed + H_rest
    H_sum = H_seed + H_rest
    print("\n" + "="*80)
    print("VERIFICATION: Does H_full = H_seed + H_rest?")
    print("="*80)
    print(f"Max difference: {np.max(np.abs(H_full - H_sum))}")
    
    if np.allclose(H_full, H_sum):
        print("✓ YES: Matrices match!")
    else:
        print("✗ NO: Matrices don't match! BUG IN CONSTRUCTION!")
        print(f"\nH_seed diagonal:\n{np.diag(H_seed)}")
        print(f"\nH_rest diagonal:\n{np.diag(H_rest)}")
        print(f"\nH_sum diagonal:\n{np.diag(H_sum)}")
        print(f"\nH_full diagonal:\n{np.diag(H_full)}")
    
    # Now test H(s) at different s values
    print("\n" + "="*80)
    print("GAP ALONG PATH")
    print("="*80)
    
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        H_s = H_seed + s * H_rest
        eigenvalues = np.linalg.eigvalsh(H_s)
        
        E_min = eigenvalues[0]
        ground_mask = np.abs(eigenvalues - E_min) < 1e-6
        excited = eigenvalues[~ground_mask]
        
        if len(excited) > 0:
            gap = excited[0] - E_min
            deg = np.sum(ground_mask)
            print(f"s={s:.2f}: gap={gap:.6f}, E_min={E_min:.6f}, deg={deg}")
        else:
            print(f"s={s:.2f}: All states degenerate!")


def test_random_instance():
    """Test an actual random instance"""
    print("\n" + "="*80)
    print("TEST 4: Random Instance from generate_random_3sat")
    print("="*80)
    
    from qlto_sat_scaffolding import generate_random_3sat
    
    problem = generate_random_3sat(4, 16, seed=42)
    
    print(f"\nProblem: N=4, M=16")
    print(f"First 5 clauses:")
    for i, clause in enumerate(problem.clauses[:5]):
        print(f"  {i}: {clause.literals}")
    
    # Full problem
    H_full = manual_sat_matrix(problem)
    
    # Seed only
    print("\n" + "-"*80)
    print("SEED HAMILTONIAN (first clause only)")
    print("-"*80)
    problem_seed = SATProblem(4, [problem.clauses[0]])
    H_seed = manual_sat_matrix(problem_seed)
    
    # Rest
    print("\n" + "-"*80)
    print("REST HAMILTONIAN (all other clauses)")
    print("-"*80)
    problem_rest = SATProblem(4, problem.clauses[1:])
    H_rest = manual_sat_matrix(problem_rest)
    
    # Check sum
    print("\n" + "="*80)
    print("CHECKING H_full = H_seed + H_rest")
    print("="*80)
    H_sum = H_seed + H_rest
    max_diff = np.max(np.abs(H_full - H_sum))
    print(f"Max difference: {max_diff}")
    
    if np.allclose(H_full, H_sum, atol=1e-10):
        print("✓ Construction is correct!")
    else:
        print("✗ BUG: Construction is wrong!")
    
    # Now test gaps
    print("\n" + "="*80)
    print("GAP PROFILE FOR THIS INSTANCE")
    print("="*80)
    
    s_values = np.linspace(0, 1, 11)
    for s in s_values:
        H_s = H_seed + s * H_rest
        eigenvalues = np.linalg.eigvalsh(H_s)
        
        E_min = eigenvalues[0]
        ground_mask = np.abs(eigenvalues - E_min) < 1e-6
        excited = eigenvalues[~ground_mask]
        
        if len(excited) > 0:
            gap = excited[0] - E_min
            deg = np.sum(ground_mask)
            print(f"s={s:.2f}: gap={gap:.8f}, E_min={E_min:.2f}, deg={deg}")


def main():
    """Run all debugging tests"""
    print("="*80)
    print("DEBUGGING SAT HAMILTONIAN CONSTRUCTION")
    print("="*80)
    
    test_single_clause()
    test_two_clauses()
    test_scaffolding_construction()
    test_random_instance()
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("""
If all tests pass:
  → Matrix construction is correct
  → The arithmetic sequence is real (needs explanation!)
  
If tests fail:
  → Found the bug in matrix construction
  → Need to fix and re-run experiments
""")


if __name__ == "__main__":
    main()

