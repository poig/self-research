"""

analyze_constant_gap.py

DIAGNOSTIC: Why is the scaffolding gap exactly constant?

This investigates whether the constant gap (0.06896552) is:
A) A genuine algorithmic property (breakthrough!)
B) A testing artifact (bug in test harness)
C) A mathematical coincidence
"""

import numpy as np
import random
from qlto_sat_scaffolding import SATProblem, SATClause, generate_random_3sat
from test_scaffolding_gap import sat_to_matrix, compute_subspace_gap, compute_scaffolding_gap


def investigate_constant_gap():
    """Diagnose the constant gap phenomenon"""
    
    print("="*80)
    print("INVESTIGATING CONSTANT GAP PHENOMENON")
    print("="*80)
    
    # Test 1: Are we testing different problems?
    print("\n[TEST 1] Problem Uniqueness Check")
    print("-" * 80)
    
    problems = []
    for N in [3, 4, 5, 6]:
        M = int(4.0 * N)
        problem = generate_random_3sat(N, M, seed=42)
        problems.append((N, problem))
        print(f"\nN={N}, M={M}")
        print(f"  First 3 clauses: {[str(c.literals) for c in problem.clauses[:3]]}")
    
    # Test 2: What is the seed clause for each?
    print("\n[TEST 2] Seed Clause Analysis")
    print("-" * 80)
    
    for N, problem in problems:
        seed_clause = problem.clauses[0]  # 'first' strategy
        print(f"\nN={N}: Seed clause = {seed_clause.literals}")
        
        # How many solutions does seed clause have?
        n_solutions = 2**N - 1  # All except all-false
        print(f"  Theoretical solutions: {n_solutions}")
        
    # Test 3: Check H_seed matrices
    print("\n[TEST 3] H_seed Matrix Analysis")
    print("-" * 80)
    
    for N, problem in problems:
        seed_clause = problem.clauses[0]
        problem_seed = SATProblem(N, [seed_clause])
        H_seed = sat_to_matrix(problem_seed)
        
        eigenvalues = np.linalg.eigvalsh(H_seed)
        gap_seed, deg_seed = compute_subspace_gap(eigenvalues)
        
        print(f"\nN={N}:")
        print(f"  H_seed dimension: {H_seed.shape}")
        print(f"  Eigenvalues: {eigenvalues[:8]}")
        print(f"  Ground degeneracy: {deg_seed}")
        print(f"  Gap at s=0: {gap_seed:.8f}")
    
    # Test 4: Check actual gap computation
    print("\n[TEST 4] Full Path Gap Analysis")
    print("-" * 80)
    
    for N, problem in problems:
        result = compute_scaffolding_gap(
            problem,
            seed_strategy='first',
            num_points=10,  # Just a few points
            verbose=False
        )
        
        gaps = result['gaps']
        print(f"\nN={N}:")
        print(f"  Gap at s=0.0: {gaps[0]:.8f}")
        print(f"  Gap at s=0.5: {gaps[len(gaps)//2]:.8f}")
        print(f"  Gap at s=1.0: {gaps[-1]:.8f}")
        print(f"  Min gap: {result['g_min_interior']:.8f}")
        print(f"  All gaps: {gaps}")
    
    # Test 5: Try completely different random instances
    print("\n[TEST 5] Different Random Seeds")
    print("-" * 80)
    
    N = 4
    M = 16
    
    for seed_val in [42, 123, 456, 789, 999]:
        problem = generate_random_3sat(N, M, seed=seed_val)
        result = compute_scaffolding_gap(
            problem,
            seed_strategy='first',
            num_points=10,
            verbose=False
        )
        
        print(f"\nSeed={seed_val}: g_min = {result['g_min_interior']:.8f}")
    
    # Test 6: Different seed selection strategies
    print("\n[TEST 6] Different Seed Strategies")
    print("-" * 80)
    
    problem = generate_random_3sat(4, 16, seed=42)
    
    for strategy in ['first', 'random', 'most_vars', 'most_common']:
        try:
            result = compute_scaffolding_gap(
                problem,
                seed_strategy=strategy,
                num_points=10,
                verbose=False
            )
            print(f"\nStrategy='{strategy}': g_min = {result['g_min_interior']:.8f}")
            print(f"  Seed clause: {result['seed_clause'].literals}")
        except Exception as e:
            print(f"\nStrategy='{strategy}': ERROR - {e}")
    
    # Test 7: Examine the minimum gap point
    print("\n[TEST 7] Where Does Minimum Gap Occur?")
    print("-" * 80)
    
    for N, problem in problems:
        result = compute_scaffolding_gap(
            problem,
            seed_strategy='first',
            num_points=50,
            verbose=False
        )
        
        interior_mask = (result['s_values'] >= 0.05) & (result['s_values'] <= 0.95)
        interior_gaps = result['gaps'][interior_mask]
        interior_s = result['s_values'][interior_mask]
        
        min_idx = np.argmin(interior_gaps)
        s_min = interior_s[min_idx]
        g_min = interior_gaps[min_idx]
        
        print(f"\nN={N}:")
        print(f"  Minimum gap: {g_min:.8f} at s={s_min:.4f}")
        print(f"  Gap range: [{np.min(interior_gaps):.8f}, {np.max(interior_gaps):.8f}]")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("""
The constant gap g_min = 0.06896552 across all N suggests one of:

1. GENUINE PROPERTY: The algorithm truly has constant gap
   → This would be extraordinary (better than polynomial!)
   → Need theoretical explanation for why gap doesn't close
   
2. TESTING ARTIFACT: Same random seed produces similar structures
   → Need to test with many different random instances
   → Need to test adversarial/structured instances
   
3. NUMERICAL COINCIDENCE: Gap happens to be similar for N=3-6
   → Need to test larger N (7, 8, 9, 10+)
   → Gap might change at larger scales
   
4. BUG IN COMPUTATION: Matrix construction or eigenvalue computation
   → Need to validate matrix construction manually
   → Need to check if H(s) is actually changing with s

CRITICAL NEXT TESTS:
- Test N = 7, 8, 9, 10 with various random seeds
- Test planted SAT instances (known solutions)
- Test adversarial instances (hard SAT competition problems)
- Manually verify H(s) matrices for a small example
""")


if __name__ == "__main__":
    investigate_constant_gap()

