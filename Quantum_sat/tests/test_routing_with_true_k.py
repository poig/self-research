"""
Test solver routing with true_k values
Verifies that the standard k method works properly when k is known
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from src.core.integrated_pipeline import integrated_dispatcher_pipeline

def generate_dummy_3sat(n_vars, n_clauses, seed=42):
    """Generate random 3-SAT for testing routing (doesn't need to be structured)"""
    random.seed(seed)
    clauses = []
    for _ in range(n_clauses):
        # Pick 3 random distinct variables
        vars = random.sample(range(1, n_vars + 1), 3)
        # Randomly negate
        clause = tuple(v * (1 if random.random() > 0.5 else -1) for v in vars)
        clauses.append(clause)
    return clauses

print("="*70)
print("SOLVER ROUTING TEST WITH TRUE_K")
print("="*70)
print()

# Test cases covering all routing thresholds
test_cases = [
    # (N, k_true, expected_solver, description)
    (20, 2, "quantum", "k ≤ log₂(N)+1 → quantum"),
    (32, 5, "quantum", "k = log₂(N) → quantum"),
    (30, 10, "hybrid_qaoa", "k ≤ N/3 → hybrid_qaoa"),
    (50, 15, "hybrid_qaoa", "k = N/3 → hybrid_qaoa"),
    (50, 25, "scaffolding_search", "k ≤ 2N/3 → scaffolding_search"),
    (60, 40, "scaffolding_search", "k = 2N/3 → scaffolding_search"),
    (50, 35, "robust_cdcl", "k > 2N/3 → robust_cdcl"),
    (30, 28, "robust_cdcl", "k near N → robust_cdcl"),
]

print(f"{'N':>4} {'k':>4} {'log₂(N)+1':>10} {'N/3':>6} {'2N/3':>6} {'Solver':>20} {'✓':>3}")
print("-"*70)

all_correct = True
for n, k_true, expected_solver, desc in test_cases:
    # Generate dummy instance (routing doesn't depend on actual clauses when true_k is given)
    M = int(n * 4.4)
    clauses = generate_dummy_3sat(n, M, seed=42)
    
    # Run pipeline with true_k
    result = integrated_dispatcher_pipeline(clauses, n, verbose=False, true_k=k_true)
    
    actual_solver = result['recommended_solver']
    is_correct = (actual_solver == expected_solver)
    all_correct = all_correct and is_correct
    
    log2_n_plus_1 = np.log2(n) + 1
    n_over_3 = n / 3
    two_n_over_3 = 2 * n / 3
    
    mark = "✓" if is_correct else "✗"
    
    print(f"{n:>4} {k_true:>4} {log2_n_plus_1:>10.1f} {n_over_3:>6.1f} {two_n_over_3:>6.1f} {actual_solver:>20} {mark:>3}")
    
    if not is_correct:
        print(f"  ⚠️  Expected: {expected_solver}, Got: {actual_solver}")
        print(f"      Reasoning: {result['reasoning']}")

print()
print("="*70)
if all_correct:
    print("✅ ALL ROUTING TESTS PASSED")
    print(f"   All {len(test_cases)} cases routed to expected solver")
else:
    print("❌ SOME ROUTING TESTS FAILED")
    print("   Check threshold logic in integrated_pipeline.py")
print("="*70)
print()

# Show the routing thresholds
print("ROUTING THRESHOLDS:")
print(f"  k ≤ log₂(N)+1     → quantum (exponential speedup)")
print(f"  k ≤ N/3           → hybrid_qaoa (polynomial speedup)")
print(f"  k ≤ 2N/3          → scaffolding_search (heuristic)")
print(f"  k > 2N/3          → robust_cdcl (classical)")
print()

# Test edge cases
print("EDGE CASE TESTS:")
print("-"*70)

# Test with k=0 (trivial)
clauses_trivial = generate_dummy_3sat(20, 88, seed=42)
result = integrated_dispatcher_pipeline(clauses_trivial, 20, verbose=False, true_k=0)
print(f"k=0 (trivial):       {result['recommended_solver']:>20} (reasoning: {result['reasoning']})")

# Test boundary: k = log₂(N)+1 exactly
n_boundary = 32
k_boundary = int(np.log2(n_boundary)) + 1  # Should be 6
clauses_b = generate_dummy_3sat(n_boundary, int(n_boundary*4.4), seed=42)
result = integrated_dispatcher_pipeline(clauses_b, n_boundary, verbose=False, true_k=k_boundary)
print(f"k=log₂(N)+1:         {result['recommended_solver']:>20} (k={k_boundary})")

# Test boundary: k = N/3 exactly
n_boundary2 = 30
k_boundary2 = n_boundary2 // 3  # Should be 10
clauses_b2 = generate_dummy_3sat(n_boundary2, int(n_boundary2*4.4), seed=42)
result = integrated_dispatcher_pipeline(clauses_b2, n_boundary2, verbose=False, true_k=k_boundary2)
print(f"k=N/3:               {result['recommended_solver']:>20} (k={k_boundary2})")

print()
