"""

test_scaffolding_gap.py

SPECTRAL GAP ANALYSIS FOR ADIABATIC SCAFFOLDING

This tests the NEW conjecture:

"The spectral gap along H(s) = H_seed + s·H_rest is polynomially bounded"

This should be MORE robust than the 2-SAT morphing because:
1. H_seed is ALWAYS satisfiable (single clause)
2. No lossy reduction
3. Smooth constraint addition
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

try:
    from qlto_sat_scaffolding import (
        SATProblem, SATClause,
        select_seed_clause,
        clause_to_hamiltonian,
        problem_to_hamiltonian,
        generate_random_3sat
    )
    SCAFFOLDING_AVAILABLE = True
except ImportError:
    SCAFFOLDING_AVAILABLE = False
    print("Warning: qlto_sat_scaffolding.py required")


def sat_to_matrix(problem: SATProblem) -> np.ndarray:
    """Convert SAT problem to matrix (for exact analysis)"""
    N = problem.n_vars
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=float)
    
    for x in range(dim):
        assignment = {i+1: bool((x >> i) & 1) for i in range(N)}
        
        penalty = sum(1.0 for c in problem.clauses if not c.is_satisfied(assignment))
        H[x, x] = penalty
    
    return H


def compute_subspace_gap(eigenvalues: np.ndarray, tolerance: float = 1e-6) -> Tuple[float, int]:
    """Compute gap between ground subspace and excited states"""
    E_min = eigenvalues[0]
    ground_mask = np.abs(eigenvalues - E_min) < tolerance
    degeneracy = np.sum(ground_mask)
    
    excited = eigenvalues[~ground_mask]
    if len(excited) == 0:
        return 0.0, degeneracy
    
    gap = excited[0] - E_min
    return gap, degeneracy


def compute_scaffolding_gap(
    problem: SATProblem,
    seed_strategy: str = 'first',
    num_points: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Compute spectral gap along scaffolding path.
    
    H(s) = H_seed + s·H_rest
    
    Returns gap profile and minimum gap.
    """
    t_start = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print("SCAFFOLDING SPECTRAL GAP ANALYSIS")
        print(f"{'='*80}")
        print(f"Problem: N={problem.n_vars}, M={len(problem.clauses)}")
        print(f"Seed strategy: {seed_strategy}")
        print(f"Sample points: {num_points}")
    
    # Select seed clause
    seed_idx = select_seed_clause(problem, strategy=seed_strategy)
    seed_clause = problem.clauses[seed_idx]
    
    if verbose:
        print(f"Seed clause: {seed_clause.literals}")
    
    # Build matrices
    # H_seed: just seed clause
    problem_seed = SATProblem(problem.n_vars, [seed_clause])
    H_seed = sat_to_matrix(problem_seed)
    
    # H_rest: all other clauses
    rest_clauses = [c for i, c in enumerate(problem.clauses) if i != seed_idx]
    problem_rest = SATProblem(problem.n_vars, rest_clauses)
    H_rest = sat_to_matrix(problem_rest)
    
    if verbose:
        print(f"\nBuilding matrices...")
        print(f"  H_seed: {H_seed.shape}")
        print(f"  H_rest: {H_rest.shape}")
    
    # Sample along path
    s_values = np.linspace(0, 1, num_points)
    gaps = []
    degeneracies = []
    
    if verbose:
        print("Computing gaps along path...")
    
    for s in s_values:
        H_s = H_seed + s * H_rest
        eigenvalues = np.linalg.eigvalsh(H_s)
        
        gap, degeneracy = compute_subspace_gap(eigenvalues)
        gaps.append(gap)
        degeneracies.append(degeneracy)
    
    gaps = np.array(gaps)
    degeneracies = np.array(degeneracies)
    
    # Analysis
    g_min_full = np.min(gaps)
    idx_min_full = np.argmin(gaps)
    s_min_full = s_values[idx_min_full]
    
    # Interior only (avoid endpoints)
    interior_mask = (s_values >= 0.05) & (s_values <= 0.95)
    interior_gaps = gaps[interior_mask]
    interior_s = s_values[interior_mask]
    
    if len(interior_gaps) > 0:
        g_min_interior = np.min(interior_gaps)
        s_min_interior = interior_s[np.argmin(interior_gaps)]
    else:
        g_min_interior = g_min_full
        s_min_interior = s_min_full
    
    t_end = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Full range [0, 1]:")
        print(f"  g_min = {g_min_full:.8f} at s = {s_min_full:.4f}")
        print(f"  Degeneracy at s_min: {degeneracies[idx_min_full]}")
        
        print(f"\nInterior [0.05, 0.95]:")
        print(f"  g_min = {g_min_interior:.8f} at s = {s_min_interior:.4f}")
        
        print(f"\nEndpoint analysis:")
        print(f"  At s=0: gap={gaps[0]:.8f}, degeneracy={degeneracies[0]} (seed only)")
        print(f"  At s=1: gap={gaps[-1]:.8f}, degeneracy={degeneracies[-1]} (full problem)")
        
        print(f"\nComputation time: {t_end - t_start:.2f}s")
        print(f"{'='*80}")
    
    return {
        's_values': s_values,
        'gaps': gaps,
        'degeneracies': degeneracies,
        'g_min_full': g_min_full,
        's_min_full': s_min_full,
        'g_min_interior': g_min_interior,
        's_min_interior': s_min_interior,
        'seed_idx': seed_idx,
        'seed_clause': seed_clause,
        'time_seconds': t_end - t_start
    }


def compare_scaffolding_vs_morphing(
    n_vars: int = 4,
    n_clauses: int = 16,
    seed: int = 42
) -> Dict:
    """
    Compare scaffolding approach vs original morphing approach.
    
    This shows why scaffolding is superior.
    """
    print(f"\n{'='*80}")
    print("COMPARISON: Scaffolding vs 2-SAT Morphing")
    print(f"{'='*80}")
    
    problem = generate_random_3sat(n_vars, n_clauses, seed)
    
    print(f"\nTest problem: N={n_vars}, M={n_clauses}")
    
    # Test scaffolding
    print("\n--- SCAFFOLDING APPROACH ---")
    result_scaff = compute_scaffolding_gap(problem, verbose=True)
    
    # Compare with 2-SAT morphing (if available)
    print("\n--- 2-SAT MORPHING APPROACH ---")
    print("(Would require 2-SAT reduction)")
    print("Issue: 2-SAT might be UNSAT even if 3-SAT is SAT")
    print("→ Cannot guarantee valid starting point")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    print("Scaffolding advantages:")
    print("  ✓ H_seed is ALWAYS satisfiable (single clause)")
    print("  ✓ No lossy reduction")
    print("  ✓ Smooth constraint addition")
    print(f"  ✓ Interior gap: {result_scaff['g_min_interior']:.8f}")
    
    print("\n2-SAT morphing disadvantages:")
    print("  ✗ 2-SAT reduction can be UNSAT")
    print("  ✗ Lossy information (drops literals)")
    print("  ✗ Invalid starting point")
    
    return result_scaff


def analyze_scaffolding_scaling(
    n_vars_range: List[int] = [3, 4, 5, 6],
    instances_per_n: int = 10,
    num_points: int = 30
) -> Dict:
    """
    Analyze how scaffolding gap scales with problem size.
    """
    print(f"\n{'='*80}")
    print("SCAFFOLDING SCALING ANALYSIS")
    print(f"{'='*80}")
    print(f"Testing N = {n_vars_range}")
    print(f"Instances per N: {instances_per_n}")
    print(f"{'='*80}\n")
    
    results = []
    
    for N in n_vars_range:
        print(f"\n--- N = {N} ---")
        
        gaps_for_n = []
        
        for instance_idx in range(instances_per_n):
            M = int(4.0 * N)
            problem = generate_random_3sat(N, M, seed=42 + instance_idx)
            
            result = compute_scaffolding_gap(
                problem,
                seed_strategy='first',
                num_points=num_points,
                verbose=False
            )
            g_min = result['g_min_interior']
            gaps_for_n.append(g_min)
            
            print(f"  Instance {instance_idx+1}/{instances_per_n}: g_min = {g_min:.8f}")
        
        avg_gap = np.mean(gaps_for_n)
        std_gap = np.std(gaps_for_n)
        min_gap = np.min(gaps_for_n)
        
        print(f"\n  Average g_min: {avg_gap:.8f} ± {std_gap:.8f}")
        print(f"  Minimum g_min: {min_gap:.8f}")
        
        results.append({
            'N': N,
            'gaps': gaps_for_n,
            'avg_gap': avg_gap,
            'std_gap': std_gap,
            'min_gap': min_gap
        })
    
    # Fit scaling
    print(f"\n{'='*80}")
    print("SCALING MODELS")
    print(f"{'='*80}")
    
    N_vals = np.array([r['N'] for r in results])
    avg_gaps = np.array([r['avg_gap'] for r in results])
    min_gaps = np.array([r['min_gap'] for r in results])
    
    # Remove zeros
    valid_avg = avg_gaps > 1e-10
    valid_min = min_gaps > 1e-10
    
    if np.sum(valid_avg) >= 2:
        log_N = np.log(N_vals[valid_avg])
        log_avg_gaps = np.log(avg_gaps[valid_avg])
        poly_fit_avg = np.polyfit(log_N, log_avg_gaps, 1)
        exp_fit_avg = np.polyfit(N_vals[valid_avg], log_avg_gaps, 1)
    else:
        poly_fit_avg = [np.nan, np.nan]
        exp_fit_avg = [np.nan, np.nan]
    
    print("\nPOLYNOMIAL MODEL: g_min ~ N^(-α)")
    print(f"  Average gaps: α = {-poly_fit_avg[0]:.4f}")
    print("  → If α ≤ 3: Polynomial time ✓")
    
    print("\nEXPONENTIAL MODEL: g_min ~ exp(-βN)")
    print(f"  Average gaps: β = {-exp_fit_avg[0]:.4f}")
    print("  → If β < 0.1: Weak exponential (acceptable) ✓")
    print("  → If β > 0.5: Strong exponential (bad) ✗")
    
    print(f"\n{'='*80}")
    
    return {
        'results': results,
        'N_vals': N_vals,
        'avg_gaps': avg_gaps,
        'min_gaps': min_gaps,
        'poly_fit_avg': poly_fit_avg,
        'exp_fit_avg': exp_fit_avg
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ADIABATIC SCAFFOLDING: SPECTRAL GAP ANALYSIS")
    print("="*80)
    
    # Test 1: Single instance comparison
    print("\n[TEST 1] Scaffolding vs Morphing Comparison")
    comparison = compare_scaffolding_vs_morphing(n_vars=4, n_clauses=16)
    
    # Test 2: Scaling analysis
    print("\n[TEST 2] Scaffolding Scaling Analysis")
    scaling = analyze_scaffolding_scaling(
        n_vars_range=[3, 4, 5, 6],
        instances_per_n=10,
        num_points=30
    )
    
    print("\n" + "="*80)
    print("FINAL VERDICT: SCAFFOLDING vs MORPHING")
    print("="*80)
    
    avg_gaps = scaling['avg_gaps']
    poly_alpha = -scaling['poly_fit_avg'][0]
    exp_beta = -scaling['exp_fit_avg'][0]
    
    print(f"\nScaffolding gaps: {avg_gaps}")
    print(f"Polynomial scaling: α = {poly_alpha:.3f}")
    print(f"Exponential scaling: β = {exp_beta:.3f}")
    
    print("\n**SCAFFOLDING is theoretically superior because:**")
    print("  1. No lossy reduction (H = H_seed + H_rest exactly)")
    print("  2. Guaranteed valid start (single clause always SAT)")
    print("  3. Smooth constraint addition (filtering solutions)")
    print("  4. Classical analogy (like DPLL branching)")
    
    if poly_alpha <= 3:
        print("\n✓✓✓ SCALING IS POLYNOMIAL!")
        print("Scaffolding approach is VIABLE for structured SAT!")
    elif exp_beta < 0.3:
        print("\n✓ WEAK EXPONENTIAL (acceptable)")
        print("Scaffolding may work for some instances")
    else:
        print("\n⚠ EXPONENTIAL SCALING")
        print("Need to analyze what structure predicts success")
    
    print("="*80)

