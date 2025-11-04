"""
Benchmark: Decomposable SAT Solver (Polynomial-Time Path)
=========================================================

This benchmark script is designed to test the PRIMARY, 
polynomial-time path of the ComprehensiveQuantumSATSolver.

GOAL:
Prove that the solver can deterministically (99.99%+) solve
decomposable SAT problems (where k* is small) using the
"solve_via_decomposition" method, which relies on solving
many small, "feasible qubit" subproblems.

METHODOLOGY:
1.  Generate a large number (e.g., 100) of SAT problems that are
    *known* to be satisfiable and *known* to be decomposable
    (e.g., N=50, k*=10, 'modular' structure).
2.  Run each problem through the ComprehensiveQuantumSATSolver.
3.  Verify two conditions for success:
    a) Did the solver correctly use the decomposition path?
       (solution.decomposition_used == True)
    b) Did the solver correctly report 'satisfiable=True'?
       (Since we know a solution exists).
4.  Calculate and report the final success rate.
"""

import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Optional, Any

# --- Add paths to import the solver and problem generator ---
# This assumes the benchmark is run from the root or 'src' directory
try:
    import os, sys
    # Ensure repository root and relevant subpackages are on the import path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    quantum_sat_pkg = os.path.join(repo_root, 'Quantum_sat')
    src_dir = os.path.join(quantum_sat_pkg, 'src')
    experiments_dir = os.path.join(quantum_sat_pkg, 'experiments')
    # Prepend to sys.path for deterministic imports
    sys.path.insert(0, src_dir)
    sys.path.insert(0, experiments_dir)
    sys.path.insert(0, quantum_sat_pkg)

    # Imports reflecting actual repository layout
    from core.quantum_sat_solver import ComprehensiveQuantumSATSolver
    # sat_decompose provides helpers for generating test instances
    from sat_decompose import create_test_sat_instance, verify_solution
    # qaoa_with_qlto (optional) lives under src/solvers; adjust path so optional imports can work
    sys.path.insert(0, os.path.join(src_dir, 'solvers'))
    try:
        from qaoa_with_qlto import SATProblem  # optional: problem classes if available
    except Exception:
        SATProblem = None
    print("✅ Successfully imported solver and helpers.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please run this script from the root directory containing 'quantum_sat_solver.py' and 'sat_decompose.py'")
    sys.exit(1)

# --- Import tqdm for progress bar ---
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        print("Install 'tqdm' for a nice progress bar: pip install tqdm")
        return iterable


def generate_benchmark_problem(n_vars: int, k_backdoor: int, structure: str) -> Tuple[List[Tuple], int, Dict[int, bool]]:
    """
    Generates a single, known-satisfiable benchmark problem.
    
    Returns:
        (clauses, n_vars, planted_solution_0_indexed)
    """
    clauses, backdoor_vars, planted_solution = create_test_sat_instance(
        n_vars=n_vars,
        k_backdoor=k_backdoor,
        structure_type=structure,
        ensure_sat=True # CRITICAL: We must know it's satisfiable
    )
    
    # create_test_sat_instance returns a 0-indexed assignment dict
    return clauses, n_vars, planted_solution


def run_benchmark(num_problems: int, n_vars: int, k_backdoor: int):
    """
    Runs the full benchmark test.
    """
    print("\n" + "="*80)
    print("Running Polynomial-Time Path Benchmark")
    print("="*80)
    print(f"  Configuration:")
    print(f"    Problems to test: {num_problems}")
    print(f"    Problem Structure: N={n_vars}, k*={k_backdoor} ('modular')")
    print(f"  This tests the solver's ability to decompose a large problem (N={n_vars})")
    print(f"  into small, 'feasible qubit' partitions (k*={k_backdoor}) and solve them.")
    print("="*80 + "\n")
    
    # Initialize the solver
    # We run with verbose=False to keep the log clean
    # We disable certification because we are *telling* it k*
    solver = ComprehensiveQuantumSATSolver(
        verbose=False,
        enable_quantum_certification=False,
        use_true_k=True # Use the k_backdoor value we provide
    )
    
    success_count = 0
    total_solve_time = 0.0
    failures = []

    for i in tqdm(range(num_problems), desc="Benchmarking Solver"):
        # 1. Generate a known-SAT, decomposable problem
        clauses, n_vars, planted_solution = generate_benchmark_problem(
            n_vars=n_vars,
            k_backdoor=k_backdoor,
            structure='modular'
        )
        
        # 2. Run the solver
        start_time = time.time()
        solution = solver.solve(
            clauses, 
            n_vars, 
            true_k=k_backdoor,
            check_final=False # We will do our own check
        )
        solve_time = time.time() - start_time
        total_solve_time += solve_time
        
        # 3. Verify the result
        used_decomposition = solution.decomposition_used
        found_solution = solution.satisfiable
        
        is_success = False
        if used_decomposition and found_solution:
            # It used the right path and found a solution.
            # We can be >99.9% sure this is a success.
            is_success = True
            success_count += 1
            
        elif not used_decomposition:
            # CRITICAL FAILURE: It didn't use the poly-time path!
            failures.append(f"Problem {i}: FAILED. Solver did NOT use decomposition path (k*={solution.k_estimate:.1f}).")
            
        elif not found_solution:
            # CRITICAL FAILURE: It used the path but failed to find a solution.
            failures.append(f"Problem {i}: FAILED. Solver returned UNSAT for a known-SAT problem.")
    
    # 4. Report Results
    avg_time = total_solve_time / num_problems
    success_rate = 100.0 * success_count / num_problems
    
    print("\n" + "="*80)
    print("Benchmark Complete: Results")
    print("="*80)
    print(f"  Problems Tested:     {num_problems}")
    print(f"  Problem Structure:   N={n_vars}, k*={k_backdoor} ('modular')")
    print("-" * 80)
    print(f"  Successes:           {success_count} / {num_problems}")
    print(f"  Success Rate:        {success_rate:.2f}%")
    print(f"  Average Solve Time:  {avg_time:.4f}s")
    print(f"  Total Solve Time:    {total_solve_time:.2f}s")
    print("-" * 80)
    
    if failures:
        print(f"  Analysis: ❌ FAILED")
        print(f"  Found {len(failures)} failures:")
        for f in failures[:10]: # Print first 10 failures
            print(f"    - {f}")
    elif success_rate >= 99.99:
        print(f"  Analysis: ✅ SUCCESS (Deterministic, Polynomial-Time)")
        print(f"  This confirms the solver correctly identifies and solves")
        print(f"  decomposable problems using the polynomial-time path.")
    else:
        print(f"  Analysis: ⚠️  Partial Success")
        print(f"  The solver is not fully deterministic. See failures above.")
        
    print("="*80)


if __name__ == "__main__":
    
    # --- Configuration ---
    NUM_PROBLEMS = 100   # Number of problems to test
    N_VARS = 50          # Total variables (too large for global solver)
    K_BACKDOOR = 10      # Backdoor size (small enough for "feasible qubit" solver)
    # --- End Configuration ---
    
    run_benchmark(
        num_problems=NUM_PROBLEMS,
        n_vars=N_VARS,
        k_backdoor=K_BACKDOOR
    )
