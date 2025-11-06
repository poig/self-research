#!/usr/bin/env python3
"""
FPT Scaling Benchmark
=====================

This script demonstrates the two key bottlenecks in our hybrid solver:

1.  **Quantum FPT ("Lockpick"):** Benchmarks `run_fpt_pipeline` from qlto_qaoa_sat.py.
    This path is limited by N (the number of variables/qubits) because
    simulating N qubits requires O(2^N) classical memory and time.
    We expect to hit a "Simulation Wall" around N=22-24.

2.  **Classical FPT ("Sledgehammer"):** Benchmarks `try_candidate_backdoor_pysat`
    from qlto_qaoa_sat.py. This is the solver that *should* be
    used by the treewidth decomposer.
    This path is limited by k (the backdoor/treewidth size).
    YOUR LOGS PROVED THIS IS NOT AN EXPONENTIAL BOTTLENECK.
"""
import time
import os
import sys
import random
import numpy as np
import itertools
from typing import List, Tuple

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Solvers ---
try:
    from src.solvers.qlto_qaoa_sat import (
        SATProblem,
        SATClause,
        run_fpt_pipeline,
        try_candidate_backdoor_pysat,
        QISKIT_AVAILABLE
    )
    from pysat.solvers import Glucose3
    PYSAT_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure qlto_qaoa_sat.py, quantum_sat_solver.py, and pysat are available.")
    sys.exit(1)

if not QISKIT_AVAILABLE:
    print("Error: Qiskit is not available. Cannot run Benchmark 1.")
    sys.exit(1)

# --- SAT Problem Generator ---
def gen_random_ksat(n_vars: int, m_clauses: int, k: int = 3, ensure_sat=False) -> Tuple[List[Tuple[int, ...]], List[int]]:
    """Generates a random K-SAT instance."""
    clauses = []
    vars_list = list(range(1, n_vars + 1))
    
    planted_solution = None
    if ensure_sat:
        planted_solution = [random.choice([1, -1]) * v for v in vars_list]

    for _ in range(m_clauses):
        clause_vars = random.sample(vars_list, k)
        
        if ensure_sat:
            # Ensure the clause is satisfied by the planted solution
            if not any(v in planted_solution for v in clause_vars) and \
               not any(-v in planted_solution for v in clause_vars):
                # None of the vars are in the clause, let's fix one
                idx_to_fix = random.randint(0, k - 1)
                var_to_fix = abs(clause_vars[idx_to_fix])
                
                # Find the sign in the planted solution
                sign = 1 if var_to_fix in planted_solution else -1
                clause_vars[idx_to_fix] = var_to_fix * sign
        
        # Randomize signs for non-fixed vars (if not ensure_sat)
        lits = []
        for v in clause_vars:
            if abs(v) in lits or -abs(v) in lits: # avoid duplicates
                continue
                
            if ensure_sat and (v in planted_solution or -v in planted_solution):
                lits.append(v) # Already has correct sign
            else:
                lits.append(v * random.choice([1, -1]))
        clauses.append(tuple(lits))
        
    return clauses, planted_solution

# ==============================================================================
# BENCHMARK 1: THE "SIMULATION WALL" (O(2^N))
# ==============================================================================

def benchmark_quantum_fpt():
    """
    Benchmarks the full Quantum FPT pipeline (`run_fpt_pipeline`).
    We expect runtime to explode exponentially with N.
    """
    print("\n" + "="*80)
    print("BENCHMARK 1: The Quantum FPT 'Simulation Wall' - Scaling with N")
    print("Running `run_fpt_pipeline` on random SAT problems.")
    print("Expected cost: O(2^N) due to classical simulation of N qubits.")
    print("="*80)
    print(f"{'N (Qubits)':<12} {'M (Clauses)':<12} {'Time (s)':<12} {'Result':<12}")
    print("-"*80)

    # Ns to test. This will get very slow, very fast.
    N_values = [10, 12, 14, 16, 18, 20] # 22+ will take minutes/hours
    
    # Create dummy config and trial seed
    dummy_cfg = {
        "p_layers": 2,
        "bits_per_param": 3,
        "N_MASK_BITS": 10,
        "shots": 1024,
        "top_T_candidates": 30,
        "freq_threshold": 0.15,
        "verbose": False
    }

    for i, N in enumerate(N_values):
        M = int(N * 4.2) # 3-SAT phase transition
        clauses, solution = gen_random_ksat(N, M, 3, ensure_sat=True)
        
        # We don't need the SATProblem object for this call
        # sat_clauses = [SATClause(c) for c in clauses]
        # problem = SATProblem(n_vars=N, clauses=sat_clauses)
        
        start_time = time.time()
        try:
            # We run the pipeline. This simulates the circuit.
            # FIX: Pass `clauses` (List[Tuple]) not `problem` (SATProblem object)
            result = run_fpt_pipeline(
                clauses, # <-- FIX
                n_vars=N,
                mode_or_name="benchmark",
                cfg=dummy_cfg,
                trial_seed=(1234 + i)
            )
            elapsed = time.time() - start_time
            
            if result.get('success'):
                print(f"{N:<12} {M:<12} {elapsed:<12.4f} {'Success'}_k'={result.get('k_prime_final')}")
            else:
                print(f"{N:<12} {M:<12} {elapsed:<12.4f} {'Failure'}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{N:<12} {M:<12} {elapsed:<12.4f} {f'Error: {e}'}")
        
        if elapsed > 30.0: # 30 second timeout
             print("--- HIT SIMULATION WALL ---")
             break # Stop benchmark if it's clearly too slow

# ==============================================================================
# BENCHMARK 2: THE "TREEWIDTH WALL" (O(N * 2^k))
# ==============================================================================

def benchmark_classical_fpt():
    """
    Benchmarks the classical FPT solver (`try_candidate_backdoor_pysat`).
    This is the algorithm that *should* be run by the treewidth decomposer.
    We fix N and expect runtime to explode exponentially with k.
    """
    print("\n" + "="*80)
    print("BENCHMARK 2: The Classical FPT 'Treewidth Wall' - Scaling with k")
    print("Running `try_candidate_backdoor_pysat` (the classical O(N*2^k) part).")
    print("Expected cost: O(N * 2^k) as we scale the backdoor size k.")
    print("="*80)
    print(f"{'N (Vars)':<12} {'k (Backdoor)':<12} {'Checks (2^k)':<12} {'Time (s)':<12} {'Result':<12}")
    print("-"*80)

    # Fix N to show scaling is polynomial in N
    N = 600
    M = int(N * 4.2) # 1680 clauses
    
    # k's to test. This will get very slow, very fast.
    k_values = [10, 12, 14, 16, 18, 20, 22] # 24+ will take minutes/hours

    clauses, solution = gen_random_ksat(N, M, 3, ensure_sat=True)
    sat_clauses = [SATClause(c) for c in clauses]
    problem = SATProblem(n_vars=N, clauses=sat_clauses)

    for k in k_values:
        # Pick k random variables to be our "backdoor"
        # (This simulates what the decomposer *should* give us)
        backdoor_vars = random.sample(range(N), k) # 0-indexed list
        
        max_checks = 2**k
        
        start_time = time.time()
        try:
            # We run the classical O(N * 2^k) solver
            # This does NOT simulate any quantum circuits
            solution_map = try_candidate_backdoor_pysat(
                problem,
                backdoor_vars
            )
            elapsed = time.time() - start_time
            
            if solution_map:
                print(f"{N:<12} {k:<12} {max_checks:<12} {elapsed:<12.4f} {'Success'}")
            else:
                # This is expected, a random backdoor won't solve a random problem
                print(f"{N:<12} {k:<12} {max_checks:<12} {elapsed:<12.4f} {'Failure (OK)'}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{N:<12} {k:<12} {max_checks:<12} {f'Error: {e}'}")
        
        if elapsed > 60.0:
            print("--- HIT TREEWIDTH/FPT WALL ---")
            break # Stop benchmark if it's clearly too slow

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Running FPT Solver Scaling Benchmarks")
    print(f"This script will demonstrate the O(2^N) 'Simulation Wall'")
    print(f"and the O(N * 2^k) 'Treewidth Wall'.")
    print("="*80)
    
    if QISKIT_AVAILABLE:
        benchmark_quantum_fpt()
    else:
        print("\nSkipping Benchmark 1: Qiskit not available.")
    
    if PYSAT_AVAILABLE:
        benchmark_classical_fpt()
    else:
        print("\nSkipping Benchmark 2: PySAT not available.")

    
    print("\n" + "="*80)
    print("Benchmark Complete.")
    print("="*80)