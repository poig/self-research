"""
Minimal Integrated Quantum SAT Solver (Working Version)
========================================================

Connects dispatcher with QLTO quantum solver only.
"""

import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

# Import our analysis pipeline
sys.path.append('.')
from src.core.integrated_pipeline import integrated_dispatcher_pipeline

# Import QLTO solver
try:
    from Quantum_sat.experiments.qaoa_sat_formal import solve_sat_qlto, SATProblem, SATClause
    print("✓ QLTO Quantum Solver loaded")
    QLTO_AVAILABLE = True
except Exception as e:
    print(f"✗ QLTO not available: {e}")
    QLTO_AVAILABLE = False
    SATProblem = None
    SATClause = None


class QuantumSATSolver:
    """
    Minimal quantum SAT solver with intelligent routing.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def solve(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int,
        true_k: Optional[int] = None,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Solve SAT with quantum advantage where possible.
        """
        
        total_start = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print("QUANTUM SAT SOLVER")
            print("="*80)
            print(f"Problem: N={n_vars}, M={len(clauses)}, k_true={true_k}")
            print()
        
        # Phase 1: Analyze
        if self.verbose:
            print("[1/2] Analyzing structure...")
        
        routing = integrated_dispatcher_pipeline(
            clauses, n_vars, verbose=False, true_k=true_k
        )
        
        k_est = routing['k_estimate']
        solver_choice = routing['recommended_solver']
        
        if self.verbose:
            print(f"  k ≈ {k_est:.1f}")
            print(f"  Route: {solver_choice}")
            print()
        
        # Phase 2: Solve
        if self.verbose:
            print(f"[2/2] Solving with {solver_choice}...")
        
        if solver_choice == "quantum" and QLTO_AVAILABLE:
            result = self._quantum_solve(clauses, n_vars, k_est)
        elif solver_choice == "hybrid_qaoa" and QLTO_AVAILABLE:
            result = self._hybrid_solve(clauses, n_vars, k_est)
        else:
            result = self._classical_solve(clauses, n_vars, timeout)
        
        result['total_time'] = time.time() - total_start
        result['k_estimate'] = k_est
        result['routed_to'] = solver_choice
        
        if self.verbose:
            print()
            print("="*80)
            status = "✅ SAT" if result['satisfiable'] else "❌ UNSAT/TIMEOUT"
            print(f"{status} by {result['method']} in {result['total_time']:.2f}s")
            print("="*80)
        
        return result
    
    def _quantum_solve(self, clauses, n_vars, k_est):
        """Use QLTO quantum algorithm"""
        try:
            # Convert to SATProblem format
            sat_clauses = [SATClause(tuple(c)) for c in clauses]
            problem = SATProblem(n_vars, sat_clauses)
            
            if self.verbose:
                print(f"  QLTO: p={int(np.log2(max(2, n_vars)))+1} layers, O(N²log²N) complexity")
            
            result = solve_sat_qlto(
                problem,
                max_iterations=min(50, int(n_vars * np.log2(max(2, n_vars)))),
                shots_per_iteration=1024,
                p_layers=int(np.log2(max(2, n_vars))) + 1,
                verbose=False
            )
            
            return {
                'satisfiable': result.get('satisfiable', False),
                'assignment': result.get('assignment'),
                'method': 'QLTO Quantum',
                'quantum_used': True
            }
        except Exception as e:
            if self.verbose:
                print(f"  QLTO failed: {e}, falling back")
            return self._classical_solve(clauses, n_vars, 5.0)
    
    def _hybrid_solve(self, clauses, n_vars, k_est):
        """Use Hybrid QAOA"""
        try:
            # Convert to SATProblem format
            sat_clauses = [SATClause(tuple(c)) for c in clauses]
            problem = SATProblem(n_vars, sat_clauses)
            
            if self.verbose:
                print(f"  Hybrid QAOA: p={min(10, int(k_est))} layers")
            
            result = solve_sat_qlto(
                problem,
                max_iterations=min(100, int(k_est * np.log2(max(2, n_vars)))),
                shots_per_iteration=2048,
                p_layers=min(10, int(k_est)),
                verbose=False
            )
            
            return {
                'satisfiable': result.get('satisfiable', False),
                'assignment': result.get('assignment'),
                'method': 'Hybrid QAOA',
                'quantum_used': True
            }
        except Exception as e:
            if self.verbose:
                print(f"  QAOA failed: {e}, falling back")
            return self._classical_solve(clauses, n_vars, 5.0)
    
    def _classical_solve(self, clauses, n_vars, timeout):
        """Simple classical DPLL"""
        if self.verbose:
            print(f"  Classical DPLL with {timeout}s timeout")
        
        start = time.time()
        
        def dpll(assignment):
            if time.time() - start > timeout:
                return None
            
            # Check if satisfied
            all_sat = True
            for clause in clauses:
                clause_sat = any(
                    (lit > 0 and assignment.get(abs(lit), None) == True) or
                    (lit < 0 and assignment.get(abs(lit), None) == False)
                    for lit in clause
                )
                if not clause_sat:
                    all_sat = False
                    break
            
            if all_sat:
                return assignment
            
            # Find unassigned
            for v in range(1, n_vars + 1):
                if v not in assignment:
                    for val in [True, False]:
                        result = dpll({**assignment, v: val})
                        if result:
                            return result
                    return None
            return None
        
        solution = dpll({})
        
        return {
            'satisfiable': solution is not None,
            'assignment': solution,
            'method': 'Classical DPLL',
            'quantum_used': False
        }


# Quick test
if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUANTUM SAT SOLVER - QUICK TEST")
    print("="*80)
    
    # Test 1: Trivial (should use quantum)
    print("\nTest 1: Trivial SAT (N=5, k=2)")
    clauses1 = [(1, 2), (-1, 3), (2, -3), (4, 5), (-4, -5)]
    
    solver = QuantumSATSolver(verbose=True)
    result1 = solver.solve(clauses1, n_vars=5, true_k=2)
    
    # Test 2: Medium (should use hybrid)
    print("\nTest 2: Medium SAT (N=10, k=5)")
    np.random.seed(42)
    clauses2 = []
    for _ in range(44):
        vars = np.random.choice(range(1, 11), 3, replace=False)
        signs = np.random.choice([-1, 1], 3)
        clauses2.append(tuple(int(vars[i] * signs[i]) for i in range(3)))
    
    result2 = solver.solve(clauses2, n_vars=10, true_k=5)
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
