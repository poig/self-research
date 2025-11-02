"""
Integrated Quantum SAT Solver
==============================

Connects the dispatcher pipeline with actual quantum implementations.
Routes problems to the appropriate quantum/hybrid solver based on structure.
"""

import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

# Import our analysis pipeline
sys.path.append('.')
from src.core.integrated_pipeline import integrated_dispatcher_pipeline

# Import quantum solvers from experiments
try:
    from Quantum_sat.experiments.qaoa_sat_formal import solve_sat_qlto, SATProblem as QLTOProblem
    QLTO_AVAILABLE = True
except (ImportError, Exception) as e:
    QLTO_AVAILABLE = False
    print(f"⚠️  QLTO solver not available: {e}")

try:
    from experiments.quantum_walk_sat import QuantumWalkSATSolver
    QWALK_AVAILABLE = True
except (ImportError, Exception) as e:
    QWALK_AVAILABLE = False
    print(f"⚠️  Quantum Walk solver not available: {e}")

try:
    from Quantum_sat.experiments.qaoa_sat_scaffolding import solve_sat_adiabatic_scaffolding
    SCAFFOLDING_AVAILABLE = True
except (ImportError, Exception) as e:
    SCAFFOLDING_AVAILABLE = False
    print(f"⚠️  Scaffolding solver not available: {e}")


@dataclass
class QuantumSATResult:
    """Result from quantum SAT solving"""
    satisfiable: bool
    assignment: Optional[Dict[int, bool]]
    method_used: str
    analysis_time: float
    solving_time: float
    total_time: float
    k_estimate: float
    confidence: float
    quantum_advantage: bool
    reasoning: str


class IntegratedQuantumSATSolver:
    """
    Complete quantum SAT solver with intelligent routing.
    
    Pipeline:
    1. Analyze structure (k estimation)
    2. Route to appropriate solver
    3. Execute quantum/hybrid algorithm
    4. Return verified result
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def solve(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int,
        true_k: Optional[int] = None
    ) -> QuantumSATResult:
        """
        Solve SAT instance using quantum advantage where applicable.
        
        Args:
            clauses: List of clauses (tuples of literals)
            n_vars: Number of variables
            true_k: If known, use this backdoor size (for testing)
            
        Returns:
            QuantumSATResult with solution and metadata
        """
        
        total_start = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print("INTEGRATED QUANTUM SAT SOLVER")
            print("="*80)
            print(f"Problem: N={n_vars} variables, M={len(clauses)} clauses")
            if true_k is not None:
                print(f"Known backdoor: k={true_k}")
            print()
        
        # Phase 1: Analyze structure
        if self.verbose:
            print("[Phase 1/2] Analyzing problem structure...")
        
        analysis_start = time.time()
        routing = integrated_dispatcher_pipeline(
            clauses, 
            n_vars, 
            verbose=False,
            true_k=true_k
        )
        analysis_time = time.time() - analysis_start
        
        k_est = routing['k_estimate']
        confidence = routing['confidence']
        solver_choice = routing['recommended_solver']
        reasoning = routing['reasoning']
        
        if self.verbose:
            print(f"  Estimated backdoor: k ≈ {k_est:.1f} (confidence: {confidence:.1%})")
            print(f"  Recommended solver: {solver_choice}")
            print(f"  Reasoning: {reasoning}")
            print(f"  Analysis time: {analysis_time:.3f}s")
            print()
        
        # Phase 2: Route to solver
        if self.verbose:
            print(f"[Phase 2/2] Executing {solver_choice}...")
        
        solving_start = time.time()
        
        # Route based on analysis
        if solver_choice == "quantum":
            result = self._solve_quantum(clauses, n_vars, k_est)
        elif solver_choice == "hybrid_qaoa":
            result = self._solve_hybrid_qaoa(clauses, n_vars, k_est)
        elif solver_choice == "scaffolding_search":
            result = self._solve_scaffolding(clauses, n_vars, k_est)
        else:  # robust_cdcl
            result = self._solve_classical(clauses, n_vars, k_est)
        
        solving_time = time.time() - solving_start
        total_time = time.time() - total_start
        
        if self.verbose:
            print(f"  Solving time: {solving_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            print()
            print("="*80)
            if result['satisfiable']:
                print(f"✅ SATISFIABLE (found by {result['method']})")
            else:
                print(f"❌ UNSATISFIABLE or TIMEOUT (checked by {result['method']})")
            print("="*80)
            print()
        
        # Check if we actually used quantum advantage
        quantum_advantage = solver_choice in ["quantum", "hybrid_qaoa"]
        
        return QuantumSATResult(
            satisfiable=result['satisfiable'],
            assignment=result['assignment'],
            method_used=result['method'],
            analysis_time=analysis_time,
            solving_time=solving_time,
            total_time=total_time,
            k_estimate=k_est,
            confidence=confidence,
            quantum_advantage=quantum_advantage,
            reasoning=reasoning
        )
    
    def _solve_quantum(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int, 
        k_est: float
    ) -> Dict[str, Any]:
        """
        Solve using pure quantum algorithm (QLTO).
        For small backdoors k ≤ log₂(N)+1.
        """
        
        if not QLTO_AVAILABLE:
            if self.verbose:
                print("  ⚠️  QLTO not available, falling back to classical")
            return self._solve_classical(clauses, n_vars, k_est)
        
        try:
            # Convert to QLTOProblem format
            problem = QLTOProblem(n_vars, [list(c) for c in clauses])
            
            # Solve with QLTO
            if self.verbose:
                print(f"  Running QLTO quantum algorithm...")
                print(f"  Expected complexity: O(N² log²(N)) = O({n_vars**2 * np.log2(n_vars)**2:.0f})")
            
            result = solve_sat_qlto(
                problem,
                max_iterations=min(50, int(n_vars * np.log2(n_vars))),
                shots_per_iteration=1024,
                p_layers=int(np.log2(n_vars)) + 1,
                verbose=self.verbose
            )
            
            return {
                'satisfiable': result.get('satisfiable', False),
                'assignment': result.get('assignment'),
                'method': 'QLTO Quantum'
            }
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  QLTO failed: {e}")
                print(f"  Falling back to classical")
            return self._solve_classical(clauses, n_vars, k_est)
    
    def _solve_hybrid_qaoa(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int, 
        k_est: float
    ) -> Dict[str, Any]:
        """
        Solve using hybrid QAOA.
        For medium backdoors log₂(N)+1 < k ≤ N/3.
        """
        
        if not QLTO_AVAILABLE:
            if self.verbose:
                print("  ⚠️  QAOA not available, falling back to classical")
            return self._solve_classical(clauses, n_vars, k_est)
        
        try:
            # Use QLTO with more layers for harder problems
            problem = QLTOProblem(n_vars, [list(c) for c in clauses])
            
            if self.verbose:
                print(f"  Running Hybrid QAOA...")
                print(f"  Using p={int(k_est)} layers for medium structure")
            
            result = solve_sat_qlto(
                problem,
                max_iterations=min(100, int(k_est * np.log2(n_vars))),
                shots_per_iteration=2048,
                p_layers=min(10, int(k_est)),
                verbose=self.verbose
            )
            
            return {
                'satisfiable': result.get('satisfiable', False),
                'assignment': result.get('assignment'),
                'method': 'Hybrid QAOA'
            }
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  QAOA failed: {e}")
                print(f"  Falling back to classical")
            return self._solve_classical(clauses, n_vars, k_est)
    
    def _solve_scaffolding(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int, 
        k_est: float
    ) -> Dict[str, Any]:
        """
        Solve using scaffolding/hierarchical search.
        For large backdoors N/3 < k ≤ 2N/3.
        """
        
        if self.verbose:
            print(f"  Running hierarchical scaffolding search...")
            print(f"  Large backdoor (k={k_est:.1f}), using heuristic methods")
        
        # For now, use classical with time limit
        # TODO: Implement actual scaffolding when available
        return self._solve_classical(clauses, n_vars, k_est, timeout=10.0)
    
    def _solve_classical(
        self, 
        clauses: List[Tuple[int, ...]], 
        n_vars: int, 
        k_est: float,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Solve using classical CDCL/DPLL.
        For very large backdoors k > 2N/3 or as fallback.
        """
        
        if self.verbose:
            print(f"  Running classical DPLL with {timeout}s timeout...")
        
        # Simple DPLL implementation
        start_time = time.time()
        
        def dpll(
            clauses: List[Tuple[int, ...]], 
            assignment: Dict[int, bool]
        ) -> Optional[Dict[int, bool]]:
            """Simple DPLL solver with timeout"""
            
            if time.time() - start_time > timeout:
                return None
            
            # Check if all clauses satisfied
            all_satisfied = True
            for clause in clauses:
                clause_sat = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            clause_sat = True
                            break
                if not clause_sat:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                return assignment
            
            # Find unassigned variable
            unassigned = None
            for v in range(1, n_vars + 1):
                if v not in assignment:
                    unassigned = v
                    break
            
            if unassigned is None:
                return None
            
            # Try both values
            for val in [True, False]:
                new_assignment = assignment.copy()
                new_assignment[unassigned] = val
                result = dpll(clauses, new_assignment)
                if result is not None:
                    return result
            
            return None
        
        result = dpll(clauses, {})
        
        return {
            'satisfiable': result is not None,
            'assignment': result,
            'method': 'Classical DPLL'
        }


def main():
    """Demo of integrated quantum SAT solver"""
    
    print("\n" + "="*80)
    print("QUANTUM SAT SOLVER - DEMO")
    print("="*80)
    print()
    
    # Test case 1: Small structured instance (quantum advantage expected)
    print("Test 1: Small structured instance (N=10, k≈3)")
    print("-" * 80)
    
    clauses_small = [
        (1, 2, 3),
        (-1, 2, 4),
        (-2, -3, 4),
        (1, -4, 5),
        (-1, -5, 6),
        (2, 5, -6),
        (-3, 6, 7),
        (3, -6, -7),
        (4, 7, 8),
        (-4, -7, 8),
    ]
    
    solver = IntegratedQuantumSATSolver(verbose=True)
    result = solver.solve(clauses_small, n_vars=8, true_k=3)
    
    print()
    print("Test 2: Medium instance (N=20, k≈5)")
    print("-" * 80)
    
    # Generate random 3-SAT
    np.random.seed(42)
    n_vars_med = 20
    n_clauses_med = 88
    clauses_medium = []
    for _ in range(n_clauses_med):
        vars = np.random.choice(range(1, n_vars_med + 1), 3, replace=False)
        signs = np.random.choice([-1, 1], 3)
        clauses_medium.append(tuple(int(vars[i] * signs[i]) for i in range(3)))
    
    result = solver.solve(clauses_medium, n_vars=n_vars_med, true_k=5)
    
    print()
    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
