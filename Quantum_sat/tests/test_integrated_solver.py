"""
Test the integrated quantum SAT solver on various problem types.

This tests the full pipeline: analyze → route → solve
"""

import sys
sys.path.append('..')

import numpy as np
from src.core.quantum_sat_solver import ComprehensiveQuantumSATSolver

def test_small_structured():
    """Test 1: Small structured instance (should use QAOA)"""
    print("\n" + "="*80)
    print("TEST 1: Small Structured Instance")
    print("="*80)
    print("Expected: Route to QAOA Formal (k ≤ log₂(N)+1)")
    print()
    
    clauses = [
        (1, 2, 3),
        (-1, 2, 4),
        (-2, -3, 4),
        (1, -4, 5),
        (-1, -5, -2),
    ]
    
    solver = ComprehensiveQuantumSATSolver(verbose=True)
    result = solver.solve(clauses, n_vars=5, true_k=2)
    
    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  Satisfiable: {result.satisfiable}")
    print(f"  Method used: {result.method_used}")
    print(f"  Quantum advantage: {'✅' if result.quantum_advantage_applied else '❌'}")
    print(f"  k estimate: {result.k_estimate:.1f}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Total time: {result.total_time:.3f}s")
    print(f"{'='*80}\n")
    
    return result.quantum_advantage_applied


def test_medium_instance():
    """Test 2: Medium instance (should test routing)"""
    print("\n" + "="*80)
    print("TEST 2: Medium Random 3-SAT")
    print("="*80)
    print("Expected: Route based on k estimate")
    print()
    
    np.random.seed(42)
    n_vars = 10
    n_clauses = 44  # 4.4 * N (random 3-SAT)
    
    clauses = []
    for _ in range(n_clauses):
        vars_idx = np.random.choice(range(1, n_vars + 1), 3, replace=False)
        signs = np.random.choice([-1, 1], 3)
        clause = tuple(int(vars_idx[i] * signs[i]) for i in range(3))
        clauses.append(clause)
    
    solver = ComprehensiveQuantumSATSolver(verbose=True)
    result = solver.solve(clauses, n_vars=n_vars, timeout=15.0)
    
    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  Satisfiable: {result.satisfiable}")
    print(f"  Method used: {result.method_used}")
    print(f"  Quantum advantage: {'✅' if result.quantum_advantage_applied else '❌'}")
    print(f"  k estimate: {result.k_estimate:.1f}")
    print(f"  Recommended: {result.recommended_solver}")
    print(f"  Reasoning: {result.reasoning}")
    print(f"  Total time: {result.total_time:.3f}s")
    print(f"{'='*80}\n")
    
    return result


def test_forced_methods():
    """Test 3: Force each method to verify they all work"""
    print("\n" + "="*80)
    print("TEST 3: Force Each Method (Verify All Work)")
    print("="*80)
    
    # Simple SAT instance
    clauses = [
        (1, 2),
        (-1, 3),
        (-2, -3, 4),
        (1, -4),
    ]
    n_vars = 4
    
    from src.core.quantum_sat_solver import SolverMethod
    
    methods = [
        SolverMethod.QAOA_FORMAL,
        SolverMethod.QAOA_MORPHING,
        SolverMethod.QAOA_SCAFFOLDING,
        SolverMethod.QUANTUM_WALK,
        SolverMethod.QSVT,
    ]
    
    results = {}
    solver = ComprehensiveQuantumSATSolver(verbose=False)
    
    for method in methods:
        print(f"\nTesting {method.value}...")
        try:
            result = solver.solve(clauses, n_vars, method=method, timeout=10.0)
            results[method.value] = {
                'success': True,
                'satisfiable': result.satisfiable,
                'method': result.method_used,
                'time': result.total_time
            }
            print(f"  ✅ {result.method_used}: SAT={result.satisfiable}, time={result.total_time:.3f}s")
        except Exception as e:
            results[method.value] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed: {e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    successes = sum(1 for r in results.values() if r['success'])
    print(f"  Methods tested: {len(methods)}")
    print(f"  Successful: {successes}/{len(methods)}")
    print(f"{'='*80}\n")
    
    return results


def test_known_sat():
    """Test 4: Known SAT instance"""
    print("\n" + "="*80)
    print("TEST 4: Known SAT Instance")
    print("="*80)
    print("(x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)")
    print("Valid solution: x1=True, x2=False, x3=True (or others)")
    print()
    
    clauses = [
        (1, 2),
        (-1, 3),
        (-2, -3)
    ]
    
    solver = ComprehensiveQuantumSATSolver(verbose=True)
    result = solver.solve(clauses, n_vars=3, true_k=1)
    
    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  Satisfiable: {result.satisfiable}")
    if result.satisfiable and result.assignment:
        print(f"  Assignment: {result.assignment}")
        # Verify solution is valid
        def verify(clauses, assignment):
            for clause in clauses:
                sat = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            sat = True
                            break
                if not sat:
                    return False
            return True
        
        correct = verify(clauses, result.assignment)
        print(f"  Valid: {'✅' if correct else '❌'}")
    print(f"  Method: {result.method_used}")
    print(f"  Time: {result.total_time:.3f}s")
    print(f"{'='*80}\n")
    
    return result.satisfiable


def test_known_unsat():
    """Test 5: Known UNSAT instance"""
    print("\n" + "="*80)
    print("TEST 5: Known UNSAT Instance")
    print("="*80)
    print("(x1) ∧ (¬x1) - contradictory clauses")
    print()
    
    clauses = [
        (1,),
        (-1,)
    ]
    
    solver = ComprehensiveQuantumSATSolver(verbose=True)
    result = solver.solve(clauses, n_vars=1, true_k=1)
    
    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"  Satisfiable: {result.satisfiable}")
    print(f"  Expected UNSAT: {'✅' if not result.satisfiable else '❌ WRONG!'}")
    print(f"  Method: {result.method_used}")
    print(f"  Time: {result.total_time:.3f}s")
    print(f"{'='*80}\n")
    
    return not result.satisfiable


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE QUANTUM SAT SOLVER - INTEGRATION TESTS")
    print("="*80)
    print("Testing full pipeline: Analyze → Route → Solve")
    print("="*80)
    
    test_results = {}
    
    # Test 1: Small structured
    test_results['small_structured'] = test_small_structured()
    
    # Test 2: Medium instance  
    result2 = test_medium_instance()
    test_results['medium_instance'] = result2.satisfiable is not None
    
    # Test 3: Force all methods
    method_results = test_forced_methods()
    test_results['all_methods'] = all(r['success'] for r in method_results.values())
    
    # Test 4: Known SAT
    test_results['known_sat'] = test_known_sat()
    
    # Test 5: Known UNSAT
    test_results['known_unsat'] = test_known_unsat()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    total = len(test_results)
    passed = sum(test_results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Quantum SAT solver is fully integrated!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review above for details")


if __name__ == "__main__":
    main()
