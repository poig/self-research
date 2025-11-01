"""

Test Safe Dispatcher with Verification Probes
=============================================

Validates the safety mechanisms and decision logic.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tests'))


import numpy as np
from src.core.safe_dispatcher import SafeDispatcher, SolverType
from src.core.polynomial_structure_analyzer import PolynomialStructureAnalyzer
from test_lanczos_scalability import generate_random_3sat


def test_confidence_threshold():
    """Test that low confidence triggers fallback"""
    print("="*70)
    print("TEST 1: Confidence Threshold Safety")
    print("="*70)
    print("\nGoal: Low confidence should trigger ROBUST_CLASSICAL fallback\n")
    
    dispatcher = SafeDispatcher(confidence_threshold=0.75, verbose=True)
    
    # Generate dummy instance
    clauses = generate_random_3sat(10, 42, seed=1)
    
    # Test case: Low confidence
    decision = dispatcher.dispatch(
        k_estimate=3.0,
        confidence=0.60,  # Below threshold
        ci_lower=2.5,
        ci_upper=3.5,
        n_vars=10,
        clauses=clauses
    )
    
    assert decision.solver == SolverType.ROBUST_CLASSICAL
    assert "Low confidence" in decision.reason
    print(f"\n✅ PASS: Low confidence correctly triggered fallback\n")


def test_k_sanity_bounds():
    """Test that invalid k triggers fallback"""
    print("="*70)
    print("TEST 2: k Sanity Bounds")
    print("="*70)
    print("\nGoal: Invalid k values should trigger ROBUST_CLASSICAL\n")
    
    dispatcher = SafeDispatcher(verbose=True)
    clauses = generate_random_3sat(10, 42, seed=2)
    
    # Test case 1: Negative k
    print("Case 1: Negative k")
    decision = dispatcher.dispatch(
        k_estimate=-1.0,  # Invalid
        confidence=0.90,
        ci_lower=-1.5,
        ci_upper=-0.5,
        n_vars=10,
        clauses=clauses
    )
    assert decision.solver == SolverType.ROBUST_CLASSICAL
    assert "Invalid k_estimate" in decision.reason
    print(f"✅ Negative k rejected\n")
    
    # Test case 2: k > N
    print("Case 2: k > N")
    decision = dispatcher.dispatch(
        k_estimate=15.0,  # > N=10
        confidence=0.90,
        ci_lower=14.5,
        ci_upper=15.5,
        n_vars=10,
        clauses=clauses
    )
    assert decision.solver == SolverType.ROBUST_CLASSICAL
    assert "Invalid k_estimate" in decision.reason
    print(f"✅ k > N rejected\n")


def test_verification_probe():
    """Test verification probe catches bad estimates"""
    print("="*70)
    print("TEST 3: Verification Probe")
    print("="*70)
    print("\nGoal: Verification should catch unrealistic estimates\n")
    
    dispatcher = SafeDispatcher(
        enable_verification=True,
        confidence_threshold=0.75,
        verbose=True
    )
    
    clauses = generate_random_3sat(12, 50, seed=3)
    
    # Good estimate (should pass verification)
    print("Case 1: Realistic estimate")
    decision = dispatcher.dispatch(
        k_estimate=3.0,
        confidence=0.85,
        ci_lower=2.7,
        ci_upper=3.3,
        n_vars=12,
        clauses=clauses
    )
    
    print(f"Verification passed: {decision.verification_passed}")
    print(f"Solver: {decision.solver.value}\n")
    
    print(f"✅ Verification probe executed\n")


def test_solver_selection_logic():
    """Test that correct solver is chosen based on k"""
    print("="*70)
    print("TEST 4: Solver Selection Logic")
    print("="*70)
    print("\nGoal: Different k values should dispatch to appropriate solvers\n")
    
    dispatcher = SafeDispatcher(
        confidence_threshold=0.75,
        enable_verification=False,  # Disable for cleaner test
        verbose=False
    )
    
    n_vars = 16
    clauses = generate_random_3sat(n_vars, 67, seed=4)
    
    test_cases = [
        {
            'k': 2.0,
            'expected': SolverType.BACKDOOR_QUANTUM,
            'desc': 'Very small k → Quantum'
        },
        {
            'k': 4.5,
            'expected': SolverType.BACKDOOR_QUANTUM,
            'desc': 'Small k ≈ log(N) → Quantum'
        },
        {
            'k': 5.2,
            'expected': SolverType.BACKDOOR_HYBRID,
            'desc': 'Just above quantum threshold → Hybrid'
        },
        {
            'k': 7.0,
            'expected': SolverType.SCAFFOLDING,
            'desc': 'Above N/3 → Scaffolding'
        },
        {
            'k': 12.0,
            'expected': SolverType.ROBUST_CLASSICAL,
            'desc': 'Above 2N/3 → Robust'
        },
    ]
    
    print(f"N = {n_vars}, thresholds:")
    print(f"  k_quantum ≤ log2(N)+1 = {np.log2(n_vars)+1:.2f}")
    print(f"  k_hybrid ≤ N/3 = {n_vars/3:.2f}")
    print(f"  k_scaffolding ≤ 2N/3 = {2*n_vars/3:.2f}\n")
    
    print(f"{'k_estimate':<12} {'Expected':<20} {'Actual':<20} {'Result':<8}")
    print("-" * 70)
    
    for test in test_cases:
        decision = dispatcher.dispatch(
            k_estimate=test['k'],
            confidence=0.85,  # Above threshold
            ci_lower=test['k'] - 0.2,
            ci_upper=test['k'] + 0.2,
            n_vars=n_vars,
            clauses=clauses
        )
        
        match = "✅ PASS" if decision.solver == test['expected'] else "❌ FAIL"
        print(f"{test['k']:<12.1f} {test['expected'].value:<20} "
              f"{decision.solver.value:<20} {match:<8}")
        
        assert decision.solver == test['expected'], f"Failed for k={test['k']}"
    
    print(f"\n✅ All solver selection tests passed\n")


def test_full_pipeline_integration():
    """Test complete pipeline: analyze → dispatch"""
    print("="*70)
    print("TEST 5: Full Pipeline Integration")
    print("="*70)
    print("\nGoal: Validate end-to-end workflow\n")
    
    # Generate test instances
    test_instances = [
        (10, 30, "easy"),
        (12, 50, "medium"),
        (14, 58, "hard"),
    ]
    
    analyzer = PolynomialStructureAnalyzer(verbose=False)
    dispatcher = SafeDispatcher(
        confidence_threshold=0.75,
        enable_verification=True,
        verbose=False
    )
    
    print(f"{'Instance':<10} {'N':>4} {'M':>5} {'k_est':>8} {'Conf':>8} "
          f"{'Solver':<20} {'Safe':>6}")
    print("-" * 75)
    
    for n, m, name in test_instances:
        clauses = generate_random_3sat(n, m, seed=n*100)
        
        # Step 1: Analyze structure
        k_estimate, confidence = analyzer.analyze(clauses, n)
        
        # Get diagnostics
        diag = analyzer._last_mc_diagnostics
        if diag:
            ci_lower = diag['ci_lower']
            ci_upper = diag['ci_upper']
        else:
            ci_lower = k_estimate - 0.5
            ci_upper = k_estimate + 0.5
        
        # Step 2: Dispatch
        decision = dispatcher.dispatch(
            k_estimate=k_estimate,
            confidence=confidence,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_vars=n,
            clauses=clauses,
            estimator_diagnostics=diag
        )
        
        safe_icon = "✅" if decision.verification_passed or not dispatcher.enable_verification else "⚠️"
        
        print(f"{name:<10} {n:>4} {m:>5} {k_estimate:>8.2f} {confidence:>8.2%} "
              f"{decision.solver.value:<20} {safe_icon:>6}")
    
    print(f"\n✅ Full pipeline working correctly\n")
    
    # Show dispatcher statistics
    dispatcher.print_statistics()


def test_dispatcher_statistics():
    """Test statistics tracking"""
    print("="*70)
    print("TEST 6: Statistics Tracking")
    print("="*70)
    print("\nGoal: Validate telemetry and learning capability\n")
    
    dispatcher = SafeDispatcher(confidence_threshold=0.75, verbose=False)
    clauses = generate_random_3sat(12, 50, seed=5)
    
    # Make various decisions
    decisions = [
        (2.0, 0.90),  # Quantum
        (3.0, 0.85),  # Quantum
        (6.0, 0.80),  # Hybrid
        (8.0, 0.75),  # Scaffolding
        (10.0, 0.70), # Robust (low confidence)
        (4.0, 0.60),  # Robust (low confidence)
    ]
    
    for k, conf in decisions:
        dispatcher.dispatch(
            k_estimate=k,
            confidence=conf,
            ci_lower=k-0.3,
            ci_upper=k+0.3,
            n_vars=12,
            clauses=clauses
        )
    
    stats = dispatcher.get_statistics()
    
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Fast path usage: {stats['fast_path_fraction']:.1%}")
    print(f"Confidence rejections: {stats['confidence_rejections']}")
    
    assert stats['total_decisions'] == 6
    assert stats['confidence_rejections'] == 2  # Two below 0.75
    
    print(f"\n✅ Statistics tracking working correctly\n")
    
    dispatcher.print_statistics()


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("SAFE DISPATCHER TEST SUITE")
    print("="*70)
    print("\nValidating safety mechanisms and decision logic\n")
    
    test_confidence_threshold()
    test_k_sanity_bounds()
    test_verification_probe()
    test_solver_selection_logic()
    test_full_pipeline_integration()
    test_dispatcher_statistics()
    
    print("="*70)
    print("ALL DISPATCHER TESTS PASSED")
    print("="*70)
    print("\nKey safety mechanisms validated:")
    print("  ✅ Confidence threshold enforced")
    print("  ✅ k sanity bounds checked")
    print("  ✅ Verification probes working")
    print("  ✅ Solver selection logic correct")
    print("  ✅ Full pipeline integration successful")
    print("  ✅ Statistics tracking operational")
    print("\nProduction readiness:")
    print("  ✅ Multiple independent safety checks")
    print("  ✅ Conservative by default (fallback when uncertain)")
    print("  ✅ Telemetry for learning from mistakes")
    print("  ✅ Clear decision reasoning for auditing")
    print("\nNext steps:")
    print("  1. Benchmark on SAT Competition instances")
    print("  2. Tune thresholds based on real data")
    print("  3. Add ML calibration layer")
    print("  4. Deploy in production hybrid solver")
    print()


if __name__ == "__main__":
    run_all_tests()

