"""

Small Benchmark Suite for Quantum Structure Analyzer
====================================================

CI-friendly benchmarks to detect performance regressions and validate
correctness on representative SAT instances.

Usage:
    python bench_small.py               # Run all benchmarks
    python bench_small.py --quick       # Run quick subset only
    python bench_small.py --profile     # Enable detailed profiling
"""

import time
import argparse
import numpy as np
from typing import List, Tuple
from quantum_structure_analyzer import QuantumStructureAnalyzer
from test_lanczos_scalability import generate_random_3sat


def bench_instance(name: str, clauses: List[Tuple[int, ...]], n_vars: int, 
                   verbose: bool = True) -> dict:
    """
    Benchmark single SAT instance through full pipeline.
    
    Returns:
        dict with timing and correctness metrics
    """
    qsa = QuantumStructureAnalyzer(verbose=False)
    
    # Phase 1: Build Hamiltonian
    t0 = time.time()
    H = qsa._build_hamiltonian(clauses, n_vars)
    t1 = time.time()
    build_time = t1 - t0
    
    # Phase 2: Compute diagonal
    t2 = time.time()
    if hasattr(H, 'matrix'):
        H_matrix = H.to_matrix()
        diag = np.diag(H_matrix).real
    else:
        diag = qsa._compute_diagonal_vectorized(clauses, n_vars)
    t3 = time.time()
    diag_time = t3 - t2
    
    # Phase 3: Validate trace
    t4 = time.time()
    expected_trace = sum(2**(n_vars - len(c)) for c in clauses)
    actual_trace = np.sum(diag)
    trace_error = abs(actual_trace - expected_trace)
    t5 = time.time()
    trace_time = t5 - t4
    
    # Phase 4: Compute ground state energy
    ground_energy = np.min(diag)
    
    total_time = t5 - t0
    
    result = {
        'name': name,
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'dim': 2**n_vars,
        'build_time': build_time,
        'diag_time': diag_time,
        'trace_time': trace_time,
        'total_time': total_time,
        'ground_energy': ground_energy,
        'expected_trace': expected_trace,
        'actual_trace': actual_trace,
        'trace_error': trace_error,
        'trace_valid': trace_error < 1e-10
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmark: {name}")
        print(f"{'='*70}")
        print(f"  Variables:        N = {n_vars}")
        print(f"  Clauses:          M = {len(clauses)}")
        print(f"  Hilbert space:    2^{n_vars} = {2**n_vars:,}")
        print(f"  Ground energy:    E_0 = {ground_energy:.6f}")
        print(f"  Trace validation: {'✅ PASS' if result['trace_valid'] else '❌ FAIL'}")
        print(f"    Expected:       {expected_trace:.2f}")
        print(f"    Actual:         {actual_trace:.2f}")
        print(f"    Error:          {trace_error:.2e}")
        print(f"  Timing:")
        print(f"    Hamiltonian:    {build_time:.4f}s")
        print(f"    Diagonal:       {diag_time:.4f}s")
        print(f"    Trace check:    {trace_time:.4f}s")
        print(f"    Total:          {total_time:.4f}s")
        print(f"  Throughput:       {2**n_vars / total_time:,.0f} states/sec")
    
    return result


def run_benchmark_suite(quick: bool = False) -> List[dict]:
    """
    Run standardized benchmark suite.
    
    Args:
        quick: If True, run only quick subset (N ≤ 12)
    
    Returns:
        List of result dictionaries
    """
    print("="*70)
    print("QUANTUM STRUCTURE ANALYZER - BENCHMARK SUITE")
    print("="*70)
    
    results = []
    
    # Benchmark 1: Tiny instance (smoke test)
    print("\n[1/6] Tiny instance (N=4)...")
    clauses_tiny = [(1, 2, 3), (-1, 2, 4), (1, -3, 4), (-2, 3, 4)]
    results.append(bench_instance("Tiny_N4", clauses_tiny, 4, verbose=True))
    
    # Benchmark 2: Small random (N=8)
    print("\n[2/6] Small random 3-SAT (N=8)...")
    clauses_8 = generate_random_3sat(8, 33, seed=800)
    results.append(bench_instance("Random_N8", clauses_8, 8, verbose=True))
    
    # Benchmark 3: Medium random (N=10)
    print("\n[3/6] Medium random 3-SAT (N=10)...")
    clauses_10 = generate_random_3sat(10, 42, seed=1000)
    results.append(bench_instance("Random_N10", clauses_10, 10, verbose=True))
    
    # Benchmark 4: Large random (N=12)
    print("\n[4/6] Large random 3-SAT (N=12)...")
    clauses_12 = generate_random_3sat(12, 50, seed=1200)
    results.append(bench_instance("Random_N12", clauses_12, 12, verbose=True))
    
    if not quick:
        # Benchmark 5: Very large (N=14)
        print("\n[5/6] Very large random 3-SAT (N=14)...")
        clauses_14 = generate_random_3sat(14, 58, seed=1400)
        results.append(bench_instance("Random_N14", clauses_14, 14, verbose=True))
        
        # Benchmark 6: Extreme (N=16)
        print("\n[6/6] Extreme random 3-SAT (N=16)...")
        clauses_16 = generate_random_3sat(16, 67, seed=1600)
        results.append(bench_instance("Random_N16", clauses_16, 16, verbose=True))
    else:
        print("\n[Quick mode] Skipping N=14 and N=16 benchmarks")
    
    return results


def print_summary(results: List[dict]):
    """Print summary table of all benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Name':<15} {'N':>3} {'M':>4} {'Dim':>10} {'Time':>8} {'States/s':>12} {'Valid':>6}")
    print("-"*70)
    
    for r in results:
        valid_str = "✅" if r['trace_valid'] else "❌"
        throughput = r['dim'] / r['total_time']
        print(f"{r['name']:<15} {r['n_vars']:>3} {r['n_clauses']:>4} "
              f"{r['dim']:>10,} {r['total_time']:>7.3f}s {throughput:>11,.0f}  {valid_str:>6}")
    
    print("-"*70)
    
    # Check for failures
    failures = [r for r in results if not r['trace_valid']]
    if failures:
        print(f"\n❌ {len(failures)} BENCHMARK(S) FAILED trace validation!")
        for f in failures:
            print(f"   - {f['name']}: error = {f['trace_error']:.2e}")
        return False
    else:
        print(f"\n✅ All {len(results)} benchmarks PASSED!")
        return True


def detect_regression(results: List[dict], baseline: dict = None):
    """
    Check for performance regressions against baseline.
    
    Baseline format: {'Random_N8': 0.05, 'Random_N10': 0.20, ...}
    """
    if baseline is None:
        # Default baselines (approximate, adjust based on your hardware)
        baseline = {
            'Tiny_N4': 0.01,
            'Random_N8': 0.05,
            'Random_N10': 0.50,
            'Random_N12': 15.0,
            'Random_N14': 300.0,
            'Random_N16': 6000.0
        }
    
    print("\n" + "="*70)
    print("REGRESSION DETECTION")
    print("="*70)
    
    regression_threshold = 2.0  # Flag if >2x slower than baseline
    regressions = []
    
    for r in results:
        name = r['name']
        if name not in baseline:
            continue
        
        baseline_time = baseline[name]
        actual_time = r['total_time']
        slowdown = actual_time / baseline_time
        
        status = "✅" if slowdown <= regression_threshold else "⚠️ REGRESSION"
        print(f"{name:<15}: {actual_time:>7.3f}s (baseline: {baseline_time:>7.3f}s, "
              f"slowdown: {slowdown:>5.2f}x) {status}")
        
        if slowdown > regression_threshold:
            regressions.append((name, slowdown))
    
    if regressions:
        print(f"\n⚠️  {len(regressions)} performance regression(s) detected:")
        for name, slowdown in regressions:
            print(f"   - {name}: {slowdown:.2f}x slower than baseline")
        return False
    else:
        print(f"\n✅ No performance regressions detected")
        return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark Quantum Structure Analyzer")
    parser.add_argument('--quick', action='store_true', 
                       help="Run quick subset only (N ≤ 12)")
    parser.add_argument('--profile', action='store_true',
                       help="Enable detailed profiling (not yet implemented)")
    parser.add_argument('--baseline', type=str,
                       help="Path to baseline JSON file for regression detection")
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmark_suite(quick=args.quick)
    
    # Print summary
    correctness_ok = print_summary(results)
    
    # Check regressions
    regression_ok = detect_regression(results)
    
    # Exit code
    if correctness_ok and regression_ok:
        print("\n" + "="*70)
        print("✅ ALL CHECKS PASSED")
        print("="*70)
        exit(0)
    else:
        print("\n" + "="*70)
        print("❌ SOME CHECKS FAILED")
        print("="*70)
        exit(1)


if __name__ == "__main__":
    main()

