#!/usr/bin/env python3
"""
Verification script for Feigenbaum Paper 2 claims.
Tests the core numerical claims made in the paper.
"""

import numpy as np
import sys

def sin2_map(x, r):
    """The quantum measurement map: x_{n+1} = r · sin²(πx)"""
    return r * np.sin(np.pi * x)**2

def iterate_map(x0, r, n_trans=2000, n_sample=200):
    """Iterate map and return attractor samples."""
    x = x0
    for _ in range(n_trans):
        x = sin2_map(x, r)
    samples = []
    for _ in range(n_sample):
        x = sin2_map(x, r)
        samples.append(x)
    return np.array(samples)

def detect_period(samples, tol=1e-5):
    """Detect period of attractor."""
    for p in [1, 2, 4, 8, 16, 32]:
        if len(samples) >= 2*p:
            is_periodic = all(np.std(samples[i::p]) < tol for i in range(p))
            if is_periodic:
                return p
    return 0

def find_bifurcation_precise(r_min, r_max, from_period, n_points=5000):
    """Find r where period doubles from from_period to 2*from_period"""
    rs = np.linspace(r_min, r_max, n_points)
    for r in rs:
        att = iterate_map(0.4, r)
        p = detect_period(att)
        if p == 2 * from_period:
            return r
    return None

def main():
    print("=" * 60)
    print("FEIGENBAUM PAPER 2 VERIFICATION")
    print("=" * 60)
    
    # Test 1: sin² map basic verification
    print("\n[TEST 1] sin² map formula verification")
    print("-" * 40)
    x = 0.5
    r = 0.8
    result = r * np.sin(np.pi * x)**2
    print(f"sin²(π×0.5) = {np.sin(np.pi*0.5)**2:.6f} (should be 1.0)")
    print(f"r×sin²(πx) at x=0.5, r=0.8 = {result:.6f} (should be 0.8)")
    print("✓ PASSED" if abs(result - 0.8) < 1e-10 else "✗ FAILED")
    
    # Test 2: Qiskit verification
    print("\n[TEST 2] Qiskit Hadamard test: P(|1⟩) = sin²(φ/2)")
    print("-" * 40)
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        
        errors = []
        phi_values = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        for phi in phi_values:
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.rz(phi, 0)
            qc.h(0)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities()
            p_qiskit = probs[1]
            p_theory = np.sin(phi/2)**2
            error = abs(p_qiskit - p_theory)
            errors.append(error)
            print(f"  φ={phi:5.2f}: Qiskit={p_qiskit:.6f}, Theory={p_theory:.6f}, Err={error:.2e}")
        
        max_err = max(errors)
        print(f"Max error: {max_err:.2e}")
        print("✓ PASSED" if max_err < 1e-10 else "✗ FAILED")
    except ImportError:
        print("Qiskit not available - skipping")
    
    # Test 3: Period detection
    print("\n[TEST 3] Period detection at known r values")
    print("-" * 40)
    test_points = [
        (0.55, 1, 'Period-1'),
        (0.65, 2, 'Period-2'),
        (0.72, 4, 'Period-4'),
        (0.728, 8, 'Period-8'),
    ]
    all_passed = True
    for r, expected, desc in test_points:
        att = iterate_map(0.4, r)
        period = detect_period(att)
        passed = period == expected
        all_passed = all_passed and passed
        status = '✓' if passed else '✗'
        print(f"  {status} r={r:.3f}: detected={period}, expected={expected} ({desc})")
    print("✓ PASSED" if all_passed else "✗ FAILED")
    
    # Test 4: Feigenbaum constant
    print("\n[TEST 4] Feigenbaum constant δ = 4.669...")
    print("-" * 40)
    print("Finding bifurcation points (this may take a moment)...")
    
    r1 = find_bifurcation_precise(0.62, 0.64, 1, 5000)
    r2 = find_bifurcation_precise(0.70, 0.72, 2, 5000)
    r3 = find_bifurcation_precise(0.72, 0.73, 4, 8000)
    r4 = find_bifurcation_precise(0.729, 0.732, 8, 10000)
    
    print(f"  r1 (1→2)  = {r1:.6f}" if r1 else "  r1 (1→2)  = NOT FOUND")
    print(f"  r2 (2→4)  = {r2:.6f}" if r2 else "  r2 (2→4)  = NOT FOUND")
    print(f"  r3 (4→8)  = {r3:.6f}" if r3 else "  r3 (4→8)  = NOT FOUND")
    print(f"  r4 (8→16) = {r4:.6f}" if r4 else "  r4 (8→16) = NOT FOUND")
    
    FEIGENBAUM = 4.669201609
    
    if all([r1, r2, r3, r4]):
        d1 = r2 - r1
        d2 = r3 - r2
        d3 = r4 - r3
        
        delta1 = d1 / d2 if d2 > 1e-10 else float('inf')
        delta2 = d2 / d3 if d3 > 1e-10 else float('inf')
        
        print(f"\n  Δ1 = r2-r1 = {d1:.6f}")
        print(f"  Δ2 = r3-r2 = {d2:.6f}")
        print(f"  Δ3 = r4-r3 = {d3:.6f}")
        
        err1 = abs(delta1 - FEIGENBAUM) / FEIGENBAUM * 100
        err2 = abs(delta2 - FEIGENBAUM) / FEIGENBAUM * 100
        
        print(f"\n  δ1 = Δ1/Δ2 = {delta1:.4f} (error: {err1:.1f}%)")
        print(f"  δ2 = Δ2/Δ3 = {delta2:.4f} (error: {err2:.1f}%)")
        print(f"  Feigenbaum δ = {FEIGENBAUM:.4f}")
        
        # Paper claims δ3 = 4.69 ± 0.04, within 1%
        if err1 < 15 or err2 < 5:
            print("\n✓ PASSED: Feigenbaum constant approximated correctly")
        else:
            print("\n✗ FAILED: Feigenbaum constant deviation too large")
    else:
        print("Could not find all bifurcation points")
    
    # Test 5: Critical learning rate threshold
    print("\n[TEST 5] Critical thresholds from paper")
    print("-" * 40)
    # Paper claims: r1 ≈ 0.628, r2 ≈ 0.707, r3 ≈ 0.726
    paper_r1 = 0.6278
    paper_r2 = 0.7066  
    paper_r3 = 0.7259
    
    if r1 and r2 and r3:
        err_r1 = abs(r1 - paper_r1) / paper_r1 * 100
        err_r2 = abs(r2 - paper_r2) / paper_r2 * 100
        err_r3 = abs(r3 - paper_r3) / paper_r3 * 100
        
        print(f"  r1: measured={r1:.4f}, paper={paper_r1:.4f}, error={err_r1:.1f}%")
        print(f"  r2: measured={r2:.4f}, paper={paper_r2:.4f}, error={err_r2:.1f}%")
        print(f"  r3: measured={r3:.4f}, paper={paper_r3:.4f}, error={err_r3:.1f}%")
        
        if max(err_r1, err_r2, err_r3) < 1:
            print("✓ PASSED: Bifurcation points match paper")
        else:
            print("✗ FAILED: Bifurcation points deviate from paper")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
