"""
basis_universality.py

Experiment 1: Measurement Basis Independence
============================================

Goal: Show that measuring in X, Y, or Z basis after phase accumulation 
gives the SAME Feigenbaum δ = 4.669...

This tests the claim that Feigenbaum universality is fundamental to
quantum measurement, not just an artifact of the sin² function.

Key Question: Do different measurement bases give the same δ?

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from measurement_maps import (
    iterate_map, find_bifurcation_points, 
    compute_feigenbaum_delta, compute_lyapunov_exponent
)


# =============================================================================
# MEASUREMENT BASIS CIRCUITS
# =============================================================================

def hadamard_test_z_basis(phi: float) -> float:
    """
    Standard Hadamard test with Z-basis measurement.
    
    Circuit: H - Rz(φ) - H - measure
    Result: P(|1⟩) = sin²(φ/2)
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.h(0)  # Convert to Z basis
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


def hadamard_test_y_basis(phi: float) -> float:
    """
    Hadamard test with Y-basis measurement.
    
    Circuit: H - Rz(φ) - S† - H - measure
    Result: P(|1⟩) = sin²(φ/2 + π/4) = (1 - sin φ)/2
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.sdg(0)  # S† gate for Y rotation
    qc.h(0)
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


def hadamard_test_x_basis(phi: float) -> float:
    """
    Hadamard test with X-basis measurement (no final H).
    
    Circuit: H - Rz(φ) - measure (in X basis)
    
    Note: This is trivial - gives P = 0.5 always (no phase information)
    This serves as a CONTROL experiment.
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    # No final gate - direct measurement in computational basis
    # But the state is |+⟩ phase-rotated, which in Z-basis gives 0.5
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


def ramsey_rz_measurement(phi: float) -> float:
    """
    Alternative circuit: Ramsey with Rz encoding.
    
    Circuit: H - Rz(φ) - H - measure
    Same as standard Z-basis, included for completeness.
    """
    return hadamard_test_z_basis(phi)


def ramsey_rx_measurement(phi: float) -> float:
    """
    Ramsey with Rx encoding instead of Rz.
    
    Circuit: H - Rx(φ) - H - measure
    P(|1⟩) = sin²(φ/2) - same functional form!
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rx(phi, 0)
    qc.h(0)
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


def ramsey_ry_measurement(phi: float) -> float:
    """
    Ramsey with Ry encoding.
    
    Circuit: H - Ry(φ) - H - measure
    Different functional form due to Ry rotation structure.
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(phi, 0)
    qc.h(0)
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


# =============================================================================
# EXPERIMENT: COMPARE ALL BASES
# =============================================================================

def run_basis_comparison(
    r_min: float = 0.5,
    r_max: float = 1.0,
    n_r_points: int = 10000,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run bifurcation analysis for all measurement bases.
    
    Returns:
        Dictionary with results for each basis:
        - bifurcations: list of bifurcation points
        - deltas: interval widths
        - ratios: δ estimates
        - best_delta: best δ estimate
    """
    FEIGENBAUM = 4.669201609
    
    # Define measurement functions
    measurements = {
        'Z-basis (H-Rz-H)': hadamard_test_z_basis,
        'Y-basis (H-Rz-S†-H)': hadamard_test_y_basis,
        # 'X-basis (control)': hadamard_test_x_basis,  # Trivial, skip
        'Ramsey-Rx': ramsey_rx_measurement,
        'Ramsey-Ry': ramsey_ry_measurement,
    }
    
    results = {}
    
    for name, meas_func in measurements.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing: {name}")
            print('='*60)
        
        # Find bifurcations
        bifurcations = find_bifurcation_points(
            meas_func, r_min=r_min, r_max=r_max, n_points=n_r_points
        )
        
        if verbose:
            print(f"Found {len(bifurcations)} bifurcations")
            for b in bifurcations[:4]:
                print(f"  Period {b['from_period']:2d} → {b['to_period']:2d} at r = {b['r']:.6f}")
        
        # Compute δ
        deltas, ratios = compute_feigenbaum_delta(bifurcations)
        
        best_delta = ratios[-1] if ratios else None
        error = abs(best_delta - FEIGENBAUM) / FEIGENBAUM * 100 if best_delta else None
        
        if verbose and best_delta:
            print(f"Best δ estimate: {best_delta:.5f}")
            print(f"Feigenbaum δ:    {FEIGENBAUM:.5f}")
            print(f"Error:           {error:.2f}%")
        
        results[name] = {
            'bifurcations': bifurcations,
            'deltas': deltas,
            'ratios': ratios,
            'best_delta': best_delta,
            'error_percent': error
        }
    
    return results


def plot_basis_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    Create comparison plot of bifurcation diagrams for different bases.
    """
    FEIGENBAUM = 4.669201609
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    measurements = {
        'Z-basis (H-Rz-H)': (hadamard_test_z_basis, axes[0, 0]),
        'Y-basis (H-Rz-S†-H)': (hadamard_test_y_basis, axes[0, 1]),
        'Ramsey-Rx': (ramsey_rx_measurement, axes[1, 0]),
        'Ramsey-Ry': (ramsey_ry_measurement, axes[1, 1]),
    }
    
    for name, (meas_func, ax) in measurements.items():
        # Generate bifurcation diagram
        rs = np.linspace(0.5, 1.0, 300)
        all_rs, all_xs = [], []
        
        for r in rs:
            attractor = iterate_map(meas_func, 0.4, r, n_transient=1000, n_sample=50)
            for x in attractor[::2]:
                all_rs.append(r)
                all_xs.append(x)
        
        ax.scatter(all_rs, all_xs, s=0.2, c='blue', alpha=0.5)
        
        # Mark bifurcation points
        if name in results and results[name]['bifurcations']:
            for b in results[name]['bifurcations'][:5]:
                ax.axvline(x=b['r'], color='red', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Control Parameter r', fontsize=10)
        ax.set_ylabel('x (attractor)', fontsize=10)
        ax.set_title(f'{name}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add δ value annotation
        if name in results and results[name]['best_delta']:
            delta = results[name]['best_delta']
            ax.text(0.95, 0.95, f'δ ≈ {delta:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Measurement Basis Independence: All Bases → Same δ = 4.669?', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_delta_convergence(results: Dict[str, Dict], save_path: str = None):
    """
    Plot convergence of δ estimates across measurement bases.
    """
    FEIGENBAUM = 4.669201609
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))
    
    for i, (name, data) in enumerate(results.items()):
        if data['ratios']:
            ax.scatter(range(1, len(data['ratios'])+1), data['ratios'], 
                      s=80, label=name, color=colors[i], zorder=5)
            ax.plot(range(1, len(data['ratios'])+1), data['ratios'], 
                   color=colors[i], alpha=0.5)
    
    ax.axhline(y=FEIGENBAUM, color='red', linestyle='--', linewidth=2,
               label=f'Feigenbaum δ = {FEIGENBAUM:.4f}')
    
    ax.set_xlabel('Ratio Index n', fontsize=12)
    ax.set_ylabel('δ_n = Δ_n / Δ_{n+1}', fontsize=12)
    ax.set_title('Convergence to Feigenbaum Constant Across Measurement Bases', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(3.0, 6.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 1: MEASUREMENT BASIS INDEPENDENCE")
    print("Paper 5: Beyond Phase Expression")
    print("=" * 70)
    print("""
Question: Do different measurement bases give the same Feigenbaum δ?

Theory Prediction:
- Z-basis (H-Rz-H): P(|1⟩) = sin²(φ/2) → δ = 4.669
- Y-basis (H-Rz-S†-H): P(|1⟩) = sin²(φ/2 + π/4) → δ = 4.669 (same!)
- Ramsey-Rx: Similar sin² form → δ = 4.669
- Ramsey-Ry: Different form → δ = ???

If ALL give δ ≈ 4.669, this proves universality is FUNDAMENTAL to 
quantum measurement, not just an artifact of the specific sin² function.
    """)
    
    # Run comparison
    results = run_basis_comparison(r_min=0.5, r_max=0.85, n_r_points=15000)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BASIS INDEPENDENCE RESULTS")
    print("=" * 70)
    
    FEIGENBAUM = 4.669201609
    all_close = True
    
    for name, data in results.items():
        if data['best_delta']:
            print(f"{name:25s}: δ = {data['best_delta']:.4f} (error: {data['error_percent']:.1f}%)")
            if data['error_percent'] > 10:
                all_close = False
        else:
            print(f"{name:25s}: Not enough bifurcations found")
            all_close = False
    
    print("\n" + "-" * 70)
    if all_close:
        print("✓ ALL BASES GIVE δ ≈ 4.669!")
        print("  Conclusion: Feigenbaum universality is FUNDAMENTAL to quantum measurement")
    else:
        print("⚠ Some bases show different behavior - investigate further")
    print("-" * 70)
    
    # Generate plots
    print("\nGenerating figures...")
    
    plot_basis_comparison(results, save_path='figures/basis_bifurcation_comparison.png')
    plot_delta_convergence(results, save_path='figures/basis_delta_convergence.png')
    
    print("\n✓ Experiment complete!")
