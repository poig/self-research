"""
visualize_all_maps.py

Fast Vectorized Bifurcation Visualization for Paper 5
======================================================

Uses NumPy vectorization for fast computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
import os
import time

# Import maps from measurement_maps.py (already vectorized with NumPy)
from measurement_maps import (
    sin2_map, cos2_map, cubic_map, gaussian_map, tent_map
)


# =============================================================================
# FAST BIFURCATION DIAGRAM (FULLY VECTORIZED)
# =============================================================================

def generate_bifurcation_fast(
    M: Callable[[np.ndarray], np.ndarray],
    r_min: float = 0.5,
    r_max: float = 1.0,
    n_r: int = 1000,
    n_transient: int = 500,
    n_sample: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAST vectorized bifurcation diagram generation.
    
    Instead of iterating one r at a time, we iterate ALL r values in parallel.
    """
    rs = np.linspace(r_min, r_max, n_r)
    
    # Initialize x for all r values at once
    x = np.full(n_r, 0.4)
    
    # Transient - vectorized across all r
    for _ in range(n_transient):
        x = rs * M(x)
        x = np.clip(x, 0.001, 0.999)
    
    # Sample - collect results
    all_rs = []
    all_xs = []
    
    for _ in range(n_sample):
        x = rs * M(x)
        x = np.clip(x, 0.001, 0.999)
        all_rs.extend(rs)
        all_xs.extend(x)
    
    return np.array(all_rs), np.array(all_xs)


def find_bifurcations_fast(
    M: Callable[[np.ndarray], np.ndarray],
    r_min: float = 0.5,
    r_max: float = 1.0,
    n_r: int = 20000,
    n_transient: int = 1000,
    n_sample: int = 200,
    tol: float = 1e-5
) -> List[Dict]:
    """
    Fast bifurcation point detection using vectorized iteration.
    """
    rs = np.linspace(r_min, r_max, n_r)
    x = np.full(n_r, 0.4)
    
    # Transient
    for _ in range(n_transient):
        x = rs * M(x)
        x = np.clip(x, 0.001, 0.999)
    
    # Collect samples for period detection
    samples = np.zeros((n_r, n_sample))
    for i in range(n_sample):
        x = rs * M(x)
        x = np.clip(x, 0.001, 0.999)
        samples[:, i] = x
    
    # Detect periods for each r
    periods = np.zeros(n_r, dtype=int)
    
    for p in [1, 2, 4, 8, 16, 32]:
        if n_sample >= 2 * p:
            # Check periodicity for all r at once
            for offset in range(p):
                subset = samples[:, offset::p]
                is_periodic = np.std(subset, axis=1) < tol
                
                # Only mark if not already marked with lower period
                mask = is_periodic & (periods == 0)
                periods[mask] = p
    
    # Find bifurcation points
    bifurcations = []
    seen_from = set()
    
    for i in range(1, n_r):
        if periods[i] > periods[i-1] > 0 and periods[i] == 2 * periods[i-1]:
            if periods[i-1] not in seen_from:
                bifurcations.append({
                    'r': rs[i],
                    'from_period': periods[i-1],
                    'to_period': periods[i]
                })
                seen_from.add(periods[i-1])
    
    return bifurcations


def compute_lyapunov_fast(
    M: Callable[[np.ndarray], np.ndarray],
    dM: Callable[[np.ndarray], np.ndarray],  # Derivative
    rs: np.ndarray,
    n_iter: int = 5000
) -> np.ndarray:
    """
    Fast vectorized Lyapunov exponent computation.
    
    Args:
        M: Map function
        dM: Derivative of map
        rs: Array of r values
    """
    x = np.full(len(rs), 0.4)
    lyap_sum = np.zeros(len(rs))
    
    for _ in range(n_iter):
        # Derivative of f(x) = r*M(x) is r*M'(x)
        derivative = rs * dM(x)
        
        # Accumulate log|derivative|
        valid = np.abs(derivative) > 1e-12
        lyap_sum[valid] += np.log(np.abs(derivative[valid]))
        
        # Iterate
        x = rs * M(x)
        x = np.clip(x, 1e-10, 1 - 1e-10)
    
    return lyap_sum / n_iter


# Map derivatives for Lyapunov calculation
def sin2_deriv(x: np.ndarray) -> np.ndarray:
    """d/dx[sin²(πx)] = 2π sin(πx) cos(πx) = π sin(2πx)"""
    return np.pi * np.sin(2 * np.pi * x)

def logistic_deriv(x: np.ndarray) -> np.ndarray:
    """d/dx[4x(1-x)] = 4 - 8x"""
    return 4 - 8 * x

def gaussian_deriv(x: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """d/dx[exp(-(x-0.5)²/σ²)] = -2(x-0.5)/σ² * exp(...)"""
    return -2 * (x - 0.5) / (sigma**2) * gaussian_map(x, sigma)

def tent_deriv(x: np.ndarray) -> np.ndarray:
    """d/dx[1-|2x-1|] = ±2"""
    return np.where(x < 0.5, 2, -2)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_all_bifurcations(save_path: str = None):
    """
    Create comparison figure of all measurement maps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    maps = [
        ('sin²(πx) - Hadamard Test', sin2_map, 0.5, 0.78, 'blue'),
        ('cos²(πx) - Phase Shifted', cos2_map, 0.5, 0.75, 'green'),
        ('4x(1-x) - Logistic Map', cubic_map, 0.7, 1.0, 'red'),
        ('Gaussian - Weak Meas.', lambda x: gaussian_map(x, 0.3), 0.5, 0.78, 'purple'),
        ('Tent Map - Cusp', tent_map, 0.3, 1.0, 'orange'),
        ('sin²(πx) ZOOMED', sin2_map, 0.72, 0.76, 'darkblue'),
    ]
    
    for ax, (name, M, r_min, r_max, color) in zip(axes.flat, maps):
        print(f"  Generating {name}...")
        t0 = time.time()
        rs, xs = generate_bifurcation_fast(M, r_min, r_max, n_r=3000)
        print(f"    Done in {time.time()-t0:.2f}s")
        
        ax.scatter(rs, xs, s=0.1, c=color, alpha=0.3)
        ax.set_xlabel('Control Parameter r', fontsize=10)
        ax.set_ylabel('x (attractor)', fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Bifurcation Diagrams: Different Measurement Maps\n' + 
                 'Paper 5: Testing Feigenbaum Universality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_delta_extraction(save_path: str = None):
    """
    Extract Feigenbaum δ for all maps.
    """
    print("\nExtracting Feigenbaum δ with high precision...")
    
    maps = {
        'sin²(πx)': (sin2_map, 0.5, 0.78),
        '4x(1-x)': (cubic_map, 0.7, 1.0),
        'Gaussian': (lambda x: gaussian_map(x, 0.3), 0.5, 0.78),
    }
    
    FEIGENBAUM = 4.669201609
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(maps)))
    
    results = {}
    
    for i, (name, (M, r_min, r_max)) in enumerate(maps.items()):
        print(f"  Analyzing {name}...")
        t0 = time.time()
        
        bifs = find_bifurcations_fast(M, r_min, r_max, n_r=50000)
        
        print(f"    Found {len(bifs)} bifurcations in {time.time()-t0:.2f}s")
        
        rs = [b['r'] for b in bifs]
        periods = [b['to_period'] for b in bifs]
        
        # Plot bifurcation points
        axes[0].scatter(rs, [np.log2(p) for p in periods], 
                       s=100, color=colors[i], label=name, zorder=5)
        axes[0].plot(rs, [np.log2(p) for p in periods], 
                    color=colors[i], alpha=0.5, linewidth=2)
        
        # Compute δ
        if len(bifs) >= 3:
            deltas = [rs[j+1] - rs[j] for j in range(len(rs)-1)]
            ratios = [deltas[j] / deltas[j+1] for j in range(len(deltas)-1) 
                     if deltas[j+1] > 1e-10]
            results[name] = {
                'bifurcations': bifs,
                'deltas': deltas,
                'ratios': ratios,
                'best_delta': ratios[-1] if ratios else None
            }
            
            for b in bifs[:5]:
                print(f"      Period {b['from_period']} → {b['to_period']} at r = {b['r']:.6f}")
    
    axes[0].set_xlabel('Bifurcation Point r', fontsize=12)
    axes[0].set_ylabel('Period (log₂ scale)', fontsize=12)
    axes[0].set_title('Bifurcation Point Locations', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Right: δ convergence
    for i, (name, data) in enumerate(results.items()):
        if data['ratios']:
            axes[1].scatter(range(1, len(data['ratios'])+1), data['ratios'],
                           s=80, color=colors[i], label=name, zorder=5)
            axes[1].plot(range(1, len(data['ratios'])+1), data['ratios'],
                        color=colors[i], alpha=0.5, linewidth=2)
    
    axes[1].axhline(y=FEIGENBAUM, color='red', linestyle='--', linewidth=2,
                   label=f'Feigenbaum δ = {FEIGENBAUM:.4f}')
    axes[1].set_xlabel('Ratio Index n', fontsize=12)
    axes[1].set_ylabel('δₙ = Δₙ / Δₙ₊₁', fontsize=12)
    axes[1].set_title('Convergence to Feigenbaum Constant', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(2.0, 8.0)  # Expanded range to show all δ values
    
    plt.suptitle('Feigenbaum δ Extraction: Universality Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, results


def plot_lyapunov_comparison(save_path: str = None):
    """
    Compare Lyapunov exponents across maps (FAST).
    """
    print("\nComputing Lyapunov exponents (vectorized)...")
    
    maps = [
        ('sin²(πx)', sin2_map, sin2_deriv, 0.5, 0.78),
        ('4x(1-x)', cubic_map, logistic_deriv, 0.7, 1.0),
        ('Gaussian', lambda x: gaussian_map(x, 0.3), lambda x: gaussian_deriv(x, 0.3), 0.5, 0.78),
        ('Tent', tent_map, tent_deriv, 0.3, 1.0),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, (name, M, dM, r_min, r_max) in zip(axes.flat, maps):
        print(f"  Computing {name}...")
        t0 = time.time()
        
        rs = np.linspace(r_min, r_max, 500)
        lyaps = compute_lyapunov_fast(M, dM, rs, n_iter=5000)
        
        print(f"    Done in {time.time()-t0:.2f}s")
        
        ax.plot(rs, lyaps, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.fill_between(rs, lyaps, 0, where=lyaps > 0, alpha=0.3, color='red', label='Chaotic')
        ax.fill_between(rs, lyaps, 0, where=lyaps < 0, alpha=0.3, color='blue', label='Stable')
        
        ax.set_xlabel('Control Parameter r', fontsize=10)
        ax.set_ylabel('Lyapunov Exponent λ', fontsize=10)
        ax.set_title(f'{name}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Lyapunov Exponent Comparison: Route to Chaos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_overlay_comparison(save_path: str = None):
    """
    Overlay sin² and logistic to show universality.
    """
    print("\nGenerating overlay comparison...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate for sin²
    print("  Generating sin²...")
    rs1, xs1 = generate_bifurcation_fast(sin2_map, 0.5, 0.78, n_r=3000)
    rs1_norm = (rs1 - 0.5) / (0.78 - 0.5)
    
    # Generate for logistic
    print("  Generating logistic...")
    rs2, xs2 = generate_bifurcation_fast(cubic_map, 0.7, 1.0, n_r=3000)
    rs2_norm = (rs2 - 0.7) / (1.0 - 0.7)
    
    ax.scatter(rs1_norm, xs1, s=0.2, c='blue', alpha=0.4, label='sin²(πx)')
    ax.scatter(rs2_norm, xs2, s=0.2, c='red', alpha=0.3, label='4x(1-x)')
    
    ax.set_xlabel('Normalized Control Parameter (r - r_min) / (r_max - r_min)', fontsize=12)
    ax.set_ylabel('x (attractor)', fontsize=12)
    ax.set_title('Bifurcation Diagrams Overlay: Testing Universality\n' + 
                 '(sin² in blue, logistic in red - same structure!)', fontsize=13)
    ax.legend(fontsize=11, markerscale=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    total_start = time.time()
    
    print("=" * 70)
    print("FAST VECTORIZED BIFURCATION VISUALIZATION")
    print("Paper 5: Beyond Phase Expression")
    print("=" * 70)
    
    os.makedirs('figures', exist_ok=True)
    
    # 1. All bifurcations
    print("\n1. Generating all bifurcation diagrams...")
    plot_all_bifurcations(save_path='figures/all_bifurcations.png')
    
    # 2. Delta extraction
    print("\n2. Extracting Feigenbaum δ...")
    fig, results = plot_delta_extraction(save_path='figures/delta_extraction.png')
    
    # 3. Lyapunov comparison
    print("\n3. Computing Lyapunov exponents...")
    plot_lyapunov_comparison(save_path='figures/lyapunov_comparison.png')
    
    # 4. Overlay comparison
    print("\n4. Creating overlay diagram...")
    plot_overlay_comparison(save_path='figures/bifurcation_overlay.png')
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    FEIGENBAUM = 4.669201609
    
    print("\nFeigenbaum δ estimates:")
    for name, data in results.items():
        if data['best_delta']:
            error = abs(data['best_delta'] - FEIGENBAUM) / FEIGENBAUM * 100
            status = "✓" if error < 10 else "⚠"
            print(f"  {status} {name:15s}: δ = {data['best_delta']:.4f} (error: {error:.1f}%)")
    
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    print("\nFigures saved to figures/")
    
    plt.show()
