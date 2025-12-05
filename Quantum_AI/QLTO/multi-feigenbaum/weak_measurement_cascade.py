"""
weak_measurement_cascade.py

Experiment 2: Weak Measurement Phase Transition
================================================

Goal: Find the critical measurement strength g_c where chaos onsets.

Key Question: At what measurement strength does period-doubling begin?

Theory:
- g = 1: Strong (projective) measurement → sin² map → chaos possible
- g = 0: No measurement → flat map → no chaos
- g = g_c: Critical point where chaos first appears

This connects to Measurement-Induced Phase Transitions (MIPT) literature!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import os
import time


# =============================================================================
# WEAK MEASUREMENT MAP (VECTORIZED)
# =============================================================================

def weak_measurement_map(x: np.ndarray, g: float) -> np.ndarray:
    """
    Weak measurement with strength g ∈ [0, 1].
    
    P(|1⟩; g) = (1-g)·0.5 + g·sin²(πx)
    
    - g = 0: P = 0.5 (no information, flat map)
    - g = 1: P = sin²(πx) (projective, full Feigenbaum)
    - 0 < g < 1: Interpolation (reduced nonlinearity)
    """
    if g < 1e-10:
        return np.full_like(x, 0.5)
    
    projective = np.sin(np.pi * x) ** 2
    return (1 - g) * 0.5 + g * projective


# =============================================================================
# FAST VECTORIZED ANALYSIS
# =============================================================================

def compute_lyapunov_fast(g: float, rs: np.ndarray, n_iter: int = 5000) -> np.ndarray:
    """
    Fast vectorized Lyapunov exponent for weak measurement map.
    """
    x = np.full(len(rs), 0.4)
    lyap_sum = np.zeros(len(rs))
    
    for _ in range(n_iter):
        # Derivative of f(x) = r * M(x, g)
        # d/dx[r * ((1-g)*0.5 + g*sin²(πx))] = r * g * π * sin(2πx)
        derivative = rs * g * np.pi * np.sin(2 * np.pi * x)
        
        valid = np.abs(derivative) > 1e-12
        lyap_sum[valid] += np.log(np.abs(derivative[valid]))
        
        x = rs * weak_measurement_map(x, g)
        x = np.clip(x, 1e-10, 1 - 1e-10)
    
    return lyap_sum / n_iter


def generate_bifurcation_weak(g: float, r_min: float, r_max: float, 
                               n_r: int = 2000, n_transient: int = 500, n_sample: int = 100):
    """Fast bifurcation diagram for given measurement strength g."""
    rs = np.linspace(r_min, r_max, n_r)
    x = np.full(n_r, 0.4)
    
    for _ in range(n_transient):
        x = rs * weak_measurement_map(x, g)
        x = np.clip(x, 0.001, 0.999)
    
    all_rs, all_xs = [], []
    for _ in range(n_sample):
        x = rs * weak_measurement_map(x, g)
        x = np.clip(x, 0.001, 0.999)
        all_rs.extend(rs)
        all_xs.extend(x)
    
    return np.array(all_rs), np.array(all_xs)


def find_chaos_onset_fast(g_values: np.ndarray, r_chaos: float = 0.9) -> Tuple[Optional[float], Dict]:
    """
    Find critical g_c where Lyapunov exponent first becomes positive.
    """
    lyapunov_vs_g = []
    
    for g in g_values:
        # Compute Lyapunov at fixed high r
        rs = np.array([r_chaos])
        lyap = compute_lyapunov_fast(g, rs, n_iter=3000)[0]
        lyapunov_vs_g.append(lyap)
    
    lyapunov_vs_g = np.array(lyapunov_vs_g)
    
    # Find transition point
    g_c = None
    for i in range(len(g_values) - 1):
        if lyapunov_vs_g[i] < 0 and lyapunov_vs_g[i+1] >= 0:
            # Linear interpolation
            g_c = g_values[i] + (g_values[i+1] - g_values[i]) * \
                  (0 - lyapunov_vs_g[i]) / (lyapunov_vs_g[i+1] - lyapunov_vs_g[i])
            break
    
    return g_c, {'g': g_values, 'lyapunov': lyapunov_vs_g}


def compute_phase_diagram_fast(g_values: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    """
    Compute 2D phase diagram: Lyapunov(g, r).
    """
    lyap_grid = np.zeros((len(g_values), len(r_values)))
    
    for i, g in enumerate(g_values):
        lyap_grid[i, :] = compute_lyapunov_fast(g, r_values, n_iter=2000)
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(g_values)}")
    
    return lyap_grid


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_phase_diagram(g_values, r_values, lyap_grid, g_c=None, save_path=None):
    """Plot 2D phase diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    R, G = np.meshgrid(r_values, g_values)
    im = ax.pcolormesh(R, G, lyap_grid, cmap='RdBu_r', 
                       vmin=-0.5, vmax=0.5, shading='auto')
    
    # Chaos boundary (λ = 0)
    ax.contour(R, G, lyap_grid, levels=[0], colors='yellow', linewidths=2)
    
    if g_c is not None:
        ax.axhline(y=g_c, color='lime', linestyle='--', linewidth=2, 
                   label=f'g_c ≈ {g_c:.3f}')
        ax.legend(loc='lower right', fontsize=12)
    
    ax.set_xlabel('Control Parameter r', fontsize=12)
    ax.set_ylabel('Measurement Strength g', fontsize=12)
    ax.set_title('Weak Measurement Phase Diagram\n(Yellow = Chaos Boundary λ=0)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Lyapunov Exponent λ', fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_bifurcation_gallery(g_values: List[float], save_path=None):
    """Plot bifurcation diagrams for different g values."""
    n = len(g_values)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for ax, g in zip(axes, g_values):
        print(f"  Generating g = {g:.2f}...")
        rs, xs = generate_bifurcation_weak(g, 0.5, 1.0, n_r=2000)
        
        ax.scatter(rs, xs, s=0.3, c='blue', alpha=0.5)
        ax.set_xlabel('Control r', fontsize=10)
        ax.set_ylabel('x', fontsize=10)
        ax.set_title(f'g = {g:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Bifurcation Cascade vs Measurement Strength g', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_lyapunov_vs_g(g_data, g_c=None, save_path=None):
    """Plot Lyapunov exponent vs measurement strength."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(g_data['g'], g_data['lyapunov'], 'b-', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', label='λ = 0 (chaos threshold)')
    
    if g_c is not None:
        ax.axvline(x=g_c, color='green', linestyle='--', linewidth=2, label=f'g_c ≈ {g_c:.3f}')
    
    ax.fill_between(g_data['g'], g_data['lyapunov'], 0, 
                    where=g_data['lyapunov'] > 0, alpha=0.3, color='red', label='Chaotic (λ > 0)')
    ax.fill_between(g_data['g'], g_data['lyapunov'], 0, 
                    where=g_data['lyapunov'] < 0, alpha=0.3, color='blue', label='Stable (λ < 0)')
    
    ax.set_xlabel('Measurement Strength g', fontsize=12)
    ax.set_ylabel('Lyapunov Exponent λ', fontsize=12)
    ax.set_title('Measurement-Induced Phase Transition\n(λ=0 marks chaos onset at r=0.9)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
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
    print("EXPERIMENT 2: WEAK MEASUREMENT PHASE TRANSITION (FAST)")
    print("Paper 5: Beyond Phase Expression")
    print("=" * 70)
    print("""
Question: At what measurement strength g_c does chaos first appear?

Physical Interpretation:
- g = 0: No information → No chaos (flat map)
- g = 1: Full measurement → Feigenbaum chaos (sin² map)
- g = g_c: Critical threshold (MIPT-like)
    """)
    
    os.makedirs('figures', exist_ok=True)
    
    # 1. Find chaos onset
    print("\n" + "-" * 70)
    print("Step 1: Finding critical measurement strength g_c")
    print("-" * 70)
    
    g_values = np.linspace(0.1, 1.0, 40)
    r_chaos = 0.9  # High r where g=1 would be chaotic
    
    t0 = time.time()
    g_c, lyap_data = find_chaos_onset_fast(g_values, r_chaos)
    print(f"  Computed in {time.time()-t0:.1f}s")
    
    if g_c is not None:
        print(f"\n✓ Critical measurement strength: g_c ≈ {g_c:.3f}")
        print(f"  Interpretation: Chaos requires g > {g_c:.3f}")
    else:
        # Check if always chaotic or never chaotic
        if lyap_data['lyapunov'][-1] > 0:
            print("✓ System is chaotic at g=1, checking lower values...")
            # Find where it transitions
            for i, (g, lyap) in enumerate(zip(g_values, lyap_data['lyapunov'])):
                if lyap > 0:
                    g_c = g
                    print(f"✓ Chaos starts around g ≈ {g_c:.2f}")
                    break
        else:
            print("⚠ No chaos detected at r=0.9. Trying higher r...")
    
    # 2. Phase diagram
    print("\n" + "-" * 70)
    print("Step 2: Computing 2D phase diagram")
    print("-" * 70)
    
    t0 = time.time()
    g_grid = np.linspace(0.2, 1.0, 25)
    r_grid = np.linspace(0.5, 1.0, 40)
    lyap_grid = compute_phase_diagram_fast(g_grid, r_grid)
    print(f"  Computed in {time.time()-t0:.1f}s")
    
    # 3. Bifurcation gallery
    print("\n" + "-" * 70)
    print("Step 3: Generating bifurcation gallery")
    print("-" * 70)
    
    plot_bifurcation_gallery([0.3, 0.5, 0.7, 1.0], save_path='figures/weak_bifurcation_gallery.png')
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if g_c:
        print(f"\n✓ Critical measurement strength: g_c ≈ {g_c:.3f}")
        print(f"  • g < {g_c:.2f}: Weak measurement → Stable (no chaos)")
        print(f"  • g > {g_c:.2f}: Strong measurement → Feigenbaum chaos")
    
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    
    # Generate plots
    print("\nGenerating figures...")
    plot_lyapunov_vs_g(lyap_data, g_c, save_path='figures/weak_lyapunov_vs_g.png')
    plot_phase_diagram(g_grid, r_grid, lyap_grid, g_c, save_path='figures/weak_phase_diagram.png')
    
    print("\n✓ Experiment complete!")
