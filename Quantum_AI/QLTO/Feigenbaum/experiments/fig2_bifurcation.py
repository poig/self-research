#!/usr/bin/env python3
"""
Figure 2: Quantum Bifurcation 2D
Classic bifurcation diagram with Feigenbaum constant verification
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    FIGURES_DIR, DELTA,
    compute_bifurcation, find_bifurcation_points
)


def plot_quantum_bifurcation_2d(fast_mode=False):
    """Classic bifurcation diagram with Feigenbaum analysis"""
    print("Generating: quantum_bifurcation_2d.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Full bifurcation diagram
    ax1 = axes[0]
    n_r = 800 if not fast_mode else 300
    r_values = np.linspace(0.5, 1.0, n_r)
    
    r_data, x_data = compute_bifurcation(
        r_values, 
        n_iter=150 if fast_mode else 300,
        n_last=80 if fast_mode else 150
    )
    
    ax1.scatter(r_data, x_data, s=0.1, c='darkblue', alpha=0.4)
    
    # Set axis limits first so vertical lines span full range
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0, 1)
    
    # Shade the collapse region (r > 0.88)
    ax1.axvspan(0.88, 1.0, alpha=0.15, color='red', label='Collapse region')
    ax1.annotate('Collapse\n(x→0)', xy=(0.94, 0.5), fontsize=11, 
                ha='center', va='center', color='darkred', fontweight='bold')
    
    # Mark bifurcation points with full-height lines
    bif_points = find_bifurcation_points(6)
    colors = ['red', 'orange', 'green', 'purple', 'cyan', 'magenta']
    labels = ['r₁ (1→2)', 'r₂ (2→4)', 'r₃ (4→8)', 'r₄ (8→16)', 'r₅ (16→32)', 'r₆ (32→64)']
    
    for r_bif, color, label in zip(bif_points, colors, labels):
        ax1.axvline(r_bif, color=color, linestyle='--', lw=1.5, alpha=0.7, label=label, ymin=0, ymax=1)
    
    ax1.set_xlabel('r (learning rate)', fontsize=12)
    ax1.set_ylabel('x* (steady state)', fontsize=12)
    ax1.set_title('(A) Period-Doubling Cascade', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Feigenbaum constant verification
    ax2 = axes[1]
    
    if len(bif_points) >= 4:
        # Compute ratios - need at least 3 consecutive points for each δ
        deltas = []
        for i in range(len(bif_points) - 2):
            delta_r1 = bif_points[i+1] - bif_points[i]
            delta_r2 = bif_points[i+2] - bif_points[i+1]
            if delta_r2 > 1e-12:  # Smaller threshold for higher ratios
                deltas.append(delta_r1 / delta_r2)
        
        x_pos = np.arange(len(deltas))
        bars = ax2.bar(x_pos, deltas, color='steelblue', edgecolor='black', lw=2, alpha=0.8)
        ax2.axhline(DELTA, color='red', linestyle='--', lw=3, 
                   label=f'Feigenbaum δ = {DELTA:.4f}')
        
        ax2.set_xlabel('Ratio index', fontsize=12)
        ax2.set_ylabel('δₙ = (rₙ - rₙ₋₁)/(rₙ₊₁ - rₙ)', fontsize=12)
        ax2.set_title('(B) Feigenbaum Constant Convergence', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'δ{i+1}' for i in range(len(deltas))])
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for i, (bar, val) in enumerate(zip(bars, deltas)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantum_bifurcation_2d.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved quantum_bifurcation_2d.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 2')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    args = parser.parse_args()
    plot_quantum_bifurcation_2d(fast_mode=args.fast)
