#!/usr/bin/env python3
"""
Figure 3: Unified Fractal Bifurcation
3D-style visualization combining bifurcation with fractal structure
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import FIGURES_DIR, sin2_map, compute_lyapunov


def plot_unified_fractal_bifurcation(fast_mode=False):
    """3D-style visualization combining bifurcation with fractal structure"""
    print("Generating: unified_fractal_bifurcation.png")
    
    fig = plt.figure(figsize=(16, 6))
    
    # Panel A: 3D bifurcation surface
    ax1 = fig.add_subplot(131, projection='3d')
    
    n_r = 150 if fast_mode else 300
    n_x0 = 20 if fast_mode else 40
    r_range = np.linspace(0.6, 1.0, n_r)
    x0_range = np.linspace(0.1, 0.9, n_x0)
    
    for x0 in x0_range:
        x_finals = []
        for r in r_range:
            x = x0
            for _ in range(100):
                x = sin2_map(x, r)
            x_finals.append(x)
        
        ax1.plot(r_range, [x0]*len(r_range), x_finals, 
                c=plt.cm.viridis(x0), alpha=0.6, lw=0.8)
    
    ax1.set_xlabel('r', fontsize=10)
    ax1.set_ylabel('x₀', fontsize=10)
    ax1.set_zlabel('x*', fontsize=10)
    ax1.set_title('(A) Bifurcation Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    
    # Panel B: Mandelbrot-style trainability map
    ax2 = fig.add_subplot(132)
    
    res = 80 if fast_mode else 200
    r_vals = np.linspace(0.5, 1.0, res)
    x0_vals = np.linspace(0.01, 0.99, res)
    
    trainability = np.zeros((res, res))
    
    for i, r in enumerate(r_vals):
        for j, x0 in enumerate(x0_vals):
            lyap = compute_lyapunov(r, n_iter=200 if fast_mode else 500, x0=x0)
            trainability[j, i] = lyap
    
    # Clip for visualization
    trainability = np.clip(trainability, -2, 2)
    
    im = ax2.imshow(trainability, extent=[0.5, 1.0, 0.01, 0.99],
                   origin='lower', aspect='auto', cmap='RdBu_r')
    
    # Mark trainability boundary (λ = 0)
    ax2.contour(r_vals, x0_vals, trainability, levels=[0], 
               colors='black', linewidths=2, linestyles='--')
    
    ax2.set_xlabel('r (learning rate)', fontsize=10)
    ax2.set_ylabel('x₀ (initial state)', fontsize=10)
    ax2.set_title('(B) Lyapunov Exponent Map', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('λ (Lyapunov)', fontsize=10)
    
    # Panel C: Fractal boundary detail
    ax3 = fig.add_subplot(133)
    
    # Zoom into chaotic transition region
    r_zoom = np.linspace(0.85, 1.0, res)
    x0_zoom = np.linspace(0.3, 0.7, res)
    
    fractal = np.zeros((res, res))
    
    for i, r in enumerate(r_zoom):
        for j, x0 in enumerate(x0_zoom):
            x = x0
            # Count iterations to escape/converge
            for n in range(100):
                x = sin2_map(x, r)
                if x < 0.01 or x > 0.99:
                    break
            fractal[j, i] = n
    
    ax3.imshow(fractal, extent=[0.85, 1.0, 0.3, 0.7],
              origin='lower', aspect='auto', cmap='magma')
    ax3.set_xlabel('r', fontsize=10)
    ax3.set_ylabel('x₀', fontsize=10)
    ax3.set_title('(C) Fractal Basin Boundary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "unified_fractal_bifurcation.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved unified_fractal_bifurcation.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 3')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    args = parser.parse_args()
    plot_unified_fractal_bifurcation(fast_mode=args.fast)
