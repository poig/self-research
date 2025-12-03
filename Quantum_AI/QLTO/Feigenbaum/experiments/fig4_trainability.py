#!/usr/bin/env python3
"""
Figure 4: Quantum Trainability Fractal
Trainability phase diagram with fractal structure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils import FIGURES_DIR, sin2_map, compute_lyapunov


def plot_quantum_trainability_fractal(fast_mode=False):
    """Trainability phase diagram with fractal structure"""
    print("Generating: quantum_trainability_fractal.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Lyapunov vs r
    ax1 = axes[0]
    
    n_r = 300 if not fast_mode else 100
    r_values = np.linspace(0.5, 1.0, n_r)
    lyap_values = [compute_lyapunov(r, n_iter=500 if not fast_mode else 200) for r in r_values]
    
    ax1.plot(r_values, lyap_values, 'b-', lw=2)
    ax1.axhline(0, color='red', linestyle='--', lw=2, label='λ = 0 (chaos threshold)')
    ax1.fill_between(r_values, lyap_values, 0, 
                     where=[l < 0 for l in lyap_values],
                     color='green', alpha=0.3, label='Trainable (λ < 0)')
    ax1.fill_between(r_values, lyap_values, 0,
                     where=[l > 0 for l in lyap_values],
                     color='red', alpha=0.3, label='Chaotic (λ > 0)')
    
    ax1.set_xlabel('r (learning rate)', fontsize=12)
    ax1.set_ylabel('Lyapunov exponent λ', fontsize=12)
    ax1.set_title('(A) Trainability Transition', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-2, 1.5)
    
    # Panel B: Phase diagram (r, n_qubits proxy)
    ax2 = axes[1]
    
    res = 100 if fast_mode else 200
    r_vals = np.linspace(0.5, 1.0, res)
    depth_vals = np.linspace(1, 20, res)  # Circuit depth as proxy
    
    phase = np.zeros((res, res))
    
    for i, r in enumerate(r_vals):
        for j, d in enumerate(depth_vals):
            # Effective learning rate scales with depth
            r_eff = r * (1 + 0.02 * d)
            lyap = compute_lyapunov(min(r_eff, 1.0), n_iter=100)
            phase[j, i] = 1 if lyap > 0 else 0
    
    ax2.imshow(phase, extent=[0.5, 1.0, 1, 20],
              origin='lower', aspect='auto', cmap='RdYlGn_r')
    ax2.set_xlabel('r (learning rate)', fontsize=12)
    ax2.set_ylabel('Circuit depth L', fontsize=12)
    ax2.set_title('(B) Trainability Phase Diagram', fontsize=14, fontweight='bold')
    
    # Add legend patches
    legend_elements = [Patch(facecolor='green', label='Trainable'),
                      Patch(facecolor='red', label='Chaotic')]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Panel C: Convergence rate map
    ax3 = axes[2]
    
    r_conv = np.linspace(0.5, 0.9, res)  # Stay in trainable region
    x0_conv = np.linspace(0.1, 0.9, res)
    
    conv_rate = np.zeros((res, res))
    
    for i, r in enumerate(r_conv):
        for j, x0 in enumerate(x0_conv):
            x = x0
            # Find convergence rate
            for n in range(200):
                x_new = sin2_map(x, r)
                if abs(x_new - x) < 1e-6:
                    break
                x = x_new
            conv_rate[j, i] = n
    
    im = ax3.imshow(conv_rate, extent=[0.5, 0.9, 0.1, 0.9],
                   origin='lower', aspect='auto', cmap='viridis_r')
    ax3.set_xlabel('r (learning rate)', fontsize=12)
    ax3.set_ylabel('x₀ (initial state)', fontsize=12)
    ax3.set_title('(C) Convergence Iterations', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Iterations to converge', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantum_trainability_fractal.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved quantum_trainability_fractal.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 4')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    args = parser.parse_args()
    plot_quantum_trainability_fractal(fast_mode=args.fast)
