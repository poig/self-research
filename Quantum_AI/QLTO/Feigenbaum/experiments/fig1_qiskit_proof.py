#!/usr/bin/env python3
"""
Figure 1: Feigenbaum Qiskit Proof
2-panel: (A) Hadamard test verification, (B) Cobweb diagram
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    FIGURES_DIR, HAS_QISKIT,
    sin2_map, qiskit_hadamard_measurement_1qubit
)


def plot_feigenbaum_qiskit_proof(fast_mode=False):
    """
    2-panel figure showing:
    (A) Qiskit verification of Hadamard test: P(|1⟩) = sin²(φ/2)
    (B) Cobweb diagram showing period-doubling dynamics
    
    Bifurcation diagram and δ convergence are in Figure 2.
    """
    print("Generating: feigenbaum_qiskit_proof.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Hadamard test verification
    ax1 = axes[0]
    phi = np.linspace(0, 4*np.pi, 400)
    
    # Analytical sin²(φ/2)
    p_analytical = np.sin(phi/2)**2
    ax1.plot(phi, p_analytical, 'b-', lw=3, label=r'Theory: $\sin^2(\phi/2)$')
    
    # Qiskit verification
    if HAS_QISKIT:
        n_pts = 40 if fast_mode else 80
        phi_qiskit = np.linspace(0, 4*np.pi, n_pts)
        p_qiskit = [qiskit_hadamard_measurement_1qubit(p) for p in phi_qiskit]
        ax1.scatter(phi_qiskit, p_qiskit, c='red', s=40, zorder=5, 
                   label='Qiskit Statevector', edgecolors='darkred', linewidths=1, alpha=0.8)
    
    ax1.set_xlabel(r'$\phi$ (phase angle)', fontsize=12)
    ax1.set_ylabel(r'$P(|1\rangle)$', fontsize=12)
    ax1.set_title('(A) Hadamard Test: Measurement Probability', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4*np.pi)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax1.set_xticklabels(['0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
    
    # Panel B: Cobweb diagram
    ax2 = axes[1]
    r = 0.85  # Period-4 regime for interesting dynamics
    x = np.linspace(0, 1, 200)
    y = sin2_map(x, r)
    
    ax2.plot(x, y, 'b-', lw=3, label=rf'$f(x) = {r} \cdot \sin^2(\pi x)$')
    ax2.plot(x, x, 'k--', lw=2, alpha=0.5, label='$y = x$')
    
    # Show cobweb for iterations
    x_cob = 0.2
    ax2.plot([x_cob, x_cob], [0, sin2_map(x_cob, r)], 'r-', lw=1.5, alpha=0.8)
    for i in range(30):
        x_new = sin2_map(x_cob, r)
        ax2.plot([x_cob, x_new], [x_new, x_new], 'r-', lw=1.5, alpha=0.7)
        ax2.plot([x_new, x_new], [x_new, sin2_map(x_new, r)], 'r-', lw=1.5, alpha=0.7)
        x_cob = x_new
    
    ax2.set_xlabel(r'$x_n$', fontsize=12)
    ax2.set_ylabel(r'$x_{n+1}$', fontsize=12)
    ax2.set_title('(B) Cobweb: Period-4 Orbit', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feigenbaum_qiskit_proof.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved feigenbaum_qiskit_proof.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 1')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    args = parser.parse_args()
    plot_feigenbaum_qiskit_proof(fast_mode=args.fast)
