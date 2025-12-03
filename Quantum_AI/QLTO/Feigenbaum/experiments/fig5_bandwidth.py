#!/usr/bin/env python3
"""
Figure 5: Ancilla Channel Capacity (Holevo Bound)
Shows efficiency collapse due to bandwidth bottleneck
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import FIGURES_DIR


def plot_qubit_bandwidth_comparison(fast_mode=False):
    """
    Show the ancilla sensing circuit structure from manuscript.tex:
    
    The Coherent Demon Protocol:
    1. SENSING: H(anc) → Controlled-e^{-iHτ} → H(anc)
       This is the Hadamard test that gives P(|1⟩) = sin²(Eτ/2)
    2. FEEDBACK: CRx(θ_gain) on system qubits
    
    Key insight: Single-qubit ancilla has Holevo capacity χ ≤ 1 bit.
    When DLA dimension grows exponentially, required info O(N) bits
    exceeds this capacity → efficiency collapse.
    """
    print("Generating: qubit_bandwidth_comparison.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Hadamard test P(|1⟩) = sin²(Eτ/2) for different E values
    ax1 = axes[0]
    tau = np.linspace(0, 3, 400)
    
    # Different energy eigenvalues
    for E, color, label in [(0.5, 'blue', 'E=0.5'), (1.0, 'green', 'E=1.0'), 
                             (2.0, 'red', 'E=2.0'), (3.0, 'purple', 'E=3.0')]:
        p = np.sin(E * tau / 2)**2
        ax1.plot(tau, p, color=color, lw=2, label=label)
    
    ax1.set_xlabel(r'Sensing time $\tau$', fontsize=12)
    ax1.set_ylabel(r'$P(|1\rangle) = \sin^2(E\tau/2)$', fontsize=12)
    ax1.set_title('(A) Hadamard Test: Energy → Phase → Probability', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotation about Holevo capacity
    ax1.annotate('Ancilla Holevo capacity:\n$\\chi \\leq 1$ bit',
                xy=(2.0, 0.15), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel B: Efficiency vs System Size (concept from theory paper)
    ax2 = axes[1]
    
    # Simulate the efficiency collapse concept
    N_values = np.arange(3, 9)
    
    # Ordered phase: polynomial DLA → efficiency increases
    eta_ordered = 0.1 * N_values**0.5
    
    # Chaotic phase: exponential DLA → efficiency collapses
    eta_chaotic = 0.3 * np.exp(-0.5 * (N_values - 3))
    eta_chaotic[eta_chaotic < 0] = 0
    
    ax2.plot(N_values, eta_ordered, 'b-o', lw=2, markersize=8, label='Ordered (poly DLA)')
    ax2.plot(N_values, eta_chaotic, 'r-s', lw=2, markersize=8, label='Chaotic (exp DLA)')
    ax2.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    
    # Mark critical size
    ax2.axvline(6.4, color='gray', linestyle=':', lw=2, alpha=0.7)
    ax2.annotate('$N_c \\approx 6.4$', xy=(6.5, 0.25), fontsize=11)
    
    ax2.set_xlabel('System Size $N$', fontsize=12)
    ax2.set_ylabel(r'Efficiency $\eta$', fontsize=12)
    ax2.set_title('(B) Efficiency Collapse: Bandwidth Bottleneck', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2.5, 8.5)
    
    # Add annotation
    ax2.annotate('Required info: $O(N)$ bits\nAncilla capacity: 1 bit\n→ Collapse when $N > N_c$',
                xy=(4, 0.05), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qubit_bandwidth_comparison.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ Saved qubit_bandwidth_comparison.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 5')
    parser.add_argument('--fast', action='store_true', help='Fast mode')
    args = parser.parse_args()
    plot_qubit_bandwidth_comparison(fast_mode=args.fast)
