"""
Figure 1: Bifurcation Diagram with Period-Doubling Cascade

Clean, focused visualization of the core result:
VQA dynamics exhibit Feigenbaum period-doubling as learning rate increases.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import sin2_map, FIGURES_DIR


def main():
    print("Generating Bifurcation Diagram...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute bifurcation data
    n_r = 1000
    r_range = np.linspace(0.50, 0.88, n_r)
    all_r, all_x = [], []
    
    for r in r_range:
        x = 0.5
        # Warmup
        for _ in range(800):
            x = sin2_map(x, r)
        # Collect steady state
        for _ in range(200):
            x = sin2_map(x, r)
            all_r.append(r)
            all_x.append(x)
    
    # Plot
    ax.scatter(all_r, all_x, s=0.1, c='darkblue', alpha=0.5)
    
    # Mark key bifurcation points
    bifurcations = [
        (0.50, 'Period-1', 'blue'),
        (0.63, 'Period-2', 'green'),
        (0.70, 'Period-4', 'orange'),
        (0.72, 'Period-8', 'red'),
    ]
    
    for r, label, color in bifurcations:
        ax.axvline(r, color=color, linestyle='--', lw=1.5, alpha=0.7)
        ax.text(r + 0.005, 0.95, label, fontsize=9, color=color, rotation=90, va='top')
    
    # Mark chaos onset
    ax.axvline(0.731, color='black', linestyle='-', lw=2)
    ax.text(0.735, 0.5, r'$r_\infty$ (Chaos)', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Control Parameter r (Effective Learning Rate)', fontsize=12)
    ax.set_ylabel('Steady-State x', fontsize=12)
    ax.set_title('Period-Doubling Cascade in VQA Dynamics', fontsize=14, fontweight='bold')
    ax.set_xlim(0.50, 0.88)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add Feigenbaum constant annotation
    ax.text(0.82, 0.15, r'$\delta = 4.669...$' + '\n(Feigenbaum)', 
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig1_bifurcation.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
