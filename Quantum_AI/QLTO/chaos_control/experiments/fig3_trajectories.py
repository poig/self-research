"""
Figure 3: VQA Trajectories at Different Learning Rates

Clean comparison of trajectory behavior across regimes.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import sin2_map, compute_trajectory, FIGURES_DIR


def main():
    print("Generating VQA Trajectories...")
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    regimes = [
        (0.55, 'Period-1 (Stable)', 'blue'),
        (0.68, 'Period-2', 'green'),
        (0.72, 'Period-4', 'orange'),
        (0.78, 'Chaos', 'red'),
    ]
    
    for ax, (r, label, color) in zip(axes, regimes):
        traj = compute_trajectory(0.5, r, 300)
        
        # Plot last 60 iterations (steady state)
        ax.plot(range(60), traj[240:300], '-o', color=color, 
                markersize=3, lw=1.2, alpha=0.8)
        
        ax.set_xlabel('Iteration n', fontsize=10)
        ax.set_ylabel('x', fontsize=10)
        ax.set_title(f'{label}\nr = {r}', fontsize=11, fontweight='bold', color=color)
        ax.set_ylim(0.3, 0.8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('VQA Trajectory Behavior Across Regimes', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig3_trajectories.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
