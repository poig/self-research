"""
Figure 8: Return Map - The Hidden 1D Dynamics

Clean visualization showing E_{n+1} vs E_n collapses onto sin² curve.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import sin2_map, FIGURES_DIR


def main():
    print("Generating Return Map...")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    regimes = [
        ("Stable", 0.60, 'blue'),
        ("Periodic", 0.72, 'orange'),
        ("Chaotic", 0.82, 'red'),
    ]
    
    for ax, (name, r, color) in zip(axes, regimes):
        # Generate trajectory
        np.random.seed(42)
        n_steps = 150
        
        traj = [0.5]
        x = 0.5
        for _ in range(n_steps):
            x = sin2_map(x, r) + np.random.normal(0, 0.005)
            x = np.clip(x, 0.01, 0.99)
            traj.append(x)
        
        traj = np.array(traj)
        x_n = traj[50:-1]  # Skip transient
        x_np1 = traj[51:]
        
        # Plot return map
        scatter = ax.scatter(x_n, x_np1, c=np.arange(len(x_n)), 
                            cmap='viridis', s=15, alpha=0.7)
        
        # Theoretical curve
        x_theory = np.linspace(min(x_n), max(x_n), 100)
        y_theory = sin2_map(x_theory, r)
        ax.plot(x_theory, y_theory, 'k--', lw=2, alpha=0.6, label=f'Theory: r·sin²(πx)')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(r'$x_n$', fontsize=11)
        ax.set_ylabel(r'$x_{n+1}$', fontsize=11)
        ax.set_title(f'{name} (r = {r})', fontsize=12, fontweight='bold', color=color)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, 0.85)
        ax.set_ylim(0.2, 0.85)
    
    plt.suptitle('Return Map: Data Collapses onto sin² Curve', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig8_return_map.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
