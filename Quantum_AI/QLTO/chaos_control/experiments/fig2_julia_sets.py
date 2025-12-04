"""
Figure 2: Julia Sets - Structure Behind VQA Dynamics

Four Julia sets showing the fractal structure corresponding to each dynamical regime.
Separate from bifurcation diagram for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import FIGURES_DIR


def julia_set(c, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, 
              width=500, height=500, max_iter=150):
    """Generate Julia set for complex parameter c."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    output = np.zeros(Z.shape)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + c
        output[mask] = i
    
    return output


def main():
    print("Generating Julia Sets visualization...")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Julia set parameters for each regime
    julia_params = [
        (-0.4 + 0.6j, 'Period-1\n(Connected)', 'blue', 0.55),
        (-0.74543 + 0.11301j, 'Period-2\n(Dendrite)', 'green', 0.68),
        (-0.1 + 0.8j, 'Period-4\n(Fragmenting)', 'orange', 0.72),
        (-0.8 + 0.2j, 'Chaos\n(Cantor Dust)', 'red', 0.85),
    ]
    
    for ax, (c, label, color, r) in zip(axes, julia_params):
        print(f"  Computing Julia set for c = {c}...")
        julia = julia_set(c, max_iter=150, width=500, height=500)
        
        ax.imshow(julia, cmap='hot', extent=[-1.5, 1.5, -1.5, 1.5])
        ax.set_title(f'{label}\nr = {r}', fontsize=12, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Julia Set Structure ↔ VQA Dynamical Regime', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig2_julia_sets.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
