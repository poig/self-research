"""
Fractal-Bifurcation Duality Visualization

Demonstrates the deep connection between Julia set fractals 
and the VQA bifurcation dynamics.

Key insight: Both emerge from z² + c iteration (quantum measurement nonlinearity)
- Connected Julia → Stable optimization
- Cantor dust Julia → Chaotic (untrainable)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from core import sin2_map, compute_trajectory, FIGURES_DIR


def julia_set(c, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, 
              width=400, height=400, max_iter=100):
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


def mandelbrot_set(xmin=-2.2, xmax=0.8, ymin=-1.2, ymax=1.2, 
                   width=400, height=320, max_iter=100):
    """Generate Mandelbrot set (parameter space)."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    output = np.zeros(C.shape)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        output[mask] = i
    
    return output


def generate_bifurcation_data(r_min=0.50, r_max=0.88, n_r=800, 
                               n_warmup=600, n_plot=150):
    """Generate bifurcation diagram data."""
    r_range = np.linspace(r_min, r_max, n_r)
    all_r, all_x = [], []
    
    for r in r_range:
        x = 0.5
        for _ in range(n_warmup):
            x = sin2_map(x, r)
        for _ in range(n_plot):
            x = sin2_map(x, r)
            all_r.append(r)
            all_x.append(x)
    
    return np.array(all_r), np.array(all_x)


def main():
    print("Generating Fractal-Bifurcation Duality visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1.2, 0.8], 
                  hspace=0.3, wspace=0.25)

    # Julia sets for different regimes
    julia_params = [
        (-0.4 + 0.6j, 'Period-1\n(Connected)', 'blue'),
        (-0.74543 + 0.11301j, 'Period-2\n(Dendrite)', 'green'),
        (-0.1 + 0.8j, 'Period-4\n(Fragmenting)', 'orange'),
        (-0.8 + 0.2j, 'Chaos\n(Cantor Dust)', 'red'),
    ]
    r_values = [0.55, 0.68, 0.72, 0.85]
    colors_list = ['blue', 'green', 'orange', 'red']

    # Row 1: Julia sets
    print("  Computing Julia sets...")
    for i, ((c, label, color), r) in enumerate(zip(julia_params, r_values)):
        ax = fig.add_subplot(gs[0, i])
        julia = julia_set(c, max_iter=100, width=350, height=350)
        ax.imshow(julia, cmap='hot', extent=[-1.5, 1.5, -1.5, 1.5])
        ax.set_title(f'{label}\nr = {r}, c = {c:.2f}', fontsize=11, 
                    color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Row 2: Bifurcation diagram
    print("  Computing bifurcation diagram...")
    ax_bif = fig.add_subplot(gs[1, :3])
    
    all_r, all_x = generate_bifurcation_data()
    ax_bif.scatter(all_r, all_x, s=0.15, c='darkblue', alpha=0.6)

    # Mark regime boundaries
    for r, col in zip(r_values, colors_list):
        ax_bif.axvline(r, color=col, linestyle='--', lw=2.5, alpha=0.9)

    ax_bif.axvline(0.731, color='black', linestyle='-', lw=2.5)
    ax_bif.text(0.735, 0.92, 'r∞\n(Chaos)', fontsize=10, ha='left', fontweight='bold')

    ax_bif.set_xlabel('Control Parameter r (Learning Rate γ)', fontsize=12)
    ax_bif.set_ylabel('Steady-State x', fontsize=12)
    ax_bif.set_title('Period-Doubling Cascade: Observable VQA Dynamics', 
                    fontsize=14, fontweight='bold')
    ax_bif.set_xlim(0.50, 0.88)
    ax_bif.set_ylim(0, 1)
    ax_bif.grid(True, alpha=0.3)

    # Mandelbrot set (parameter space)
    print("  Computing Mandelbrot set...")
    ax_mandel = fig.add_subplot(gs[1, 3])
    mandel = mandelbrot_set(xmin=-2, xmax=0.6, ymin=-1.1, ymax=1.1, 
                           width=450, height=350)
    ax_mandel.imshow(mandel, cmap='hot', extent=[-2, 0.6, -1.1, 1.1], aspect='auto')

    # Mark Julia set parameters on Mandelbrot
    for (c, label, color), r in zip(julia_params, r_values):
        ax_mandel.scatter([c.real], [c.imag], color=color, s=150, 
                         edgecolors='white', lw=2, zorder=5)

    ax_mandel.set_xlabel('Re(c)', fontsize=11)
    ax_mandel.set_ylabel('Im(c)', fontsize=11)
    ax_mandel.set_title('Mandelbrot Set\n(Parameter Space)', fontsize=12, fontweight='bold')

    # Row 3: Explanatory panels
    
    # The sin² map
    ax_map = fig.add_subplot(gs[2, 1])
    x_vals = np.linspace(0, 1, 200)
    for r, col in zip([0.55, 0.68, 0.72, 0.85], colors_list):
        y_vals = sin2_map(x_vals, r)
        ax_map.plot(x_vals, y_vals, color=col, lw=2.5, label=f'r={r}')

    ax_map.plot(x_vals, x_vals, 'k--', lw=1.5, alpha=0.5)
    ax_map.set_xlabel('xₙ', fontsize=11)
    ax_map.set_ylabel('f(x) = r·sin²(πx)', fontsize=11)
    ax_map.set_title('The sin² Map', fontsize=12, fontweight='bold')
    ax_map.legend(fontsize=9, loc='upper right')
    ax_map.grid(True, alpha=0.3)

    # Trajectories
    ax_traj = fig.add_subplot(gs[2, 2])
    for r, col, name in zip([0.55, 0.72], ['blue', 'orange'], ['Stable', 'Period-4']):
        traj = compute_trajectory(0.5, r, 100)
        ax_traj.plot(range(60), traj[:60], color=col, lw=1.5, label=f'{name} (r={r})')
        
    ax_traj.set_xlabel('Iteration n', fontsize=11)
    ax_traj.set_ylabel('x', fontsize=11)
    ax_traj.set_title('VQA Trajectories', fontsize=12, fontweight='bold')
    ax_traj.legend(fontsize=9)
    ax_traj.grid(True, alpha=0.3)

    # Key insight box
    ax_insight = fig.add_subplot(gs[2, 3])
    insight_text = """The Duality:

Julia Set ↔ Bifurcation
(Structure)   (Dynamics)

• Connected Julia
  → Stable optimization
  
• Cantor dust Julia  
  → Chaotic (untrainable)

Both emerge from:
z² + c iteration
(quantum measurement)"""
    ax_insight.text(0.5, 0.5, insight_text, fontsize=10, ha='center', va='center',
                   transform=ax_insight.transAxes, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.5))
    ax_insight.axis('off')
    ax_insight.set_title('Key Insight', fontsize=12, fontweight='bold')

    # Save
    output_path = FIGURES_DIR / 'fractal_bifurcation_duality.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
