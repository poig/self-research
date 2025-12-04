#!/usr/bin/env python3
"""
3D Bifurcation-Julia Combined Visualization

Creates the famous 3D structure where:
- Julia sets are stacked along the r (bifurcation) axis
- Side view reveals the bifurcation diagram
- Shows the deep connection between fractal topology and dynamical chaos

Based on the sin² map: x_{n+1} = r·sin²(πx)
with c = i·cot(πr/2) from the H-Rz-H quantum circuit.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path
import os

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def sin2_map(x, r):
    """The sin² map: x_{n+1} = r·sin²(πx)"""
    return r * np.sin(np.pi * x) ** 2


def get_c_from_r(r):
    """
    Map sin² map parameter r to Julia set parameter c.
    Derived from H-Rz-H circuit: c = i·cot(πr/2)
    """
    phi = np.pi * r
    if np.abs(np.sin(phi / 2)) < 1e-10:
        return 1e10j
    return 1j * np.cos(phi / 2) / np.sin(phi / 2)


def julia_set_boundary(c, resolution=100, max_iter=50):
    """
    Compute Julia set boundary points for 3D visualization.
    Returns points that are on/near the Julia set boundary.
    """
    x = np.linspace(-1.5, 1.5, resolution)
    y = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    M = np.zeros(Z.shape)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + c
        M[mask] = i
    
    # Normalize escape time
    M = M / max_iter
    
    return X, Y, M


def compute_bifurcation_attractor(r, n_iter=200, n_sample=50):
    """Compute attractor points for bifurcation diagram."""
    x = 0.5
    for _ in range(n_iter):
        x = sin2_map(x, r)
    
    attractor = []
    for _ in range(n_sample):
        x = sin2_map(x, r)
        if 0 < x < 1:
            attractor.append(x)
    
    return np.array(attractor)


def create_3d_bifurcation_julia(n_r_slices=15, julia_res=80, save=True):
    """
    Create 3D visualization combining bifurcation and Julia sets.
    
    The structure shows:
    - Julia sets stacked perpendicular to r-axis
    - Bifurcation attractor points along the r-axis
    - Connection between fractal topology and period-doubling
    """
    print("Generating 3D Bifurcation-Julia Structure...")
    
    fig = plt.figure(figsize=(18, 8))
    
    # =========================================
    # Panel 1: Stacked Julia Sets (Main 3D View)
    # =========================================
    ax1 = fig.add_subplot(131, projection='3d')
    
    r_values = np.linspace(0.55, 0.85, n_r_slices)
    
    print(f"  Computing {n_r_slices} Julia sets...")
    
    for i, r in enumerate(r_values):
        c = get_c_from_r(r)
        X, Y, M = julia_set_boundary(c, resolution=julia_res, max_iter=40)
        
        # Only plot boundary points (intermediate escape times)
        boundary_mask = (M > 0.1) & (M < 0.9)
        
        # Scatter the boundary points at this r-slice
        x_pts = X[boundary_mask]
        y_pts = Y[boundary_mask]
        r_pts = np.full_like(x_pts, r)
        
        # Color by escape time
        colors = M[boundary_mask]
        
        ax1.scatter(r_pts, x_pts, y_pts, c=colors, cmap='magma', 
                   s=1.5, alpha=0.8)
        
        print(f"    r={r:.3f}: c={c:.3f}, {len(x_pts)} boundary points")
    
    ax1.set_xlabel('r (bifurcation)', fontsize=10)
    ax1.set_ylabel('Re(z)', fontsize=10)
    ax1.set_zlabel('Im(z)', fontsize=10)
    ax1.set_title('(A) Julia Sets Stacked Along r\n(Quantum Measurement Parameter)', fontsize=11, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # =========================================
    # Panel 2: Side View = Bifurcation Diagram
    # =========================================
    ax2 = fig.add_subplot(132)
    
    print("  Computing bifurcation diagram...")
    
    r_bif = np.linspace(0.5, 0.9, 300)
    for r in r_bif:
        attractor = compute_bifurcation_attractor(r)
        if len(attractor) > 0:
            ax2.scatter([r] * len(attractor), attractor, s=0.2, c='darkblue', alpha=0.5)
    
    # Mark bifurcation points
    bif_points = [0.6278, 0.7066, 0.7259, 0.7302]
    for r_b in bif_points:
        ax2.axvline(r_b, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('r', fontsize=10)
    ax2.set_ylabel('x* (attractor)', fontsize=10)
    ax2.set_title('(B) Bifurcation Diagram\n(Side View of 3D Structure)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0.5, 0.9)
    ax2.set_ylim(0, 0.85)
    ax2.grid(alpha=0.3)
    
    # =========================================
    # Panel 3: Julia Set Gallery at Key r Values
    # =========================================
    ax3 = fig.add_subplot(133)
    
    # Create 2x2 inset of Julia sets
    key_r_values = [0.55, 0.68, 0.74, 0.85]
    labels = ['Period-1\nr=0.55', 'Period-2\nr=0.68', 'Period-4\nr=0.74', 'Chaos\nr=0.85']
    
    for idx, (r, label) in enumerate(zip(key_r_values, labels)):
        c = get_c_from_r(r)
        
        # Create inset axes
        left = 0.68 + (idx % 2) * 0.16
        bottom = 0.55 - (idx // 2) * 0.25 
        width = 0.14
        height = 0.22
        
        ax_inset = fig.add_axes([left, bottom, width, height])
        
        X, Y, M = julia_set_boundary(c, resolution=150, max_iter=60)
        ax_inset.imshow(M, cmap='magma', extent=[-1.5, 1.5, -1.5, 1.5], origin='lower')
        ax_inset.set_title(label, fontsize=8)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
    
    ax3.axis('off')
    ax3.set_title('(C) Julia Sets at Key Bifurcations', fontsize=11, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / "bifurcation_julia_3d_combined.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n  ✓ Saved: {output_path}")
    
    plt.show()
    print("Done!")


def create_3d_julia_tower(n_slices=25, resolution=100, save=True):
    """
    Create the famous 3D Julia set "tower" visualization.
    
    This shows Julia sets stacked along the imaginary c-axis,
    revealing the Mandelbrot set in the side view.
    """
    print("\nGenerating 3D Julia Tower...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # For sin² map, c = i·cot(πr/2) is on imaginary axis
    # So we stack along Im(c)
    
    r_values = np.linspace(0.52, 0.88, n_slices)
    
    all_r = []
    all_re = []
    all_im = []
    all_colors = []
    
    for r in r_values:
        c = get_c_from_r(r)
        
        # Julia set computation
        x = np.linspace(-1.5, 1.5, resolution)
        y = np.linspace(-1.5, 1.5, resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        M = np.zeros(Z.shape)
        
        for i in range(40):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            M[mask] = i
        
        M = M / 40  # Normalize
        
        # Keep only boundary points
        boundary = (M > 0.15) & (M < 0.85)
        
        x_pts = X[boundary].flatten()
        y_pts = Y[boundary].flatten()
        r_pts = np.full_like(x_pts, r)
        c_pts = M[boundary].flatten()
        
        all_r.extend(r_pts)
        all_re.extend(x_pts)
        all_im.extend(y_pts)
        all_colors.extend(c_pts)
    
    # Convert to arrays
    all_r = np.array(all_r)
    all_re = np.array(all_re)
    all_im = np.array(all_im)
    all_colors = np.array(all_colors)
    
    # Use ALL points for dense visualization (no subsampling)
    n_points = len(all_r)  # Use all points
    
    scatter = ax.scatter(all_r, all_re, all_im,
                        c=all_colors, cmap='hot', s=0.5, alpha=0.8)
    
    ax.set_xlabel('r (bifurcation parameter)', fontsize=11)
    ax.set_ylabel('Re(z)', fontsize=11)
    ax.set_zlabel('Im(z)', fontsize=11)
    ax.set_title('3D Julia Set Tower\nConnecting Quantum Measurement to Fractal Structure', 
                fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=0.5, pad=0.1)
    cbar.set_label('Escape time (normalized)', fontsize=10)
    
    ax.view_init(elev=15, azim=60)
    
    if save:
        output_path = FIGURES_DIR / "julia_tower_3d.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("3D Bifurcation-Julia Visualization (HIGH RESOLUTION)")
    print("=" * 60)
    
    # High resolution: 80 r-slices, 150x150 Julia grid
    create_3d_bifurcation_julia(n_r_slices=80, julia_res=150)
    # High resolution tower: 100 slices, 150x150 grid
    create_3d_julia_tower(n_slices=100, resolution=150)
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
