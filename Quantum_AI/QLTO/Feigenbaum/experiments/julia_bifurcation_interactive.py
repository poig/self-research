#!/usr/bin/env python3
"""
Interactive 3D Julia-Bifurcation Visualization using Plotly

Creates a smooth, interactive 3D surface that can be rotated in browser.
Uses isosurface rendering for a true 3D shape appearance.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def get_c_from_r(r):
    """
    Map sin² map parameter r to Julia set parameter c.
    Derived from H-Rz-H circuit: c = i·cot(πr/2)
    """
    phi = np.pi * r
    if np.abs(np.sin(phi / 2)) < 1e-10:
        return 1e10j
    return 1j * np.cos(phi / 2) / np.sin(phi / 2)


def compute_julia_volume(n_r=50, resolution=80, max_iter=40):
    """
    Compute a 3D volume where each r-slice is a Julia set escape time.
    Returns coordinates and values for isosurface rendering.
    """
    r_values = np.linspace(0.52, 0.88, n_r)
    x_coords = np.linspace(-1.5, 1.5, resolution)
    y_coords = np.linspace(-1.5, 1.5, resolution)
    
    # 3D volume grid
    R, X, Y = np.meshgrid(r_values, x_coords, y_coords, indexing='ij')
    volume = np.zeros((n_r, resolution, resolution))
    
    print(f"Computing {n_r} Julia slices at {resolution}x{resolution} resolution...")
    
    for i, r in enumerate(r_values):
        c = get_c_from_r(r)
        
        Z = X[i] + 1j * Y[i]
        M = np.zeros(Z.shape)
        
        for it in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            M[mask] = it
        
        volume[i] = M / max_iter
        
        if i % 10 == 0:
            print(f"  Slice {i+1}/{n_r} done...")
    
    return R, X, Y, volume, r_values, x_coords, y_coords


def create_interactive_isosurface(n_r=40, resolution=60):
    """
    Create interactive 3D isosurface visualization.
    The isosurface shows the Julia set boundary as a smooth 3D shape.
    """
    print("=" * 60)
    print("Creating Interactive 3D Julia-Bifurcation Surface")
    print("=" * 60)
    
    R, X, Y, volume, r_vals, x_vals, y_vals = compute_julia_volume(n_r, resolution)
    
    print("Generating isosurface...")
    
    # Create figure with isosurface
    fig = go.Figure(data=go.Isosurface(
        x=R.flatten(),
        y=X.flatten(),
        z=Y.flatten(),
        value=volume.flatten(),
        isomin=0.3,
        isomax=0.7,
        surface_count=3,  # Multiple surfaces at different escape time thresholds
        colorscale='Magma',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6,
        showscale=True,
        colorbar=dict(title='Escape Time'),
    ))
    
    fig.update_layout(
        title=dict(
            text='3D Julia-Bifurcation Structure<br><sup>Interactive: Click and drag to rotate</sup>',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='r (bifurcation parameter)',
            yaxis_title='Re(z)',
            zaxis_title='Im(z)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1000,
        height=800,
    )
    
    output_path = FIGURES_DIR / "julia_bifurcation_3d_interactive.html"
    fig.write_html(str(output_path))
    print(f"\n✓ Saved interactive HTML: {output_path}")
    
    return fig


def create_interactive_surface_mesh(n_r=60, resolution=100):
    """
    Create a surface mesh visualization for each Julia set slice.
    This creates multiple semi-transparent surfaces stacked together.
    """
    print("\nCreating Surface Mesh Visualization...")
    
    r_values = np.linspace(0.52, 0.88, n_r)
    
    fig = go.Figure()
    
    for i, r in enumerate(r_values):
        c = get_c_from_r(r)
        
        x = np.linspace(-1.5, 1.5, resolution)
        y = np.linspace(-1.5, 1.5, resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        M = np.zeros(Z.shape)
        
        for it in range(40):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            M[mask] = it
        
        M = M / 40
        
        # Create constant-r surface
        R = np.full_like(X, r)
        
        # Add surface at this r value
        fig.add_trace(go.Surface(
            x=R, y=X, z=Y,
            surfacecolor=M,
            colorscale='Magma',
            opacity=0.7,
            showscale=(i == 0),  # Only show colorbar for first
            colorbar=dict(title='Escape Time') if i == 0 else None,
            name=f'r={r:.2f}'
        ))
        
        if i % 10 == 0:
            print(f"  Surface {i+1}/{n_r} added...")
    
    fig.update_layout(
        title=dict(
            text='3D Julia Sets Stacked Along Bifurcation Parameter<br><sup>Smooth surfaces - click and drag to rotate</sup>',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='r (bifurcation)',
            yaxis_title='Re(z)',
            zaxis_title='Im(z)',
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.8)),
            aspectmode='cube'
        ),
        width=1200,
        height=900,
    )
    
    output_path = FIGURES_DIR / "julia_bifurcation_surfaces_interactive.html"
    fig.write_html(str(output_path))
    print(f"\n✓ Saved interactive surfaces: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Create isosurface (true 3D shape)
    create_interactive_isosurface(n_r=40, resolution=50)
    
    # Create stacked surfaces (smoother appearance)
    create_interactive_surface_mesh(n_r=30, resolution=80)
    
    print("\n" + "=" * 60)
    print("Interactive visualizations complete!")
    print("Open the .html files in a browser to interact.")
    print("=" * 60)
