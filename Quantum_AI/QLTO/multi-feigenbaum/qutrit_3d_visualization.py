"""
3D Qutrit Bifurcation Surface Visualization

Creates a 3D bifurcation diagram where:
- X-axis: bifurcation parameter r
- Y-axis: probability coordinate x1
- Z-axis: probability coordinate x2

This shows how the 2D attractor evolves as r changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def coupled_qutrit_map(x: np.ndarray, r: float, coupling: float = 0.15) -> np.ndarray:
    """Coupled qutrit bifurcation map."""
    x1, x2 = x
    new_x1 = r * np.sin(np.pi * (x1 + coupling * x2)) ** 2
    new_x2 = r * np.sin(np.pi * (x2 + coupling * x1)) ** 2
    return np.array([new_x1, new_x2])


def generate_3d_bifurcation_data(r_min=0.5, r_max=1.0, n_r=150, 
                                   n_iterations=100, n_discard=200,
                                   coupling=0.15):
    """Generate 3D bifurcation data for qutrit."""
    r_values = np.linspace(r_min, r_max, n_r)
    
    all_r = []
    all_x1 = []
    all_x2 = []
    
    for r in r_values:
        x = np.array([0.3, 0.7])
        
        # Discard transient
        for _ in range(n_discard):
            x = coupled_qutrit_map(x, r, coupling)
        
        # Record attractor
        for _ in range(n_iterations):
            x = coupled_qutrit_map(x, r, coupling)
            all_r.append(r)
            all_x1.append(x[0])
            all_x2.append(x[1])
    
    return np.array(all_r), np.array(all_x1), np.array(all_x2)


def plot_3d_bifurcation_surface():
    """Create 3D bifurcation surface visualization."""
    
    print("Generating 3D qutrit bifurcation data...")
    r_data, x1_data, x2_data = generate_3d_bifurcation_data(
        r_min=0.5, r_max=1.0, n_r=200, n_iterations=80, coupling=0.15
    )
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(18, 12))
    
    # 3D scatter view 1 - Front angle
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = cm.plasma((r_data - 0.5) / 0.5)
    ax1.scatter(r_data, x1_data, x2_data, s=0.3, c=colors, alpha=0.4)
    ax1.set_xlabel('r (bifurcation param)')
    ax1.set_ylabel('x₁ (P₀ coordinate)')
    ax1.set_zlabel('x₂ (P₁ coordinate)')
    ax1.set_title('3D Qutrit Bifurcation\n(Front View)')
    ax1.view_init(elev=20, azim=45)
    
    # 3D scatter view 2 - Side angle
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(r_data, x1_data, x2_data, s=0.3, c=colors, alpha=0.4)
    ax2.set_xlabel('r')
    ax2.set_ylabel('x₁')
    ax2.set_zlabel('x₂')
    ax2.set_title('3D Qutrit Bifurcation\n(Side View)')
    ax2.view_init(elev=10, azim=120)
    
    # 3D scatter view 3 - Top-down
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.scatter(r_data, x1_data, x2_data, s=0.3, c=colors, alpha=0.4)
    ax3.set_xlabel('r')
    ax3.set_ylabel('x₁')
    ax3.set_zlabel('x₂')
    ax3.set_title('3D Qutrit Bifurcation\n(Top View)')
    ax3.view_init(elev=80, azim=0)
    
    # 2D projection: r vs x1
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(r_data, x1_data, s=0.1, c='blue', alpha=0.2)
    ax4.set_xlabel('r')
    ax4.set_ylabel('x₁')
    ax4.set_title('Projection: r vs x₁')
    ax4.grid(True, alpha=0.3)
    
    # 2D projection: r vs x2
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(r_data, x2_data, s=0.1, c='red', alpha=0.2)
    ax5.set_xlabel('r')
    ax5.set_ylabel('x₂')
    ax5.set_title('Projection: r vs x₂')
    ax5.grid(True, alpha=0.3)
    
    # 2D projection: x1 vs x2 (attractor shape at different r)
    ax6 = fig.add_subplot(2, 3, 6)
    sc = ax6.scatter(x1_data, x2_data, s=0.3, c=r_data, cmap='plasma', alpha=0.3)
    ax6.set_xlabel('x₁')
    ax6.set_ylabel('x₂')
    ax6.set_title('Attractor Shape\n(color = r value)')
    plt.colorbar(sc, ax=ax6, label='r')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.suptitle('QUTRIT (d=3) BIFURCATION: 3D STRUCTURE\n' +
                 'Coupled map: x₁\' = r·sin²(π(x₁ + 0.15·x₂)), x₂\' = r·sin²(π(x₂ + 0.15·x₁))',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qutrit_3d_bifurcation.png', dpi=200)
    plt.close()
    print("Saved: qutrit_3d_bifurcation.png")


def plot_bifurcation_slices():
    """Show 2D slices at different r values."""
    
    r_values = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, r in enumerate(r_values):
        ax = axes.flat[idx]
        
        x = np.array([0.3, 0.7])
        x1_list, x2_list = [], []
        
        # Warmup
        for _ in range(500):
            x = coupled_qutrit_map(x, r, coupling=0.15)
        
        # Record
        for _ in range(2000):
            x = coupled_qutrit_map(x, r, coupling=0.15)
            x1_list.append(x[0])
            x2_list.append(x[1])
        
        ax.scatter(x1_list, x2_list, s=0.5, c='purple', alpha=0.3)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'r = {r}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('QUTRIT ATTRACTOR EVOLUTION\nSlices at Different r Values', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qutrit_attractor_slices.png', dpi=150)
    plt.close()
    print("Saved: qutrit_attractor_slices.png")


def plot_qubit_vs_qutrit_3d():
    """Compare qubit (2D ribbon) vs qutrit (3D volume)."""
    
    fig = plt.figure(figsize=(16, 6))
    
    # Qubit: Create "fake" 3D by adding small noise to z
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    r_qubit = np.linspace(0.5, 1.0, 200)
    r_list, x_list = [], []
    
    for r in r_qubit:
        x = 0.5
        for _ in range(200):
            x = r * np.sin(np.pi * x) ** 2
        for _ in range(100):
            x = r * np.sin(np.pi * x) ** 2
            r_list.append(r)
            x_list.append(x)
    
    r_arr = np.array(r_list)
    x_arr = np.array(x_list)
    z_arr = np.zeros_like(x_arr)  # Qubit is 1D - flat in z
    
    ax1.scatter(r_arr, x_arr, z_arr, s=0.3, c='blue', alpha=0.5)
    ax1.set_xlabel('r')
    ax1.set_ylabel('P(|1⟩)')
    ax1.set_zlabel('(none)')
    ax1.set_title('QUBIT (d=2)\n1D Bifurcation = Ribbon in 3D')
    ax1.view_init(elev=20, azim=45)
    
    # Qutrit: True 3D structure
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    r_data, x1_data, x2_data = generate_3d_bifurcation_data(
        r_min=0.5, r_max=1.0, n_r=150, n_iterations=80, coupling=0.15
    )
    
    colors = cm.plasma((r_data - 0.5) / 0.5)
    ax2.scatter(r_data, x1_data, x2_data, s=0.3, c=colors, alpha=0.4)
    ax2.set_xlabel('r')
    ax2.set_ylabel('x₁')
    ax2.set_zlabel('x₂')
    ax2.set_title('QUTRIT (d=3)\n2D Bifurcation = Volume in 3D')
    ax2.view_init(elev=20, azim=45)
    
    plt.suptitle('QUBIT vs QUTRIT: DIMENSIONAL DIFFERENCE\n' +
                 'Qubit = 1D curve, Qutrit = 2D surface → Different chaos structure?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qubit_vs_qutrit_3d.png', dpi=200)
    plt.close()
    print("Saved: qubit_vs_qutrit_3d.png")


if __name__ == "__main__":
    print("=" * 60)
    print("3D QUTRIT BIFURCATION VISUALIZATION")
    print("=" * 60)
    
    plot_3d_bifurcation_surface()
    plot_bifurcation_slices()
    plot_qubit_vs_qutrit_3d()
    
    print("\n" + "=" * 60)
    print("Complete! Generated:")
    print("  1. qutrit_3d_bifurcation.png - Full 3D structure")
    print("  2. qutrit_attractor_slices.png - 2D slices at each r")
    print("  3. qubit_vs_qutrit_3d.png - Dimensional comparison")
    print("=" * 60)
