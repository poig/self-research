"""
visualize_ancilla.py

Real-time visualization of QLTO ancilla sensing dynamics.
Creates 3D Julia set fractals based on activation_rate from quantum measurements.

Usage:
    from visualize_ancilla import AncillaVisualizer
    
    viz = AncillaVisualizer()
    viz.record(epoch=0, activation_rate=0.45, energy=-2.3)
    viz.record(epoch=1, activation_rate=0.52, energy=-2.8)
    ...
    viz.save_3d_fractal("optimization_fractal.html")
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Optional imports (only needed for visualization)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class AncillaVisualizer:
    """
    Records QLTO sensing diagnostics and generates fractal visualizations.
    
    The activation_rate from ancilla measurements is mapped to Julia set
    parameters, creating a 3D structure that shows how the optimization
    landscape evolves during training.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving figures. Defaults to current dir.
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(exist_ok=True)
        
        # History storage
        self.history: List[Dict] = []
        
    def record(
        self, 
        epoch: int, 
        activation_rate: float, 
        energy: float,
        entropy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        layer_id: Optional[int] = None
    ):
        """
        Record a snapshot of optimization state.
        
        Args:
            epoch: Current optimization epoch
            activation_rate: P(ancilla=|1⟩) from sensing
            energy: Current energy expectation value
            entropy: Optional normalized entropy
            gradient_norm: Optional ||∇E||
            layer_id: Optional layer identifier for layer-wise training
        """
        self.history.append({
            'epoch': epoch,
            'activation_rate': activation_rate,
            'energy': energy,
            'entropy': entropy if entropy is not None else 0.0,
            'gradient_norm': gradient_norm if gradient_norm is not None else 0.0,
            'layer_id': layer_id
        })
    
    def get_history(self) -> List[Dict]:
        """Return recorded history."""
        return self.history
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
    
    @staticmethod
    def activation_to_c(activation_rate: float, energy: float = 0.0, 
                        energy_scale: float = 1.0) -> complex:
        """
        Map activation_rate and energy to Julia set parameter c.
        
        The mapping:
        - Real part: activation_rate scaled to [-0.5, 1.5]
        - Imaginary part: normalized energy contribution
        
        This creates real-valued c for most cases, giving beautiful
        Mandelbrot-like Julia sets instead of the abstract imaginary ones.
        
        Args:
            activation_rate: P(|1⟩) ∈ [0, 1]
            energy: Current energy value
            energy_scale: Scale factor for energy normalization
        
        Returns:
            Complex c parameter for Julia set
        """
        # Map activation [0,1] → c_real [-0.5, 1.5]
        # This covers the interesting Mandelbrot/Julia region
        c_real = activation_rate * 2.0 - 0.5
        
        # Energy contribution to imaginary part (normalized)
        c_imag = np.clip(energy / energy_scale, -1.0, 1.0) * 0.5
        
        return complex(c_real, c_imag)
    
    @staticmethod
    def compute_julia_set(c: complex, resolution: int = 100, 
                          max_iter: int = 50) -> np.ndarray:
        """
        Compute Julia set for parameter c.
        
        Returns escape time array normalized to [0, 1].
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
        
        return M / max_iter
    
    def generate_2d_summary(self, save: bool = True, filename: str = "ancilla_summary.png"):
        """
        Generate 2D summary plots of optimization history.
        
        Creates a 2x2 figure:
        - Activation rate over epochs
        - Energy over epochs
        - Entropy over epochs
        - Sample Julia sets at key epochs
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Cannot generate 2D plots.")
            return None
        
        if not self.history:
            print("Warning: No history to visualize.")
            return None
        
        epochs = [h['epoch'] for h in self.history]
        activations = [h['activation_rate'] for h in self.history]
        energies = [h['energy'] for h in self.history]
        entropies = [h['entropy'] for h in self.history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Activation Rate
        ax1 = axes[0, 0]
        ax1.plot(epochs, activations, 'b-o', markersize=4)
        ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Target 50%')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Activation Rate P(|1⟩)')
        ax1.set_title('Ancilla Sensing Quality')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: Energy
        ax2 = axes[0, 1]
        ax2.plot(epochs, energies, 'g-o', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Energy ⟨H⟩')
        ax2.set_title('Energy Convergence')
        ax2.grid(alpha=0.3)
        
        # Panel 3: Entropy
        ax3 = axes[1, 0]
        ax3.plot(epochs, entropies, 'm-o', markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Normalized Entropy')
        ax3.set_title('Solution Diversity')
        ax3.grid(alpha=0.3)
        
        # Panel 4: Sample Julia Sets
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('Julia Sets at Key Epochs')
        
        # Select 4 key epochs
        n = len(self.history)
        key_indices = [0, n//3, 2*n//3, n-1] if n >= 4 else list(range(n))
        
        # Create mini Julia set insets
        energy_scale = max(abs(min(energies)), abs(max(energies)), 1.0)
        
        for i, idx in enumerate(key_indices[:4]):
            h = self.history[idx]
            c = self.activation_to_c(h['activation_rate'], h['energy'], energy_scale)
            julia = self.compute_julia_set(c, resolution=80)
            
            # Position insets
            left = 0.52 + (i % 2) * 0.22
            bottom = 0.05 + (1 - i // 2) * 0.22
            
            ax_inset = fig.add_axes([left, bottom, 0.18, 0.18])
            ax_inset.imshow(julia, cmap='magma', extent=[-1.5, 1.5, -1.5, 1.5])
            ax_inset.set_title(f'Epoch {h["epoch"]}\na={h["activation_rate"]:.2f}', fontsize=8)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        
        return fig
    
    def generate_3d_fractal(self, save: bool = True, 
                            filename: str = "optimization_fractal.html",
                            resolution: int = 80):
        """
        Generate interactive 3D fractal showing optimization dynamics.
        
        X-axis: Epoch (optimization progress)
        Y-Z plane: Julia set boundary for that epoch's activation_rate
        
        Args:
            save: Whether to save to file
            filename: Output filename (HTML for Plotly)
            resolution: Julia set resolution
        
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            print("Warning: plotly not available. Cannot generate 3D visualization.")
            return None
        
        if not self.history:
            print("Warning: No history to visualize.")
            return None
        
        print("Generating 3D optimization fractal...")
        
        # Collect all boundary points
        all_epoch = []
        all_x = []
        all_y = []
        all_colors = []
        
        energy_scale = max(
            abs(min(h['energy'] for h in self.history)),
            abs(max(h['energy'] for h in self.history)),
            1.0
        )
        
        for h in self.history:
            epoch = h['epoch']
            c = self.activation_to_c(h['activation_rate'], h['energy'], energy_scale)
            
            julia = self.compute_julia_set(c, resolution=resolution)
            
            # Get coordinate grids
            x = np.linspace(-1.5, 1.5, resolution)
            y = np.linspace(-1.5, 1.5, resolution)
            X, Y = np.meshgrid(x, y)
            
            # Extract boundary points
            boundary = (julia > 0.15) & (julia < 0.85)
            
            if np.sum(boundary) > 0:
                all_epoch.extend(np.full(np.sum(boundary), epoch))
                all_x.extend(X[boundary])
                all_y.extend(Y[boundary])
                all_colors.extend(julia[boundary])
        
        if not all_epoch:
            print("Warning: No boundary points found.")
            return None
        
        print(f"  Total points: {len(all_epoch)}")
        
        # Create Plotly scatter
        fig = go.Figure(data=go.Scatter3d(
            x=all_epoch,
            y=all_x,
            z=all_y,
            mode='markers',
            marker=dict(
                size=1.5,
                color=all_colors,
                colorscale='Plasma',
                opacity=0.7,
                colorbar=dict(title='Escape Time')
            ),
        ))
        
        fig.update_layout(
            title=dict(
                text='QLTO Optimization Fractal<br><sup>Julia sets from activation_rate</sup>',
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title='Epoch',
                yaxis_title='Re(z)',
                zaxis_title='Im(z)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                bgcolor='#111111',
            ),
            paper_bgcolor='#1a1a1a',
            width=1000,
            height=800,
        )
        
        if save:
            output_path = self.output_dir / filename
            fig.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
        
        return fig


# Convenience function for quick visualization
def visualize_optimization(history: List[Dict], output_dir: str = "."):
    """
    Quick visualization from a list of history dictionaries.
    
    Args:
        history: List of dicts with keys: epoch, activation_rate, energy
        output_dir: Output directory
    """
    viz = AncillaVisualizer(output_dir)
    for h in history:
        viz.record(**h)
    
    viz.generate_2d_summary()
    viz.generate_3d_fractal()


if __name__ == "__main__":
    # Demo: Generate sample visualization
    print("Demo: Generating sample optimization trajectory...")
    
    viz = AncillaVisualizer(output_dir="./figures")
    
    # Simulate optimization trajectory
    np.random.seed(42)
    energy = -0.5
    
    for epoch in range(20):
        # Simulate improving activation and decreasing energy
        activation = 0.2 + 0.6 * (1 - np.exp(-epoch / 5)) + np.random.normal(0, 0.05)
        activation = np.clip(activation, 0, 1)
        
        energy = energy - 0.15 * np.exp(-epoch / 10) + np.random.normal(0, 0.05)
        entropy = 0.8 - 0.3 * (epoch / 20) + np.random.normal(0, 0.05)
        entropy = np.clip(entropy, 0, 1)
        
        viz.record(epoch=epoch, activation_rate=activation, energy=energy, entropy=entropy)
    
    # Generate visualizations
    viz.generate_2d_summary()
    viz.generate_3d_fractal()
    
    print("\nDone! Check ./figures/ for outputs.")
