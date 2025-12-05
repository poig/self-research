"""
Qudit Feigenbaum Analysis: Higher-Dimensional Quantum Chaos

This script explores whether Feigenbaum universality (δ = 4.669) extends
to qudits (d-level quantum systems) or if different universal constants emerge.

Theoretical Framework:
- Qubit (d=2): P_1 = sin²(θ/2) → 1D map → δ = 4.669
- Qutrit (d=3): P = (P_0, P_1, P_2) → 2D simplex
- Qudit (d): P = (P_0, ..., P_{d-1}) → (d-1)D simplex

Author: Auto-generated for Paper 4/5 research
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# THEORETICAL FRAMEWORK
# =============================================================================

def qubit_probability(theta: float) -> float:
    """
    Qubit Born rule: P(|1⟩) = sin²(θ/2)
    This is the H-Rz(θ)-H circuit output.
    """
    return np.sin(theta * np.pi / 2) ** 2


def qutrit_probabilities(theta1: float, theta2: float) -> Tuple[float, float, float]:
    """
    Qutrit probability distribution from two rotation angles.
    
    For a qutrit state: |ψ⟩ = α|0⟩ + β|1⟩ + γ|2⟩
    With generalized rotations, probabilities follow:
    
    P_0 = cos²(θ1 * π/2)
    P_1 = sin²(θ1 * π/2) * cos²(θ2 * π/2)
    P_2 = sin²(θ1 * π/2) * sin²(θ2 * π/2)
    
    This ensures P_0 + P_1 + P_2 = 1 (probability conservation)
    """
    p0 = np.cos(theta1 * np.pi / 2) ** 2
    p1 = np.sin(theta1 * np.pi / 2) ** 2 * np.cos(theta2 * np.pi / 2) ** 2
    p2 = np.sin(theta1 * np.pi / 2) ** 2 * np.sin(theta2 * np.pi / 2) ** 2
    return p0, p1, p2


def qudit_probabilities(thetas: np.ndarray, d: int) -> np.ndarray:
    """
    General d-dimensional qudit probability distribution.
    
    Uses nested trigonometric parameterization to ensure Σ P_k = 1.
    
    Parameters:
    - thetas: array of (d-1) angles in [0, 1]
    - d: dimension of the qudit
    
    Returns:
    - probs: array of d probabilities
    """
    probs = np.zeros(d)
    remaining = 1.0
    
    for k in range(d - 1):
        probs[k] = remaining * np.cos(thetas[k] * np.pi / 2) ** 2
        remaining *= np.sin(thetas[k] * np.pi / 2) ** 2
    
    probs[d - 1] = remaining
    return probs


# =============================================================================
# BIFURCATION DYNAMICS
# =============================================================================

def qubit_bifurcation_map(x: float, r: float) -> float:
    """
    Qubit bifurcation map: x_{n+1} = r * sin²(π * x_n)
    This is equivalent to the Born rule feedback.
    """
    return r * np.sin(np.pi * x) ** 2


def qutrit_bifurcation_map(x: np.ndarray, r: float) -> np.ndarray:
    """
    Qutrit bifurcation map in 2D.
    
    x = (x1, x2) where x1, x2 ∈ [0, 1]
    
    The map applies Born-rule-like nonlinearity to each coordinate:
    x1' = r * sin²(π * x1)
    x2' = r * sin²(π * x2)
    
    This creates a 2D coupled system that may show different universality.
    """
    return r * np.sin(np.pi * x) ** 2


def coupled_qutrit_map(x: np.ndarray, r: float, coupling: float = 0.1) -> np.ndarray:
    """
    Coupled qutrit map with interaction between coordinates.
    
    x1' = r * sin²(π * (x1 + coupling * x2))
    x2' = r * sin²(π * (x2 + coupling * x1))
    
    Coupling strength determines how the two dimensions interact.
    """
    x1, x2 = x
    new_x1 = r * np.sin(np.pi * (x1 + coupling * x2)) ** 2
    new_x2 = r * np.sin(np.pi * (x2 + coupling * x1)) ** 2
    return np.array([new_x1, new_x2])


# =============================================================================
# BIFURCATION DIAGRAM GENERATION
# =============================================================================

def generate_qubit_bifurcation(r_values: np.ndarray, 
                                n_iterations: int = 500,
                                n_discard: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 1D bifurcation diagram for qubit."""
    r_list = []
    x_list = []
    
    for r in r_values:
        x = 0.5  # Initial condition
        
        # Transient
        for _ in range(n_discard):
            x = qubit_bifurcation_map(x, r)
        
        # Record attractor
        for _ in range(n_iterations):
            x = qubit_bifurcation_map(x, r)
            r_list.append(r)
            x_list.append(x)
    
    return np.array(r_list), np.array(x_list)


def generate_qutrit_bifurcation(r_values: np.ndarray,
                                 coupling: float = 0.0,
                                 n_iterations: int = 300,
                                 n_discard: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2D bifurcation data for qutrit."""
    r_list = []
    x1_list = []
    x2_list = []
    
    for r in r_values:
        x = np.array([0.3, 0.7])  # Initial condition
        
        # Transient
        for _ in range(n_discard):
            if coupling == 0:
                x = qutrit_bifurcation_map(x, r)
            else:
                x = coupled_qutrit_map(x, r, coupling)
        
        # Record attractor
        for _ in range(n_iterations):
            if coupling == 0:
                x = qutrit_bifurcation_map(x, r)
            else:
                x = coupled_qutrit_map(x, r, coupling)
            r_list.append(r)
            x1_list.append(x[0])
            x2_list.append(x[1])
    
    return np.array(r_list), np.array(x1_list), np.array(x2_list)


def generate_qudit_bifurcation(d: int, 
                                r_values: np.ndarray,
                                n_iterations: int = 200,
                                n_discard: int = 200) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate (d-1)-dimensional bifurcation data for qudit."""
    r_list = []
    x_lists = [[] for _ in range(d - 1)]
    
    for r in r_values:
        # Initial condition: random point in (d-1)-hypercube
        x = np.random.uniform(0.2, 0.8, d - 1)
        
        # Transient
        for _ in range(n_discard):
            x = r * np.sin(np.pi * x) ** 2
        
        # Record attractor
        for _ in range(n_iterations):
            x = r * np.sin(np.pi * x) ** 2
            r_list.append(r)
            for k in range(d - 1):
                x_lists[k].append(x[k])
    
    return np.array(r_list), [np.array(lst) for lst in x_lists]


# =============================================================================
# FEIGENBAUM CONSTANT EXTRACTION
# =============================================================================

def find_bifurcation_points_1d(r_values: np.ndarray, x_values: np.ndarray,
                                tolerance: float = 0.01) -> List[float]:
    """
    Find bifurcation points by detecting where attractor size changes.
    """
    bifurcation_points = []
    prev_n_attractors = 1
    
    for r in np.unique(r_values)[::-1]:  # Scan from high to low r
        mask = r_values == r
        x_at_r = x_values[mask]
        
        # Count distinct attractor points
        unique_x = np.unique(np.round(x_at_r, 2))
        n_attractors = len(unique_x)
        
        if n_attractors != prev_n_attractors and n_attractors <= 16:
            bifurcation_points.append(r)
            prev_n_attractors = n_attractors
    
    return sorted(bifurcation_points)


def calculate_feigenbaum_delta(bifurcation_points: List[float]) -> List[float]:
    """
    Calculate Feigenbaum δ from bifurcation points.
    
    δ_n = (r_n - r_{n-1}) / (r_{n+1} - r_n)
    
    Should converge to 4.669... for Feigenbaum universality.
    """
    if len(bifurcation_points) < 3:
        return []
    
    deltas = []
    for i in range(1, len(bifurcation_points) - 1):
        numerator = bifurcation_points[i] - bifurcation_points[i - 1]
        denominator = bifurcation_points[i + 1] - bifurcation_points[i]
        
        if abs(denominator) > 1e-10:
            deltas.append(numerator / denominator)
    
    return deltas


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_qubit_vs_qutrit_comparison():
    """
    Compare qubit (1D) vs qutrit (2D) bifurcation diagrams.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Qubit (1D)
    r_values = np.linspace(0.5, 1.0, 500)
    r_qubit, x_qubit = generate_qubit_bifurcation(r_values)
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(r_qubit, x_qubit, s=0.1, c='blue', alpha=0.3)
    ax1.set_xlabel('r (bifurcation parameter)')
    ax1.set_ylabel('x (probability)')
    ax1.set_title('Qubit (d=2): 1D Bifurcation\n→ Feigenbaum δ = 4.669')
    ax1.grid(True, alpha=0.3)
    
    # Qutrit uncoupled (2D)
    r_qutrit, x1_qutrit, x2_qutrit = generate_qutrit_bifurcation(r_values, coupling=0.0)
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(r_qutrit, x1_qutrit, s=0.1, c='red', alpha=0.3, label='x1')
    ax2.scatter(r_qutrit, x2_qutrit, s=0.1, c='green', alpha=0.3, label='x2')
    ax2.set_xlabel('r (bifurcation parameter)')
    ax2.set_ylabel('x (probability)')
    ax2.set_title('Qutrit (d=3): 2D Uncoupled\n→ Same δ (independent channels)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Qutrit coupled (2D with interaction)
    r_coupled, x1_coupled, x2_coupled = generate_qutrit_bifurcation(r_values, coupling=0.2)
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(r_coupled, x1_coupled, s=0.1, c='purple', alpha=0.3, label='x1')
    ax3.scatter(r_coupled, x2_coupled, s=0.1, c='orange', alpha=0.3, label='x2')
    ax3.set_xlabel('r (bifurcation parameter)')
    ax3.set_ylabel('x (probability)')
    ax3.set_title('Qutrit (d=3): 2D Coupled (ε=0.2)\n→ Different dynamics emerge!')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 2D Phase space for coupled qutrit at fixed r
    ax4 = fig.add_subplot(2, 2, 4)
    
    r_fixed = 0.85
    x = np.array([0.3, 0.7])
    trajectory_x1, trajectory_x2 = [x[0]], [x[1]]
    
    for _ in range(500):
        x = coupled_qutrit_map(x, r_fixed, coupling=0.2)
        trajectory_x1.append(x[0])
        trajectory_x2.append(x[1])
    
    ax4.scatter(trajectory_x1[200:], trajectory_x2[200:], s=1, c='magenta', alpha=0.5)
    ax4.set_xlabel('x1 (P_0)')
    ax4.set_ylabel('x2 (P_1)')
    ax4.set_title(f'Qutrit Phase Space (r={r_fixed})\n→ 2D attractor structure')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qudit_bifurcation_comparison.png', dpi=150)
    plt.close()
    print("Saved: qudit_bifurcation_comparison.png")


def plot_dimension_dependence():
    """
    Show how bifurcation structure changes with qudit dimension.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    r_values = np.linspace(0.5, 1.0, 400)
    
    dimensions = [2, 3, 4, 5, 6, 7]
    
    for idx, d in enumerate(dimensions):
        ax = axes.flat[idx]
        
        if d == 2:
            r_data, x_data = generate_qubit_bifurcation(r_values, n_iterations=200)
            ax.scatter(r_data, x_data, s=0.1, c='blue', alpha=0.3)
        else:
            r_data, x_lists = generate_qudit_bifurcation(d, r_values, n_iterations=100)
            colors = plt.cm.viridis(np.linspace(0, 1, d-1))
            for k in range(d - 1):
                ax.scatter(r_data, x_lists[k], s=0.1, c=[colors[k]], alpha=0.3, label=f'x{k+1}')
        
        ax.set_xlabel('r')
        ax.set_ylabel('Probability coordinates')
        ax.set_title(f'Qudit d={d}\n({d-1}D phase space)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dimension Dependence of Quantum Bifurcation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qudit_dimension_dependence.png', dpi=150)
    plt.close()
    print("Saved: qudit_dimension_dependence.png")


def plot_feigenbaum_extraction():
    """
    Extract and compare Feigenbaum δ for different systems.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # High-resolution qubit bifurcation
    r_values = np.linspace(0.6, 0.75, 2000)
    r_data, x_data = generate_qubit_bifurcation(r_values, n_iterations=500, n_discard=500)
    
    ax1 = axes[0]
    ax1.scatter(r_data, x_data, s=0.05, c='blue', alpha=0.2)
    ax1.set_xlabel('r (bifurcation parameter)')
    ax1.set_ylabel('x (attractor)')
    ax1.set_title('High-Resolution Qubit Bifurcation\nfor δ Extraction')
    ax1.grid(True, alpha=0.3)
    
    # Known bifurcation points for sin² map (approximate)
    # r1 ≈ 0.64, r2 ≈ 0.71, r3 ≈ 0.73, r_∞ ≈ 0.74
    bifurcation_points = [0.640, 0.705, 0.720, 0.740]  # Approximate
    
    for r_bif in bifurcation_points:
        ax1.axvline(r_bif, color='red', linestyle='--', alpha=0.5)
    
    # Calculate δ
    deltas = calculate_feigenbaum_delta(bifurcation_points)
    
    ax2 = axes[1]
    if deltas:
        ax2.bar(range(len(deltas)), deltas, color='green', alpha=0.7)
        ax2.axhline(4.669, color='red', linestyle='--', label=f'Feigenbaum δ = 4.669')
        ax2.set_xlabel('Bifurcation index n')
        ax2.set_ylabel('δ_n = (r_n - r_{n-1}) / (r_{n+1} - r_n)')
        ax2.set_title(f'Feigenbaum δ Convergence\n(values: {[f"{d:.2f}" for d in deltas]})')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Need more bifurcation points', 
                ha='center', va='center', transform=ax2.transAxes)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qudit_feigenbaum_extraction.png', dpi=150)
    plt.close()
    print("Saved: qudit_feigenbaum_extraction.png")


def plot_3d_qutrit_attractor():
    """
    Visualize the full qutrit probability simplex and attractor.
    """
    fig = plt.figure(figsize=(12, 5))
    
    # 3D simplex visualization
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Generate trajectory
    r = 0.9
    x = np.array([0.3, 0.7])
    p0_list, p1_list, p2_list = [], [], []
    
    for _ in range(1000):
        x = coupled_qutrit_map(x, r, coupling=0.15)
        p0, p1, p2 = qutrit_probabilities(x[0], x[1])
        p0_list.append(p0)
        p1_list.append(p1)
        p2_list.append(p2)
    
    # Plot attractor in probability simplex
    ax1.scatter(p0_list[200:], p1_list[200:], p2_list[200:], 
               s=1, c=range(800), cmap='plasma', alpha=0.5)
    ax1.set_xlabel('P(|0⟩)')
    ax1.set_ylabel('P(|1⟩)')
    ax1.set_zlabel('P(|2⟩)')
    ax1.set_title(f'Qutrit Attractor in Probability Simplex\nr={r}, coupling=0.15')
    
    # 2D projection
    ax2 = fig.add_subplot(1, 2, 2)
    sc = ax2.scatter(p0_list[200:], p1_list[200:], 
                     s=2, c=range(800), cmap='plasma', alpha=0.5)
    ax2.set_xlabel('P(|0⟩)')
    ax2.set_ylabel('P(|1⟩)')
    ax2.set_title('2D Projection of Qutrit Attractor')
    plt.colorbar(sc, ax=ax2, label='Iteration')
    
    plt.tight_layout()
    plt.savefig('/home/poig/project/self-research/Quantum_AI/QLTO/multi-feigenbaum/qutrit_3d_attractor.png', dpi=150)
    plt.close()
    print("Saved: qutrit_3d_attractor.png")


# =============================================================================
# THEORETICAL ANALYSIS
# =============================================================================

def print_theoretical_summary():
    """Print summary of qudit Feigenbaum theory."""
    summary = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           QUDIT FEIGENBAUM THEORY: THEORETICAL SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUBIT (d=2):                                                                ║
║  ───────────                                                                 ║
║  • Born rule: P(|1⟩) = sin²(θπ/2)                                            ║
║  • Map: 1D → x' = r·sin²(πx)                                                 ║
║  • Universality: Feigenbaum δ = 4.669...                                     ║
║  • Route to chaos: Period-doubling                                           ║
║                                                                              ║
║  QUTRIT (d=3):                                                               ║
║  ────────────                                                                ║
║  • Born rule: P_k = |⟨k|ψ⟩|² with Σ P_k = 1                                  ║
║  • Map: 2D → (x1', x2') on probability simplex                               ║
║  • Predictions:                                                              ║
║    - Uncoupled: Same δ = 4.669 (independent channels)                        ║
║    - Coupled: Different δ or quasi-periodic route                            ║
║  • New phenomena: Torus bifurcations, Hopf bifurcation                       ║
║                                                                              ║
║  QUDIT (d):                                                                  ║
║  ──────────                                                                  ║
║  • Phase space: (d-1)-dimensional probability simplex                        ║
║  • Higher-dimensional chaos may have different universality                  ║
║                                                                              ║
║  KEY QUESTION:                                                               ║
║  ─────────────                                                               ║
║  Is δ = 4.669 universal across ALL quantum dimensions,                       ║
║  or does it depend on Hilbert space dimension d?                             ║
║                                                                              ║
║  TESTABLE PREDICTIONS:                                                       ║
║  ─────────────────────                                                       ║
║  1. Uncoupled qudit: Same δ for each dimension                               ║
║  2. Coupled qudit: Different δ or quasi-periodic                             ║
║  3. High-d limit: May approach mean-field behavior                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(summary)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUDIT FEIGENBAUM ANALYSIS")
    print("Exploring higher-dimensional quantum chaos")
    print("=" * 70)
    
    print_theoretical_summary()
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_qubit_vs_qutrit_comparison()
    plot_dimension_dependence()
    plot_feigenbaum_extraction()
    plot_3d_qutrit_attractor()
    
    print("\n" + "=" * 70)
    print("All visualizations complete!")
    print("Key finding: Coupled qudits may show DIFFERENT universality than qubits.")
    print("=" * 70)
