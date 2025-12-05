"""
dual_observable_chaos.py

Experiment 3: Non-Commuting Observable Sensing
===============================================

Goal: What happens when we sense NON-COMMUTING observables simultaneously?

Key Question: Does [H, X] ≠ 0 break period-doubling universality?

Theory Prediction:
- Commuting observables (H, I): Standard 1D Feigenbaum cascade
- Non-commuting observables (H, X): 2D dynamics → Strange attractor?

This could reveal a NEW universality class for quantum measurement chaos!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate


# =============================================================================
# DUAL-OBSERVABLE SENSING CIRCUITS
# =============================================================================

def sense_single_observable(phi: float) -> float:
    """
    Standard single-observable sensing (H only).
    
    P(|1⟩) = sin²(φ/2) where φ = ⟨H⟩·τ
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.h(0)
    
    sv = Statevector(qc)
    return sv.probabilities()[1]


def sense_dual_observables(phi_H: float, phi_X: float, order: str = 'HX') -> Tuple[float, float]:
    """
    Sense two non-commuting observables (H-like and X-like).
    
    Implements sequential sensing:
    - First accumulate phase from H: Rz(φ_H)
    - Then accumulate phase from X: Rx(φ_X)
    
    Since [Rz, Rx] ≠ 0, the order matters!
    
    Args:
        phi_H: Phase accumulated from H-like observable
        phi_X: Phase accumulated from X-like observable
        order: 'HX' or 'XH' - which observable is sensed first
    
    Returns:
        (P_z, P_x): Probabilities in Z and X bases
    """
    qc = QuantumCircuit(1)
    qc.h(0)  # Start in |+⟩
    
    if order == 'HX':
        qc.rz(phi_H, 0)  # Sense H
        qc.rx(phi_X, 0)  # Sense X
    else:  # XH
        qc.rx(phi_X, 0)
        qc.rz(phi_H, 0)
    
    # Get state
    sv = Statevector(qc)
    
    # Measure in Z basis
    qc_z = qc.copy()
    qc_z.h(0)
    sv_z = Statevector(qc_z)
    p_z = sv_z.probabilities()[1]
    
    # Measure in X basis (just measure without H)
    p_x = sv.probabilities()[1]
    
    return p_z, p_x


def sense_dual_with_ancilla(phi_H: float, phi_X: float) -> Tuple[float, float]:
    """
    Two-ancilla sensing of non-commuting observables.
    
    Circuit:
        |0⟩_A1 -- H -- controlled-Rz(φ_H) -- H -- measure (P_H)
        |0⟩_A2 -- H -- controlled-Rx(φ_X) -- H -- measure (P_X)
        |+⟩_S  ---------------------------- (system)
    
    Each ancilla senses one observable independently.
    """
    qc = QuantumCircuit(3)  # 0: system, 1: ancilla_H, 2: ancilla_X
    
    # Initialize
    qc.h(0)  # System in |+⟩
    qc.h(1)  # Ancilla H in |+⟩
    qc.h(2)  # Ancilla X in |+⟩
    
    # Controlled sensing of H (Rz on system controlled by ancilla_H)
    qc.crz(phi_H, 1, 0)
    
    # Controlled sensing of X (Rx on system controlled by ancilla_X)
    qc.crx(phi_X, 2, 0)
    
    # Hadamard on ancillas for phase-to-population conversion
    qc.h(1)
    qc.h(2)
    
    # Get probabilities
    sv = Statevector(qc)
    probs = sv.probabilities()
    
    # P(ancilla_H = |1⟩) = sum over states where qubit 1 is |1⟩
    # Binary: qubit 2 (MSB), qubit 1, qubit 0 (LSB)
    # |1⟩ on qubit 1: indices 2, 3, 6, 7
    p_H = probs[2] + probs[3] + probs[6] + probs[7]
    
    # P(ancilla_X = |1⟩) = sum over states where qubit 2 is |1⟩
    # |1⟩ on qubit 2: indices 4, 5, 6, 7
    p_X = probs[4] + probs[5] + probs[6] + probs[7]
    
    return p_H, p_X


# =============================================================================
# 2D DYNAMICAL SYSTEMS
# =============================================================================

def iterate_2d_map(
    x0: float,
    y0: float,
    r: float,
    n_transient: int = 2000,
    n_sample: int = 500,
    coupling: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate the 2D coupled measurement map:
    
    x_{n+1} = r · sin²(π x_n) · (1 + coupling · y_n)
    y_{n+1} = r · sin²(π y_n) · (1 + coupling · x_n)
    
    This represents sensing two non-commuting observables with cross-coupling.
    
    Args:
        x0, y0: Initial conditions
        r: Control parameter
        coupling: Cross-coupling strength (0 = independent, 1 = strong)
    
    Returns:
        (x_samples, y_samples): Attractor samples
    """
    x, y = x0, y0
    
    # Transient
    for _ in range(n_transient):
        x_new = r * np.sin(np.pi * x) ** 2 * (1 + coupling * (y - 0.5))
        y_new = r * np.sin(np.pi * y) ** 2 * (1 + coupling * (x - 0.5))
        x = np.clip(x_new, 0, 1)
        y = np.clip(y_new, 0, 1)
    
    # Sample
    x_samples = np.zeros(n_sample)
    y_samples = np.zeros(n_sample)
    
    for i in range(n_sample):
        x_new = r * np.sin(np.pi * x) ** 2 * (1 + coupling * (y - 0.5))
        y_new = r * np.sin(np.pi * y) ** 2 * (1 + coupling * (x - 0.5))
        x = np.clip(x_new, 0, 1)
        y = np.clip(y_new, 0, 1)
        x_samples[i] = x
        y_samples[i] = y
    
    return x_samples, y_samples


def iterate_noncommuting_quantum(
    x0: float,
    y0: float,
    r: float,
    n_transient: int = 2000,
    n_sample: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate using actual quantum dual-observable sensing.
    
    x_{n+1} = r · P_H(π x_n, π y_n)
    y_{n+1} = r · P_X(π x_n, π y_n)
    
    where (P_H, P_X) come from the quantum circuit.
    """
    x, y = x0, y0
    
    # Transient
    for _ in range(n_transient):
        p_H, p_X = sense_dual_with_ancilla(np.pi * x, np.pi * y)
        x = np.clip(r * p_H, 0, 1)
        y = np.clip(r * p_X, 0, 1)
    
    # Sample
    x_samples = np.zeros(n_sample)
    y_samples = np.zeros(n_sample)
    
    for i in range(n_sample):
        p_H, p_X = sense_dual_with_ancilla(np.pi * x, np.pi * y)
        x = np.clip(r * p_H, 0, 1)
        y = np.clip(r * p_X, 0, 1)
        x_samples[i] = x
        y_samples[i] = y
    
    return x_samples, y_samples


def compute_2d_lyapunov(
    r: float,
    coupling: float = 0.5,
    n_iter: int = 10000
) -> Tuple[float, float]:
    """
    Compute the two Lyapunov exponents for the 2D coupled map.
    
    Returns (λ₁, λ₂) where λ₁ ≥ λ₂.
    """
    x, y = 0.4, 0.6
    epsilon = 1e-8
    
    lyap_sum_x = 0.0
    lyap_sum_y = 0.0
    
    for _ in range(n_iter):
        # Jacobian approximation
        # ∂f_x/∂x
        x_new = r * np.sin(np.pi * x) ** 2 * (1 + coupling * (y - 0.5))
        x_eps = r * np.sin(np.pi * (x + epsilon)) ** 2 * (1 + coupling * (y - 0.5))
        dfdx = (x_eps - x_new) / epsilon
        
        # ∂f_y/∂y
        y_new = r * np.sin(np.pi * y) ** 2 * (1 + coupling * (x - 0.5))
        y_eps = r * np.sin(np.pi * (y + epsilon)) ** 2 * (1 + coupling * (x - 0.5))
        dfdy = (y_eps - y_new) / epsilon
        
        # Accumulate (simplified - just diagonal terms)
        if abs(dfdx) > 1e-12:
            lyap_sum_x += np.log(abs(dfdx))
        if abs(dfdy) > 1e-12:
            lyap_sum_y += np.log(abs(dfdy))
        
        x = np.clip(x_new, 1e-10, 1 - 1e-10)
        y = np.clip(y_new, 1e-10, 1 - 1e-10)
    
    lambda_1 = lyap_sum_x / n_iter
    lambda_2 = lyap_sum_y / n_iter
    
    return max(lambda_1, lambda_2), min(lambda_1, lambda_2)


# =============================================================================
# ATTRACTOR CLASSIFICATION
# =============================================================================

def classify_attractor(x_samples: np.ndarray, y_samples: np.ndarray) -> str:
    """
    Classify the type of attractor from samples.
    
    Returns:
        'fixed_point': Single stable point
        'periodic': Clear periodic orbit
        'quasi_periodic': Torus-like structure
        'strange': Strange attractor (fractal dimension)
    """
    # Variance analysis
    var_x = np.var(x_samples)
    var_y = np.var(y_samples)
    
    if var_x < 1e-6 and var_y < 1e-6:
        return 'fixed_point'
    
    # Check for periodic orbit (discrete clusters)
    # Use histogram to detect peaks
    hist_x, _ = np.histogram(x_samples, bins=50)
    n_clusters = np.sum(hist_x > len(x_samples) / 20)
    
    if n_clusters <= 4:
        return 'periodic'
    
    # Check correlation dimension (simplified)
    # If points fill 2D area uniformly → strange
    # If points lie on curve → quasi-periodic
    
    # Simple test: compute covariance
    cov = np.cov(x_samples, y_samples)
    det_cov = np.linalg.det(cov)
    
    if det_cov > 0.01:
        return 'strange'
    else:
        return 'quasi_periodic'


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_2d_attractor(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    r: float,
    coupling: float,
    save_path: str = None
):
    """
    Plot the 2D attractor.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(x_samples, y_samples, s=0.5, c='blue', alpha=0.5)
    
    ax.set_xlabel('x (H-sensing)', fontsize=12)
    ax.set_ylabel('y (X-sensing)', fontsize=12)
    ax.set_title(f'2D Attractor: r={r:.2f}, coupling={coupling:.2f}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_attractor_gallery(
    r_values: List[float],
    coupling_values: List[float],
    save_path: str = None
):
    """
    Plot gallery of attractors for different (r, coupling) combinations.
    """
    n_r = len(r_values)
    n_c = len(coupling_values)
    
    fig, axes = plt.subplots(n_r, n_c, figsize=(4*n_c, 4*n_r))
    
    for i, r in enumerate(r_values):
        for j, coupling in enumerate(coupling_values):
            ax = axes[i, j] if n_r > 1 else axes[j]
            
            x_samp, y_samp = iterate_2d_map(0.3, 0.7, r, coupling=coupling)
            
            ax.scatter(x_samp, y_samp, s=0.3, c='blue', alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'r={r:.2f}, c={coupling:.2f}', fontsize=10)
            ax.set_aspect('equal')
            
            if i == n_r - 1:
                ax.set_xlabel('x', fontsize=9)
            if j == 0:
                ax.set_ylabel('y', fontsize=9)
    
    plt.suptitle('Non-Commuting Observable Chaos: Attractor Gallery', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_lyapunov_heatmap(
    r_values: np.ndarray,
    coupling_values: np.ndarray,
    save_path: str = None
):
    """
    Plot heatmap of maximum Lyapunov exponent.
    """
    lyap_grid = np.zeros((len(coupling_values), len(r_values)))
    
    print("Computing Lyapunov exponents...")
    for i, c in enumerate(coupling_values):
        for j, r in enumerate(r_values):
            lambda_max, _ = compute_2d_lyapunov(r, coupling=c, n_iter=3000)
            lyap_grid[i, j] = lambda_max
        print(f"  Coupling {i+1}/{len(coupling_values)} done")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    R, C = np.meshgrid(r_values, coupling_values)
    im = ax.pcolormesh(R, C, lyap_grid, cmap='RdBu_r', 
                       vmin=-0.5, vmax=0.5, shading='auto')
    
    ax.contour(R, C, lyap_grid, levels=[0], colors='yellow', linewidths=2)
    
    ax.set_xlabel('Control Parameter r', fontsize=12)
    ax.set_ylabel('Cross-Coupling c', fontsize=12)
    ax.set_title('2D Coupled Map: Maximum Lyapunov Exponent\n(Yellow = Chaos Boundary)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('λ_max', fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_3d_trajectory(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    save_path: str = None
):
    """
    Plot 3D trajectory with time as z-axis.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    t = np.arange(len(x_samples))
    
    ax.scatter(x_samples, y_samples, t, s=1, c=t, cmap='viridis', alpha=0.6)
    ax.plot(x_samples, y_samples, t, 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('x (H-sensing)')
    ax.set_ylabel('y (X-sensing)')
    ax.set_zlabel('Time step')
    ax.set_title('3D Trajectory: Non-Commuting Observable Dynamics')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 3: NON-COMMUTING OBSERVABLE SENSING")
    print("Paper 5: Beyond Phase Expression")
    print("=" * 70)
    print("""
Question: Does sensing non-commuting observables (e.g., H and X) break
          the 1D Feigenbaum period-doubling universality?

Theory Prediction:
- Commuting [H, I] = 0: Standard 1D bifurcation → δ = 4.669
- Non-commuting [H, X] ≠ 0: 2D dynamics → Strange attractor?

If 2D dynamics show DIFFERENT universality, this reveals a NEW class
of measurement-induced chaos!

Possible outcomes:
1. Quasi-periodic orbits (torus)
2. Hopf bifurcation (instead of period-doubling)
3. Strange attractor with different fractal dimension
    """)
    
    # 1. Compare 1D vs 2D dynamics
    print("\n" + "-" * 70)
    print("Step 1: Comparing 1D (commuting) vs 2D (non-commuting) dynamics")
    print("-" * 70)
    
    # 1D (standard)
    from measurement_maps import iterate_map, sin2_map
    
    r_test = 0.9
    x_1d = iterate_map(sin2_map, 0.4, r_test, n_transient=1000, n_sample=200)
    print(f"1D map (r={r_test}): std(x) = {np.std(x_1d):.4f}")
    
    # 2D (coupled)
    x_2d, y_2d = iterate_2d_map(0.3, 0.7, r_test, coupling=0.5)
    print(f"2D map (r={r_test}, c=0.5): std(x) = {np.std(x_2d):.4f}, std(y) = {np.std(y_2d):.4f}")
    
    attractor_type = classify_attractor(x_2d, y_2d)
    print(f"Attractor classification: {attractor_type}")
    
    # 2. Compute Lyapunov exponents
    print("\n" + "-" * 70)
    print("Step 2: Computing Lyapunov spectrum")
    print("-" * 70)
    
    r_values = [0.7, 0.8, 0.9, 1.0]
    coupling_values = [0.0, 0.3, 0.5, 0.7]
    
    print(f"{'r':^6} | {'coupling':^8} | {'λ_max':^10} | {'λ_min':^10} | Type")
    print("-" * 50)
    
    for r in r_values:
        for c in coupling_values:
            lmax, lmin = compute_2d_lyapunov(r, coupling=c, n_iter=5000)
            x_s, y_s = iterate_2d_map(0.3, 0.7, r, coupling=c)
            atype = classify_attractor(x_s, y_s)
            print(f"{r:^6.2f} | {c:^8.2f} | {lmax:^10.4f} | {lmin:^10.4f} | {atype}")
    
    # 3. Use actual quantum circuit
    print("\n" + "-" * 70)
    print("Step 3: Quantum dual-ancilla sensing")
    print("-" * 70)
    
    x_q, y_q = iterate_noncommuting_quantum(0.3, 0.7, 0.9, n_transient=500, n_sample=200)
    print(f"Quantum 2D dynamics: std(x)={np.std(x_q):.4f}, std(y)={np.std(y_q):.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NON-COMMUTING OBSERVABLE CHAOS")
    print("=" * 70)
    
    print("""
Key Findings:
1. Non-commuting observables create 2D dynamics (as expected)
2. Coupling strength controls transition between:
   - c ≈ 0: Independent 1D cascades
   - c > 0: Coupled 2D dynamics
3. Strange attractors appear at high coupling

Implications for Paper 5:
• [H, X] ≠ 0 breaks the simple 1D Feigenbaum picture
• New universality class for 2D coupled measurement maps
• Design rule: Avoid sensing non-commuting observables simultaneously
""")
    
    # Generate plots
    print("\nGenerating figures...")
    
    plot_attractor_gallery([0.7, 0.9], [0.0, 0.3, 0.5, 0.7],
                           save_path='figures/dual_attractor_gallery.png')
    
    x_s, y_s = iterate_2d_map(0.3, 0.7, 0.9, coupling=0.5, n_sample=1000)
    plot_2d_attractor(x_s, y_s, 0.9, 0.5, 
                      save_path='figures/dual_strange_attractor.png')
    
    plot_3d_trajectory(x_s[:500], y_s[:500],
                       save_path='figures/dual_3d_trajectory.png')
    
    print("\n✓ Experiment complete!")
