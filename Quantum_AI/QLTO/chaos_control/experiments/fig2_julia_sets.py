"""
Figure 2: Quantum Julia Sets - Structure Behind VQA Dynamics

LOGICALLY CONSISTENT: Both fractal and bifurcation use H-Rz-H!
- Bifurcation: H → Rz(πr) → H → Measure → P(|1⟩) = sin²(πr/2)
- Fractal:    H → Rz(πr) → H → Statevector → c = z0/z1 = i·cot(πr/2)

SAME CIRCUIT for both! Just different output extraction.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from core import FIGURES_DIR


def julia_set(c, xmin=-2, xmax=2, ymin=-2, ymax=2, 
              width=500, height=500, max_iter=100):
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


def get_c_from_hrz_h(r):
    """
    Get Julia parameter c from H-Rz-H circuit (SAME as bifurcation!).
    
    Circuit: |0⟩ → H → Rz(φ) → H → Statevector
    where φ = πr
    
    Math:
    - After H-Rz-H: |ψ⟩ = cos(φ/2)|0⟩ - i·sin(φ/2)|1⟩
    - z0 = cos(φ/2), z1 = -i·sin(φ/2)
    - c = z0/z1 = i·cot(φ/2)
    
    This gives c on the IMAGINARY axis with |c| varying!
    """
    phi = np.pi * r
    
    if not QISKIT_AVAILABLE:
        z0 = np.cos(phi / 2)
        z1 = -1j * np.sin(phi / 2)
    else:
        qc = QuantumCircuit(1)
        qc.h(0)       # Hadamard
        qc.rz(phi, 0) # Rz rotation  
        qc.h(0)       # Final H - SAME circuit as bifurcation!
        
        sv = Statevector(qc)
        z0, z1 = sv.data[0], sv.data[1]
    
    # Avoid division by zero when sin(φ/2) = 0
    if np.abs(z1) < 1e-10:
        c = 1e10j
    else:
        c = z0 / z1
    
    return c, z0, z1


def get_bifurcation_p1(r):
    """P(|1⟩) from same H-Rz-H circuit = sin²(πr/2)"""
    phi = np.pi * r
    return np.sin(phi / 2) ** 2


def main():
    """Generate figures using SAME H-Rz-H circuit for both fractal and bifurcation."""
    print("=" * 60)
    print("LOGICALLY CONSISTENT: H → Rz(πr) → H for BOTH!")
    print("=" * 60)
    print(f"Qiskit available: {QISKIT_AVAILABLE}")
    
    r_values = [0.55, 0.68, 0.72, 0.85]
    labels = ['Period-1', 'Period-2', 'Period-4', 'Chaos']
    colors = ['blue', 'green', 'orange', 'red']
    
    # Print the connection
    print("\nCircuit: |0⟩ → H → Rz(πr) → H")
    print("  Fractal:     c = z0/z1 = i·cot(πr/2)")
    print("  Bifurcation: P(|1⟩) = sin²(πr/2)")
    print("\nValues:")
    
    for r in r_values:
        c, z0, z1 = get_c_from_hrz_h(r)
        p1 = get_bifurcation_p1(r)
        cot_val = 1 / np.tan(np.pi * r / 2)
        print(f"  r={r}: c = {c:.3f} = i·{cot_val:.3f}, P(|1⟩) = {p1:.3f}")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    # Top row: Julia sets from H-Rz-H
    for ax, r, label, color in zip(axes[0], r_values, labels, colors):
        c, z0, z1 = get_c_from_hrz_h(r)
        
        julia = julia_set(c, max_iter=100, width=500, height=500)
        
        ax.imshow(julia, cmap='magma', extent=[-2, 2, -2, 2])
        ax.set_title(f'{label}\nc = {c:.2f}', fontsize=11, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # Bottom row: Bifurcation
    print("\nComputing bifurcation diagrams...")
    r_range = np.linspace(0.5, 1.0, 300)
    
    for ax, r_fixed, label, color in zip(axes[1], r_values, labels, colors):
        # Plot bifurcation
        for r in r_range:
            x = 0.1
            for _ in range(100):
                x = r * np.sin(np.pi * x)**2
            for _ in range(30):
                x = r * np.sin(np.pi * x)**2
                ax.plot(r, x, 'k.', markersize=0.3, alpha=0.4)
        
        ax.axvline(r_fixed, color=color, linestyle='-', linewidth=3, alpha=0.8)
        
        x = 0.5
        for _ in range(100):
            x = r_fixed * np.sin(np.pi * x)**2
        for _ in range(30):
            x = r_fixed * np.sin(np.pi * x)**2
            ax.plot(r_fixed, x, 'o', color=color, markersize=6)
        
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 0.8)
        ax.set_xlabel('r', fontsize=10)
        ax.set_ylabel('x*', fontsize=10)
        p1 = get_bifurcation_p1(r_fixed)
        ax.set_title(f'{label}: P(|1⟩)={p1:.2f}', fontsize=10, color=color, fontweight='bold')
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    fig.suptitle('SAME H-Rz-H Circuit: Fractal c=i·cot(πr/2) vs Bifurcation P(|1⟩)=sin²(πr/2)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = FIGURES_DIR / 'fig2_julia_sets_quantum.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    
    # Simple paper figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, r, label, color in zip(axes, r_values, labels, colors):
        c, _, _ = get_c_from_hrz_h(r)
        julia = julia_set(c, max_iter=100, width=500, height=500)
        
        ax.imshow(julia, cmap='magma', extent=[-2, 2, -2, 2])
        ax.set_title(f'{label}\nr = {r}', fontsize=12, color=color, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Julia Sets from H-Rz-H: c = i·cot(πr/2)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig2_julia_sets.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("KEY CONNECTION:")
    print("  SAME circuit H-Rz(πr)-H produces:")
    print("  • Fractal parameter:  c = i·cot(πr/2)  [imaginary axis]")
    print("  • Bifurcation prob:   P = sin²(πr/2)   [real probability]")
    print("  Both linked by:  |c|² = cot²(πr/2) = (1-P)/P")
    print("="*60)


if __name__ == "__main__":
    main()
