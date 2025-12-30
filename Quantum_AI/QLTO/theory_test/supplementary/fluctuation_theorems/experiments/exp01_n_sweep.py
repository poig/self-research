"""
Experiment 1: Efficiency vs System Size (N-Sweep)
==================================================

Measures how efficiency η(N) changes with system size.
Key finding: η peaks at N≈4, then collapses (R² degrades at transition).
"""

import numpy as np
import matplotlib.pyplot as plt
from core import CoherentDemonEngine, setup_plot_style, save_figure


def run_n_sweep(N_range=(2, 6), n_tau=25, kick_strength=0.2):
    """
    Sweep system size N and measure efficiency.
    """
    print("=" * 60)
    print("EXPERIMENT 1: EFFICIENCY vs SYSTEM SIZE")
    print("=" * 60)
    
    N_values = list(range(N_range[0], N_range[1] + 1))
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        engine = CoherentDemonEngine(n_qubits=N)
        eff = engine.measure_efficiency(n_tau=n_tau, kick_strength=kick_strength)
        results.append(eff)
        
        print(f"  η = {eff.eta:.4f} ± {eff.eta_error:.4f}")
        print(f"  R² = {eff.r_squared:.4f}")
        
        if eff.r_squared < 0.7:
            print(f"  ⚠ Low R² indicates transition region!")
    
    # Plot
    plot_n_sweep(results)
    
    return results


def plot_n_sweep(results):
    """Generate efficiency vs N plot with transition annotation."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    N_vals = [r.N for r in results]
    eta_vals = [r.eta for r in results]
    eta_errs = [r.eta_error for r in results]
    r2_vals = [r.r_squared for r in results]
    
    # Color by R² quality
    colors = ['#2E86AB' if r > 0.7 else '#E94F37' for r in r2_vals]
    
    # Plot with error bars
    for i, (N, eta, err, r2, color) in enumerate(zip(N_vals, eta_vals, eta_errs, r2_vals, colors)):
        ax.errorbar(N, eta, yerr=err, fmt='o', markersize=12, 
                    color=color, capsize=5, linewidth=2)
        # Add R² annotation
        ax.annotate(f'R²={r2:.2f}', (N, eta), 
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=color)
    
    # Connect with line
    ax.plot(N_vals, eta_vals, '--', color='gray', alpha=0.5, linewidth=1)
    
    # Highlight transition
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Find peak
    peak_idx = np.argmax(eta_vals)
    peak_N = N_vals[peak_idx]
    ax.axvline(peak_N, color='green', linestyle=':', linewidth=2, 
               label=f'Peak at N={peak_N}')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Good fit (R² > 0.7)'),
        Patch(facecolor='#E94F37', label='Transition region (R² < 0.7)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel('System Size N', fontsize=14)
    ax.set_ylabel('Efficiency η = dW/dI', fontsize=14)
    ax.set_title('Efficiency Collapse: η peaks then degrades\n(Red = transition where W ∝ I breaks down)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "01_n_sweep.png")
    # plt.show()


if __name__ == "__main__":
    results = run_n_sweep(N_range=(2, 8))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✓ Good" if r.r_squared > 0.7 else "⚠ Transition"
        print(f"N={r.N}: η={r.eta:.4f}, R²={r.r_squared:.3f} {status}")
