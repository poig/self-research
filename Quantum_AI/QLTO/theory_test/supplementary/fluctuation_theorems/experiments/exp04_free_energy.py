"""
Experiment 4: Constitutive Law Breakdown
========================================

Shows HOW the constitutive law W = η·I breaks down at the transition.
Replaces unclear free energy analysis with direct W vs I scatter.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from core import CoherentDemonEngine, setup_plot_style, save_figure


def run_breakdown_test(N_values=[3, 4, 5, 6], n_samples=30):
    """
    Show the breakdown of W ∝ I across system sizes.
    """
    print("=" * 60)
    print("EXPERIMENT 4: CONSTITUTIVE LAW BREAKDOWN")
    print("W = η·I should hold, but breaks at transition")
    print("=" * 60)
    
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        
        engine = CoherentDemonEngine(n_qubits=N)
        eff = engine.measure_efficiency(n_tau=n_samples)
        
        results.append({
            'N': N,
            'W': eff.W_values,
            'I': eff.I_values,
            'eta': eff.eta,
            'r_squared': eff.r_squared
        })
        
        print(f"  η = {eff.eta:.4f}")
        print(f"  R² = {eff.r_squared:.4f}")
        
        if eff.r_squared < 0.7:
            print(f"  ⚠ LAW BREAKS DOWN!")
    
    # Plot
    plot_breakdown(results)
    
    return results


def plot_breakdown(results):
    """Generate W vs I scatter plots showing law breakdown."""
    setup_plot_style()
    
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (ax, r) in enumerate(zip(axes, results)):
        W = r['W']
        I = r['I']
        eta = r['eta']
        r2 = r['r_squared']
        N = r['N']
        
        # Scatter
        color = '#2E86AB' if r2 > 0.7 else '#E94F37'
        ax.scatter(I, W, alpha=0.6, s=60, color=color, edgecolors='white')
        
        # Fit line
        if len(I) > 2:
            slope, intercept, _, _, _ = linregress(I, W)
            I_fit = np.linspace(min(I), max(I), 100)
            ax.plot(I_fit, slope * I_fit + intercept, '--', 
                    color='black', linewidth=2, label=f'η={eta:.3f}')
        
        ax.set_xlabel('Mutual Information I(S:A)', fontsize=11)
        ax.set_ylabel('Work Extracted W', fontsize=11)
        ax.set_title(f'N={N}: R²={r2:.2f}', fontsize=12,
                     color='green' if r2 > 0.7 else 'red')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    
    fig.suptitle('Constitutive Law: W = η·I\n(Green title = law holds, Red = breakdown)', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    save_figure(fig, "04_breakdown.png")
    # plt.show()


if __name__ == "__main__":
    results = run_breakdown_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Constitutive Law")
    print("=" * 60)
    for r in results:
        status = "✓ HOLDS" if r['r_squared'] > 0.7 else "✗ BREAKS"
        print(f"N={r['N']}: R²={r['r_squared']:.3f} {status}")
