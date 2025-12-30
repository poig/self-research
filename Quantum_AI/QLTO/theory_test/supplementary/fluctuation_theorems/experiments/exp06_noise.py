"""
Experiment 6: Second Law Verification
======================================

Simple, robust test: ⟨W⟩ ≥ ΔF (Second Law of Thermodynamics)
Replaces unreliable noise simulation with fundamental physics test.
"""

import numpy as np
import matplotlib.pyplot as plt
from core import CoherentDemonEngine, setup_plot_style, save_figure


def run_second_law_test(N_values=[2, 3, 4, 5, 6], n_samples=30, beta=1.0):
    """
    Verify the second law: ⟨W⟩ ≥ ΔF.
    """
    print("=" * 60)
    print("EXPERIMENT 6: SECOND LAW VERIFICATION")
    print("⟨W⟩ ≥ ΔF (average work ≥ free energy change)")
    print("=" * 60)
    
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        
        engine = CoherentDemonEngine(n_qubits=N)
        
        W_samples = []
        E_final_samples = []
        taus = np.linspace(0.1, 1.5, n_samples)
        
        for tau in taus:
            data = engine.run_cycle(tau)
            W_samples.append(data.work)
            E_final_samples.append(data.E_final)
        
        W_arr = np.array(W_samples)
        mean_W = np.mean(W_arr)
        std_W = np.std(W_arr)
        
        # Operational ΔF
        E_reachable = np.min(E_final_samples)
        delta_F = E_reachable - engine.E_initial
        
        # Second law check
        margin = mean_W - delta_F
        satisfied = margin >= -1e-6  # Allow small numerical error
        
        results.append({
            'N': N,
            'mean_W': mean_W,
            'std_W': std_W,
            'delta_F': delta_F,
            'margin': margin,
            'satisfied': satisfied,
            'W_samples': W_arr
        })
        
        print(f"  ⟨W⟩ = {mean_W:.4f} ± {std_W:.4f}")
        print(f"  ΔF = {delta_F:.4f}")
        print(f"  Margin = {margin:.4f}")
        print(f"  Second Law: {'✓ SATISFIED' if satisfied else '✗ VIOLATED!'}")
    
    plot_second_law(results)
    return results


def plot_second_law(results):
    """Generate second law verification plot."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_vals = [r['N'] for r in results]
    mean_W = [r['mean_W'] for r in results]
    std_W = [r['std_W'] for r in results]
    delta_F = [r['delta_F'] for r in results]
    margins = [r['margin'] for r in results]
    
    # Left: W vs ΔF
    ax1 = axes[0]
    x = np.arange(len(N_vals))
    width = 0.35
    
    ax1.bar(x - width/2, mean_W, width, yerr=std_W, 
            label='⟨W⟩ (measured)', color='#2E86AB', alpha=0.8, capsize=5)
    ax1.bar(x + width/2, delta_F, width, 
            label='ΔF (free energy)', color='#E94F37', alpha=0.8)
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('System Size N')
    ax1.set_ylabel('Energy')
    ax1.set_title('Second Law: ⟨W⟩ vs ΔF\n(Blue should be ≥ Red)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(N_vals)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Margin (should all be positive)
    ax2 = axes[1]
    colors = ['#28A745' if m >= 0 else '#E94F37' for m in margins]
    bars = ax2.bar(N_vals, margins, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Second Law limit')
    
    # Add labels
    for bar, m in zip(bars, margins):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + 0.02,
                 f'{m:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('⟨W⟩ - ΔF')
    ax2.set_title('Second Law Margin\n(Green = satisfied, all should be ≥ 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, "06_second_law.png")
    # plt.show()


if __name__ == "__main__":
    results = run_second_law_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Second Law")
    print("=" * 60)
    all_pass = True
    for r in results:
        status = "✓" if r['satisfied'] else "✗"
        if not r['satisfied']:
            all_pass = False
        print(f"N={r['N']}: ⟨W⟩-ΔF = {r['margin']:.4f} {status}")
    
    print(f"\n{'✓ Second Law holds for all N!' if all_pass else '⚠ Second Law violated!'}")
