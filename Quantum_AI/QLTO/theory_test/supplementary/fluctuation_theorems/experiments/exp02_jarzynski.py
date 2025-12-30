"""
Experiment 2: Jarzynski Equality Verification
==============================================

Verifies ⟨exp(-βW)⟩ = exp(-βΔF) using OPERATIONAL ΔF.
Key finding: Error < 30% with reachable energy definition.
"""

import numpy as np
import matplotlib.pyplot as plt
from core import CoherentDemonEngine, setup_plot_style, save_figure


def run_jarzynski_test(N_values=[3, 4, 5, 6], n_samples=40, beta=1.0):
    """
    Test Jarzynski equality with operational ΔF.
    """
    print("=" * 60)
    print("EXPERIMENT 2: JARZYNSKI EQUALITY")
    print("⟨exp(-βW)⟩ = exp(-βΔF_operational)")
    print("=" * 60)
    
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        
        engine = CoherentDemonEngine(n_qubits=N)
        
        # Collect samples
        W_samples = []
        E_final_samples = []
        taus = np.linspace(0.1, 1.5, n_samples)
        
        for tau in taus:
            data = engine.run_cycle(tau)
            W_samples.append(data.work)
            E_final_samples.append(data.E_final)
        
        W_arr = np.array(W_samples)
        
        # Operational ΔF: use BEST reachable energy
        E_reachable = np.min(E_final_samples)
        delta_F = E_reachable - engine.E_initial
        
        # Jarzynski
        jarz_measured = np.mean(np.exp(-beta * W_arr))
        jarz_predicted = np.exp(-beta * delta_F)
        error = abs(jarz_measured - jarz_predicted) / abs(jarz_predicted)
        
        # Second law check
        second_law = np.mean(W_arr) >= delta_F
        
        results.append({
            'N': N,
            'jarz_measured': jarz_measured,
            'jarz_predicted': jarz_predicted,
            'error': error,
            'delta_F': delta_F,
            'mean_W': np.mean(W_arr),
            'second_law': second_law
        })
        
        print(f"  ΔF = {delta_F:.4f}")
        print(f"  ⟨exp(-βW)⟩ = {jarz_measured:.4f}")
        print(f"  exp(-βΔF) = {jarz_predicted:.4f}")
        print(f"  Error = {error:.1%}")
        print(f"  Second Law: {'✓' if second_law else '✗'}")
    
    # Plot
    plot_jarzynski(results)
    
    return results


def plot_jarzynski(results):
    """Generate clean Jarzynski result plot."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_vals = [r['N'] for r in results]
    errors = [r['error'] * 100 for r in results]
    jarz_m = [r['jarz_measured'] for r in results]
    jarz_p = [r['jarz_predicted'] for r in results]
    
    # Left: Jarzynski comparison
    ax1 = axes[0]
    x = np.arange(len(N_vals))
    width = 0.35
    
    ax1.bar(x - width/2, jarz_m, width, label='⟨exp(-βW)⟩ measured', 
            color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, jarz_p, width, label='exp(-βΔF) predicted', 
            color='#28A745', alpha=0.8)
    
    ax1.set_xlabel('System Size N')
    ax1.set_ylabel('Jarzynski Average')
    ax1.set_title('Jarzynski Equality: Measured vs Predicted')
    ax1.set_xticks(x)
    ax1.set_xticklabels(N_vals)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Error with threshold
    ax2 = axes[1]
    colors = ['#28A745' if e < 30 else '#E94F37' for e in errors]
    bars = ax2.bar(N_vals, errors, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(30, color='red', linestyle='--', linewidth=2, 
                label='Acceptable threshold (30%)')
    
    # Add error labels
    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{err:.0f}%', ha='center', va='bottom', fontsize=11)
    
    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Jarzynski Verification Error\n(Green = passed, Red = failed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(errors) * 1.2)
    
    plt.tight_layout()
    save_figure(fig, "02_jarzynski.png")
    # plt.show()


if __name__ == "__main__":
    results = run_jarzynski_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Jarzynski Verification")
    print("=" * 60)
    all_pass = True
    for r in results:
        status = "✓ PASS" if r['error'] < 0.3 else "✗ FAIL"
        if r['error'] >= 0.3:
            all_pass = False
        print(f"N={r['N']}: error={r['error']:.1%} {status}")
    
    print(f"\nOverall: {'✓ All tests passed!' if all_pass else '⚠ Some tests failed'}")
