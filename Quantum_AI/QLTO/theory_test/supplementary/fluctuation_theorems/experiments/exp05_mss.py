"""
Experiment 5: Scrambling Rate Analysis
======================================

Shows operator spreading rate and compares to efficiency.
Better visualization: shows λ directly, not just saturation %.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp
from core import (CoherentDemonEngine, build_random_hamiltonian, 
                  setup_plot_style, save_figure)


def compute_operator_spreading(H: SparsePauliOp, n_qubits: int, 
                                times: np.ndarray) -> np.ndarray:
    """
    Compute operator spreading via commutator norm.
    ||[W(t), V]||² measures how operators spread.
    """
    H_mat = H.to_matrix()
    dim = 2**n_qubits
    
    # Local operators
    W_label = ["I"] * n_qubits
    W_label[0] = "Z"
    V_label = ["I"] * n_qubits
    V_label[min(1, n_qubits - 1)] = "Z"
    
    W = SparsePauliOp.from_list([("".join(W_label[::-1]), 1.0)]).to_matrix()
    V = SparsePauliOp.from_list([("".join(V_label[::-1]), 1.0)]).to_matrix()
    
    spreading = []
    
    for t in times:
        U_t = expm(-1j * H_mat * t)
        U_t_dag = expm(1j * H_mat * t)
        W_t = U_t_dag @ W @ U_t
        
        commutator = W_t @ V - V @ W_t
        norm_sq = np.real(np.trace(commutator @ commutator.conj().T)) / dim
        spreading.append(norm_sq)
    
    return np.array(spreading)


def fit_scrambling_rate(times: np.ndarray, spreading: np.ndarray) -> float:
    """Fit λ from early-time exponential growth."""
    early_mask = times < np.median(times)
    t_early = times[early_mask]
    s_early = np.clip(spreading[early_mask], 1e-20, None)
    
    try:
        slope, _, _, _, _ = linregress(t_early, np.log(s_early))
        return max(0, slope / 2)
    except:
        return 0.0


def run_scrambling_test(N_values=[3, 4, 5], n_times=20):
    """
    Measure scrambling rate and compare to efficiency.
    """
    print("=" * 60)
    print("EXPERIMENT 5: SCRAMBLING RATE")
    print("λ = operator spreading rate")
    print("=" * 60)
    
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        
        engine = CoherentDemonEngine(n_qubits=N)
        eff = engine.measure_efficiency(n_tau=15)
        
        times = np.linspace(0.01, 2.0, n_times)
        spreading = compute_operator_spreading(engine.H, N, times)
        lyapunov = fit_scrambling_rate(times, spreading)
        
        results.append({
            'N': N,
            'times': times,
            'spreading': spreading,
            'lyapunov': lyapunov,
            'eta': eff.eta,
            'r_squared': eff.r_squared
        })
        
        print(f"  η = {eff.eta:.4f}")
        print(f"  λ = {lyapunov:.4f}")
        print(f"  λ/η ratio = {lyapunov/eff.eta:.1f}" if eff.eta > 1e-6 else "  λ/η ratio = ∞")
    
    plot_scrambling(results)
    return results


def plot_scrambling(results):
    """Generate scrambling analysis plot."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    
    # Left: Operator spreading curves
    ax1 = axes[0]
    for i, r in enumerate(results):
        ax1.semilogy(r['times'], r['spreading'] + 1e-12, 'o-', 
                     color=colors[i], markersize=5, linewidth=2,
                     label=f"N={r['N']}: λ={r['lyapunov']:.2f}")
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Operator Spreading ||[W(t), V]||²')
    ax1.set_title('Operator Spreading (log scale)\nSlope = scrambling rate λ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: λ vs η comparison
    ax2 = axes[1]
    N_vals = [r['N'] for r in results]
    lambdas = [r['lyapunov'] for r in results]
    etas = [r['eta'] for r in results]
    
    x = np.arange(len(N_vals))
    width = 0.35
    
    ax2.bar(x - width/2, lambdas, width, label='Scrambling rate λ', 
            color='#E94F37', alpha=0.8)
    ax2.bar(x + width/2, etas, width, label='Efficiency η', 
            color='#2E86AB', alpha=0.8)
    
    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('Rate')
    ax2.set_title('Scrambling vs Efficiency\n(λ >> η = bottleneck)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(N_vals)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, (lam, eta) in enumerate(zip(lambdas, etas)):
        ratio = lam / eta if eta > 1e-6 else float('inf')
        ax2.text(i, max(lam, eta) + 0.3, f'λ/η={ratio:.0f}x', 
                 ha='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, "05_scrambling.png")
    # plt.show()


if __name__ == "__main__":
    results = run_scrambling_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Scrambling Analysis")
    print("=" * 60)
    for r in results:
        ratio = r['lyapunov'] / r['eta'] if r['eta'] > 1e-6 else float('inf')
        print(f"N={r['N']}: λ={r['lyapunov']:.2f}, η={r['eta']:.4f}, ratio={ratio:.0f}x")
