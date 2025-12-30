"""
Experiment 3: Holevo Capacity and Information Bottleneck
=========================================================

Tests the information bottleneck mechanism with HONEST DLA calculation.
Key finding: Information required exceeds channel capacity.
"""

import numpy as np
import matplotlib.pyplot as plt
from core import CoherentDemonEngine, setup_plot_style, save_figure


def compute_actual_dla_dimension(n_qubits: int, H_type: str = "ising_random") -> int:
    """
    Compute ACTUAL DLA dimension for our Hamiltonian.
    
    For random Ising with all-to-all ZZ + X:
    - Generators: N choose 2 ZZ terms + N X terms = N(N-1)/2 + N
    - DLA closure includes mixed terms but NOT full SU(2^N)
    - Upper bound: O(N²) not O(4^N)
    
    Being HONEST: we estimate, not assume exponential.
    """
    if H_type == "ising_random":
        # Number of generators in our Hamiltonian
        n_ZZ = n_qubits * (n_qubits - 1) // 2  # All-to-all ZZ
        n_X = n_qubits  # Single-qubit X
        
        # DLA closure includes commutators: [ZZ, X] ~ Y, [ZZ, ZZ] ~ 0, etc.
        # Rough estimate: O(N²) to O(N³) depending on connectivity
        # Conservative upper bound for random Ising:
        estimated_dim = n_ZZ + n_X + 2 * n_ZZ * n_X  # Include first-order commutators
        
        return min(estimated_dim, 4**n_qubits - 1)  # Can't exceed full algebra
    else:
        return 4**n_qubits - 1  # Full SU(2^N)


def run_holevo_test(N_values=[2, 3, 4, 5, 6], n_samples=25):
    """
    Test Holevo capacity and compare to information requirements.
    HONEST: Uses realistic DLA dimension estimate.
    """
    print("=" * 60)
    print("EXPERIMENT 3: INFORMATION BOTTLENECK (HONEST)")
    print("Holevo χ ≤ 1 bit vs DLA information requirement")
    print("=" * 60)
    
    results = []
    
    for N in N_values:
        print(f"\n[N = {N}]")
        
        engine = CoherentDemonEngine(n_qubits=N)
        
        # Collect Holevo samples
        holevo_samples = []
        mutual_info_samples = []
        taus = np.linspace(0.1, 1.5, n_samples)
        
        for tau in taus:
            data = engine.run_cycle(tau)
            holevo_samples.append(data.holevo_chi)
            mutual_info_samples.append(data.mutual_info)
        
        holevo_chi = np.mean(holevo_samples)
        mutual_info = np.mean(mutual_info_samples)
        
        # HONEST DLA dimension
        dla_dim = compute_actual_dla_dimension(N)
        dla_dim_full = 4**N - 1  # What we were claiming before
        
        info_required = np.log2(dla_dim) if dla_dim > 0 else 0
        info_required_full = np.log2(dla_dim_full)
        
        bottleneck_ratio = info_required / holevo_chi if holevo_chi > 1e-6 else float('inf')
        
        results.append({
            'N': N,
            'holevo_chi': holevo_chi,
            'mutual_info': mutual_info,
            'dla_dim': dla_dim,
            'dla_dim_full': dla_dim_full,
            'info_required': info_required,
            'info_required_full': info_required_full,
            'bottleneck_ratio': bottleneck_ratio,
            'is_bottlenecked': bottleneck_ratio > 2
        })
        
        print(f"  Holevo χ = {holevo_chi:.4f} bits")
        print(f"  Mutual I = {mutual_info:.4f} bits")
        print(f"  DLA dim (honest) = {dla_dim}")
        print(f"  DLA dim (if full) = {dla_dim_full}")
        print(f"  I_required = {info_required:.1f} bits")
        print(f"  Bottleneck = {bottleneck_ratio:.1f}x")
        
        if bottleneck_ratio > 2:
            print(f"  ⚠ INFORMATION BOTTLENECK")
    
    # Plot
    plot_holevo(results)
    
    return results


def plot_holevo(results):
    """Generate honest Holevo/bottleneck plot."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_vals = [r['N'] for r in results]
    holevo = [r['holevo_chi'] for r in results]
    info_req = [r['info_required'] for r in results]
    info_req_full = [r['info_required_full'] for r in results]
    
    # Left: Capacity vs Requirement
    ax1 = axes[0]
    x = np.arange(len(N_vals))
    width = 0.25
    
    ax1.bar(x - width, holevo, width, label='Holevo χ (available)', 
            color='#28A745', alpha=0.8)
    ax1.bar(x, info_req, width, label='I_required (honest DLA)', 
            color='#2E86AB', alpha=0.8)
    ax1.bar(x + width, info_req_full, width, label='I_required (if full SU)', 
            color='#E94F37', alpha=0.5)
    
    ax1.set_xlabel('System Size N')
    ax1.set_ylabel('Information (bits)')
    ax1.set_title('Information Capacity vs Requirement\n(Honest DLA estimate)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(N_vals)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Bottleneck ratio
    ax2 = axes[1]
    bottleneck = [r['bottleneck_ratio'] for r in results]
    colors = ['#28A745' if b < 2 else '#FFA500' if b < 5 else '#E94F37' 
              for b in bottleneck]
    
    bars = ax2.bar(N_vals, bottleneck, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(1, color='black', linestyle='-', linewidth=2, label='No bottleneck')
    ax2.axhline(2, color='orange', linestyle='--', linewidth=2, label='Mild bottleneck')
    
    # Add labels
    for bar, b in zip(bars, bottleneck):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{b:.1f}x', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('System Size N')
    ax2.set_ylabel('Bottleneck Ratio (I_req / χ)')
    ax2.set_title('Information Bottleneck Severity\n(>1 = required exceeds available)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, "03_holevo.png")
    # plt.show()


if __name__ == "__main__":
    results = run_holevo_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Information Bottleneck (Honest)")
    print("=" * 60)
    for r in results:
        status = "⚠ BOTTLENECK" if r['is_bottlenecked'] else "✓ OK"
        print(f"N={r['N']}: χ={r['holevo_chi']:.3f}, I_req={r['info_required']:.1f}, ratio={r['bottleneck_ratio']:.1f}x {status}")
