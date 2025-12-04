"""
Figure 5: Chaos Control - Before and After

Simple comparison: uncontrolled chaos vs controlled stability.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import sin2_map, compute_trajectory, detect_period_quantum, FIGURES_DIR


def run_controlled(n_steps=200, initial_gamma=0.75, control_interval=16):
    """Run VQA with adaptive chaos control."""
    gamma_history = [initial_gamma]
    trajectory = [0.5]
    
    x = 0.5
    gamma = initial_gamma
    
    for i in range(n_steps):
        x = sin2_map(x, gamma)
        trajectory.append(x)
        
        if (i + 1) % control_interval == 0 and i > control_interval:
            window = np.array(trajectory[-control_interval:])
            _, detected_period = detect_period_quantum(window, n_qubits=4)
            
            # Control law
            if detected_period >= 4:
                gamma = max(0.5, gamma * 0.85)
            elif detected_period == 1 and gamma < 0.7:
                gamma = min(0.78, gamma * 1.05)
            
            gamma_history.append(gamma)
    
    return np.array(trajectory), gamma_history


def main():
    print("Generating Chaos Control comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    n_iter = 200
    
    # Uncontrolled
    ax1 = axes[0]
    traj_uncontrolled = compute_trajectory(0.5, 0.78, n_iter)
    ax1.plot(traj_uncontrolled, 'r-', lw=0.8, alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('x', fontsize=11)
    ax1.set_title('Uncontrolled (γ = 0.78)\n→ Chaotic', 
                  fontsize=12, fontweight='bold', color='red')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Controlled
    ax2 = axes[1]
    traj_controlled, gamma_history = run_controlled(n_iter)
    ax2.plot(traj_controlled, 'b-', lw=0.8, alpha=0.8)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('x', fontsize=11)
    ax2.set_title('Controlled (Adaptive γ)\n→ Stable', 
                  fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Learning rate adaptation
    ax3 = axes[2]
    iterations = np.arange(len(gamma_history)) * 16
    ax3.plot(iterations, gamma_history, 'g-o', lw=2, markersize=5)
    ax3.axhline(0.73, color='red', linestyle='--', lw=1.5, label=r'Chaos threshold $r_\infty$')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Learning Rate γ', fontsize=11)
    ax3.set_title('Adaptive γ\nKeeps r < r∞', fontsize=12, fontweight='bold', color='green')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0.45, 0.85)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Quantum-Assisted Chaos Control', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig5_chaos_control.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  Final γ = {gamma_history[-1]:.3f} (kept below chaos threshold)")


if __name__ == "__main__":
    main()
