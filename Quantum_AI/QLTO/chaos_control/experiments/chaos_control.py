"""
Quantum Chaos Control System

Demonstrates the complete chaos control loop:
1. VQA Optimizer produces trajectory θ(n)
2. Trajectory Buffer collects recent N points
3. Quantum Period Detector (QFT) extracts current period
4. Controller adjusts learning rate γ to maintain stability
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from core import (sin2_map, compute_trajectory, detect_period_quantum,
                  run_controlled_optimization, FIGURES_DIR)


def main():
    print("Generating Quantum Chaos Control System visualization...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Row 1: Quantum Circuit and Detection
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Drawing quantum circuit diagram...")
    ax_circuit = fig.add_subplot(gs[0, :2])
    ax_circuit.set_xlim(0, 10)
    ax_circuit.set_ylim(0, 5)

    y_idx, y_val = 3.5, 1.5
    ax_circuit.hlines([y_idx], 0.5, 9.5, colors='black', lw=1.5)
    ax_circuit.hlines([y_val], 0.5, 9.5, colors='black', lw=1.5)
    ax_circuit.text(0.2, y_idx, '|n⟩', fontsize=12, va='center', fontweight='bold')
    ax_circuit.text(0.2, y_val, '|x⟩', fontsize=12, va='center', fontweight='bold')

    # Hadamard gate
    rect = Rectangle((1.3, y_idx-0.3), 0.4, 0.6, fill=True, 
                     facecolor='lightblue', edgecolor='black')
    ax_circuit.add_patch(rect)
    ax_circuit.text(1.5, y_idx, 'H', fontsize=10, ha='center', va='center', fontweight='bold')

    # Oracle
    rect_oracle = Rectangle((2.5, y_val-0.8), 1.5, 2.8, fill=True, 
                            facecolor='lightyellow', edgecolor='black', lw=2)
    ax_circuit.add_patch(rect_oracle)
    ax_circuit.text(3.25, y_idx-0.8, 'Oracle\nU_traj', fontsize=10, 
                   ha='center', va='center', fontweight='bold')

    # QFT
    rect_qft = Rectangle((5, y_idx-0.4), 1.2, 0.8, fill=True, 
                         facecolor='lightgreen', edgecolor='black', lw=2)
    ax_circuit.add_patch(rect_qft)
    ax_circuit.text(5.6, y_idx, 'QFT', fontsize=11, ha='center', va='center', fontweight='bold')

    # Measurement
    ax_circuit.plot([7.5], [y_idx], 'ko', markersize=15)
    ax_circuit.plot([7.5], [y_idx], 'k|', markersize=20, mew=2)
    ax_circuit.text(7.5, y_idx+0.5, 'Measure k', fontsize=9, ha='center')
    ax_circuit.text(8.5, y_idx, '→ Period = N/gcd(k,N)', fontsize=10, va='center')

    ax_circuit.set_title('Quantum Circuit for VQA Period Detection\n(Shor-like approach)', 
                        fontsize=13, fontweight='bold')
    ax_circuit.axis('off')

    # Detection results
    print("  Computing QFT period detection...")
    ax_detect = fig.add_subplot(gs[0, 2:])

    r_test_values = [0.55, 0.68, 0.72, 0.78]
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Period-1', 'Period-2', 'Period-4', 'Chaos']

    bar_width = 0.2
    x_positions = np.arange(16)

    for i, (r, col, label) in enumerate(zip(r_test_values, colors, labels)):
        traj = compute_trajectory(0.5, r, 64)
        freq_probs, detected = detect_period_quantum(traj[32:], n_qubits=4)
        ax_detect.bar(x_positions + i*bar_width, freq_probs, bar_width, 
                      color=col, alpha=0.7, label=f'{label} (r={r}) → Det: {detected}')

    ax_detect.set_xlabel('Frequency k', fontsize=11)
    ax_detect.set_ylabel('Probability', fontsize=11)
    ax_detect.set_title('QFT Frequency Spectrum from VQA Trajectories', 
                       fontsize=12, fontweight='bold')
    ax_detect.legend(fontsize=8, loc='upper right')
    ax_detect.set_xlim(-0.5, 16)
    ax_detect.grid(True, alpha=0.3, axis='y')

    # ═══════════════════════════════════════════════════════════════════
    # Row 2: Control System Block Diagram
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Drawing control system diagram...")
    ax_control = fig.add_subplot(gs[1, :])
    ax_control.set_xlim(0, 14)
    ax_control.set_ylim(0, 4)

    blocks = [
        (0.5, 'VQA\nOptimizer', 'lightblue'),
        (3.5, 'Trajectory\nBuffer', 'lightyellow'),
        (6.5, 'Quantum\nPeriod Detector\n(QFT)', 'lightgreen'),
        (10, 'Learning\nRate\nController', 'lightcoral'),
    ]

    for x, text, color in blocks:
        w = 2.5 if 'Period' in text else 2
        rect = Rectangle((x, 1.5), w, 1.5, fill=True, 
                         facecolor=color, edgecolor='black', lw=2)
        ax_control.add_patch(rect)
        ax_control.text(x + w/2, 2.25, text, fontsize=10, 
                       ha='center', va='center', fontweight='bold')

    # Arrows between blocks
    for x1, x2, label in [(2.5, 3.5, 'θ(n)'), (5.5, 6.5, ''), (9, 10, 'Period k')]:
        ax_control.annotate('', xy=(x2, 2.25), xytext=(x1, 2.25), 
                           arrowprops=dict(arrowstyle='->', lw=2))
        if label:
            ax_control.text((x1+x2)/2, 2.6, label, fontsize=10, ha='center')

    # Feedback arrow
    ax_control.annotate('', xy=(0.5, 1.5), xytext=(10, 1.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                                      connectionstyle='arc3,rad=-0.3'))
    ax_control.text(5.5, 0.5, 'Adjust γ to maintain Period ≤ 2', fontsize=11, 
                   ha='center', color='red', fontweight='bold')

    # Control law
    ax_control.text(13, 2.25, 'if k > 2:\n  γ ← γ × 0.9\nelif k == 1:\n  γ ← γ × 1.05', 
                   fontsize=9, family='monospace', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_control.set_title('Quantum-Assisted VQA Control System', fontsize=13, fontweight='bold')
    ax_control.axis('off')

    # ═══════════════════════════════════════════════════════════════════
    # Row 3: Control Demonstration
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Running control simulation...")
    np.random.seed(42)
    n_iter = 200

    # Uncontrolled trajectory
    ax_uncontrolled = fig.add_subplot(gs[2, 0])
    traj_uncontrolled = compute_trajectory(0.5, 0.78, n_iter)
    ax_uncontrolled.plot(traj_uncontrolled, 'r-', lw=0.8, alpha=0.8)
    ax_uncontrolled.set_xlabel('Iteration', fontsize=10)
    ax_uncontrolled.set_ylabel('x', fontsize=10)
    ax_uncontrolled.set_title('Uncontrolled (γ = 0.78)\n→ CHAOS', 
                              fontsize=11, fontweight='bold', color='red')
    ax_uncontrolled.set_ylim(0, 1)
    ax_uncontrolled.grid(True, alpha=0.3)

    # Controlled trajectory
    ax_controlled = fig.add_subplot(gs[2, 1])
    traj_controlled, gamma_history, period_history = run_controlled_optimization(
        n_steps=n_iter, initial_gamma=0.75, control_interval=16
    )
    
    ax_controlled.plot(traj_controlled, 'b-', lw=0.8, alpha=0.8)
    ax_controlled.set_xlabel('Iteration', fontsize=10)
    ax_controlled.set_ylabel('x', fontsize=10)
    ax_controlled.set_title('Controlled (Adaptive γ)\n→ STABLE', 
                           fontsize=11, fontweight='bold', color='blue')
    ax_controlled.set_ylim(0, 1)
    ax_controlled.grid(True, alpha=0.3)

    # Gamma history
    ax_gamma = fig.add_subplot(gs[2, 2])
    ax_gamma.plot(np.arange(len(gamma_history)) * 16, gamma_history, 
                 'g-o', lw=2, markersize=4)
    ax_gamma.axhline(0.73, color='red', linestyle='--', label='Chaos threshold')
    ax_gamma.set_xlabel('Iteration', fontsize=10)
    ax_gamma.set_ylabel('Learning Rate γ', fontsize=10)
    ax_gamma.set_title('Adaptive Learning Rate', fontsize=11, fontweight='bold')
    ax_gamma.legend(fontsize=9)
    ax_gamma.grid(True, alpha=0.3)
    ax_gamma.set_ylim(0.4, 0.85)

    # Period histogram comparison
    ax_hist = fig.add_subplot(gs[2, 3])
    
    # Compute periods for uncontrolled
    periods_uncontrolled = []
    for i in range(10, n_iter, 16):
        window = traj_uncontrolled[i:i+16]
        if len(window) == 16:
            _, p = detect_period_quantum(window, n_qubits=4)
            periods_uncontrolled.append(p)

    ax_hist.hist([periods_uncontrolled, period_history], 
                bins=[0.5, 1.5, 2.5, 4.5, 8.5, 16.5],
                label=['Uncontrolled', 'Controlled'],
                color=['red', 'blue'], alpha=0.7)
    ax_hist.set_xlabel('Detected Period', fontsize=10)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_title('Period Distribution', fontsize=11, fontweight='bold')
    ax_hist.legend(fontsize=9)
    ax_hist.set_xticks([1, 2, 4, 8, 16])

    fig.suptitle('Quantum-Assisted Chaos Control for VQA Optimization', 
                 fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = FIGURES_DIR / 'quantum_chaos_control.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ Saved: {output_path}")
    print(f"\n📊 Control Results:")
    print(f"   Uncontrolled: Fixed γ = 0.78, mostly chaotic")
    print(f"   Controlled:   Final γ = {gamma_history[-1]:.2f}, kept stable through adaptation")


if __name__ == "__main__":
    main()
