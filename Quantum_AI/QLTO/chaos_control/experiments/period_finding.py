"""
Shor vs Feigenbaum Period Finding Comparison

Both use Fourier analysis to extract periodicity, but in different contexts:

| Aspect           | Shor's Algorithm    | Feigenbaum/VQA        |
|------------------|---------------------|----------------------|
| What has period  | Function f(x)       | Dynamical orbit x_n  |
| Period means     | f(x) = f(x+r)       | x_n = x_{n+k}        |
| Tool             | QFT on input        | QFT/FFT on trajectory|
| Goal             | Find hidden r       | Detect/control chaos |
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core import compute_trajectory, detect_period_quantum, FIGURES_DIR


def main():
    print("Generating Shor vs Feigenbaum Period Finding comparison...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Row 1: Shor's Period Finding
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Illustrating Shor's period finding...")
    ax_shor = fig.add_subplot(gs[0, :2])
    N, period = 16, 4
    x_shor = np.arange(N)
    f_shor = x_shor % period  # Simple periodic function

    ax_shor.stem(x_shor, f_shor, linefmt='b-', markerfmt='bo', basefmt='k-')
    # Highlight period boundaries
    for i in range(0, N, period):
        ax_shor.axvspan(i, i+0.5, alpha=0.3, color='red')
    
    ax_shor.text(2, 3.5, f'Period r = {period}', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax_shor.set_xlabel('x (input)', fontsize=11)
    ax_shor.set_ylabel('f(x)', fontsize=11)
    ax_shor.set_title("Shor's Period Finding: f(x) = f(x + r)", 
                     fontsize=12, fontweight='bold')
    ax_shor.grid(True, alpha=0.3)

    # QFT reveals the period
    ax_qft = fig.add_subplot(gs[0, 2:])
    qft_result = np.fft.fft(f_shor)
    qft_power = np.abs(qft_result) ** 2
    
    ax_qft.stem(range(N), qft_power, linefmt='g-', markerfmt='go', basefmt='k-')
    # Mark expected peaks
    for k in range(0, N, N//period):
        ax_qft.axvline(k, color='red', linestyle='--', alpha=0.5)
    
    ax_qft.text(8, max(qft_power)*0.8, 
               f'Peaks at k = 0,4,8,12\n→ Period = {N}//{N//period} = {period}', 
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax_qft.set_xlabel('Frequency k', fontsize=11)
    ax_qft.set_ylabel('|QFT|²', fontsize=11)
    ax_qft.set_title('QFT Spectrum Reveals Period', fontsize=12, fontweight='bold')
    ax_qft.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Row 2: Feigenbaum Trajectories
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Computing VQA trajectories...")
    r_values = [0.55, 0.68, 0.72, 0.78]
    period_labels = ['Period-1', 'Period-2', 'Period-4', 'Chaos']
    colors = ['blue', 'green', 'orange', 'red']

    for i, (r, label, col) in enumerate(zip(r_values, period_labels, colors)):
        ax_traj = fig.add_subplot(gs[1, i])
        traj = compute_trajectory(0.5, r, 300)
        ax_traj.plot(range(50), traj[250:300], '-o', color=col, markersize=3, lw=1)
        ax_traj.set_xlabel('Iteration n', fontsize=10)
        ax_traj.set_ylabel('x_n', fontsize=10)
        ax_traj.set_title(f'{label}\nr = {r}', fontsize=11, fontweight='bold', color=col)
        ax_traj.set_ylim(0.3, 0.8)
        ax_traj.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Row 3: FFT of Trajectories (Feigenbaum period detection)
    # ═══════════════════════════════════════════════════════════════════
    
    print("  Computing FFT spectra...")
    for i, (r, label, col) in enumerate(zip(r_values, period_labels, colors)):
        ax_fft = fig.add_subplot(gs[2, i])
        traj = compute_trajectory(0.5, r, 500)
        steady = traj[200:]  # Use steady-state part
        
        # Compute FFT
        fft = np.fft.fft(steady - np.mean(steady))
        freqs = np.fft.fftfreq(len(steady))
        power = np.abs(fft) ** 2
        power[0] = 0  # Remove DC component
        
        n_freqs = len(freqs) // 2
        ax_fft.plot(freqs[:n_freqs], power[:n_freqs], color=col, lw=1.5)
        ax_fft.fill_between(freqs[:n_freqs], 0, power[:n_freqs], color=col, alpha=0.3)
        ax_fft.set_xlabel('Frequency', fontsize=10)
        ax_fft.set_ylabel('Power', fontsize=10)
        ax_fft.set_title('FFT of Trajectory', fontsize=10, fontweight='bold')
        ax_fft.set_xlim(0, 0.5)
        ax_fft.grid(True, alpha=0.3)

    # Title and caption
    fig.suptitle('Period Finding: Shor (QFT on function) vs Feigenbaum (FFT on dynamics)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.01, 
             'Both use Fourier Transform to extract periodicity!\n'
             'Shor: Period of f(x) in input space  |  Feigenbaum: Period of orbit in iteration space',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Save
    output_path = FIGURES_DIR / 'period_finding_connection.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
