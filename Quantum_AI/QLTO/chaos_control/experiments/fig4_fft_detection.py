"""
Figure 4: Period Detection via FFT

Shows how Fourier analysis reveals the dynamical period.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import compute_trajectory, FIGURES_DIR


def main():
    print("Generating FFT Period Detection...")
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    regimes = [
        (0.55, 'Period-1', 'blue'),
        (0.68, 'Period-2', 'green'),
        (0.72, 'Period-4', 'orange'),
        (0.78, 'Chaos', 'red'),
    ]
    
    for ax, (r, label, color) in zip(axes, regimes):
        # Compute trajectory
        traj = compute_trajectory(0.5, r, 600)
        steady = traj[300:]  # Use steady-state
        
        # FFT
        signal = steady - np.mean(steady)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        power = np.abs(fft) ** 2
        
        # Only positive frequencies
        n_half = len(freqs) // 2
        freqs_pos = freqs[1:n_half]
        power_pos = power[1:n_half]
        
        # Handle Period-1 case (constant signal → no frequency content)
        max_power = np.max(power_pos) if len(power_pos) > 0 else 0
        if max_power < 1e-10:  # Essentially zero → Period-1
            ax.text(0.5, 0.5, 'No oscillation\n(Period-1)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, color=color, fontweight='bold')
            ax.set_ylim(0, 1)
        else:
            ax.fill_between(freqs_pos, 0, power_pos, color=color, alpha=0.3)
            ax.plot(freqs_pos, power_pos, color=color, lw=1.5)
            
            # Mark dominant frequency
            peak_idx = np.argmax(power_pos)
            peak_freq = freqs_pos[peak_idx]
            if peak_freq > 0.01:  # Avoid DC
                period = int(round(1 / peak_freq)) if peak_freq > 0 else 1
                ax.axvline(peak_freq, color='black', linestyle='--', alpha=0.5)
                ax.text(peak_freq + 0.02, power_pos[peak_idx] * 0.8, 
                       f'T≈{period}', fontsize=9)
        
        ax.set_xlabel('Frequency', fontsize=10)
        ax.set_ylabel('Power', fontsize=10)
        ax.set_title(f'{label} (r={r})', fontsize=11, fontweight='bold', color=color)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Period Detection via Fourier Analysis', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig4_fft_detection.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
