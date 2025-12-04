"""
Run All Figure Generation Scripts

Paper 3: Quantum-Assisted Chaos Control for VQA
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("Paper 3: Quantum-Assisted Chaos Control")
    print("Generating All Figures")
    print("=" * 60)
    print()
    
    figures = [
        ("fig1_bifurcation", "Bifurcation Diagram"),
        ("fig2_julia_sets", "Julia Sets"),
        ("fig3_trajectories", "VQA Trajectories"),
        ("fig4_fft_detection", "FFT Period Detection"),
        ("fig5_chaos_control", "Chaos Control Comparison"),
        ("fig6_control_diagram", "Control System Diagram"),
        ("fig7_catch22", "Catch-22 Bifurcation"),
        ("fig8_return_map", "Return Map Analysis"),
    ]
    
    for i, (module_name, description) in enumerate(figures, 1):
        print(f"[{i}/{len(figures)}] {description}")
        print("-" * 40)
        
        module = __import__(module_name)
        module.main()
        print()
    
    print("=" * 60)
    print("All figures generated!")
    print("=" * 60)
    
    from core import FIGURES_DIR
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob("fig*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
