#!/usr/bin/env python3
"""
Feigenbaum Universality in VQA Optimization - Figure Generation
Main entry point that imports and runs individual figure modules.

Usage:
    python main_experiment.py              # Generate all figures
    python main_experiment.py --fast       # Fast mode (lower resolution)
    python main_experiment.py --figure 1   # Generate specific figure only
"""

import argparse

from utils import FIGURES_DIR, HAS_QISKIT
from fig1_qiskit_proof import plot_feigenbaum_qiskit_proof
from fig2_bifurcation import plot_quantum_bifurcation_2d
from fig3_fractal import plot_unified_fractal_bifurcation
from fig4_trainability import plot_quantum_trainability_fractal
from fig5_bandwidth import plot_qubit_bandwidth_comparison


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fast', action='store_true', help='Fast mode with lower resolution')
    parser.add_argument('--figure', type=int, choices=[1, 2, 3, 4, 5],
                       help='Generate specific figure only')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Feigenbaum Universality in VQA - Figure Generation")
    print("=" * 60)
    print(f"Output directory: {FIGURES_DIR}")
    print(f"Qiskit available: {HAS_QISKIT}")
    print(f"Fast mode: {args.fast}")
    print("=" * 60)
    
    figures = {
        1: ("feigenbaum_qiskit_proof.png", plot_feigenbaum_qiskit_proof),
        2: ("quantum_bifurcation_2d.png", plot_quantum_bifurcation_2d),
        3: ("unified_fractal_bifurcation.png", plot_unified_fractal_bifurcation),
        4: ("quantum_trainability_fractal.png", plot_quantum_trainability_fractal),
        5: ("qubit_bandwidth_comparison.png", plot_qubit_bandwidth_comparison),
    }
    
    if args.figure:
        print(f"\n[Figure {args.figure}]")
        figures[args.figure][1](fast_mode=args.fast)
    else:
        for fig_num, (name, fig_func) in figures.items():
            print(f"\n[Figure {fig_num}]")
            fig_func(fast_mode=args.fast)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
