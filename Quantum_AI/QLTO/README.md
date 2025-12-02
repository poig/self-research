

# QLTO: Quantum Landscape Tunneling Optimizer

A geometry-aware quantum optimization framework for variational quantum algorithms.

## Abstract

This project investigates the information-theoretic foundations of variational quantum optimization. I demonstrate that the trainability of variational quantum algorithms is governed by thermodynamic constraints, specifically the information channel capacity of ancilla-mediated feedback control. By analyzing systems with different Dynamical Lie Algebra (DLA) scaling, I identify a phase transition between trainable ("Ordered") and untrainable ("Chaotic") regimes.

## Papers

### Paper 1: Theory

**Title**: *Information-Theoretic Constraints on Variational Quantum Optimization: Efficiency Transitions and the Dynamical Lie Algebra*

**Summary**: I reframe the variational optimizer as a quantum Maxwell's Demon and establish an empirical constitutive relation $\Delta E \leq \eta \cdot I(S:A)$ linking work extraction to mutual information. Key findings:

- Quantum entanglement provides a factor-of-2 advantage over classical Landauer bounds
- Systems with polynomial DLA ($O(N^3)$) exhibit sustained positive efficiency
- Systems with exponential DLA ($O(4^N)$) undergo efficiency collapse at $N \approx 6$ qubits
- The efficiency coefficient $\eta$ serves as a trainability diagnostic

### Paper 2: Algorithm

**Title**: *Scalable Riemannian Quantum Optimization via Commuting-Block Decomposition*

**Summary**: I introduce QLTO, a practical optimizer that maintains trainability by decomposing the ansatz into commuting blocks. Key features:

- Reduces metric tensor estimation from $O(N^2)$ to $O(L)$ circuits
- Reduces matrix inversion from $O(N^3)$ to $O(N)$ complexity
- Ancilla-controlled quantum walk for geometry-aware exploration
- Achieves competitive accuracy with $2.5\times$ fewer function evaluations

## Method Overview

| Component | Function | Complexity |
|-----------|----------|------------|
| QNN Circuit ($U_{\text{QNN}}$) | Parameterized trial state ansatz | Quantum |
| Commuting-Block FIM | Efficient metric tensor estimation | $O(L)$ |
| Riemannian Mixer/Drift | Curvature-scaled exploration | $O(N)$ |
| Ancilla Sensing | Hadamard-test energy oracle | 1 qubit |

## Repository Structure

```
QLTO/
├── Application/      # Benchmark implementations
├── theory_test/      # Numerical experiments for theory paper
└── README.md
```

## Requirements

- Python 3.8+
- Qiskit >= 2.0
- NumPy, SciPy
- Matplotlib (for visualization)

## Benchmarks

Tested on:
- Heisenberg models (2-12 qubits)
- MaxCut problems (vs. QAOA p=3)
- Molecular ground states (H₂, LiH)

## Citation

If you use this code in your research, please cite the associated papers (arXiv links to be added upon acceptance).

## License

See the repo-level `LICENSE` file.

## Acknowledgments

The author used a large language model to assist with manuscript preparation and takes full intellectual responsibility for all scientific content.