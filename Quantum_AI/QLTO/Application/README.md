# NISQ V2: Riemannian Coherent QLTO (MPS Enabled)

## Overview
This project implements **Riemannian Coherent Quantum Learning via Trajectory Optimization (QLTO)**, a novel quantum optimization architecture that merges the Quantum nature gradient and the geometric efficiency of **Riemannian optimization** for quantum optimization tools.

The core innovation is the "Symmetric Sandwich" quantum walk, which evolves a parameter wavefunction on a curved Riemannian manifold defined by the Quantum Fisher Information Matrix (QFIM). This allows the optimizer to navigate the energy landscape using **Natural Gradient flow** coherently, bypassing the "blind search" limitations of standard Grover-based methods.

This V2 implementation includes:
- **Matrix Product State (MPS) Support**: Enables simulation of larger systems with significant entanglement.
- **Commuting-Block Geometry Sensing**: Achieves $O(L)$ scaling for Gradient and Metric computation, making it feasible to train deep, overparameterized circuits.

## Key Features
- **Riemannian Coherent Walk**: Uses a metric-scaled mixer to perform geodesic descent on the parameter manifold.
- **Efficient Sensing**: Exploits commuting-block structure to measure gradients and curvature with linear circuit depth cost.
- **Hybrid Architecture**: Combines classical geometry sensing with coherent quantum evolution.
- **Benchmark Suite**: Includes a comprehensive benchmark against Classical QNG, AdamW, and SPSA.

## Research Summary

### Plan: Fusing QWOA and QLTO
The research plan focused on integrating the structural benefits of QWOA into the QLTO framework.
- **Goal**: To prevent Barren Plateaus and ensure trainability.
- **Strategy**:
    1.  **Structure-Aware Ansatz**: Using commuting-block Hamiltonians (like QWOA) to guarantee polynomial Dynamical Lie Algebra (DLA) dimension.
    2.  **Indexed Parameter Space**: Compressing the search space to valid parameter manifolds.
    3.  **Hybrid Mixing**: Combining Riemannian diffusion (local descent) with Complete Graph mixing (global tunneling) to escape local minima.

### Verdict: Complexity & Performance
The theoretical analysis concludes that QLTO represents a **"Geometric Quantum Descent"** paradigm.
- **Complexity Class**: Efficiently solves problems in **BPPO** (Bounded-Error Probabilistic Polynomial-Time Optimization) and **BP-APX** (Approximable problems).
- **The "Curse" Resolution**:
    - **Expressivity**: QLTO cannot break the physical limit of the ansatz; hard problems still require deep circuits ($O(\sqrt{2^N})$).
    - **Trainability**: QLTO **breaks the optimization bottleneck**. It enables the training of the deep, overparameterized circuits required for hard problems by using efficient $O(L)$ geometry sensing and Riemannian navigation.
- **Conclusion**: QLTO is a "Polynomial-Time Solver" for representable problems and a robust "Approximate Solver" for hard physical systems, offering exponential convergence in the optimization loop compared to standard blind search.

## References

### implimentaiton reference for efficient NQG (commute_fim.py, commute_gradient.py) 
- **1909.02108v3**: *Quantum Natural Gradient* - https://arxiv.org/pdf/1909.02108
- **2505.09818v1**: *Efficient protocol to estimate the Quantum Fisher Information Matrix for Commuting-Block Circuits* - https://arxiv.org/pdf/2505.09818
- **q-2025-10-02-1873**: *Backpropagation scaling in parameterised quantum circuits* - https://quantum-journal.org/papers/q-2025-10-02-1873/

### implimentaiton reference for QLTO (nisq_v2.py)
- **2508.05749v1**: *Expressivity Limits and Trainability Guarantees in Quantum Walk-based Optimization* - **Key Theoretical Basis for QLTO Limits.** https://arxiv.org/html/2508.05749v1 
- **PhysRevResearch.2.023302**: *Combinatorial optimization via highly efficient quantum walks* - https://arxiv.org/pdf/PhysRevResearch.2.023302
- **s11128-019-2171-3**: *A quantum walk-assisted approximate algorithm for bounded NP optimisation problems* - https://link.springer.com/article/10.1007/s11128-019-2171-3
- **2309.09342v3**: *A Lie Algebraic Theory of Barren Plateaus for Deep Parameterized Quantum Circuits* - https://arxiv.org/pdf/2309.09342v3
- **2407.12587v1**: *On the dynamical Lie algebras of quantum approximate optimization algorithms* - https://arxiv.org/pdf/2407.12587v1

These papers provide the mathematical foundation for the Commuting-Block ansatz structure, the Riemannian metric sensing protocols, and the convergence guarantees utilized in this project.

## Usage

### Running the Optimizer
To run the main QLTO optimizer on a Heisenberg spin chain:
```bash
python nisq_v2.py
```

### Running Benchmarks
To compare QLTO against other optimizers (Correct QNG, AdamW, SPSA):
```bash
python benchmark.py
```
This will generate performance plots (e.g., `benchmark_Heisenberg_N4.png`) showing Energy vs. NEFV (Number of Function Evaluations).
