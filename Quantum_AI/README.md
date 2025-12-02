

# Quantum AI Research

This repository contains research on information-theoretic approaches to variational quantum optimization.

## Overview

We investigate the fundamental thermodynamic constraints governing variational quantum algorithms (VQAs), with a focus on understanding the trainability transition (barren plateau phenomenon) through the lens of quantum information theory and the Dynamical Lie Algebra (DLA).

## Projects

### [QLTO](./QLTO/) - Quantum Landscape Tunneling Optimizer

A geometry-aware quantum optimization framework that addresses the barren plateau problem through:

1. **Theory Paper**: *Information-Theoretic Constraints on Variational Quantum Optimization*
   - Establishes an empirical constitutive relation linking work extraction to mutual information
   - Identifies efficiency transitions governed by DLA dimension (polynomial vs. exponential)
   - Demonstrates quantum entanglement provides a factor-of-2 advantage over classical Landauer bounds

2. **Algorithm implementation**: *Scalable Riemannian Quantum Optimization via Commuting-Block Decomposition* `Application/algorithm.pdf`
   - Introduces QLTO with $O(N)$ metric tensor estimation via commuting-block structure
   - Implements ancilla-controlled quantum walk for geometry-aware optimization
   - Achieves competitive accuracy with reduced function evaluations on benchmark problems

## Key Contributions

- **Thermodynamic perspective** on VQA trainability: optimization as a Maxwell's Demon heat engine
- **Efficiency coefficient** $\eta = dW/dI$ as a diagnostic for algorithm trainability
- **Commuting-block decomposition** reducing QNG complexity from $O(N^3)$ to $O(N)$
- **Empirical phase transition** characterization between trainable (Ordered) and untrainable (Chaotic) regimes

## References

The theoretical framework builds upon:
- Ragone et al. (2024) - DLA theory for barren plateaus
- Francica et al. (2017) - Daemonic ergotropy
- Stokes et al. (2020) - Quantum Natural Gradient

## License

This research is for academic purposes. See individual paper directories for specific licensing.
