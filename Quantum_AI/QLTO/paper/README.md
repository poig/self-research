# Feigenbaum Universality in Quantum Computing
## Research Paper Series

[![DOI](https://zenodo.org/badge/1079381087.svg)](https://doi.org/10.5281/zenodo.17834056)

**Author:** Tan Jun Liang  
**Affiliation:** University of Queensland  
**Contact:** junliang.tan@student.uq.edu.au

---

## Overview

This paper series explores the discovery that **quantum measurement dynamics follow Feigenbaum universality** — the same period-doubling route to chaos observed in classical nonlinear systems. The Born rule probability P = |ψ|² = sin²(θ/2) creates a unimodal map that produces the universal constant δ = 4.669...

## Papers

### Paper 1: Information-Theoretic Foundations
**Title:** *Information-Theoretic Constraints on Variational Quantum Optimization*

Establishes the thermodynamic bound on quantum optimization: ΔE ≤ η · I(S:A), where efficiency η is constrained by the Holevo capacity of the measurement channel.

---

### Paper 2: Feigenbaum Discovery
**Title:** *Feigenbaum Universality in Variational Quantum Algorithm Optimization*

**Key Discovery:** The quantum measurement update rule
```
θ_{n+1} = θ_n - γ · sin²(E(θ)τ/2)
```
follows the Feigenbaum route to chaos with δ ≈ 4.669.

---

### Paper 3: Chaos Control
**Title:** *Feigenbaum-Guided Chaos Control for Variational Quantum Algorithms*

Demonstrates spectral period detection (QFT-based) for adaptive learning rate control. Includes **hardware verification** on Rigetti Ankaa-3.

---

### Paper 4: Future Directions (Planning)
**File:** [paper4.md](./paper4.md)

Outlines experiments for:
- Cross-platform verification (IonQ, QuEra)
- Precision δ extraction to ±0.01
- Connection to 't Hooft's Cellular Automaton Interpretation

---

### Paper 5: Scaling Structure Algorithm
**Title:** *Scaling Structure as a Quantum Resource*

Proposes that Feigenbaum scaling structure enables super-polynomial quantum speedup, analogous to how Abelian group structure enables Shor's algorithm.

---

## Key Results

| Finding | Evidence |
|---------|----------|
| sin² map from Born rule | Mathematical derivation |
| δ = 4.669 in simulations | `feigenbaum_qiskit_proof.py` |
| Bifurcation on hardware | Rigetti Ankaa-3 data |
| Thermodynamic efficiency drop | 2× efficiency collapse at chaos onset |

## Code

All simulation and hardware code is available in:
- [`../multi-feigenbaum/`](../multi-feigenbaum/) — Main experiment code
- [`../Feigenbaum/`](../Feigenbaum/) — Original discovery code
- [`../chaos_control/`](../chaos_control/) — Chaos control experiments

## Citation

```bibtex
@misc{tan2024feigenbaum,
  author = {Tan, Jun Liang},
  title = {Feigenbaum Universality in Quantum Computing},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/poig/self-research}},
  doi = {10.5281/zenodo.17834056}
}
```

## License

This work is released as a preprint for academic use. All papers remain the intellectual property of the author.

---

*Last updated: December 2024*
