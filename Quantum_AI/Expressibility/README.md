# Paper 6: Chaos-Enhanced Expressibility

Implementation of experiments from Paper 6: *Chaos-Enhanced Expressibility: Exploring the Full Bloch Sphere via Feigenbaum Dynamics*

## Quick Start

```bash
cd /home/poig/project/self-research/Quantum_AI/Expressibility
python run_paper6.py
```

## Experiments

| Experiment | Description | Script |
|------------|-------------|--------|
| **6.1** | Bloch Sphere Coverage | `experiments/exp6_1_bloch_coverage.py` |
| **6.2** | Barren Plateau Stress Test | `experiments/exp6_2_barren_plateau.py` |
| **6.3** | DLA Saturation & Scalability | `experiments/exp6_3_scalability.py` |

## Key Predictions

1. **Exp 6.1:** ChaosOpt achieves D₂ ≈ 1.5 (structured exploration)
2. **Exp 6.2:** ChaosOpt Var(Δθ) = const (survives deep circuits)
3. **Exp 6.3:** ChaosOpt maintains fidelity up to N ≈ 10-12 qubits

## Dependencies

- numpy
- matplotlib
- qiskit (optional, for real quantum simulation)

## Structure

```
Expressibility/
├── experiments/           # Experiment scripts
│   ├── exp6_1_bloch_coverage.py
│   ├── exp6_2_barren_plateau.py
│   └── exp6_3_scalability.py
├── figures/               # Generated plots
├── results/               # Saved .npy results
├── utils/                 # Helper functions
├── run_paper6.py          # Main runner
└── README.md              # This file
```

## Connection to ChaosOpt

These experiments use the sin² map from ChaosOpt:

```python
θ_{n+1} = θ_n - γ · sin²(E(θ) · τ)
```

This is conjugate to the logistic map and exhibits Feigenbaum universality.

## References

- Paper 6: QLTO/paper/paper6.md
- ChaosOpt: /home/poig/project/ChaosOpt
- GMC Theory: arXiv:2311.04027
