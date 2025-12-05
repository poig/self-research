# Paper 5: Beyond Phase Expression
## Multi-Feigenbaum Experiment Suite

This directory contains the experimental code for **Paper 5: "Universality Classes of Quantum Measurement Back-Action"**.

## Key Questions

1. **Basis Independence**: Do X, Y, Z measurement bases all give δ = 4.669?
2. **Weak Measurement Threshold**: What is the critical $g_c$ where chaos onsets?
3. **Non-Commuting Observables**: Does $[H, X] \neq 0$ break period-doubling universality?
4. **MIPT Connection**: Explicit link between measurement rate and bifurcation cascade

## Directory Structure

```
multi-feigenbaum/
├── README.md                    # This file
├── measurement_maps.py          # Core library: sin², cos², weak, cusp maps
├── basis_universality.py        # Exp 1: X, Y, Z basis comparison
├── weak_measurement_cascade.py  # Exp 2: Find critical g_c
├── dual_observable_chaos.py     # Exp 3: Non-commuting 2D dynamics
├── run_all_experiments.py       # Master script
└── figures/                     # Generated plots
```

## Quick Start

```bash
# Run quick test
python -c "from run_all_experiments import run_quick_test; run_quick_test()"

# Run all experiments
python run_all_experiments.py

# Run individual experiments
python basis_universality.py
python weak_measurement_cascade.py
python dual_observable_chaos.py
```

## Expected Output

### Experiment 1: Basis Independence
- Bifurcation diagrams for Z, Y, Rx, Ry bases
- δ convergence plot showing all bases → 4.669
- **Expected result**: ALL bases give same δ

### Experiment 2: Weak Measurement Transition
- Phase diagram: (g, r) → Lyapunov exponent
- Critical threshold $g_c \approx 0.5-0.7$
- **Expected result**: Sharp transition at $g_c$

### Experiment 3: Non-Commuting Observables
- 2D attractor gallery for different coupling strengths
- Strange attractor at high coupling
- **Expected result**: Break of 1D universality

## Dependencies

```python
numpy
matplotlib
qiskit
qiskit-aer  # optional, for shot-based simulation
```

## Key Equations

### Standard (Paper 1-4)
$$P(|1\rangle) = \sin^2(\phi/2) \quad \Rightarrow \quad \delta = 4.669$$

### Weak Measurement (Paper 5)
$$P(|1\rangle; g) = (1-g) \cdot 0.5 + g \cdot \sin^2(\phi/2)$$

### 2D Coupled Map (Paper 5)
$$x_{n+1} = r \cdot \sin^2(\pi x_n)(1 + c(y_n - 0.5))$$
$$y_{n+1} = r \cdot \sin^2(\pi y_n)(1 + c(x_n - 0.5))$$

## Theoretical Background

See `/bin/QLTO/src/ancilla_test/paper/paper5.md` for full theoretical context.

Key references:
- Feigenbaum (1978): Universal constants δ = 4.669, α = 2.502
- Eastman et al. (2017, 2019): Quantum chaos control via measurement
- Li et al. (2018): Measurement-induced phase transitions
- Wiseman & Milburn (2010): Quantum measurement and control

## Author

QLTO Research Team
December 2024
