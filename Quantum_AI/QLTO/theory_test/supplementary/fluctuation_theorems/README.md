# Paper 4: Fluctuation Theorem Experiments

Numerical experiments supporting **Paper 4**: *"Quantum Optimization as a Fluctuation-Driven Engine"*

## Directory Structure

```
paper4_fluctuation_theorems/
├── experiments/
│   ├── critical_exponents.py    # Extract β from efficiency collapse
│   └── work_distribution.py     # Verify Jarzynski/Sagawa-Ueda
└── figures/
    └── (generated outputs)
```

## Experiments

### 1. Critical Exponent Extraction (`critical_exponents.py`)

**Goal**: Measure β from efficiency scaling: η ~ |N - N_c|^β

**Predictions**:
- Mean-field: β = 0.5
- Ising: β ≈ 0.33

**Run**:
```bash
cd experiments
python critical_exponents.py
```

**Outputs**:
- `figures/critical_exponents.png` — η(N) plot + log-log fit
- Console: β ± error, N_c estimate

---

### 2. Work Distribution Measurement (`work_distribution.py`)

**Goal**: Verify fluctuation theorems:
- **Jarzynski**: ⟨exp(-βW)⟩ = exp(-βΔF)
- **Sagawa-Ueda**: ⟨exp(-β(W - k_B T I))⟩ ≈ 1

**Run**:
```bash
cd experiments
python work_distribution.py
```

**Outputs**:
- `figures/work_distribution.png` — P(W), W-vs-I correlation, verification bars
- Console: Mean work, mutual information, theorem satisfaction

---

## Dependencies

```bash
pip install numpy matplotlib scipy qiskit qiskit-aer
```

For mock mode (no Qiskit), scripts will run with simulated data.

---

## Connection to Paper 4

| Experiment | Paper Section | Key Claim |
|------------|---------------|-----------|
| `critical_exponents.py` | Section IV | β ≈ 0.5 → mean-field universality |
| `work_distribution.py` | Section II-III | Derive η from Sagawa-Ueda |

---

## Quick Start

```bash
cd /home/poig/project/self-research/Quantum_AI/paper4_fluctuation_theorems/experiments

# Test run (reduced samples)
python critical_exponents.py

# Full run
python work_distribution.py
```
