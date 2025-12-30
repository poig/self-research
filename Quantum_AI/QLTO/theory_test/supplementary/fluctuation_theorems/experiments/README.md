# Paper 4: Quantum Thermodynamics of VQA

## Experiments

Clean, modular experiment suite for Paper 4: "Fluctuation Theorems in Quantum Optimization".

### Structure

```
experiments/
├── core.py              # Shared library (CoherentDemonEngine, utilities)
├── exp01_n_sweep.py     # Efficiency vs system size
├── exp02_jarzynski.py   # Jarzynski equality (operational ΔF)
├── exp03_holevo.py      # Holevo capacity saturation
├── exp04_free_energy.py # Free energy landscape
├── exp05_mss.py         # MSS scrambling bound
├── exp06_noise.py       # Hardware noise comparison
├── run_all.py           # Master script
└── README.md            # This file
```

### Usage

Run individual experiments:
```bash
python exp01_n_sweep.py
python exp02_jarzynski.py
# etc.
```

Run all experiments:
```bash
python run_all.py
```

### Key Findings

| Experiment | Key Result |
|------------|------------|
| N-Sweep | η peaks at N=4, collapses at N=6 |
| Jarzynski | Error < 30% with operational ΔF |
| Holevo | I_req/χ = 10-14x (bottleneck) |
| Free Energy | Curvature changes at transition |
| MSS | 34% saturation (moderate scrambler) |
| Noise | DLA collapse dominates over noise |

### Dependencies

- numpy
- scipy
- matplotlib
- qiskit
- qiskit-aer
