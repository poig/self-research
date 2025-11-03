# Project File Structure - Clean Organization

## Files to KEEP (Essential)

### Main Tools
```
can_we_crack_aes.py          ✅ Main AES analysis tool (interactive)
```

### Core Solvers
```
src/core/
├── quantum_sat_solver.py     ✅ Main solver class
├── integrated_pipeline.py    ✅ Analysis pipeline
└── __init__.py
```

### AES Encoders
```
src/solvers/
├── aes_full_encoder.py       ✅ Full 10-round AES encoding
├── aes_sbox_encoder.py       ✅ S-box SAT encoding  
├── aes_mixcolumns_encoder.py ✅ MixColumns encoding
├── structure_aligned_qaoa.py  ✅ QAOA solver
└── __init__.py
```

### Decomposition
```
experiments/
├── sat_decompose.py           ✅ Decomposition algorithms
└── sat_undecomposable_quantum.py  ✅ Hardness certification
```

### Documentation
```
README.md                      ✅ Project overview
docs/
├── AES_CRACKING_GUIDE.md      ✅ Step-by-step tutorial
├── FINAL_SUMMARY.md           ✅ Results summary
├── BREAKTHROUGH_AES_CRACKABLE.md  ✅ Research findings
├── SPECTRAL_ANALYSIS_EXPLAINED.md ✅ Technical deep-dive
└── archive/                   📁 Old/redundant docs
```

---

## Files to ARCHIVE (Redundant/Old)

### Test Files (Move to archive/)
```
test_1round_aes.py            → archive/tests/
test_real_aes_certification.py → archive/tests/
quick_aes_test.py             → archive/tests/
verify_aes_key.py             → archive/tests/
interactive_aes_cracker.py    → archive/tests/ (old version, will recreate clean one)
```

### Old Documentation (Already moved)
```
docs/archive/
├── BUG_ANALYSIS_WHY_KEY_IS_WRONG.md
├── HONEST_ASSESSMENT.md
├── WHY_AES_CERTIFICATION_IS_SLOW.md
├── WHAT_TO_DO_NOW.md
├── CAN_WE_CRACK_AES.md
├── CAN_WE_CRACK_AES_SUMMARY.md
├── CAN_WE_CRACK_REAL_CRYPTO.md
└── AES_ANALYSIS_RESULTS.md
```

---

## Proposed Clean Structure

```
Quantum_sat/
│
├── README.md                           # Main overview
├── can_we_crack_aes.py                # Main tool
├── requirements.txt                    # Dependencies
│
├── docs/
│   ├── AES_CRACKING_GUIDE.md          # Tutorial
│   ├── FINAL_SUMMARY.md               # Results
│   ├── BREAKTHROUGH_AES_CRACKABLE.md  # Research
│   ├── SPECTRAL_ANALYSIS_EXPLAINED.md # Technical
│   └── archive/                       # Old docs
│
├── src/
│   ├── core/
│   │   ├── quantum_sat_solver.py      # Main solver
│   │   └── integrated_pipeline.py     # Pipeline
│   └── solvers/
│       ├── aes_full_encoder.py        # AES encoding
│       ├── aes_sbox_encoder.py        # S-box
│       ├── aes_mixcolumns_encoder.py  # MixColumns
│       └── structure_aligned_qaoa.py  # QAOA
│
├── experiments/
│   ├── sat_decompose.py               # Decomposition
│   └── sat_undecomposable_quantum.py  # Certification
│
└── archive/
    ├── tests/                         # Old test files
    └── scripts/                       # Old scripts
```

---

## Files Statistics

### Before Cleanup
- Python files: ~50
- Documentation: ~40 MD files
- Total size: Large, confusing

### After Cleanup
- Essential Python: ~10 files
- Documentation: 4 main docs + archive
- Total: Clean, organized

---

## Cleanup Commands

### Move test files to archive
```powershell
cd Quantum_sat
New-Item -ItemType Directory -Path "archive\tests" -Force
Move-Item -Path "test_*.py", "quick_*.py", "verify_*.py" -Destination "archive\tests\" -ErrorAction SilentlyContinue
```

### List remaining files
```powershell
Get-ChildItem -Path "." -Filter "*.py" -Recurse | Where-Object {$_.FullName -notmatch "archive|__pycache__|venv"} | Select-Object FullName
```

---

## Essential Dependencies

```txt
# Core
numpy>=1.20.0
scipy>=1.7.0
networkx>=2.6.0

# SAT Solving
python-sat>=0.1.7

# Quantum
qiskit>=0.39.0
qiskit-aer>=0.11.0

# UI/Progress
tqdm>=4.62.0

# Optional
matplotlib>=3.5.0
jupyter>=1.0.0
```

---

## What Each File Does

### can_we_crack_aes.py
- Interactive tool for AES analysis
- Config options: rounds (1/2/10), cores, methods
- Progress tracking with tqdm
- Outputs k* and analysis time

### quantum_sat_solver.py
- Main solver class with 6 quantum methods
- Routing logic based on k*
- Decomposition integration
- Classical fallback

### aes_full_encoder.py
- Encodes full 10-round AES-128
- Returns (clauses, n_vars, round_keys)
- Uses S-box and MixColumns encoders
- 941,824 clauses total

### sat_decompose.py
- Louvain community detection
- Treewidth decomposition  
- FisherInfo spectral clustering
- Hypergraph bridge breaking

### structure_aligned_qaoa.py
- Structure-Aligned QAOA algorithm
- Coupling matrix construction
- Spectral analysis (optional)
- Backdoor estimation

---

## Maintenance

### Keep Updated
- ✅ can_we_crack_aes.py - main tool
- ✅ Documentation (README, GUIDE, SUMMARY)
- ✅ Core solvers

### Can Archive
- Test files once validated
- Experiment scripts
- Old documentation
- Redundant examples

### Must Test After Changes
- AES encoding (941k clauses)
- k* estimation (should get ~105)
- Decomposition (Louvain + Treewidth)
- Progress bars (tqdm)

---

Last Updated: November 3, 2025
