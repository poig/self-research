# BQP ≠ NP: Lean Formalization via Dynamical Lie Algebras

**Goal:** Prove (or refute) that quantum computers cannot efficiently solve NP-complete problems.

## Project Structure

```
BQP_NP_Lean/
├── BQP_NP.lean              # Main theorem statement
├── BQP_NP/
│   ├── Basic/
│   │   ├── LieAlgebra.lean  # DLA, Killing form, spectral gap
│   │   └── PauliOperators.lean  # Operator sparsity
│   ├── Year1/
│   │   └── SpectralGap.lean # Conjecture 1: λ_min ≤ e^{-cN}
│   ├── Year2/               # (TODO) Anderson localization
│   └── Year3/               # (TODO) Main separation theorem
├── lakefile.toml            # Build config (Mathlib + QuantumInfo)
└── lean-toolchain           # Lean 4.12.0
```

## The Three Conjectures

1. **Spectral Gap Collapse** (Year 1): For NP-hard Hamiltonians, the Killing form has λ_min ≤ e^{-cN}
2. **Curvature Explosion** (Year 2): Geodesic curvature κ ≤ -e^{αN}
3. **Anderson Localization** (Year 3): IPR ≥ e^{βN} (information bottleneck)

## Dependencies

- [Mathlib](https://github.com/leanprover-community/mathlib4) - Standard math library
- [Lean-QuantumInfo](https://github.com/Timeroot/Lean-QuantumInfo) - Quantum info definitions (entropy, states)

## Build

```bash
cd BQP_NP_Lean
lake update  # Download dependencies
lake build   # Compile project

# Cancel current build (Ctrl+C)
# Then run:
lake exe cache get
# After cache downloads, rebuild (should be instant for Mathlib):
lake build
```

## Status

| Component | Status |
|-----------|--------|
| Project structure | ✅ Done |
| Basic definitions | ⚠️ Placeholder (`sorry`) |
| Year 1 theorem | ⚠️ Stated, not proven |
| Year 2/3 | ❌ Not started |
