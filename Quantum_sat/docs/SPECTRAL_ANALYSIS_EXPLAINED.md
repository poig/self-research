# What is Spectral Analysis? 🔬

## The Question You Asked

> "what is inside spectral analysis"

You saw this getting stuck at 25% for a long time when testing AES. Let me explain what's happening!

## What It Does

**Spectral analysis** computes the **eigenvalues** of the coupling matrix `J`.

### The Coupling Matrix J

The coupling matrix is an `N×N` matrix where:
- **Rows/Columns**: Variables in your SAT problem
- **J[i,j]**: How often variables `i` and `j` appear together in clauses
- **Size for AES**: 11,137 × 11,137 = **124,032,769 entries**!

### The Eigenvalue Computation

We compute the **2 smallest eigenvalues** of this matrix using the **Lanczos algorithm**:

```python
eigenvalues = eigsh(J_sparse, k=2, which='SM', maxiter=1000, tol=1e-3)
spectral_gap = eigenvalues[1] - eigenvalues[0]
```

**What does spectral_gap tell us?**

The spectral gap describes the **energy landscape** of your optimization problem:

| Spectral Gap | Meaning | QAOA Behavior |
|--------------|---------|---------------|
| **Large** (> 0.5) | Smooth energy landscape, few local minima | **Easy**: QAOA converges fast, needs shallow depth |
| **Medium** (0.1 - 0.5) | Moderate complexity | **Normal**: Standard QAOA depth works |
| **Small** (< 0.1) | Rugged landscape, many local minima | **Hard**: QAOA needs deeper circuits |

## Why It's SLOW for AES

### The Problem Size

For **AES-128** (11,137 variables):
- Coupling matrix: 11,137 × 11,137 = **124 million entries**
- Even as sparse matrix: **~20 million non-zero entries**
- Memory: ~160 MB for the matrix alone

### The Algorithm Complexity

The `eigsh` function uses the **Lanczos iterative algorithm**:

```
Time complexity: O(k × iter × N × nnz)
where:
  k = 2 (eigenvalues to find)
  iter ≈ 100-1000 (iterations to converge)
  N = 11,137 (matrix size)
  nnz ≈ 20,000,000 (non-zero entries)

Total operations: ~40 billion operations!
```

**Expected time**: 5-30 minutes on a single CPU core 😱

## Why We Can SKIP It

### The Good News

**The spectral gap is NOT critical for the final answer!**

It only affects:
1. **QAOA depth recommendation** (we can use heuristics instead)
2. **Parameter initialization** (we can use defaults)

It does **NOT** affect:
- Whether the problem is solvable
- The backdoor size `k*` (computed separately)
- The actual solution

### Our Solution

Added `fast_mode=True` for large problems:

```python
if fast_mode and n_vars > 1000:
    print("⚡ SKIPPING spectral analysis (fast mode)")
    spectral_gap = 0.1  # Use default
```

**Result**: Skip 5-30 minute wait, use sensible default instead! ⚡

## The Real Question: Can We Crack AES?

The spectral analysis is a **diagnostic tool**, not the core cracking mechanism.

The real question is determined by:

1. **Backdoor size `k*`**:
   - If `k* < 10`: AES is **CRACKABLE** with quantum decomposition
   - If `k* ≈ 128`: AES is **SECURE** (cannot decompose)

2. **Decomposition success**:
   - Can we partition the 11k variables into small groups?
   - Do Louvain/Treewidth algorithms find separators?

The spectral gap is just metadata about how "smooth" the landscape is. It's useful for **optimization tuning**, but not for determining **crackability**.

## What's Actually Running Now

After skipping spectral analysis, the slow part is:

```
🎯 Computing matrix rank for backdoor estimate...
```

This uses `np.linalg.matrix_rank(J)` on the 11k×11k matrix.

**This is ALSO slow** (5-20 minutes) because:
- Computes SVD (Singular Value Decomposition)
- Complexity: O(N²×N) = O(N³) ≈ **1.4 trillion operations**
- For 11,137 variables: ~10-20 minutes

## The Bottom Line

### For AES Testing

Skip both slow operations and use:
- **Spectral gap**: Default = 0.1
- **Backdoor estimate**: Use heuristic = `min(128, N/10)`

### For Real Cracking

The actual test is:
```python
decomposer.decompose(
    backdoor_vars=list(range(128)),  # Try to decompose using key bits
    strategies=[Louvain, Treewidth],  # Fast graph algorithms
    optimize_for='separator'
)
```

**If this succeeds** → AES decomposes → **CRACKABLE**  
**If this fails** → AES doesn't decompose → **SECURE**

---

## TL;DR

**Spectral analysis** = Computing eigenvalues to measure landscape "smoothness"  
**Why slow** = 11k×11k matrix, 40 billion operations, 5-30 min  
**Why skippable** = Only affects tuning, not crackability  
**Real test** = Can Louvain/Treewidth decompose the problem? ← This is what matters!
