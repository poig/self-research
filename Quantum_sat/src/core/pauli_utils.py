"""
Pauli Expansion Utilities for SAT Hamiltonian Construction
===========================================================

This module provides efficient clause-to-Pauli expansion for SAT problems,
enabling SparsePauliOp construction without 4^N complexity.

Key Idea:
---------
Each k-literal clause projects onto states violating that clause.
The projector can be expanded as a product of single-qubit projectors:

- Literal x_i (positive):  violated when x_i=0  →  |0⟩⟨0| = (I + Z_i)/2
- Literal ¬x_i (negative): violated when x_i=1  →  |1⟩⟨1| = (I - Z_i)/2
- Variables not in clause: factor is I (identity, no effect)

Expanding the product (I ± Z_i)/2 for each variable in the clause yields
2^k Pauli terms (for k-literal clause), each with coefficient ±1/(2^k).

For 3-SAT: each clause → 8 Pauli terms (manageable!)
For m clauses: total ~8m Pauli terms (before deduplication)

CRITICAL: Bit Ordering Convention
----------------------------------
We use STANDARD COMPUTATIONAL BASIS ordering:
- Variable index i (1-indexed in SAT) → qubit position i-1 (0-indexed)
- State |x⟩ encoded as integer: x = x₀ + 2·x₁ + 4·x₂ + ... = Σᵢ xᵢ·2^i
- Bit extraction: bit_i = (state >> i) & 1
- Pauli string: 'IZI' means I on qubit 0, Z on qubit 1, I on qubit 2

This matches:
- numpy bit operations: (state >> pos) & 1
- Qiskit little-endian convention (rightmost char = highest qubit number in some contexts)
- Standard binary representation

IMPORTANT: Keep this convention consistent across:
1. clause_to_paulis() - Pauli string construction
2. compute_diagonal_vectorized() - state enumeration
3. Any matrix/vector indexing

Complexity:
-----------
- Traditional matrix-to-Pauli: O(4^N) - iterate all N-qubit Pauli strings
- This approach: O(m * 2^k) where k=clause size (~3 for 3-SAT)
- Memory: O(m * 2^k) unique Pauli strings (typically ~few thousand for realistic instances)

Example:
--------
Clause (x1 ∨ x2 ∨ x3) with N=3 variables:
Violated when x1=0, x2=0, x3=0 (binary: state=0 = 000₂)

Projector: |0⟩⟨0|_1 ⊗ |0⟩⟨0|_2 ⊗ |0⟩⟨0|_3
         = [(I+Z1)/2] ⊗ [(I+Z2)/2] ⊗ [(I+Z3)/2]
         = (1/8) * [III + IIZ + IZI + IZZ + ZII + ZIZ + ZZI + ZZZ]
         
8 terms with coefficient 1/8 each.
"""

from itertools import product
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    SparsePauliOp = None


# ============================================================================
# Fast Walsh-Hadamard Transform (FWHT) for efficient diagonal → Z-Pauli conversion
# ============================================================================

def fwht_inplace(a: np.ndarray):
    """
    In-place Fast Walsh-Hadamard Transform on 1D array of length 2^n.
    
    This is the key optimization that replaces O(2^N × 2^N) naive transform
    with O(N × 2^N) algorithm, essential for N > 16.
    
    Args:
        a: 1D numpy array of length 2^n (will be modified in-place)
    
    Raises:
        ValueError: If length is not a power of two
    
    Algorithm:
        Standard radix-2 Cooley-Tukey style decimation in time.
        For each stage h = 1, 2, 4, ..., 2^(n-1):
            Apply butterfly operations: (x, y) → (x+y, x-y)
    
    Complexity: O(N × 2^N) arithmetic operations, excellent cache locality
    """
    n = a.shape[0]
    if n & (n - 1):
        raise ValueError(f"FWHT requires power-of-two length, got {n}")
    
    h = 1
    while h < n:
        # Process blocks of size 2*h
        for i in range(0, n, h * 2):
            # Apply butterfly to pairs (j, j+h)
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def diag_to_zpauli_coeffs(diag: np.ndarray, prune_threshold: float = 1e-12) -> Dict[str, float]:
    """
    Convert diagonal (length 2^N) to Z-basis Pauli coefficients using FWHT.
    
    This is dramatically faster than the naive O(2^N × 2^N) approach used in
    _matrix_to_sparse_pauli. Use this when you have a diagonal Hamiltonian
    and need the full Pauli decomposition.
    
    Args:
        diag: Diagonal elements of Hamiltonian (length 2^N)
        prune_threshold: Drop Pauli terms with |coeff| < threshold
    
    Returns:
        Dict mapping Pauli strings (e.g., 'IIZZI') to coefficients
    
    Example:
        >>> diag = np.array([0., 1., 1., 2.])  # N=2, dim=4
        >>> pauli_dict = diag_to_zpauli_coeffs(diag)
        >>> # Returns Z-basis decomposition of diagonal operator
    
    Complexity: O(N × 2^N + 2^N) vs naive O(4^N) for full Pauli decomposition
    
    Mathematical Background:
        For diagonal H with H[i,i] = h_i, we can write:
        H = Σ_{z∈{0,1}^N} c_z Z_z  where Z_z = ⊗ᵢ (Z if z_i=1 else I)
        
        The coefficients are given by Walsh-Hadamard transform:
        c_z = (1/2^N) Σᵢ h_i (-1)^(i·z)  where i·z = XOR of bits
        
        FWHT computes this in O(N × 2^N) instead of O(2^N × 2^N).
    """
    dim = diag.shape[0]
    n_vars = int(np.log2(dim))
    
    if 2**n_vars != dim:
        raise ValueError(f"Diagonal length {dim} is not a power of two")
    
    # Copy array (FWHT modifies in-place)
    arr = diag.astype(np.float64).copy()
    
    # Apply Fast Walsh-Hadamard Transform
    fwht_inplace(arr)
    
    # Normalize: c_z = (FWHT output) / 2^N
    arr /= float(dim)
    
    # Build Pauli dictionary, pruning small coefficients
    pauli_dict = {}
    for z_idx in range(dim):
        coeff = arr[z_idx]
        if abs(coeff) <= prune_threshold:
            continue
        
        # Build Pauli string: bit i = 1 → Z at position i, bit i = 0 → I
        # Follows our bit-ordering convention: bit 0 = rightmost qubit
        pauli_str = ''.join('Z' if ((z_idx >> pos) & 1) else 'I' 
                           for pos in range(n_vars))
        pauli_dict[pauli_str] = coeff
    
    return pauli_dict


def clause_to_paulis(clause: Tuple[int, ...], n_vars: int) -> Tuple[List[str], List[float]]:
    """
    Expand a single clause into Pauli strings and coefficients.
    
    Args:
        clause: Tuple of literals (e.g., (1, -2, 3) means x1 ∨ ¬x2 ∨ x3)
        n_vars: Total number of variables in the problem
    
    Returns:
        pauli_strings: List of Pauli strings (e.g., ['III', 'IIZ', ...])
        coeffs: List of coefficients corresponding to each Pauli string
    
    Example:
        >>> clause_to_paulis((1, 2), 3)
        # Clause (x1 ∨ x2) with 3 variables
        # Violated when x1=0 AND x2=0 (x3 can be anything)
        # Returns 4 Pauli strings with coefficients 0.25 each
    """
    # Build literal map: variable index → sign
    # sign = +1 means want |0⟩⟨0| = (I + Z)/2
    # sign = -1 means want |1⟩⟨1| = (I - Z)/2
    literal_map = {}
    for lit in clause:
        v = abs(lit) - 1  # Convert to 0-indexed
        if 0 <= v < n_vars:
            if lit > 0:
                literal_map[v] = +1  # Positive literal → violated when 0
            else:
                literal_map[v] = -1  # Negative literal → violated when 1
    
    if not literal_map:
        # Empty clause or all literals out of range → treat as unsatisfiable
        # Return identity with coefficient 1.0
        return ['I' * n_vars], [1.0]
    
    # Get positions that matter (sorted for consistency)
    clause_positions = sorted(literal_map.keys())
    k = len(clause_positions)
    
    # Iterate over 2^k choices: for each variable in clause, pick I or Z term
    # from the expansion (I ± Z)/2
    pauli_strings = []
    coeffs = []
    
    for pattern in product([0, 1], repeat=k):
        # Initialize: all qubits start as 'I'
        pauli = ['I'] * n_vars
        coeff = 1.0
        
        for j, pos in enumerate(clause_positions):
            sign = literal_map[pos]  # +1 for (I+Z)/2, -1 for (I-Z)/2
            
            if pattern[j] == 0:
                # Choose 'I' term from (I ± Z)/2
                # Contributes factor of 1/2
                coeff *= 0.5
                # pauli[pos] stays 'I'
            else:
                # Choose 'Z' term from (I ± Z)/2
                # Contributes factor of (sign)/2
                coeff *= (sign * 0.5)
                pauli[pos] = 'Z'
        
        pauli_strings.append(''.join(pauli))
        coeffs.append(coeff)
    
    return pauli_strings, coeffs


def clauses_to_pauli_dict(clauses: List[Tuple[int, ...]], n_vars: int) -> Dict[str, float]:
    """
    Convert multiple clauses to a dictionary of Pauli strings and coefficients.
    Automatically deduplicates and sums coefficients for identical Pauli strings.
    
    Args:
        clauses: List of clauses, each clause is tuple of literals
        n_vars: Total number of variables
    
    Returns:
        Dictionary mapping Pauli string → total coefficient
    
    Example:
        >>> clauses = [(1, 2, 3), (-1, 2, -3)]
        >>> pauli_dict = clauses_to_pauli_dict(clauses, 3)
        >>> len(pauli_dict)  # Should be ≤ 16 (8 per clause, some may cancel)
    """
    pauli_dict = defaultdict(float)
    
    for clause in clauses:
        strings, coeffs = clause_to_paulis(clause, n_vars)
        for pauli_str, coeff in zip(strings, coeffs):
            pauli_dict[pauli_str] += coeff
    
    # Remove terms with near-zero coefficients (numerical cancellation)
    pauli_dict = {k: v for k, v in pauli_dict.items() if abs(v) > 1e-12}
    
    return pauli_dict


def create_sparse_pauli_op(clauses: List[Tuple[int, ...]], n_vars: int):
    """
    Create a Qiskit SparsePauliOp representing H = Σ_c Π_c.
    
    Uses modern Qiskit API (from_list) with fallback for older versions.
    
    Args:
        clauses: List of clauses (tuples of literals)
        n_vars: Number of variables
    
    Returns:
        SparsePauliOp instance representing the Hamiltonian
    
    Raises:
        ImportError: If Qiskit is not installed
    
    Example:
        >>> clauses = [(1, 2, 3), (-1, 2, -3), (1, -2, 3)]
        >>> H_op = create_sparse_pauli_op(clauses, 3)
        >>> print(H_op)
        # SparsePauliOp with ~24 terms (3 clauses × 8 terms, some deduplicated)
    
    Note:
        Handles empty clause list by returning zero operator.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is not installed. Install with: pip install qiskit"
        )
    
    # Get deduplicated Pauli dictionary
    pauli_dict = clauses_to_pauli_dict(clauses, n_vars)
    
    if not pauli_dict:
        # Empty Hamiltonian → return zero operator
        return SparsePauliOp(['I' * n_vars], [0.0])
    
    # Convert to list format: [(label, coeff), ...]
    pauli_list_with_coeffs = [(label, float(coeff)) for label, coeff in pauli_dict.items()]
    
    # Modern Qiskit API: SparsePauliOp.from_list([(label, coeff), ...])
    # Fallback for older versions: SparsePauliOp(labels, coeffs)
    try:
        return SparsePauliOp.from_list(pauli_list_with_coeffs)
    except AttributeError:
        # Older Qiskit versions don't have from_list
        pauli_list = list(pauli_dict.keys())
        coeffs = list(pauli_dict.values())
        return SparsePauliOp(pauli_list, coeffs)


def validate_hamiltonian_trace(clauses: List[Tuple[int, ...]], n_vars: int, 
                                H_matrix: np.ndarray = None,
                                H_op = None,
                                tolerance: float = 1e-10) -> bool:
    """
    Validate that Hamiltonian trace matches expected formula: trace(H) = Σ_c 2^(N-|c|)
    
    This is a critical sanity check for Hamiltonian construction. Each clause contributes
    a projector with trace = 2^(N-k) where k is the clause size (number of literals).
    
    Args:
        clauses: List of clauses (tuples of literals)
        n_vars: Number of variables
        H_matrix: Optional explicit matrix representation (numpy array)
        H_op: Optional SparsePauliOp representation (will be converted to matrix)
        tolerance: Numerical tolerance for trace comparison
    
    Returns:
        True if trace matches expected value within tolerance
    
    Raises:
        ValueError: If trace doesn't match (indicates bug in Hamiltonian construction)
        AssertionError: If neither H_matrix nor H_op provided
    
    Example:
        >>> clauses = [(1, 2, 3), (-1, 2)]  # One 3-literal, one 2-literal clause
        >>> H_matrix = build_hamiltonian_matrix(clauses, 3)
        >>> validate_hamiltonian_trace(clauses, 3, H_matrix=H_matrix)
        True
        >>> # Expected trace = 2^(3-3) + 2^(3-2) = 1 + 2 = 3
    
    Mathematical Background:
        Each k-literal clause projects onto violating assignments:
        P_c = Π_{i∈c} (I ± Z_i)/2
        
        trace(P_c) = product of traces: trace((I±Z)/2)^k × I^(N-k)
                   = (1)^k × 2^(N-k)  [since trace(I±Z) = 2, trace(I) = 2]
                   = 2^(N-k)
        
        For Hamiltonian H = Σ_c P_c:
        trace(H) = Σ_c trace(P_c) = Σ_c 2^(N-|c|)
    """
    assert H_matrix is not None or H_op is not None, "Must provide either H_matrix or H_op"
    
    # Compute expected trace from clause structure
    expected_trace = sum(2**(n_vars - len(clause)) for clause in clauses)
    
    # Get actual trace
    if H_matrix is not None:
        actual_trace = np.trace(H_matrix)
    else:
        # Convert SparsePauliOp to matrix
        H_matrix = H_op.to_matrix()
        actual_trace = np.trace(H_matrix)
    
    # Check if traces match
    trace_diff = abs(actual_trace - expected_trace)
    
    if trace_diff > tolerance:
        raise ValueError(
            f"Hamiltonian trace validation FAILED!\n"
            f"  Expected trace: {expected_trace}\n"
            f"  Actual trace:   {actual_trace}\n"
            f"  Difference:     {trace_diff}\n"
            f"  Tolerance:      {tolerance}\n"
            f"This indicates a bug in Hamiltonian construction."
        )
    
    return True


def verify_pauli_expansion(clauses: List[Tuple[int, ...]], n_vars: int, 
                          reference_matrix: np.ndarray = None,
                          tolerance: float = 1e-10) -> bool:
    """
    Verify that Pauli expansion matches direct matrix construction.
    
    Args:
        clauses: List of clauses
        n_vars: Number of variables
        reference_matrix: Optional reference matrix to compare against
        tolerance: Numerical tolerance for comparison
    
    Returns:
        True if Pauli expansion matches reference (or if constructed correctly)
    
    Example:
        >>> from quantum_structure_analyzer import QuantumStructureAnalyzer
        >>> qsa = QuantumStructureAnalyzer(use_ml=False)
        >>> clauses = [(1, 2, 3), (-1, 2, -3)]
        >>> H_ref = qsa._build_hamiltonian(clauses, 3)
        >>> verify_pauli_expansion(clauses, 3, H_ref.to_matrix())
        True
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available, cannot verify SparsePauliOp")
        return False
    
    # Create Pauli operator
    H_pauli = create_sparse_pauli_op(clauses, n_vars)
    H_pauli_matrix = H_pauli.to_matrix()
    
    if reference_matrix is not None:
        # Compare against provided reference
        return np.allclose(H_pauli_matrix, reference_matrix, atol=tolerance)
    else:
        # Just check that it's Hermitian and diagonal (for projector Hamiltonians)
        is_hermitian = np.allclose(H_pauli_matrix, H_pauli_matrix.conj().T, atol=tolerance)
        # Check if diagonal (off-diagonal elements are zero)
        dim = H_pauli_matrix.shape[0]
        off_diag = H_pauli_matrix - np.diag(np.diag(H_pauli_matrix))
        is_diagonal = np.allclose(off_diag, 0.0, atol=tolerance)
        
        return is_hermitian and is_diagonal


def get_expansion_stats(clauses: List[Tuple[int, ...]], n_vars: int) -> Dict[str, int]:
    """
    Get statistics about the Pauli expansion without creating the operator.
    Useful for estimating memory requirements.
    
    Args:
        clauses: List of clauses
        n_vars: Number of variables
    
    Returns:
        Dictionary with statistics:
        - 'num_clauses': Number of clauses
        - 'theoretical_terms': m * 2^k (before deduplication)
        - 'unique_terms': Number of unique Pauli strings (after deduplication)
        - 'avg_coeff': Average absolute coefficient value
        - 'max_coeff': Maximum absolute coefficient
    """
    pauli_dict = clauses_to_pauli_dict(clauses, n_vars)
    
    coeffs_abs = [abs(c) for c in pauli_dict.values()]
    
    # Estimate theoretical terms (assuming 3-SAT)
    avg_clause_size = np.mean([len(c) for c in clauses])
    theoretical_terms = len(clauses) * (2 ** avg_clause_size)
    
    stats = {
        'num_clauses': len(clauses),
        'theoretical_terms': int(theoretical_terms),
        'unique_terms': len(pauli_dict),
        'avg_coeff': np.mean(coeffs_abs) if coeffs_abs else 0.0,
        'max_coeff': np.max(coeffs_abs) if coeffs_abs else 0.0,
        'compression_ratio': theoretical_terms / len(pauli_dict) if pauli_dict else 1.0
    }
    
    return stats


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Pauli Expansion Utility - Example Usage")
    print("="*70)
    
    # Example 1: Single clause expansion
    print("\n1. Single Clause Expansion")
    print("-" * 70)
    clause = (1, 2, 3)  # x1 ∨ x2 ∨ x3
    n_vars = 3
    
    pauli_strs, coeffs = clause_to_paulis(clause, n_vars)
    print(f"Clause: {clause} (x1 ∨ x2 ∨ x3)")
    print(f"Expands to {len(pauli_strs)} Pauli terms:")
    for ps, c in zip(pauli_strs, coeffs):
        print(f"  {ps}: {c:+.4f}")
    
    # Example 2: Multiple clauses with deduplication
    print("\n2. Multiple Clauses with Deduplication")
    print("-" * 70)
    clauses = [(1, 2, 3), (-1, 2, -3), (1, -2, 3)]
    pauli_dict = clauses_to_pauli_dict(clauses, n_vars)
    
    print(f"Clauses: {clauses}")
    print(f"Total unique Pauli terms: {len(pauli_dict)}")
    print(f"Top 5 terms by coefficient:")
    sorted_terms = sorted(pauli_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for ps, c in sorted_terms[:5]:
        print(f"  {ps}: {c:+.6f}")
    
    # Example 3: Expansion statistics
    print("\n3. Expansion Statistics")
    print("-" * 70)
    stats = get_expansion_stats(clauses, n_vars)
    print(f"Number of clauses: {stats['num_clauses']}")
    print(f"Theoretical terms (before dedup): {stats['theoretical_terms']}")
    print(f"Unique terms (after dedup): {stats['unique_terms']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Average |coefficient|: {stats['avg_coeff']:.6f}")
    print(f"Max |coefficient|: {stats['max_coeff']:.6f}")
    
    # Example 4: Create SparsePauliOp (if Qiskit available)
    if QISKIT_AVAILABLE:
        print("\n4. Qiskit SparsePauliOp Creation")
        print("-" * 70)
        H_op = create_sparse_pauli_op(clauses, n_vars)
        print(f"Created SparsePauliOp with {len(H_op)} terms")
        print(f"Hamiltonian:\n{H_op}")
        
        # Convert to matrix and check properties
        H_matrix = H_op.to_matrix()
        print(f"\nMatrix shape: {H_matrix.shape}")
        print(f"Is Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}")
        print(f"Diagonal elements: {np.diag(H_matrix).real}")
        print(f"Ground state energy: {np.min(np.diag(H_matrix).real):.6f}")
    else:
        print("\n4. Qiskit SparsePauliOp Creation")
        print("-" * 70)
        print("Qiskit not installed - skipping SparsePauliOp creation")
        print("Install with: pip install qiskit")
    
    print("\n" + "="*70)
    print("✅ All examples completed successfully!")
    print("="*70)
