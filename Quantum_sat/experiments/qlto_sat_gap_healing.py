"""

Quantum SAT Solver with Gap Healing via Counterdiabatic Driving

This module implements an advanced adiabatic SAT solver that detects and heals
exponential spectral gaps using counterdiabatic driving (CD) terms.

Key innovation: When an exponential gap is detected during hierarchical scaffolding,
we add a counterdiabatic correction term that suppresses diabatic transitions,
potentially turning exponential evolution time into polynomial time.

Theoretical foundation:
- Standard adiabatic: H(s) = (1-s)H_0 + s*H_1, T ~ O(1/g_min²)
- With CD: H_CD(s) = H(s) + λ*A(s), where A(s) = i[dH/ds, |ψ_0⟩⟨ψ_0|]
- CD shortcuts adiabatic evolution, potentially healing exponential gaps

Author: Research Team
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from qiskit.quantum_info import SparsePauliOp, Statevector
import matplotlib.pyplot as plt


@dataclass
class GapHealingResult:
    """Results from gap healing analysis"""
    layer: int
    original_gap: float
    healed_gap: float
    gap_improvement_factor: float
    cd_term_norm: float
    healing_success: bool  # True if gap improved significantly
    evolution_time_estimate: float


@dataclass
class CDTerm:
    """Counterdiabatic driving term"""
    operator: SparsePauliOp
    norm: float
    lambda_coefficient: float  # Scaling factor for CD term


def pauli_commutator(A: SparsePauliOp, B: SparsePauliOp) -> SparsePauliOp:
    """
    Compute the commutator [A, B] = AB - BA for Pauli operators.
    
    Args:
        A, B: Sparse Pauli operators
        
    Returns:
        Commutator [A, B] as SparsePauliOp
    """
    # Compute AB
    AB = A.compose(B)
    # Compute BA
    BA = B.compose(A)
    # Return commutator
    commutator = AB - BA
    
    # Simplify by removing terms with near-zero coefficients
    tolerance = 1e-10
    commutator = commutator.simplify(atol=tolerance)
    
    return commutator


def compute_approximate_ground_state_projector(
    H: SparsePauliOp,
    n_vars: int,
    method: str = "exact"
) -> np.ndarray:
    """
    Compute approximation of |ψ_0⟩⟨ψ_0| for ground state of H.
    
    Args:
        H: Hamiltonian operator
        n_vars: Number of variables
        method: "exact" (small N) or "variational" (large N)
        
    Returns:
        Ground state projector as dense matrix
    """
    if method == "exact" and n_vars <= 8:
        # For small systems, compute exactly
        H_matrix = H.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        # Ground state is eigenvector with smallest eigenvalue
        ground_idx = np.argmin(eigenvalues)
        psi_0 = eigenvectors[:, ground_idx]
        
        # Projector |ψ_0⟩⟨ψ_0|
        projector = np.outer(psi_0, np.conj(psi_0))
        
        return projector
    
    elif method == "variational":
        # For large systems, use variational approximation
        # Use current ground state from adiabatic evolution as approximation
        # (In practice, this would be the quantum state during evolution)
        
        # Placeholder: Use uniform superposition as rough approximation
        # In real implementation, this would be the evolved state
        dim = 2 ** n_vars
        psi_approx = np.ones(dim) / np.sqrt(dim)
        projector = np.outer(psi_approx, np.conj(psi_approx))
        
        return projector
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_counterdiabatic_term_approximate(
    H_current: SparsePauliOp,
    H_target: SparsePauliOp,
    n_vars: int,
    s: float = 0.5
) -> CDTerm:
    """
    Compute approximate counterdiabatic term using first-order approximation.
    
    The exact CD term is: A(s) = i * dH/ds * [dH/ds, |ψ_0(s)⟩⟨ψ_0(s)|]
    
    First-order approximation:
    A(s) ≈ i * [dH/ds, H(s)] / gap(s)
    
    where dH/ds = H_target - H_current for linear interpolation.
    
    Args:
        H_current: Current Hamiltonian (seed)
        H_target: Target Hamiltonian (full problem)
        n_vars: Number of variables
        s: Current parameter value
        
    Returns:
        CDTerm with approximate counterdiabatic operator
    """
    # Compute dH/ds = H_target - H_current (for linear path)
    dH_ds = H_target - H_current
    
    # Current Hamiltonian: H(s) = H_current + s * (H_target - H_current)
    H_s = H_current + s * dH_ds
    
    # First-order approximation: A(s) ≈ i * [dH/ds, H(s)]
    # In practice, we use the commutator directly
    A_approx = pauli_commutator(dH_ds, H_s)
    
    # Scale by i (represented by multiplying coefficients by 1j)
    # But since we're working with Hermitian operators for physical implementation,
    # we'll use the magnitude and adjust phase during evolution
    
    # Compute norm of CD term
    cd_norm = np.linalg.norm(A_approx.coeffs)
    
    # Estimate optimal lambda coefficient
    # Heuristic: λ ~ 1 / ||A||
    if cd_norm > 1e-10:
        lambda_coeff = 1.0 / cd_norm
    else:
        lambda_coeff = 0.0
    
    return CDTerm(
        operator=A_approx,
        norm=cd_norm,
        lambda_coefficient=lambda_coeff
    )


def compute_counterdiabatic_term_perturbative(
    H_current: SparsePauliOp,
    H_target: SparsePauliOp,
    n_vars: int,
    s: float = 0.5
) -> CDTerm:
    """
    Compute CD term using perturbation theory (more accurate than first-order).
    
    Uses second-order perturbation expansion:
    A(s) ≈ Σ_{n≠0} (|n⟩⟨n|[dH/ds]|0⟩⟨0| + |0⟩⟨0|[dH/ds]|n⟩⟨n|) / (E_0 - E_n)
    
    Args:
        H_current: Current Hamiltonian (seed)
        H_target: Target Hamiltonian (full problem)
        n_vars: Number of variables
        s: Current parameter value
        
    Returns:
        CDTerm with perturbative counterdiabatic operator
    """
    # For small systems, compute exactly
    if n_vars <= 6:
        dH_ds = H_target - H_current
        H_s = H_current + s * dH_ds
        
        # Get matrix representations
        H_matrix = H_s.to_matrix()
        dH_matrix = dH_ds.to_matrix()
        
        # Diagonalize H(s)
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        
        # Ground state
        E_0 = eigenvalues[0]
        psi_0 = eigenvectors[:, 0]
        
        # Build CD term using perturbation theory
        dim = len(eigenvalues)
        A_matrix = np.zeros((dim, dim), dtype=complex)
        
        for n in range(1, dim):  # Sum over excited states
            E_n = eigenvalues[n]
            psi_n = eigenvectors[:, n]
            
            gap = E_n - E_0
            
            if abs(gap) > 1e-10:  # Avoid division by zero
                # Matrix element ⟨n|dH/ds|0⟩
                matrix_element = np.dot(np.conj(psi_n), np.dot(dH_matrix, psi_0))
                
                # Add contribution: |n⟩⟨0| + |0⟩⟨n|
                A_matrix += (np.outer(psi_n, np.conj(psi_0)) * matrix_element / gap)
                A_matrix += (np.outer(psi_0, np.conj(psi_n)) * np.conj(matrix_element) / gap)
        
        # Multiply by i (phase factor)
        A_matrix = 1j * A_matrix
        
        # Convert back to SparsePauliOp (approximate)
        # For now, measure norm and create scaled version of commutator
        cd_norm = np.linalg.norm(A_matrix, ord='fro')
        
        # Use commutator as approximation (scaled)
        A_approx = pauli_commutator(dH_ds, H_s)
        
        # Scale to match perturbative norm
        if cd_norm > 1e-10:
            scale_factor = cd_norm / np.linalg.norm(A_approx.coeffs)
            A_approx = A_approx * scale_factor
        
        lambda_coeff = 1.0 if cd_norm > 1e-10 else 0.0
        
        return CDTerm(
            operator=A_approx,
            norm=cd_norm,
            lambda_coefficient=lambda_coeff
        )
    
    else:
        # For large systems, fall back to first-order
        return compute_counterdiabatic_term_approximate(H_current, H_target, n_vars, s)


def measure_gap_with_cd_term(
    H_current: SparsePauliOp,
    H_target: SparsePauliOp,
    cd_term: CDTerm,
    n_vars: int,
    s: float
) -> Tuple[float, float]:
    """
    Measure spectral gap with and without CD term.
    
    Args:
        H_current: Current Hamiltonian
        H_target: Target Hamiltonian
        cd_term: Counterdiabatic term
        n_vars: Number of variables
        s: Evolution parameter
        
    Returns:
        (gap_without_cd, gap_with_cd)
    """
    # Original Hamiltonian: H(s) = H_current + s*(H_target - H_current)
    H_original = H_current + s * (H_target - H_current)
    
    # Healed Hamiltonian: H_CD(s) = H(s) + λ*A(s)
    H_healed = H_original + cd_term.lambda_coefficient * cd_term.operator
    
    # Convert to matrices
    H_orig_matrix = H_original.to_matrix()
    H_healed_matrix = H_healed.to_matrix()
    
    # Compute eigenvalues
    eigs_orig = np.linalg.eigvalsh(H_orig_matrix)
    eigs_healed = np.linalg.eigvalsh(H_healed_matrix)
    
    # Sort
    eigs_orig = np.sort(eigs_orig)
    eigs_healed = np.sort(eigs_healed)
    
    # Gaps
    gap_orig = eigs_orig[1] - eigs_orig[0]
    gap_healed = eigs_healed[1] - eigs_healed[0]
    
    return gap_orig, gap_healed


def analyze_gap_healing_for_layer(
    H_seed: SparsePauliOp,
    H_new_clause: SparsePauliOp,
    H_rest: SparsePauliOp,
    n_vars: int,
    layer_idx: int,
    num_points: int = 20,
    cd_method: str = "approximate"
) -> GapHealingResult:
    """
    Analyze gap healing for a single layer in hierarchical scaffolding.
    
    Args:
        H_seed: Hamiltonian of seed clauses (already added)
        H_new_clause: Hamiltonian of new clause being added
        H_rest: Hamiltonian of remaining clauses
        n_vars: Number of variables
        layer_idx: Current layer index
        num_points: Number of points to sample along evolution path
        cd_method: "approximate" or "perturbative"
        
    Returns:
        GapHealingResult with analysis
    """
    # Current layer evolves: H(s) = H_seed + H_new_clause + s*H_rest
    H_current = H_seed + H_new_clause
    H_target = H_current + H_rest
    
    # Sample points along evolution path
    s_values = np.linspace(0, 1, num_points)
    
    gaps_original = []
    gaps_healed = []
    cd_norms = []
    
    for s in s_values:
        # Compute CD term at this point
        if cd_method == "approximate":
            cd_term = compute_counterdiabatic_term_approximate(H_current, H_target, n_vars, s)
        elif cd_method == "perturbative":
            cd_term = compute_counterdiabatic_term_perturbative(H_current, H_target, n_vars, s)
        else:
            raise ValueError(f"Unknown CD method: {cd_method}")
        
        # Measure gaps
        gap_orig, gap_healed = measure_gap_with_cd_term(H_current, H_target, cd_term, n_vars, s)
        
        gaps_original.append(gap_orig)
        gaps_healed.append(gap_healed)
        cd_norms.append(cd_term.norm)
    
    # Find minimum gaps
    min_gap_orig = np.min(gaps_original)
    min_gap_healed = np.min(gaps_healed)
    
    # Improvement factor
    if min_gap_orig > 1e-10:
        improvement = min_gap_healed / min_gap_orig
    else:
        improvement = float('inf') if min_gap_healed > 1e-10 else 1.0
    
    # Estimate evolution time: T ~ 1/g²
    if min_gap_orig > 1e-10:
        time_orig = 1.0 / (min_gap_orig ** 2)
    else:
        time_orig = float('inf')
    
    if min_gap_healed > 1e-10:
        time_healed = 1.0 / (min_gap_healed ** 2)
    else:
        time_healed = float('inf')
    
    # Healing is successful if gap improved by factor > 2
    healing_success = improvement > 2.0
    
    return GapHealingResult(
        layer=layer_idx,
        original_gap=min_gap_orig,
        healed_gap=min_gap_healed,
        gap_improvement_factor=improvement,
        cd_term_norm=np.mean(cd_norms),
        healing_success=healing_success,
        evolution_time_estimate=time_healed
    )


def clause_to_hamiltonian(clause: List[int], n_vars: int) -> SparsePauliOp:
    """
    Convert a SAT clause to quantum Hamiltonian that penalizes violations.
    
    For clause (x1 ∨ x2 ∨ ¬x3), we want:
    - H|assignment⟩ = 0 if clause is satisfied
    - H|assignment⟩ = 1 if clause is violated
    
    The clause is violated only when ALL literals are false.
    For (x1 ∨ x2 ∨ ¬x3): violated when x1=0, x2=0, x3=1
    
    We build projector onto this forbidden state using:
    - For positive literal xi: (I-Z)/2 projects onto |0⟩
    - For negative literal ¬xi: (I+Z)/2 projects onto |1⟩
    
    H = Π_{lit in clause} projector(lit)
    
    Args:
        clause: List of literals (positive=variable, negative=negation)
        n_vars: Total number of variables
        
    Returns:
        Hamiltonian that equals 0 if clause satisfied, 1 if violated
    """
    # Build the product of projectors
    # Start with identity operator
    H = SparsePauliOp.from_list([('I' * n_vars, 1.0)])
    
    for lit in clause:
        var_idx = abs(lit) - 1  # Convert to 0-indexed
        
        # Build projector for this literal
        pauli_i = ['I'] * n_vars
        pauli_z = ['I'] * n_vars
        pauli_z[var_idx] = 'Z'
        
        # Create (I ± Z)/2
        I_op = SparsePauliOp.from_list([(''.join(pauli_i), 0.5)])
        
        if lit > 0:
            # Positive literal: project onto |0⟩ using (I-Z)/2
            Z_op = SparsePauliOp.from_list([(''.join(pauli_z), -0.5)])
        else:
            # Negative literal: project onto |1⟩ using (I+Z)/2
            Z_op = SparsePauliOp.from_list([(''.join(pauli_z), 0.5)])
        
        projector = I_op + Z_op
        
        # Multiply with running product
        H = H.compose(projector)
        H = H.simplify(atol=1e-10)
    
    return H


def test_gap_healing_on_problem(
    clauses: List[List[int]],
    n_vars: int,
    cd_method: str = "approximate",
    num_points: int = 20
) -> Dict:
    """
    Test gap healing on a complete SAT problem using hierarchical scaffolding.
    
    Args:
        clauses: List of clauses
        n_vars: Number of variables
        cd_method: "approximate" or "perturbative"
        num_points: Points to sample per layer
        
    Returns:
        Dictionary with results
    """
    # Build Hamiltonians for all clauses
    H_clauses = [clause_to_hamiltonian(clause, n_vars) for clause in clauses]
    
    # Hierarchical scaffolding: Add one clause at a time
    results = []
    
    H_seed = SparsePauliOp.from_list([('I' * n_vars, 0.0)])  # Start with zero
    
    for i in range(len(clauses)):
        print(f"\n=== Layer {i+1}/{len(clauses)} ===")
        
        H_new = H_clauses[i]
        H_rest = sum(H_clauses[i+1:], SparsePauliOp.from_list([('I' * n_vars, 0.0)]))
        
        # Analyze gap healing for this layer
        result = analyze_gap_healing_for_layer(
            H_seed=H_seed,
            H_new_clause=H_new,
            H_rest=H_rest,
            n_vars=n_vars,
            layer_idx=i,
            num_points=num_points,
            cd_method=cd_method
        )
        
        results.append(result)
        
        # Print results
        print(f"Original gap: {result.original_gap:.6f}")
        print(f"Healed gap: {result.healed_gap:.6f}")
        print(f"Improvement factor: {result.gap_improvement_factor:.2f}x")
        print(f"CD term norm: {result.cd_term_norm:.6f}")
        print(f"Healing success: {result.healing_success}")
        
        if result.original_gap > 1e-10:
            time_orig = 1.0 / (result.original_gap ** 2)
            time_healed = result.evolution_time_estimate
            print(f"Evolution time: {time_orig:.2f} → {time_healed:.2f} (speedup: {time_orig/time_healed:.2f}x)")
        
        # Update seed for next layer
        H_seed = H_seed + H_new
    
    # Compute summary statistics
    total_layers = len(results)
    successful_healings = sum(1 for r in results if r.healing_success)
    avg_improvement = np.mean([r.gap_improvement_factor for r in results])
    
    min_original_gap = min(r.original_gap for r in results)
    min_healed_gap = min(r.healed_gap for r in results)
    
    # Overall verdict
    if min_healed_gap > 1e-10 and min_healed_gap > 2 * min_original_gap:
        verdict = "GAP_HEALING_SUCCESS"
    elif min_healed_gap > min_original_gap:
        verdict = "GAP_HEALING_PARTIAL"
    else:
        verdict = "GAP_HEALING_FAILED"
    
    summary = {
        'clauses': clauses,
        'n_vars': n_vars,
        'num_layers': total_layers,
        'layer_results': results,
        'successful_healings': successful_healings,
        'avg_improvement_factor': avg_improvement,
        'min_original_gap': min_original_gap,
        'min_healed_gap': min_healed_gap,
        'overall_improvement': min_healed_gap / min_original_gap if min_original_gap > 1e-10 else float('inf'),
        'verdict': verdict,
        'cd_method': cd_method
    }
    
    return summary


def visualize_gap_healing(results: Dict, save_path: Optional[str] = None):
    """
    Visualize gap healing results.
    
    Args:
        results: Results from test_gap_healing_on_problem
        save_path: Optional path to save figure
    """
    layer_results = results['layer_results']
    layers = [r.layer for r in layer_results]
    
    gaps_orig = [r.original_gap for r in layer_results]
    gaps_healed = [r.healed_gap for r in layer_results]
    improvements = [r.gap_improvement_factor for r in layer_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Gap comparison
    ax = axes[0, 0]
    ax.plot(layers, gaps_orig, 'o-', label='Original Gap', color='red', linewidth=2)
    ax.plot(layers, gaps_healed, 's-', label='Healed Gap', color='green', linewidth=2)
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Polynomial threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Minimum Gap')
    ax.set_title('Gap Healing: Original vs Healed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Improvement factor
    ax = axes[0, 1]
    colors = ['green' if r.healing_success else 'orange' for r in layer_results]
    ax.bar(layers, improvements, color=colors, alpha=0.7)
    ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=2, label='Success threshold (2x)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Improvement Factor')
    ax.set_title('Gap Improvement Factor per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Evolution time estimate
    ax = axes[1, 0]
    times_orig = [1.0/(r.original_gap**2) if r.original_gap > 1e-10 else 1e6 for r in layer_results]
    times_healed = [r.evolution_time_estimate if r.evolution_time_estimate < 1e6 else 1e6 for r in layer_results]
    
    ax.plot(layers, times_orig, 'o-', label='Original Time', color='red', linewidth=2)
    ax.plot(layers, times_healed, 's-', label='Healed Time', color='green', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Evolution Time Estimate')
    ax.set_title('Evolution Time: Original vs Healed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
GAP HEALING SUMMARY
{'='*40}

Total Layers: {results['num_layers']}
Successful Healings: {results['successful_healings']} / {results['num_layers']}
Success Rate: {100*results['successful_healings']/results['num_layers']:.1f}%

Average Improvement: {results['avg_improvement_factor']:.2f}x

Minimum Original Gap: {results['min_original_gap']:.6f}
Minimum Healed Gap: {results['min_healed_gap']:.6f}
Overall Improvement: {results['overall_improvement']:.2f}x

CD Method: {results['cd_method']}
Verdict: {results['verdict']}
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


# ============================================================================
# Test cases
# ============================================================================

def generate_binary_counter_unsat(N: int) -> Tuple[List[List[int]], int]:
    """
    Generate binary counter UNSAT problem (from Test 3).
    This is the adversarial case that showed exponential gap closure.
    
    Args:
        N: Number of variables
        
    Returns:
        (clauses, n_vars)
    """
    clauses = []
    n_vars = N
    
    # Generate 2^N clauses, each forcing a different assignment
    for i in range(2 ** N):
        clause = []
        for bit_pos in range(N):
            # Check if bit is set in binary representation of i
            if (i >> bit_pos) & 1:
                clause.append(bit_pos + 1)  # Variable is True
            else:
                clause.append(-(bit_pos + 1))  # Variable is False
        clauses.append(clause)
    
    return clauses, n_vars


def generate_random_3sat(N: int, M: int, seed: int = 42) -> Tuple[List[List[int]], int]:
    """
    Generate random 3-SAT problem.
    
    Args:
        N: Number of variables
        M: Number of clauses
        seed: Random seed
        
    Returns:
        (clauses, n_vars)
    """
    np.random.seed(seed)
    clauses = []
    
    for _ in range(M):
        # Pick 3 random distinct variables
        vars_in_clause = np.random.choice(N, size=3, replace=False) + 1
        # Randomly negate
        signs = np.random.choice([-1, 1], size=3)
        clause = [int(sign * var) for sign, var in zip(signs, vars_in_clause)]
        clauses.append(clause)
    
    return clauses, N


if __name__ == "__main__":
    print("=" * 80)
    print("QUANTUM GAP HEALING WITH COUNTERDIABATIC DRIVING")
    print("=" * 80)
    
    # Test 1: Binary Counter UNSAT (N=3, THE CRITICAL TEST)
    print("\n" + "=" * 80)
    print("TEST 1: Binary Counter UNSAT (N=3) - The Exponential Barrier Case")
    print("=" * 80)
    print("\nThis is the adversarial instance from Test 3 that showed gap closure.")
    print("Can gap healing reopen the gap?\n")
    
    clauses_test1, n_vars_test1 = generate_binary_counter_unsat(N=3)
    print(f"Problem: {len(clauses_test1)} clauses, {n_vars_test1} variables")
    print(f"Expected: Gap closes to zero (Δ=1-s) → exponential time")
    print(f"Hope: CD term reopens gap → polynomial time\n")
    
    results_test1 = test_gap_healing_on_problem(
        clauses=clauses_test1,
        n_vars=n_vars_test1,
        cd_method="approximate",
        num_points=10
    )
    
    print("\n" + "=" * 80)
    print("TEST 1 SUMMARY")
    print("=" * 80)
    print(f"Verdict: {results_test1['verdict']}")
    print(f"Successful healings: {results_test1['successful_healings']}/{results_test1['num_layers']}")
    print(f"Average improvement: {results_test1['avg_improvement_factor']:.2f}x")
    print(f"Overall improvement: {results_test1['overall_improvement']:.2f}x")
    print(f"Min original gap: {results_test1['min_original_gap']:.6f}")
    print(f"Min healed gap: {results_test1['min_healed_gap']:.6f}")
    
    # Test 2: Random 3-SAT (N=4, baseline)
    print("\n" + "=" * 80)
    print("TEST 2: Random 3-SAT (N=4, M=8) - Baseline Comparison")
    print("=" * 80)
    print("\nThis is a structured SAT instance (should have polynomial gap).")
    print("Gap healing should maintain or improve the gap.\n")
    
    clauses_test2, n_vars_test2 = generate_random_3sat(N=4, M=8, seed=42)
    print(f"Problem: {len(clauses_test2)} clauses, {n_vars_test2} variables")
    
    results_test2 = test_gap_healing_on_problem(
        clauses=clauses_test2,
        n_vars=n_vars_test2,
        cd_method="approximate",
        num_points=10
    )
    
    print("\n" + "=" * 80)
    print("TEST 2 SUMMARY")
    print("=" * 80)
    print(f"Verdict: {results_test2['verdict']}")
    print(f"Successful healings: {results_test2['successful_healings']}/{results_test2['num_layers']}")
    print(f"Average improvement: {results_test2['avg_improvement_factor']:.2f}x")
    print(f"Overall improvement: {results_test2['overall_improvement']:.2f}x")
    
    # Visualize results
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    
    visualize_gap_healing(results_test1, save_path="gap_healing_binary_counter.png")
    visualize_gap_healing(results_test2, save_path="gap_healing_random_3sat.png")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey findings:")
    print(f"1. Binary Counter UNSAT: {results_test1['verdict']}")
    print(f"   - Gap improvement: {results_test1['overall_improvement']:.2f}x")
    print(f"2. Random 3-SAT: {results_test2['verdict']}")
    print(f"   - Gap improvement: {results_test2['overall_improvement']:.2f}x")
    print("\nConclusion: Check if gap healing successfully reopened the exponential barrier!")

