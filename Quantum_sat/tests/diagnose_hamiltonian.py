"""

Diagnostic script to understand Hamiltonian construction and gap structure
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List

def clause_to_hamiltonian_correct(clause: List[int], n_vars: int) -> SparsePauliOp:
    """
    Correct Hamiltonian construction for SAT clause.
    
    For clause (x1 ∨ x2 ∨ ¬x3):
    - Violated state: x1=0, x2=0, x3=1
    - All other states: satisfied
    
    We build: H = projector onto violated state
    """
    # Build the product of projectors
    H = SparsePauliOp.from_list([('I' * n_vars, 1.0)])
    
    for lit in clause:
        var_idx = abs(lit) - 1
        
        pauli_i = ['I'] * n_vars
        pauli_z = ['I'] * n_vars
        pauli_z[var_idx] = 'Z'
        
        I_op = SparsePauliOp.from_list([(''.join(pauli_i), 0.5)])
        
        if lit > 0:
            # Positive literal: project onto |0⟩ using (I-Z)/2
            Z_op = SparsePauliOp.from_list([(''.join(pauli_z), -0.5)])
        else:
            # Negative literal: project onto |1⟩ using (I+Z)/2
            Z_op = SparsePauliOp.from_list([(''.join(pauli_z), 0.5)])
        
        projector = I_op + Z_op
        H = H.compose(projector)
        H = H.simplify(atol=1e-10)
    
    return H


def test_single_clause():
    """Test single clause Hamiltonian"""
    print("="*80)
    print("TEST: Single Clause Hamiltonian")
    print("="*80)
    
    # Test clause: (x1 ∨ x2 ∨ ¬x3) for N=3 variables
    clause = [1, 2, -3]
    n_vars = 3
    
    print(f"\nClause: {clause} (x1 ∨ x2 ∨ ¬x3)")
    print(f"Violated state: x1=0, x2=0, x3=1 → |001⟩")
    print(f"All other states should have energy 0")
    
    H = clause_to_hamiltonian_correct(clause, n_vars)
    print(f"\nHamiltonian Pauli terms: {len(H.paulis)}")
    
    # Convert to matrix
    H_matrix = H.to_matrix()
    
    # Check energies for all basis states
    print("\n" + "="*60)
    print("Energies for all basis states:")
    print("="*60)
    for i in range(2**n_vars):
        # Create basis state |i⟩
        state = np.zeros(2**n_vars)
        state[i] = 1.0
        
        # Compute energy: ⟨i|H|i⟩
        energy = np.real(state @ H_matrix @ state)
        
        # Decode assignment
        x1 = (i >> 0) & 1
        x2 = (i >> 1) & 1
        x3 = (i >> 2) & 1
        
        # Check if clause is satisfied
        clause_sat = (x1 == 1) or (x2 == 1) or (x3 == 0)
        
        print(f"|{i:03b}⟩ (x1={x1}, x2={x2}, x3={x3}): E={energy:.6f} {'✓ SAT' if clause_sat else '✗ VIOLATED'}")
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    eigenvalues = np.sort(eigenvalues)
    
    print("\n" + "="*60)
    print("Eigenvalue spectrum:")
    print("="*60)
    for i, eig in enumerate(eigenvalues):
        print(f"E_{i} = {eig:.6f}")
    
    gap = eigenvalues[1] - eigenvalues[0]
    print(f"\nSpectral gap: Δ = {gap:.6f}")
    

def test_binary_counter():
    """Test binary counter UNSAT problem"""
    print("\n\n" + "="*80)
    print("TEST: Binary Counter UNSAT (N=3)")
    print("="*80)
    
    N = 3
    clauses = []
    
    # Generate 2^N clauses, each forcing a different assignment
    for i in range(2 ** N):
        clause = []
        for bit_pos in range(N):
            if (i >> bit_pos) & 1:
                clause.append(bit_pos + 1)
            else:
                clause.append(-(bit_pos + 1))
        clauses.append(clause)
    
    print(f"\nGenerated {len(clauses)} clauses (2^{N} = {2**N})")
    print("\nClauses:")
    for i, clause in enumerate(clauses):
        print(f"  C{i}: {clause}")
    
    print("\n" + "="*60)
    print("Analysis: Each clause forbids exactly one assignment")
    print("="*60)
    
    for i, clause in enumerate(clauses):
        # Decode which assignment is forbidden
        forbidden = []
        for lit in clause:
            if lit > 0:
                forbidden.append(f"x{abs(lit)}=0")
            else:
                forbidden.append(f"x{abs(lit)}=1")
        print(f"  C{i} violated when: {', '.join(forbidden)} → |{i:03b}⟩")
    
    # Build total Hamiltonian
    print("\n" + "="*60)
    print("Building total Hamiltonian H = Σ H_clause")
    print("="*60)
    
    H_total = SparsePauliOp.from_list([('I' * N, 0.0)])
    
    for i, clause in enumerate(clauses):
        H_clause = clause_to_hamiltonian_correct(clause, N)
        H_total = H_total + H_clause
        print(f"  Added C{i}, running norm: {np.linalg.norm(H_total.coeffs):.6f}")
    
    H_total = H_total.simplify(atol=1e-10)
    
    print(f"\nTotal Hamiltonian terms: {len(H_total.paulis)}")
    
    # Convert to matrix
    H_matrix = H_total.to_matrix()
    
    # Check energies
    print("\n" + "="*60)
    print("Energies for all basis states:")
    print("="*60)
    
    for i in range(2**N):
        state = np.zeros(2**N)
        state[i] = 1.0
        energy = np.real(state @ H_matrix @ state)
        print(f"|{i:03b}⟩: E={energy:.6f}")
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    eigenvalues = np.sort(eigenvalues)
    
    print("\n" + "="*60)
    print("Eigenvalue spectrum:")
    print("="*60)
    for i, eig in enumerate(eigenvalues):
        print(f"E_{i} = {eig:.6f}")
    
    gap = eigenvalues[1] - eigenvalues[0]
    print(f"\nSpectral gap: Δ = {gap:.6f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("Each clause penalizes one assignment.")
    print("Since there are 2^N clauses and 2^N assignments,")
    print("EVERY assignment is penalized exactly once.")
    print("Therefore, ALL states have the same energy!")
    print("This creates a DEGENERATE ground space with ZERO gap.")
    print("\nThis is why scaffolding fails: we're adding contradictions,")
    print("not filtering solutions!")


def test_hierarchical_binary_counter():
    """Test hierarchical scaffolding on binary counter"""
    print("\n\n" + "="*80)
    print("TEST: Hierarchical Scaffolding on Binary Counter")
    print("="*80)
    
    N = 3
    clauses = []
    
    for i in range(2 ** N):
        clause = []
        for bit_pos in range(N):
            if (i >> bit_pos) & 1:
                clause.append(bit_pos + 1)
            else:
                clause.append(-(bit_pos + 1))
        clauses.append(clause)
    
    print(f"\nTesting hierarchical addition of {len(clauses)} clauses")
    
    H_seed = SparsePauliOp.from_list([('I' * N, 0.0)])
    
    for layer in range(len(clauses)):
        print(f"\n{'='*60}")
        print(f"Layer {layer}: Adding clause C{layer} = {clauses[layer]}")
        print('='*60)
        
        # Add this clause
        H_new = clause_to_hamiltonian_correct(clauses[layer], N)
        H_current = H_seed + H_new
        
        # Remaining clauses
        H_rest = SparsePauliOp.from_list([('I' * N, 0.0)])
        for j in range(layer + 1, len(clauses)):
            H_rest = H_rest + clause_to_hamiltonian_correct(clauses[j], N)
        
        # Check gap at s=0.5
        s = 0.5
        H_interp = H_current + s * H_rest
        H_interp = H_interp.simplify(atol=1e-10)
        
        # Eigenvalues
        H_matrix = H_interp.to_matrix()
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        eigenvalues = np.sort(eigenvalues)
        
        gap = eigenvalues[1] - eigenvalues[0]
        
        print(f"  Ground state energy: E_0 = {eigenvalues[0]:.6f}")
        print(f"  First excited energy: E_1 = {eigenvalues[1]:.6f}")
        print(f"  Spectral gap: Δ = {gap:.6f}")
        
        # Count degeneracy
        degeneracy_ground = np.sum(np.abs(eigenvalues - eigenvalues[0]) < 1e-6)
        print(f"  Ground state degeneracy: {degeneracy_ground}")
        
        # Update seed
        H_seed = H_current
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("The binary counter creates a degenerate ground space")
    print("because it's UNSAT - every assignment is penalized.")
    print("The gap doesn't close exponentially; it's ZERO from the start!")


if __name__ == "__main__":
    test_single_clause()
    test_binary_counter()
    test_hierarchical_binary_counter()

