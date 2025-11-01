"""Debug script to understand the Hamiltonian construction issue"""

import numpy as np
import sys
sys.path.insert(0, '.')
from quantum_structure_analyzer import QuantumStructureAnalyzer

# Create QSA instance
qsa = QuantumStructureAnalyzer(use_ml=False)

# Test simple 2-SAT: (x1 ∨ x2) ∧ (¬x1 ∨ x2)
clauses = [(1, 2), (-1, 2)]
n_vars = 2

print("="*70)
print("DEBUG: Simple 2-SAT Instance")
print("Clauses: (x1 ∨ x2) ∧ (¬x1 ∨ x2)")
print("="*70)

# Build Hamiltonian
H = qsa._build_hamiltonian(clauses, n_vars)
H_matrix = H.to_matrix()
H_diag = np.diag(H_matrix).real

print("\nFull Hamiltonian matrix:")
print(H_matrix.real)

print("\nDiagonal elements:")
for assignment in range(4):
    x1 = (assignment >> 0) & 1
    x2 = (assignment >> 1) & 1
    
    # Check clauses manually
    clause1 = (x1 == 1) or (x2 == 1)  # x1 ∨ x2
    clause2 = (x1 == 0) or (x2 == 1)  # ¬x1 ∨ x2
    
    violated = (not clause1) + (not clause2)
    
    print(f"Assignment {assignment} (x1={x1}, x2={x2}):")
    print(f"  Clause 1 (x1 ∨ x2): {clause1}")
    print(f"  Clause 2 (¬x1 ∨ x2): {clause2}")
    print(f"  Expected violations: {violated}")
    print(f"  H_diag[{assignment}]: {H_diag[assignment]}")
    print()

print("\n" + "="*70)
print("Analysis of clause violating states:")
print("="*70)

for i, clause in enumerate(clauses, 1):
    print(f"\nClause {i}: {clause}")
    
    # Compute violating state manually
    violating_state = 0
    for lit in clause:
        var = abs(lit) - 1
        if lit < 0:
            violating_state |= (1 << var)
    
    print(f"  Violating state (binary): {violating_state:0{n_vars}b}")
    print(f"  Violating state (decimal): {violating_state}")
    
    # Check what this assignment means
    x1_viol = (violating_state >> 0) & 1
    x2_viol = (violating_state >> 1) & 1
    print(f"  This means: x1={x1_viol}, x2={x2_viol}")
    
    # Verify this actually violates the clause
    satisfied = False
    for lit in clause:
        var = abs(lit) - 1
        var_value = (violating_state >> var) & 1
        
        if lit > 0:  # Positive literal
            if var_value == 1:
                satisfied = True
                break
        else:  # Negative literal
            if var_value == 0:
                satisfied = True
                break
    
    print(f"  Clause satisfied? {satisfied} (should be False)")
    print(f"  Clause violated? {not satisfied} (should be True)")

print("\n" + "="*70)
print("Direct Matrix Construction Check:")
print("="*70)

# Build matrix directly to verify
dim = 2 ** n_vars
H_direct = np.zeros((dim, dim))

for clause in clauses:
    violating_state = 0
    for lit in clause:
        var = abs(lit) - 1
        if var >= n_vars:
            continue
        if lit < 0:
            violating_state |= (1 << var)
    
    print(f"Clause {clause} violates at state {violating_state}")
    H_direct[violating_state, violating_state] += 1.0

print("\nDirect H_matrix diagonal:")
print(np.diag(H_direct))

print("\nConverted H_matrix diagonal:")
print(H_diag)

print("\nAre they equal?", np.allclose(np.diag(H_direct), H_diag))
