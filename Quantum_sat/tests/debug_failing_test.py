"""Debug for failing test_satisfiable_with_solution"""

import numpy as np
import sys
sys.path.insert(0, '.')
from quantum_structure_analyzer import QuantumStructureAnalyzer

def count_violated_clauses(assignment, clauses, n_vars):
    """Count violated clauses"""
    violated = 0
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit) - 1
            if var >= n_vars:
                continue
            var_value = (assignment >> var) & 1
            if lit > 0:
                if var_value == 1:
                    clause_satisfied = True
                    break
            else:
                if var_value == 0:
                    clause_satisfied = True
                    break
        if not clause_satisfied:
            violated += 1
    return violated

# Test case that's failing
clauses = [(1, 2), (-1, 3), (-2, -3)]
n_vars = 3

qsa = QuantumStructureAnalyzer(use_ml=False)
H = qsa._build_hamiltonian(clauses, n_vars)
H_matrix = H.to_matrix()
H_diag = np.diag(H_matrix).real

print("Clauses: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)")
print("="*70)
print("\nH_matrix diagonal:", H_diag)
print()

for assignment in range(2 ** n_vars):
    x1 = (assignment >> 0) & 1
    x2 = (assignment >> 1) & 1
    x3 = (assignment >> 2) & 1
    
    expected = count_violated_clauses(assignment, clauses, n_vars)
    actual = H_diag[assignment]
    
    match = "✓" if abs(expected - actual) < 1e-6 else "✗"
    
    print(f"{match} Assignment {assignment} ({x1}{x2}{x3}): expected={expected}, actual={actual:.1f}")
    
    if abs(expected - actual) > 1e-6:
        print(f"   MISMATCH! Let's trace through clauses:")
        for i, clause in enumerate(clauses, 1):
            satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                var_value = (assignment >> var) & 1
                if (lit > 0 and var_value == 1) or (lit < 0 and var_value == 0):
                    satisfied = True
                    break
            print(f"   Clause {i} {clause}: {'SAT' if satisfied else 'VIOLATED'}")

print("\n" + "="*70)
print("Clause violating states:")
for i, clause in enumerate(clauses, 1):
    violating_state = 0
    for lit in clause:
        var = abs(lit) - 1
        if var >= n_vars:
            continue
        if lit < 0:
            violating_state |= (1 << var)
    
    x1_v = (violating_state >> 0) & 1
    x2_v = (violating_state >> 1) & 1
    x3_v = (violating_state >> 2) & 1
    
    print(f"Clause {i} {clause}: violates at state {violating_state} ({x1_v}{x2_v}{x3_v})")
