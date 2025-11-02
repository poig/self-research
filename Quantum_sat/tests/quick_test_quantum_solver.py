"""
Quick test of integrated quantum solver
"""
import sys
sys.path.append('..')

from integrated_quantum_solver import IntegratedQuantumSATSolver

print("Testing Integrated Quantum SAT Solver...")
print()

# Simple 3-variable instance
clauses = [
    (1, 2, 3),
    (-1, 2),
    (-2, -3),
]

solver = IntegratedQuantumSATSolver(verbose=True)
result = solver.solve(clauses, n_vars=3, true_k=2)

print(f"\nResult: {'SAT' if result.satisfiable else 'UNSAT'}")
print(f"Method: {result.method_used}")
print(f"Quantum advantage: {result.quantum_advantage}")
print(f"Total time: {result.total_time:.3f}s")
