from src.core.quantum_sat_solver import ComprehensiveQuantumSATSolver

# Initialize comprehensive quantum solver (quiet mode)
solver = ComprehensiveQuantumSATSolver(verbose=False, use_true_k=True)

# Define a 3-SAT problem (small backdoor for quantum advantage)
clauses = [
    (1, 2, 3),      # x1 OR x2 OR x3
    (-1, 4, 5),     # NOT x1 OR x4 OR x5
    (-2, -3, 4),    # NOT x2 OR NOT x3 OR x4
    (1, -4, -5),    # x1 OR NOT x4 OR NOT x5
    (2, 3, -4)      # x2 OR x3 OR NOT x4
]

# Solve automatically - analyzes structure and routes to best method
result = solver.solve(clauses, n_vars=5)

# Results with full metadata
print(f"Satisfiable: {result.satisfiable}")
print(f"Solution: {result.assignment}")
print(f"Method used: {result.method_used}")
print(f"Quantum advantage: {result.quantum_advantage_applied}")
print(f"Backdoor size: k ≈ {result.k_estimate:.1f}")
print(f"Total time: {result.total_time:.3f}s")