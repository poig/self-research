from qiskit.quantum_info import SparsePauliOp
import numpy as np
from nisq_v2 import force_heisenberg_hamiltonian, generate_frustrated_hamiltonian


H = force_heisenberg_hamiltonian(4)
matrix = H.to_matrix()
eigvals = np.linalg.eigvalsh(matrix)
print(f"Ground State Energy: {eigvals[0]}")
print(f"All Eigenvalues: {eigvals}")

H = generate_frustrated_hamiltonian(4, seed=42)
matrix = H.to_matrix()
eigvals = np.linalg.eigvalsh(matrix)
print(f"frustrated Ground State Energy: {eigvals[0]}")
print(f"frustrated All Eigenvalues: {eigvals}")