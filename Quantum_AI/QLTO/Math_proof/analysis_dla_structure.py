"""
analyze_dla_structure.py: The Physics-to-Math Bridge (Localization Update)
==========================================================================
Testing the "Algebraic Localization Conjecture" for Computational Hardness.

Hypothesis:
    - Tractable (Ordered): Adjoint Eigenvectors are Delocalized (Low IPR).
      The algebra is "connected."
    - Intractable (Chaotic): Adjoint Eigenvectors are Anderson Localized (High IPR).
      The algebra is "fragmented."

Methodology:
    1. Generate DLA Basis.
    2. Compute Adjoint Matrix (Commutator Graph).
    3. Compute Eigenvectors of the Adjoint Operator.
    4. Measure Inverse Participation Ratio (IPR).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings

warnings.filterwarnings("ignore")

try:
    from qiskit.quantum_info import SparsePauliOp
except ImportError:
    print("CRITICAL: Qiskit missing.")
    exit(1)

def get_commutator(op_a, op_b):
    return (op_a.compose(op_b) - op_b.compose(op_a)).simplify()

class DLA_Analyzer:
    def __init__(self, generators, n_qubits):
        self.generators = generators
        self.n = n_qubits
        self.basis = [] 

    def generate_dla(self):
        current_basis = [g for g in self.generators]
        self.basis = current_basis
        new_found = True
        depth = 0
        MAX_DEPTH = 6 # Limit depth for speed
        
        print(f"  Start: {len(self.basis)} generators.")
        
        while new_found and depth < MAX_DEPTH:
            new_found = False
            temp_new = []
            depth += 1
            
            # Optimization: Only commutate New vs All
            # To keep it simple and robust for small N, we do All vs All but skip processed pairs?
            # For N=3/4, All vs All is fine.
            
            for i in range(len(self.basis)):
                for j in range(i + 1, len(self.basis)):
                    comm = get_commutator(self.basis[i], self.basis[j])
                    if np.all(np.isclose(np.abs(comm.coeffs), 0.0)): continue
                        
                    if self._is_independent(comm, self.basis + temp_new):
                        norm = np.linalg.norm(comm.coeffs)
                        comm = SparsePauliOp(comm.paulis, comm.coeffs / norm)
                        temp_new.append(comm)
                        new_found = True
            
            if temp_new:
                self.basis.extend(temp_new)
                print(f"  Depth {depth}: Added {len(temp_new)} ops. Total Dim: {len(self.basis)}")
            else:
                print(f"  Depth {depth}: Closure reached.")
                
    def _is_independent(self, candidate, basis):
        vec_c = self._op_to_vec(candidate)
        vec_basis = [self._op_to_vec(b) for b in basis]
        if not vec_basis: return True
        mat = np.array(vec_basis + [vec_c])
        return np.linalg.matrix_rank(mat) > np.linalg.matrix_rank(mat[:-1])

    def _op_to_vec(self, op):
        vec = np.zeros(4**self.n, dtype=complex)
        for pauli, coeff in zip(op.paulis, op.coeffs):
            s = pauli.to_label()
            idx = 0
            for char in s:
                idx *= 4
                if char == 'I': idx += 0
                elif char == 'X': idx += 1
                elif char == 'Y': idx += 2
                elif char == 'Z': idx += 3
            vec[idx] = coeff
        return vec

    def compute_localization(self):
        """
        Computes Inverse Participation Ratio (IPR) of Adjoint Eigenvectors.
        IPR = sum(|psi_i|^4) / (sum(|psi_i|^2))^2
        IPR ~ 1/Dim (Delocalized)
        IPR ~ 1 (Localized)
        """
        dim = len(self.basis)
        if dim > 300:
            print("  [Warning] Dimension too large for full diagonalization.")
            return 0.0, 0.0

        # Build Adjoint Matrix (Killing Form approximation)
        # K_ij = || [B_i, B_j] ||
        adj_mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                comm = get_commutator(self.basis[i], self.basis[j])
                adj_mat[i, j] = np.linalg.norm(comm.coeffs)
                
        # Diagonalize
        evals, evecs = np.linalg.eigh(adj_mat)
        
        # Compute IPR for each eigenvector
        iprs = []
        for k in range(dim):
            psi = evecs[:, k]
            ipr = np.sum(np.abs(psi)**4)
            iprs.append(ipr)
            
        avg_ipr = np.mean(iprs)
        
        # Normalized IPR: IPR * Dim. 
        # If 1 -> Delocalized. If Dim -> Localized.
        norm_ipr = avg_ipr * dim
        
        return evals[1] if len(evals)>1 else 0, norm_ipr

# ==============================================================================
# MAIN COMPARISON
# ==============================================================================

def run_localization_test():
    print("\n=== ALGEBRAIC LOCALIZATION TEST (PATH A) ===")
    print("Hypothesis: 'Ordered' -> Delocalized (IPR ~ 1/D).")
    print("            'Chaotic' -> Localized (IPR ~ 1).")
    
    N_LIST = [3, 4]
    
    for n in N_LIST:
        print(f"\n--- System Size N={n} ---")
        
        # 1. Ordered
        ops_ord = []
        hc_list = []
        for i in range(n):
            for j in range(i+1, n):
                s = ["I"]*n; s[i]="Z"; s[j]="Z"
                hc_list.append(("".join(s), 1.0))
        ops_ord.append(SparsePauliOp.from_list(hc_list))
        hb_list = []
        for i in range(n):
            s = ["I"]*n; s[i]="X"
            hb_list.append(("".join(s), 1.0))
        ops_ord.append(SparsePauliOp.from_list(hb_list))
        
        dla_ord = DLA_Analyzer(ops_ord, n)
        dla_ord.generate_dla()
        gap_ord, ipr_ord = dla_ord.compute_localization()
        print(f"Ordered Gap: {gap_ord:.4f} | Norm IPR: {ipr_ord:.4f}")
        
        # 2. Chaotic
        ops_ch = []
        hc_list = []
        np.random.seed(42)
        for i in range(n):
            for j in range(i+1, n):
                s = ["I"]*n; s[i]="Z"; s[j]="Z"
                hc_list.append(("".join(s), np.random.uniform(-1, 1)))
        ops_ch.append(SparsePauliOp.from_list(hc_list))
        hb_list = []
        for i in range(n):
            s = ["I"]*n; s[i]="X"
            hb_list.append(("".join(s), np.random.uniform(-1, 1)))
        ops_ch.append(SparsePauliOp.from_list(hb_list))
        
        dla_ch = DLA_Analyzer(ops_ch, n)
        dla_ch.generate_dla()
        gap_ch, ipr_ch = dla_ch.compute_localization()
        print(f"Chaotic Gap: {gap_ch:.4f} | Norm IPR: {ipr_ch:.4f}")
        
    print("\n=== CONCLUSION ===")
    print("Compare Normalized IPR.")
    print("If Chaotic IPR >> Ordered IPR, we have Anderson Localization in the Algebra.")

if __name__ == "__main__":
    run_localization_test()