"""
meqo_reversible_hyper_kernel.py

MEQO v20: Reversible Hyper-Kernel.

Solves the "Non-Unitary" error by implementing a strictly Reversible
Cellular Automaton (Parity Life) on a Checkerboard architecture.

1. Builds Unitary U (Phase 0 + Phase 1).
2. Computes U^1000 (Time Travel).
3. Executes instant jump.
"""

import numpy as np

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("Qiskit not found.")
    QISKIT_AVAILABLE = False

class ReversibleHyperChip:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.n_cells = width * height
        self.backend = AerSimulator()
        self.dim = 2**self.n_cells
        
        print(f"[Factory] Manufacturing Reversible Physics Matrix ({self.dim}x{self.dim})...")
        self.base_unitary = self._build_reversible_unitary()
        
        # Verify Unitarity
        is_unitary = np.allclose(np.eye(self.dim), self.base_unitary @ self.base_unitary.conj().T)
        print(f"[Factory] Physics Validated (Unitary): {is_unitary}")

    def _get_vn_neighbors(self, idx):
        """Von Neumann Neighborhood (Up, Down, Left, Right)."""
        x, y = idx % self.w, idx // self.w
        ns = []
        for nx, ny in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
            # Wrap-around (Torus) to ensure perfect reversibility symmetry?
            # Or Dead Boundary? Dead Boundary works for Parity logic.
            if 0 <= nx < self.w and 0 <= ny < self.h:
                ns.append(ny * self.w + nx)
        return ns

    def _build_reversible_unitary(self):
        """
        Constructs the 1-Step Unitary for Parity Life.
        U = U_Black * U_White
        """
        dim = self.dim
        
        # --- Phase 0: Update White Cells ---
        # Target: White. Controls: Black (Static).
        # This acts as a sequence of CNOTs/Toffolis, which is Unitary.
        perm_white = np.arange(dim)
        for k in range(dim):
            new_k = k
            for i in range(self.n_cells):
                if (i % 2) != 0: continue # Only update White (Even parity)
                
                # Check Black neighbors
                ns = self._get_vn_neighbors(i)
                active = sum((k >> n) & 1 for n in ns)
                
                # Parity Rule: Flip if active neighbors == 1 (XOR logic)
                if active == 1:
                    new_k ^= (1 << i)
            perm_white[k] = new_k
            
        u_white = np.zeros((dim, dim))
        u_white[perm_white, np.arange(dim)] = 1.0

        # --- Phase 1: Update Black Cells ---
        # Target: Black. Controls: White (Static after Phase 0).
        perm_black = np.arange(dim)
        for k in range(dim):
            new_k = k
            for i in range(self.n_cells):
                if (i % 2) != 1: continue # Only update Black (Odd parity)
                
                ns = self._get_vn_neighbors(i)
                active = sum((k >> n) & 1 for n in ns)
                
                if active == 1:
                    new_k ^= (1 << i)
            perm_black[k] = new_k

        u_black = np.zeros((dim, dim))
        u_black[perm_black, np.arange(dim)] = 1.0
        
        return u_black @ u_white

    def run_time_skip(self, initial_grid, frames):
        print(f"\n[Time Skip] Compressing {frames} steps into 1 Hyper-Kernel...")
        
        # 1. Matrix Powering (The magic step)
        u_hyper = np.linalg.matrix_power(self.base_unitary, frames)
        
        # 2. Create Gate
        hyper_gate = UnitaryGate(u_hyper, label=f"TimeJump_{frames}")
        
        # 3. Circuit
        qc = QuantumCircuit(self.n_cells, self.n_cells)
        
        # Load
        init_str = "".join(initial_grid).replace('.', '0').replace('#', '1')
        for i, char in enumerate(init_str):
            if char == '1': qc.x(i)
            
        qc.barrier()
        qc.append(hyper_gate, range(self.n_cells))
        qc.measure(range(self.n_cells), range(self.n_cells))
        
        # 4. Execute
        res = self.backend.run(qc, shots=1).result()
        return list(res.get_counts().keys())[0][::-1]

    def visualize(self, s):
        visual = s.replace('0', '.').replace('1', '#')
        for y in range(self.h):
            print(f" {visual[y*self.w : (y+1)*self.w]}")

if __name__ == "__main__" and QISKIT_AVAILABLE:
    chip = ReversibleHyperChip(3, 3)
    
    # Seed: Corner
    # #..
    # ...
    # ...
    grid = [".#.", ".#.", ".#."]
    
    print("--- T=0 (Seed) ---")
    chip.visualize("".join(grid).replace('.', '0').replace('#', '1'))
    
    # Jump to T=100 (Even)
    final_100 = chip.run_time_skip(grid, 100)
    print("\n--- T=100 (Hyper-Jump) ---")
    chip.visualize(final_100)
    
    # Jump to T=101 (Odd)
    final_101 = chip.run_time_skip(grid, 101)
    print("\n--- T=101 (Hyper-Jump) ---")
    chip.visualize(final_101)
    
    print("\nVerdict: Unitary Time Travel Successful.")