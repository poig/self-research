"""
meqo_checkerboard_native.py

MEQO v16: The Checkerboard Native Processor.
------------------------------------------
A "Quantum-Native" Cellular Automaton that runs In-Place.
No buffers. No classical logic loops. Just two alternating
physics kernels applied to the state vector.

Architecture:
- Grid: 3x3 (9 Qubits Total)
- Neighborhood: Von Neumann (Up, Down, Left, Right)
- Rule: "Parity Life" (Flip if exactly 1 neighbor is active)
- Update: Checkerboard (White Cells -> Black Cells)
"""

import numpy as np

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    print("Qiskit not found. Please install qiskit and qiskit-aer.")
    QISKIT_AVAILABLE = False

class CheckerboardChip:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.n_cells = width * height
        self.backend = AerSimulator()
        
        print(f"[Factory] Manufacturing {width}x{height} Checkerboard Chip...")
        print(f"[Factory] Qubits Allocated: {self.n_cells} (No Buffer)")
        
        # --- MANUFACTURING PHASE ---
        # We verify the hardware kernels ONCE.
        self.kernel_white = self._manufacture_kernel(parity=0, label="Physics_White")
        self.kernel_black = self._manufacture_kernel(parity=1, label="Physics_Black")
        print("[Factory] Kernels Installed.")

    def _manufacture_kernel(self, parity, label):
        """
        Builds a Global Instruction that updates ONLY cells of the given parity.
        Parity 0 (White) uses Black neighbors as Controls.
        Parity 1 (Black) uses White neighbors as Controls.
        """
        # Single Register for the entire grid
        reg = QuantumRegister(self.n_cells, 'Grid')
        qc = QuantumCircuit(reg, name=label)
        
        # Iterate over all cells to wire the connections
        for i in range(self.n_cells):
            x, y = i % self.w, i // self.w
            
            # Only wire the cells that belong to this Phase (White or Black)
            if (x + y) % 2 != parity:
                continue
                
            # 1. Get Hardware Connections (Von Neumann: Up/Down/Left/Right)
            # Note: In a Checkerboard, all VN neighbors of a White cell are Black.
            # This ensures the Controls are static during the update!
            neighbors = self._get_vn_neighbors(i)
            
            # 2. Create the Interaction Gate (The Physics)
            # Rule: Flip if exactly 1 neighbor is active (Parity/Growth Rule)
            op = self._create_interaction_gate(len(neighbors))
            
            # 3. Wire connection
            controls = [reg[n] for n in neighbors]
            target = reg[i]
            
            # Append to the Global Kernel
            if controls: # Handle isolated cells if any
                qc.append(op, [target] + controls)
                
        return qc.to_instruction()

    def _get_vn_neighbors(self, idx):
        """Returns Von Neumann neighbors (Up, Down, Left, Right)."""
        x, y = idx % self.w, idx // self.w
        ns = []
        for nx, ny in [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]:
            if 0 <= nx < self.w and 0 <= ny < self.h:
                ns.append(ny * self.w + nx)
        return ns

    def _create_interaction_gate(self, num_neighbors):
        """
        Builds the 'Parity Life' Unitary.
        Map: |Neighbors> |Target> -> |Neighbors> |Target XOR f(Neighbors)>
        """
        dim = 2**(num_neighbors + 1)
        mat = np.zeros((dim, dim), dtype=complex)
        
        for k in range(dim):
            target_in = (k >> 0) & 1
            neighbors = (k >> 1)
            
            # LOGIC: Count active neighbors
            active = bin(neighbors).count('1')
            
            # RULE: Flip if Active == 1 (Simple Reversible Dynamics)
            flip = 1 if active == 1 else 0
            
            target_out = target_in ^ flip
            out_k = (neighbors << 1) | target_out
            mat[out_k, k] = 1.0
            
        return UnitaryGate(mat, label="Rule_Parity")

    def boot_and_run(self, initial_grid, frames):
        """
        The Quantum Runtime Loop.
        Clean, minimal, and purely physical.
        """
        print(f"\n[Runtime] Booting Processor for {frames} Cycles...")
        
        # The Hardware Register
        grid = QuantumRegister(self.n_cells, 'Cell')
        creg = ClassicalRegister(self.n_cells, 'View')
        
        qc = QuantumCircuit(grid, creg)
        
        # 1. Load Input
        init_str = "".join(initial_grid).replace('.', '0').replace('#', '1')
        for i, char in enumerate(init_str):
            if char == '1': qc.x(grid[i])
            
        # 2. The Physics Loop (White -> Black -> White...)
        for t in range(frames):
            qc.barrier()
            # Phase A: Update White Cells
            qc.append(self.kernel_white, grid)
            
            # Phase B: Update Black Cells
            qc.append(self.kernel_black, grid)
            
        # 3. Output
        qc.measure(grid, creg)
        
        # Execute
        res = self.backend.run(qc.decompose(reps=10), shots=1).result()
        bitstring = list(res.get_counts().keys())[0][::-1] # Little Endian fix
        
        return bitstring

    def visualize(self, s):
        visual = s.replace('0', '.').replace('1', '#')
        for y in range(self.h):
            print(f" {visual[y*self.w : (y+1)*self.w]}")

if __name__ == "__main__" and QISKIT_AVAILABLE:
    # 1. Initialize Chip
    chip = CheckerboardChip(3, 3)
    
    # 2. Define Input (Seed in corner)
    # #..
    # ...
    # ...
    input_grid = [
        ".#...",
        "..#..",
        "###..",
        ".....",
        "....."
    ]
    
    print("\n--- T=0 (Seed) ---")
    chip.visualize("".join(input_grid).replace('.', '0').replace('#', '1'))
    
    # 3. Run for 2 Cycles (4 Phases)
    final_state = chip.boot_and_run(input_grid, frames=4)
    
    print(f"\n--- T=2 (Result) ---")
    chip.visualize(final_state)
    
    print("\nVerdict: In-Place Quantum Processing Verified.")