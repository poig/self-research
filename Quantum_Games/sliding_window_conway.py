import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class SlidingWindowConway:
    """
    Memory-Optimized Quantum Life using a Sliding Window.
    Simulates a W x 3 window to compute the next state of the middle row.
    Qubits: ~3*W + Ancilla (Linear in Width, Constant in Height).
    """
    def __init__(self, width):
        self.w = width
        
        # 3 Rows (Top, Mid, Bot)
        self.qr_top = QuantumRegister(self.w, 'row_top')
        self.qr_mid = QuantumRegister(self.w, 'row_mid')
        self.qr_bot = QuantumRegister(self.w, 'row_bot')
        
        # Output Row (Next state of Mid)
        self.qr_out = QuantumRegister(self.w, 'row_out')
        
        # Shared Ancilla for calculation (4 qubits for counting 0-8)
        self.qr_math = QuantumRegister(4, 'calculator') 
        
        # Measurement
        self.cr = ClassicalRegister(self.w, 'measure')
        
        self.backend = AerSimulator(method='matrix_product_state') 

    def _add_neighbor_to_counter(self, qc, ctrl_qubit):
        """Adds 1 to the 4-qubit math register, controlled by ctrl_qubit."""
        c = self.qr_math
        qc.mcx([ctrl_qubit, c[0], c[1], c[2]], c[3])
        qc.mcx([ctrl_qubit, c[0], c[1]], c[2])
        qc.ccx(ctrl_qubit, c[0], c[1])
        qc.cx(ctrl_qubit, c[0])
        
    def _uncompute_counter(self, qc, ctrl_qubit):
        """Inverse of add_neighbor."""
        c = self.qr_math
        qc.cx(ctrl_qubit, c[0])
        qc.ccx(ctrl_qubit, c[0], c[1])
        qc.mcx([ctrl_qubit, c[0], c[1]], c[2])
        qc.mcx([ctrl_qubit, c[0], c[1], c[2]], c[3])

    def _apply_life_logic(self, qc, target_qubit, current_state_qubit):
        """Applies B3/S23 rules."""
        c = self.qr_math
        
        # Rule 1: Birth (Count == 3 AND Current == 0) -> Target = 1
        qc.x([c[2], c[3], current_state_qubit])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[2], c[3], current_state_qubit]) 
        
        # Rule 2: Survival (Count == 2 OR 3 AND Current == 1) -> Target = 1
        # Case A: Count == 2 (0010) AND Current == 1
        qc.x([c[0], c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[0], c[2], c[3]])
        
        # Case B: Count == 3 (0011) AND Current == 1
        qc.x([c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[2], c[3]])

    def process_window(self, row_top_str, row_mid_str, row_bot_str):
        """
        Computes the next state of the middle row given the 3 input rows.
        """
        qc = QuantumCircuit(self.qr_top, self.qr_mid, self.qr_bot, self.qr_out, self.qr_math, self.cr)
        
        # 1. Load Input State
        for i, char in enumerate(row_top_str):
            if char == '#': qc.x(self.qr_top[i])
        for i, char in enumerate(row_mid_str):
            if char == '#': qc.x(self.qr_mid[i])
        for i, char in enumerate(row_bot_str):
            if char == '#': qc.x(self.qr_bot[i])
            
        qc.barrier()
        
        # 2. Compute Next State for each cell in Mid Row
        for i in range(self.w):
            # Identify Neighbors
            # Top: i-1, i, i+1
            # Mid: i-1,    i+1
            # Bot: i-1, i, i+1
            # Handle Boundaries (Toroidal X)
            left = (i - 1) % self.w
            right = (i + 1) % self.w
            
            neighbors = [
                self.qr_top[left], self.qr_top[i], self.qr_top[right],
                self.qr_mid[left],                 self.qr_mid[right],
                self.qr_bot[left], self.qr_bot[i], self.qr_bot[right]
            ]
            
            # Compute
            for n_q in neighbors:
                self._add_neighbor_to_counter(qc, n_q)
            
            self._apply_life_logic(qc, self.qr_out[i], self.qr_mid[i])
            
            for n_q in reversed(neighbors):
                self._uncompute_counter(qc, n_q)
                
        # 3. Measure Output
        qc.measure(self.qr_out, self.cr)
        
        # Execute
        # Decompose for simulator
        qc_transpiled = qc.decompose(reps=3)
        job = self.backend.run(qc_transpiled, shots=1)
        res = job.result()
        counts = res.get_counts()
        
        raw_bitstring = list(counts.keys())[0]
        return raw_bitstring[::-1] # Reverse to match 0..W

    def simulate_frame(self, grid):
        """
        Simulates one frame of the entire grid by sliding the window.
        grid: List of strings (rows).
        """
        height = len(grid)
        new_grid = []
        
        for y in range(height):
            # Define Window (Toroidal Y)
            top = grid[(y - 1) % height]
            mid = grid[y]
            bot = grid[(y + 1) % height]
            
            # Process
            new_row_bits = self.process_window(top, mid, bot)
            
            # Convert to string
            new_row_str = new_row_bits.replace('0', '.').replace('1', '#')
            new_grid.append(new_row_str)
            
        return new_grid

if __name__ == "__main__":
    # Demo: Blinker on 5x5 Grid (to avoid toroidal self-interaction)
    # .....
    # ..#..
    # ..#..
    # ..#..
    # .....
    sw = SlidingWindowConway(5)
    grid = [
        ".....",
        "..#..",
        "..#..",
        "..#..",
        "....."
    ]
    
    print("Initial:")
    for r in grid: print(r)
    
    print("\nSimulating 1 Frame...")
    new_grid = sw.simulate_frame(grid)
    
    print("Result:")
    for r in new_grid: print(r)
