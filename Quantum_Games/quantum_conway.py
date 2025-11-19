import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class OptimizedPingPong:
    """
    Optimized Quantum Life using Shared Ancilla for B3/S23 logic.
    Uses "Ping-Pong" buffering: World A -> World B, then Reset A.
    """
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.n_cells = width * height
        
        # Two Grids (Current and Next)
        self.qr_a = QuantumRegister(self.n_cells, 'grid_a')
        self.qr_b = QuantumRegister(self.n_cells, 'grid_b')
        
        # Shared Ancilla for calculation (4 qubits for counting 0-8)
        self.qr_math = QuantumRegister(4, 'calculator') 
        
        # Measurement
        self.cr = ClassicalRegister(self.n_cells, 'measure')
        
        # We build the circuit dynamically in simulate()
        # But we can define the backend here.
        # MPS is good for larger grids with low entanglement depth.
        self.backend = AerSimulator(method='matrix_product_state') 

    def get_neighbors(self, idx):
        """Returns indices of 8 neighbors (Toroidal/Wrap-around)."""
        x, y = idx % self.w, idx // self.w
        ns = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = (x + dx) % self.w, (y + dy) % self.h
                ns.append(ny * self.w + nx)
        return ns

    def _add_neighbor_to_counter(self, qc, ctrl_qubit):
        """Adds 1 to the 4-qubit math register, controlled by ctrl_qubit."""
        c = self.qr_math
        # Toffoli Ladder for Increment
        # c3 = c3 ^ (ctrl & c0 & c1 & c2)
        qc.mcx([ctrl_qubit, c[0], c[1], c[2]], c[3])
        # c2 = c2 ^ (ctrl & c0 & c1)
        qc.mcx([ctrl_qubit, c[0], c[1]], c[2])
        # c1 = c1 ^ (ctrl & c0)
        qc.ccx(ctrl_qubit, c[0], c[1])
        # c0 = c0 ^ ctrl
        qc.cx(ctrl_qubit, c[0])
        
    def _uncompute_counter(self, qc, ctrl_qubit):
        """Inverse of add_neighbor."""
        c = self.qr_math
        qc.cx(ctrl_qubit, c[0])
        qc.ccx(ctrl_qubit, c[0], c[1])
        qc.mcx([ctrl_qubit, c[0], c[1]], c[2])
        qc.mcx([ctrl_qubit, c[0], c[1], c[2]], c[3])

    def _apply_life_logic(self, qc, target_qubit, current_state_qubit):
        """
        Applies B3/S23 rules to target_qubit based on counter and current state.
        Target is assumed to be |0>.
        """
        c = self.qr_math
        
        # Rule 1: Birth (Count == 3 AND Current == 0) -> Target = 1
        # 3 is '0011' (c3=0, c2=0, c1=1, c0=1)
        # We want to trigger if c=0011 and current=0.
        # Flip 0s to 1s for MCX
        qc.x([c[2], c[3], current_state_qubit])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[2], c[3], current_state_qubit]) # Restore
        
        # Rule 2: Survival (Count == 2 OR 3 AND Current == 1) -> Target = 1
        # We already handled Birth (3, 0).
        # Now we need (2, 1) and (3, 1).
        
        # Case A: Count == 2 (0010) AND Current == 1
        qc.x([c[0], c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[0], c[2], c[3]])
        
        # Case B: Count == 3 (0011) AND Current == 1
        qc.x([c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3], current_state_qubit], target_qubit)
        qc.x([c[2], c[3]])

    def run_step(self, qc, source_reg, target_reg):
        """
        Evolves source_reg -> target_reg.
        """
        for i in range(self.n_cells):
            current_cell = source_reg[i]
            target_cell = target_reg[i]
            neighbors = [source_reg[n] for n in self.get_neighbors(i)]
            
            # 1. Compute Neighbors into Ancilla
            for n_q in neighbors:
                self._add_neighbor_to_counter(qc, n_q)
            
            # 2. Apply Logic to Target
            self._apply_life_logic(qc, target_cell, current_cell)
            
            # 3. Uncompute Ancilla
            for n_q in reversed(neighbors):
                self._uncompute_counter(qc, n_q)

    def simulate(self, initial_grid, frames):
        """
        Runs the simulation for `frames` steps.
        Returns the final bitstring.
        """
        qc = QuantumCircuit(self.qr_a, self.qr_b, self.qr_math, self.cr)
        
        # 1. Load Initial State into A
        s = "".join(initial_grid).replace('.', '0').replace('#', '1')
        for i, char in enumerate(s):
            if char == '1': qc.x(self.qr_a[i])
            
        active_source = self.qr_a
        active_target = self.qr_b
        
        for t in range(frames):
            qc.barrier()
            
            # Evolve A -> B
            self.run_step(qc, active_source, active_target)
            
            # Reset Source (The "Ping-Pong" non-unitary step)
            # In a real device, we'd measure and reset.
            # Here we just use the reset instruction.
            qc.reset(active_source)
            
            # Swap roles
            active_source, active_target = active_target, active_source
            
        # Measure the final active grid
        qc.measure(active_source, self.cr)
        
        # Execute
        print(f"Executing Optimized Ping-Pong ({frames} steps) on MPS...")
        # Decompose Toffolis for the simulator
        qc_transpiled = qc.decompose(reps=3) 
        job = self.backend.run(qc_transpiled, shots=1)
        res = job.result()
        counts = res.get_counts()
        
        # Parse result
        # Qiskit returns bitstring in reverse order of registers?
        # And reverse qubit order?
        # Let's assume standard little-endian for now and reverse it back.
        raw_bitstring = list(counts.keys())[0]
        # The bitstring contains all measured bits. We only measured 'active_source'.
        # But we defined CR as size n_cells.
        # So the bitstring should be length n_cells.
        
        return raw_bitstring[::-1] # Reverse to match array order 0..N

if __name__ == "__main__":
    # Demo
    W, H = 4, 4
    engine = OptimizedPingPong(W, H)
    
    # Blinker
    # . # . .
    # . # . .
    # . # . .
    # . . . .
    grid = [
        ".#..",
        ".#..",
        ".#..",
        "...."
    ]
    
    print("Initial:")
    for r in grid: print(r)
    
    res = engine.simulate(grid, 1)
    
    print("\nResult T=1:")
    # Reshape
    for y in range(H):
        row = res[y*W : (y+1)*W]
        print(row.replace('0','.').replace('1','#'))
