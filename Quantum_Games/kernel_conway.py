import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class QuantumKernelConway:
    def __init__(self, N):
        self.N = N
        self.grid = np.zeros((N, N), dtype=int)
        
        # The Kernel Circuit (Fixed Size)
        # Inputs: 9 qubits (8 neighbors + 1 center)
        # Aux: 4 qubits (Counter)
        # Output: 1 qubit (Result) -> We can reuse center? No, let's use a dedicated target.
        # Total: 9 + 4 + 1 = 14 qubits.
        
        self.backend = AerSimulator()
        self._kernel_circuit = self._build_kernel_circuit()

    def _build_kernel_circuit(self):
        """
        Builds the parameterized kernel circuit.
        We will append state prep (X gates) to copies of this.
        """
        qr_in = QuantumRegister(9, 'in') # 0-7: Neighbors, 8: Center
        qr_cnt = QuantumRegister(4, 'count')
        qr_out = QuantumRegister(1, 'out')
        cr = ClassicalRegister(1, 'res')
        
        qc = QuantumCircuit(qr_in, qr_cnt, qr_out, cr)
        
        # 1. Compute Neighbor Count (Sum n0..n7 into qr_cnt)
        # We use a simple adder ladder.
        # Since we only need to distinguish 2, 3, and "other", we don't need a full 8-bit adder.
        # But a 4-bit adder is safe.
        
        # Simple incrementer for each neighbor
        for i in range(8):
            # Increment qr_cnt by 1 controlled by qr_in[i]
            # C-Increment-4
            # Toffoli ladder:
            # if in[i]:
            #   if c0 and c1 and c2: flip c3
            #   if c0 and c1: flip c2
            #   if c0: flip c1
            #   flip c0
            
            ctrl = qr_in[i]
            c = qr_cnt
            
            qc.mcx([ctrl, c[0], c[1], c[2]], c[3])
            qc.mcx([ctrl, c[0], c[1]], c[2])
            qc.ccx(ctrl, c[0], c[1])
            qc.cx(ctrl, c[0])
            
        # 2. Apply Rules
        # Survival: Count=2 (0010) AND Center=1
        # Birth: Count=3 (0011) AND Center=0
        # Combined: (Count=2 AND Center=1) OR (Count=3) -> Alive
        # Wait, standard rules:
        # Alive if: (Count=2 AND Center=1) OR (Count=3)
        
        c = qr_cnt
        center = qr_in[8]
        target = qr_out[0]
        
        # Condition A: Count == 3 (0011)
        # Logic: if c3=0, c2=0, c1=1, c0=1 -> Flip Target
        qc.x([c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3]], target) # If 3, Alive
        qc.x([c[2], c[3]])
        
        # Condition B: Count == 2 (0010) AND Center == 1
        # Logic: if c3=0, c2=0, c1=1, c0=0 AND center=1 -> Flip Target
        # Note: If we already flipped for 3, we shouldn't flip again?
        # No, Count cannot be 2 and 3 at the same time. So we can just accumulate flips (XOR).
        # Target starts at 0.
        
        qc.x([c[0], c[2], c[3]])
        qc.mcx([c[0], c[1], c[2], c[3], center], target) # If 2 and Alive, Stay Alive
        qc.x([c[0], c[2], c[3]])
        
        # 3. Measure
        qc.measure(target, cr)
        
        return qc

    def insertGlider(self, top_left):
        x, y = top_left
        glider = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for dy, dx in glider:
            ny, nx = (y + dy) % self.N, (x + dx) % self.N
            self.grid[ny, nx] = 1

    def evolve(self):
        """
        Evolves the grid by one step using Quantum Batch Processing.
        """
        circuits = []
        
        # 1. Prepare Circuits for all cells
        for y in range(self.N):
            for x in range(self.N):
                # Get neighborhood
                # Neighbors: TL, T, TR, L, R, BL, B, BR
                # Center: (x,y)
                
                # Map to input qubits:
                # 0: x-1, y-1
                # 1: x,   y-1
                # 2: x+1, y-1
                # 3: x-1, y
                # 4: x+1, y
                # 5: x-1, y+1
                # 6: x,   y+1
                # 7: x+1, y+1
                # 8: x,   y (Center)
                
                offsets = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1, 0),           (1, 0),
                    (-1, 1),  (0, 1),  (1, 1)
                ]
                
                # Create a shallow copy of the kernel
                # We will prepend State Prep
                # Actually, better to compose?
                # Or just create a new circuit and append the kernel instruction?
                # Let's use `compose`.
                
                prep_qc = QuantumCircuit(14, 1)
                
                # Encode Neighbors
                for i, (dx, dy) in enumerate(offsets):
                    nx, ny = (x + dx) % self.N, (y + dy) % self.N
                    if self.grid[ny, nx] == 1:
                        prep_qc.x(i) # qr_in[i]
                        
                # Encode Center
                if self.grid[y, x] == 1:
                    prep_qc.x(8) # qr_in[8]
                    
                # Append Kernel
                # We can convert kernel to instruction for speed
                # But compose is fine for N=32 (1000 circuits)
                
                full_qc = prep_qc.compose(self._kernel_circuit)
                circuits.append(full_qc)
                
        # 2. Run Batch
        # 1024 circuits, 1 shot each.
        # Optimization: Use `memory=True`? No, get_counts is fine.
        job = self.backend.run(circuits, shots=1)
        results = job.result()
        
        # 3. Update Grid
        new_grid = np.zeros_like(self.grid)
        
        for i, qc in enumerate(circuits):
            counts = results.get_counts(i)
            # counts is {'0': 1} or {'1': 1}
            val = int(list(counts.keys())[0])
            
            y, x = i // self.N, i % self.N
            new_grid[y, x] = val
            
        self.grid = new_grid

    def getGrid(self):
        return self.grid

if __name__ == "__main__":
    # Demo
    N = 10
    game = QuantumKernelConway(N)
    game.insertGlider((1, 1))
    
    print("Initial:")
    print(game.grid)
    
    game.evolve()
    print("\nStep 1:")
    print(game.grid)
