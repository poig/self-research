import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

class FractalLandscapeEngine:
    def __init__(self, n_bits_x=3, n_bits_y=3):
        self.nx = n_bits_x
        self.ny = n_bits_y
        
        # Registers
        self.qr_x = QuantumRegister(self.nx, 'x')
        self.qr_y = QuantumRegister(self.ny, 'y')
        self.qr_state = QuantumRegister(1, 'state')
        self.cr = ClassicalRegister(self.nx + self.ny + 1, 'readout')
        
        self.qc = QuantumCircuit(self.qr_x, self.qr_y, self.qr_state, self.cr)
        
        # *** CRITICAL FIX ***
        # 1. STRETCH THE CANVAS FIRST (Time T=0)
        # We must exist everywhere before we can check constraints anywhere.
        self.qc.h(self.qr_x)
        self.qc.h(self.qr_y)

    def fuse_fractal_rule(self):
        print(f"Fusing Sierpinski Logic into Superposition...")
        # 2. FUSE LOGIC (Time T=1)
        for x in range(2**self.nx):
            for y in range(2**self.ny):
                if (x & y) == 0:
                    self._fuse_gate_at_coord(x, y)

    def _fuse_gate_at_coord(self, x, y):
        # Address Decoding (Sensor)
        x_str = format(x, f'0{self.nx}b')[::-1]
        y_str = format(y, f'0{self.ny}b')[::-1]

        # Activate Sensor
        for i, bit in enumerate(x_str):
            if bit == '0': self.qc.x(self.qr_x[i])
        for i, bit in enumerate(y_str):
            if bit == '0': self.qc.x(self.qr_y[i])
            
        # Apply Force (Actuator)
        controls = list(self.qr_x) + list(self.qr_y)
        self.qc.mcx(controls, self.qr_state[0])
        
        # Deactivate Sensor (Uncompute)
        for i, bit in enumerate(x_str):
            if bit == '0': self.qc.x(self.qr_x[i])
        for i, bit in enumerate(y_str):
            if bit == '0': self.qc.x(self.qr_y[i])

    def run_simulation(self):
        # 3. OBSERVE (Time T=2)
        self.qc.measure(self.qr_x, self.cr[:self.nx])
        self.qc.measure(self.qr_y, self.cr[self.nx : self.nx+self.ny])
        self.qc.measure(self.qr_state, self.cr[-1])
        
        sim = AerSimulator()
        return sim.run(self.qc, shots=8192).result().get_counts()

if __name__ == "__main__":
    bits = 4
    engine = FractalLandscapeEngine(n_bits_x=bits, n_bits_y=bits)
    engine.fuse_fractal_rule()
    counts = engine.run_simulation()
    
    print("\n--- FRACTAL LOGIC SURFACE (Corrected) ---")
    grid = [['.' for _ in range(2**bits)] for _ in range(2**bits)]
    
    for k, count in counts.items():
        # k format: "State Y X" (e.g. "1 011 001")
        # Remove spaces if present
        k = k.replace(" ", "")
        
        state = k[0]
        y_bits = k[1 : 1+bits] # Y is middle
        x_bits = k[1+bits : 1+2*bits] # X is last
        
        if state == '1':
            grid[int(y_bits, 2)][int(x_bits, 2)] = '█'
            
    # print("  0 1 2 3 4 5 6 7 (X)")
    print(f"{bits}x{bits} Grid")
    print("  " + " ".join([str(i) for i in range(2**bits)]))
    for y in range(2**bits):
        print(f"{y} " + " ".join(grid[y]))