import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from typing import List

# --- Configuration ---
GRID_W = 4
GRID_H = 4
N_CELLS = GRID_W * GRID_H  # 16 Cells
NUM_BUFFERS = 4            # A -> B -> C -> D -> A ...
FRAMES_TO_SIMULATE = 2    # Run enough frames to see movement

class QuantumGliderEngine:
    def __init__(self):
        print(f"Initializing 4-Buffer Ring Engine ({GRID_W}x{GRID_H} Grid)...")
        self.sim = AerSimulator(method='matrix_product_state')

    def _get_neighbors_indices(self, idx):
        """Hard Boundary Neighbors for 4x4 Grid"""
        row = idx // GRID_W
        col = idx % GRID_W
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                    neighbors.append(nr * GRID_W + nc)
        return neighbors

    def apply_physics(self, qc, src, tgt):
        """
        Applies Conway's Rules from Source Register to Target Register.
        Target is assumed |0...0> (Clean).
        """
        # 1. Birth (Dead->Alive if N=3)
        for i in range(N_CELLS):
            neighbors = self._get_neighbors_indices(i)
            # Logic: If Self=0 (Dead)
            qc.x(src[i]) 
            
            # Check for exactly 3 neighbors
            for on_neighbors in itertools.combinations(neighbors, 3):
                off_neighbors = [n for n in neighbors if n not in on_neighbors]
                
                # Flip OFF neighbors for control check
                for off in off_neighbors: qc.x(src[off])
                
                # Control on Self(0->1), ON(1), OFF(0->1)
                controls = [src[i]] + [src[n] for n in on_neighbors] + [src[n] for n in off_neighbors]
                qc.mcx(controls, tgt[i])
                
                # Uncompute OFF
                for off in off_neighbors: qc.x(src[off])
            
            qc.x(src[i]) # Uncompute Self

        # 2. Survival (Alive->Alive if N=2 or N=3)
        for i in range(N_CELLS):
            neighbors = self._get_neighbors_indices(i)
            # Check for 2 or 3 neighbors
            for count in [2, 3]:
                for on_neighbors in itertools.combinations(neighbors, count):
                    off_neighbors = [n for n in neighbors if n not in on_neighbors]
                    
                    for off in off_neighbors: qc.x(src[off])
                    
                    # Control on Self(1), ON(1), OFF(0->1)
                    controls = [src[i]] + [src[n] for n in on_neighbors] + [src[n] for n in off_neighbors]
                    qc.mcx(controls, tgt[i])
                    
                    for off in off_neighbors: qc.x(src[off])

    def run_glider_simulation(self, initial_state: List[int]):
        # Create 4 Quantum Buffers (A, B, C, D)
        # Total active qubits = 16 * 4 = 64. 
        # Note: Simulator handles this because we Measure & Reset sequentially.
        q_bufs = [QuantumRegister(N_CELLS, f'buf_{i}') for i in range(NUM_BUFFERS)]
        c_hist = [ClassicalRegister(N_CELLS, f'c_{t}') for t in range(FRAMES_TO_SIMULATE + 1)]
        
        qc = QuantumCircuit(*q_bufs, *c_hist)
        
        # --- T=0: Initialize Buffer A ---
        for i, bit in enumerate(initial_state):
            if bit == 1: qc.x(q_bufs[0][i])
        qc.measure(q_bufs[0], c_hist[0])
        
        # --- The Infinite Ring Loop ---
        for t in range(FRAMES_TO_SIMULATE):
            # Current Index (Source) -> Next Index (Target)
            src_i = t % NUM_BUFFERS
            tgt_i = (t + 1) % NUM_BUFFERS
            
            src_reg = q_bufs[src_i]
            tgt_reg = q_bufs[tgt_i]
            
            # 1. Apply Physics
            self.apply_physics(qc, src_reg, tgt_reg)
            
            # 2. Save Result
            qc.measure(tgt_reg, c_hist[t+1])
            
            # 3. DISSIPATE (Reset Source)
            # This clears the buffer so it can be reused 3 steps later
            # as the Target for buffer C.
            qc.reset(src_reg)

        # Execute
        print(f"Simulating {FRAMES_TO_SIMULATE} frames on 4x4 Grid...")
        # Using shots=1 (Deterministic Logic)
        res = self.sim.run(qc.decompose(reps=4), shots=1).result().get_counts()
        
        # Parse
        raw = list(res.keys())[0].split()
        history = raw[::-1]
        
        print("\n--- GLIDER SIMULATION (4-BUFFER RING) ---")
        for t, frame_str in enumerate(history):
            bits = [int(c) for c in frame_str][::-1] # Reverse to visual order
            print(f"\nTime T={t} (Buffer {['A','B','C','D'][t % 4]}):")
            self.print_grid(bits)

    def print_grid(self, state: List[int]):
        for r in range(GRID_H):
            row = ""
            for c in range(GRID_W):
                idx = r * GRID_W + c
                row += " O " if state[idx] else " . "
            print(row)

if __name__ == "__main__":
    engine = QuantumGliderEngine()
    
    # Glider Pattern (Top-Left)
    # . O . .
    # . . O .
    # O O O .
    # . . . .
    init = [0] * N_CELLS
    # Row 0
    init[1] = 1
    # Row 1
    init[1*4 + 2] = 1
    # Row 2
    init[2*4 + 0] = 1
    init[2*4 + 1] = 1
    init[2*4 + 2] = 1
    
    engine.run_glider_simulation(init)