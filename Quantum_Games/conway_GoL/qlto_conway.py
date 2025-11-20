import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from typing import List

# --- Configuration ---
# 3x3 Grid for the Blinker (Hard Boundaries to prevent self-interaction)
GRID_W = 3
GRID_H = 3
N_CELLS = GRID_W * GRID_H
# Simulate 3 Frames (T0 -> T1 -> T2) coherently
FRAMES = 3 

class CoherentLandscapeEngine:
    def __init__(self):
        print("Initializing Coherent Landscape Engine (Spacetime Crystal)...")
        self.sim = AerSimulator()

    def _get_neighbors_indices(self, idx):
        """Hard Boundary Neighbors for 3x3 Grid"""
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

    def apply_landscape_layer(self, qc: QuantumCircuit, src_reg: QuantumRegister, tgt_reg: QuantumRegister):
        """
        Constructs the 'Digital Cliff' Landscape between Time T and T+1.
        This is a UNITARY operation (Reversible).
        Since 'tgt_reg' starts at |0>, this acts as a coherent copy-paste of the next state.
        """
        # We construct the landscape by superimposing two force fields:
        # 1. The Birth Field (Operates on Empty Space)
        # 2. The Survival Field (Operates on Living Matter)
        
        for i in range(N_CELLS):
            neighbors = self._get_neighbors_indices(i)
            
            # --- LANDSCAPE A: BIRTH (Cliff at N=3) ---
            # Condition: Source=0 AND Neighbors=3
            # In a high-dim continuous landscape, this is a sharp peak at coords (0, 3).
            # We implement this 'Peak' using Multi-Controlled-X (The Digital Cliff).
            
            # "Rotate into the valley of Death" (Check if dead)
            qc.x(src_reg[i]) 
            
            # "Check for Resonance at N=3"
            # We iterate all combinations of 3 neighbors
            for on_neighbors in itertools.combinations(neighbors, 3):
                off_neighbors = [n for n in neighbors if n not in on_neighbors]
                
                # "Shape the Landscape" (Invert OFF neighbors)
                for off in off_neighbors: qc.x(src_reg[off])
                
                # "The Cliff Edge" (If we are exactly at this point in phase space...)
                controls = [src_reg[i]] + [src_reg[n] for n in on_neighbors] + [src_reg[n] for n in off_neighbors]
                
                # "...Fall into the next state" (Flip Target)
                qc.mcx(controls, tgt_reg[i])
                
                # "Restore the Landscape" (Uncompute)
                for off in off_neighbors: qc.x(src_reg[off])
            
            qc.x(src_reg[i]) # Uncompute Self

            # --- LANDSCAPE B: SURVIVAL (Plateau at N=2,3) ---
            # Condition: Source=1 AND Neighbors in {2, 3}
            
            # We iterate valid counts
            for count in [2, 3]:
                for on_neighbors in itertools.combinations(neighbors, count):
                    off_neighbors = [n for n in neighbors if n not in on_neighbors]
                    
                    # Landscape shaping
                    for off in off_neighbors: qc.x(src_reg[off])
                    
                    # The Cliff Edge
                    # Controls: Self(1), ON(1), OFF(0->1)
                    controls = [src_reg[i]] + [src_reg[n] for n in on_neighbors] + [src_reg[n] for n in off_neighbors]
                    qc.mcx(controls, tgt_reg[i])
                    
                    # Restore
                    for off in off_neighbors: qc.x(src_reg[off])

    def run_coherent_simulation(self, initial_state: List[int]):
        # 1. Allocate the Spacetime Crystal
        # One register per timeframe.
        # T0 is Fixed Input. T1 is Phase-Evolved from T0. T2 is Phase-Evolved from T1.
        q_frames = [QuantumRegister(N_CELLS, f't{t}') for t in range(FRAMES)]
        c_readout = [ClassicalRegister(N_CELLS, f'c{t}') for t in range(FRAMES)]
        
        qc = QuantumCircuit(*q_frames, *c_readout)
        
        # 2. Initialize T=0 (The Boundary Condition)
        for i, bit in enumerate(initial_state):
            if bit == 1: qc.x(q_frames[0][i])
            
        # 3. Construct the Logic Landscapes (The Laws of Physics)
        print(f"Building Coherent Landscape (Depth: {FRAMES})...")
        
        # Layer 1: T0 -> T1
        self.apply_landscape_layer(qc, q_frames[0], q_frames[1])
        
        # Layer 2: T1 -> T2
        self.apply_landscape_layer(qc, q_frames[1], q_frames[2])
        
        # 4. Measure the Crystal (Collapse the timeline)
        # Note: We measure ALL frames to see the history
        for t in range(FRAMES):
            qc.measure(q_frames[t], c_readout[t])
            
        # 5. Execute
        print("Simulating Blinker Evolution...")
        res = self.sim.run(qc, shots=1).result().get_counts()
        
        # 6. Visualization
        # Qiskit string is "c2 c1 c0"
        raw = list(res.keys())[0].split()
        history = raw[::-1] # T0, T1, T2
        
        print("\n--- COHERENT BLINKER RESULT ---")
        for t, frame_str in enumerate(history):
            bits = [int(c) for c in frame_str][::-1] # Reverse for grid mapping
            print(f"\nTime T={t}:")
            self.print_grid(bits)

    def print_grid(self, state: List[int]):
        for r in range(GRID_H):
            row = ""
            for c in range(GRID_W):
                idx = r * GRID_W + c
                row += " O " if state[idx] else " . "
            print(row)

if __name__ == "__main__":
    engine = CoherentLandscapeEngine()
    
    # Initial State: Vertical Blinker (Center column)
    # . O .
    # . O .
    # . O .
    # Indices: 1, 4, 7
    init = [0]*9
    init[1] = 1
    init[4] = 1
    init[7] = 1
    
    engine.run_coherent_simulation(init)