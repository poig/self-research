import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from typing import List

# --- Configuration ---
GRID_W = 3
GRID_H = 3
N_CELLS = GRID_W * GRID_H
FRAMES_TO_SIMULATE = 6

class QuantumBlinkerEngine:
    def __init__(self):
        print(f"Initializing {GRID_W}x{GRID_H} Quantum Engine (Hard Boundaries)...")
        self.sim = AerSimulator()

    def _get_neighbors_indices(self, idx):
        """
        Returns neighbors for a grid with Hard Boundaries (No wrapping).
        """
        row = idx // GRID_W
        col = idx % GRID_W
        neighbors = []
        
        # Check all 8 directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                
                nr, nc = row + dr, col + dc
                
                # Boundary Check
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                    neighbors.append(nr * GRID_W + nc)
        return neighbors

    def _apply_exact_logic(self, qc, src, tgt, cell_idx, valid_counts):
        """
        Generates the MCX gates for EXACT neighbor counts.
        To strictly enforce 'Count == 3', we must ensure the 
        other neighbors are explicitly 0.
        """
        neighbors = self._get_neighbors_indices(cell_idx)
        
        # We iterate over every valid count (e.g., 3)
        for count in valid_counts:
            # Get all combinations of neighbors that could be ON
            for on_neighbors in itertools.combinations(neighbors, count):
                off_neighbors = [n for n in neighbors if n not in on_neighbors]
                
                # Build the Control Logic
                # 1. Check ON neighbors (Standard controls)
                controls = list(on_neighbors)
                
                # 2. Check OFF neighbors (Wrap in X gates)
                for off in off_neighbors:
                    qc.x(src[off])
                    controls.append(off)
                
                # 3. Add Self to controls (State of the cell itself)
                # This function assumes we've already handled the 'Self' state check externally
                # or we add it here.
                # Let's assume the caller handles the 'Self' check (Alive vs Dead logic).
                
                # Apply Logic
                # Control on Source, Flip Target
                qc.mcx([src[c] for c in controls], tgt[cell_idx])
                
                # Uncompute X gates on OFF neighbors
                for off in off_neighbors:
                    qc.x(src[off])

    def apply_physics_layer(self, qc: QuantumCircuit, src: QuantumRegister, tgt: QuantumRegister):
        """
        Applies Birth and Survival Logic for the whole grid.
        """
        for i in range(N_CELLS):
            # --- FORCE 1: BIRTH (Dead -> Alive) ---
            # Self must be 0. Neighbors must be exactly 3.
            qc.x(src[i]) # Check if Dead
            
            # Apply logic for neighbors == 3
            self._apply_exact_logic(qc, src, tgt, i, valid_counts=[3])
            
            qc.x(src[i]) # Uncompute Self check

            # --- FORCE 2: SURVIVAL (Alive -> Alive) ---
            # Self must be 1. Neighbors must be 2 or 3.
            # Note: We don't check Self=1 here explicitly if we include it in the control list
            # But strictly, the logic is: IF Self=1 AND (N=2 or N=3) -> Target=1.
            # If we just controlled on neighbors, we'd flip Target even if Self was 0.
            
            # So we use the Self qubit as a control.
            # But `_apply_exact_logic` iterates neighbors.
            # We need to add Self to the MCX inside.
            
            # Custom loop for Survival to include Self=1 control
            neighbors = self._get_neighbors_indices(i)
            for count in [2, 3]:
                 for on_neighbors in itertools.combinations(neighbors, count):
                    off_neighbors = [n for n in neighbors if n not in on_neighbors]
                    
                    # Controls: Self=1, On_Neighbors=1, Off_Neighbors=0
                    
                    # Flip OFF neighbors
                    for off in off_neighbors: qc.x(src[off])
                    
                    controls = [src[i]] + [src[on] for on in on_neighbors] + [src[off] for off in off_neighbors]
                    qc.mcx(controls, tgt[i])
                    
                    # Uncompute OFF neighbors
                    for off in off_neighbors: qc.x(src[off])

    def run_simulation(self, initial_state: List[int], steps: int):
        # 2 Buffers of 9 Qubits
        q_a = QuantumRegister(N_CELLS, 'buf_a')
        q_b = QuantumRegister(N_CELLS, 'buf_b')
        
        # Classical Registers for History
        c_history = [ClassicalRegister(N_CELLS, f'c_{t}') for t in range(steps + 1)]
        
        qc = QuantumCircuit(q_a, q_b, *c_history)
        
        # Initialize T=0
        for i, bit in enumerate(initial_state):
            if bit == 1: qc.x(q_a[i])
        qc.measure(q_a, c_history[0])
        
        # The Ping-Pong Loop
        for t in range(steps):
            if t % 2 == 0: # A -> B
                self.apply_physics_layer(qc, q_a, q_b)
                qc.measure(q_b, c_history[t+1])
                qc.reset(q_a)
            else: # B -> A
                self.apply_physics_layer(qc, q_b, q_a)
                qc.measure(q_a, c_history[t+1])
                qc.reset(q_b)
                
        print(f"Compiling 3x3 Simulation ({steps} frames)...")
        res = self.sim.run(qc, shots=1).result().get_counts()
        
        # Parse and Visualise
        raw_data = list(res.keys())[0]
        frames = raw_data.split()
        history = frames[::-1]
        
        print("\n--- 3x3 BLINKER SIMULATION ---")
        for t, frame_str in enumerate(history):
            # String is q8...q0. Reverse to get q0...q8
            bits = [int(c) for c in frame_str][::-1]
            
            print(f"\nTime T={t}:")
            self.print_grid(bits)
            
    def print_grid(self, state: List[int]):
        # Print as 3x3
        for r in range(GRID_H):
            row_str = ""
            for c in range(GRID_W):
                idx = r * GRID_W + c
                char = 'O' if state[idx] else '.'
                row_str += f" {char} "
            print(row_str)

if __name__ == "__main__":
    engine = QuantumBlinkerEngine()
    
    # Initial State: Vertical Line (Blinker)
    # . O .
    # . O .
    # . O .
    # Indices: 1, 4, 7
    init_state = [0]*9
    init_state[1] = 1
    init_state[4] = 1
    init_state[7] = 1
    
    engine.run_simulation(init_state, steps=FRAMES_TO_SIMULATE)