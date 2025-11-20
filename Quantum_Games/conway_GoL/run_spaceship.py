import sys
import os
import numpy as np

# Ensure imports work
sys.path.append(os.getcwd())

from classical_conway import GameOfLife
from kernel_conway import QuantumKernelConway
# from quantum_conway import OptimizedPingPong # Too slow for large grids
# from sliding_window_conway import SlidingWindowConway # Better, but still heavy for 300 width

def main():
    print("--- Quantum Spaceship Runner ---")
    
    # 1. Load TXT using Classical Engine
    # Use absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "spaceship.txt")
    with open(txt_path, 'r') as f:
        txt_str = f.read()
        
    # Initialize Classical Game to parse TXT
    gol = GameOfLife(N=128) 
    
    print(f"Loading {txt_path}...")
    gol.insertFromPlainText(txt_str, pad=5)
    
    # Get the grid
    c_grid = gol.getGrid()
    # Crop to relevant area to save qubits?
    # 64x64 = 4096 qubits. That's too big for MPS on a laptop probably.
    # We need to crop it tightly around the spaceship.
    
    # Find bounding box
    rows, cols = np.where(c_grid == 1)
    if len(rows) > 0:
        min_y, max_y = np.min(rows), np.max(rows)
        min_x, max_x = np.min(cols), np.max(cols)
        
        # Add padding
        pad = 2
        min_y = max(0, min_y - pad)
        max_y = min(c_grid.shape[0], max_y + pad + 1)
        min_x = max(0, min_x - pad)
        max_x = min(c_grid.shape[1], max_x + pad + 1)
        
        crop_grid = c_grid[min_y:max_y, min_x:max_x]
    else:
        print("Error: Empty grid loaded.")
        return

    H_crop, W_crop = crop_grid.shape
    print(f"Cropped Grid Size: {W_crop}x{H_crop} ({W_crop*H_crop} cells)")
    
    if W_crop * H_crop > 100:
        print("Warning: High qubit count! Simulation might be slow.")
        
    # Convert to string list for Quantum Engine
    initial_strings = []
    for y in range(H_crop):
        row_str = ""
        for x in range(W_crop):
            row_str += "#" if crop_grid[y, x] == 1 else "."
        initial_strings.append(row_str)
        print(row_str)

    # 2. Run Quantum Simulation
    print("\nInitializing Quantum Engine (Hybrid Kernel)...")
    # We use Hybrid Kernel because 279x258 is too big for full statevector simulation.
    # Hybrid Kernel uses O(1) qubits (14 qubits).
    engine = QuantumKernelConway(max(W_crop, H_crop)) # KernelConway takes N for NxN, but handles rectangular internally if we patch it, or we just make it big enough.
    # Actually KernelConway uses a grid of size N*N. Let's use the crop size.
    # But KernelConway init creates a square grid.
    # Let's just use the max dimension.
    
    # Load the grid into the engine
    # QuantumKernelConway.grid is a numpy array.
    engine.grid = np.zeros((engine.N, engine.N), dtype=int)
    # Copy crop
    engine.grid[0:H_crop, 0:W_crop] = crop_grid
    
    frames = 1
    print(f"Running for {frames} step(s)...")
    
    try:
        for _ in range(frames):
            engine.evolve()
            
        res_grid = engine.getGrid()
        
        print(f"\nResult T={frames}:")
        # Visualize Crop
        for y in range(H_crop):
            row_str = ""
            for x in range(W_crop):
                row_str += "#" if res_grid[y, x] == 1 else "."
            print(row_str)
            
        print("\nVerdict: Spaceship loaded and processed using Hybrid Quantum Kernel.")
        
    except Exception as e:
        print(f"Simulation Failed: {e}")

if __name__ == "__main__":
    main()
