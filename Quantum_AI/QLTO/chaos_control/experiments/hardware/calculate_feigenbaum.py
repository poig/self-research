import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def analyze_bifurcations(data):
    # Sort data by r
    sorted_r = sorted(data.keys(), key=lambda x: float(x))
    r_values = np.array([float(r) for r in sorted_r])
    
    # We need to detect when the number of clusters changes
    # Simple metric: Variance of the steady state values
    # Or: Minimum distance between points?
    
    bifurcations = []
    regimes = []
    
    # Threshold for detecting splitting
    # If standard deviation is low -> Period 1
    # If standard deviation jumps -> Period 2
    
    variances = []
    
    print(f"{'r':<10} {'StdDev':<10} {'Clusters':<10}")
    print("-" * 30)
    
    for r in sorted_r:
        energies = data[r]
        # Take last 20 points as steady state
        steady = np.array(energies[-20:])
        std = np.std(steady)
        variances.append(std)
        
        # Simple clustering to count peaks
        # Histogram approach
        hist, bins = np.histogram(steady, bins=10)
        # Count non-zero bins separated by zero bins? 
        # Easier: use KMeans or just look at std dev jumps
        
        print(f"{r:<10} {std:.4f}")

    # Plot variance to find jumps
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, variances, '-o')
    plt.xlabel('Control Parameter r')
    plt.ylabel('Std Dev of Steady State')
    plt.title('Variance of Trajectory vs r (Bifurcation Detection)')
    plt.grid(True)
    plt.savefig('variance_plot.png')
    print("Saved variance_plot.png")
    
    # Manual inspection of the variance plot will likely be needed to pick exact r1, r2
    # But let's try to find the first jump
    
    # r1: Transition P1 -> P2
    # Look for sharp increase in variance
    
    # r2: Transition P2 -> P4
    # Harder to see in variance alone, might need to look at "splitting" of the two branches
    
    return r_values, variances

if __name__ == "__main__":
    filepath = "../data/hardware/scan_qbraid_rigetti_ankaa_3_20251204.json"
    data = load_data(filepath)
    analyze_bifurcations(data)
