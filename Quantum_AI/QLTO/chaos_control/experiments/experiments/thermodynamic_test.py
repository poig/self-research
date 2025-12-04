"""
thermodynamic_test.py

Verifies the thermodynamic cost of the chaos control.
Calculates the entropy production vs energy extraction efficiency.

Theory (Paper 1):
    Delta E <= eta * I(S:A)
    
    Where I(S:A) is the mutual information (bounded by Holevo limit chi <= 1).
    Landauer cost: E_dissipated >= k_B T * I(S:A) * ln(2)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
KB_T_LN2 = 1.0  # Normalized units

def entropy_binary(p):
    """Binary entropy function H(p)."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def analyze_thermodynamics(trajectory, r_values):
    """
    Analyzes the thermodynamic efficiency of the optimization.
    
    Args:
        trajectory: List of x values (normalized energy)
        r_values: List of r values used (if adaptive)
    """
    
    n_steps = len(trajectory)
    
    # 1. Energy Improvement (Work Extracted)
    # We assume the goal is x -> 0 (Ground State)
    # Work = x_initial - x_final
    work_extracted = trajectory[0] - trajectory[-1]
    
    # 2. Information Cost
    # Each step is a measurement.
    # The information gained is I(S:A) = H(P_meas) - H(Noise)
    # For pure states, I = H(P(|1>))
    
    total_info_bits = 0
    entropy_production = 0
    
    for i in range(n_steps - 1):
        x = trajectory[i]
        # In our map x_{n+1} = r * sin^2(pi*x)
        # The measurement probability was P = sin^2(pi*x)
        # But we only observe x_{n+1}. 
        # Reconstruct P: P = x_{n+1} / r
        
        r = r_values[i] if isinstance(r_values, list) else r_values
        if r == 0: continue
            
        p_meas = trajectory[i+1] / r
        
        # Clamp for numerical stability
        p_meas = np.clip(p_meas, 1e-9, 1.0 - 1e-9)
        
        # Information gained in this step
        info = entropy_binary(p_meas)
        total_info_bits += info
        
        # Landauer cost
        entropy_production += info * KB_T_LN2

    # 3. Efficiency
    eta = work_extracted / (total_info_bits * KB_T_LN2) if total_info_bits > 0 else 0
    
    return {
        'work_extracted': work_extracted,
        'total_info_bits': total_info_bits,
        'entropy_production': entropy_production,
        'efficiency': eta
    }

import json
import glob

DATA_DIR = Path('../data/hardware')

def load_latest_data():
    # Find the most recent json file
    files = []
    files.extend(glob.glob(str(DATA_DIR / 'scan_stacked_*.json')))
    files.extend(glob.glob(str(DATA_DIR / 'scan_qbraid_*.json')))
    files.extend(glob.glob(str(DATA_DIR / 'scan_batch_*.json')))
    
    if not files:
        raise FileNotFoundError("No data files found in ../data/hardware")
        
    # Prioritize Rigetti if available
    rigetti_files = [f for f in files if 'rigetti' in f.lower() or 'ankaa' in f.lower()]
    if rigetti_files:
        latest_file = max(rigetti_files, key=lambda f: Path(f).stat().st_mtime)
        print(f"Loading data from (Hardware Priority): {latest_file}")
    else:
        latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
        print(f"Loading data from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    return data

def main():
    print("Thermodynamic Verification (Hardware Data)")
    print("==========================================")
    
    try:
        data = load_latest_data()
    except Exception as e:
        print(f"Could not load hardware data: {e}")
        print("Falling back to simulation...")
        # ... (keep simulation fallback if needed, or just fail)
        return

    # Extract r values
    r_keys = sorted(data.keys(), key=float)
    
    # Select a stable r (low r) and a chaotic r (high r)
    # r values are typically 0.6 to 0.8
    r_stable_key = r_keys[0] # Lowest r
    r_chaos_key = r_keys[-1] # Highest r
    
    r_stable = float(r_stable_key)
    r_chaos = float(r_chaos_key)
    
    traj_stable = data[r_stable_key]
    traj_chaos = data[r_chaos_key]
    
    print(f"Analyzing Stable Trajectory (r={r_stable:.4f})")
    stats_stable = analyze_thermodynamics(traj_stable, r_stable)
    
    print(f"Analyzing Chaotic Trajectory (r={r_chaos:.4f})")
    stats_chaos = analyze_thermodynamics(traj_chaos, r_chaos)
    
    print("\nResults:")
    print(f"Stable (r={r_stable:.4f}):")
    print(f"  Work: {stats_stable['work_extracted']:.4f}")
    print(f"  Info Cost: {stats_stable['total_info_bits']:.4f} bits")
    print(f"  Efficiency: {stats_stable['efficiency']:.4f}")
    
    print(f"\nChaotic (r={r_chaos:.4f}):")
    print(f"  Work: {stats_chaos['work_extracted']:.4f}")
    print(f"  Info Cost: {stats_chaos['total_info_bits']:.4f} bits")
    print(f"  Efficiency: {stats_chaos['efficiency']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(traj_chaos, 'r.-', label=f'Chaos (r={r_chaos:.2f})', alpha=0.6)
    plt.plot(traj_stable, 'b.-', label=f'Stable (r={r_stable:.2f})', alpha=0.6)
    plt.title('Hardware Trajectories')
    plt.xlabel('Step')
    plt.ylabel('Energy x')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(['Stable', 'Chaos'], [stats_stable['efficiency'], stats_chaos['efficiency']])
    bars[0].set_color('blue')
    bars[1].set_color('red')
    plt.title('Thermodynamic Efficiency $\eta$')
    plt.ylabel('Efficiency (Work / Info Cost)')
    
    plt.tight_layout()
    output_path = Path('../figures/thermo_efficiency_hardware.png')
    plt.savefig(output_path)
    print(f"\nSaved figure to {output_path}")

if __name__ == "__main__":
    main()
