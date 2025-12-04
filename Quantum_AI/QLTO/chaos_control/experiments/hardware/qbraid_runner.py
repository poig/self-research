"""
qbraid_runner.py

Script to run the VQA bifurcation scan on real Quantum Hardware via qBraid.
Automatically selects the LEAST BUSY device to minimize wait time.
"""

import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import qbraid
from qiskit import QuantumCircuit

# Configuration
SHOTS = 4096
DATA_DIR = Path('../data/hardware')
DATA_DIR.mkdir(parents=True, exist_ok=True)

from qbraid.runtime import QbraidProvider
provider = QbraidProvider()
provider.save_config()

def get_cheapest_qpu(min_qubits=20):
    """
    Finds the cheapest real QPU.
    Heuristic: Rigetti is typically cheaper than IonQ or IQM on public clouds.
    """
    print("Querying qBraid for available devices...")
    devices = provider.get_devices()
    
    candidates = []
    for d in devices:
        # Check status
        try:
            status_obj = d.status() if callable(d.status) else d.status
            status_str = str(status_obj).upper()
            is_online = 'ONLINE' in status_str and 'UNAVAILABLE' not in status_str
        except Exception:
            is_online = False
            
        dev_id = d.id
        
        # Filter out simulators
        is_simulator = ('sim' in dev_id.lower() or 
                        'sv1' in dev_id.lower() or 
                        'tn1' in dev_id.lower() or 
                        'dm1' in dev_id.lower())
                        
        if 'quera' in dev_id.lower(): continue # Analog
        if 'lucy' in dev_id.lower(): continue # Too small

        if is_online and not is_simulator:
            # Cost Heuristic (lower score is cheaper)
            cost_score = 100
            if 'rigetti' in dev_id.lower():
                cost_score = 1  # Usually cheapest ($0.35/task + $0.00035/shot)
            elif 'iqm' in dev_id.lower():
                cost_score = 2
            elif 'ionq' in dev_id.lower():
                cost_score = 3  # Usually most expensive ($0.30/task + $0.01/shot)
            
            print(f"Candidate: {dev_id} (Cost Score: {cost_score})")
            candidates.append((d, cost_score))
            
    if not candidates:
        print("No online QPUs found. Falling back to simulator 'aws_sv1'.")
        return provider.get_device("aws_sv1")
        
    # Sort by cost score
    candidates.sort(key=lambda x: x[1])
    
    best_device, score = candidates[0]
    print(f"Selected cheapest device: {best_device.id}")
    return best_device

def run_bifurcation_scan(device, r_values, steps_per_r=50):
    """
    Runs the bifurcation scan on the selected qBraid device.
    """
    print(f"Starting STACKED scan on {device.id}...")
    
    # Initialize state
    current_xs = {r: 0.4 for r in r_values}
    trajectories = {r: [] for r in r_values}
    
    # We iterate through time steps
    for step in range(steps_per_r):
        print(f"Step {step+1}/{steps_per_r}...", end='\r')
        
        # 1. Construct circuit for this step
        # Since qBraid might not support Qiskit ParameterVector efficiently across all backends
        # (transpilation issues), we recreate the circuit with bound values.
        # This is safer for cross-provider compatibility.
        
        num_experiments = len(r_values)
        qc = QuantumCircuit(num_experiments)
        
        r_list = list(r_values)
        for i, r in enumerate(r_list):
            x = current_xs[r]
            # Map: x_{n+1} = r * sin^2(pi * x)
            # Prob: P(|1>) = sin^2(phi/2)
            # phi = 2 * pi * x
            phi = 2 * np.pi * x
            
            qc.h(i)
            qc.rz(phi, i)
            qc.h(i)
            
        qc.measure_all()
        
        # 2. Run on qBraid
        # qBraid handles transpilation to the target backend format (Braket, Qir, etc.)
        job = device.run(qc, shots=SHOTS)
        result = job.result()
        
        # 3. Process results
        # qBraid result object usually wraps the provider result.
        # We need to extract counts.
        try:
            # Try Qiskit style first (most common for these backends via qBraid)
            if hasattr(result, 'data') and hasattr(result.data, 'get_counts'):
                counts = result.data.get_counts()
            elif hasattr(result, 'get_counts'):
                counts = result.get_counts()
            elif hasattr(result, 'measurement_counts'):
                # Braket style (deprecated wrapper?)
                counts = result.measurement_counts()
            else:
                # Fallback
                counts = result.data.counts
        except Exception:
             # If all else fails, try the deprecated one or inspect
             counts = result.measurement_counts()
            
        # Marginalize
        ones_counts = np.zeros(num_experiments)
        total_counts = 0
        
        for bitstring, count in counts.items():
            total_counts += count
            # Handle bitstring format (some are '0101', some '0 1 0 1')
            clean_bits = bitstring.replace(' ', '')[::-1] # Reverse to match Qiskit little-endian if needed
            # Actually, let's be careful. Qiskit is little-endian (qubit 0 is rightmost).
            # Braket is big-endian (qubit 0 is leftmost).
            # qBraid usually preserves the provider convention.
            
            # Let's assume Qiskit convention since we sent a Qiskit circuit
            # qBraid transpiler usually handles mapping, but let's check provider.
            is_braket = 'aws' in device.id
            
            if is_braket:
                # Braket: "100" -> q0=1, q1=0, q2=0 (Big Endian)
                bits = bitstring.replace(' ', '')
                for i in range(num_experiments):
                    if i < len(bits) and bits[i] == '1':
                        ones_counts[i] += count
            else:
                # Qiskit/IBM: "001" -> q0=1 (Little Endian)
                bits = bitstring.replace(' ', '')
                for i in range(num_experiments):
                    # qubit i is at index -(i+1)
                    if i < len(bits) and bits[-(i+1)] == '1':
                        ones_counts[i] += count

        # Update trajectories
        for i, r in enumerate(r_list):
            p1 = ones_counts[i] / total_counts
            x_new = r * p1
            current_xs[r] = x_new
            trajectories[r].append(x_new)
            
        # Save intermediate
        if step % 5 == 0:
            save_data = {str(r): traj for r, traj in trajectories.items()}
            with open(DATA_DIR / f'scan_qbraid_{device.id.replace(":", "_")}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(save_data, f)

    # Final save
    save_data = {str(r): traj for r, traj in trajectories.items()}
    with open(DATA_DIR / f'scan_qbraid_{device.id.replace(":", "_")}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
        json.dump(save_data, f)
        
    return trajectories

def main():
    try:
        # 1. Select Device
        device = get_cheapest_qpu(min_qubits=20)
        
        # 2. Define scan range
        r_values = np.linspace(0.6, 0.8, 20)
        
        # 3. Run
        run_bifurcation_scan(device, r_values)
        
        print("\nScan complete. Data saved.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
