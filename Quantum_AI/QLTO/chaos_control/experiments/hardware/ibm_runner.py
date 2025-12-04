"""
ibm_runner.py

Script to run the VQA bifurcation scan on real IBM Quantum hardware.
This is the "Smoking Gun" experiment for Paper 3.

It uses Qiskit Runtime Primitives (Sampler/Estimator) to execute the 
Hadamard test circuits on a real backend (e.g., ibm_brisbane).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from pathlib import Path

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService

# Configuration
BACKEND_NAME = 'ibm_fez'  # Or 'ibmq_qasm_simulator' for testing
SHOTS = 4096
DATA_DIR = Path('../data/hardware')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_hadamard_test_circuit(phi):
    """
    Creates a Hadamard test circuit for a given phase phi.
    In a real VQA, 'phi' would be E(theta)*tau.
    Here we simulate the effective map directly on hardware to prove
    the measurement noise doesn't destroy the bifurcation.
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.h(0)
    qc.measure_all()
    return qc

def run_bifurcation_scan(service, backend_name, r_values, steps_per_r=50):
    """
    Runs the bifurcation scan on hardware.
    
    For each r:
      1. Initialize x = 0.4
      2. Iterate x_{n+1} = r * P(|1>)
      3. P(|1>) comes from RUNNING THE CIRCUIT on hardware
    """
    print(f"Connecting to backend: {backend_name}")
    backend = service.backend(backend_name)
    sampler = Sampler(mode=backend)
    
    results = {}
    
    print(f"Starting BATCHED scan over {len(r_values)} r-values...")
    
    # Initialize state for all r-values
    current_xs = {r: 0.4 for r in r_values}
    trajectories = {r: [] for r in r_values}
    
    # 1. Create ONE stacked circuit for all r-values
    # We use Space Parallelization: Qubit i runs experiment for r_values[i]
    from qiskit.circuit import ParameterVector
    
    num_experiments = len(r_values)
    print(f"Stacking {num_experiments} experiments onto a single circuit...")
    
    # Parameter vector for phases: phi_0, phi_1, ...
    phis = ParameterVector('phi', num_experiments)
    
    qc_stacked = QuantumCircuit(num_experiments)
    for i in range(num_experiments):
        # Hadamard test on qubit i
        qc_stacked.h(i)
        qc_stacked.rz(phis[i], i)
        qc_stacked.h(i)
        
    qc_stacked.measure_all()
    
    # 2. Transpile ONCE
    print(f"Transpiling stacked circuit for {backend_name}...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc_stacked)
    
    print(f"Running in Job Mode on {backend_name}...")
    sampler = Sampler(mode=backend)
    
    # We iterate through time steps
    for step in range(steps_per_r):
        print(f"Batch Step {step+1}/{steps_per_r}...", end='\r')
        
        # 3. Prepare parameters for this step
        # We need a single list of float values [phi_0, phi_1, ...]
        current_phis = []
        r_list = list(r_values) # Keep order consistent
        
        for r in r_list:
            x = current_xs[r]
            # FIX: To implement x_{n+1} = r * sin^2(pi * x), we need P(|1>) = sin^2(pi * x).
            # The physical probability is sin^2(phi/2).
            # Therefore, phi/2 = pi * x  =>  phi = 2 * pi * x.
            phi = 2 * np.pi * x
            current_phis.append(phi)
            
        # 4. Run the single stacked circuit
        # PUB format: (circuit, parameter_values)
        # parameter_values must be shape (1, num_params) for a single shot
        # or just a 1D list if we are running one configuration
        job = sampler.run([(isa_qc, current_phis)], shots=SHOTS)
        result = job.result()
        
        # 5. Process results (Marginalization)
        # The result contains counts for the full N-bit string
        # We need P(|1>) for each qubit individually
        
        # Get bitstrings (keys of counts)
        counts = result[0].data.meas.get_counts()
        
        # Initialize counts for each qubit
        ones_counts = np.zeros(num_experiments)
        total_counts = 0
        
        for bitstring, count in counts.items():
            total_counts += count
            # Qiskit bitstrings are little-endian (qubit 0 is rightmost)
            # But get_counts() returns them reversed compared to array indexing usually?
            # Actually, standard Qiskit is: bitstring[0] is qubit N-1, bitstring[-1] is qubit 0
            # Let's handle this carefully.
            
            for i in range(num_experiments):
                # The bit for qubit i is at index -(i+1)
                bit_val = bitstring[-(i+1)]
                if bit_val == '1':
                    ones_counts[i] += count
                    
        # Update trajectories
        for i, r in enumerate(r_list):
            p1 = ones_counts[i] / total_counts
            
            # Feedback update
            x_new = r * p1
            current_xs[r] = x_new
            trajectories[r].append(x_new)
            
        # Save intermediate results periodically
        if step % 5 == 0:
            save_data = {str(r): traj for r, traj in trajectories.items()}
            with open(DATA_DIR / f'scan_stacked_{backend_name}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(save_data, f)
                
    # Final save
    save_data = {str(r): traj for r, traj in trajectories.items()}
    with open(DATA_DIR / f'scan_stacked_{backend_name}_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
        json.dump(save_data, f)
            
    return trajectories

def main():
    # 1. Authenticate
    # Ensure you have saved your token: QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_TOKEN")
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        print("Error connecting to IBM Quantum. Make sure your API token is saved.")
        print("Run: QiskitRuntimeService.save_account(channel='ibm_quantum', token='...')")
        return

    # 2. Define scan range (focus on bifurcation region)
    r_values = np.linspace(0.6, 0.8, 20)  # 20 points from 0.6 to 0.8
    
    # 3. Run
    data = run_bifurcation_scan(service, BACKEND_NAME, r_values)
    
    print("\nScan complete. Data saved.")

if __name__ == "__main__":
    main()
