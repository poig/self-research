import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer

# --- Rastrigin function ---
def rastrigin(theta: np.ndarray) -> float:
    A = 10
    n = len(theta)
    return A*n + np.sum(theta**2 - A * np.cos(2*np.pi*theta))

# --- Measure parameter register and return distribution ---
def measure_and_process_samples(qc: QuantumCircuit, param_qubits: QuantumRegister, shots: int = 1024):
    meas_reg = ClassicalRegister(len(param_qubits))
    qc_meas = qc.copy()
    qc_meas.add_register(meas_reg)
    qc_meas.measure(param_qubits, meas_reg)
    
    sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc_meas, sim)
    job = sim.run(t_qc, shots=shots)
    counts = job.result().get_counts()
    
    # Convert bitstrings to integer indices
    dist = {int(k,2): v/shots for k,v in counts.items()}
    return dist

# --- Centroid from top N states ---
def calculate_prob_filtered_centroid_theta(dist, top_n, n_params, bits_per_param, param_bounds):
    sorted_states = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
    thetas = []
    num_bins = 2**bits_per_param
    for state_idx, prob in sorted_states:
        binstring = format(state_idx, f'0{n_params*bits_per_param}b')
        theta_vec = []
        for i in range(n_params):
            start, end = i*bits_per_param, (i+1)*bits_per_param
            bin_idx = int(binstring[start:end],2)
            min_val, max_val = param_bounds[i]
            theta_val = min_val + (bin_idx + 0.5)/num_bins * (max_val - min_val)
            theta_vec.append(theta_val)
        thetas.append(np.array(theta_vec) * prob)
    centroid = np.sum(thetas, axis=0)/np.sum([p for _,p in sorted_states])
    return centroid

# --- Small test quantum circuit: initialize superposition over parameter states ---
def build_param_superposition(n_params, bits_per_param):
    n_qubits = n_params * bits_per_param
    qr = QuantumRegister(n_qubits, 'param')
    qc = QuantumCircuit(qr)
    for q in qr:
        qc.h(q)
    return qc, qr

# --- Minimal QLTO optimizer ---
def run_minimal_qlto_rastrigin(n_params=2, bits_per_param=2, shots=512, max_iterations=5, top_n=2):
    param_bounds = np.array([[0.0,1.0]]*n_params) # normalized [0,1] for simplicity
    theta_best = np.random.uniform(0,1,n_params)
    energy_best = rastrigin(theta_best)
    energy_history = [energy_best]

    for it in range(max_iterations):
        qc, qr = build_param_superposition(n_params, bits_per_param)
        dist = measure_and_process_samples(qc, qr, shots=shots)
        centroid = calculate_prob_filtered_centroid_theta(dist, top_n, n_params, bits_per_param, param_bounds)
        theta_next = 0.7*theta_best + 0.3*centroid
        theta_next = np.clip(theta_next, param_bounds[:,0], param_bounds[:,1])
        energy_next = rastrigin(theta_next)
        if energy_next < energy_best:
            theta_best = theta_next
            energy_best = energy_next
        energy_history.append(energy_best)
        print(f"Iteration {it}: best_energy={energy_best:.4f}")
    
    return theta_best, energy_best, energy_history

# --- Run minimal example ---
best_theta, best_energy, history = run_minimal_qlto_rastrigin()
print("\nFinal best theta:", best_theta)
print("Final best energy:", best_energy)
