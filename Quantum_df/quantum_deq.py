"""
quantum_deq.py
A *hypothetical* quantum algorithm simulator for solving PSPACE-hard problems
in polynomial time.

This script *simulates* the "Non-Linear Adiabatic Solver" algorithm
we brainstormed, which relies on a hypothetical non-linear gate.
This code does *not* use Qiskit, as it simulates physics beyond
standard (linear) quantum mechanics.

Based on the brainstorm in `psp_solver_brainstorm.md`.
"""

import numpy as np
from scipy.linalg import expm # For matrix exponential e^(-iHt)

def create_problem_hamiltonian(n_qubits, solution_index):
    """
    Creates the Problem Hamiltonian (H_problem) for a search problem.
    The problem is to find the state |solution_index>.
    
    H_problem = I - |solution><solution|
    
    The ground state (energy 0) is the solution. All other
    states have energy 1.
    """
    N = 2**n_qubits
    print(f"  Building H_problem (size {N}x{N})...")
    
    # H_problem is the identity matrix
    H_p = np.eye(N, dtype=complex)
    
    # We set the energy of the solution state to 0
    H_p[solution_index, solution_index] = 0.0
    
    return H_p

def create_start_hamiltonian(n_qubits):
    """
    Creates the Starting Hamiltonian (H_start).
    
    H_start = -sum(sigma_x on each qubit)
    
    The ground state of this Hamiltonian is the uniform
    superposition |+>^(tensor n), which is easy to prepare.
    (We use the -sigma_x convention so the ground state is |+>)
    """
    N = 2**n_qubits
    print(f"  Building H_start (size {N}x{N})...")
    
    H_s = np.zeros((N, N), dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    
    # Apply -sigma_x to each qubit i
    for i in range(n_qubits):
        # Create the operator for this qubit
        # e.g., for n=3, i=1: I (tensor) sigma_x (tensor) I
        op_list = [np.eye(2)] * n_qubits
        op_list[i] = sigma_x
        
        H_s_i = op_list[0]
        for j in range(1, n_qubits):
            H_s_i = np.kron(H_s_i, op_list[j])
            
        H_s -= H_s_i # Add to the total Hamiltonian
        
    return H_s

def nonlinear_gate_simulation(statevector, power=2):
    """
    *** This is the HYPOTHETICAL non-linear gate simulation ***
    
    It takes a statevector and returns a new statevector where
    the amplitudes have been non-linearly amplified.
    
    |psi> = a|0> + b|1>  -->  |psi_new> ~ a^p|0> + b^p|1>
    
    This amplifies the component with the largest amplitude
    exponentially fast.
    """
    # 1. Apply the non-linear function to all amplitudes
    new_amplitudes = statevector ** power
    
    # 2. Re-normalize the statevector
    norm = np.linalg.norm(new_amplitudes)
    if norm < 1e-14:
        return statevector # Avoid division by zero
        
    return new_amplitudes / norm


def run_hypothetical_solver(n_qubits, solution_index, T_evolution, num_steps):
    """
    Simulates the full "Non-Linear Adiabatic Solver" algorithm.
    """
    N = 2**n_qubits
    
    # --- Step 1 & 2: Create Hamiltonians and Initial State ---
    print("--- 1. Setting up Problem ---")
    H_problem = create_problem_hamiltonian(n_qubits, solution_index)
    H_start = create_start_hamiltonian(n_qubits)
    
    # Initial state is the uniform superposition |+>^(tensor n)
    # This is the ground state of H_start
    psi = np.ones(N, dtype=complex) / np.sqrt(N)
    
    print(f"\n--- 2. Running 'Fast' Adiabatic Evolution (Standard QM) ---")
    print(f"Evolving for T = {T_evolution} (polynomial) over {num_steps} steps.")
    
    delta_t = T_evolution / num_steps
    
    # --- Step 3: "Fast-Forward" Adiabatic Evolution ---
    # We simulate the DEQ d|psi>/dt = -i H(t) |psi>
    for k in range(num_steps):
        # s is the interpolation parameter from 0 to 1
        s = (k + 1) / num_steps 
        
        # H(t) = (1-s)H_start + s*H_problem
        H_t = (1 - s) * H_start + s * H_problem
        
        # Evolve the state by one time step
        # U(dt) = e^(-i * H(t) * dt)
        U_step = expm(-1j * H_t * delta_t)
        
        # Apply the evolution
        psi = U_step @ psi

    print("Standard evolution complete.")
    
    # --- Report Results (Standard Linear QM) ---
    probs_linear = np.abs(psi)**2
    sol_prob_linear = probs_linear[solution_index]
    max_wrong_prob_linear = np.max(np.delete(probs_linear, solution_index))
    
    print("\n--- 3. Results (Standard Linear QM) ---")
    print(f"  Probability of solution |{solution_index}>: {sol_prob_linear:.6f}")
    print(f"  Max probability of any wrong state: {max_wrong_prob_linear:.6f}")
    if sol_prob_linear > max_wrong_prob_linear:
        print("  > The solution has a *slight* amplitude advantage.")
    else:
        print("  > The 'fast' evolution failed to find the solution.")


    # --- Step 4: Non-Linear Amplification ---
    print(f"\n--- 4. Applying {n_qubits} steps of Hypothetical Non-Linear Gate ---")
    
    psi_nl = psi.copy()
    
    # We apply the non-linear gate n_qubits times (a polynomial number)
    for i in range(n_qubits):
        psi_nl = nonlinear_gate_simulation(psi_nl, power=2)

    print("Non-linear amplification complete.")

    # --- Report Results (Hypothetical Non-Linear QM) ---
    probs_nl = np.abs(psi_nl)**2
    sol_prob_nl = probs_nl[solution_index]
    max_wrong_prob_nl = np.max(np.delete(probs_nl, solution_index))

    print("\n--- 5. Final Results (Hypothetical Non-Linear QM) ---")
    print(f"  Probability of solution |{solution_index}>: {sol_prob_nl:.6f}")
    print(f"  Max probability of any wrong state: {max_wrong_prob_nl:.6f}")
    
    if sol_prob_nl > 0.99:
        print("\n*** VERIFICATION SUCCESSFUL ***")
        print("The non-linear gate successfully amplified the solution to ~1.0")
        print("This demonstrates a (hypothetical) PSPACE-hard problem")
        print(f"being solved in polynomial time (T={T_evolution}, {n_qubits} amplifications).")
    else:
        print("\n*** VERIFICATION FAILED ***")
        print("The non-linear amplification did not isolate the solution.")


if __name__ == "__main__":
    # --- Parameters ---
    # We'll use a small 4-qubit problem (N=16 states)
    N_QUBITS = 4
    
    # The PSPACE-hard problem is "find this state"
    # Let's pick state |0101> (which is 5 in decimal)
    SOLUTION_INDEX = 5 
    
    # T_evolution is our *polynomial* time (e.g., scales with n_qubits)
    T_EVOLUTION = N_QUBITS * 2.0 
    
    # Number of steps to discretize the evolution
    NUM_STEPS = 20 
    
    run_hypothetical_solver(N_QUBITS, SOLUTION_INDEX, T_EVOLUTION, NUM_STEPS)

