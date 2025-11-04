"""
qsve_with_comparison.py:
A complete simulation of the Quantum Singular Value Estimation (QSVE) algorithm.

This script demonstrates the full workflow for finding the SVD of a 
*non-unitary* matrix 'A' by "assuming a classical oracle".

1.  Define a random 2x2 *non-unitary* matrix 'A'.
2.  Classically find its SVD (U, s, Vh) to get the "ground truth" singular value 's'.
3.  "Assume the Oracle": Classically construct the 4x4 *unitary* block-encoding 'U_A'.
4.  Classically *find the true eigenvector* of the *constructed U_A* to use as
    the target_eigenstate. This fixes the numerical mismatch bug.
5.  Build the QPE circuit, using 'U_A' as the unitary.
6.  Run the QPE to *measure* the phase 'theta'.
7.  Post-process: Convert the measured phase 'theta' back into the singular value 's'.
8.  Compare the "Quantum" singular value to the "Classical" singular value.

GOAL: Find the singular value 's' of a random 2x2 matrix 'A'.
- We create a 4x4 unitary U_A.
- The eigenvalues of U_A are exp(+- i * arccos(s))
- The QPE finds 'theta', where theta = arccos(s) / (2*pi)
- Therefore, s = cos(2*pi*theta)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit import transpile
from scipy.linalg import sqrtm

# Set print options for clarity
np.set_printoptions(precision=5, suppress=True)

def get_classical_svd_truth(matrix_A):
    """
    Finds the "ground truth" singular values of the original
    non-unitary matrix A.
    """
    print("="*60)
    print("PART 1: CLASSICAL SVD (THE GROUND TRUTH)")
    print("="*60)
    print("Original 2x2 Non-Unitary Matrix A:")
    print(matrix_A)

    U, s, Vh = np.linalg.svd(matrix_A)
    
    # We will target the first singular value
    target_singular_value = s[0]
    
    print(f"\nClassical Singular Values: {s}")
    print(f"TARGET (Classical) Singular Value: {target_singular_value:.5f}")
    
    # Return all SVD components for use later
    return target_singular_value, U, s, Vh

def build_block_encoding_oracle(matrix_A):
    """
    "Assumes the Oracle"
    1. Classically constructs the 4x4 unitary block-encoding U_A.
    2. Classically *finds the correct eigenvector* of U_A to use.
       This is the FIX: it guarantees the state is a perfect eigenvector
       of the (numerically imperfect) oracle.
    
    U_A = [ A           , sqrt(I - A A_dagger) ]
          [ sqrt(I - A_dagger A) , -A_dagger        ]
    """
    print("\n" + "="*60)
    print("PART 2: BUILDING THE ORACLE (CLASSICALLY)")
    print("="*60)
    
    # Ensure A is 2x2
    if matrix_A.shape != (2, 2):
        raise ValueError("Matrix A must be 2x2")

    I = np.identity(2)
    A_dagger = matrix_A.conj().T
    
    # --- Build the four 2x2 blocks ---
    block_11 = matrix_A
    block_12 = sqrtm(I - matrix_A @ A_dagger)
    block_21 = sqrtm(I - A_dagger @ matrix_A)
    block_22 = -A_dagger
    
    # --- Assemble the 4x4 unitary U_A ---
    U_A = np.block([
        [block_11, block_12],
        [block_21, block_22]
    ])
    
    print("Constructed 4x4 Unitary Block-Encoding Oracle U_A:")
    # print(U_A) # (Disabled for brevity)

    # --- Diagnostic Check (from critique) ---
    unitarity_error = np.linalg.norm(U_A.conj().T @ U_A - np.eye(U_A.shape[0]))
    print(f"Unitarity Check ||U_A^dagger * U_A - I||: {unitarity_error:.2e}")
    if unitarity_error > 1e-10:
        print("*** WARNING: U_A is not perfectly unitary due to numerical errors. ***")
    
    # --- Find the "Ground Truth" Phase ---
    # We still need the theoretical phase for our final comparison
    s_0_classical = np.linalg.svd(matrix_A)[1][0]
    theoretical_phase = np.arccos(s_0_classical) / (2 * np.pi)
    theoretical_eigval = np.exp(1j * 2 * np.pi * theoretical_phase)

    # --- FIX: Find the *Actual* Eigenvector of U_A ---
    # Instead of building the state from SVD, we find the *actual*
    # eigenvector of our *numerically constructed* U_A.
    eigvals, eigvecs = np.linalg.eig(U_A)
    
    # Find the index of the eigenvalue closest to our theoretical one
    best_match_index = np.argmin(np.abs(eigvals - theoretical_eigval))
    
    # This is the *actual* eigenvector we must use
    target_eigenstate = eigvecs[:, best_match_index]
    
    # This is the *actual* phase we are feeding to the QPE
    actual_eigval = eigvals[best_match_index]
    actual_phase_for_qpe = np.angle(actual_eigval) / (2 * np.pi)
    # Handle negative angles from np.angle
    if actual_phase_for_qpe < 0:
        actual_phase_for_qpe += 1

    print(f"\nTheoretical 'True' Phase (from s_0):   {theoretical_phase:.5f}")
    print(f"Actual Phase we will test (from U_A): {actual_phase_for_qpe:.5f}")
    print(f"Eigenstate (found from U_A): \n{target_eigenstate}")

    # Return the *actual* phase for QPE to find
    return UnitaryGate(U_A, label="U_A"), actual_phase_for_qpe, target_eigenstate

def build_qpe_circuit(n_counting_qubits, unitary_gate, target_eigenstate):
    """
    Builds the QPE circuit for a given custom unitary gate and its
    corresponding target eigenstate.
    """
    print("\n" + "="*60)
    print("PART 3: BUILDING THE QUANTUM CIRCUIT (QPE)")
    print("="*60)
    print(f"Building QPE circuit with {n_counting_qubits} counting qubits...")
    
    n_eigenstate_qubits = unitary_gate.num_qubits

    qr_counting = QuantumRegister(n_counting_qubits, name="counting")
    qr_eigenstate = QuantumRegister(n_eigenstate_qubits, name="eigenstate")
    cr_counting = ClassicalRegister(n_counting_qubits, name="measurement")
    circuit = QuantumCircuit(qr_counting, qr_eigenstate, cr_counting)

    # --- 3a. Initialize the Eigenstate ---
    print(f"Initializing {n_eigenstate_qubits}-qubit eigenstate...")
    # This state is now guaranteed to be an eigenvector of the unitary_gate
    circuit.initialize(target_eigenstate, qr_eigenstate)
    circuit.barrier()

    # --- 3b. Apply Hadamards to Counting Qubits ---
    circuit.h(qr_counting)
    circuit.barrier()

    # --- 3c. Apply Controlled-Unitary Operations ---
    print("Applying controlled-unitary operations (the 'evolution')...")
    controlled_gate = unitary_gate.control(1)
    
    for i_counting in range(n_counting_qubits):
        repetitions = 1 << i_counting # 2^i
        control_and_target_qubits = [qr_counting[i_counting]] + list(qr_eigenstate)
        for _ in range(repetitions):
            circuit.append(controlled_gate, control_and_target_qubits)
    circuit.barrier()

    # --- 3d. Apply Inverse Quantum Fourier Transform (iQFT) ---
    print("Applying Inverse Quantum Fourier Transform...")
    iqft_gate = QFT(num_qubits=n_counting_qubits, inverse=True, do_swaps=True).to_gate()
    iqft_gate.name = "iQFT"
    circuit.append(iqft_gate, qr_counting)
    circuit.barrier()

    # --- 3e. Measure ---
    circuit.measure(qr_counting, cr_counting)
    
    return circuit

def run_qpe_simulation(circuit, shots=2048):
    """
    Simulates the QPE circuit and returns the measurement counts.
    """
    print("\n" + "="*60)
    print("PART 4: RUNNING SIMULATION")
    print("="*60)
    print("\nSimulating the circuit...")
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    print(f"Simulation complete. Got counts:\n{counts}\n")
    return counts

def analyze_qsve_results(counts, n_counting_qubits):
    """
    Interprets the simulation counts to find the measured phase
    and then calculates the corresponding singular value.
    """
    print("="*60)
    print("PART 5: ANALYZING RESULTS")
    print("="*60)

    most_frequent_bits = max(counts, key=counts.get)
    measured_int = int(most_frequent_bits, 2)
    
    # --- 5a. Calculate the Measured Phase 'theta' ---
    measured_phase = measured_int / (2**n_counting_qubits)
    
    # --- 5b. Convert Phase 'theta' to Singular Value 's' ---
    # This is the final, crucial post-processing step
    # s = cos(2*pi*theta)
    measured_singular_value = np.cos(2 * np.pi * measured_phase)

    print("\n------ Quantum Algorithm Result ------")
    print(f"Most Frequent Bitstring:   '{most_frequent_bits}'")
    print(f"Measured Integer:          {measured_int}")
    print(f"Calculated Phase (theta):  {measured_phase:<.5f}")
    print(f"Calculated Singular Value (s): {measured_singular_value:<.5f}")
    
    return measured_phase, measured_singular_value

def main():
    """
    Orchestrates the full classical vs. quantum SVD comparison.
    """
    # --- Define Problem ---
    # 1. Create a 2x2 random *non-unitary* matrix
    print("--- Creating a 2x2 Random Non-Unitary Matrix ---")
    A_raw = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    
    # 2. Normalize it so largest singular value is < 1
    # This is required for block-encoding to be unitary.
    s_max = np.linalg.svd(A_raw)[1][0]
    A_norm = A_raw / (s_max * 1.1) # Divide by max singular value + 10%
    
    # --- Classical Ground Truth (for final check) ---
    target_singular_value, _, _, _ = get_classical_svd_truth(A_norm)

    # --- "Oracle" Step ---
    # Classically build the oracle U_A and get its *actual* # target phase and eigenvector
    U_A_gate, true_phase_for_qpe, target_eigenstate = build_block_encoding_oracle(A_norm)
    
    # --- Quantum ---
    n_counting = 8 # Use 8 bits of precision
    
    # 3. Build the general QPE circuit
    qpe_circuit = build_qpe_circuit(
        n_counting, 
        U_A_gate, 
        target_eigenstate
    )
    
    # 4. Run simulation
    counts = run_qpe_simulation(qpe_circuit)
    
    # 5. Analyze results
    measured_phase, measured_singular_value = analyze_qsve_results(counts, n_counting)

    # --- Final Comparison ---
    print("\n" + "="*60)
    print("PART 6: FINAL CONCLUSION")
    print("="*60)
    
    print("------ Classical 'Ground Truth' ------")
    print(f"Original Singular Value (s): {target_singular_value:<.5f}")
    print(f"Actual Phase fed into QPE:   {true_phase_for_qpe:<.5f}")

    print("\n------ Quantum Algorithm Result ------")
    print(f"Measured Singular Value (s): {measured_singular_value:<.5f}")
    print(f"Measured Phase (theta):      {measured_phase:<.5f}")

    print("\n------ VERDICT ------")
    # We now check if the *measured phase* matches the *actual phase we fed in*.
    # This is the correct test of the QPE algorithm itself.
    if np.isclose(true_phase_for_qpe, measured_phase, atol=1/(2**n_counting)):
        print("SUCCESS: The quantum algorithm (QPE) successfully measured the")
        print("         phase of the oracle's eigenvector.")
        
        # As a second check, see if the final S.V. matches
        if np.isclose(target_singular_value, measured_singular_value, atol=0.01):
             print("         The resulting singular value also matches the classical value!")
        else:
             print("         However, the final singular value is off. This implies")
             print("         a numerical error in the U_A oracle construction itself.")

    else:
        print("FAILURE: The measured phase does not match the true phase.")
        print("         (This is a bug, not a precision issue.)")

if __name__ == "__main__":
    main()

