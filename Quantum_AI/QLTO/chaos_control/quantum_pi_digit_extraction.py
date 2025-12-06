from qiskit.quantum_info import Statevector
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

def binary_expansion_pi(n_bits):
    """
    Returns ground truth binary expansion of Pi's fractional part.
    Pi = 3.14159... = 11. (binary) 00100100001111110110...
    """
    # pi - 3 = 0.14159...
    val = np.pi - 3
    bits = []
    for _ in range(n_bits):
        val *= 2
        bit = int(val)
        bits.append(bit)
        val -= bit
    return bits

def quantum_digit_extraction(n_digits, use_statevector=True):
    """
    Extracts binary digits. 
    If use_statevector=True, uses exact linear algebra (infinite shots).
    This proves whether the limitation is Shot Noise or Phase Ambiguity.
    """
    
    print(f"Extracting {n_digits} digits via {'Statevector (Exact)' if use_statevector else 'Simulator (Shots)'}...")
    
    # 1. Simulator setup
    simulator = Aer.get_backend('qasm_simulator')
    
    # 2. Initial Condition
    x0 = np.pi - 3.0
    
    extracted_bits = []
    
    print(f"\n{'Step':<5} | {'Phase (rad)':<12} | {'Prob(1)':<10} | {'Bit':<5} | {'True'}")
    print("-" * 55)
    
    tr_bits = binary_expansion_pi(n_digits)

    for k in range(n_digits):
        qc = QuantumCircuit(1)
        qc.h(0)
        
        # Evolution U^(2^k)
        angle = (2**k) * (2 * np.pi * x0)
        # Normalize angle to [0, 2pi) for display
        disp_angle = angle % (2*np.pi)
        
        qc.p(angle, 0) 
        qc.h(0)
        
        if use_statevector:
            # EXACT CALCULATION
            sv = Statevector(qc)
            probs = sv.probabilities()
            p1 = probs[1]
        else:
            # SHOT NOISE SIMULATION
            qc.measure_all()
            job = simulator.run(transpile(qc, simulator), shots=8192)
            counts = job.result().get_counts()
            p1 = counts.get('1', 0) / 8192.0
            
        # DECODING
        # If P(1) < 0.5 -> Bit 0
        # If P(1) > 0.5 -> Bit 1
        # Ambiguity: If P(1) approx 0.5, we are sensitive to noise.
        
        measured_bit = 1 if p1 > 0.5 else 0
        
        match = "✓" if measured_bit == tr_bits[k] else "✗"
        print(f"{k+1:<5} | {disp_angle:<12.4f} | {p1:<10.4f} | {measured_bit:<5} | {match}")
        
        extracted_bits.append(measured_bit)
        
    return extracted_bits

def main():
    N_DIGITS = 10
    
    # Run with Statevector (Exact)
    q_bits = quantum_digit_extraction(N_DIGITS, use_statevector=True)
    
    # Ground Truth
    true_bits = binary_expansion_pi(N_DIGITS)
    
    match_count = sum(1 for q, t in zip(q_bits, true_bits) if q == t)
     
    print("-" * 55)
    print(f"Accuracy: {match_count}/{N_DIGITS} ({match_count/N_DIGITS*100:.1f}%)")
    
    print("\nANALYSIS:")
    print("Even with infinite precision (Statevector), accuracy might not be 100%.")
    print("Why? Because measurement P(|1>) = sin^2(phi/2).")
    print("If phi is near pi/2 (prob 0.5), the bit is ambiguous in the Z-basis.")
    print("To fix this, we would need 'Phase Tomography' (measure X and Y).")

if __name__ == "__main__":
    main()
