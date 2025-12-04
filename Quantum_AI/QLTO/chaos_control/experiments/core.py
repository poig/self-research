"""
Core functions for Quantum Chaos Control experiments.
Shared utilities for all Paper 3 experiments.
"""

import numpy as np
from pathlib import Path

# Figures directory
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================
# THE SIN² MAP - Effective 1D VQA Dynamics
# ============================================================

def sin2_map(x, r):
    """
    VQA effective map: x_{n+1} = r·sin²(πx)
    
    This arises from:
    - Hadamard test gradient sensing: P(|1⟩) = sin²(Eτ/2)
    - Gradient descent update with learning rate γ
    - Effective r = γ·τ·E_max/π
    """
    return r * np.sin(np.pi * x) ** 2


def compute_trajectory(x0, r, n_steps):
    """Compute orbit trajectory for sin² map."""
    traj = [x0]
    x = x0
    for _ in range(n_steps):
        x = sin2_map(x, r)
        traj.append(x)
    return np.array(traj)


def compute_lyapunov(r, x0=0.5, n_transient=500, n_compute=500):
    """
    Compute Lyapunov exponent for sin² map at given r.
    
    λ > 0 → Chaotic (sensitive dependence on initial conditions)
    λ < 0 → Stable (trajectories converge)
    """
    x = x0
    for _ in range(n_transient):
        x = sin2_map(x, r)
    
    lyap_sum = 0
    for _ in range(n_compute):
        x = sin2_map(x, r)
        # Derivative: d/dx [r·sin²(πx)] = r·π·sin(2πx)
        deriv = abs(r * np.pi * np.sin(2 * np.pi * x))
        if deriv > 1e-12:
            lyap_sum += np.log(deriv)
    
    return lyap_sum / n_compute


# ============================================================
# FEIGENBAUM CONSTANTS
# ============================================================

# Universal Feigenbaum constant
FEIGENBAUM_DELTA = 4.669201609102990671853203820466

# Critical r values for sin² map period-doubling cascade
R_PERIOD_1 = 0.50   # Stable fixed point
R_PERIOD_2 = 0.63   # Period-2 bifurcation
R_PERIOD_4 = 0.70   # Period-4 bifurcation
R_PERIOD_8 = 0.72   # Period-8 bifurcation
R_CHAOS = 0.731     # Onset of chaos (r∞)


# ============================================================
# QUANTUM PERIOD DETECTION (Shor-like approach)
# ============================================================

def encode_trajectory_to_quantum_state(trajectory, n_qubits=4):
    """
    Encode a VQA trajectory into a quantum superposition.
    
    |ψ⟩ = (1/√N) Σ_n |n⟩|x_n⟩
    
    where n is the iteration index and x_n is the discretized trajectory value.
    """
    N = 2**n_qubits
    n_traj = min(len(trajectory), N)
    
    # Discretize trajectory to n_qubits resolution
    x_discrete = np.round(trajectory[:n_traj] * (N-1)).astype(int)
    x_discrete = np.clip(x_discrete, 0, N-1)
    
    # Create amplitude vector for 2*n_qubits system (index + value)
    total_qubits = 2 * n_qubits
    state = np.zeros(2**total_qubits, dtype=complex)
    
    for n, x_val in enumerate(x_discrete):
        idx = n * N + x_val
        state[idx] = 1.0
    
    # Normalize
    state = state / np.linalg.norm(state)
    return state


def apply_qft_to_index_register(state, n_qubits=4):
    """
    Apply QFT to the index (iteration) register.
    This extracts the period information (like Shor's algorithm).
    """
    N = 2**n_qubits
    
    # Build QFT matrix
    omega = np.exp(2j * np.pi / N)
    qft_matrix = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    
    # Apply QFT to index register
    new_state = np.zeros_like(state)
    
    for x_val in range(N):
        indices_in = [n * N + x_val for n in range(N)]
        amplitudes_in = state[indices_in]
        amplitudes_out = qft_matrix @ amplitudes_in
        indices_out = [k * N + x_val for k in range(N)]
        new_state[indices_out] = amplitudes_out
    
    return new_state


def measure_frequency_register(state_after_qft, n_qubits=4):
    """Measure the frequency register to detect period."""
    N = 2**n_qubits
    
    freq_probs = np.zeros(N)
    for k in range(N):
        for x_val in range(N):
            idx = k * N + x_val
            freq_probs[k] += np.abs(state_after_qft[idx])**2
    
    return freq_probs


def detect_period_quantum(trajectory, n_qubits=4):
    """
    Quantum-inspired period detection using QFT on trajectory.
    
    Returns: (freq_probs, detected_period)
    """
    state = encode_trajectory_to_quantum_state(trajectory, n_qubits)
    state_qft = apply_qft_to_index_register(state, n_qubits)
    freq_probs = measure_frequency_register(state_qft, n_qubits)
    
    N = 2**n_qubits
    
    # Find significant peaks (excluding DC)
    peaks = [(k, freq_probs[k]) for k in range(1, N) if freq_probs[k] > 0.05]
    
    if len(peaks) >= 1:
        peak_k = min([p[0] for p in peaks])
        detected_period = N // np.gcd(peak_k, N)
    else:
        detected_period = 1
    
    return freq_probs, detected_period


# ============================================================
# CHAOS CONTROL ALGORITHM
# ============================================================

def chaos_controller(detected_period, current_gamma, 
                     gamma_min=0.5, gamma_max=0.78,
                     reduce_factor=0.85, increase_factor=1.05):
    """
    Adaptive learning rate controller based on detected period.
    
    Strategy:
    - Period >= 4: Approaching chaos, reduce γ
    - Period == 1 and γ low: Has headroom, can increase γ
    - Otherwise: Maintain current γ
    """
    if detected_period >= 4:
        new_gamma = max(gamma_min, current_gamma * reduce_factor)
    elif detected_period == 1 and current_gamma < gamma_max * 0.9:
        new_gamma = min(gamma_max, current_gamma * increase_factor)
    else:
        new_gamma = current_gamma
    
    return new_gamma


def run_controlled_optimization(n_steps=200, initial_gamma=0.75, 
                                 control_interval=16, n_qubits=4):
    """
    Run VQA optimization with chaos control.
    
    Returns: (trajectory, gamma_history, period_history)
    """
    gamma_history = [initial_gamma]
    period_history = []
    trajectory = [0.5]
    
    x = 0.5
    gamma = initial_gamma
    
    for i in range(n_steps):
        x = sin2_map(x, gamma)
        trajectory.append(x)
        
        # Periodic control update
        if (i + 1) % control_interval == 0 and i > control_interval:
            window = np.array(trajectory[-control_interval:])
            _, detected_period = detect_period_quantum(window, n_qubits)
            period_history.append(detected_period)
            
            gamma = chaos_controller(detected_period, gamma)
            gamma_history.append(gamma)
    
    return np.array(trajectory), gamma_history, period_history


# ============================================================
# HAMILTONIAN GENERATORS
# ============================================================

def get_hamiltonian_ops(n, topology="ordered"):
    """
    Generate Hamiltonian operators for different topologies.
    
    Returns list of (pauli_string, coefficient) tuples.
    
    Topologies:
    - "ordered": TFIM 1D chain (Polynomial DLA ~ O(N²))
    - "chaotic": SK Spin Glass (Exponential DLA ~ O(4^N))
    """
    ops = []
    
    # Transverse Field (common to both)
    for i in range(n):
        lbl = ["I"] * n
        lbl[i] = "X"
        ops.append(("".join(lbl), 0.5))
    
    if topology == "ordered":
        # 1D Chain coupling
        for i in range(n-1):
            lbl = ["I"] * n
            lbl[i] = "Z"
            lbl[i+1] = "Z"
            ops.append(("".join(lbl), 1.0))
            
    elif topology == "chaotic":
        # SK Spin Glass - all-to-all random frustrated couplings
        np.random.seed(42)  # Reproducibility
        for i in range(n):
            for j in range(i+1, n):
                lbl = ["I"] * n
                lbl[i] = "Z"
                lbl[j] = "Z"
                J_ij = np.random.uniform(-1.5, 1.5)
                ops.append(("".join(lbl), J_ij))
                
    return ops


if __name__ == "__main__":
    # Quick test
    print("=== Core Module Test ===\n")
    
    print("Lyapunov exponents at different r:")
    for r in [0.55, 0.68, 0.72, 0.78, 0.85]:
        lyap = compute_lyapunov(r)
        status = "stable" if lyap < 0 else "CHAOTIC"
        print(f"  r = {r:.2f}: λ = {lyap:+.3f} ({status})")
    
    print("\nQuantum Period Detection Test:")
    for r, expected in [(0.55, 1), (0.68, 2), (0.72, 4), (0.78, 8)]:
        traj = compute_trajectory(0.5, r, 64)
        _, detected = detect_period_quantum(traj[32:], n_qubits=4)
        match = "✓" if detected == expected else "≈"
        print(f"  r = {r:.2f}: Expected Period-{expected}, Detected: {detected} {match}")
    
    print(f"\nFigures will be saved to: {FIGURES_DIR}")
