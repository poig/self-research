"""Supplement Script v57_pulse_control.py: Model-Free Pulse Control via QLTO Primitive (Path B)

Evaluates Path B: Applying QLTO's hardware-native, model-free gradient estimation primitive
to pulse-level quantum control calibration (e.g. tuning drive amplitudes u_j for target gates).

Physical system: 2-qubit system with drift Hamiltonian H_0 = Z_0 Z_1 + Z_0 + Z_1
and M control drives H_j = X_0, Y_0, X_1, Y_1, etc.
Total Hamiltonian: H(u) = H_0 + sum_{j=1}^M u_j H_j

Objective: Fidelity to target gate U_target (e.g. CNOT or Bell state preparation).
Loss: F(u) = |<psi_target | U_pulse(u) | psi_init>|^2

Sensing:
  Encode pulse amplitudes u_j = u_j^0 - R + 2R x_j on a parameter register.
  Single circuit per epoch yields all M pulse gradient components via return-bit marginals.
  
Metrics:
1. Cosine similarity of QLTO pulse gradient vs exact finite-difference gradient.
2. Convergence to F > 0.999 in pulse calibration loop.
3. Total circuit savings vs finite-difference GRAPE (1 circuit/epoch vs 2M circuits/epoch).
"""
import sys, os, contextlib, io, time
import numpy as np
from scipy.linalg import expm

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

N_sys = 2
T_pulse = 1.5
M = 4

# Target problem: 2-qubit coupled system
# H_drift = Z0 Z1 + 0.5 Z0 + 0.5 Z1
# H_controls = [X0, Y0, X1, Y1]
TERMS = ["ZZ", "ZI", "IZ", "XI", "YI", "IX", "IY"]
C_DRIFT = np.array([1.0, 0.5, 0.5])
U_TARGET_AMPS = np.array([0.80, -0.60, 0.70, -0.50]) # True optimal control amplitudes

BACKEND = AerSimulator(method='statevector')


def build_system_hamiltonian(u_params):
    # u_params is length M (4)
    coeffs = np.concatenate([C_DRIFT, u_params])
    return SparsePauliOp.from_list(list(zip(TERMS, coeffs))).simplify()


def simulate_pulse_fidelity(u_params):
    """Calculates fidelity F(u) = |<psi_target | U_pulse(u) | psi_0>|^2."""
    # Target state: state under optimal amplitudes U_TARGET_AMPS starting from |00>
    qc_target = QuantumCircuit(N_sys)
    qc_target.append(PauliEvolutionGate(build_system_hamiltonian(U_TARGET_AMPS), time=T_pulse), range(N_sys))
    psi_target = Statevector(qc_target)
    
    # Pulse state under current amplitudes
    qc_pulse = QuantumCircuit(N_sys)
    qc_pulse.append(PauliEvolutionGate(build_system_hamiltonian(u_params), time=T_pulse), range(N_sys))
    psi_pulse = Statevector(qc_pulse)
    
    fidelity = abs(np.vdot(psi_target.data, psi_pulse.data)) ** 2
    return float(fidelity)


def build_pulse_sensing_circuit(u_center, R, active_indices):
    """
    Builds single QLTO sensing circuit over M pulse amplitudes simultaneously.
    Executes controlled pulse increments e^{-i u_j H_j T} controlled by param register.
    """
    n_act = len(active_indices)
    param = QuantumRegister(n_act, 'p')
    sysr = QuantumRegister(N_sys, 's')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(n_act, 'cp'), ClassicalRegister(N_sys, 'cs'))
    
    qc.h(param)
    # Target state backward evolution U_target^\dagger
    qc.append(PauliEvolutionGate(build_system_hamiltonian(U_TARGET_AMPS), time=-T_pulse), sysr)
    
    # Base pulse evolution under u_center - R
    u_base = u_center.copy()
    u_base[active_indices] -= R
    qc.append(PauliEvolutionGate(build_system_hamiltonian(u_base), time=T_pulse), sysr)
    
    # Controlled pulse increment +2R per active parameter
    ctrl_ops = [
        SparsePauliOp.from_list([("XI", 1.0)]),
        SparsePauliOp.from_list([("YI", 1.0)]),
        SparsePauliOp.from_list([("IX", 1.0)]),
        SparsePauliOp.from_list([("IY", 1.0)])
    ]
    
    for local_idx, global_idx in enumerate(active_indices):
        op = ctrl_ops[global_idx]
        qc.append(PauliEvolutionGate(op, time=2.0 * R * T_pulse).control(1),
                  [param[local_idx]] + list(sysr))
                  
    # Project system register to |00> (return probability check)
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])
    return qc


def sense_pulse_gradient(u_center, R, active_indices, shots=16384):
    """Extracts M-component pulse gradient from single circuit execution."""
    qc = build_pulse_sensing_circuit(u_center, R, active_indices)
    counts = BACKEND.run(transpile(qc, BACKEND, optimization_level=1), shots=shots).result().get_counts()
    
    n_act = len(active_indices)
    tot = 0
    s1 = np.zeros(n_act)
    
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        sys_bits, par_bits = parts[0], parts[1]
        ret = 1.0 if set(sys_bits) == {'0'} else 0.0
        xb = par_bits[::-1]
        sg = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0 for i in range(n_act)])
        s1 += ret * sg * cnt
        tot += cnt
        
    walsh = s1 / max(tot, 1)
    g = np.zeros(M)
    g[active_indices] = walsh / R
    return g


if __name__ == "__main__":
    print("=========================================================================")
    print("PATH B: MODEL-FREE PULSE CONTROL VIA QLTO PRIMITIVE")
    print("=========================================================================")
    print(f"  Target Amplitudes: {U_TARGET_AMPS}")
    print(f"  Pulse Duration T = {T_pulse}s | M = {M} control drives")
    print()

    # Initial detuned pulse amplitudes
    rng = np.random.RandomState(42)
    u_curr = U_TARGET_AMPS + rng.uniform(-0.5, 0.5, M)
    init_fid = simulate_pulse_fidelity(u_curr)
    print(f"  Initial u: {np.round(u_curr, 4)}")
    print(f"  Initial Fidelity: {init_fid:.5f}")
    print()

    # PART 1: Gradient Accuracy Check
    R0 = 0.3
    act_all = list(range(M))
    g_qlto = np.mean([sense_pulse_gradient(u_curr, R0, act_all, shots=16384) for _ in range(5)], axis=0)

    # Finite-difference gradient for ground truth
    g_fd = np.zeros(M)
    eps = 1e-4
    for j in range(M):
        u_plus = u_curr.copy(); u_plus[j] += eps
        u_minus = u_curr.copy(); u_minus[j] -= eps
        g_fd[j] = (simulate_pulse_fidelity(u_plus) - simulate_pulse_fidelity(u_minus)) / (2.0 * eps)

    cos_sim = float(np.dot(g_qlto, g_fd) / (np.linalg.norm(g_qlto) * np.linalg.norm(g_fd) + 1e-12))
    print("PART 1: Pulse Gradient Validation")
    print("-" * 50)
    print(f"  QLTO Gradient: {np.round(g_qlto, 4)}")
    print(f"  FD Gradient:   {np.round(g_fd, 4)}")
    print(f"  Cosine Similarity: {cos_sim:.5f}")
    print()

    # PART 2: Model-Free Pulse Calibration Loop
    print("PART 2: Model-Free Pulse Calibration Loop")
    print("-" * 65)
    print(f"  {'epoch':>6}{'R':>7}{'Fidelity':>12}{'||u - u_target||':>20}{'Circuits Used':>16}")
    print("-" * 65)

    total_circuits_qlto = 0
    total_circuits_fd = 0

    for ep in range(25):
        R_ep = max(0.3 * (0.90 ** ep), 0.02)
        g_sense = sense_pulse_gradient(u_curr, R_ep, act_all, shots=16384)
        total_circuits_qlto += 1
        total_circuits_fd += 2 * M

        # Ascend pulse fidelity
        step_sz = 0.8 * R_ep
        u_curr = u_curr + step_sz * g_sense / (np.linalg.norm(g_sense) + 1e-9)
        fid_curr = simulate_pulse_fidelity(u_curr)
        dist_target = np.linalg.norm(u_curr - U_TARGET_AMPS)

        if ep % 3 == 0 or ep == 24 or fid_curr >= 0.999:
            print(f"  {ep+1:>6}{R_ep:>7.3f}{fid_curr:>12.5f}{dist_target:>20.4f}{total_circuits_qlto:>16}")
        if fid_curr >= 0.999:
            print(f"\n  [Success] Target fidelity 0.999 reached at epoch {ep+1}!")
            break

    print()
    print("=" * 65)
    print("EFFICIENCY COMPARISON")
    print("=" * 65)
    print(f"  Final Recovered Amplitudes: {np.round(u_curr, 4)}")
    print(f"  True Optimal Amplitudes:    {np.round(U_TARGET_AMPS, 4)}")
    print(f"  Max Amplitude Error:        {np.max(np.abs(u_curr - U_TARGET_AMPS)):.4f}")
    print(f"  QLTO Quantum Circuits Used: {total_circuits_qlto}")
    print(f"  FD GRAPE Circuits Required: {total_circuits_fd}  ({total_circuits_fd / total_circuits_qlto:.1f}x reduction)")
