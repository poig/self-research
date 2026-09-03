"""Supplement Script v57c_pulse_correct_sign.py: Model-Free Pulse Control with Correct Sign Derivative

Identifies and fixes the sign flip:
Physical forward pulse evolution is e^{-i H(u) T} (minus sign in exponent), whereas v6_hamlearn used e^{+i theta P T}.
Derivative d(e^{-i u H T})/du = -i H T e^{-i u H T} introduces an explicit SIGN FLIP in the gradient direction.

With the sign correctly accounted for (gradient descent vs gradient ascent on return bit):
1. Measures cos(QLTO, FD) gradient alignment.
2. Runs the model-free pulse calibration loop to verify convergence to target pulse amplitudes.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

N_sys = 2
T_pulse = 1.0
M = 4

TERMS = ["ZZ", "ZI", "IZ", "XI", "YI", "IX", "IY"]
C_DRIFT = np.array([1.0, 0.5, 0.5])
U_TARGET_AMPS = np.array([0.80, -0.60, 0.70, -0.50])

BACKEND = AerSimulator(method='statevector')

CTRL_OPS = [
    SparsePauliOp.from_list([("XI", 1.0)]),
    SparsePauliOp.from_list([("YI", 1.0)]),
    SparsePauliOp.from_list([("IX", 1.0)]),
    SparsePauliOp.from_list([("IY", 1.0)])
]
DRIFT_OP = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.5), ("IZ", 0.5)])


def build_system_hamiltonian(u_params):
    coeffs = np.concatenate([C_DRIFT, u_params])
    return SparsePauliOp.from_list(list(zip(TERMS, coeffs))).simplify()


def simulate_pulse_fidelity(u_params):
    qc_target = QuantumCircuit(N_sys)
    qc_target.append(PauliEvolutionGate(build_system_hamiltonian(U_TARGET_AMPS), time=T_pulse), range(N_sys))
    psi_target = Statevector(qc_target)
    
    qc_pulse = QuantumCircuit(N_sys)
    qc_pulse.append(PauliEvolutionGate(build_system_hamiltonian(u_params), time=T_pulse), range(N_sys))
    psi_pulse = Statevector(qc_pulse)
    
    return float(abs(np.vdot(psi_target.data, psi_pulse.data)) ** 2)


def build_pulse_sensing_circuit(u_center, R, active_indices, K_slices=8):
    """
    Pulse sensing circuit.
    Physical evolution e^{-i H(u) t} uses -i sign.
    Forward pulse evolution e^{-i H(u) t} followed by U_target^\dagger.
    """
    n_act = len(active_indices)
    param = QuantumRegister(n_act, 'p')
    sysr = QuantumRegister(N_sys, 's')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(n_act, 'cp'), ClassicalRegister(N_sys, 'cs'))
    
    dt = T_pulse / K_slices
    qc.h(param)
    
    # 1. Forward pulse evolution under u_center - R with controlled increment +2R
    for step in range(K_slices):
        qc.append(PauliEvolutionGate(DRIFT_OP, time=dt), sysr)
        for local_idx, global_idx in enumerate(active_indices):
            op = CTRL_OPS[global_idx]
            u_base_j = u_center[global_idx] - R
            qc.append(PauliEvolutionGate(op, time=u_base_j * dt), sysr)
            qc.append(PauliEvolutionGate(op, time=2.0 * R * dt).control(1),
                      [param[local_idx]] + list(sysr))
                      
    # 2. Target state backward evolution U_target^\dagger
    qc.append(PauliEvolutionGate(build_system_hamiltonian(U_TARGET_AMPS), time=-T_pulse), sysr)
                      
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])
    return qc


def sense_pulse_gradient(u_center, R, active_indices, K_slices=8, shots=32768):
    qc = build_pulse_sensing_circuit(u_center, R, active_indices, K_slices=K_slices)
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
    # Physical derivative sign correction: g_physical = - walsh / R
    g[active_indices] = - walsh / R
    return g


if __name__ == "__main__":
    print("=========================================================================")
    print("PATH B: MODEL-FREE PULSE CONTROL (WITH PHYSICAL DERIVATIVE SIGN FIX)")
    print("=========================================================================")
    
    rng = np.random.RandomState(42)
    u_curr = U_TARGET_AMPS + rng.uniform(-0.4, 0.4, M)
    init_fid = simulate_pulse_fidelity(u_curr)
    print(f"  Target Amplitudes: {U_TARGET_AMPS}")
    print(f"  Initial u: {np.round(u_curr, 4)}")
    print(f"  Initial Fidelity:  {init_fid:.5f}")
    print()
    
    # Ground truth finite difference gradient
    g_fd = np.zeros(M)
    eps = 1e-4
    for j in range(M):
        u_plus = u_curr.copy(); u_plus[j] += eps
        u_minus = u_curr.copy(); u_minus[j] -= eps
        g_fd[j] = (simulate_pulse_fidelity(u_plus) - simulate_pulse_fidelity(u_minus)) / (2.0 * eps)
        
    act_all = list(range(M))
    R0 = 0.2
    g_qlto = np.mean([sense_pulse_gradient(u_curr, R0, act_all, K_slices=8, shots=32768)
                      for _ in range(5)], axis=0)
    cos_sim = float(np.dot(g_qlto, g_fd) / (np.linalg.norm(g_qlto) * np.linalg.norm(g_fd) + 1e-12))
    
    print("PART 1: Gradient Validation")
    print("-" * 50)
    print(f"  FD Gradient:   {np.round(g_fd, 4)}")
    print(f"  QLTO Gradient: {np.round(g_qlto, 4)}")
    print(f"  Cosine Similarity: {cos_sim:.5f}")
    print()
    
    # PART 2: Calibration Loop
    print("PART 2: Model-Free Pulse Calibration Loop")
    print("-" * 65)
    print(f"  {'epoch':>6}{'R':>7}{'Fidelity':>12}{'||u - u_target||':>20}{'Circuits Used':>16}")
    print("-" * 65)
    
    total_circuits_qlto = 0
    total_circuits_fd = 0
    
    for ep in range(25):
        R_ep = max(0.25 * (0.90 ** ep), 0.02)
        g_sense = sense_pulse_gradient(u_curr, R_ep, act_all, K_slices=8, shots=32768)
        total_circuits_qlto += 1
        total_circuits_fd += 2 * M
        
        # Ascend pulse fidelity along gradient direction
        step_sz = 0.6 * R_ep
        u_curr = u_curr + step_sz * g_sense / (np.linalg.norm(g_sense) + 1e-9)
        fid_curr = simulate_pulse_fidelity(u_curr)
        dist_target = np.linalg.norm(u_curr - U_TARGET_AMPS)
        
        if ep % 3 == 0 or ep == 24 or fid_curr >= 0.995:
            print(f"  {ep+1:>6}{R_ep:>7.3f}{fid_curr:>12.5f}{dist_target:>20.4f}{total_circuits_qlto:>16}")
        if fid_curr >= 0.995:
            print(f"\n  [Success] Target fidelity 0.995 reached at epoch {ep+1}!")
            break
            
    print()
    print("=" * 65)
    print("EFFICIENCY COMPARISON")
    print("=" * 65)
    print(f"  Final Recovered Amplitudes: {np.round(u_curr, 4)}")
    print(f"  True Optimal Amplitudes:    {np.round(U_TARGET_AMPS, 4)}")
    print(f"  Max Amplitude Error:        {np.max(np.abs(u_curr - U_TARGET_AMPS)):.4f}")
    print(f"  QLTO Quantum Circuits Used: {total_circuits_qlto}")
    print(f"  FD GRAPE Circuits Required: {total_circuits_fd}  ({total_circuits_fd / total_circuits_qlto:.1f}x circuit reduction)")
