"""Supplement Script v57b_pulse_trotter.py: Trotterized Pulse Sensing for Non-Commuting Controls

Fixes v57's non-commutativity BCH error:
In v57, splitting e^{-i (H_0 + sum u_j H_j) T} into base evolution followed by separated controlled H_j gates
caused massive BCH non-commutativity errors because [H_0, H_j] != 0 and T=1.5 is large.

v57b uses Trotterized pulse slicing:
Time is sliced into K steps dt = T/K.
In each time slice dt, the evolution is:
  e^{-i H_0 dt} * prod_j [ e^{-i (u_j - R) H_j dt} * e^{-i 2 R x_j H_j dt} ]
where e^{-i 2 R x_j H_j dt} is controlled by param qubit j.

Tests gradient accuracy vs exact finite difference as Trotter steps K increase.
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


def build_trotter_pulse_sensing(u_center, R, active_indices, K_slices=8):
    """
    Builds Trotterized pulse sensing circuit with K_slices time steps.
    In each slice dt = T/K:
      1. e^{-i H_0 dt}
      2. For each j: e^{-i (u_j - R) H_j dt} followed by CR_Hj(2 R dt) controlled by param[j].
    """
    n_act = len(active_indices)
    param = QuantumRegister(n_act, 'p')
    sysr = QuantumRegister(N_sys, 's')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(n_act, 'cp'), ClassicalRegister(N_sys, 'cs'))
    
    dt = T_pulse / K_slices
    qc.h(param)
    
    # Target state backward evolution U_target^\dagger
    qc.append(PauliEvolutionGate(build_system_hamiltonian(U_TARGET_AMPS), time=-T_pulse), sysr)
    
    # Trotterized forward evolution
    for step in range(K_slices):
        qc.append(PauliEvolutionGate(DRIFT_OP, time=dt), sysr)
        for local_idx, global_idx in enumerate(active_indices):
            op = CTRL_OPS[global_idx]
            u_base_j = u_center[global_idx] - R
            # Base uncontrolled pulse slice
            qc.append(PauliEvolutionGate(op, time=u_base_j * dt), sysr)
            # Controlled increment slice
            qc.append(PauliEvolutionGate(op, time=2.0 * R * dt).control(1),
                      [param[local_idx]] + list(sysr))
                      
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])
    return qc


def sense_pulse_gradient_trotter(u_center, R, active_indices, K_slices=8, shots=16384):
    qc = build_trotter_pulse_sensing(u_center, R, active_indices, K_slices=K_slices)
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
    print("PATH B FIX: TROTTERIZED PULSE SENSING FOR NON-COMMUTING CONTROLS")
    print("=========================================================================")
    
    rng = np.random.RandomState(42)
    u_curr = U_TARGET_AMPS + rng.uniform(-0.4, 0.4, M)
    print(f"  Initial u: {np.round(u_curr, 4)}")
    print(f"  Initial Fidelity: {simulate_pulse_fidelity(u_curr):.5f}")
    print()
    
    # Ground truth finite difference gradient
    g_fd = np.zeros(M)
    eps = 1e-4
    for j in range(M):
        u_plus = u_curr.copy(); u_plus[j] += eps
        u_minus = u_curr.copy(); u_minus[j] -= eps
        g_fd[j] = (simulate_pulse_fidelity(u_plus) - simulate_pulse_fidelity(u_minus)) / (2.0 * eps)
        
    print(f"  FD Gradient: {np.round(g_fd, 4)}")
    print()
    print(f"  {'Trotter K':>10}{'QLTO Gradient':>35}{'cos(QLTO, FD)':>18}{'Norm Ratio':>14}")
    print("  " + "-" * 80)
    
    R0 = 0.2
    act_all = list(range(M))
    
    for K_slices in [1, 2, 4, 8, 16]:
        g_qlto = np.mean([sense_pulse_gradient_trotter(u_curr, R0, act_all, K_slices=K_slices, shots=32768)
                          for _ in range(3)], axis=0)
        cos_sim = float(np.dot(g_qlto, g_fd) / (np.linalg.norm(g_qlto) * np.linalg.norm(g_fd) + 1e-12))
        norm_rat = float(np.linalg.norm(g_qlto) / (np.linalg.norm(g_fd) + 1e-12))
        print(f"  {K_slices:>10}{str(np.round(g_qlto, 4)):>35}{cos_sim:>18.4f}{norm_rat:>14.4f}")
