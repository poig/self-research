"""Hamiltonian learning with the QLTO primitive - does the pivot actually work?

DESIGN (from the APPLICATION PIVOTS entry). Unknown device H_true = sum_k c_k P_k.
Model H(theta) = sum_k theta_k P_k. Circuit:

    |+..+>  ->  U_true = e^{-i H_true T}  ->  U_model(theta)^dag  ->  back to |+..+>?

At theta = c_true the two evolutions cancel and the probe returns exactly. So the
loss is the RETURN PROBABILITY, and three properties follow:

  * NO SENSING MACHINERY. No ancilla, no QPE, no Trotterised sensing evolution -
    just measure the system register and test for all-zeros (after undoing the
    probe basis). The per-shot readout is a BIT.
  * BOUNDED READOUT. A bit has variance <= 1/4 regardless of anything, so T4's
    Bernoulli argument applies and the cross-coordinate variance term b = 0
    structurally - the same property measured for the Hadamard path.
  * LINEARITY PRESERVED. The loss IS a single expectation value, not a function of
    several, so T2 holds and the degree-1 Walsh marginal is an unbiased gradient
    at any shots-per-vertex.

W-GATE FOR THIS. theta_k = c_k - R + 2R*x_k with x_k a param qubit, and
U_model^dag needs e^{+i theta_k P_k T}, i.e. PauliEvolutionGate(P_k, time=-theta_k*T).
Split into an uncontrolled base at -(c_k-R)*T and a controlled increment at
-2R*T. This is the multi-qubit generalisation of build_w_gate that the pivot
entry says is needed.

TEST CHOICES, and why. All P_k are Z-type so they COMMUTE - Trotter is then exact
and cannot confound the result. The probe is |+..+> rather than |0..0> because a
diagonal H would leave |0..0> an eigenstate and the return probability would be 1
for every theta. This makes the first test classically easy on purpose: the point
is to validate the estimator and the recovery loop, not to claim advantage.

PART 1 checks the measured gradient against the exact one.
PART 2 runs recovery from a wrong starting point.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

N = 4
T = 1.0
TERMS = ["ZZII", "IZZI", "IIZZ", "ZIII", "IIZI"]
C_TRUE = np.array([0.80, -0.55, 0.35, 0.60, -0.40])
M = len(TERMS)
BACKEND = AerSimulator(method='statevector')


def paulis():
    return [SparsePauliOp.from_list([(t, 1.0)]) for t in TERMS]


def h_of(theta):
    return SparsePauliOp.from_list(list(zip(TERMS, theta))).simplify()


def return_prob_exact(theta):
    """|<+..+| U_model(theta)^dag U_true |+..+>|^2, computed exactly."""
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.append(PauliEvolutionGate(h_of(C_TRUE), time=T), range(N))
    qc.append(PauliEvolutionGate(h_of(theta), time=-T), range(N))
    qc.h(range(N))
    sv = Statevector(qc)
    return float(abs(sv.data[0]) ** 2)


def build_sensing(center, R, act):
    """One circuit: superposition over the +-R hypercube of the ACTIVE coefficients.

    Measures the param register and the system register together, so each shot
    yields (which vertex, did the probe return).
    """
    n = len(act)
    param = QuantumRegister(n, 'p')
    sysr = QuantumRegister(N, 's')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(n, 'cp'),
                        ClassicalRegister(N, 'cs'))
    qc.h(param)
    qc.h(sysr)
    qc.append(PauliEvolutionGate(h_of(C_TRUE), time=T), sysr)   # the "device"
    P = paulis()
    for j, k in enumerate(act):
        # U_model^dag term: e^{+i theta_k P_k T},  theta_k = center_k - R + 2R x_k
        qc.append(PauliEvolutionGate(P[k], time=-(center[k] - R) * T), sysr)
        qc.append(PauliEvolutionGate(P[k], time=-2.0 * R * T).control(1),
                  [param[j]] + list(sysr))
    for k in range(M):                                          # frozen terms
        if k not in act:
            qc.append(PauliEvolutionGate(P[k], time=-center[k] * T), sysr)
    qc.h(sysr)
    qc.measure(param, qc.cregs[0]); qc.measure(sysr, qc.cregs[1])
    return qc


def sense(center, R, act, shots=8192):
    """Degree-1 Walsh marginal of the RETURN BIT -> d(return prob)/d theta_k."""
    qc = build_sensing(center, R, act)
    counts = BACKEND.run(transpile(qc, BACKEND, optimization_level=1),
                         shots=shots).result().get_counts()
    n = len(act)
    tot = 0; s1 = np.zeros(n)
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        sys_bits, par_bits = parts[0], parts[1]      # creg order: cs printed first
        ret = 1.0 if set(sys_bits) == {'0'} else 0.0
        xb = par_bits[::-1]
        sg = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0
                       for i in range(n)])
        s1 += ret * sg * cnt
        tot += cnt
    walsh = s1 / max(tot, 1)
    g = np.zeros(M)
    g[act] = walsh / R          # degree-1 coeff / R  ->  gradient estimate
    return g


print("=" * 84)
print("PART 1. Does the return-bit marginal give the gradient?")
print("=" * 84)
print(f"  N={N} qubits, {M} terms {TERMS}, T={T}")
print(f"  c_true = {np.array2string(C_TRUE, precision=3)}")
theta0 = C_TRUE + np.array([0.30, -0.25, 0.20, -0.30, 0.25])
print(f"  probe theta = {np.array2string(theta0, precision=3)}"
      f"   return prob = {return_prob_exact(theta0):.4f}")
print()
R = 0.4
act = list(range(M))
# exact smeared gradient by enumerating the hypercube
sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                for v in range(2 ** M)])
Ev = np.array([return_prob_exact(theta0 + R * s) for s in sig])
ex_walsh = np.array([np.mean(Ev * sig[:, i]) for i in range(M)])
ex_smear = ex_walsh / R
# exact analytic-ish gradient by central difference
ex_grad = np.array([(return_prob_exact(theta0 + 1e-4 * np.eye(M)[i])
                     - return_prob_exact(theta0 - 1e-4 * np.eye(M)[i]))
                    / 2e-4 for i in range(M)])
meas = np.mean([sense(theta0, R, act) for _ in range(6)], axis=0)
print(f"  {'k':<4}{'term':<7}{'d(ret)/dtheta':>15}{'smeared exact':>15}"
      f"{'measured':>12}")
print("  " + "-" * 53)
for k in range(M):
    print(f"  {k:<4}{TERMS[k]:<7}{ex_grad[k]:>15.4f}{ex_smear[k]:>15.4f}"
          f"{meas[k]:>12.4f}")
cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
print(f"\n  cos(measured, smeared) = {cs(meas, ex_smear):.5f}"
      f"   norm ratio = {np.linalg.norm(meas)/np.linalg.norm(ex_smear):.4f}")
print(f"  cos(measured, exact)   = {cs(meas, ex_grad):.5f}")
print("  cos ~ 1 vs SMEARED is the claim; vs exact it degrades by the usual R^2.")

print()
print("=" * 84)
print("PART 2. Recovery: can it find c_true from a wrong start?")
print("=" * 84)
print(f"  {'epoch':>6}{'R':>7}{'||theta-c||':>13}{'return prob':>13}")
print("  " + "-" * 39)
rng = np.random.RandomState(0)
theta = C_TRUE + rng.uniform(-0.6, 0.6, M)
print(f"  {'start':>6}{'-':>7}{np.linalg.norm(theta-C_TRUE):>13.4f}"
      f"{return_prob_exact(theta):>13.4f}")
for ep in range(30):
    R = max(0.4 * (0.92 ** ep), 0.02)
    g = sense(theta, R, act, shots=8192)
    # ASCEND the return probability
    theta = theta + 0.9 * R * g / (np.linalg.norm(g) + 1e-9)
    if ep % 5 == 4 or ep == 29:
        print(f"  {ep+1:>6}{R:>7.3f}{np.linalg.norm(theta-C_TRUE):>13.4f}"
              f"{return_prob_exact(theta):>13.4f}", flush=True)
print(f"\n  recovered = {np.array2string(theta, precision=3)}")
print(f"  c_true    = {np.array2string(C_TRUE, precision=3)}")
print(f"  max |error| per coefficient = {np.max(np.abs(theta-C_TRUE)):.4f}")
print(f"  circuits used = {30 * 1} (one per epoch, all {M} coefficients each)")
print(f"  parameter-shift equivalent = {30 * 2 * M} circuits")
