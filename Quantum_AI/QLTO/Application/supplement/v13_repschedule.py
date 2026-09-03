"""Was the Suzuki-2 change over-generalised? Gradient bias per problem.

v4_frontier validated suz2 at reps=2^a/2 on HEISENBERG N=4 only, measuring
gradient bias, and it was then shipped for every problem. Direct operator-norm
measurement now says that change makes the Trotter error WORSE - 5x worse on H2 -
because the reps schedule sets the Trotter STEP, not the accuracy:

    old  reps=2^a      -> step = 2^a tau0 / 2^a     = tau0
    new  reps=2^a/2    -> step = 2^a tau0 / (2^a/2) = 2 tau0

and tau0 = pi/(margin ||H0||) is LARGE exactly when ||H0|| is small, which is H2.
There the top ancilla evolves to t ~ 15.7 with a step of ~3.9 - far outside any
product formula's asymptotic regime, where doubling the step is simply worse.

Operator error and gradient bias can disagree, because the gradient uses
DIFFERENCES of energies across vertices and a Trotter error that is uniform over
the hypercube partly cancels. Gradient bias is the metric the algorithm actually
consumes, so measure that, per problem.

Also tested: a STEP-BOUNDED schedule, reps = ceil(t / step_max), which fixes the
Trotter step to an absolute constant instead of tying it to tau0. That is the
principled version - it adapts reps to ||H0|| automatically.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import PauliEvolutionGate, QFT
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def exact_smeared(ansatz, H, c, R, act):
    n = len(act)
    sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(n)]
                    for v in range(2 ** n)])
    E = np.empty(len(sig))
    for v, s in enumerate(sig):
        p = c.copy(); p[act] = c[act] + R * s
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
    g = np.zeros(len(c))
    for i in range(n):
        hi = sig[:, i] > 0
        g[act[i]] = (E[hi].mean() - E[~hi].mean()) / (2.0 * R)
    return g


def sense(q, c, R, act, mk):
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for A in range(k):
        t = (2 ** A) * q.tau0
        qc.append(PauliEvolutionGate(q.H_sense, time=t,
                                     synthesis=mk(A, t)).control(1),
                  [anc[A]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_gradient_qpe(q._run(qc), c, act, R)


SCHEDULES = [
    ("lie  reps=2^a      (pre-fix)", lambda A, t: LieTrotter(reps=max(1, 2 ** A))),
    ("suz2 reps=2^a/2    (SHIPPED)", lambda A, t: SuzukiTrotter(order=2, reps=max(1, (2 ** A) // 2))),
    ("suz2 reps=2^a               ", lambda A, t: SuzukiTrotter(order=2, reps=max(1, 2 ** A))),
    ("suz2 step<=0.5     (bounded)", lambda A, t: SuzukiTrotter(order=2, reps=int(max(1, np.ceil(t / 0.5))))),
    ("suz2 step<=0.25    (bounded)", lambda A, t: SuzukiTrotter(order=2, reps=int(max(1, np.ceil(t / 0.25))))),
]

R, REP = 0.6, 4
print("=" * 92)
print("Gradient bias by reps schedule, PER PROBLEM (the metric the algorithm uses)")
print("=" * 92)
for pname, fn in (("H2", B.get_h2_problem),
                  ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                  ("MaxCut N=4", lambda: B.get_maxcut_problem(4))):
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=65536, num_ancillas=4)
    act = q.layers[0]['params']
    c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
    gsm = exact_smeared(ansatz, q.H_sense, c, R, act)
    nsm = np.linalg.norm(gsm[act])
    print(f"\n  --- {pname} | ||H0||={q.H0_norm:.3f}  tau0={q.tau0:.3f}  "
          f"top t={8*q.tau0:.2f} ---")
    print(f"  {'schedule':<32}{'bias':>9}{'cos':>9}{'depth':>8}")
    print("  " + "-" * 58)
    for label, mk in SCHEDULES:
        runs = np.array([sense(q, c, R, act, mk)[act] for _ in range(REP)])
        m = runs.mean(axis=0)
        bias = np.linalg.norm(m - gsm[act]) / nsm
        cs = float(m @ gsm[act] / (np.linalg.norm(m) * nsm + 1e-18))
        print(f"  {label:<32}{bias:>9.4f}{cs:>9.4f}{q.last_circuit_depth:>8}",
              flush=True)
print()
print("  A step-bounded schedule adapts reps to ||H0|| automatically, which is")
print("  what the shipped 2^a/2 rule fails to do: it fixes the REP COUNT and lets")
print("  the STEP float with tau0 = pi/(margin ||H0||).")
