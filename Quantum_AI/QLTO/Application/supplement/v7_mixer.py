"""Does NON-UNIFORM mixing help? The cheap probe before building energy-conditioning.

T7 left the mixer as the interesting untested half: CRX is a product of
independent single-qubit rotations at a FIXED beta, identical for every
coordinate and every vertex, and nothing in the walk adapts it. Every ablation so
far varied the DRIFT.

The ambitious version conditions beta on the QPE energy register - "freeze the
good vertices, shake the bad ones" - but that needs an energy register inside the
walk circuit, which is a real rebuild. So probe the cheap question first, exactly
as the degree-2 Walsh weight was checked before building any CRZZ: does making
beta NON-UNIFORM help at all?

Per-coordinate shaping using the gradient already measured, costing nothing:

    beta_i = beta * (1 + lambda * (1 - ghat_i)),   ghat_i = |g_i| / max|g|

  lambda = 0   uniform, i.e. exactly the current walk (control)
  lambda > 0   MORE mixing where the gradient is small - explore the flat
               directions, let the driven ones be driven
  lambda < 0   the opposite: more mixing where the gradient is large

If uniform beta is already optimal, no energy-conditioned scheme will help either
and the mixer branch closes. If shaping helps, the sign of the winning lambda says
which way the fancier version should condition.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def walk_shaped(q, center, k_steps, dt, R, act, g, lam):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=dt * np.pi,
                                 synthesis=LieTrotter(reps=1)).control(1),
              [anc[0]] + list(sysr))
    gl = g[act]
    dg = 1.0 / np.sqrt(max(R, 1e-9))
    mx = float(np.max(np.abs(gl))) + 1e-12
    shape = 1.0 + lam * (1.0 - np.abs(gl) / mx)      # lam=0 -> all ones
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        for i in range(n):
            qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        for i in range(n):
            qc.crx(beta * shape[i], anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_walk(q._run(qc), center, act, R)


def run(prob, lam, seed, epochs=20, k=15, shots=8192):
    ansatz, H, _ = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    best = float('inf')
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            p[act] = walk_shaped(q, p, k, dt, R, act,
                                 q.sense_gradient(p, R, act), lam)
        best = min(best, float(np.real(Statevector(ansatz.assign_parameters(p))
                                       .expectation_value(H))))
    return best, float(np.real(Statevector(ansatz.assign_parameters(p))
                               .expectation_value(H)))


SEEDS = (42, 43, 44, 45, 46)
LAMS = (-0.5, 0.0, 0.5, 1.0, 2.0)
for pname, prob in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))):
    a, h, _ = prob()
    exact = float(np.min(np.linalg.eigvalsh(h.to_matrix())))
    print(f"\n===== {pname} | exact {exact:.4f} | {len(SEEDS)} seeds =====",
          flush=True)
    print(f"  {'lambda':<14}{'E_best':>10}{'E_final':>10}{'std':>8}{'SEM':>8}"
          f"{'time':>7}")
    print("  " + "-" * 57)
    store = {}
    for lam in LAMS:
        t0 = time.time(); bs, fs = [], []
        for s in SEEDS:
            b_, f_ = run(prob, lam, s)
            bs.append(b_); fs.append(f_)
        sd = float(np.std(fs)); sem = sd / np.sqrt(len(SEEDS))
        store[lam] = (float(np.mean(fs)), sem)
        tag = f"{lam}" + ("  (current)" if lam == 0.0 else "")
        print(f"  {tag:<14}{np.mean(bs):>10.4f}{np.mean(fs):>10.4f}{sd:>8.4f}"
              f"{sem:>8.4f}{time.time()-t0:>6.0f}s", flush=True)
    b0, s0 = store[0.0]
    print(f"\n  vs uniform (E_final, negative = shaping helps):")
    for lam in LAMS:
        if lam == 0.0:
            continue
        m, sm = store[lam]
        print(f"    lambda={lam:<5}{m - b0:+.4f}  "
              f"({abs(m - b0) / max(np.hypot(sm, s0), 1e-9):.1f} sigma)")
