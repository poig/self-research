"""Merged tilted-axis rotation: does -37% depth cost anything? Paired, 12 seeds.

v5_merge left this unresolved: N=4 said the merged walk was 0.0407 BETTER at
2.5 sigma, N=6 said 0.176 WORSE at 0.7 sigma. Signs disagreed, and the effect was
comparable to the gap between two independent measurements of the same control.

The design flaw was comparing ARM MEANS across separate runs. These trajectories
are stochastic beyond the seed - the seed fixes only the initial parameters - so
between-arm variance carries all the run-to-run drift. Fix: run BOTH arms from the
SAME initial parameters inside one loop and take the PER-SEED DIFFERENCE. The
paired statistic cancels the shared trajectory variance, which is exactly the
noise that swamped the first attempt.

The merge itself is exact: RY(-phi); CRZ(theta); RY(phi) with
theta=sqrt(a^2+b^2), phi=atan2(b,a) equals exp(-i(aZ+bX)/2) to 4e-16, and the RY
conjugation is uncontrolled so it is ONE controlled gate instead of two. It is not
a small-angle approximation of the current walk - at the angles actually used
(a=3.8, b=0.94) the two operators differ by 0.813 in operator norm. So this is
different dynamics at lower depth, and the question is only whether the difference
costs accuracy.

Depth is already measured at -37% (162->102 at N=4, 246->156 at N=6). If the
paired difference is within noise, that depth is free and the merge should ship.
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


def walk(q, center, k_steps, dt, R, act, g, merged):
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
    gl = g[act]; dg = 1.0 / np.sqrt(max(R, 1e-9))
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        if merged:
            for i in range(n):
                al = gl[i] * gamma * 0.5 * np.pi * dg
                th = float(np.hypot(al, beta)); ph = float(np.arctan2(beta, al))
                qc.ry(-ph, param[i]); qc.crz(th, anc[0], param[i]); qc.ry(ph, param[i])
        else:
            for i in range(n):
                qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
            for i in range(n):
                qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_walk(q._run(qc), center, act, R)


def one_run(prob, merged, p0, epochs=20, k=15, shots=8192):
    ansatz, H, _ = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = p0.copy()
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            p[act] = walk(q, p, k, dt, R, act,
                          q.sense_gradient(p, R, act), merged)
    return float(np.real(Statevector(ansatz.assign_parameters(p))
                         .expectation_value(H)))


SEEDS = tuple(range(40, 52))          # 12 seeds
for pname, prob in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))):
    a, h, _ = prob()
    exact = float(np.min(np.linalg.eigvalsh(h.to_matrix())))
    print(f"\n===== {pname} | exact {exact:.4f} | {len(SEEDS)} seeds, PAIRED =====",
          flush=True)
    t0 = time.time(); base, merg = [], []
    for s in SEEDS:
        p0 = np.random.RandomState(s).uniform(-np.pi, np.pi, a.num_parameters)
        base.append(one_run(prob, False, p0))      # same p0 for both arms
        merg.append(one_run(prob, True, p0))
    base = np.array(base); merg = np.array(merg)
    d = merg - base                                # per-seed difference
    sem_d = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  CRZ+CRX  mean {base.mean():.4f}  std {base.std(ddof=1):.4f}")
    print(f"  merged   mean {merg.mean():.4f}  std {merg.std(ddof=1):.4f}")
    print(f"  paired difference (merged - base): {d.mean():+.4f} "
          f"+- {sem_d:.4f}  ({abs(d.mean())/max(sem_d,1e-9):.1f} sigma)")
    print(f"  merged better on {(d < 0).sum()}/{len(d)} seeds"
          f"   [{time.time()-t0:.0f}s]", flush=True)
print()
print("  Within noise => the measured -37% depth is free and the merge ships.")
print("  Reliably worse => the CRZ-then-CRX ordering carries real dynamics.")
