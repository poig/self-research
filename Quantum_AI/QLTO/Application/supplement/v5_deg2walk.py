"""Degree-2 drift: does giving the walk the quadratic term help?

The walk's CRZ writes a phase LINEAR in the param bits, so it steers on a degree-1
model of the energy. v5_walsh.log showed that is leaving real structure on the
table: degree-2 carries 14-24% of the hypercube variance at R=0.6, degree-3+ is
numerically zero (so degree 1+2 is essentially the WHOLE landscape), and on two of
four blocks the degree-2 weight EXCEEDS degree-1 (blk2 0.41 vs 0.30, blk3 0.52 vs
0.21). Those are the small-gradient blocks that carried the worst scale bias.

The coefficients are free: every Walsh coefficient is an expectation over the same
shot record (T2), so degree-2 costs zero extra circuits on the sensing side and
measures at SNR 3-4 at 8192 shots.

Circuit cost is a controlled-RZZ per pair. It decomposes without a Toffoli:
controlled-(V W V^dag) = V * controlled-W * V^dag with V = CX, so

    CX(i,j) ; CRZ(theta; anc->j) ; CX(i,j)

with the CX gates UNCONTROLLED - 2 CX plus one CRZ per pair per step.

SCALING IS NOT DERIVED. The existing degree-1 angle is g_i*gamma*0.5pi*drift_gain
with g_i = Ehat_i/R and drift_gain = 1/sqrt(R), which is itself heuristic. Rather
than guess the matching degree-2 normalisation, sweep a gain and let gain=0
reproduce the current walk exactly as the control.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def sense_walsh12(q, c, R, act):
    """Degree-1 and degree-2 Walsh coefficients from ONE QPE sensing circuit."""
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(
            q.H_sense, time=(2 ** a) * q.tau0,
            synthesis=SuzukiTrotter(order=2, reps=max(1, (2 ** a) // 2))
        ).control(1), [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    counts = q._run(qc)

    tot = 0; s1 = np.zeros(n); s2 = np.zeros((n, n))
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        m = int(parts[0], 2); phi = m / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        e = -2.0 * np.pi * phi / (q.tau0 + 1e-12)
        xb = parts[1][::-1]
        sg = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0
                       for i in range(n)])
        s1 += e * sg * cnt
        s2 += e * np.outer(sg, sg) * cnt
        tot += cnt
    return s1 / max(tot, 1), s2 / max(tot, 1)


def walk_deg2(q, center, k_steps, dt, R, act, w1, w2, gain2):
    """The V3 walk, plus optional controlled-ZZ drift from degree-2 coefficients.

    gain2 = 0 reproduces _execute_walk exactly.
    """
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
    dg = 1.0 / np.sqrt(max(R, 1e-9))
    g1 = w1 / max(R, 1e-9)                 # degree-1 -> gradient, as V3 does
    g2 = w2 / max(R, 1e-9)                 # same normalisation, gain-swept
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        for i in range(n):
            qc.crz(g1[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        if gain2 != 0.0:
            for i in range(n):
                for j in range(i + 1, n):
                    th = gain2 * g2[i, j] * gamma * 0.5 * np.pi * dg
                    if abs(th) < 1e-9:
                        continue
                    # controlled-RZZ = CX ; CRZ ; CX   (CX uncontrolled)
                    qc.cx(param[i], param[j])
                    qc.crz(th, anc[0], param[j])
                    qc.cx(param[i], param[j])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc, q._decode_walk(q._run(qc), center, act, R)


def run(prob, gain2, seed, epochs=20, k=15, shots=8192):
    ansatz, H, _ = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    best, depth = float('inf'), 0
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            w1, w2 = sense_walsh12(q, p, R, act)
            qc, blk = walk_deg2(q, p, k, dt, R, act, w1, w2, gain2)
            p[act] = blk
        E = float(np.real(Statevector(ansatz.assign_parameters(p))
                          .expectation_value(H)))
        best = min(best, E)
    return best, E, q.max_circuit_depth


SEEDS = (42, 43, 44, 45)
GAINS = (0.0, 0.25, 0.5, 1.0, 2.0)
for pname, prob in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))):
    a, h, _ = prob()
    exact = float(np.min(np.linalg.eigvalsh(h.to_matrix())))
    print(f"\n===== {pname} | exact {exact:.4f} | {len(SEEDS)} seeds, "
          f"20 epochs, k=15 =====", flush=True)
    print(f"  {'deg2 gain':<12}{'E_best':>10}{'E_final':>10}{'std':>8}"
          f"{'SEM':>8}{'depth':>8}{'time':>7}")
    print("  " + "-" * 63)
    store = {}
    for g in GAINS:
        t0 = time.time(); bs, fs, dep = [], [], 0
        for s in SEEDS:
            b, f, d = run(prob, g, s)
            bs.append(b); fs.append(f); dep = max(dep, d)
        sd = float(np.std(fs)); sem = sd / np.sqrt(len(SEEDS))
        store[g] = (float(np.mean(fs)), sem)
        tag = f"{g}" + ("  (V3)" if g == 0.0 else "")
        print(f"  {tag:<12}{np.mean(bs):>10.4f}{np.mean(fs):>10.4f}"
              f"{sd:>8.4f}{sem:>8.4f}{dep:>8}{time.time()-t0:>6.0f}s", flush=True)
    b0, s0 = store[0.0]
    print(f"\n  vs gain=0 (E_final, negative = degree-2 helps):")
    for g in GAINS[1:]:
        m, sm = store[g]
        print(f"    gain={g:<5}{m - b0:+.4f}  "
              f"({abs(m - b0) / max(np.hypot(sm, s0), 1e-9):.1f} sigma)")
