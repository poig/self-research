"""How many MOVES does the walk need, against uniform, Grover, and the hypercube?

TIER A - Qiskit circuits on AerSimulator with shots for every quantum arm.

THE METRIC. A walk does not visit vertices one at a time; it holds the whole box
in superposition and interferes. So the honest cost is not "vertices visited" but

    moves  =  1 / P(sample a vertex in the top q% of the box)

For a uniform sampler that is 100/q. Grover on N items reaches a marked set of
size N q/100 in ~sqrt(100/q) moves - the quadratic bound, and the thing any
structured search must beat to be worth its depth.

WHAT IS BEING COMPARED, all on the SAME potential and the SAME 4096 vertices:

    cycle      d cycles of 2^kappa sites - a TORUS. v136 PART 5 measured this
               register under DeltaE = e^{-S0/h}, slope -0.998: a particle.
    hypercube  the same 4096 vertices as a 12-cube. v136 PART 7 measured THIS
               one under DeltaE = e^{-n S~}, r = -0.99994: a spin, degrading
               exponentially in the parameter count.
    uniform    no evolution at all - the control that says whether any of the
               depth bought anything.

The potential is the measured local quadratic model from qlto_walk's 3-level
design, written as RZ + RZZ (verified against the direct form at 2.6e-16).

WHY THIS IS THE RIGHT QUESTION. v139 measured the walk capturing 90.2% of the
brute-force descent but LOSING to brute force 5 times in 6, at P(best) = 0.009
against a uniform 0.00024 - 37x concentration. Grover on 4096 would give 64x.
So at that operating point the walk was below the quadratic bound, and the
question is whether that is the schedule or the structure.
"""
import sys
import itertools
import numpy as np

sys.path.insert(0, '/home/poig/project/self-research/Quantum_AI/QLTO/Application')
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from qlto_walk import QLTOWalk, _cycle_mixer_qc, _quadratic_potential_qc

BASIS = ['rz', 'sx', 'x', 'cx']


def heis(n):
    t = []
    for i in range(n - 1):
        for p in ('XX', 'YY', 'ZZ'):
            lab = ['I'] * n
            lab[i], lab[i + 1] = p[0], p[1]
            t.append((''.join(reversed(lab)), 1.0))
    return SparsePauliOp.from_list(t)


def anz(n, reps=1):
    p = ParameterVector('t', 2 * n * (reps + 1))
    qc = QuantumCircuit(n)
    i = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(p[i], q); i += 1
            qc.rz(p[i], q); i += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(p[i], q); i += 1
        qc.rz(p[i], q); i += 1
    return qc


def hypercube_mixer_qc(nq, h, dt):
    """e^{-i h^2 (D - A) dt} with A = sum X_i: D = nq I, so up to a global phase
    this is a product of RX(-2 h^2 dt). One gate per qubit, no entangler - which
    is Part VI's separability theorem in circuit form."""
    qc = QuantumCircuit(nq)
    for q in range(nq):
        qc.rx(-2.0 * h * h * dt, q)
    return qc


def run(kind, d, kappa, gsub, Hsub, R, h, t_total, steps, shots, seed):
    nq = d * kappa
    qc = QuantumCircuit(nq, nq)
    qc.h(range(nq))
    dt = t_total / steps if steps else 0.0
    pot = _quadratic_potential_qc(d, kappa, gsub, Hsub, R, dt)
    if kind == 'cycle':
        mix = _cycle_mixer_qc(kappa, h, dt)
    elif kind == 'hypercube':
        mix = hypercube_mixer_qc(nq, h, dt)
    else:
        mix = None
    if mix is not None:
        for _ in range(steps):
            if kind == 'cycle':
                for i in range(d):
                    qc.compose(mix, qubits=range(i * kappa, (i + 1) * kappa),
                               inplace=True)
            else:
                qc.compose(mix, inplace=True)
            qc.compose(pot, inplace=True)
    qc.measure(range(nq), range(nq))
    be = AerSimulator(seed_simulator=seed)
    t = transpile(qc, be, basis_gates=BASIS, optimization_level=1)
    cnt = be.run(t, shots=shots).result().get_counts()
    return cnt, t.depth(), t.count_ops().get('cx', 0)


def model_values(d, kappa, gsub, Hsub, R):
    """The measured quadratic model at every vertex, indexed as the circuit is."""
    a = 2.0 * R / ((1 << kappa) - 1)
    N = 1 << (d * kappa)
    v = np.empty(N)
    for b in range(N):
        t = np.array([a * ((b >> (i * kappa)) & ((1 << kappa) - 1)) - R
                      for i in range(d)])
        v[b] = gsub @ t + 0.5 * t @ Hsub @ t
    return v


if __name__ == '__main__':
    print(__doc__.split('\n')[0])
    print("TIER A - AerSimulator with shots.")
    print("")
    d, kappa, shots = 4, 3, 20000
    nq = d * kappa
    N = 1 << nq
    n_sys = 2
    a_ = anz(n_sys)
    Hm = heis(n_sys)
    M = a_.num_parameters
    sub = [0, 1, 2, 3]
    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, M)
    R = 0.5

    q = QLTOWalk(a_, Hm, shot_budget=1 << 16, sim_seed=11)
    g, H, _ = q.sense(th, R, sub)
    gsub, Hsub = g[sub], H[np.ix_(sub, sub)]
    h0 = QLTOWalk.suggest_h(H, R, sub)

    mv = model_values(d, kappa, gsub, Hsub, R)
    thr = np.quantile(mv, 0.01)                 # top 1% of the box
    good = set(np.nonzero(mv <= thr)[0].tolist())
    print("  box %d vertices, top-1%% set = %d, suggest_h = %.4f"
          % (N, len(good), h0))
    print("  uniform needs 100 moves; Grover's quadratic bound is 10")
    print("")
    print("  %-11s %7s %7s %10s %9s %7s %6s"
          % ("mixer", "h", "t", "P(top1%)", "moves", "depth", "cx"))

    # uniform control - no evolution
    cnt, dep, cx = run('none', d, kappa, gsub, Hsub, R, 0.0, 0.0, 1, shots, 1)
    p = sum(v for k, v in cnt.items() if int(k, 2) in good) / shots
    print("  %-11s %7s %7s %10.4f %9.1f %7d %6d"
          % ('uniform', '-', '-', p, 1.0 / max(p, 1e-9), dep, cx))

    for kind in ('cycle', 'hypercube'):
        for hm in (0.5, 1.0, 2.0):
            for tm in (0.5, 1.0, 2.0, 4.0):
                h = h0 * hm
                t_tot = tm / max(h * h, 1e-9)
                cnt, dep, cx = run(kind, d, kappa, gsub, Hsub, R, h,
                                   t_tot, 12, shots, 7)
                p = sum(v for k, v in cnt.items()
                        if int(k, 2) in good) / shots
                print("  %-11s %7.3f %7.1f %10.4f %9.1f %7d %6d"
                      % (kind, h, t_tot, p, 1.0 / max(p, 1e-9), dep, cx))
    print("")
    print("  moves = 1/P(top 1%). Below 100 beats uniform; below 10 beats")
    print("  Grover's quadratic bound and would mean the walk is exploiting")
    print("  landscape structure rather than just amplitude concentration.")
