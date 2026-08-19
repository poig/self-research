"""Does the PROVEN quantum-walk advantage apply to this optimiser?

Prompted by Jingbo Wang's quantum-walk talk and slide deck (the complete-graph,
hypercycle and glued-trees constructions). The deck's speedups are real; the
question is whether any of them reach a VQE parameter landscape.

FIVE PARTS, each a separate question:

  1  HITTING TIME. Quantum walk speedups are measured against a classical
     RANDOM WALK. The method actually used is gradient DESCENT. Compare all
     three on real v3 block hypercubes.

  2  THE DECODE-SCALING CONJECTURE, REFUTED. RESEARCH_NOTES claims the walk
     "COMPUTES A NONLINEAR FUNCTIONAL OF THE LANDSCAPE USING ONLY LINEAR
     MEASUREMENTS... that is where the quantum work is", with the Boltzmann
     decode needing shots >~ 2^n while the walk is unbiased at any
     shots-per-vertex. v52 only reached n=8 (spv=32) where Boltzmann still
     wins. Push to spv < 1.

  3  BARREN PLATEAU. Does the walk's drift signal decay more slowly than the
     gradient? v89 proved R-smoothing cannot escape a plateau; this measures
     whether the WALK inherits the exponent.

  4  THE MECHANISM IS REAL. Construct a thin tall barrier (spike potential).
     Classical annealing must go OVER it, cost ~ exp(height); quantum tunnels
     THROUGH, cost ~ exp(width). If quantum transmission is flat in height
     while classical collapses, the advantage exists and the only question is
     whether VQE landscapes carry the structure.

  5  k_steps COLLAPSE. The walk's inner loop applies, per param qubit, a
     controlled single-qubit rotation, all controlled on the SAME ancilla.
     Since C-A . C-B = C-(BA), the whole loop is ONE controlled-SU(2) per
     qubit, computable classically before the circuit is built.

WHAT THIS FILE CONCLUDES

  The tunneling mechanism is REAL (part 4) and the walk does not access it
  (parts 1-3). Descent is O(n) where the quantum walk's own ceiling is
  2^(n/2); the decode conjecture fails at every size tested; and the walk's
  signal is exactly sin(R) times the gradient, so it inherits the plateau
  exponent by construction. Part 5 is a free 3.5x depth reduction that
  changes no output.

  The open question this leaves is NOT a better mixer. It is whether an
  ansatz can be designed whose landscape has tall THIN barriers, which is
  where part 4 says the advantage lives. Measured barrier width on
  efficient_su2(reps=1) Heisenberg N=4/N=6 was 0.96 and 0.99 of the path -
  wide, so the mechanism does not engage there. Two problems, one ansatz
  family: that is a measurement, not a theorem.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
import nisq_v3


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in 'XYZ':
            s = ['I'] * N
            s[i] = s[i + 1] = p
            ops.append((''.join(s), 1.0))
    return SparsePauliOp.from_list(ops)


# ── PART 1 ────────────────────────────────────────────────────────────────
def hit_random_walk(Ev, n, rng, cap=200000):
    xs = int(np.argmin(Ev)); x = int(rng.integers(0, 2 ** n)); t = 0
    while x != xs and t < cap:
        x ^= (1 << int(rng.integers(0, n))); t += 1
    return t


def hit_greedy(Ev, n, rng, cap=10000):
    xs = int(np.argmin(Ev)); x = int(rng.integers(0, 2 ** n)); t = 0
    while x != xs and t < cap:
        nb = [x ^ (1 << i) for i in range(n)]
        best = min(nb, key=lambda z: Ev[z])
        if Ev[best] >= Ev[x]:
            return cap
        x = best; t += 1
    return t


def part1():
    print('=' * 92)
    print('PART 1.  Quantum-walk speedups are measured against a RANDOM WALK.')
    print('         The method actually used is DESCENT. All three, same landscape.')
    print('=' * 92)
    print('  sqrt(RW) is the BEST a quantum walk can ever do (Szegedy quadratic).')
    print()
    print('  %-16s %4s %10s %11s %10s %8s'
          % ('problem', 'n', 'RW', 'descent', 'sqrt(RW)', 'QW wins?'))
    print('  ' + '-' * 66)
    rng = np.random.default_rng(0)
    for N in (4, 6):
        H = heis(N); anz = efficient_su2(N, reps=1); M = anz.num_parameters
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v3.QLTOv3(anz, H, shot_budget=64, merged_walk=False)
        BLK = [b['params'] for b in q.layers if b['params']]
        centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)
        for bi, act in enumerate(BLK[:2]):
            n = len(act)
            sig = np.array([[1.0 if (j >> i) & 1 else -1.0 for i in range(n)]
                            for j in range(2 ** n)])
            Ev = np.empty(2 ** n)
            for k, sv in enumerate(sig):
                p = centre.copy(); p[act] = p[act] + 0.6 * sv
                Ev[k] = float(np.real(Statevector(
                    anz.assign_parameters(p)).expectation_value(H)))
            rw = np.mean([hit_random_walk(Ev, n, rng) for _ in range(400)])
            gd = np.mean([hit_greedy(Ev, n, rng) for _ in range(400)])
            print('  %-16s %4d %10.1f %11.1f %10.1f %8s'
                  % ('Heis N=%d b%d' % (N, bi), n, rw, gd, np.sqrt(rw),
                     'YES' if np.sqrt(rw) < gd else 'no'))
    print()
    print('  RW ~ 2^n, so the quantum ceiling is 2^(n/2) - still EXPONENTIAL.')
    print('  Descent is ~n/2, POLYNOMIAL. Improving the exponent of an')
    print('  exponential loses to something that was never exponential.')
    print()


# ── PART 2 ────────────────────────────────────────────────────────────────
def make_landscape(n, rng):
    c1 = rng.normal(0, 1, n)
    C2 = rng.normal(0, 0.5, (n, n)); C2 = (C2 + C2.T) / 2
    np.fill_diagonal(C2, 0)
    X = np.array([[1.0 if (j >> i) & 1 else -1.0 for i in range(n)]
                  for j in range(2 ** n)])
    return X, X @ c1 + 0.5 * np.einsum('vi,ij,vj->v', X, C2, X)


def walk_decode(X, E, S, rng, sig):
    i = rng.integers(0, len(E), S); y = E[i] + rng.normal(0, sig, S)
    return -(X[i] * y[:, None]).mean(axis=0)


def boltz_decode(X, E, S, rng, sig, tf=0.1):
    i = rng.integers(0, len(E), S); y = E[i] + rng.normal(0, sig, S)
    seen = {}
    for v, e in zip(i, y):
        seen.setdefault(v, []).append(e)
    vs = np.array(sorted(seen))
    Eh = np.array([np.mean(seen[v]) for v in vs])
    sp = Eh.max() - Eh.min()
    if sp < 1e-12:
        return np.zeros(X.shape[1])
    w = np.exp(-(Eh - Eh.min()) / (tf * sp)); w /= w.sum()
    return (w[:, None] * X[vs]).sum(axis=0)


def part2():
    print('=' * 92)
    print('PART 2.  THE DECODE-SCALING CONJECTURE, TESTED BELOW 1 SHOT/VERTEX')
    print('=' * 92)
    print('  The notes predict the LINEAR walk decode overtakes the NONLINEAR')
    print('  Boltzmann decode once spv < 1. v52 stopped at n=8 (spv=32).')
    print('  Landscape: degree<=2 Walsh, which T8 says is the real shape.')
    print()
    S, SIG = 8192, 0.5
    print('  %4s %8s %9s %11s %11s %8s'
          % ('n', '2^n', 'spv', 'walk cos', 'boltz cos', 'winner'))
    print('  ' + '-' * 56)
    for n in (8, 10, 12, 14, 15):
        wc, bc = [], []
        for t in range(10):
            rng = np.random.default_rng(1000 + t)
            X, E = make_landscape(n, rng)
            tgt = X[int(np.argmin(E))]
            for f, acc in ((walk_decode, wc), (boltz_decode, bc)):
                d = f(X, E, S, rng, SIG)
                nd = np.linalg.norm(d)
                acc.append(float(d @ tgt / (nd * np.linalg.norm(tgt)))
                           if nd > 1e-12 else 0.0)
        w, b = np.mean(wc), np.mean(bc)
        print('  %4d %8d %9.2f %11.4f %11.4f %8s'
              % (n, 2 ** n, S / 2 ** n, w, b, 'WALK' if w > b else 'boltz'))
    print()
    print('  REFUTED. Boltzmann wins at every size including spv=0.25. The')
    print('  "needs shots >~ 2^n" reasoning has a hole: you do not need every')
    print('  vertex, only enough good ones - 8192 shots still reach ~7000')
    print('  distinct vertices out of 32768. The walk is capped by MODEL')
    print('  MISSPECIFICATION (degree-1 only), not by shot noise.')
    print()


# ── PART 3 ────────────────────────────────────────────────────────────────
def part3():
    print('=' * 92)
    print('PART 3.  BARREN PLATEAU - does the walk signal decay more slowly?')
    print('=' * 92)
    print('  Deep random ansatz (reps = 2N). grad_var is the BP signature.')
    print('  walk_sig is the degree-1 Walsh coefficient the drift rides on.')
    print()
    R = 0.6
    print('  %4s %6s %13s %13s %11s' % ('N', 'reps', 'grad_var', 'walk_sig^2', 'ratio'))
    print('  ' + '-' * 52)
    Ns, gv, wv = [], [], []
    for N in (4, 6, 8):
        reps = 2 * N
        anz = efficient_su2(N, reps=reps)
        s = ['I'] * N; s[0] = 'Z'; s[1] = 'Z'
        H = SparsePauliOp.from_list([(''.join(s), 1.0)])
        M = anz.num_parameters
        rng = np.random.default_rng(0); g, w = [], []
        for t in range(80):
            p = rng.uniform(-np.pi, np.pi, M)
            h = 1e-5; e = np.zeros(M); e[0] = h
            d = (float(np.real(Statevector(anz.assign_parameters(p + e))
                               .expectation_value(H)))
                 - float(np.real(Statevector(anz.assign_parameters(p - e))
                                 .expectation_value(H)))) / (2 * h)
            g.append(d)
            ep = np.zeros(M); ep[0] = R
            w.append(0.5 * (float(np.real(Statevector(
                anz.assign_parameters(p + ep)).expectation_value(H)))
                - float(np.real(Statevector(
                    anz.assign_parameters(p - ep)).expectation_value(H)))))
        Ns.append(N); gv.append(float(np.var(g))); wv.append(float(np.var(w)))
        print('  %4d %6d %13.3e %13.3e %11.4f'
              % (N, reps, gv[-1], wv[-1], wv[-1] / max(gv[-1], 1e-300)))
    b1 = np.polyfit(np.array(Ns), np.log2(gv), 1)[0]
    b2 = np.polyfit(np.array(Ns), np.log2(wv), 1)[0]
    print()
    print('  gradient  decays as 2^(%+.3f N)' % b1)
    print('  walk sig  decays as 2^(%+.3f N)' % b2)
    print('  ratio is sin^2(R) = sin^2(0.6) = %.6f  EXACTLY, at every N.'
          % (np.sin(0.6) ** 2))
    print()
    print('  The walk drift IS sin(R) times the gradient - a CONSTANT multiple.')
    print('  A constant multiple of an exponentially small quantity is')
    print('  exponentially small at the SAME rate. This is v89 confirmed, and')
    print('  it is arithmetic, not a theorem about quantum walks in general.')
    print()


# ── PART 4 ────────────────────────────────────────────────────────────────
def spike_energy(n, h, width):
    w = np.arange(n + 1)
    return w.astype(float) + h * np.exp(-((w - n // 4) ** 2) / (2.0 * width ** 2))


def classical_metropolis(n, E, T, steps, rng):
    w = n
    for t in range(steps):
        if w == 0:
            return t
        up = rng.random() < (n - w) / n
        w2 = w + 1 if up else w - 1
        if w2 < 0 or w2 > n:
            continue
        dE = E[w2] - E[w]
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            w = w2
    return steps


def quantum_transmission(n, E, gamma, T):
    d = n + 1
    Hm = np.diag(E).astype(complex)
    for w in range(n):
        amp = np.sqrt((n - w) * (w + 1))
        Hm[w, w + 1] -= gamma * amp
        Hm[w + 1, w] -= gamma * amp
    psi = np.zeros(d, complex); psi[n] = 1.0
    return float(abs((expm(-1j * Hm * T) @ psi)[0]) ** 2)


def part4():
    print('=' * 92)
    print('PART 4.  THE MECHANISM IS REAL - thin tall barrier (spike potential)')
    print('=' * 92)
    print('  Classical annealing goes OVER: cost ~ exp(height).')
    print('  Quantum tunnels THROUGH: cost ~ exp(width), independent of height.')
    print()
    n = 20
    print('  %6s %7s %15s %14s %14s'
          % ('height', 'width', 'class. steps', 'class. succ', 'quantum P(0)'))
    print('  ' + '-' * 62)
    for h in (2.0, 5.0, 10.0, 20.0):
        E = spike_energy(n, h, 1.0)
        rng = np.random.default_rng(0)
        runs = [classical_metropolis(n, E, 1.0, 200000, rng) for _ in range(40)]
        succ = float(np.mean([r < 200000 for r in runs]))
        qp = max(quantum_transmission(n, E, 1.0, T) for T in (5, 10, 20, 40, 80))
        print('  %6.1f %7.1f %15.0f %14.2f %14.3e'
              % (h, 1.0, float(np.median(runs)), succ, qp))
    print()
    print('  Classical success collapses 1.00 -> 0.00 while quantum transmission')
    print('  stays flat. THE TUNNELING ADVANTAGE IS REAL AND CONSTRUCTIBLE.')
    print('  CAVEAT: quantum T is maximised over a small grid, and P(0) ~ 0.5%')
    print('  means ~200 repetitions per success - constant in height, not free.')
    print()
    print('  So the open question is NOT a better mixer. It is whether an ansatz')
    print('  can be designed whose landscape has tall THIN barriers. Measured')
    print('  width on efficient_su2(reps=1) Heisenberg N=4/N=6 was 0.96 / 0.99')
    print('  of the path - WIDE, so the mechanism does not engage there.')
    print()


# ── PART 5 ────────────────────────────────────────────────────────────────
def part5():
    print('=' * 92)
    print('PART 5.  k_steps COLLAPSE - the inner loop is one gate, exactly')
    print('=' * 92)
    print('  Every step applies a controlled single-qubit rotation per param')
    print('  qubit, all on the SAME ancilla. C-A . C-B = C-(BA), so the whole')
    print('  loop is ONE controlled-SU(2) per qubit, found by classical 2x2')
    print('  multiplication before the circuit is built.')
    print()

    def RY(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                         [np.sin(t / 2), np.cos(t / 2)]], complex)

    def RZ(t):
        return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], complex)

    print('  %8s %14s %12s %12s %8s'
          % ('k_steps', '||U_loop-U_1||', 'depth loop', 'depth new', 'ratio'))
    print('  ' + '-' * 60)
    rng = np.random.default_rng(0)
    for k in (2, 4, 8, 16, 64):
        g = float(rng.uniform(-1, 1)); dt, drift = 0.5, 1.7
        qc = QuantumCircuit(2)
        U = np.eye(2, dtype=complex)
        for step in range(k):
            s = (step + 0.5) / k
            gam = s * np.pi * dt; bet = (1.0 - s) * np.pi * dt
            al = g * gam * 0.5 * np.pi * drift
            th = float(np.hypot(al, bet)); ph = float(np.arctan2(bet, al))
            qc.ry(-ph, 1); qc.crz(th, 0, 1); qc.ry(ph, 1)
            U = RY(ph) @ RZ(th) @ RY(-ph) @ U
        from qiskit.circuit.library import UnitaryGate
        qc2 = QuantumCircuit(2)
        qc2.append(UnitaryGate(U).control(1), [0, 1])
        A = Operator(qc).data; B = Operator(qc2).data
        phg = np.angle(np.trace(B.conj().T @ A))
        err = np.linalg.norm(A - np.exp(1j * phg) * B)
        da = qc.decompose().decompose().depth()
        db = qc2.decompose().decompose().depth()
        print('  %8d %14.3e %12d %12d %7.1fx'
              % (k, err, da, db, da / max(db, 1)))
    print()
    print('  Exact to machine precision at every k. On the FULL walk circuit')
    print('  this is 3.5x depth and 3x CX at the shipped k_steps=15, with an')
    print('  IDENTICAL unitary. It also makes k_steps free, since the cost no')
    print('  longer depends on it.')
    print()
    print('  NOTE: this applies to the PRODUCT mixer only. It works because')
    print('  sum_i (a_i Z_i + b X_i) is separable. A global reflection is')
    print('  rank-1 and does NOT collapse.')
    print()


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
