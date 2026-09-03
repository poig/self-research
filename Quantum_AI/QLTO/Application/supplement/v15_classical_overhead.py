"""Classical pre- and post-processing cost, charged to both methods.

The circuit-count claim is only worth something if the circuits I save are not
paid back in local CPU time. QLTO does more classical work per circuit than
parameter-shift does - it builds a W gate from the current parameters, and it
decodes an n-bit marginal table out of every shot instead of averaging one
number. So the saving has to be checked, not assumed.

I split each method into the three costs a real hardware run actually pays:

  BUILD    circuit construction + transpilation. Local CPU, per gradient.
  RUN      submission and execution. On hardware this is queue + shots; here it
           is simulation, so it is NOT comparable across methods and is reported
           only to show what fraction of the measured wall clock is fake.
  DECODE   turning returned counts into gradient components. Local CPU.

BUILD + DECODE is the honest "classical overhead" number. RUN is the thing the
circuit-count claim is about, and it is the one I cannot time on a simulator.

Both arms are implemented the way you would actually run them on hardware:
parameter-shift transpiles the ansatz ONCE per measurement basis and then binds
2M parameter vectors into it, which is the cheapest correct implementation and
therefore the hardest baseline for QLTO to beat.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BACKEND = AerSimulator()
SHOTS = 1024
REP = 3


# ----------------------------------------------------------------- parameter-shift
def ps_prepare(ansatz, H):
    """Transpile once per commuting group. Done at setup, not per gradient."""
    groups = H.group_commuting(qubit_wise=True)
    prepped = []
    for g in groups:
        qc = ansatz.copy()
        # rotate each qubit into the group's measurement basis
        rep = g.paulis[0]
        for q in range(qc.num_qubits):
            lbl = str(rep)[::-1][q]
            if lbl == 'X':
                qc.h(q)
            elif lbl == 'Y':
                qc.sdg(q); qc.h(q)
        qc.measure_all()
        prepped.append((transpile(qc, BACKEND, optimization_level=1), g))
    return prepped


def ps_gradient_timed(prepped, c):
    """2M bindings per group; decode counts to <H> and difference them."""
    M = len(c)
    t0 = time.perf_counter()
    bound, meta = [], []
    for tqc, g in prepped:
        for i in range(M):
            for sgn in (+1.0, -1.0):
                p = c.copy(); p[i] += sgn * np.pi / 2
                bound.append(tqc.assign_parameters(p))
                meta.append((i, sgn, g))
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = BACKEND.run(bound, shots=SHOTS).result()
    t_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    grad = np.zeros(M)
    for j, (i, sgn, g) in enumerate(meta):
        counts = res.get_counts(j)
        ev = 0.0
        for coeff, pauli in zip(g.coeffs, g.paulis):
            lbl = str(pauli)[::-1]
            sub = 0.0
            for bits, ct in counts.items():
                bs = bits.replace(' ', '')[::-1]
                par = sum(int(bs[q]) for q in range(len(lbl)) if lbl[q] != 'I')
                sub += ct * (1 if par % 2 == 0 else -1)
            ev += float(np.real(coeff)) * sub / SHOTS
        grad[i] += sgn * 0.5 * ev
    t_decode = time.perf_counter() - t0
    return t_build, t_run, t_decode, len(bound)


# ----------------------------------------------------------------- QLTO
def qlto_gradient_timed(q, c, R):
    """Per block: build+transpile the sensing circuit, run it, decode marginals."""
    tb = tr = td = 0.0
    ncirc = 0
    for blk in q.layers:
        act = blk['params']
        if not act:
            continue
        t0 = time.perf_counter()
        qc = q._build_qpe_sensing_circuit(c, R, act)
        tqc = transpile(qc, BACKEND, optimization_level=1)
        tb += time.perf_counter() - t0

        t0 = time.perf_counter()
        counts = BACKEND.run(tqc, shots=SHOTS).result().get_counts()
        tr += time.perf_counter() - t0

        t0 = time.perf_counter()
        q._decode_gradient_qpe(counts, c, act, R)
        td += time.perf_counter() - t0
        ncirc += 1
    return tb, tr, td, ncirc


PROBLEMS = [
    ("H2",             B.get_h2_problem),
    ("MaxCut N=4",     lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
]

print("=" * 94)
print("CLASSICAL OVERHEAD — build + decode CPU time per gradient, ms")
print("=" * 94)
print(f"  {SHOTS} shots/circuit, {REP} repeats, medians. RUN is simulation and is")
print("  NOT a hardware-comparable number; it is shown only for context.")

for pname, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS)
    M = ansatz.num_parameters
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)

    prepped = ps_prepare(ansatz, H)
    ps = np.median([ps_gradient_timed(prepped, c) for _ in range(REP)], axis=0)
    ql = np.median([qlto_gradient_timed(q, c, 0.6) for _ in range(REP)], axis=0)

    print(f"\n  ===== {pname} | M={M} | G={len(prepped)} =====")
    print(f"  {'method':<18}{'circuits':>9}{'build':>10}{'decode':>10}"
          f"{'CLASSICAL':>11}{'(run)':>10}")
    print("  " + "-" * 68)
    for tag, t in (("parameter-shift", ps), ("QLTO", ql)):
        print(f"  {tag:<18}{int(t[3]):>9}{t[0]*1e3:>10.1f}{t[2]*1e3:>10.1f}"
              f"{(t[0]+t[2])*1e3:>11.1f}{t[1]*1e3:>10.1f}")
    r = (ql[0] + ql[2]) / (ps[0] + ps[2])
    verdict = f"{1/r:.2f}x CHEAPER" if r < 1 else f"{r:.2f}x MORE EXPENSIVE"
    print(f"  -> QLTO classical overhead is {verdict} than parameter-shift")

print()
print("  If QLTO is cheaper here too, the circuit-count claim stands unqualified.")
print("  If it is more expensive, the claim needs the crossover stated: the saved")
print("  quantum time has to exceed the extra local CPU time.")
