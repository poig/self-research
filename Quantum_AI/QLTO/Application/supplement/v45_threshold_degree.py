"""The per-problem number: what degree does the THRESHOLD indicator need?

phase_degree_bound settled the capability of the circuit family exactly:

    MAX ENHANCEMENT(n, d) = sum_{j<=d} C(n, j)

so a degree-1 drift caps at n+1 against a 2^n search space, and a phase
PROPORTIONAL to energy caps near n even with full degree available, because
sin^2 of a proportional phase is a Boltzmann reweighting and cannot peak. Grover
reaches 2^n because its phase is a THRESHOLD: pi below the cut, 0 above.

That makes one quantity decisive, and it is a property of the PROBLEM rather than
of the mixer, the schedule or the shot budget:

    how well is 1[E(x) <= t] approximated by a low-degree polynomial on the
    hypercube?

This file measures it directly. For each block of each problem the full 2^n
hypercube is enumerated, the indicator of the best m corners is formed, and its
Walsh spectrum is taken. Reported is the cumulative fraction of the indicator's
variance captured at degree <= d, and the effective degree d90 - the smallest d
capturing 90%.

A RANDOM landscape is included as the control. A random threshold indicator has
an essentially flat Walsh spectrum, so its weight sits at degree ~n/2 and a
low-degree drift can do nothing with it - that is the unstructured case Grover's
sqrt(N) is optimal for. If the physical Hamiltonians concentrate their indicator
weight at LOW degree, that is exactly the structure a low-degree drift could
exploit, and the gap between them and random is the size of the opportunity.

This is the "per problem class" question made quantitative. It replaces asking
which mixer suits which Hamiltonian with a measurable number per landscape.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def maxcut(N):
    return SparsePauliOp.from_list(
        [("".join("Z" if q in (i, i + 1) else "I" for q in range(N)), 1.0)
         for i in range(N - 1)])


def h2():
    return SparsePauliOp.from_list([("II", -1.0523), ("IZ", 0.3979),
                                    ("ZI", -0.3979), ("ZZ", -0.0113),
                                    ("XX", 0.1809)])


def E(a, H, p):
    return float(np.real(Statevector(a.assign_parameters(p)).expectation_value(H)))


def walsh_spectrum(f, n):
    """Fast Walsh-Hadamard transform; returns coefficients indexed by subset mask."""
    a = f.astype(float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    return a / len(a)


def degree_profile(f, n):
    """Fraction of the (mean-removed) variance at each Walsh degree."""
    c = walsh_spectrum(f, n)
    c[0] = 0.0                                    # drop the mean
    deg = np.array([bin(i).count('1') for i in range(len(c))])
    tot = float(np.sum(c ** 2))
    if tot < 1e-18:
        return np.zeros(n + 1)
    return np.array([float(np.sum(c[deg == d] ** 2)) / tot for d in range(n + 1)])


R = 0.6
PROBLEMS = [("H2", h2(), 1), ("MaxCut N=4", maxcut(4), 1),
            ("MaxCut N=6", maxcut(6), 1), ("Heisenberg N=4", heis(4), 1),
            ("Heisenberg N=6", heis(6), 1), ("Heisenberg N=8", heis(8), 1)]
FRACS = [1, 2, 4]                                  # mark the best m corners

print("=" * 100)
print("THRESHOLD-INDICATOR DEGREE — the per-problem number the bound makes decisive")
print("=" * 100)
print(f"  R={R}. Full hypercube enumeration per block. The indicator marks the")
print(f"  best m corners; its Walsh spectrum says what degree a phase must reach")
print(f"  to express it. d90 is the smallest degree capturing 90% of the variance.")
print()
print(f"  {'problem':>15}{'blk':>4}{'n':>3}{'m':>3}"
      f"{'deg1':>8}{'<=2':>8}{'<=3':>8}{'<=n/2':>8}{'d90':>6}{'cap@1':>8}")
print("  " + "-" * 78)

rows = []
for name, H, reps in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        if n > 10 or bi > 1:
            continue
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        # index by bitmask so the FWHT ordering matches
        order = np.array([int(''.join('1' if s[i] > 0 else '0'
                                      for i in range(n))[::-1], 2) for s in sig])
        e_by = np.empty(2 ** n)
        e_by[order] = vals

        for m in FRACS:
            if m >= 2 ** n:
                continue
            thr = np.sort(e_by)[m - 1]
            f = (e_by <= thr).astype(float)
            prof = degree_profile(f, n)
            cum = np.cumsum(prof)
            d90 = int(np.argmax(cum >= 0.90)) if cum[-1] >= 0.90 else n
            half = max(1, n // 2)
            rows.append((name, n, m, prof[1], cum[min(2, n)], cum[min(3, n)],
                         cum[half], d90))
            print(f"  {name if m == FRACS[0] else '':>15}"
                  f"{bi if m == FRACS[0] else '':>4}{n if m == FRACS[0] else '':>3}"
                  f"{m:>3}{prof[1]:>8.3f}{cum[min(2, n)]:>8.3f}"
                  f"{cum[min(3, n)]:>8.3f}{cum[half]:>8.3f}{d90:>6}"
                  f"{1 + n:>8}", flush=True)
        print("  " + "." * 78)

print("\n  RANDOM LANDSCAPE CONTROL — the unstructured case Grover is optimal for")
print(f"  {'n':>5}{'m':>4}{'deg1':>8}{'<=2':>8}{'<=3':>8}{'<=n/2':>8}{'d90':>6}")
print("  " + "-" * 47)
for n in (4, 6, 8):
    rng = np.random.RandomState(5 + n)
    e = rng.randn(2 ** n)
    for m in FRACS:
        thr = np.sort(e)[m - 1]
        f = (e <= thr).astype(float)
        prof = degree_profile(f, n)
        cum = np.cumsum(prof)
        d90 = int(np.argmax(cum >= 0.90)) if cum[-1] >= 0.90 else n
        half = max(1, n // 2)
        print(f"  {n:>5}{m:>4}{prof[1]:>8.3f}{cum[min(2, n)]:>8.3f}"
              f"{cum[min(3, n)]:>8.3f}{cum[half]:>8.3f}{d90:>6}")

print()
print("  'deg1' is the fraction of the indicator expressible at degree 1, which is")
print("  all the shipped drift has. 'cap@1' is n+1, the enhancement ceiling there.")
print("  Physical landscapes concentrating indicator weight at LOW degree, well")
print("  above the random control, is the structure a low-degree drift could")
print("  exploit - and the size of that gap is the size of the opportunity. If")
print("  the physical rows match the random rows, then these landscapes are")
print("  unstructured from the drift's point of view and no low-degree phase")
print("  helps, whatever the mixer.")
