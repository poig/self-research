"""Before building a THC block-encoding, confirm the scaling it is meant to fix.

The chemistry case for V3 rests on two claimed scalings I have quoted but never
measured: T = Theta(N^4) Pauli terms and G = Theta(N^3) qubit-wise-commuting
groups, giving T/G = Theta(N) - the factor that decides whether QPE's
G-independence beats direct measurement's grouping. Every problem in this suite
has G = 1, 2 or 3 including LiH, so NOTHING HERE EXERCISES THAT REGIME, and a
build costed on an unverified exponent is exactly the mistake pattern this
session has already produced several times.

No chemistry package is needed, because grouping depends only on the PAULI
STRINGS and not on the integral values. The Jordan-Wigner image of the
electronic-structure Hamiltonian has a fixed string structure:

  one-body   a_p^dag a_q + h.c., p < q
             -> (X_p Z...Z X_q + Y_p Z...Z Y_q)/2, Z on the open interval
  number     a_p^dag a_p -> (I - Z_p)/2
  two-body   a_p^dag a_q^dag a_r a_s with four distinct indices
             -> 8 strings, {X,Y}^4 at the four positions with an EVEN number of
                Ys, times Z-strings on the open intervals (p,q) and (r,s)

That is enough to count T and G exactly. Coefficients are set to 1 - they affect
neither the term count nor the grouping.

Reports T, G, T/G and their fitted exponents, then prices V3 against V4 with the
MEASURED G rather than an assumed one.
"""
import sys, os, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import SparsePauliOp


def zstring(lbl, a, b):
    for i in range(a + 1, b):
        if lbl[i] == 'I':
            lbl[i] = 'Z'


def es_hamiltonian_terms(N):
    """Distinct Pauli strings of a JW-mapped electronic-structure Hamiltonian."""
    terms = set()
    for p in range(N):                                   # number operators
        lbl = ['I'] * N; lbl[p] = 'Z'
        terms.add(''.join(lbl))
    for p, q in itertools.combinations(range(N), 2):     # one-body hopping
        for pauli in ('X', 'Y'):
            lbl = ['I'] * N
            lbl[p] = lbl[q] = pauli
            zstring(lbl, p, q)
            terms.add(''.join(lbl))
        lbl = ['I'] * N; lbl[p] = lbl[q] = 'Z'           # Coulomb ZZ
        terms.add(''.join(lbl))
    for quad in itertools.combinations(range(N), 4):     # two-body, 4 distinct
        p, q, r, s = quad
        for combo in itertools.product('XY', repeat=4):
            if combo.count('Y') % 2:
                continue
            lbl = ['I'] * N
            for idx, ch in zip(quad, combo):
                lbl[idx] = ch
            zstring(lbl, p, q)
            zstring(lbl, r, s)
            terms.add(''.join(lbl))
    return sorted(terms)


SIZES = (6, 8, 10, 12)
print("=" * 92)
print("CHEMISTRY SCALING — measured T and G for a JW electronic-structure Hamiltonian")
print("=" * 92)
print("  Pauli strings generated from the JW structure; coefficients are irrelevant")
print("  to both the term count and the qubit-wise-commuting grouping.")
print()
print(f"  {'N':>4}{'T terms':>10}{'G groups':>10}{'T/G':>8}"
      f"{'T/N^4':>9}{'G/N^3':>9}")
print("  " + "-" * 50)

rows = []
for N in SIZES:
    labels = es_hamiltonian_terms(N)
    op = SparsePauliOp.from_list([(s, 1.0) for s in labels])
    G = len(op.group_commuting(qubit_wise=True))
    T = len(labels)
    rows.append((N, T, G))
    print(f"  {N:>4}{T:>10}{G:>10}{T / G:>8.2f}"
          f"{T / N ** 4:>9.3f}{G / N ** 3:>9.3f}", flush=True)

ns = np.array([r[0] for r in rows], float)
aT = np.polyfit(np.log(ns), np.log([r[1] for r in rows]), 1)[0]
aG = np.polyfit(np.log(ns), np.log([r[2] for r in rows]), 1)[0]
print()
print(f"  fitted exponents:   T ~ N^{aT:.2f}    G ~ N^{aG:.2f}    T/G ~ N^{aT - aG:.2f}")
print("  claimed:            T ~ N^4.00    G ~ N^3.00    T/G ~ N^1.00")

print()
print("=" * 92)
print("WHAT THAT DOES TO THE V3 / V4 CHOICE, using MEASURED G")
print("=" * 92)
T_GATE, REP, IBM = 70e-9, 250e-6, 96.0 / 60.0
S = 4096
print("  V3: 3 circuits/epoch, depth ~ Sigma_a r_a * T * 16 gates (kappa=3 -> Sigma=4)")
print("  V4: (G+2) circuits/epoch, depth ~ ansatz")
print()
print(f"  {'N':>4}{'T':>9}{'G':>8}{'V3 depth':>11}{'V3 dur':>10}"
      f"{'V3 $/epoch':>12}{'V4 $/epoch':>12}{'winner':>9}")
print("  " + "-" * 76)
for N, T, G in rows:
    d3 = 4 * T * 16
    dur3 = d3 * T_GATE
    c3 = 3 * S * (REP + dur3) * IBM
    c4 = (G + 2) * S * (REP + (N + 15) * T_GATE) * IBM
    print(f"  {N:>4}{T:>9}{G:>8}{d3:>11}{dur3 * 1e6:>9.0f}us"
          f"{c3:>12.1f}{c4:>12.1f}{'V3' if c3 < c4 else 'V4':>9}")

print()
print("  The crossover here is the whole chemistry case. If V3 loses at every N")
print("  the THC build is what rescues it - depth Theta(T) -> Theta(N^2) - and if")
print("  V3 already wins, the build is unnecessary and the priority is wrong.")
