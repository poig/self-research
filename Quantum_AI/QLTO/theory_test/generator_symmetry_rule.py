"""Which feedback generators can extract work at all? A symmetry selection rule.

su2_check.py showed the isotropic Heisenberg / Y = sum_i X_i pairing gives W = 0
for EVERY state, because the chain is SU(2)-symmetric and therefore commutes with
every total-spin operator. That was one observation. The general statement is a
selection rule, and it follows from Corollary 1 with no further assumptions.

THE RULE. Let C(H) = {O : [H,O] = 0} be the commutant of H. Decompose any
Hermitian generator as

    Y  =  Y_par  +  Y_perp ,      Y_par in C(H) ,

so that [H, Y] = [H, Y_perp]. By Corollary 1 the first-order work depends on Y
only through [H,Y]. Therefore:

    ONLY THE COMPONENT OF THE FEEDBACK GENERATOR OUTSIDE THE HAMILTONIAN'S
    COMMUTANT CAN EXTRACT WORK. A generator lying entirely inside the symmetry
    algebra is inert, for every input state and every sensing time.

This is stronger and more actionable than "check whether your initial state is an
eigenstate": that is a property of the input and is repaired by choosing another
one, whereas this is a property of the H/Y PAIRING and no input repairs it.

WHAT THIS FILE MEASURES
  (a) a family of generators against the isotropic Heisenberg chain, reporting
      ||[H,Y]|| beside the exact work - the rule predicts these vanish together
  (b) that the total-spin generators sum_i X_i, sum_i Y_i, sum_i Z_i are ALL
      inert, not just the X one that happened to be in the protocol
  (c) an explicit split Y = Y_par + Y_perp on a generator with both components,
      confirming that the work tracks Y_perp alone and is blind to Y_par
  (d) the same sweep on a symmetry-BROKEN chain (anisotropic XXZ), where the
      total-spin operators stop being inert - the rule has to distinguish the
      two cases, not merely report zero everywhere
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 4
TAU, THETA = 1.042, 0.2
d = 2 ** N


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def heisenberg(n, delta=1.0):
    """Isotropic at delta=1; anisotropic XXZ otherwise (breaks SU(2) to U(1))."""
    ops = []
    for i in range(n - 1):
        ops.append((lbl(n, **{str(i): "X", str(i + 1): "X"}), 1.0))
        ops.append((lbl(n, **{str(i): "Y", str(i + 1): "Y"}), 1.0))
        ops.append((lbl(n, **{str(i): "Z", str(i + 1): "Z"}), delta))
    return SparsePauliOp.from_list(ops).to_matrix()


def total(n, p):
    return sum(SparsePauliOp(lbl(n, **{str(i): p})).to_matrix() for i in range(n))


def single(n, p, i):
    return SparsePauliOp(lbl(n, **{str(i): p})).to_matrix()


def work(Hm, Ym, psi, tau=TAU, theta=THETA):
    psi1 = expm(-1j * Hm * tau) @ psi
    Psi1 = np.zeros(2 * d, dtype=complex)
    Psi1[:d] = psi / np.sqrt(2)
    Psi1[d:] = psi1 / np.sqrt(2)
    A = np.kron(np.eye(2), Hm)
    K = np.kron(np.diag([0.0, 1.0]), Ym)
    U = expm(-1j * (theta / 2.0) * K)
    return float(np.real(Psi1.conj() @ (A - U.conj().T @ A @ U) @ Psi1))


def comm_norm(Hm, Ym):
    return float(np.linalg.norm(Hm @ Ym - Ym @ Hm))


rng = np.random.RandomState(7)
v = rng.normal(size=d) + 1j * rng.normal(size=d)
psi_rand = v / np.linalg.norm(v)          # deliberately NOT an eigenstate
plus = np.ones(d) / np.sqrt(d)

GENERATORS = [
    ("sum_i X_i          (total spin)", lambda n: total(n, "X")),
    ("sum_i Y_i          (total spin)", lambda n: total(n, "Y")),
    ("sum_i Z_i          (total spin)", lambda n: total(n, "Z")),
    ("X_0                (single site)", lambda n: single(n, "X", 0)),
    ("Z_0                (single site)", lambda n: single(n, "Z", 0)),
    ("X_0 + X_1          (partial sum)", lambda n: single(n, "X", 0) + single(n, "X", 1)),
    ("Z_0 Z_1            (two-body)",
     lambda n: SparsePauliOp(lbl(n, **{"0": "Z", "1": "Z"})).to_matrix()),
]

print("=" * 88)
print("GENERATOR SELECTION RULE — only the part outside the commutant does work")
print("=" * 88)

for tag, delta in (("ISOTROPIC Heisenberg (SU(2) symmetric)", 1.0),
                   ("ANISOTROPIC XXZ, delta=0.4 (SU(2) broken)", 0.4)):
    Hm = heisenberg(N, delta)
    print(f"\n  ===== {tag} =====")
    print(f"  {'generator Y':<34}{'||[H,Y]||':>12}{'W (|+>^n)':>13}"
          f"{'W (random)':>13}{'inert?':>9}")
    print("  " + "-" * 81)
    for name, gen in GENERATORS:
        Ym = gen(N)
        c = comm_norm(Hm, Ym)
        w1 = work(Hm, Ym, plus)
        w2 = work(Hm, Ym, psi_rand)
        inert = "YES" if c < 1e-10 else ""
        print(f"  {name:<34}{c:>12.2e}{w1:>13.2e}{w2:>13.2e}{inert:>9}")

print()
print("  (c) EXPLICIT SPLIT — Y = Y_par + Y_perp on the isotropic chain")
print("  The rule says [H, Y_par + Y_perp] = [H, Y_perp], so the FIRST-ORDER work")
print("  cannot see Y_par. It says nothing about finite theta: U = exp(-i(theta/2)")
print("  P_1 (x) Y) depends on the whole of Y, and [Y_par, Y_perp] != 0 means the")
print("  commuting part re-enters at second order and beyond.")
Hm = heisenberg(N, 1.0)
Y_par = total(N, "X")                      # in the commutant
Y_perp = single(N, "Z", 0)                 # outside it


def first_order(Hm, Ym, psi, tau=TAU, theta=THETA):
    """(theta/4) <psi_1| i[H,Y] |psi_1> - depends on Y only through [H,Y]."""
    psi1 = expm(-1j * Hm * tau) @ psi
    C = 1j * (Hm @ Ym - Ym @ Hm)
    return float(np.real(psi1.conj() @ C @ psi1)) * theta / 4.0


print()
print(f"  {'generator':<24}{'1st order':>12}{'W @0.20':>11}{'W @0.02':>11}"
      f"{'W@0.02 /0.1':>13}")
print("  " + "-" * 71)
for name, Ym in (("Y_perp = Z_0", Y_perp),
                 ("Y_par + Y_perp", Y_par + Y_perp),
                 ("3*Y_par + Y_perp", 3 * Y_par + Y_perp)):
    fo = first_order(Hm, Ym, psi_rand)
    w_big = work(Hm, Ym, psi_rand, theta=0.20)
    w_sml = work(Hm, Ym, psi_rand, theta=0.02)
    print(f"  {name:<24}{fo:>12.6f}{w_big:>11.6f}{w_sml:>11.6f}"
          f"{w_sml / 0.1:>13.6f}")

print()
print("  The 'first order' column must be IDENTICAL across the three rows - that")
print("  is the selection rule. The 'W @0.20' column need not be, and is not.")
print("  Rescaling the small-theta work by 0.1 puts it on the same footing as")
print("  theta=0.2; convergence of that column onto the first-order one is the")
print("  check that the discrepancy is higher-order and not a broken rule.")
print()
print("  SCOPE, therefore. Corollary 1 is EXACT: [H,Y]=0 gives W=0 to all orders,")
print("  because U then commutes with A outright. The DECOMPOSITION statement is")
print("  first-order only. A generator is inert exactly when it lies wholly in the")
print("  commutant; a generator that merely CONTAINS a commuting part is not")
print("  equivalent to its perpendicular component at finite feedback strength.")
