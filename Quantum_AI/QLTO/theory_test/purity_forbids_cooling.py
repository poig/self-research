"""Purity, not the generator, is what forbids directional cooling.

asymmetric_generator.py tried to break the sign degeneracy by choosing a
generator with an asymmetric spectrum, and failed: rank-1 through rank-15
projectors all gave the SAME symmetric interval. The reason is Corollary 3.

For a PURE post-sensing state, M11 = (i/2)[|psi_1><psi_1|, H] has rank at most
two, with eigenvalues +Delta_H, -Delta_H and every other eigenvalue zero. Theorem
2's pairing then reads

    hi = sum_k lam_dn(M11) lam_dn(Y) = Delta_H (lam_max(Y) - lam_min(Y))
    lo = sum_k lam_dn(M11) lam_up(Y) = Delta_H (lam_min(Y) - lam_max(Y)) = -hi

because the zeros annihilate every interior eigenvalue of Y and only the two
extremes survive - antisymmetrically. Therefore:

    FOR A PURE POST-SENSING STATE THE FIRST-ORDER REACHABLE SET IS SYMMETRIC
    ABOUT ZERO FOR EVERY HERMITIAN GENERATOR.

No choice of Y makes this protocol prefer cooling. The defect is not the
unfiltered kick, and not the generator's spectrum: it is the PURITY of the
post-sensing state. A rank-2 M11 cannot pair asymmetrically against anything.

THE PREDICTION THAT FOLLOWS. Mixing the state raises rank(M11) above two, interior
eigenvalues of Y stop being annihilated, and the two orderings stop being
negatives of each other. So the interval should become asymmetric - and the
degree of asymmetry should grow with mixedness.

If that holds it explains, structurally rather than empirically, why every
working single-ancilla cooling scheme is Lindbladian rather than unitary: the
reset is not bookkeeping between cycles, it is where the directionality has to
come from.

CHECKED HERE
  (a) rank(M11) against purity
  (b) the interval endpoints |lo| vs |hi| as the state is mixed
  (c) that the asymmetry is not an artefact of one generator
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


def paper_H(n, seed=42):
    rng = np.random.RandomState(seed)
    ops = []
    for i in range(n):
        for j in range(i + 1, n):
            ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}), rng.uniform(-1, 1)))
    for i in range(n):
        ops.append((lbl(n, **{str(i): "X"}), rng.uniform(-0.5, 0.5)))
    return SparsePauliOp.from_list(ops).to_matrix()


Hm = paper_H(N)
plus = np.ones(d) / np.sqrt(d)
Uev = expm(-1j * Hm * TAU)


def M11_from_rho(rho_S):
    """<1| i[rho, A] |1> for a post-sensing state whose |1> branch is rho_S."""
    return 0.5j * (rho_S @ Hm - Hm @ rho_S)


def interval(M, Ym):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    hi = float(np.sum(lm * ly)) * THETA / 2.0
    lo = float(np.sum(lm * ly[::-1])) * THETA / 2.0
    return lo, hi


GENS = {
    "sum_i X_i": sum(SparsePauliOp(lbl(N, **{str(i): "X"})).to_matrix()
                     for i in range(N)),
    "X_0": SparsePauliOp(lbl(N, **{"0": "X"})).to_matrix(),
}
evals, evecs = np.linalg.eigh(Hm)
GENS["Pi_low(r=4)"] = evecs[:, :4] @ evecs[:, :4].conj().T

print("=" * 88)
print("PURITY FORBIDS DIRECTIONAL COOLING")
print("=" * 88)
print("  Post-sensing |1> branch is rho_p = (1-p)|psi_1><psi_1| + p I/d.")
print("  p = 0 is the protocol as analysed in the manuscript.")

print()
print("  DEPOLARISING MIXING IS THE WRONG TEST, and it is worth recording why.")
print("  rho_p = (1-p)|psi_1><psi_1| + p I/d has i[rho_p, H] = (1-p) i[|psi_1><psi_1|, H],")
print("  because the identity commutes with H. Rank stays 2, everything scales, the")
print("  interval stays exactly symmetric. Admixing the identity adds no asymmetry -")
print("  which is the resource theory speaking: the maximally mixed state is FREE.")
print("  The same argument kills the thermal state: [e^{-beta H}, H] = 0 exactly.")
print()
print("  A mixture must be of states that do NOT commute with H to raise rank(M11).")


def cycle(rho_S, Ym, tau=TAU, theta=THETA):
    """One full sense-actuate-reset cycle; returns the system state after it."""
    Uv = expm(-1j * Hm * tau)
    big = np.zeros((2 * d, 2 * d), dtype=complex)
    # |Psi_1><Psi_1| for ancilla |+> and controlled evolution on the |1> branch
    b0 = np.eye(d) / np.sqrt(2)
    b1 = Uv / np.sqrt(2)
    Wmat = np.vstack([b0, b1])                       # d -> 2d isometry
    big = Wmat @ rho_S @ Wmat.conj().T
    K = np.kron(np.diag([0.0, 1.0]), Ym)
    U = expm(-1j * (theta / 2.0) * K)
    out = U @ big @ U.conj().T
    return out[:d, :d] + out[d:, d:]                 # reset = trace out ancilla


print()
print("  MIXTURES THE PROTOCOL ACTUALLY PRODUCES — state after k cycles")
for gname, Ym in GENS.items():
    print(f"\n  ===== generator: {gname} =====")
    print(f"  {'cycle':>7}{'purity':>10}{'rank M11':>10}{'W_lo':>11}{'W_hi':>11}"
          f"{'|hi|-|lo|':>12}{'asym %':>9}")
    print("  " + "-" * 70)
    rho = np.outer(plus, plus.conj())
    for k in range(6):
        psi1_rho = Uev @ rho @ Uev.conj().T          # |1> branch of this cycle
        M = M11_from_rho(psi1_rho)
        rk = int(np.sum(np.abs(np.linalg.eigvalsh(M)) > 1e-10))
        pur = float(np.real(np.trace(rho @ rho)))
        lo, hi = interval(M, Ym)
        asym = abs(abs(hi) - abs(lo)) / max(abs(hi), 1e-15) * 100
        print(f"  {k:>7}{pur:>10.4f}{rk:>10}{lo:>11.5f}{hi:>11.5f}"
              f"{abs(hi) - abs(lo):>12.2e}{asym:>9.2f}")
        rho = cycle(rho, Ym)

print()
print("  If rank(M11) climbs above 2 once the state is genuinely mixed and the")
print("  interval goes asymmetric, then directionality enters through the RESET,")
print("  not the kick - and Corollary 3's symmetry is a first-cycle artefact of")
print("  starting pure. If it stays symmetric, the obstruction is deeper than")
print("  purity and no single-ancilla unitary feedback can prefer cooling at all.")
