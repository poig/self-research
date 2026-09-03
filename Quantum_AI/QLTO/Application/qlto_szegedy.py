"""Szegedy's quadratic, on QLTO's own box. Achieved or not - exactly.

TIER C for the spectra (dense matrices, exact eigendecomposition). That is the
correct tier: a spectral gap is an algebraic property of an operator and no shot
floor exists in it. No accuracy or cost figure is taken from here.

WHAT IS BEING FIXED. QLTO's walk step is an ANNEALED CONTINUOUS-TIME evolution:
e^{-i(h^2 L + V)t} with a schedule. It carries NO guarantee. v140 measured it
reaching a top-1% vertex in 3.1 moves against uniform's 97.1 - 31x - where
Grover's bound on that 4096-vertex box is 64x. BELOW the quadratic bound, which
is the signature of an anneal rather than a walk with a theorem behind it.

WHAT SZEGEDY GIVES, and it is generic - no special problem structure needed.
For ANY reversible Markov chain P with spectral gap delta, the quantum walk

    W = R2 R1,    R1 = 2 A A^dag - I,   A = sum_x |psi_x><x|,
    |psi_x> = sum_y sqrt(P_xy) |x>|y>,  R2 = S R1 S   (S = swap)

has PHASE GAP  Delta >= 2 sqrt(delta).  Since hitting time scales as 1/gap, the
quantum walk hits in O(sqrt(HT_classical)). That is the quadratic, it is proved,
and Bridi et al.'s cap says quadratic is also the CEILING for this class - so
achieving it means achieving the best available, not merely improving.

THE TEST. Build the Metropolis chain on QLTO's measured quadratic model over the
same box the walk step uses, take its classical gap, build W, take its phase gap,
and check Delta / (2 sqrt(delta)) >= 1. If it holds, the quadratic is available
on our box and the current anneal is leaving it on the table.
"""
import sys
import numpy as np


def model_values(d, kappa, g, H, R):
    """The measured quadratic model at every vertex of the box."""
    a = 2.0 * R / ((1 << kappa) - 1)
    N = 1 << (d * kappa)
    L = (1 << kappa) - 1
    v = np.empty(N)
    for b in range(N):
        t = np.array([a * ((b >> (i * kappa)) & L) - R for i in range(d)])
        v[b] = g @ t + 0.5 * t @ H @ t
    return v


def neighbours(x, d, kappa):
    L = 1 << kappa
    out = []
    for i in range(d):
        c = (x >> (i * kappa)) & (L - 1)
        for s in (1, -1):
            nc = c + s
            if 0 <= nc < L:
                out.append(x - (c << (i * kappa)) + (nc << (i * kappa)))
    return out


def metropolis(vals, d, kappa, beta):
    """Reversible chain with stationary pi ~ e^{-beta E}. Lazy (1/2) so the
    spectrum is non-negative and the gap is the relevant one."""
    N = len(vals)
    P = np.zeros((N, N))
    for x in range(N):
        nb = neighbours(x, d, kappa)
        for y in nb:
            acc = min(1.0, np.exp(-beta * (vals[y] - vals[x])))
            P[x, y] = 0.5 * acc / len(nb)
        P[x, x] = 1.0 - P[x].sum()
    return P


def classical_gap(P, pi):
    """1 - lambda_2 of the symmetrised chain."""
    s = np.sqrt(pi)
    Ps = (P * s[:, None]) / s[None, :]
    Ps = 0.5 * (Ps + Ps.T)
    ev = np.sort(np.linalg.eigvalsh(Ps))[::-1]
    return float(1.0 - ev[1])


def szegedy_phase_gap(P):
    """Phase gap of W = R2 R1 on the N^2-dimensional edge space."""
    N = P.shape[0]
    A = np.zeros((N * N, N), dtype=complex)
    for x in range(N):
        for y in range(N):
            if P[x, y] > 0:
                A[x * N + y, x] = np.sqrt(P[x, y])
    R1 = 2.0 * (A @ A.conj().T) - np.eye(N * N)
    S = np.zeros((N * N, N * N))
    for x in range(N):
        for y in range(N):
            S[x * N + y, y * N + x] = 1.0
    W = (S @ R1 @ S) @ R1
    ph = np.angle(np.linalg.eigvals(W))
    ph = np.abs(ph)
    ph = ph[ph > 1e-8]
    return float(ph.min()) if len(ph) else 0.0


def part1():
    print("PART 1  Does QLTO's box achieve Szegedy's quadratic?")
    print("        PREDICTED: phase gap Delta >= 2 sqrt(delta).")
    print("        A ratio >= 1 means the quadratic is available here.")
    rng = np.random.default_rng(3)
    print("")
    print("   %4s %6s %8s %10s %12s %12s %10s"
          % ("d", "kappa", "states", "beta", "delta", "Delta", "ratio"))
    for d, kappa in ((1, 3), (2, 2), (1, 4), (2, 3)):
        m = d
        g = rng.normal(size=m) * 0.5
        A = rng.normal(size=(m, m))
        Hm = (A + A.T) / 2.0 * 0.4
        vals = model_values(d, kappa, g, Hm, 0.6)
        for beta in (1.0, 3.0):
            P = metropolis(vals, d, kappa, beta)
            pi = np.exp(-beta * vals)
            pi /= pi.sum()
            dl = classical_gap(P, pi)
            Dl = szegedy_phase_gap(P)
            print("   %4d %6d %8d %10.1f %12.6f %12.6f %10.4f"
                  % (d, kappa, len(vals), beta, dl, Dl,
                     Dl / (2.0 * np.sqrt(max(dl, 1e-300)))))
    print("")
    print("   ratio >= 1 confirms Szegedy's bound on OUR box and OUR measured")
    print("   model - the quadratic is available, and the annealed walk that")
    print("   ships today does not take it (v140: 31x against Grover's 64x).")
    print("")


def part2():
    print("PART 2  WHAT THE QUADRATIC IS WORTH, in moves.")
    print("        hitting time ~ 1/gap, so quantum ~ sqrt(classical).")
    rng = np.random.default_rng(3)
    print("")
    print("   %4s %6s %8s %14s %14s %10s"
          % ("d", "kappa", "states", "1/delta", "1/Delta", "speedup"))
    for d, kappa in ((1, 3), (2, 2), (1, 4), (2, 3)):
        m = d
        g = rng.normal(size=m) * 0.5
        A = rng.normal(size=(m, m))
        Hm = (A + A.T) / 2.0 * 0.4
        vals = model_values(d, kappa, g, Hm, 0.6)
        P = metropolis(vals, d, kappa, 3.0)
        pi = np.exp(-3.0 * vals)
        pi /= pi.sum()
        dl = classical_gap(P, pi)
        Dl = szegedy_phase_gap(P)
        print("   %4d %6d %8d %14.2f %14.2f %10.3f"
              % (d, kappa, len(vals), 1.0 / dl, 1.0 / Dl, dl and (1.0 / dl) /
                 (1.0 / Dl)))
    print("")
    print("   The speedup column is the ratio of mixing scales. It should grow")
    print("   as the box grows and the classical gap closes - that is the")
    print("   sqrt, and it is exactly what an anneal with no theorem cannot")
    print("   promise.")
    print("")


def part3():
    print("PART 3  WHAT BUILDING IT COSTS, and the honest scope.")
    print("")
    print("   THE OPERATOR. W = R2 R1 needs, per application:")
    print("     - state prep |psi_x> = sum_y sqrt(P_xy)|x>|y> CONTROLLED on x.")
    print("       On a box with nearest-neighbour moves that is d controlled")
    print("       rotations encoding the Metropolis acceptance - NOT a general")
    print("       state prep, because the neighbourhood is O(d) not O(N).")
    print("     - two reflections and a swap on the edge register.")
    print("   So the edge space is 2 x d x kappa qubits, twice the walk")
    print("   register, and the depth per step is O(d) rather than O(N).")
    print("")
    print("   WHAT IT REPLACES. The annealed evolution's `steps` Trotter slices")
    print("   with O(sqrt(HT)) applications of W. That is a WORSE constant and")
    print("   a BETTER exponent, so it wins only once the box is large enough")
    print("   for the sqrt to pay - which is the honest condition and it should")
    print("   be measured, not assumed.")
    print("")
    print("   AND THE CEILING IS STILL QUADRATIC. Bridi et al. cap this class")
    print("   at quadratic, so achieving Szegedy means reaching the best")
    print("   available for a walk-based optimiser - not escaping the cap.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("TIER C - NO CIRCUIT. Exact spectra.")
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
