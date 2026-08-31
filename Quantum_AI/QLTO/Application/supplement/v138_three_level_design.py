"""What does a 3-level design buy over V6's +-1 one, and is it exact?

TIER B - exact E from Statevector on a full 3^m factorial. R1 admits tier B for
exactness identities and mechanism; no number here is an accuracy or cost figure.

THE SETUP. Every parameter enters through a generator with P^2 = I, so the
landscape's Fourier support is exactly {-1,0,1}^M (v135: 2nd harmonic 5.6e-16)
and for a shift u_j = R sigma_j,

    E(theta + R sigma) = sum_k A_k prod_{j: k_j != 0} g_{k_j}(sigma_j),
    g_kappa(sigma) = 1 + sigma^2 (cos R - 1) + i kappa sigma sin R.

AT q=2, sigma^2 == 1 identically, so g collapses to cos R + i kappa sigma sin R
and the SIGMA-FREE part of every spectator factor is cos R. THAT is the whole
origin of V6's low-pass filter A: c_k -> cos(R)^{|k|-1} c_k. It is an artifact of
two levels, not of finite radius.

AT q=3 the sigma-free part is 1 and sigma^2 is a genuine basis function, so:

  (a) the DIAGONAL curvature d2E/dtheta_j^2 becomes visible. At q=2 it is
      identically invisible - sum_k k_j^2 A_k is degenerate with the constant
      because k_j^2 = 1 whenever k_j != 0 and sigma_j^2 = 1 always.

  (b) the spectator attenuation becomes TUNABLE. Projecting on the orthogonal
      basis {P0=1, P1=sigma, P2=sigma^2-<sigma^2>} attenuates by alpha^{|k|-1}
      with

          alpha = p0 + 2 p1 cos R          p0 + 2 p1 = 1

      so p0, the fraction of rows that leave a coordinate ALONE, trades bias
      against noise. q=2 forces p0 = 0, alpha = cos R, and has no such knob.

WHAT IS BEING TESTED, and (b) is a prediction that could be wrong.
  PART 1  main effects vs exact gradient, sweeping R at p0 = 1/3
  PART 2  the sigma^2 projection vs the exact DIAGONAL Hessian - the thing q=2
          cannot see at all
  PART 3  sigma_j sigma_l vs the exact OFF-DIAGONAL Hessian
  PART 4  sweep p0 and check alpha = p0 + 2 p1 cos R against measured bias
"""
import itertools
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector

LEVELS = np.array([-1.0, 0.0, 1.0])


def heis(n):
    t = []
    for i in range(n - 1):
        for p in ('XX', 'YY', 'ZZ'):
            lab = ['I'] * n
            lab[i], lab[i + 1] = p[0], p[1]
            t.append((''.join(reversed(lab)), 1.0))
    return SparsePauliOp.from_list(t)


def exact_gH(anz, H, th):
    M = len(th)
    E = lambda x: float(np.real(
        Statevector(anz.assign_parameters(x)).expectation_value(H)))
    s = np.pi / 2
    g = np.zeros(M)
    Hs = np.zeros((M, M))
    for j in range(M):
        p, q = th.copy(), th.copy()
        p[j] += s; q[j] -= s
        g[j] = 0.5 * (E(p) - E(q))
    for j in range(M):
        for l in range(M):
            a, b, c, d = th.copy(), th.copy(), th.copy(), th.copy()
            a[j] += s; a[l] += s
            b[j] += s; b[l] -= s
            c[j] -= s; c[l] += s
            d[j] -= s; d[l] -= s
            Hs[j, l] = 0.25 * (E(a) - E(b) - E(c) + E(d))
    return g, Hs


def design_readout(anz, H, th, R, m, p0):
    """Full 3^m factorial with level probabilities (p1, p0, p1); orthogonal
    projections onto P1(sigma_j), P2(sigma_j), P1(sigma_j)P1(sigma_l)."""
    p1 = 0.5 * (1.0 - p0)
    w = np.array([p1, p0, p1])
    s2 = float(w @ (LEVELS ** 2))                # <sigma^2>
    P2 = LEVELS ** 2 - s2
    nP1, nP2 = float(w @ LEVELS ** 2), float(w @ P2 ** 2)

    rows = list(itertools.product(range(3), repeat=m))
    Ev = np.empty(len(rows))
    pw = np.empty(len(rows))
    for i, r in enumerate(rows):
        x = th.copy()
        for j in range(m):
            x[j] += R * LEVELS[r[j]]
        Ev[i] = float(np.real(
            Statevector(anz.assign_parameters(x)).expectation_value(H)))
        pw[i] = np.prod([w[r[j]] for j in range(m)])

    lin = np.zeros(m)
    quad = np.zeros(m)
    cross = np.zeros((m, m))
    for j in range(m):
        f1 = np.array([LEVELS[r[j]] for r in rows])
        f2 = np.array([P2[r[j]] for r in rows])
        lin[j] = float(pw @ (Ev * f1)) / nP1
        # nP2 == 0 exactly when p0 == 0: with levels +-1 only, sigma^2 - <sigma^2>
        # is identically zero. THAT is the q=2 degeneracy, not a numerical edge.
        quad[j] = float(pw @ (Ev * f2)) / nP2 if nP2 > 1e-14 else float("nan")
        for l in range(j + 1, m):
            g1 = np.array([LEVELS[r[l]] for r in rows])
            cross[j, l] = cross[l, j] = \
                float(pw @ (Ev * f1 * g1)) / (nP1 * nP1)
    return lin, quad, cross


if __name__ == '__main__':
    print(__doc__.split('\n')[0])
    print("TIER B - full 3^m factorial from Statevector, no shots.")
    print("")
    n_sys, m = 2, 4                      # 3^4 = 81 rows
    anz = efficient_su2(n_sys, reps=1).decompose()
    Hm = heis(n_sys)
    M = anz.num_parameters
    rng = np.random.default_rng(3)
    th = rng.uniform(-np.pi, np.pi, M)
    g_ex, H_ex = exact_gH(anz, Hm, th)
    sub = list(range(m))

    print("PART 1  main effect / sin R  vs exact gradient   (p0 = 1/3)")
    print("  %8s %12s %12s %10s" % ("R", "rel err", "cos", "alpha pred"))
    for R in (0.6, 0.4, 0.25, 0.15, 0.08):
        lin, _, _ = design_readout(anz, Hm, th, R, m, 1.0 / 3.0)
        est = lin / np.sin(R)
        ex = g_ex[sub]
        rel = float(np.linalg.norm(est - ex) / np.linalg.norm(ex))
        cos = float(est @ ex / (np.linalg.norm(est) * np.linalg.norm(ex)))
        alpha = 1.0 / 3.0 + (2.0 / 3.0) * np.cos(R)
        print("  %8.3f %12.4e %12.6f %10.6f" % (R, rel, cos, alpha))

    print("")
    print("PART 2  sigma^2 projection vs the exact DIAGONAL Hessian")
    print("        (q=2 cannot see this at all: sigma_j^2 == 1)")
    print("  %8s %14s %14s %10s" % ("R", "est diag[0]", "exact diag[0]", "rel"))
    for R in (0.6, 0.4, 0.25, 0.15):
        _, quad, _ = design_readout(anz, Hm, th, R, m, 1.0 / 3.0)
        est = quad / (1.0 - np.cos(R))     # sigma^2 coeff = -(1-cos R) d2E
        ex = np.diag(H_ex)[sub]
        rel = float(np.linalg.norm(est - ex) / np.linalg.norm(ex))
        print("  %8.3f %14.6f %14.6f %10.4f" % (R, est[0], ex[0], rel))

    print("")
    print("PART 3  sigma_j sigma_l vs the exact OFF-DIAGONAL Hessian")
    print("  %8s %14s %14s %10s" % ("R", "est [0,1]", "exact [0,1]", "rel"))
    off = ~np.eye(m, dtype=bool)
    for R in (0.6, 0.4, 0.25, 0.15):
        _, _, cr = design_readout(anz, Hm, th, R, m, 1.0 / 3.0)
        est = cr / (np.sin(R) ** 2)        # sigma_j sigma_l coeff = sin^2R d2E
        ex = H_ex[np.ix_(sub, sub)]
        rel = float(np.linalg.norm((est - ex)[off]) /
                    np.linalg.norm(ex[off]))
        print("  %8.3f %14.6f %14.6f %10.4f" % (R, est[0, 1], ex[0, 1], rel))

    print("")
    print("PART 4  does p0 control the attenuation as alpha = p0 + 2 p1 cos R?")
    print("        gradient rel-err at R = 0.4, sweeping p0")
    print("  %8s %10s %14s %14s" % ("p0", "alpha", "rel err", "1-alpha^3"))
    R = 0.4
    for p0 in (0.0, 0.2, 1.0 / 3.0, 0.5, 0.7, 0.9):
        lin, _, _ = design_readout(anz, Hm, th, R, m, p0)
        p1 = 0.5 * (1 - p0)
        alpha = p0 + 2 * p1 * np.cos(R)
        # lin is ALREADY divided by nP1 = 2 p1 inside design_readout; dividing
        # again was a double normalisation and made the sweep read backwards.
        est = lin / np.sin(R)
        ex = g_ex[sub]
        rel = float(np.linalg.norm(est - ex) / np.linalg.norm(ex))
        print("  %8.3f %10.6f %14.4e %14.4e"
              % (p0, alpha, rel, abs(alpha ** 3 - 1.0)))
    print("")
    print("  p0=0 is the q=2 case (alpha = cos R). If the rel-err falls with")
    print("  p0 the extra level is buying a real knob; if it does not, the")
    print("  alpha model is wrong and q=3's only gain is the diagonal.")
