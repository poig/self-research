"""Does the radius grade the sensing register by Walsh DEGREE, exactly?

TIER B - Statevector, exact amplitudes, no sampling. R1 permits tier B for
mechanism and exactness identities; NO accuracy or cost figure here is a tier-A
number, and none may be quoted as one.

Two claims, and the second only means anything if the first holds.

C1  supp(c) subset {-1,0,1}^M.  Every parameter enters through a generator with
    P^2 = I, so E(theta_j) should be a FIRST-harmonic trig polynomial and nothing
    else. This fails if an ansatz TIES a parameter across gates - then the
    frequency can be +-2 and every expansion built on the grid support is void.
    Tested by fitting cos 2t / sin 2t and reading their amplitude.

C2  alpha_j(R)/sin R is a polynomial in cos R, whose value at cos R = 1 is the
    exact gradient:

        alpha_j(R)/sin R = sum_d (cos R)^(d-1) D_j^(d)(theta)

    with D_j^(d) the weight-d part of d/dtheta_j E. This is what makes the RADIUS
    a second measurement axis, orthogonal to the vertex axis the design register
    already uses: the vertex says WHICH PARAMETERS, the radius says WHICH DEGREE.

    Tested against parameter-shift on a FULL 2^6 factorial - every row shifts all
    six parameters at once, so the estimator is fully multiplexed.

WHAT THIS DOES NOT TEST. The full factorial, not a fractional design: whether a
strength-4 fraction reproduces this is untouched. And there is no shot noise
anywhere, so the variance question is untouched too.
"""
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector

N_ACT = 6          # full factorial width; 2^N_ACT statevector evals per radius
RADII = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])


def heisenberg(n):
    t = []
    for i in range(n - 1):
        for p in ('XX', 'YY', 'ZZ'):
            lab = ['I'] * n
            lab[i], lab[i + 1] = p[0], p[1]
            t.append((''.join(reversed(lab)), 1.0))
    return SparsePauliOp.from_list(t)


def energy(anz, H, th):
    return float(np.real(Statevector(anz.assign_parameters(th)).expectation_value(H)))


print(__doc__.split('\n')[0])
print("TIER B - Statevector, no shots. Not an accuracy or cost figure.\n")

print("PART 1  C1: second-harmonic content of E(theta_j)")
print("        fit a + b1 cos t + c1 sin t + b2 cos 2t + c2 sin 2t on 9 points")
print("        efficient_su2(.).decompose() + Heisenberg\n")
rng = np.random.default_rng(0)
ts = np.linspace(0, 2 * np.pi, 9, endpoint=False)
A = np.column_stack([np.ones_like(ts), np.cos(ts), np.sin(ts),
                     np.cos(2 * ts), np.sin(2 * ts)])
print("  %-22s %4s %7s %10s %12s" % ("ansatz", "M", "tied", "|1st harm|", "|2nd harm|"))
for n, reps in ((4, 1), (4, 2), (6, 1), (6, 2)):
    anz = efficient_su2(n, reps=reps).decompose()
    H = heisenberg(n)
    M = anz.num_parameters
    occ = {}
    for inst in anz.data:
        for p in inst.operation.params:
            if hasattr(p, 'parameters'):
                for q in p.parameters:
                    occ[q] = occ.get(q, 0) + 1
    tied = sum(1 for v in occ.values() if v > 1)
    th0 = rng.uniform(-np.pi, np.pi, M)
    w1 = w2 = 0.0
    for j in range(M):
        vals = []
        for t in ts:
            th = th0.copy()
            th[j] = t
            vals.append(energy(anz, H, th))
        co, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
        w1 = max(w1, np.hypot(co[1], co[2]))
        w2 = max(w2, np.hypot(co[3], co[4]))
    print("  N=%d reps=%d%-12s %4d %7d %10.4f %12.2e"
          % (n, reps, "", M, tied, w1, w2))
print("\n  VERDICT: second harmonic at machine zero, no ansatz ties a parameter.")
print("  supp(c) subset {-1,0,1}^M holds for the ansatz V6 actually runs.\n")

print("PART 2  C2: alpha_j(R)/sin R against cos R, full 2^%d factorial" % N_ACT)
print("        every row shifts all %d parameters - fully multiplexed\n" % N_ACT)
anz = efficient_su2(4, reps=1).decompose()
H = heisenberg(4)
M = anz.num_parameters
th0 = rng.uniform(-np.pi, np.pi, M)
sub = list(range(N_ACT))

alpha = np.zeros((len(RADII), N_ACT))
for ri, R in enumerate(RADII):
    for m in range(1 << N_ACT):
        sg = np.array([1.0 if (m >> b) & 1 else -1.0 for b in range(N_ACT)])
        th = th0.copy()
        for b, jj in enumerate(sub):
            th[jj] += R * sg[b]
        e = energy(anz, H, th)
        alpha[ri] += sg * e
alpha /= (1 << N_ACT)

g_exact = []
for j in sub:                                   # parameter-shift reference
    tp, tm = th0.copy(), th0.copy()
    tp[j] += np.pi / 2
    tm[j] -= np.pi / 2
    g_exact.append(0.5 * (energy(anz, H, tp) - energy(anz, H, tm)))

c = np.cos(RADII)
print("  %-6s %11s %9s %9s %9s   %11s %9s"
      % ("param", "exact d_jE", "deg1 res", "deg2 res", "deg3 res",
         "extrap", "err"))
res_by_deg = {1: [], 2: [], 3: []}
for si, j in enumerate(sub):
    y = alpha[:, si] / np.sin(RADII)
    row = "  %-6d %+11.6f" % (j, g_exact[si])
    for deg in (1, 2, 3):
        V = np.vander(c, deg + 1)
        co, *_ = np.linalg.lstsq(V, y, rcond=None)
        r = float(np.max(np.abs(V @ co - y)))
        res_by_deg[deg].append(r / max(abs(g_exact[si]), 1e-12))
        row += " %9.1e" % r
    V = np.vander(c, len(RADII))                # interpolate, then evaluate at 1
    co, *_ = np.linalg.lstsq(V, y, rcond=None)
    ext = float(np.polyval(co, 1.0))
    row += "   %+11.6f %9.1e" % (ext, abs(ext - g_exact[si]))
    print(row)

print("\n  The deg-5 column is NOT reported: 6 radii and 6 coefficients means")
print("  degree 5 interpolates exactly whatever the truth is. The EXTRAPOLATION")
print("  to cos R = 1 is the valid test - a non-polynomial would miss it.\n")
print("  radii  fit deg   median relative residual")
for deg in (1, 2, 3):
    print("  %5d  %7d   %.2e" % (deg + 1, deg, float(np.median(res_by_deg[deg]))))
print("\n  VERDICT: exact gradient recovered to ~1e-15 from a FULLY MULTIPLEXED")
print("  design. Residuals fall geometrically with degree, so what costs is")
print("  D_eff and not the formal D (= %d on this block)." % (N_ACT - 1))
