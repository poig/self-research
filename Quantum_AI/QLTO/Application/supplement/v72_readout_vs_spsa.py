"""Is the READOUT the only quantum content? The half v67 could not see.

v67 established the negative half exactly: with the displacement radii matched,
QLTO's degree-1 Walsh marginal under antithetic sampling IS the SPSA estimator.
Measured gap 0.0000 at every sample count and both sizes tested. The superposition
over the +-R hypercube reproduces classical random perturbation and buys nothing
by itself.

But v67 ran at EXACT ENERGIES, which is precisely where a readout cannot matter.
It therefore proved that the sampling designs coincide and said nothing about the
claim that actually carries weight:

    the quantum content of this family of in-circuit optimisers is the READOUT,
    not the superposition.

T4 gives the mechanism. SPSA assigns one scalar (grad E . sigma) to every
coordinate, so each component inherits the noise of all the others:

    Var_SPSA(g_i)  ~  |grad E|^2 - (d_iE)^2      grows with M, no shot budget
                                                 removes it - it is the
                                                 estimator's own structure

QLTO's shot returns a BOUNDED +-1 ancilla bit whose variance cannot exceed 1
however much the energy varies across the hypercube, so the cross-coordinate term
is absent (T4 measured b/a = -0.004).

THE TEST. Both estimators at MATCHED TOTAL SHOTS, scored against the same exact
gradient. SPSA is charged honestly:
  - it needs G measurement settings per energy, as QLTO does, so a sigma sample
    costs 2G circuits against QLTO's G;
  - its energies are estimated from real sampling, with Var(<H_g>) computed
    EXACTLY from the statevector rather than taken from an estimator that returns
    exact values plus fixed-width noise. That subsidy is the documented trap in
    these notes and is avoided here;
  - it is given the SAME displacement radius, so neither carries a bias the other
    does not.

Swept over shot budget and over M, because the claim is about the cross-talk term
and that term is the one that grows with M.

WHAT WOULD CONFIRM IT: QLTO ahead at matched shots, with the margin WIDENING in M.
WHAT WOULD KILL IT: parity or better for SPSA, which would mean the bounded
readout buys nothing either, and the whole family reduces to classical stochastic
approximation with extra steps. That outcome is worth as much as the other and
should be reported as plainly.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def exact_grad(ansatz, Hm, theta):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            g[i] += s * float(np.real(np.conj(v) @ (Hm @ v))) / 2
    return g


def energy_sampled(ansatz, groups, gmats, theta, shots, rng):
    """<H> with honest shot noise: Var = sum_g Var(H_g)/shots, each Var computed
    exactly from the statevector. No estimator subsidy."""
    v = Statevector(ansatz.assign_parameters(theta)).data
    tot = 0.0
    for Hg, Hg2 in gmats:
        m1 = float(np.real(np.conj(v) @ (Hg @ v)))
        m2 = float(np.real(np.conj(v) @ (Hg2 @ v)))
        var = max(m2 - m1 * m1, 0.0)
        tot += m1 + rng.normal(0.0, np.sqrt(var / max(shots, 1)))
    return tot


R = 0.45
REPEATS = 8
BUDGETS = (2 ** 13, 2 ** 15, 2 ** 17)

print("=" * 100)
print("IS THE READOUT THE QUANTUM PART?  QLTO vs SPSA at matched TOTAL shots")
print("=" * 100)
print(f"  Same radius R = {R} for both, so neither carries a bias the other does not.")
print(f"  SPSA charged 2G circuits per sigma sample, QLTO G per block; both spend")
print(f"  the same TOTAL shots. Scored by cos(g_hat, grad E). {REPEATS} repeats.")
print()
print(f"  {'N':>3}{'M':>4}{'T total':>10}{'cos QLTO':>11}{'cos SPSA':>11}"
      f"{'1-cos QL':>11}{'1-cos SP':>11}{'ratio':>8}{'winner':>9}")
print("  " + "-" * 78)

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
    groups = probe.groups
    G = len(groups)
    blocks = [b['params'] for b in probe.layers if b['params']]
    L = len(blocks)
    gmats = [(g.to_matrix(), (g @ g).simplify().to_matrix()) for g in groups]

    with contextlib.redirect_stdout(io.StringIO()):
        qs = [nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct',
                             sim_seed=700 + r) for r in range(REPEATS)]

    for T in BUDGETS:
        # QLTO: G*L circuits, T/(G*L) shots each
        Sq = max(1, T // (G * L))
        cq = []
        for q in qs:
            q.shot_budget = int(Sq)
            gh = np.zeros(M)
            for act in blocks:
                gi, _ = q.sense(theta, R, act)
                gh += gi
            cq.append(cosine(gh, g_ex))

        # SPSA: K sigma samples, each 2G circuits, so shots per energy is
        # T / (2*G*K). K chosen to balance sampling against per-energy noise,
        # swept and the BEST taken, which favours SPSA.
        best_sp = -2.0
        for K in (M // 4, M // 2, M, 2 * M, 4 * M):
            if K < 1:
                continue
            Sp = T // (2 * G * K)
            if Sp < 1:
                continue
            cs = []
            for rep in range(REPEATS):
                rng = np.random.RandomState(9000 + rep)
                g_sp = np.zeros(M)
                for _ in range(K):
                    sig = rng.choice([-1.0, 1.0], size=M)
                    ep = energy_sampled(ansatz, groups, gmats, theta + R * sig,
                                        Sp, rng)
                    em = energy_sampled(ansatz, groups, gmats, theta - R * sig,
                                        Sp, rng)
                    g_sp += ((ep - em) / (2.0 * R)) * sig
                g_sp /= K
                cs.append(cosine(g_sp, g_ex))
            best_sp = max(best_sp, float(np.mean(cs)))

        mq, ms = float(np.mean(cq)), best_sp
        eq, es = max(1 - mq, 1e-9), max(1 - ms, 1e-9)
        win = 'QLTO' if eq < es else 'SPSA'
        if abs(eq - es) / max(eq, es) < 0.05:
            win = 'tie'
        print(f"  {N:>3}{M:>4}{T:>10}{mq:>11.4f}{ms:>11.4f}{eq:>11.5f}"
              f"{es:>11.5f}{es / eq:>8.2f}{win:>9}", flush=True)
    print("  " + "." * 78)

print()
print("  'ratio' is SPSA error over QLTO error: above 1 means the bounded readout")
print("  is winning. The claim under test is that it wins AND that the margin")
print("  grows with M, since the term it removes is |grad E|^2 - (d_iE)^2 and that")
print("  is what grows. A flat or inverted ratio says the superposition and the")
print("  readout both buy nothing, and this family of in-circuit optimisers is")
print("  classical stochastic approximation with extra steps.")
