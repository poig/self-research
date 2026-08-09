"""Is v14's plateau a property of the estimator, or an artefact of fixing R?

THE STANDING CONCLUSION, from the CORRECTION in "IS IT ACTUALLY CHEAPER?":

    "QLTO PLATEAUS. The last two rows of each QLTO column are equal within noise:
     more shots buy nothing, because the error is bias, not variance. The floor
     sits at cos ~= 0.977 ... So at MATCHED SHOTS parameter-shift wins beyond a
     modest budget. QLTO's advantage is circuit count and nothing else. '3.2x
     fewer total shots' is withdrawn."

That measurement held R = 0.6 at every shot budget. But the two error terms move
in OPPOSITE directions in R:

    bias      ~ c R^2            (T3: Ehat({i})/R = d_iE + O(R^2 sup|d^3 E|))
    variance  ~ a / (R^2 S)      (T4: Var(g_i) = a/S with a = 1/(S R^2 tau^2))

    total^2   ~ c^2 R^4 + a/(R^2 S)

Minimising over R gives R* ~ S^(-1/6) and total error ~ S^(-1/3). So the RIGHT
protocol shrinks R as the budget grows, and the error should keep falling - just
at exponent 1/3 instead of the unbiased 1/2. A FIXED R CANNOT DO THAT AND WILL
PLATEAU AT EXACTLY c R^2 NO MATTER THE BUDGET, which is what v14 reported.

If that is what happened, "QLTO's advantage is circuit count and nothing else" was
withdrawn on the strength of a protocol artefact, and the shot question is open
again. If instead the error still plateaus with R free, the withdrawal stands on
firmer ground than it did and the bias floor is real.

THE COMPARISON IS AT MATCHED TOTAL SHOTS, which is the currency the CORRECTION
rightly insisted on. For one full gradient:

    QLTO           C_q = G * L circuits          S_q = T / C_q shots each
    parameter-shift C_p = 2 M G circuits         S_p = T / C_p shots each

QLTO gets FEWER circuits so it can afford MORE shots in each - that is the whole
trade, and only a matched-T comparison prices it.

BASELINE HONESTY. Parameter-shift's noise is NOT simulated with Aer's estimator:
the notes record that trap (precision=p returns exact + Gaussian noise with no
dependence on Var(H), a ~27x subsidy). Instead Var(<H>) = sum_g Var(H_g)/S_p is
computed EXACTLY from the statevector for each shifted circuit, which is the fix
that section already prescribes.

QLTO IS GIVEN THE ORACLE R at each budget - the best R on the grid, chosen after
seeing the answer. That is deliberately generous and is stated as such: the
question here is whether the SCALING exists at all, not whether a practical
schedule attains it. If the oracle-R curve plateaus, no schedule can rescue it.
If it does not plateau, then finding a real schedule becomes the next question.
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


def pshift_noisy(ansatz, H, groups, theta, S_p, rng):
    """Parameter-shift with HONEST shot noise: Var(<H>) = sum_g Var(H_g)/S_p,
    computed exactly from the statevector rather than taken from an estimator
    that silently returns exact values plus a fixed-width Gaussian."""
    gmats = [(g.to_matrix(), (g @ g).simplify().to_matrix()) for g in groups]
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            tot = 0.0
            for Hg, Hg2 in gmats:
                m1 = float(np.real(np.conj(v) @ (Hg @ v)))
                m2 = float(np.real(np.conj(v) @ (Hg2 @ v)))
                var = max(m2 - m1 * m1, 0.0)
                tot += m1 + rng.normal(0.0, np.sqrt(var / S_p))
            g[i] += s * tot / 2
    return g


RADII = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9)
BUDGETS = (2 ** 13, 2 ** 15, 2 ** 17, 2 ** 19)
REPEATS = 6

print("=" * 100)
print("IS THE PLATEAU REAL, OR DID v14 FREEZE R?")
print("=" * 100)
print("  Matched TOTAL shots T. QLTO spends T over G*L circuits, parameter-shift")
print("  over 2MG. QLTO is given the ORACLE best R on the grid at each budget.")
print("  v14 held R=0.6 at every budget and reported a plateau at cos ~ 0.977.")
print()

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024)
    G = len(probe.groups)
    blocks = [b['params'] for b in probe.layers if b['params']]
    L = len(blocks)
    C_q, C_p = G * L, 2 * M * G

    theta = np.random.RandomState(17).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    print(f"  Heisenberg N={N}:  M={M}  G={G}  L={L}   QLTO {C_q} circuits,"
          f"  p-shift {C_p} circuits  ({C_p / C_q:.0f}x)")
    print(f"  {'T total':>10}{'S_q':>8}{'S_p':>7}{'R*':>7}{'cos QLTO':>11}"
          f"{'cos PS':>10}{'1-cos QL':>11}{'1-cos PS':>11}")
    print("  " + "-" * 75)

    # ONE instance per repeat, reused across every radius and budget. R is a
    # Parameter in _direct_template and shot_budget is read at run time, so the
    # transpiled template is valid for all of them - building a fresh QLTOv5 per
    # (radius, repeat, budget) would pay 18 cold transpiles each and make this
    # run transpile-bound rather than shot-bound.
    with contextlib.redirect_stdout(io.StringIO()):
        qs = [nisq_v5.QLTOv5(ansatz, H, shot_budget=1024,
                             gradient_mode='direct', sim_seed=900 + rep)
              for rep in range(REPEATS)]

    rows = []
    for T in BUDGETS:
        S_q, S_p = max(1, T // C_q), max(1, T // C_p)
        for q in qs:
            q.shot_budget = int(S_q)

        best = (-2.0, None)
        for R in RADII:
            cs = []
            for q in qs:
                gh = np.zeros(M)
                for act in blocks:
                    gi, _ = q.sense(theta, R, act)
                    gh += gi
                cs.append(cosine(gh, g_ex))
            m = float(np.mean(cs))
            if m > best[0]:
                best = (m, R)
        cq, Rstar = best

        cps = []
        for rep in range(REPEATS):
            rng = np.random.RandomState(4000 + rep)
            cps.append(cosine(pshift_noisy(ansatz, H, probe.groups, theta,
                                           S_p, rng), g_ex))
        cp = float(np.mean(cps))

        rows.append((T, Rstar, cq, cp))
        print(f"  {T:>10}{S_q:>8}{S_p:>7}{Rstar:>7.2f}{cq:>11.4f}{cp:>10.4f}"
              f"{1 - cq:>11.5f}{1 - cp:>11.5f}", flush=True)

    # does the QLTO error keep falling, and at what exponent?
    Ts = np.array([r[0] for r in rows], float)
    eq = np.array([max(1 - r[2], 1e-9) for r in rows])
    ep = np.array([max(1 - r[3], 1e-9) for r in rows])
    aq = np.polyfit(np.log(Ts), np.log(eq), 1)[0]
    ap = np.polyfit(np.log(Ts), np.log(ep), 1)[0]
    print(f"  fitted (1-cos) ~ T^alpha:   QLTO alpha = {aq:+.3f}"
          f"    p-shift alpha = {ap:+.3f}")
    print(f"  R* trend: " + " -> ".join(f"{r[1]:.2f}" for r in rows))
    print()

print("  READING IT. alpha ~ 0 for QLTO means the plateau is REAL and the")
print("  withdrawal in the notes stands. alpha clearly negative, with R* SHRINKING")
print("  as T grows, means v14's plateau was an artefact of freezing R and the")
print("  shot question has to be reopened.")
print()
print("  THE EXPONENTS TO EXPECT, and note the metric SQUARES them. The algebra")
print("  gives R* ~ T^(-1/6) and gradient ERROR ~ T^(-1/3) for QLTO against the")
print("  unbiased T^(-1/2). But 1-cos ~ (1/2)(e_perp/|g|)^2 for small errors, so")
print("  it tracks error SQUARED and the fitted alpha should be:")
print("        QLTO      alpha ~ -2/3 = -0.667")
print("        p-shift   alpha ~ -1.0")
print("  A QLTO alpha near -2/3 confirms the bias-variance trade is live and the")
print("  plateau was the protocol. A QLTO alpha near 0 confirms the floor is real.")
