"""Does V6's amortisation survive the radius it has to pay for it?

v82 measured tr(Cov) ~ M^1.006 for V6 against M^2.000 for parameter-shift at fixed
gradient norm, and reported that as a full power of M in amortisation. IT HELD R
FIXED AT 0.45 ACROSS THE WHOLE SWEEP. V6 cannot do that.

A block of n parameters displaces the state by about sqrt(n)*R, so the radius must
shrink as R ~ R_N sqrt(N/n) or the linearisation degrades - which is exactly why
V6 rescales internally, and v79 measured the cost of not doing so as cos 0.886
against 0.975. But the variance carries 1/R^2, so

    R^2 ~ N/M   =>   v/R^2 ~ vM/N

and the radius penalty alone contributes a FACTOR OF M to the variance. v82
therefore measured the benefit of the wide block without the price of it.

WHAT THIS DECIDES. From v75's law, Var_V6 ~ (G/T)[C + v/R^2]:

    R fixed        v/R^2 constant, C constant at fixed norm  ->  tr(Cov) ~ M
    R ~ 1/sqrt(M)  v/R^2 grows as M                          ->  tr(Cov) ~ M^2

and parameter-shift is M^2 either way. So if the second row is what V6 actually
faces, the amortisation reported in v82 does not exist and the circuit advantage is
paid back in full through the radius.

THREE ARMS, identical except for the radius rule:
    R fixed             what v82 measured; the optimistic bound
    R ~ sqrt(N/M)       what V6's _radius() actually does
    R optimised per M   the honest best case: R chosen to minimise total error at
                        each M, balancing bias c*n*R^2 against variance a/(R^2 S)
                        rather than following a fixed rule

The third arm matters because sqrt(N/M) is a HEURISTIC fitted at two sizes (v79
measured 0.18 best at N=6/M=36 against 0.184 predicted, and 0.10 at N=4/M=24
against 0.184 predicted). If the optimal rule is weaker than 1/sqrt(M) the penalty
is smaller than the middle arm suggests, and if it is stronger the penalty is
worse. Fitting the exponent of R*(M) directly answers that.

WHAT WOULD SALVAGE v82's CLAIM: the optimised arm scaling clearly below M^2.
WHAT WOULD KILL IT: all three of bias-limited, variance-limited and optimised
sitting at M^2, which would mean V6's wide block buys circuits and nothing else.
"""
import numpy as np

rng = np.random.default_rng(53)


def design_cols(n, k=1):
    m_row = max(1, int(np.ceil(np.log2(n + 1))))
    gray = lambda t: t ^ (t >> 1)
    if k > 1:
        per = -(-n // k)
        m_lo = max(1, int(np.ceil(np.log2(per + 1))))
        m_hi = int(np.ceil(np.log2(k)))
        if m_lo + m_hi <= m_row:
            cols = [((p % k) << m_lo) ^ gray(p // k + 1) for p in range(n)]
            if len(set(cols)) == n and 0 not in cols:
                return m_row, cols
    return m_row, [gray(j + 1) for j in range(n)]


def design_sigma(M, T, k=3):
    m_row, cols = design_cols(M, k)
    d = rng.integers(0, 1 << m_row, size=T)
    f = rng.integers(0, 2, size=T)
    c = np.asarray(cols)
    pc = np.zeros((T, M), dtype=np.int64)
    x = d[:, None] & c[None, :]
    while np.any(x):
        pc += x & 1
        x >>= 1
    return np.where(((pc + f[:, None]) & 1) == 1, -1.0, 1.0)


def cubic_terms(M, n_terms=None, seed=None):
    """A GENERIC degree-3 Walsh polynomial: random triples, random coefficients.

    The obvious choice, (sum_j s_j)^3, is NOT generic and must not be used here.
    It is the mode in which every parameter moves together, and the Hadamard
    design contains the all-ones row where that term reaches M^3. Measuring with
    it reports an interaction between the landscape and the design rather than
    finite-radius bias, and it inverted the exponent in the first version of this
    script.
    """
    r = np.random.default_rng(seed)
    n_terms = n_terms or 4 * M
    tri = np.sort(r.integers(0, M, size=(n_terms, 3)), axis=1)
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])
    tri = tri[keep]
    return tri, r.normal(0.0, 1.0, len(tri))


def mse_v6(g, T, R, v, curv, tri, tc):
    """Total MSE of the V6 marginal estimator, bias included.

    E = R sum_j g_j s_j + curv R^3 sum_{(j,k,l)} c s_j s_k s_l + noise(v).
    The cubic scale carries R^3 so the induced bias in the degree-1 coefficient is
    the genuine O(R^2) finite-radius term rather than an assumed constant.
    """
    M = len(g)
    S = design_sigma(M, T)
    lin = R * (S @ g)
    cube = curv * (R ** 3) * (
        (S[:, tri[:, 0]] * S[:, tri[:, 1]] * S[:, tri[:, 2]]) @ tc)
    E = lin + cube + rng.normal(0.0, np.sqrt(v), T)
    Z = S * E[:, None]
    est = Z.mean(axis=0) / R
    return float(np.mean((est - g) ** 2))


def mse_ps(M, T, G, v, s=np.pi / 2):
    per = max(1.0, T / (2.0 * M * G))
    return float((2.0 * v / per) / (4.0 * s * s))


G, T, V, CURV, N_SYS, R_N = 3, 200000, 1.0, 0.35, 6, 0.45
MS = (8, 16, 32, 64, 128)
GRID = np.array([0.6, 0.45, 0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035])

print("=" * 100)
print("DOES V6's AMORTISATION SURVIVE THE RADIUS IT PAYS FOR IT?")
print("=" * 100)
print(f"  Gradient norm held at 1 for every M, so any M-dependence is the RADIUS")
print(f"  and not the landscape. G={G}, T={T}, readout var={V}, cubic={CURV}.")
print()
print(f"  {'M':>5}{'R fixed':>12}{'R~sqrt(N/M)':>14}{'R optimised':>13}"
      f"{'R* used':>10}{'PS':>12}")
print("  " + "-" * 66)

rows = {'fix': [], 'rule': [], 'opt': [], 'ps': [], 'rstar': []}
for M in MS:
    g = rng.normal(0.0, 1.0, M)
    g = g / np.linalg.norm(g)
    tri, tc = cubic_terms(M, seed=1000 + M)
    m_fix = mse_v6(g, T, R_N, V, CURV, tri, tc)
    m_rule = mse_v6(g, T, R_N * np.sqrt(N_SYS / M), V, CURV, tri, tc)
    cand = [(mse_v6(g, T, r, V, CURV, tri, tc), r) for r in GRID]
    m_opt, r_star = min(cand)
    m_ps = mse_ps(M, T, G, V)
    for k, val in (('fix', m_fix), ('rule', m_rule), ('opt', m_opt),
                   ('ps', m_ps), ('rstar', r_star)):
        rows[k].append(val)
    print(f"  {M:>5}{m_fix:>12.3e}{m_rule:>14.3e}{m_opt:>13.3e}"
          f"{r_star:>10.3f}{m_ps:>12.3e}")

lm = np.log(np.array(MS, dtype=float))
print()
print("  fitted per-component MSE ~ M^alpha:")
for k, tag in (('fix', 'R fixed        '), ('rule', 'R ~ sqrt(N/M)  '),
               ('opt', 'R optimised    '), ('ps', 'parameter-shift')):
    a = float(np.polyfit(lm, np.log(np.array(rows[k])), 1)[0])
    print(f"      {tag}  alpha = {a:+.3f}")
ar = float(np.polyfit(lm, np.log(np.array(rows['rstar'])), 1)[0])
print(f"      optimal radius R*(M) ~ M^{ar:+.3f}"
      f"   (the sqrt(N/M) heuristic assumes -0.500)")

print()
print("=" * 100)
print("READING IT")
print("=" * 100)
print("  Per-component MSE is reported, so parameter-shift's alpha is +1 here")
print("  rather than the +2 v82 reported for tr(Cov); the comparison between arms")
print("  is what matters, not the absolute exponent.")
print()
print("  If 'R fixed' sits near 0 and the other two near +1, then v82 measured an")
print("  amortisation that V6 cannot actually collect, and the radius penalty")
print("  returns the circuit saving in full.")
print()
print("  If 'R optimised' sits clearly below 'R ~ sqrt(N/M)', the heuristic is")
print("  over-shrinking and V6 should solve for R rather than follow a rule - which")
print("  would be a real and cheap improvement to nisq_v6._radius.")
