"""Does V6 amortise gradient information across M, or hide an M-penalty in Cov?

V6 computes an M-component gradient in G circuits where parameter-shift needs
2MG. That is a CIRCUIT count. It says nothing about whether the information per
shot was amortised, because the circuit saving could be paid back entirely in
estimator variance. This measures the covariance and settles it.

THE PREDICTION, from v75's variance law, so the measurement has something to
falsify rather than merely describe:

    Var_V6(g_i)  ~ (G/T) [ C_i + v/R^2 ],     C_i = sum_{j!=i} g_j^2 ~ |grad E|^2
    Var_PS(g_i)  ~ (M G/T) v/s^2

    ratio = M v/s^2 / ( |grad E|^2 + v/R^2 )

which SPLITS, and the split is the whole result:

    cross term dominant   |grad E|^2 ~ M gbar^2 cancels the M, ratio CONSTANT,
                          V6 amortises nothing and its circuit win is paid back
    readout dominant      ratio grows LINEARLY in M, V6 genuinely amortises

TWO REGIMES MUST BE SEPARATED or the exponent is meaningless. Sweeping M while
holding the per-component magnitude fixed makes |grad E|^2 grow with M; holding
the total norm fixed does not. Those give OPPOSITE answers and the literature
conflates them, so both are run here:

    'per-component fixed'   g_j ~ N(0,1), so |grad E|^2 grows as M.
                            More parameters, each doing work: a wider ansatz.
    'total norm fixed'      g scaled to |grad E| = 1 at every M.
                            The same problem parameterised more finely.

WHAT IS MEASURED. The full covariance of both estimators at MATCHED TOTAL SHOTS,
swept over M with N, G, T and R held fixed, from which:

    tr(Cov)             total variance, the quantity an optimiser feels
    tr(Cov)/M           per-component variance
    lambda_max(Cov)     worst direction, which a step size must respect
    mean |corr_ij|      off-diagonal structure: whether "one shot informs all M"
                        is information reuse or correlated noise wearing its coat

then a fit of tr(Cov) ~ M^alpha for each estimator in each regime.

WHAT WOULD CONFIRM AMORTISATION: alpha_V6 < alpha_PS. Equal exponents would mean
V6 moves variance around rather than reducing it, and the circuit advantage is
then a billing advantage only, not an information one.

This is an ESTIMATOR-level measurement on a synthetic landscape where g is known
exactly, which is what allows M to reach 128. It says nothing about circuit depth
or hardware, and the finite-radius bias is excluded by construction: the landscape
is linear in sigma, so any M-dependence found here is variance, not bias.
"""
import numpy as np

rng = np.random.default_rng(41)


def design_cols(n, k=1):
    """Gray-ordered resolution-IV columns, per-wire slices, as V6 uses."""
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
    """T shots of V6's design: sample a row index and foldover bit, expand."""
    m_row, cols = design_cols(M, k)
    d = rng.integers(0, 1 << m_row, size=T)
    f = rng.integers(0, 2, size=T)
    c = np.asarray(cols)
    # popcount of (d & c_j) per shot per parameter, then parity with f
    pc = np.zeros((T, M), dtype=np.int64)
    x = d[:, None] & c[None, :]
    while np.any(x):
        pc += x & 1
        x >>= 1
    return np.where(((pc + f[:, None]) & 1) == 1, -1.0, 1.0)


def cov_v6(g, T, R, v, k=3):
    """Cov of the self-normalised marginal estimator, computed in closed form
    from the design rather than by Monte Carlo over repeats.

    Under E = E0 + R sum_j g_j sigma_j + noise(v), the estimator is
        g_i = [mean(E|s_i=+1) - mean(E|s_i=-1)] / 2R
    which for a balanced design equals (1/RT) sum_t sigma_i E_t up to O(1/T).
    Its covariance is therefore (1/(R^2 T)) Cov(sigma_i E, sigma_j E).
    """
    M = len(g)
    S = design_sigma(M, T, k)
    E = R * (S @ g) + rng.normal(0.0, np.sqrt(v), T)
    Z = S * E[:, None]                       # per-shot influence, sigma_i * E
    return np.cov(Z, rowvar=False) / (T * R * R)


def cov_ps(g, T, M, G, v, s=np.pi / 2):
    """Cov of symmetric parameter-shift at the same TOTAL budget.

    2*M*G circuits share T shots, so each energy gets T/(2 M G). Components are
    independent by construction: each uses its own pair of circuits, which is
    exactly the property V6 gives up.
    """
    per = max(1.0, T / (2.0 * M * G))
    var = (2.0 * v / per) / (4.0 * s * s)
    return np.diag(np.full(M, var))


def report(tag, C, M):
    tr = float(np.trace(C))
    lam = float(np.linalg.eigvalsh(C).max())
    d = np.sqrt(np.diag(C))
    R_ = C / np.outer(d, d)
    off = float(np.mean(np.abs(R_[~np.eye(M, dtype=bool)])))
    return tr, tr / M, lam, off


G, T, R, V = 3, 200000, 0.45, 1.0
MS = (8, 16, 32, 64, 128)

for regime in ('per-component fixed', 'total norm fixed'):
    print("=" * 100)
    print(f"REGIME: {regime}")
    print("=" * 100)
    if regime == 'per-component fixed':
        print("  g_j ~ N(0,1), so |grad E|^2 grows as M: a WIDER ansatz.")
    else:
        print("  g rescaled to |grad E| = 1 at every M: the SAME problem, finer.")
    print(f"  G={G}, T={T}, R={R}, readout var={V}, both at matched TOTAL shots.")
    print()
    print(f"  {'M':>5}{'tr V6':>11}{'tr PS':>11}{'tr/M V6':>10}{'tr/M PS':>10}"
          f"{'lmax V6':>10}{'|corr| V6':>11}{'PS/V6':>8}")
    print("  " + "-" * 76)
    trs = {'V6': [], 'PS': []}
    for M in MS:
        g = rng.normal(0.0, 1.0, M)
        if regime == 'total norm fixed':
            g = g / np.linalg.norm(g)
        C6 = cov_v6(g, T, R, V)
        CP = cov_ps(g, T, M, G, V)
        t6, p6, l6, o6 = report('V6', C6, M)
        tp, pp, lp, _ = report('PS', CP, M)
        trs['V6'].append(t6)
        trs['PS'].append(tp)
        print(f"  {M:>5}{t6:>11.5f}{tp:>11.5f}{p6:>10.6f}{pp:>10.6f}"
              f"{l6:>10.5f}{o6:>11.4f}{tp / t6:>8.2f}")
    lm = np.log(np.array(MS, dtype=float))
    a6 = float(np.polyfit(lm, np.log(np.array(trs['V6'])), 1)[0])
    ap = float(np.polyfit(lm, np.log(np.array(trs['PS'])), 1)[0])
    print()
    print(f"  fitted tr(Cov) ~ M^alpha :   V6 alpha = {a6:+.3f}"
          f"     parameter-shift alpha = {ap:+.3f}")
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  Parameter-shift splits T over 2MG circuits, so its per-component variance")
print("  rises linearly in M and tr(Cov) rises as M^2. That is the baseline.")
print()
print("  V6's exponent is the result. If it is BELOW parameter-shift's, the")
print("  one-shot-informs-all-M property is genuine information reuse. If the two")
print("  match, V6 has moved variance rather than removed it and the circuit")
print("  advantage is a billing advantage only.")
print()
print("  The two regimes are expected to DISAGREE, and that disagreement is the")
print("  point: with per-component magnitude fixed the cross term |grad E|^2 grows")
print("  as M and eats the amortisation, while at fixed total norm it does not.")
print("  Any claim about V6 scaling therefore has to state which regime it means.")
