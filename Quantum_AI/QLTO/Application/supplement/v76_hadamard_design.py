"""Can an orthogonal design replace the random hypercube, and what does it cost?

THE TWO WALLS THIS ATTACKS, both established here rather than assumed:

  REGISTER WIDTH. The advantage over parameter-shift is 2n in block width n, and
  v71 found accuracy FLAT within noise from n=1 to n=18, so widening is free in
  accuracy. What stops it is one control qubit per parameter (the requirement
  grows as 0.65*2N(r+1) + N, linear in M).

  THE CROSS TERM. v75 derived Var(V5) ~ (1/T)[C + v/R^2] with
  C = sum_{j!=i} g_j^2, and showed C is what pulls the QLTO/SPSA ratio toward 2
  as the parameter count grows. It survives because random sigma leaves residual
  cross-correlation of order 1/sqrt(S).

THE PROPOSAL. Replace uniform sigma over {+-1}^n by rows of a Sylvester-Hadamard
matrix, H[d,j] = (-1)^popcount(d AND j), indexing D rows on log2(D) qubits rather
than n parameters on n qubits. Two consequences, pulling in OPPOSITE directions,
which is the whole point of measuring rather than asserting:

  GOOD. Columns are orthogonal EXACTLY, sum_d H[d,i] H[d,j] = 0 for i != j, so
  the cross term cancels by construction instead of to O(1/sqrt(S)).

  BAD. A saturated design is resolution III. The product of columns i and j is
  column (i XOR j), so a degree-2 Walsh term q_ij sigma_i sigma_j ALIASES onto the
  main effect of column (i XOR j) and biases that estimate. The random design has
  no such bias: E[sigma_i * sigma_j sigma_k] = 0 for distinct indices, so it is
  unbiased at every order.

The register saving and the aliasing come from the SAME choice. Using only
generator columns (single bit set) removes aliasing but then n = log2(D) and D =
2^n rows, which IS the present hypercube and saves nothing. So the honest question
is not whether an orthogonal design is nicer, it is:

    does the variance removed by exact orthogonality exceed the bias
    introduced by resolution-III aliasing, at realistic degree-2 weight?

WHAT IS MEASURED. Both designs on the same landscape, same total shots, same
readout noise, over many repeats: per-component bias, variance and MSE, plus the
cosine to the true gradient. Swept over the degree-2 weight, because that is the
quantity the tradeoff turns on. Register width is reported alongside, since it is
the resource actually being bought.

WHAT WOULD CONFIRM: Hadamard MSE at or below random across the plausible degree-2
range, while needing log2(n) qubits instead of n. That would lift the block-width
cap and make the 2n advantage grow rather than sit pinned.
WHAT WOULD KILL IT: aliasing bias dominating at realistic degree-2 weight. Then
the register saving is real but bought with an error the random design does not
have, and resolution IV (twice the rows) is the next thing to price.

This is an estimator-level test on a synthetic landscape where g is known exactly.
It says nothing yet about circuit cost; the parity CNOTs still have to be priced
against the n control qubits they replace.
"""
import numpy as np

rng = np.random.default_rng(23)


def sylvester(D):
    """Sylvester-Hadamard of order D, a power of two. H[d,j] = (-1)^<d,j>."""
    H = np.ones((1, 1), dtype=float)
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    return H


def landscape(sig, g, q, R, E0):
    """E = E0 + R * sum_j g_j s_j + R^2 * sum_{j<k} q_jk s_j s_k."""
    lin = sig @ g
    quad = np.einsum('sj,sk,jk->s', sig, sig, q)
    return E0 + R * lin + (R ** 2) * quad


def marginal_estimate(sig, E, R):
    """Self-normalised per-parameter marginal, exactly as V5 forms it."""
    n = sig.shape[1]
    out = np.zeros(n)
    for i in range(n):
        p = sig[:, i] > 0
        if p.all() or (~p).all():
            continue
        out[i] = (E[p].mean() - E[~p].mean()) / (2 * R)
    return out


def run_random(n, g, q, R, E0, T, v):
    sig = rng.choice([-1.0, 1.0], size=(T, n))
    E = landscape(sig, g, q, R, E0) + rng.normal(0.0, np.sqrt(v), T)
    return marginal_estimate(sig, E, R)


def run_hadamard(n, g, q, R, E0, T, v, H, cols, balanced=True):
    """Orthogonality is EXACT only when every design row gets the same number of
    shots. Sampling rows at random leaves an O(1/sqrt(T)) imbalance, which puts
    the cross term straight back and reproduces the random design's variance. That
    is a real allocation requirement, not a detail: `balanced` toggles it so the
    difference is visible rather than assumed."""
    D = H.shape[0]
    if balanced:
        reps = T // D
        rows = np.repeat(np.arange(D), reps)
        if len(rows) < T:                      # spread any remainder, once each
            rows = np.concatenate([rows, rng.permutation(D)[:T - len(rows)]])
    else:
        rows = rng.integers(0, D, size=T)
    sig = H[np.ix_(rows, cols)]
    E = landscape(sig, g, q, R, E0) + rng.normal(0.0, np.sqrt(v), T)
    return marginal_estimate(sig, E, R)


def run_foldover(n, g, q, R, E0, T, v, H, cols):
    """Resolution IV by foldover: the design plus its own sign reversal. Odd-order
    effects become orthogonal to even-order ones, so main effects are clear of ALL
    two-factor interactions and the degree-2 aliasing disappears. Costs twice the
    rows, hence one extra register qubit, and nothing else."""
    base = H[:, cols]
    rowsig = np.vstack([base, -base])
    Df = rowsig.shape[0]
    reps = T // Df
    idx = np.repeat(np.arange(Df), reps)
    if len(idx) < T:
        idx = np.concatenate([idx, rng.permutation(Df)[:T - len(idx)]])
    sig = rowsig[idx]
    E = landscape(sig, g, q, R, E0) + rng.normal(0.0, np.sqrt(v), T)
    return marginal_estimate(sig, E, R)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


N_PARAM = 15
R, E0, T, v, REPS = 0.45, 1.7, 8192, 1.0, 300

D = 1
while D < N_PARAM + 1:
    D *= 2
H = sylvester(D)
cols = list(range(1, N_PARAM + 1))             # skip the all-ones column 0
qubits_rand = N_PARAM
qubits_had = int(np.log2(D))

print("=" * 100)
print("ORTHOGONAL DESIGN vs RANDOM HYPERCUBE")
print("=" * 100)
print(f"  n = {N_PARAM} parameters, T = {T} shots, R = {R}, readout var = {v},")
print(f"  {REPS} repeats. Hadamard order D = {D}.")
print()
print(f"  REGISTER WIDTH:   random {qubits_rand} qubits   ->   Hadamard"
      f" {qubits_had} qubits   ({qubits_rand - qubits_had} saved)")
print()
print("  Aliasing structure: column i times column j is column (i XOR j), so a")
print("  degree-2 term on (i,j) lands on the main effect of (i XOR j) whenever")
print("  that column is in use. Random sigma has no such term at any order.")
print()
print("  Shot allocation across design rows is BALANCED (exactly T/D per row).")
print("  Random allocation leaves an O(1/sqrt(T)) imbalance that restores the")
print("  cross term and erases the whole effect; the last block below shows that.")
print()
print(f"  {'deg-2 wt':>9}{'|bias| rnd':>12}{'|bias| had':>12}"
      f"{'var rnd':>11}{'var had':>11}{'MSE rnd':>11}{'MSE had':>11}"
      f"{'cos rnd':>10}{'cos had':>10}{'winner':>9}")
print("  " + "-" * 106)

for qw in (0.0, 0.1, 0.25, 0.5, 1.0):
    g = rng.normal(0.0, 1.0, N_PARAM)
    q = np.triu(rng.normal(0.0, qw, (N_PARAM, N_PARAM)), 1)

    er = np.array([run_random(N_PARAM, g, q, R, E0, T, v) for _ in range(REPS)])
    eh = np.array([run_hadamard(N_PARAM, g, q, R, E0, T, v, H, cols)
                   for _ in range(REPS)])

    br = np.abs(er.mean(axis=0) - g).mean()
    bh = np.abs(eh.mean(axis=0) - g).mean()
    vr = er.var(axis=0, ddof=1).mean()
    vh = eh.var(axis=0, ddof=1).mean()
    mr = ((er - g) ** 2).mean()
    mh = ((eh - g) ** 2).mean()
    cr = float(np.mean([cosine(e, g) for e in er]))
    ch = float(np.mean([cosine(e, g) for e in eh]))
    win = 'Hadamard' if mh < mr else 'random'
    if abs(mh - mr) / max(mh, mr) < 0.05:
        win = 'tie'
    print(f"  {qw:>9.2f}{br:>12.5f}{bh:>12.5f}{vr:>11.5f}{vh:>11.5f}"
          f"{mr:>11.5f}{mh:>11.5f}{cr:>10.4f}{ch:>10.4f}{win:>9}")

print()
print("  ALLOCATION CHECK, at zero degree-2 weight so nothing aliases.")
g0 = rng.normal(0.0, 1.0, N_PARAM)
q0 = np.zeros((N_PARAM, N_PARAM))
eb = np.array([run_hadamard(N_PARAM, g0, q0, R, E0, T, v, H, cols, True)
               for _ in range(REPS)])
eu = np.array([run_hadamard(N_PARAM, g0, q0, R, E0, T, v, H, cols, False)
               for _ in range(REPS)])
er0 = np.array([run_random(N_PARAM, g0, q0, R, E0, T, v) for _ in range(REPS)])
C = float(np.sum(g0 ** 2) - np.mean(g0 ** 2))
print(f"    balanced rows   var = {eb.var(axis=0, ddof=1).mean():.6f}")
print(f"    random rows     var = {eu.var(axis=0, ddof=1).mean():.6f}")
print(f"    random sigma    var = {er0.var(axis=0, ddof=1).mean():.6f}")
print(f"    predicted with cross term    (C + v/R^2)/T = "
      f"{(C + v / R ** 2) / T:.6f}")
print(f"    predicted without cross term (v/R^2)/T     = "
      f"{(v / R ** 2) / T:.6f}")
print("    Balanced allocation should shed C = sum_{j!=i} g_j^2 and keep only the")
print("    readout term. If it does not separate from the other two, the")
print("    orthogonality argument is wrong rather than merely mis-implemented.")
print()
print("  The first row has NO degree-2 content, so it isolates the orthogonality")
print("  gain with nothing aliasing: any Hadamard advantage there is the cross-term")
print("  cancellation alone. Later rows add the aliasing the saving pays for.")

print()
print("=" * 100)
print("RESOLUTION IV BY FOLDOVER:  does one extra qubit remove the aliasing?")
print("=" * 100)
print(f"  Design plus its sign reversal: {2 * D} rows, {int(np.log2(2 * D))} register")
print(f"  qubits against {N_PARAM} for the random hypercube. Main effects become")
print("  clear of every two-factor interaction, so the bias should vanish while the")
print("  variance gain is kept.")
print()
print(f"  {'deg-2 wt':>9}{'|bias| rnd':>12}{'|bias| IV':>12}"
      f"{'var rnd':>11}{'var IV':>11}{'MSE rnd':>11}{'MSE IV':>11}"
      f"{'cos rnd':>10}{'cos IV':>10}{'winner':>9}")
print("  " + "-" * 106)
for qw in (0.0, 0.1, 0.25, 0.5, 1.0):
    g = rng.normal(0.0, 1.0, N_PARAM)
    q = np.triu(rng.normal(0.0, qw, (N_PARAM, N_PARAM)), 1)
    er = np.array([run_random(N_PARAM, g, q, R, E0, T, v) for _ in range(REPS)])
    ef = np.array([run_foldover(N_PARAM, g, q, R, E0, T, v, H, cols)
                   for _ in range(REPS)])
    br = np.abs(er.mean(axis=0) - g).mean()
    bf = np.abs(ef.mean(axis=0) - g).mean()
    mr = ((er - g) ** 2).mean()
    mf = ((ef - g) ** 2).mean()
    win = 'foldover' if mf < mr else 'random'
    if abs(mf - mr) / max(mf, mr) < 0.05:
        win = 'tie'
    print(f"  {qw:>9.2f}{br:>12.5f}{bf:>12.5f}"
          f"{er.var(axis=0, ddof=1).mean():>11.5f}"
          f"{ef.var(axis=0, ddof=1).mean():>11.5f}{mr:>11.5f}{mf:>11.5f}"
          f"{float(np.mean([cosine(e, g) for e in er])):>10.4f}"
          f"{float(np.mean([cosine(e, g) for e in ef])):>10.4f}{win:>9}")
print()
print("  If the bias column stays flat while the variance gain holds, the register")
print("  drops from n to log2(n)+1 with no accuracy cost, the block-width cap lifts,")
print("  and the 2n advantage grows with n rather than sitting at the qubit limit.")
print()
print("  If Hadamard holds up to realistic degree-2 weight, the block-width cap")
print("  lifts and the 2n advantage grows with n instead of sitting pinned at the")
print("  register limit. If it does not, the register saving is real but bought")
print("  with a bias the random design never has, and resolution IV at twice the")
print("  rows is the next thing to price.")
