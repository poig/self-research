"""v76's variance win needs EXACT row balance. A quantum register cannot give it.

v76 measured a 5x variance reduction from a resolution-IV design, but only under
DETERMINISTIC allocation of exactly T/D shots per design row. A superposed
parameter register does not allocate: it is measured, and the row counts are
multinomial. The column sums then fluctuate as O(sqrt(T)) exactly as they do for
random sigma, so the orthogonality that removed the cross term
C = sum_{j!=i} g_j^2 is not available on hardware.

That threatens the whole v76 conclusion, so it is tested here before anything is
built on it.

WHAT MIGHT STILL SURVIVE, and it is worth separating from what does not:

  THE REGISTER SAVING IS INDEPENDENT OF THE ALLOCATION. Indexing D design rows on
  log2(D) qubits rather than n parameters on n qubits is a property of the
  encoding, not of the shot statistics. If the sampled design merely MATCHES the
  random hypercube on accuracy, the qubit saving is still real and still lifts the
  block-width cap.

  THE DECODER CAN EXPLOIT A KNOWN DESIGN EVEN WHEN THE ALLOCATION IS RANDOM. The
  marginal decoder throws information away: it conditions on sigma_i and averages,
  ignoring that the realised design matrix is known exactly for every shot. Least
  squares on that matrix is the best linear unbiased estimator GIVEN the observed
  allocation, so it should recover part of what imbalance costs. That is a change
  to the classical decoder only, with no circuit consequence at all.

THREE ARMS, all at matched shots and identical readout noise:
  1. random hypercube  + marginal decoder      (what V5 ships)
  2. sampled design    + marginal decoder      (v76's encoding, honest allocation)
  3. sampled design    + least-squares decoder (same circuit, better decoder)
and, as the unreachable reference, the deterministic allocation v76 measured.

WHAT WOULD CONFIRM A REAL IMPROVEMENT: arm 3 at or below arm 1 in MSE while using
log2(n)+1 qubits instead of n. That is a strictly better encoding even without the
orthogonality bonus.
WHAT WOULD KILL IT: arm 2 and arm 3 both landing at arm 1's variance. Then the
design buys only qubits, the v76 headline was an artefact of an allocation that
hardware cannot perform, and it must be said so plainly.
"""
import numpy as np

rng = np.random.default_rng(29)


def sylvester(D):
    H = np.ones((1, 1), dtype=float)
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    return H


def landscape(sig, g, q, R, E0):
    return (E0 + R * (sig @ g)
            + (R ** 2) * np.einsum('sj,sk,jk->s', sig, sig, q))


def marginal_decode(sig, E, R):
    n = sig.shape[1]
    out = np.zeros(n)
    for i in range(n):
        p = sig[:, i] > 0
        if p.all() or (~p).all():
            continue
        out[i] = (E[p].mean() - E[~p].mean()) / (2 * R)
    return out


def ols_decode(sig, E, R):
    """Least squares of E on [1, sigma]. Uses the realised design matrix, which is
    known exactly shot by shot, so it corrects for whatever imbalance the sampling
    happened to produce. Classical post-processing only."""
    X = np.hstack([np.ones((len(E), 1)), sig])
    beta, *_ = np.linalg.lstsq(X, E, rcond=None)
    return beta[1:] / R


def sample_rows(rowsig, T, deterministic):
    D = rowsig.shape[0]
    if deterministic:
        idx = np.repeat(np.arange(D), T // D)
        if len(idx) < T:
            idx = np.concatenate([idx, rng.permutation(D)[:T - len(idx)]])
    else:
        idx = rng.integers(0, D, size=T)      # what measuring a register gives
    return rowsig[idx]


N_PARAM, R, E0, T, v, REPS = 15, 0.45, 1.7, 8192, 1.0, 300
D = 1
while D < N_PARAM + 1:
    D *= 2
H = sylvester(D)
cols = list(range(1, N_PARAM + 1))
base = H[:, cols]
rowsig = np.vstack([base, -base])             # resolution IV by foldover
qubits_design = int(np.ceil(np.log2(rowsig.shape[0])))

print("=" * 104)
print("DOES THE v76 WIN SURVIVE A REGISTER THAT IS MEASURED RATHER THAN ALLOCATED?")
print("=" * 104)
print(f"  n = {N_PARAM}, T = {T}, R = {R}, readout var = {v}, {REPS} repeats.")
print(f"  Design rows {rowsig.shape[0]} on {qubits_design} qubits, against"
      f" {N_PARAM} qubits for the random hypercube.")
print()
print("  rand+OLS is the control that decides ATTRIBUTION: least squares is best")
print("  linear unbiased for any known design, so if it matches samp+OLS then the")
print("  variance win belongs to the DECODER and the design buys only qubits.")
print()
print(f"  {'deg-2':>7}{'rand+marg':>12}{'rand+OLS':>12}{'samp+marg':>12}"
      f"{'samp+OLS':>12}{'det+marg':>12}{'best realisable':>17}")
print("  " + "-" * 86)

for qw in (0.0, 0.25, 1.0):
    g = rng.normal(0.0, 1.0, N_PARAM)
    q = np.triu(rng.normal(0.0, qw, (N_PARAM, N_PARAM)), 1)

    def mse(fn, det=None):
        acc = []
        for _ in range(REPS):
            if det is None:
                s = rng.choice([-1.0, 1.0], size=(T, N_PARAM))
            else:
                s = sample_rows(rowsig, T, det)
            E = landscape(s, g, q, R, E0) + rng.normal(0.0, np.sqrt(v), T)
            acc.append(fn(s, E, R))
        return float(((np.array(acc) - g) ** 2).mean())

    m_rand = mse(marginal_decode, None)
    m_rols = mse(ols_decode, None)
    m_samp = mse(marginal_decode, False)
    m_ols = mse(ols_decode, False)
    m_det = mse(marginal_decode, True)
    cand = {'rand+marg': m_rand, 'rand+OLS': m_rols,
            'samp+marg': m_samp, 'samp+OLS': m_ols}
    best = min(cand, key=cand.get)
    print(f"  {qw:>7.2f}{m_rand:>12.5f}{m_rols:>12.5f}{m_samp:>12.5f}"
          f"{m_ols:>12.5f}{m_det:>12.5f}{best:>17}")

print()
print("  'det+marg' is v76's deterministic allocation and is NOT realisable by")
print("  measuring a superposed register; it is here only as the reference the")
print("  other columns are trying to reach.")
print()
print("  If samp+OLS matches or beats rand+marg, the encoding is strictly better:")
print(f"  same accuracy on {qubits_design} qubits instead of {N_PARAM}, which is what")
print("  lifts the block-width cap. If every sampled column sits at rand+marg, the")
print("  design buys qubits and nothing else, and v76's 5x was an artefact of an")
print("  allocation hardware cannot perform.")
