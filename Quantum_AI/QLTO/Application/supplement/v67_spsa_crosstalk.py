"""QLTO vs SPSA at the estimator level: is the difference Theta(M), or a constant?

RESEARCH_NOTES already rules on the other methods. Parameter-shift, AdamW and QNG
all need 2MG circuits per gradient against QLTO's G*M/N, a Theta(M) circuit win
(v65: the ratio is exactly 2N). But the same passage concedes "NEITHER beats SPSA
by more than a constant - SPSA is already Th(1) in circuits, and that is the limit
of the cost claim." That concession is about CIRCUITS. It leaves the axis that
actually decides an optimiser - how much gradient information arrives per energy
evaluation - unmeasured against the one competitor that matters.

THE STRUCTURAL CLAIM, stated before measuring. Both estimators probe the
landscape with random +-1 vectors. They differ in what they read back:

  SPSA   picks ONE random sigma, evaluates E(theta +- c sigma), and assigns the
         same scalar to every coordinate:
             g_i = (grad E . sigma) sigma_i = d_iE + sum_{j!=i} d_jE sigma_j sigma_i
         The cross term has mean zero and variance |grad E|^2 - (d_iE)^2. Since
         |grad E|^2 grows with M while (d_iE)^2 does not, SPSA's per-component
         noise GROWS WITH M. That is intrinsic to the estimator, not to shots.

  QLTO   reads the degree-1 Walsh coefficients of a superposition over ALL sigma:
             Ehat({i}) = E_sigma[ E(theta + R sigma) sigma_i ]
         T4 measured the cross-coordinate term at b = 0 STRUCTURALLY for this
         path. Zero cross-talk. So QLTO should be SPSA with the M-growing noise
         term removed - a Theta(M) advantage in evaluations-to-fixed-accuracy at
         the same Theta(1) circuits.

WHY THIS RUN IS CHEAP AND WHY THAT MATTERS. The claim is about the ESTIMATORS,
not about shot noise, so it needs no sampled circuits at all - only exact
statevector energies. Both methods are charged the SAME currency: one energy
evaluation is one unit, whoever spends it. SPSA spends 2 per sigma; QLTO's
marginal spends 2 per sigma as well (E at theta+R sigma and the paired
-R sigma), so K sigma-samples cost 2K for both and the comparison is like for
like. What differs is only how many coordinates each extracts from the same
spend.

CONTROL, so this is not a rigged comparison: the exact gradient is computed by
parameter-shift on the statevector, and BOTH estimators are scored by
cos(g_hat, grad E) against it - the same common target that the CORRECTION in
"IS IT ACTUALLY CHEAPER?" insisted on after the first cost claim was withdrawn
for normalising each method to its own target.

WHAT WOULD FALSIFY IT. If the two curves converge at the same rate, or if QLTO's
curve is merely shifted by a constant, the notes' "constant factor" concession
stands as written and the Theta(M) reading is wrong. The M-dependence is the
whole claim, so it is measured at three sizes.

=============================================================================
RESULT, AND THIS SCRIPT'S PREMISE IS WRONG. The measured gap is EXACTLY 0.0000
at every sample count, at N=4 (M=24) and N=6 (M=36) alike. Not "small" - zero to
every printed digit, with the sign of the rounding error flipping. That is the
signature of an identity, not of a null effect, and it is: with the radii matched
(c = R), the two lines below are the same expression.

    g_s += ((ep - em) / (2C)) * s          -> ((E+ - E-)/2R) sigma
    g_q += (ep * s - em * s) / 2 ; /= R    -> ((E+ - E-)/2R) sigma

THE DEGREE-1 WALSH MARGINAL UNDER ANTITHETIC SAMPLING IS THE SPSA ESTIMATOR.
Same sigma design, same O(R^2) displacement bias, same cross-talk term. The
comparison this script set out to make does not exist classically, and the "QLTO
extracts n coordinates per evaluation, SPSA extracts 1" framing in the docstring
above is wrong - SPSA also writes to all M coordinates, just with a rank-1
outer product that carries every other coordinate's gradient as noise. So does
the Walsh marginal, identically, when the energies are exact.

WHERE THE REAL DIFFERENCE LIVES, and this run cannot see it BY CONSTRUCTION,
because it was made cheap by removing shot noise - which is the only place the
two differ. T4: SPSA's per-sample variance is |grad E|^2 - (d_iE)^2, which grows
with M and which no shot budget removes, because it is the estimator's own
structure. QLTO's quantum readout returns a BOUNDED +-1 ancilla bit per shot
whose variance cannot exceed 1 however much the energy varies across the
hypercube, and T4 measured that cross-coordinate coefficient at b/a = -0.004.

Kept as-run rather than repaired. The zero column is the cleanest possible
statement that the sampling designs coincide, which is worth more than the
comparison originally wanted, and it localises the whole question to the readout.
A replacement would have to run at MATCHED SHOTS, not matched evaluations.
=============================================================================
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def make_energy(ansatz, H):
    Hm = H.to_matrix()

    def E(theta):
        v = Statevector(ansatz.assign_parameters(theta)).data
        return float(np.real(np.conj(v) @ (Hm @ v)))
    return E


def exact_grad(E, theta):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            g[i] += s * E(t) / 2
    return g


def cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


R = 0.6          # QLTO search radius, the shipped default
C = 0.6          # SPSA displacement, matched so neither gets a bias advantage
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
REPEATS = 12

print("=" * 100)
print("QLTO MARGINAL vs SPSA — information per energy evaluation")
print("=" * 100)
print(f"  Exact statevector energies, no shot noise. R = C = {R}, so both carry")
print(f"  the same O(R^2) displacement bias and differ ONLY in the readout.")
print(f"  K sigma-samples = 2K energy evaluations for BOTH methods.")
print(f"  Scored as cos(g_hat, grad E) against the parameter-shift gradient,")
print(f"  {REPEATS} repeats. QLTO's advantage should GROW with M if it is the")
print(f"  cross-talk term; stay flat if it is a constant factor.")
print()

summary = []
for N in (4, 6, 8):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    E = make_energy(ansatz, H)
    theta = np.random.RandomState(5).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(E, theta)

    print(f"  Heisenberg N={N},  M={M},  |grad E| = {np.linalg.norm(g_ex):.4f},"
          f"  max|d_iE| = {np.max(np.abs(g_ex)):.4f}")
    print(f"  {'K':>6}{'evals':>8}{'cos SPSA':>12}{'cos QLTO':>12}{'gap':>9}")
    print("  " + "-" * 47)

    rows = []
    for K in KS:
        cs, cq = [], []
        for rep in range(REPEATS):
            rng = np.random.RandomState(1000 * rep + K)
            sig = rng.choice([-1.0, 1.0], size=(K, M))
            g_s = np.zeros(M)
            g_q = np.zeros(M)
            for k in range(K):
                s = sig[k]
                ep = E(theta + C * s)
                em = E(theta - C * s)
                g_s += ((ep - em) / (2.0 * C)) * s          # SPSA
                g_q += (ep * s - em * s) / 2.0              # Walsh degree-1
            g_s /= K
            g_q /= (K * R)
            cs.append(cos(g_s, g_ex))
            cq.append(cos(g_q, g_ex))
        a, b = float(np.mean(cs)), float(np.mean(cq))
        rows.append((K, a, b))
        print(f"  {K:>6}{2 * K:>8}{a:>12.4f}{b:>12.4f}{b - a:>9.4f}", flush=True)

    # evaluations each needs to first reach cos >= 0.9
    def first(rows, idx, thr=0.9):
        for K, a, b in rows:
            if (a if idx == 0 else b) >= thr:
                return 2 * K
        return None
    es, eq = first(rows, 0), first(rows, 1)
    summary.append((N, M, es, eq))
    print(f"  evals to reach cos >= 0.90:  SPSA {es}   QLTO {eq}")
    print()

print("=" * 100)
print(f"  {'N':>4}{'M':>5}{'SPSA evals':>13}{'QLTO evals':>13}{'ratio':>9}")
print("  " + "-" * 44)
for N, M, es, eq in summary:
    r = (es / eq) if (es and eq) else float('nan')
    print(f"  {N:>4}{M:>5}{str(es):>13}{str(eq):>13}{r:>9.1f}")
print()
print("  A ratio that GROWS with M is the cross-talk term, and it would sharpen")
print("  the notes' 'neither beats SPSA by more than a constant' - true of")
print("  CIRCUITS, false of evaluations. A flat ratio means the concession stands")
print("  exactly as written and SPSA remains the real competitor.")
print()
print("  NOTE what this does NOT measure: both estimators here are given exact")
print("  energies, so the O(R^2) displacement bias is present but shot noise is")
print("  not. The bias floor that made parameter-shift overtake QLTO at ~8k shots")
print("  (v14) applies to BOTH arms equally here and is why neither cosine will")
print("  reach 1.0.")
