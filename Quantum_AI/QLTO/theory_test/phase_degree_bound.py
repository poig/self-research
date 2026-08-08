"""What can a degree-d phase oracle concentrate? Approximation theory, not sweeps.

Nine empirical interventions failed to improve QLTO's walk oracle. Nine failed
guesses are weak evidence about a DESIGN SPACE and prove nothing; the question of
whether a better low-degree phase exists has a definite yes or no, and the closed
form (v37b, validated to 0.00241) makes it answerable analytically. So answer it.

THE SETUP. With the drift diagonal in the param basis and the anc=1
post-selection,

    P(x)  ~  |<x|(I - U)|s>|^2  =  4 sin^2( phi(x) / 2 ).

Perfect concentration on a single corner x* needs phi(x*) = pi and phi(x) = 0 for
every other x - that is, phi must approximate pi * 1[x = x*], THE INDICATOR
FUNCTION. So the achievable concentration of this circuit family is a question
about how well a degree-d polynomial on the hypercube can approximate an
indicator, and nothing about mixers or schedules enters.

TWO CONSEQUENCES ARE ALREADY IMPLIED BY KNOWN RESULTS.

  A phase LINEAR in the energy cannot concentrate. If phi = lambda (E - E_bar),
  then P ~ sin^2 of a linear function of E: a smooth, Boltzmann-like reweighting
  whose contrast across the spectrum is bounded by a small constant, not by 2^n.
  That is exactly what these notes measured from the other direction - "ONE
  decoder ties it: a Boltzmann-weighted average over all sampled vertices" - and
  it is why the walk ties a classical Boltzmann decode. The walk IS a Boltzmann
  reweighting, because its phase is linear in E.

  Grover-like concentration needs a THRESHOLD phase. The approximate degree of
  the AND function on n bits is Theta(sqrt n) (Nisan-Szegedy; Paturi), and the
  polynomial method of Beals, Buhrman, Cleve, Mosca and de Wolf turns exactly
  that fact into Grover's Omega(sqrt N) query lower bound. A degree-d phase
  oracle is therefore subject to the same bound.

MEASURED HERE, exactly, with no simulator: for each degree d, optimise a degree-d
Walsh polynomial phi to maximise P(x*) under P ~ sin^2(phi/2), and report the
achievable enhancement 2^n P(x*). The curve of max enhancement against d is the
circuit family's capability, and it is a property of n and d alone.

WHY THIS ANSWERS THE PROBLEM-BASED QUESTION. If the achievable enhancement is
governed by the approximate degree of the TARGET INDICATOR, then the right
per-problem quantity is that approximate degree - not the mixer family and not
the coefficient degree. Problems whose good-set indicator has low approximate
degree are the ones a low-degree drift can concentrate on; problems needing
Theta(sqrt n) are the ones where it provably cannot. That is a criterion with a
yes/no answer per problem class, which is what a design rule has to be.
"""
import itertools
import numpy as np
from scipy.optimize import minimize


def walsh_basis(n, deg):
    """Columns of the Walsh basis up to the given degree, over all 2^n vertices."""
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    cols, keys = [np.ones(len(sig))], [()]
    for d in range(1, deg + 1):
        for S in itertools.combinations(range(n), d):
            cols.append(np.prod(sig[:, S], axis=1))
            keys.append(S)
    return sig, np.stack(cols, axis=1), keys


def best_enhancement(n, deg, i_star=0, restarts=6, seed=0):
    """Max 2^n P(x*) over degree-<=deg phases, with P ~ sin^2(phi/2)."""
    sig, A, keys = walsh_basis(n, deg)
    N = 2 ** n
    rng = np.random.RandomState(seed)

    def neg(c):
        phi = A @ c
        p = np.sin(phi / 2.0) ** 2
        tot = p.sum()
        return -N * p[i_star] / tot if tot > 1e-12 else 0.0

    best = 0.0
    for r in range(restarts):
        c0 = rng.randn(A.shape[1]) * 0.8
        res = minimize(neg, c0, method='BFGS',
                       options={'maxiter': 800, 'gtol': 1e-10})
        best = max(best, -res.fun)
    return best


print("=" * 88)
print("WHAT A DEGREE-d PHASE ORACLE CAN CONCENTRATE")
print("=" * 88)
print("  P(x) ~ sin^2(phi(x)/2). Perfect concentration needs phi to approximate")
print("  pi * 1[x = x*], so this is the approximate degree of an indicator.")
print("  Enhancement is 2^n P(x*); 1.0 is uniform, 2^n is a delta function.")
print()
print(f"  {'n':>4}{'2^n':>7}" + "".join(f"{'d=' + str(d):>10}" for d in range(1, 6)))
print("  " + "-" * (11 + 10 * 5))

for n in (3, 4, 5, 6):
    row = []
    for d in range(1, 6):
        row.append(best_enhancement(n, min(d, n), seed=n) if d <= n else np.nan)
    print(f"  {n:>4}{2 ** n:>7}"
          + "".join(f"{v:>10.2f}" if np.isfinite(v) else f"{'-':>10}"
                    for v in row), flush=True)

print()
print("  (2) A PHASE LINEAR IN THE ENERGY — the shipped drift. Best achievable")
print("      enhancement when phi is CONSTRAINED to phi = lambda (E - E_bar),")
print("      optimised over lambda, on random landscapes.")
print(f"  {'n':>4}{'2^n':>7}{'best enh':>11}{'at lambda':>11}"
      f"{'deg-n free':>12}")
print("  " + "-" * 45)
for n in (3, 4, 5, 6):
    rng = np.random.RandomState(7 + n)
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = rng.randn(2 ** n)
    E = E - E.mean()
    i_star = int(np.argmin(E))
    lams = np.linspace(0.01, 20.0, 4000)
    best, bl = 0.0, 0.0
    for lam in lams:
        p = np.sin(lam * (E - E.mean()) / 2.0) ** 2
        t = p.sum()
        if t < 1e-12:
            continue
        v = 2 ** n * p[i_star] / t
        if v > best:
            best, bl = v, lam
    free = best_enhancement(n, n, i_star=i_star, seed=n)
    print(f"  {n:>4}{2 ** n:>7}{best:>11.2f}{bl:>11.2f}{free:>12.2f}")

print()
print("  The first table is the capability of the circuit family by degree; the")
print("  second is what the SHIPPED drift can reach, since its phase is linear in")
print("  E by construction. A large gap between the two columns of table (2) is")
print("  the price of using a Boltzmann-shaped phase instead of a threshold one,")
print("  and it is a theorem about polynomials rather than a property of any")
print("  mixer, schedule or shot budget.")
