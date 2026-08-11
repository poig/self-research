"""Are the two symmetry conditions two mechanisms, or one criterion seen twice?

The manuscript states two sufficient conditions for the first-order reachable
interval to be symmetric about zero:

    (A)  spec(Y)    is symmetric about zero
    (B)  spec(M_11) is symmetric about zero

and proves each separately. Stating two unrelated-looking conditions invites the
obvious question, which a referee will ask: are they NECESSARY, and are they all?

By von Neumann's trace inequality the endpoints of Eq. (interval) are

    W_hi = (theta/2) <a, y_down>,     W_lo = (theta/2) <a, y_up>,

with a = lam_down(M_11), y_down/y_up the spectrum of Y sorted down/up. Adding
them, and using y_up_k = y_down_{n+1-k}:

    W_hi + W_lo = (theta/2) <a, b>,   b_k = y_down_k + y_down_{n+1-k}.

So symmetry is EXACTLY the orthogonality <a, b> = 0, with no hypothesis on
either spectrum. That reframes (A) and (B) as the two degenerate ways an inner
product vanishes:

    (A) kills the second factor outright, b = 0
    (B) makes a antisymmetric under index reversal while b is symmetric under it
        by construction, so the k <-> n+1-k terms cancel pairwise

WHAT IS MEASURED, in three parts:

  1. that W_hi + W_lo = (theta/2)<a,b> is an identity, not an approximation
  2. that (A) and (B) each force <a,b> = 0, i.e. they really are special cases
  3. whether a THIRD route exists: <a,b> = 0 with neither spectrum symmetric

Part 3 is the one that decides whether the manuscript's list is exhaustive. If
such spectra exist, then (A) and (B) are sufficient but not necessary, and
breaking both is necessary but NOT sufficient for an asymmetric interval. That
would be the correct reading of the directional_fraction.py result: the search
there breaks both conditions and still finds |D| no larger than 0.0147, which is
consistent with the symmetric configurations forming a surface that a generic
perturbation does not travel far from.

WHAT WOULD FALSIFY THE UNIFICATION. Any row of part 1 where the identity fails
by more than rounding, or any draw in part 2 where a symmetric spectrum leaves
<a,b> nonzero. If part 3 finds nothing in many draws, the two conditions would
be exhaustive after all and the manuscript's framing would stand as written.

Operators here are drawn directly at the level of spectra and random Hermitian
matrices rather than produced by the protocol. That is deliberate: the claim is
about the endpoint formula, which depends on the two operators only through
their spectra, so restricting to protocol-reachable states would test less.
"""
import numpy as np

rng = np.random.default_rng(7)
THETA = 0.1


def herm(n):
    """Generic Hermitian: no spectral symmetry expected."""
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (X + X.conj().T) / 2


def sym_spectrum(n):
    """Hermitian whose spectrum is symmetric about zero by construction."""
    half = rng.normal(size=n // 2)
    ev = np.concatenate([half, -half, np.zeros(n % 2)])
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return Q @ np.diag(ev) @ Q.conj().T


def a_vec(M):
    return np.sort(np.linalg.eigvalsh(M))[::-1]


def b_vec(Y):
    yd = np.sort(np.linalg.eigvalsh(Y))[::-1]
    return yd + yd[::-1]


def endpoints(M, Y, t=THETA):
    """Eq. (interval): the von Neumann extrema over the isospectral orbit."""
    a, yd = a_vec(M), np.sort(np.linalg.eigvalsh(Y))[::-1]
    return (t / 2) * float(a @ yd), (t / 2) * float(a @ yd[::-1])


def defect(v):
    """How far a spectrum is from being symmetric about zero."""
    return float(np.max(np.abs(np.asarray(v) + np.asarray(v)[::-1])))


print("=" * 96)
print("(1)  IS  W_hi + W_lo = (theta/2)<a,b>  AN IDENTITY?")
print("=" * 96)
print("  a = lam_down(M_11);  b_k = lam_down_k(Y) + lam_down_{n+1-k}(Y).")
print("  Generic M_11 and Y, no symmetry imposed on either.")
print()
print(f"  {'n':>4}{'W_hi + W_lo':>18}{'(theta/2)<a,b>':>18}{'abs difference':>18}")
print("  " + "-" * 56)
worst = 0.0
for n in (2, 3, 4, 5, 6, 8, 12, 16):
    M, Y = herm(n), herm(n)
    hi, lo = endpoints(M, Y)
    pred = (THETA / 2) * float(a_vec(M) @ b_vec(Y))
    d = abs((hi + lo) - pred)
    worst = max(worst, d)
    print(f"  {n:>4}{hi + lo:>18.12f}{pred:>18.12f}{d:>18.2e}")
print()
print(f"  worst deviation over all n: {worst:.2e}")
print("  The endpoint sum is the inner product exactly, so symmetry of the")
print("  interval IS the orthogonality <a,b> = 0 and nothing else.")

print()
print("=" * 96)
print("(2)  ARE (A) AND (B) SPECIAL CASES OF THAT ORTHOGONALITY?")
print("=" * 96)
print(f"  {'case':>32}{'defect spec(Y)':>17}{'defect spec(M11)':>19}"
      f"{'<a,b>':>13}{'W_hi + W_lo':>15}")
print("  " + "-" * 94)
for n in (4, 6, 8, 12):
    M, Y = herm(n), herm(n)
    Ys, Ms = sym_spectrum(n), sym_spectrum(n)
    for tag, MM, YY in (("(A) spec(Y) symmetric", M, Ys),
                        ("(B) spec(M11) symmetric", Ms, Y),
                        ("neither", M, Y)):
        hi, lo = endpoints(MM, YY)
        print(f"  {tag + ', n=' + str(n):>32}"
              f"{defect(np.linalg.eigvalsh(YY)):>17.2e}"
              f"{defect(np.linalg.eigvalsh(MM)):>19.2e}"
              f"{float(a_vec(MM) @ b_vec(YY)):>13.2e}{hi + lo:>15.2e}")
print()
print("  Both conditions drive the inner product to zero while the OTHER operator")
print("  keeps a defect of order one. Neither is doing anything but making the")
print("  inner product vanish, by the two different available routes.")

print()
print("=" * 96)
print("(3)  IS THERE A THIRD ROUTE: <a,b> = 0 WITH NEITHER SPECTRUM SYMMETRIC?")
print("=" * 96)
print("  Fix a generic Y. Draw a descending spectrum for M_11, then solve for its")
print("  smallest entry so that <a,b> = 0, keeping the ordering valid. If such")
print("  spectra exist, the manuscript's two conditions are not exhaustive.")
print()
n = 6
Y = herm(n)
b = b_vec(Y)
found, shown = 0, False
for _ in range(500):
    a = np.sort(rng.normal(size=n))[::-1]
    if abs(b[-1]) < 1e-9:
        continue
    s = -(a[:-1] @ b[:-1]) / b[-1]
    if s > a[-2]:
        continue                       # would violate descending order
    a2 = np.concatenate([a[:-1], [s]])
    if defect(a2) > 1e-6 and defect(np.linalg.eigvalsh(Y)) > 1e-6:
        found += 1
        if not shown:
            shown = True
            print(f"    example:  <a,b> = {float(a2 @ b):.2e}")
            print(f"              defect spec(M11) = {defect(a2):.4f}"
                  f"   (0 would mean condition (B) holds)")
            print(f"              defect spec(Y)   = "
                  f"{defect(np.linalg.eigvalsh(Y)):.4f}"
                  f"   (0 would mean condition (A) holds)")
print()
print(f"  {found} such spectra in 500 draws.")
print()
print("  CONSEQUENCE. Symmetry is one orthogonality condition, and (A) and (B) are")
print("  its two degenerate solutions rather than two independent mechanisms. The")
print("  practical form of this: breaking both conditions is NECESSARY for an")
print("  asymmetric interval but NOT SUFFICIENT, since a protocol can satisfy")
print("  neither and still land on <a,b> = 0. The pure-branch obstruction is")
print("  untouched, being exactly case (B), which holds for every Hermitian Y.")
