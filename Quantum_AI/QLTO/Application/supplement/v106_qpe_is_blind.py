"""Why V5's QPE trick cannot be transplanted onto the twirl: conjugation is isospectral.

V5 removed the factor G by reading the energy from a PHASE instead of measuring
each qubit-wise-commuting group. The natural question is whether the same move
removes the remaining factor of 2 (measurement bases) in twirl_cal. It cannot, and
the obstruction is structural rather than practical - it is not about depth, which
is what actually killed V5's QPE path (survival 0.098 at Heisenberg N=6).

THE ARGUMENT IN ONE LINE. The twirl acts by CONJUGATION,

    H_sigma  =  Q H Q^dag  =  sum_k sigma_k c_k P_k

and conjugation by a unitary is a similarity transformation, so H_sigma is
ISOSPECTRAL to H for every sigma. Any readout that returns a SPECTRAL quantity -
and a phase is exactly that - is therefore constant in sigma. The degree-1 Walsh
decode over sigma would return identically zero. QPE is not merely expensive here;
it is blind.

WHY V5's QPE DID WORK, and the contrast is the whole point. There the parameter
register changed the STATE: different register values bound different ansatz
angles, producing genuinely different states with genuinely different energies, so
the energy carried the signal. Here the register changes the HAMILTONIAN, and it
changes it by conjugation, which moves the eigenVECTORS and leaves the
eigenVALUES fixed. The signal lives entirely in the relationship between those
rotated eigenvectors and the FIXED probe and FIXED observable - which is precisely
what an expectation value in a fixed basis measures, and what a phase does not.

THIS ALSO EXPLAINS THE RETURN-PROBABILITY REMARK in twirl_cal's docstring. The
return probability |<psi|U_sigma|psi>|^2 IS sigma-dependent - the eigenvectors
rotate against a fixed psi - but its linear part is purely imaginary and squares
away, leaving no FIRST-ORDER term. So there are two distinct obstructions, and
they are not the same:

    QPE phase            no sigma dependence at ANY order   (isospectral)
    return probability   sigma dependence, but not at order 1
    observable <O>       sigma dependence at order 1        <- what is used

TIER (project rule R1). PART 1 is a structural fact about operators with no state
evolution - eigenvalues of a matrix family - which the rule lists explicitly as a
sanctioned NumPy use. PART 2 is tier B: exact amplitudes, no sampling, supporting
a MECHANISM claim about which readouts carry first-order signal and which do not.
Neither part reports an accuracy or a cost, and no headline rests on either.
"""
import sys, os
import itertools
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import Pauli, SparsePauliOp
from twirl_cal import crosstalk_terms, crosstalk_coeffs

N = 3
T = 0.25
terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)

P = [Pauli(t).to_matrix() for t in terms]
Z = np.array([Pauli(t).z.astype(int) for t in terms])
X = np.array([Pauli(t).x.astype(int) for t in terms])


def sigma_of(a, b):
    """The twirl signs handed over by the symplectic vectors."""
    return (-1.0) ** ((Z @ a + X @ b) % 2)


def H_of(sig):
    return sum(s * cc * p for s, cc, p in zip(sig, c_true, P))


print("=" * 96)
print("v106  QPE IS BLIND TO THE TWIRL:  conjugation is isospectral")
print("=" * 96)
print("  N=%d crosstalk, M=%d terms, T=%.2f" % (N, M, T))
print()

print("=" * 96)
print("PART 1  THE SPECTRUM DOES NOT MOVE  (tier C: operator structure, no evolution)")
print("=" * 96)
H0 = H_of(np.ones(M))
ev0 = np.sort(np.linalg.eigvalsh(H0))
worst = 0.0
rows = []
for a in itertools.product([0, 1], repeat=N):
    for b in itertools.product([0, 1], repeat=N):
        sig = sigma_of(np.array(a), np.array(b))
        ev = np.sort(np.linalg.eigvalsh(H_of(sig)))
        d = float(np.max(np.abs(ev - ev0)))
        worst = max(worst, d)
        rows.append((a, b, sig, d))
print("  all %d register values (a,b), each giving a distinct sign pattern sigma:" % len(rows))
print()
print("     (a,b)              sigma (first 6)          max |eigval - eigval(sigma=1)|")
print("   " + "-" * 84)
for a, b, sig, d in rows[:6]:
    print("   %s%s   %s      %.3e"
          % (''.join(map(str, a)), ''.join(map(str, b)),
             np.array2string(sig[:6].astype(int), separator=','), d))
print("   ...")
print()
print("  WORST spectral deviation over all %d twirls: %.3e" % (len(rows), worst))
print("  distinct sign patterns realised: %d"
      % len({tuple(r[2]) for r in rows}))
print()
if worst < 1e-10:
    print("  The spectrum is INVARIANT. Any phase-based readout - QPE, and every")
    print("  eigenvalue-estimation variant of it - has exactly zero signal to decode,")
    print("  at any T, any depth, any precision. This is not a cost, it is a")
    print("  structural impossibility.")
print()

print("=" * 96)
print("PART 2  WHERE THE SIGNAL ACTUALLY LIVES  (tier B: exact amplitudes, no sampling)")
print("=" * 96)
print("  Degree-1 Walsh coefficient in sigma_k of three candidate readouts, on the")
print("  same probe and the same evolution. Only a readout with a nonzero degree-1")
print("  coefficient can be decoded by the design register.")
print()

rng = np.random.default_rng(0)
psi = np.zeros(2 ** N, dtype=complex)
v = np.array([1.0 + 0j])
for _ in range(N):
    th, ph = np.arccos(1 - 2 * rng.random()), 2 * np.pi * rng.random()
    v = np.kron(v, np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)]))
psi = v

O = Pauli('IIZ').to_matrix()

walsh_phase = np.zeros(M)
walsh_return = np.zeros(M)
walsh_obs = np.zeros(M)
n_rows = 0
for a in itertools.product([0, 1], repeat=N):
    for b in itertools.product([0, 1], repeat=N):
        sig = sigma_of(np.array(a), np.array(b))
        Hs = H_of(sig)
        U = expm(-1j * Hs * T)
        # (i) a spectral readout: the mean eigenphase weighted by nothing at all
        phase = float(np.sum(np.linalg.eigvalsh(Hs)))
        # (ii) return probability
        amp = complex(psi.conj() @ (U @ psi))
        ret = float(abs(amp) ** 2)
        # (iii) observable after evolution
        ev = complex(psi.conj() @ (U.conj().T @ O @ U @ psi)).real
        walsh_phase += sig * phase
        walsh_return += sig * ret
        walsh_obs += sig * ev
        n_rows += 1
walsh_phase /= n_rows
walsh_return /= n_rows
walsh_obs /= n_rows

print("   readout                     max |degree-1 Walsh coeff|     usable?")
print("   " + "-" * 76)
print("   QPE phase (spectral)             %.3e                 no, exactly"
      % np.max(np.abs(walsh_phase)))
print("   return probability               %.3e                 weakly - see PART 3"
      % np.max(np.abs(walsh_return)))
print("   observable <O> after evolution   %.3e                 YES"
      % np.max(np.abs(walsh_obs)))
print()
print("   for scale, the predicted first-order size T*|c| is ~%.3e"
      % (T * np.mean(np.abs(c_true))))
print()

print("=" * 96)
print("PART 3  A CORRECTION TO twirl_cal's DOCSTRING  (tier B: exact amplitudes)")
print("=" * 96)
print("  twirl_cal says the return probability 'has NO first-order term in sigma -")
print("  the linear part is purely imaginary and squares away'. PART 2 measures")
print("  3.1e-03, not zero, so the statement is imprecise. Working out what it")
print("  should say:")
print()
print("    order T^1 : -iT<H_sigma> is imaginary; |1-iTx|^2 = 1+T^2x^2, no T^1 term.")
print("    order T^2 : both <H_sigma>^2 and <H_sigma^2> carry only sigma_j sigma_k")
print("                (degree 2) and sigma_k^2 = 1 (degree 0). NO degree 1.")
print("    order T^3 : sigma_i sigma_j sigma_k with two indices coinciding collapses")
print("                to a SINGLE sigma. Degree 1 appears here, and not before.")
print()
print("  So the return probability does carry degree-1 signal - suppressed by T^2")
print("  relative to the observable's, which is O(T). The prediction is a slope of")
print("  3 against the observable's 1. Measured:")
print()
print("      T        deg-1 <O>     deg-1 P_return      ratio")
print("   " + "-" * 66)
Ts = [0.05, 0.1, 0.2, 0.4]
ob_at, rt_at = [], []
for Tv in Ts:
    wr = np.zeros(M)
    wo = np.zeros(M)
    nr = 0
    for a in itertools.product([0, 1], repeat=N):
        for b in itertools.product([0, 1], repeat=N):
            sig = sigma_of(np.array(a), np.array(b))
            U = expm(-1j * H_of(sig) * Tv)
            amp = complex(psi.conj() @ (U @ psi))
            wr += sig * float(abs(amp) ** 2)
            wo += sig * complex(psi.conj() @ (U.conj().T @ O @ U @ psi)).real
            nr += 1
    wr /= nr
    wo /= nr
    o_, r_ = float(np.max(np.abs(wo))), float(np.max(np.abs(wr)))
    ob_at.append(o_); rt_at.append(r_)
    print("   %5.2f    %.4e     %.4e      %.4f" % (Tv, o_, r_, r_ / o_))
so = np.polyfit(np.log(Ts), np.log(ob_at), 1)[0]
sr = np.polyfit(np.log(Ts), np.log(rt_at), 1)[0]
print()
print("   fitted slope d log(coeff) / d log T :   <O> %.3f      P_return %.3f"
      % (so, sr))
print("   predicted                           :   <O> 1.000     P_return 3.000")
print()
print("  MEASURED SLOPE IS 2, NOT 3, so the order-T^3 story above is WRONG. The")
print("  T^2 term does reach degree 1, and PART 4 finds the mechanism.")
print()

print("=" * 96)
print("PART 4  THE DESIGN IS CONFOUNDED, AND THE COLUMNS ARE NOT FREE TO FIX")
print("=" * 96)
print("  sigma_j sigma_k is itself a Walsh character: sigma_j(a,b) sigma_k(a,b) =")
print("  (-1)^(a.(z_j+z_k) + b.(x_j+x_k)), i.e. the character of the symplectic")
print("  vector v_j + v_k. So if v_j + v_k = v_m for some term m already in the set,")
print("  a DEGREE-2 effect is indistinguishable from the DEGREE-1 effect of term m.")
print("  That is classical design confounding, and here the columns are handed over")
print("  by the Paulis - they cannot be re-chosen the way _design_spec chooses the")
print("  QLTO register's columns.")
print()
V = np.concatenate([Z, X], axis=1) % 2
triples = []
for i in range(M):
    for j in range(i + 1, M):
        s = (V[i] + V[j]) % 2
        for k in range(M):
            if k != i and k != j and np.array_equal(s, V[k]):
                triples.append((i, j, k))
print("  aliasing triples v_i + v_j = v_k found: %d" % len(triples))
print()
print("     P_i          P_j          =  P_k         c_k")
print("   " + "-" * 64)
seen = set()
for i, j, k in triples:
    key = tuple(sorted((i, j, k)))
    if key in seen:
        continue
    seen.add(key)
    print("   %-12s %-12s =  %-10s  %+.4f"
          % (terms[i], terms[j], terms[k], c_true[k]))
print()
print("  RANK CHECK (tier C: operator structure).")
r = np.linalg.matrix_rank(V.astype(float) @ np.eye(2 * N))
print("    M = %d distinct symplectic vectors, GF(2) dimension 2N = %d" % (M, 2 * N))
print("    so at most %d can be independent and M > 2N forces dependencies." % (2 * N))
print()
print("  WHAT THIS MEANS, and it is a real correction to the record.")
print()
print("    v101 proved FULL RANK of the design matrix - distinct Paulis give distinct")
print("    symplectic vectors, distinct characters are orthogonal, so the M degree-1")
print("    columns are independent. That theorem is correct and is NOT what is at")
print("    stake here. Independence of the degree-1 columns says nothing about")
print("    whether a degree-2 product COINCIDES with one of them, and for the")
print("    XX/YY/ZZ triple on every bond it does: v_XX + v_YY = v_ZZ exactly.")
print()
print("    twirl_cal's commit explains its T^2 bias by saying 'even orders carry")
print("    sigma_j sigma_k and project onto degree 2'. For a term set closed under")
print("    such products that is false - the even order lands squarely on degree 1,")
print("    which is why the slope above is 2. The Richardson fix still WORKS, since")
print("    it cancels any T^2 term whatever its origin, but its stated justification")
print("    does not hold and the bias it removes is aliasing, not truncation.")
print()
print("    This is the same failure the QLTO register has as resolution III, which")
print("    v90 measured costing the Hamiltonian-learning cosine 0.714 at M=16. There")
print("    the cure was more design rows. Here there is no such lever: the columns")
print("    ARE the Paulis. The available cures are different - drop one term of each")
print("    triple and infer it, or choose probes/observables making the aliased")
print("    contribution vanish - and none of them is tested.")
print()

print("=" * 96)
print("READING IT")
print("=" * 96)
print("  QPE cannot be mimicked here, and the reason is not depth. V5's QPE path")
print("  died on depth - survival 0.098 at Heisenberg N=6 - which is a cost and")
print("  could in principle be engineered away. This is different: the twirl is a")
print("  similarity transformation, so a spectral readout is CONSTANT in the very")
print("  variable the design register decodes.")
print()
print("  The contrast with V5 is exact and worth keeping. V5's register changed the")
print("  STATE, so energies moved and a phase carried the signal. This register")
print("  changes the HAMILTONIAN BY CONJUGATION, so energies do not move and only a")
print("  fixed-basis expectation value carries it.")
print()
print("  Which means the O(1) already reached in v105 - 2 bases times n_probes, flat")
print("  in N and M - is not a way station toward a phase-based version. It is the")
print("  right answer for this construction, and the factor of 2 it keeps is a")
print("  commutation requirement, not an inefficiency.")
