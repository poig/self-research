"""Which Hamiltonian coefficients are Heisenberg-learnable without control.

CLAIM. Probe |psi>, evolve under the unknown H for time T, then apply the model
inverse e^{+iH(theta)T} and ask whether the probe returned. No control is
interleaved into the device evolution. Then the number of coefficient
directions carrying Heisenberg (T^2) Fisher information is

    rank span{ diag_H(P_k) }

where diag_H projects onto the commutant of H, i.e. the diagonal part of P_k in
an eigenbasis of H. The remaining M - rank directions are blind to EVERY probe
state and EVERY evolution time.

DERIVATION. The parameter derivative of the evolution is governed by

    A_k = int_0^T e^{iHs} P_k e^{-iHs} ds .

In an eigenbasis of H, (A_k)_ab = (P_k)_ab * I_ab with

    I_ab = T                                     if E_a = E_b
    I_ab = (e^{i(E_a-E_b)T} - 1)/(i(E_a-E_b))    otherwise, BOUNDED by
                                                 2/|E_a-E_b| uniformly in T.

So A_k = T * diag_H(P_k) + O(1). The quantum Fisher matrix of the state family
e^{-iH(c)T}|psi> is F = 4 Cov_psi(A_k, A_l), hence

    F = 4 T^2 Cov(diag_H(P_k), diag_H(P_l)) + O(T) .

The T^2 block is a Gram matrix of the commutant projections, so its rank - and
therefore the count of Fisher eigenvalues growing as T^2 - is the rank of their
span. Summing F over a spanning set of probes realises the operator Gram rank;
no single probe can exceed it.

WHY THIS DOES NOT CONTRADICT THE CONTROL-FREE LOWER BOUND. arXiv:2606.19486
proves Omega(Lambda/eps^2 log(Lambda/eps)) for CONTROL-FREE protocols, meaning
prepare / evolve / measure with no ancillas and no intermediate operations. The
terminal model inversion e^{+iH(theta)T} is an intermediate operation and sits
outside that class. It is nonetheless weaker than the interleaved fast pulses
Hamiltonian reshaping needs: nothing is applied to the system DURING its
evolution under the unknown H.

WHAT IT IS GOOD FOR. The criterion is computable from the term structure before
any experiment is run, and it names WHICH directions require control instead of
forcing reshaping of all M terms.

MEASURED. Criterion vs directly fitted Fisher-eigenvalue growth exponents, exact
A_k in the eigenbasis, T in [50, 800], 32 probe states summed:

    case                M   gram rank   T^2 dirs   match
    commuting Z-type    6       6           6      YES
    non-commuting ZX    6       4           4      YES
    transverse Ising    6       4           4      YES
    generic mixed       6       4           4      YES

Blind count d = M - rank across families:

    transverse Ising   d = N-1 ~ M/2, PERSISTENT   (see below: INTEGRABILITY,
                                                     not the Z2 symmetry)
    Heisenberg XYZ     d = 0 for N >= 4
    random Pauli sets  d = 0 for N >= 6

WHY TFIM IS PERSISTENTLY DEFICIENT: INTEGRABILITY, NOT ITS Z2 SYMMETRY. TFIM's
spin-flip symmetry S = prod_i X_i turns out not to be the cause - both term
types (Z_iZ_{i+1} and X_i) are EVEN under S, so the standard "odd operators
have vanishing diagonal" argument does not apply and does not explain the gap.

The real cause is that TFIM is Jordan-Wigner integrable: it maps to N free-
fermion modes with conserved occupations n_1..n_N, and every local quadratic
operator's commutant-diagonal part is confined to that same N-dimensional
space regardless of which operator it is. Measured directly, N=3..8: the N
field terms {X_i} ALONE already achieve rank N - the full measured rank - and
adding the N-1 coupling terms {Z_iZ_{i+1}} contributes ZERO new directions.
The couplings are not individually deficient (rank{Z_iZ_{i+1}} alone is N-1
already), they are collectively confined to the span the fields cover.

So d = M - rank is a symptom of integrability, not of the specific symmetry:
free-fermion-solvable H is a second classically-tractable rung below the
commuting case, still not where an ancilla register does real work. The
regime where it might is chaotic, non-commuting H with d = 0 and no efficient
classical description - e.g. the ZZ+XY tunable-coupler crosstalk case in the
application screen, not TFIM or any other integrable model.

So the obstruction is a SYMMETRY/INTEGRABILITY phenomenon, not a
non-commutativity one:
generic non-commuting Hamiltonians have no blind directions at all. An earlier
non-commuting benchmark in this project measured SQL scaling and was read as
evidence against the method; its test Hamiltonian (ZII IZI IIZ XII IXI XXI) has
rank 4 of 6, so it measured a degenerate case rather than the generic one.

RANK IS NOT THE OPERATIONAL BOTTLENECK - CONDITIONING IS. Everything above
concerns the QUANTUM FISHER INFORMATION, what an optimal estimator could
extract. The QLTO gradient-ascent estimator does NOT attain it, and full rank
does not rescue it. Measured, 64 shots/gradient, 48 steps/level, 3 seeds, error
against total evolution time:

    case                 rank    d   cond(F)     fitted exponent
    commuting Z-type      6/6    0   1.00e+00        -0.924
    deficient ZX          4/6    2   4.29e+06        -0.126
    Heisenberg XYZ N=4    9/9    0   8.76e+15        +0.012
                                     Heisenberg -1.000, SQL -0.500

Heisenberg XYZ at N=4 has NO blind directions - all 9 carry T^2 information -
and the estimator still fails completely, because cond(F) = 8.8e15 is
numerically singular despite formal full rank. The relationship is monotone in
cond(F) across all three cases and is not explained by rank.

The reason is structural. Commuting terms give A_k = T*P_k with the P_k
orthogonal, so F is proportional to the IDENTITY and cond(F) = 1.000 exactly.
Non-commuting terms give A_k -> T*diag_H(P_k), and commutant projections are far
from orthogonal even when they are linearly independent. So this method reaches
the Heisenberg limit precisely when F is isotropic, which is the commuting case
and essentially nothing else.

Metric preconditioning is the textbook repair and it was tested and FAILED here:
with an exact Hessian recomputed per level and its probes billed, the
non-commuting exponent went from -0.435 to +0.033, i.e. the error stopped
falling entirely. Dividing by small eigenvalues amplifies exactly the directions
where a cheap 32-64 shot gradient is pure noise, and the max-normalised step
then commits its full length to that amplified noise. The cheap-noisy-gradient
design that gives QLTO its one-circuit-per-M-coefficients property is in direct
conflict with the preconditioning that non-commuting H requires. That tension
looks structural rather than incidental.
"""
import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
MAP = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def pauli(s):
    r = np.array([[1]], complex)
    for ch in s:
        r = np.kron(r, MAP[ch])
    return r


def _put(N, d):
    s = ['I'] * N
    for i, ch in d.items():
        s[i] = ch
    return ''.join(s)


def tfim(N):
    return ([_put(N, {i: 'Z', i + 1: 'Z'}) for i in range(N - 1)]
            + [_put(N, {i: 'X'}) for i in range(N)])


def heisenberg(N):
    return [_put(N, {i: ch, i + 1: ch})
            for i in range(N - 1) for ch in 'XYZ']


def A_exact(w, V, Ps, T):
    """A_k = int_0^T e^{iHs}P_k e^{-iHs} ds, closed form in the eigenbasis."""
    d = w[:, None] - w[None, :]
    small = np.abs(d) < 1e-12
    ker = np.where(small, T, (np.exp(1j * d * T) - 1) / (1j * np.where(small, 1, d)))
    return [V @ ((V.conj().T @ P @ V) * ker) @ V.conj().T for P in Ps]


def commutant_rank(H, Ps, tol=1e-8, dtol=1e-9):
    """rank span{diag_H(P_k)} - the criterion, computed with no experiment.

    diag_H is projection onto the COMMUTANT of H, which is block diagonal over
    degenerate eigenvalue groups, not merely diagonal. A_k grows as T wherever
    E_a = E_b, and for degenerate H that includes off-diagonal elements INSIDE
    a degenerate block, so the strict diagonal is the wrong projector whenever
    the spectrum is degenerate.

    Measured: on every case in this file the strict diagonal happens to give
    the same rank as the block projection, including Heisenberg N=3 and N=5,
    which carry 4 and 16 degenerate pairs. That agreement is an observation,
    not a theorem, so the block form is used because it is the one the
    derivation specifies.
    """
    w, V = np.linalg.eigh(H)
    groups, i = [], 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and abs(w[j + 1] - w[i]) < dtol:
            j += 1
        groups.append((i, j))
        i = j + 1
    Dk = []
    for P in Ps:
        Pt = V.conj().T @ P @ V
        B = np.zeros_like(Pt)
        for a, b in groups:
            B[a:b + 1, a:b + 1] = Pt[a:b + 1, a:b + 1]
        Dk.append(V @ B @ V.conj().T)
    G = np.array([[np.trace(a.conj().T @ b).real for b in Dk] for a in Dk])
    e = np.sort(np.linalg.eigvalsh(G))[::-1]
    return int(np.sum(e > tol * max(e[0], 1e-30)))


def fisher(A, psi):
    M = len(A)
    m = np.array([np.vdot(psi, Ak @ psi).real for Ak in A])
    F = np.zeros((M, M))
    for k in range(M):
        for l in range(k, M):
            v = np.vdot(psi, (A[k] @ A[l] + A[l] @ A[k]) @ psi).real / 2
            F[k, l] = F[l, k] = 4 * (v - m[k] * m[l])
    return F


def t2_directions(H, Ps, Ts=(50., 100., 200., 400., 800.), nprobe=32, seed=3):
    """Count Fisher eigendirections growing as T^2, by direct fit."""
    w, V = np.linalg.eigh(H)
    D = H.shape[0]
    rng = np.random.default_rng(seed)
    probes = [np.ones(D, complex) / np.sqrt(D)]
    for _ in range(nprobe - 1):
        v = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        probes.append(v / np.linalg.norm(v))
    evs = []
    for T in Ts:
        A = A_exact(w, V, Ps, T)
        F = sum(fisher(A, p) for p in probes)
        evs.append(np.sort(np.linalg.eigvalsh(F))[::-1])
    evs = np.array(evs)
    lt = np.log(np.array(Ts))
    ex = [np.polyfit(lt, np.log(np.maximum(evs[:, i], 1e-20)), 1)[0]
          for i in range(len(Ps))]
    return sum(1 for e in ex if e > 1.5), ex


def main():
    cases = {
        'commuting Z-type': ['ZII', 'IZI', 'IIZ', 'ZZI', 'ZIZ', 'IZZ'],
        'non-commuting ZX': ['ZII', 'IZI', 'IIZ', 'XII', 'IXI', 'XXI'],
        'transverse Ising': ['ZZI', 'IZZ', 'ZIZ', 'XII', 'IXI', 'IIX'],
        'generic mixed   ': ['ZII', 'IXI', 'IIY', 'ZZI', 'XIY', 'IZX'],
    }
    print('CRITERION vs MEASURED T^2 DIRECTION COUNT')
    print('%-18s %4s %10s %10s   %s'
          % ('case', 'M', 'criterion', 'measured', 'match'))
    print('-' * 60)
    ok = True
    for name, terms in cases.items():
        Ps = [pauli(t) for t in terms]
        c = np.random.default_rng(11).uniform(-0.6, 0.6, len(Ps))
        H = sum(a * P for a, P in zip(c, Ps))
        H = (H + H.conj().T) / 2
        r = commutant_rank(H, Ps)
        n2, ex = t2_directions(H, Ps)
        ok &= (r == n2)
        print('%-18s %4d %10d %10d   %s'
              % (name, len(Ps), r, n2, 'YES' if r == n2 else 'NO'))
        print('   exponents: ' + ' '.join('%+.2f' % e for e in ex))
    print()
    print('BLIND COUNT d = M - rank ACROSS FAMILIES')
    print('%-18s %4s %4s %6s %5s' % ('family', 'N', 'M', 'rank', 'd'))
    print('-' * 42)
    for label, fn, Ns in (('transverse Ising', tfim, range(3, 9)),
                          ('Heisenberg XYZ', heisenberg, range(3, 8))):
        for N in Ns:
            terms = fn(N)
            Ps = [pauli(t) for t in terms]
            c = np.random.default_rng(11).uniform(-0.6, 0.6, len(Ps))
            H = sum(a * P for a, P in zip(c, Ps))
            H = (H + H.conj().T) / 2
            r = commutant_rank(H, Ps)
            print('%-18s %4d %4d %6d %5d'
                  % (label, N, len(Ps), r, len(Ps) - r))
    print()
    print('criterion matched measurement in every case: %s' % ok)
    print()
    print('WHY TFIM IS DEFICIENT: do the N field terms {X_i} ALONE already')
    print('span the full measured rank, making the N-1 couplings redundant?')
    print('%-5s %4s %10s %12s %14s %s'
          % ('N', 'M', 'rank_all', 'rank_X_only', 'rank_ZZ_only', 'X spans all?'))
    print('-' * 62)
    for N in range(3, 9):
        zz = [_put(N, {i: 'Z', i + 1: 'Z'}) for i in range(N - 1)]
        xx = [_put(N, {i: 'X'}) for i in range(N)]
        Pzz, Pxx = [pauli(t) for t in zz], [pauli(t) for t in xx]
        c = np.random.default_rng(11).uniform(-0.6, 0.6, len(zz) + len(xx))
        H = sum(a * P for a, P in zip(c, Pzz + Pxx))
        H = (H + H.conj().T) / 2
        r_all = commutant_rank(H, Pzz + Pxx)
        r_x = commutant_rank(H, Pxx) if Pxx else 0
        r_zz = commutant_rank(H, Pzz) if Pzz else 0
        print('%-5d %4d %10d %12d %14d   %s'
              % (N, len(zz) + len(xx), r_all, r_x, r_zz,
                 'YES' if r_x == r_all else 'no'))


if __name__ == '__main__':
    main()
