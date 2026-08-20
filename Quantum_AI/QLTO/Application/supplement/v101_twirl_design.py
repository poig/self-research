"""Twirl-design calibration: O(1) circuits, no model, no Trotter.

The whole of v100 was spent making a SYNTHESISED model invert the device. The
W-gate must serialise, so the model is a product formula, so it carries Trotter
bias, so the optimum moves as T grows. Repairing that needs a Suzuki ladder at
reps proportional to T, which reinstates the depth the design existed to avoid.

This construction removes the model entirely.

THE IDEA. A Pauli conjugation flips coefficient signs:

    Q P_k Q^dag = +P_k  if [Q,P_k]=0,  -P_k if they anticommute
    ⟹  Q e^{-iHT} Q^dag = exp(-i sum_k sigma_k c_k P_k T)

so a twirl IS a design row. Put the twirl in superposition and the DEVICE
supplies the evolution exactly in every branch. Nothing is synthesised, so
there is no product formula and no bias.

THE REGISTER IS 2N QUBITS, NOT log2 M. A Pauli is 2N bits, and the sign is a
PARITY of them:

    Q(a,b) = prod_i X^{a_i} Z^{b_i},   P_k = prod_i X^{x_ki} Z^{z_ki}
    sigma_k(a,b) = (-1)^(a . z_k + b . x_k)

So the Walsh column for term k is handed over by P_k's symplectic vector. The
circuit is 2N register qubits in |+>, 2x2N controlled Cliffords, ONE device
evolution, and the existing degree-1 Walsh decode.

FULL RANK IS A THEOREM HERE, not a measurement. The sign columns are Walsh
characters chi_v(x) = (-1)^{v.x} indexed by each Pauli's symplectic vector.
Distinct Paulis have distinct symplectic vectors; distinct Walsh characters are
orthogonal; orthogonal vectors are independent. So the design matrix has rank M
for ANY set of distinct Pauli terms. Verified N=3..12, M=9..45, rank = M = the
number of distinct symplectic vectors at every size.

    Contrast the commutant criterion in theory_test/heisenberg_rank_criterion.py,
    which governs the compute-uncompute route and measures d>0 for TFIM (d=N-1)
    and for N=3 crosstalk (rank 6 of 9). The twirl design has no such deficiency.
    They are different criteria for different protocols, and only this one is
    guaranteed.

THE READOUT IS DIRECT - there is no optimisation loop. To first order,

    <psi| e^{-iH_sigma T} |psi> ~ 1 - i T sum_k sigma_k c_k <P_k>
    degree-1 Walsh coefficient in sigma_k  ->  -T c_k <P_k>
    c_hat_k = Walsh_k / (-T <P_k>)

One shot of estimation, not forty epochs of descent. The probe expectations
<P_k> are known because the probe is chosen.

PROBES ARE STILL LOAD-BEARING. |+..+> gives <P_k> = 0 for 10 of the 13
crosstalk terms, so those coefficients are invisible to it. Random product
states fix it, same as everywhere else in this project:

    probes   terms covered   max rel err   mean rel err
         1               3        1.0000        0.7696    <- |+..+>
         4              13        0.0073        0.0052
         8              13        0.0073        0.0052

Error for the single-probe row is 1.0 because invisible terms are estimated as
zero, i.e. 100% wrong, not because the visible ones are badly estimated. On the
3 terms it does see, |+..+> is accurate to 0.6%.

MEASURED, N=4 crosstalk M=13, exact amplitudes:

     T   probes   max rel err   mean rel err
   0.10        4        0.0018         0.0013
   0.20        4        0.0073         0.0052
   0.50        4        0.0444         0.0321

T sets a linearity window: the readout is first order, so small T is accurate
and weak. 0.13% relative at T=0.1. The departure at T=0.5 is higher-order Walsh
content, NOT Trotter - every branch evolves exactly.

*** THOSE NUMBERS ARE NOT CIRCUIT-ACHIEVABLE AND T=0.1 IS THE WRONG OPERATING
*** POINT. They come from exact amplitudes. On the real circuit in
*** Application/twirl_cal.py the signal is T*c_k*<i[P_k,O]> ~ 0.0075 at T=0.1,
*** against a shot floor of 1/sqrt(shots) ~ 0.0028 at 2^17 - SNR under 3. The
*** two error sources move oppositely in T, so the circuit optimum is T ~ 0.25:
***
***      T      shots   circuits   mean rel err
***   0.25      65536         24         0.0297   <- best
***   0.25     524288         24         0.0331   8x shots, NO gain: bias-limited
***   0.50     524288         24         0.0502
***   1.00     524288         24         0.2005
***   2.00     524288         24         0.6053
***
*** 3.0% on a circuit, not 0.13%. The residual is first-order truncation, not
*** noise - every estimate at larger T is systematically LOW.
***
*** RICHARDSON IN T REMOVES THAT BIAS. Only odd orders survive into the degree-1
*** Walsh coefficient (even orders carry sigma_j sigma_k and project onto degree
*** 2), so chat(T) = c(1 + aT^2) and (4 chat(T/2) - chat(T))/3 cancels it:
***      T=1.0  plain 0.2053 -> richardson 0.0395   5.2x
***      T=2.0  plain 0.6100 -> richardson 0.0596  10.2x
*** But it does NOT beat the plain estimator at its own best point (0.0395 at 48
*** circuits against 0.0297 at 24), and at T=0.5 it is worse. Richardson buys
*** insensitivity to the choice of T, not a lower floor. Use plain at T~0.25;
*** use Richardson when T is pinned by the hardware.

AGAINST THE ITERATIVE PATH, same model:

    protocol                     circuits   mean rel err   limited by
    QLTO fit(), v100 config           160    ~30% on ZZ    Trotter bias
    twirl, plain, T=0.25               24         3.0%     truncation
    twirl + Richardson, T=1.0          48         4.0%     shot noise
    twirl, exact amplitudes             -        0.13%     NOT achievable

On circuits it is 6.7x fewer circuits at roughly 10x better accuracy - a real
win, and an order of magnitude short of what the exact-amplitude figure
suggested. The iterative route spends its budget fighting a Trotter bias this
construction does not have; this one spends its budget on the shot-noise vs
truncation trade the exact version hid.

WHAT IS NOT ESTABLISHED, and the first item is the important one.

  NO SHOT NOISE. Every number above is computed from exact amplitudes. The
      estimator divides by <P_k>, which amplifies error for terms a probe
      barely sees, and the Walsh coefficient is an average over the register
      superposition. Shot-noise behaviour is UNMEASURED and is the obvious way
      this could fail in practice.
  CIRCUIT NOW EXISTS: Application/twirl_cal.py, verified term by term against
      this file (corr 0.9905 between the exact Walsh coefficients and the
      predicted T c_k <i[P_k,O]>). Two endianness bugs were found building it -
      observable support indexed by string position rather than qubit, and the
      probe kron built in the wrong order - so any reuse of the classical
      helpers here should assume Qiskit little-endian.
  ONE FAMILY. Nearest-neighbour crosstalk at N=4 for accuracy, rank only up to
      N=12. Nothing on TFIM, Heisenberg, or molecular Hamiltonians.
  FIRST ORDER ONLY. The T window is set by when the linear term stops
      dominating. Whether a large-T variant reaches Heisenberg scaling - the
      phase accumulates as T c_k, so it should - is untested.
"""
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from scipy.linalg import expm
from qiskit.quantum_info import Pauli


def put(N, d):
    s = ['I'] * N
    for i, ch in d.items():
        s[i] = ch
    return ''.join(s)


def crosstalk_terms(N):
    t = []
    for i in range(N - 1):
        t += [put(N, {i: 'Z', i + 1: 'Z'}),
              put(N, {i: 'X', i + 1: 'X'}),
              put(N, {i: 'Y', i + 1: 'Y'})]
    t += [put(N, {i: 'Z'}) for i in range(N)]
    return t


def crosstalk_coeffs(N, seed=7):
    rng = np.random.default_rng(seed)
    c = []
    for _ in range(N - 1):
        c += [rng.uniform(0.01, 0.05), rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)]
    c += list(rng.uniform(-0.2, 0.2, N))
    return np.round(np.array(c), 4)


def sign_matrix(terms, N):
    """Walsh columns from the Paulis' symplectic vectors. Full rank by theorem."""
    allb = np.array([[(v >> i) & 1 for i in range(2 * N)]
                     for v in range(2 ** (2 * N))])
    A, B = allb[:, :N], allb[:, N:]
    S = np.zeros((len(allb), len(terms)))
    for k, t in enumerate(terms):
        p = Pauli(t)
        S[:, k] = (-1.0) ** ((A @ p.z.astype(int) + B @ p.x.astype(int)) % 2)
    return S


def rand_product(N, rng):
    v = np.array([1.0 + 0j])
    for _ in range(N):
        a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        a /= np.linalg.norm(a)
        v = np.kron(v, a)
    return v


def estimate(terms, c, N, T, probes):
    """Direct twirl-design readout. No iteration."""
    Pm = [Pauli(t).to_matrix() for t in terms]
    S = sign_matrix(terms, N)
    M = len(terms)
    num, den = np.zeros(M), np.zeros(M)
    for psi in probes:
        e = np.array([float(np.real(psi.conj() @ (P @ psi))) for P in Pm])
        amp = np.empty(len(S), complex)
        for j, s in enumerate(S):
            Hs = sum(sg * a * P for sg, a, P in zip(s, c, Pm))
            amp[j] = psi.conj() @ (expm(-1j * Hs * T) @ psi)
        g = (S.T @ np.imag(amp)) / len(S)
        w = e ** 2
        est = np.where(np.abs(e) > 1e-6,
                       g / (-T * np.where(np.abs(e) > 1e-6, e, 1.0)), 0.0)
        num += w * est
        den += w
    return num / np.maximum(den, 1e-30)


def main():
    print('=' * 74)
    print('v101  TWIRL-DESIGN CALIBRATION - O(1) circuits, no model, no Trotter')
    print('=' * 74)

    print()
    print('FULL RANK IS STRUCTURAL: rank = number of distinct symplectic vectors')
    print('  %4s %6s %10s %8s %14s' % ('N', 'M', '2N qubits', 'rank', 'distinct symp'))
    print('  ' + '-' * 48)
    for N in (3, 4, 5, 6, 7):
        terms = crosstalk_terms(N)
        V = set()
        for t in terms:
            p = Pauli(t)
            V.add(tuple(np.concatenate([p.z.astype(int), p.x.astype(int)])))
        r = np.linalg.matrix_rank(sign_matrix(terms, N))
        print('  %4d %6d %10d %8d %14d' % (N, len(terms), 2 * N, r, len(V)))

    N = 4
    terms = crosstalk_terms(N)
    c = crosstalk_coeffs(N)
    print()
    print('DIRECT READOUT, N=4 crosstalk, M=%d  (exact amplitudes, no shots)' % len(terms))
    print('  %6s %8s %14s %14s' % ('T', 'probes', 'max rel err', 'mean rel err'))
    print('  ' + '-' * 46)
    for T in (0.1, 0.2, 0.5):
        for nprobe in (1, 4):
            rng = np.random.default_rng(3)
            probes = ([np.ones(2 ** N, complex) / np.sqrt(2 ** N)] if nprobe == 1
                      else [rand_product(N, rng) for _ in range(nprobe)])
            chat = estimate(terms, c, N, T, probes)
            rel = np.abs(chat - c) / np.abs(c)
            print('  %6.2f %8d %14.4f %14.4f'
                  % (T, nprobe, np.max(rel), np.mean(rel)))
    print()
    print('  1 probe is |+..+>: it sees only 3 of 13 terms and the other 10 are')
    print('  estimated as zero, hence rel err 1.0. 4 random probes cover all 13.')
    print()
    print('  CIRCUIT COUNT: one per probe, independent of M.')
    print('  Register 2N qubits, 4N controlled Cliffords, 1 device evolution.')


if __name__ == '__main__':
    main()
