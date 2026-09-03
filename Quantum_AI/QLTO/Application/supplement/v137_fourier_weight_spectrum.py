"""Where does a variational landscape's Fourier energy actually SIT by weight?

TIER B - exact amplitudes from Statevector on a 3^M grid, no sampling. R1
permits tier B for structure; no accuracy or cost figure here is tier A.

WHY THIS WAS RUN. qlto_walk.py encodes the register value d as a LINEAR PHASE,
u_j(d) = (2 pi / N) c_j d, so E(theta + u(d)) is a Fourier series in d with
frequency <k,c>. A DFT over d then separates modes by frequency, and bin c_j was
supposed to return A_{e_j}, the weight-1 coefficient - giving an EXACT gradient
with no finite-radius bias, at D log2(M) register qubits.

It returned cos(g, exact) = -0.0666. The decode is not the bug.

WHAT THE SPECTRUM ACTUALLY IS. Sampling E on a 3^M grid (3 points per axis is
exactly Nyquist for support {-1,0,1}) and taking the M-dimensional FFT gives
every A_k. Binning |A_k|^2 by weight |k|_0:

    the weight-1 and weight-2 energy is IDENTICALLY ZERO, and the bulk sits at
    weight ~ M/2.

So bin c_j reads a coefficient that does not exist, and the -0.0666 is aliased
high-weight content. This is not a small effect to be corrected; it is the
opposite of the assumption.

WHY WEIGHT-1 VANISHES. A_{e_j} is the first harmonic in theta_j after averaging
over the FULL period of every other parameter. Averaging a rotation over its
period projects onto its commutant, and averaging over all M-1 of them projects
so far that no first harmonic in theta_j survives. Weight 0 vanishes for the
same reason and more simply: the full-torus average of <H> is Tr(H)/2^n = 0 for
a traceless H.

WHAT THIS SETTLES, AND IT IS NOT WHAT THE SCRIPT WAS BUILT TO TEST.

  A FULL-PERIOD DFT CANNOT GIVE A LOCAL GRADIENT AT LOG WIDTH. dE/dtheta_j
  = sum_{k: k_j != 0} A_k (i k_j) needs every bin containing j, and the bin
  label <k,c> does not carry k_j. Separating them needs <k,c> distinct over all
  of {-1,0,1}^M - a Sidon set of order M - hence max c ~ 2^M, N ~ 2^M, and a
  register of M qubits. The log advantage is gone.

  V6's SMALL-R BOX IS NOT A DEFECT TO BE REMOVED. Its marginal is a finite
  difference, so EVERY weight contributes to the same bin, attenuated by
  cos(R)^{|k|-1} but never separated:

      alpha_j(R)/sin R = sum_d (cos R)^{d-1} D_j^{(d)}

  The attenuation is exactly what lets one log-width register carry all weights
  at once. Trading it for exactness trades away the multiplexing with it.

  AND D_eff IS NOT SMALL FOR THIS ANSATZ. The bulk at weight ~M/2 means the
  cos R polynomial has high degree, so v135's geometric decay of FIT RESIDUALS
  (3.8e-2 -> 4.1e-3 -> 9.8e-5 at 2/3/4 radii) is a statement about how well a
  low-degree polynomial APPROXIMATES that sum on the sampled radii - not
  evidence that the landscape is low-weight. Those are different claims and this
  file separates them.
"""
import itertools
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector


def heis(n):
    t = []
    for i in range(n - 1):
        for p in ('XX', 'YY', 'ZZ'):
            lab = ['I'] * n
            lab[i], lab[i + 1] = p[0], p[1]
            t.append((''.join(reversed(lab)), 1.0))
    return SparsePauliOp.from_list(t)


def tfim(n):
    t = [('I' * (n - i - 2) + 'ZZ' + 'I' * i, 1.0) for i in range(n - 1)]
    t += [('I' * (n - i - 1) + 'X' + 'I' * i, 0.5) for i in range(n)]
    return SparsePauliOp.from_list(t)


def weight_spectrum(anz, H, theta, seed=0):
    """|A_k|^2 binned by |k|_0, from an exact 3^M grid FFT."""
    M = anz.num_parameters
    G = 3                                   # Nyquist for support {-1,0,1}
    off = 2 * np.pi * np.arange(G) / G
    vals = np.zeros([G] * M)
    for idx in itertools.product(range(G), repeat=M):
        x = theta + off[list(idx)]
        vals[idx] = float(np.real(
            Statevector(anz.assign_parameters(x)).expectation_value(H)))
    Ah = np.fft.fftn(vals) / G ** M
    w = np.zeros(M + 1)
    for idx in itertools.product(range(G), repeat=M):
        d = sum(1 for i in idx if i != 0)
        w[d] += abs(Ah[idx]) ** 2
    return w / max(w.sum(), 1e-300)


if __name__ == '__main__':
    print(__doc__.split('\n')[0])
    print("TIER B - exact 3^M grid FFT, no shots.")
    print("")
    rng = np.random.default_rng(0)
    for name, mk in (('Heisenberg', heis), ('TFIM', tfim)):
        for n_sys, reps in ((2, 1), (2, 2), (3, 1)):
            anz = efficient_su2(n_sys, reps=reps).decompose()
            M = anz.num_parameters
            if 3 ** M > 4_000_000:
                print("  %-11s N=%d reps=%d  M=%2d   skipped (3^M too large)"
                      % (name, n_sys, reps, M))
                continue
            th = rng.uniform(-np.pi, np.pi, M)
            w = weight_spectrum(anz, mk(n_sys), th)
            nz = [d for d in range(M + 1) if w[d] > 1e-12]
            print("  %-11s N=%d reps=%d  M=%2d   support weights %d..%d, "
                  "peak at %d"
                  % (name, n_sys, reps, M, min(nz), max(nz), int(np.argmax(w))))
            print("      " + "  ".join("w%d:%.3f" % (d, w[d])
                                       for d in range(M + 1) if w[d] > 1e-4))
    print("")
    print("  VERDICT: weight-1 and weight-2 energy is identically zero and the")
    print("  bulk sits near M/2. A DFT bin at frequency c_j therefore reads a")
    print("  coefficient that does not exist. See the docstring for what that")
    print("  settles about full-period designs and about V6's small-R box.")
