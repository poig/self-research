"""QLTO on ZZ+XY tunable-coupler crosstalk: what makes it converge.

The rank criterion (theory_test/heisenberg_rank_criterion.py) passes this model
- rank 13 of 13, every coefficient carries T^2 Fisher information - and the
first attempts still failed, error RISING from 0.0597 to 0.17-1.18. Four things
had to be fixed, three of which were mine.

MODEL. A line of transmons with tunable couplers, N=4:
    ZZ_{i,i+1}  dispersive, always on, 0.010 - 0.035
    XX_{i,i+1}  exchange,             0.16  - 0.28
    YY_{i,i+1}  exchange
    Z_i         detuning,            -0.013 - -0.098
M = 13, non-commuting, with a 27x spread between the weakest ZZ and the
strongest XX. Realistic: dispersive shifts are small by design and exchange is
the thing being tuned.

WHAT WAS ACTUALLY WRONG, in the order the failures were resolved.

  1  THE PROBE.  |+..+> is near an eigenstate for this H and drives cond(F) to
     3.4e17 while the operators themselves sit at cond 64. A random product
     state gives cond(F) ~ 271. See qlto_hl probe='random'.

  2  T TOO SMALL.  At T=1 the interferometer sits at P ~ 0.98, nearly blind.
     Signal grows as T^2||d||^2, so a small residual needs a long evolution.

  3  MODEL INVERSION.  The W-gate must serialise - each theta_k carries its own
     ancilla-controlled sign - so the model is a product formula and does not
     invert the device at theta = c_true. Measured on the sensing circuit:
         T=1  P(c)=0.987     T=4  P(c)=0.334     T=16  P(c)=0.140
     A Trotter ladder in the W-gate repairs this, and the sweep MUST BE
     SYMMETRIC. Second-order Suzuki against first-order Lie at equal reps:
         T=4  reps=4   sym 0.0240   vs  asym 0.0457
         T=8  reps=8   sym 0.0149   vs  asym 0.0872     5.9x
     Holding P ~ 0.999 requires reps proportional to T.

  4  STEP SIZE.  alpha=0.9 with r0=0.5 steps 0.45 rad when the residual is
     0.06 - a 7x overshoot that random-walks the iterate. fit() defaults to
     alpha=0.35 and is fine. An earlier test here used 0.9 and produced a
     0.97 error that was mistaken for an L_infinity geometry failure; the same
     L_infinity rule at alpha~0.3 gives 0.0192, better than L2 normalisation.

PHASE 1, VERIFIED. T=4, reps=8, alpha=0.3, probe='random' with 4 probes,
65536 shots, 40 epochs, 5 seeds:

    seed   start err   final err   ratio
       0      0.0597      0.0192   3.11x
       1      0.0553      0.0231   2.39x
       2      0.0592      0.0255   2.32x
       3      0.0582      0.0132   4.39x
       4      0.0595      0.0181   3.30x
    mean 3.10x, min 2.32x, max 4.39x
    160 circuits against a parameter-shift equivalent of 4160  ->  26x

PHASE 3, THE DEPTH-PRECISION LAW, and it is worse than it first looked.
Holding P ~ 0.999 by setting reps = T:

    T      P(theta=c)    final err
    2        0.9996        0.0247
    4        0.9989        0.0240
    8        0.9992        0.0149
   16        0.9933        0.0115

    4-point fit   err ~ T^-0.400        BELOW the standard quantum limit
    T=4->8 only   err ~ T^-0.688        <- WITHDRAWN, a two-point artefact

Depth scales with reps, so err ~ depth^-0.40: halving the error costs about
5.7x the circuit depth. The T=16 reps=16 circuit is 416 controlled evolutions
per gradient. The ladder is correct and effective and it reinstates exactly the
depth cost the single-stage design existed to avoid - the same shape as the
filtered-jump repair in the feedback manuscript, which buys directional cooling
at ~32 Hamiltonian evolutions per cycle.

PHASE 2, PEELING. Allocate the design register to a subset and apply the rest
unperturbed. It does NOT reduce Trotter error - all M terms still appear in the
product either way, measured P(theta=c) = 0.99925 monolithic against 0.99930
with the design on ZZ alone. What it buys is R_eff = R/sqrt(M_active), so
splitting 13 into 10 and 3 gives the weak group a 2.1x larger radius and the
whole shot budget.

Alternating A-B-C-D over (XYZ,T=4), (ZZ,T=4), (XYZ,T=8), (ZZ,T=8), 10 epochs
each, against monolithic at the same 40 circuits:

THE RADIUS MUST CARRY ACROSS STAGES, and that is the only firm finding here.
Restarting r at r0 each stage throws away the refinement four times over:

    scheme                  err ALL    err ZZ
    bootstrap, r RESET       0.3585    0.0917     ~20x worse than monolithic
    bootstrap, r CARRIED     0.0370    0.0158

PEELING ITSELF SHOWS NO MEASURED ADVANTAGE. Six seeds, matched 40 circuits:

    seed     mono ALL   boot ALL    mono ZZ    boot ZZ
       0       0.0230     0.0259     0.0230     0.0065
       1       0.0978     0.0404     0.0860     0.0285
       2       0.0181     0.0222     0.0095     0.0138
       3       0.0221     0.0638     0.0209     0.0480
       4       0.0225     0.0426     0.0121     0.0213
       5       0.0203     0.0513     0.0203     0.0465
    mean       0.0340     0.0410     0.0286     0.0274
     std       0.0286     0.0142     0.0261     0.0155

The ZZ means are indistinguishable (0.0286 vs 0.0274, both std ~0.02) and the
bootstrap wins on ZZ in only 2 of 6 seeds, while losing on overall error. A
single earlier run showed 0.0051 against 0.0164 and was written up here as a
3.2x win on the weak terms; it did not replicate and is WITHDRAWN. The R_eff
argument - splitting 13 into 10 and 3 gives the weak group a 2.1x larger radius
- remains theoretically sound and is not visible at this size and budget.

What the six seeds do lean toward is VARIANCE, not accuracy: the bootstrap runs
at about half the spread of monolithic on both metrics. Six seeds cannot
establish that either, and it is recorded as an observation, not a result.

WHAT IS NOT ESTABLISHED. One Hamiltonian family at N=4, noiseless simulation,
one initial-error scale. Nothing here has seen T1/T2 decay or readout error,
which cap the usable T and would bite the deep end of the Phase 3 curve first.
The rank criterion ruled this model IN at N=4 and OUT at N=3 (rank 6 of 9), so
the pre-flight works, but d=0 says the information is present and says nothing
about whether an estimator can reach it - which is what the four fixes above
were needed for.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from nisq_v6 import _design_spec, _design_sign
from qlto_hl import QLTOHamiltonianLearner


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


def crosstalk_coeffs(N, seed):
    rng = np.random.default_rng(seed)
    c = []
    for _ in range(N - 1):
        c += [rng.uniform(0.01, 0.05), rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)]
    c += list(rng.uniform(-0.2, 0.2, N))
    return np.round(np.array(c), 4)


class SuzukiPeelHL(QLTOHamiltonianLearner):
    """Symmetric second-order W-gate; design register on `active` only."""

    def __init__(self, *a, trotter_reps=4, device_reps=12, active=None, **k):
        super().__init__(*a, **k)
        self.trotter_reps = int(trotter_reps)
        self.device_reps = int(device_reps)
        self.active = list(range(self.M)) if active is None else list(active)

    def _sense_circuit(self, c_true, centre, R, pi=0):
        A = self.active
        Ma = len(A)
        m_row, cols = _design_spec(Ma, self.n_scratch, self.design_resolution)
        nreg = m_row + 1
        ns = max(1, min(self.n_scratch, Ma))
        param = QuantumRegister(nreg, 'p')
        sysr = QuantumRegister(self.N, 's')
        scr = QuantumRegister(ns, 'a')
        qc = QuantumCircuit(param, sysr, scr,
                            ClassicalRegister(nreg, 'cp'),
                            ClassicalRegister(self.N, 'cs'))
        qc.h(param)
        self._prep(qc, sysr, pi)
        qc.append(PauliEvolutionGate(self._H(c_true), time=self.T,
                                     synthesis=SuzukiTrotter(
                                         order=2, reps=self.device_reps)), sysr)
        K = self.trotter_reps

        def sweep(order):
            for i in order:
                if i in A:
                    j = A.index(i)
                    s = j % ns
                    qc.append(PauliEvolutionGate(
                        self._paulis[i],
                        time=-(centre[i] + R) * self.T * 0.5 / K), sysr)
                    for b in range(m_row):
                        if (cols[j] >> b) & 1:
                            qc.cx(param[b], scr[s])
                    qc.cx(param[m_row], scr[s])
                    qc.append(PauliEvolutionGate(
                        self._paulis[i],
                        time=+2.0 * R * self.T * 0.5 / K).control(1),
                        [scr[s]] + list(sysr))
                    qc.cx(param[m_row], scr[s])
                    for b in range(m_row):
                        if (cols[j] >> b) & 1:
                            qc.cx(param[b], scr[s])
                else:
                    qc.append(PauliEvolutionGate(
                        self._paulis[i],
                        time=-centre[i] * self.T * 0.5 / K), sysr)

        for _ in range(K):
            sweep(range(self.M))
            sweep(range(self.M - 1, -1, -1))
        self._prep(qc, sysr, pi, inverse=True)
        qc.measure(param, qc.cregs[0])
        qc.measure(sysr, qc.cregs[1])
        return qc, m_row, cols, 1

    def gradient(self, c_true, centre, R, rescale=True):
        A = self.active
        Reff = R * (1.0 / np.sqrt(max(len(A), 1)) if rescale else 1.0)
        qc, m_row, cols, fold = self._sense_circuit(c_true, centre, Reff)
        tqc = transpile(qc, self.backend, optimization_level=1)
        counts = self.backend.run(tqc, shots=self.shots).result().get_counts()
        self.nefv += 1
        tot, acc = 0, np.zeros(len(A))
        for bs, cnt in counts.items():
            parts = bs.split()
            if len(parts) != 2:
                continue
            sysb, parb = parts[0], parts[1]
            tot += cnt
            if set(sysb) != {'0'}:
                continue
            xb = parb[::-1]
            row = sum(1 << b for b in range(m_row)
                      if b < len(xb) and xb[b] == '1')
            f = 1 if (m_row < len(xb) and xb[m_row] == '1') else 0
            acc += np.array([_design_sign(row, f, cols[j])
                             for j in range(len(A))]) * cnt
        g = np.zeros(self.M)
        g[A] = acc / max(tot, 1) / Reff
        return g


def descend(q, c, th, epochs, r, alpha=0.3, rd=0.93):
    """Returns (theta, final radius) - the radius MUST carry across stages."""
    for _ in range(epochs):
        g = q.gradient(c, th, r)
        mx = float(np.max(np.abs(g)))
        if mx > 1e-12:
            th = th + alpha * r * g / mx
        r = max(r * rd, 1e-4)
    return th, r


def main():
    N = 4
    terms = crosstalk_terms(N)
    M = len(terms)
    ZZ = list(range(0, 9, 3))
    XYZ = [i for i in range(M) if i not in ZZ]

    print('=' * 78)
    print('v100  QLTO on ZZ+XY crosstalk: monolithic vs alternating bootstrap')
    print('=' * 78)
    print('  N=4, M=13, 27x coefficient spread. T=4, symmetric Suzuki reps=4,')
    print('  alpha=0.3, probe=random(4), 65536 shots, 40 circuits each arm.')
    print()

    c = crosstalk_coeffs(N, 7)
    th0 = c + np.random.default_rng(70).uniform(-0.06, 0.06, M)
    print('  start   err ALL %.4f   err ZZ %.4f'
          % (float(np.max(np.abs(th0 - c))),
             float(np.max(np.abs((th0 - c)[ZZ])))))
    print()
    print('  %-24s %11s %11s %9s' % ('scheme', 'err ALL', 'err ZZ', 'circuits'))
    print('  ' + '-' * 58)

    q = SuzukiPeelHL(terms, evolution_time=4.0, shots=65536, seed=0,
                     probe='random', n_probes=4, trotter_reps=4, active=None)
    n0 = q.nefv
    th, _ = descend(q, c, th0.copy(), 40, 0.5)
    print('  %-24s %11.4f %11.4f %9d'
          % ('monolithic M=13', float(np.max(np.abs(th - c))),
             float(np.max(np.abs((th - c)[ZZ]))), q.nefv - n0))

    for label, carry in (('bootstrap, r RESET', False),
                         ('bootstrap, r CARRIED', True)):
        th = th0.copy()
        tot = 0
        r = 0.5
        for act, T, ep in [(XYZ, 4.0, 10), (ZZ, 4.0, 10),
                           (XYZ, 8.0, 10), (ZZ, 8.0, 10)]:
            q = SuzukiPeelHL(terms, evolution_time=T, shots=65536, seed=0,
                             probe='random', n_probes=4,
                             trotter_reps=int(T), active=act)
            n0 = q.nefv
            th, rend = descend(q, c, th, ep, r if carry else 0.5)
            if carry:
                r = rend
            tot += q.nefv - n0
        print('  %-24s %11.4f %11.4f %9d'
              % (label, float(np.max(np.abs(th - c))),
                 float(np.max(np.abs((th - c)[ZZ]))), tot))
    print()
    print('  The radius must carry across stages - resetting it is ~20x worse.')
    print('  Peeling itself: no measured advantage over 6 seeds (see docstring).')


if __name__ == '__main__':
    main()
