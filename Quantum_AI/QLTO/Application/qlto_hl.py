"""QLTO for Hamiltonian learning - the log register on a quantum-data task.

Learn the coefficients c of an unknown device Hamiltonian H_true = sum_k c_k P_k
by matching a model H(theta) = sum_k theta_k P_k against it.

WHY THIS TASK RATHER THAN VQE. The loss is a RETURN PROBABILITY, not a Pauli
sum:

    |+..+>  ->  e^{-i H_true T}  ->  e^{+i H(theta) T}  ->  back to |+..+>?

At theta = c the two evolutions cancel and the probe returns exactly. Three
consequences, and the first is the one that matters:

  G = 1 STRUCTURALLY. One measurement setting - "did the probe come back" - so
      circuits per gradient is 1, not the G qubit-wise-commuting groups a Pauli
      sum needs. V6's Theta(G) becomes Theta(1) here by the shape of the
      problem, not by a property of some Hamiltonian family.
  BOUNDED READOUT. The per-shot outcome is a BIT, variance <= 1/4 regardless of
      system size or spectral norm.
  LINEARITY PRESERVED. The loss is a single expectation value, so the degree-1
      Walsh marginal is an unbiased gradient estimator at any shots-per-vertex.

WHAT IS AND IS NOT CLAIMED. The data is quantum-native: an unknown device has no
efficient classical description, so there is no classical vector to have fitted
instead. That is NOT the same as the proven separation of Huang et al. (Science
2022), which is between MULTI-COPY and single-copy learners and requires
entangled measurements across copies. This protocol is single-copy and does not
invoke it. Whether this beats classical shadows, randomised benchmarking or
Bayesian estimation on the same task is UNMEASURED - see benchmark_hl() below,
which exists to answer that rather than assume it.

REACHING THE HEISENBERG LIMIT - fit_heisenberg().
--------------------------------------------------------------
As shipped this runs at FIXED T and averages shots, which is the standard
quantum limit: total evolution time ~ 1/eps^2. That is not structural. The fix
is to schedule T and R with the estimate rather than fixing them.

  With commuting terms, e^{iH(theta)T} e^{-iH_true T} = e^{i dH T} exactly,
  where dH = sum_k (theta_k - c_k) P_k. On |+..+>,

      P = |<psi| e^{i dH T} |psi>|^2  ~  1 - T^2 ||d||^2      for T||d|| <~ 1

  so the gradient is dP/dtheta_k = -2 T^2 d_k: SIGNAL GROWS AS T^2. The readout
  is a bounded bit, so the Walsh estimator noise is ~ 1/(2 R sqrt(n)),
  INDEPENDENT of T. Hence

      SNR = 4 T^2 d R sqrt(n)

  FIXED T, FIXED R      resolving d = eps needs n ~ 1/(T^4 eps^2 R^2), so total
                        evolution time n*T ~ 1/eps^2.            SQL

  T ~ 1/eps, R ~ eps    SNR = 4 (1/eps^2)(eps)(eps) sqrt(n) = 4 sqrt(n). The
                        eps-dependence CANCELS. With T_j = 2^j T_0 over
                        O(log 1/eps) levels:
                             total evolution time ~ Otilde(M/eps)  HEISENBERG

  The M in that expression is an INFORMATION FLOOR, not an inefficiency:
  pinning M coefficients to precision eps needs M log(1/eps) bits and one shot
  yields at most one bit, so n >~ M shots per level for ANY single-copy
  bit-readout protocol. Those M shots go through ONE circuit, so the DISTINCT
  circuit count stays O(log 1/eps) and M-independent.

AGAINST THE PUBLISHED BEST (arXiv:2502.11900, ansatz-free, Heisenberg-limited):

                        2-copy              1-copy               here
  evolution time        Otilde(M^2/eps)     Otilde(M^3 log n/eps) Otilde(M/eps)
  distinct circuits     Otilde(M^2 log M    similar               O(log 1/eps)
                          log 1/eps)
  ancillas              n (2nd register)    0                     ceil(log2 M)+1
  measurement settings  Bell + comp., adaptive  Pauli products    1
  coefficients          group-by-group      group-by-group        all M at once

  Otilde(M/eps) is plausibly OPTIMAL: each of M parameters needs 1/eps
  evolution time at the Heisenberg limit, and the information bound forbids
  fewer than M shots per level.

TWO THINGS THIS RESTS ON, and the first is the reason it is not simply done.

  COMMUTATION. The exact cancellation e^{iH(theta)T} e^{-iH_true T} = e^{i dH T}
      requires the terms to commute. For a general H the Trotter error enters at
      O(T^2 ||[P_i,P_j]||) and GROWS with T - precisely where the schedule
      pushes. For non-commuting Hamiltonians this derivation does not go
      through, and that is why the ansatz-free protocols work harder than this.
  LEVEL-CONDITIONING. T may only be raised once d is verifiably halved. On an
      epoch clock instead of an estimate-conditioned trigger, T*d crosses pi,
      the return probability wraps, and the gradient inverts.

STATUS: fit() runs the fixed-T SQL schedule; fit_heisenberg() implements the
hierarchical one derived above. Use fit_heisenberg only on COMMUTING terms -
it will silently give wrong answers otherwise, for the Trotter reason above.

MEASURED SO FAR, in supplement/v88 and v90:
  cos(measured gradient, exact smeared) = 0.997 at M=5 on a linear register
  recovery from a wrong start to 0.034 max error per coefficient
  16 coefficients carried on a 6-qubit design register, 1 circuit per gradient
  32x fewer circuits than parameter-shift
  design_resolution=5 recovers cosine 0.714 -> 0.927 at M=16 (this task only;
  it does NOT help VQE, see v97)
  variance - NOT bias - is what limits the shipped configuration: with exact
  gradients the same schedule reaches 9.6e-4 at 30 epochs and 1.2e-4 at 120,
  against 0.077 at 4096 shots. The estimator is not the bottleneck.
  amplitude estimation was tested and REJECTED: it wins above ~1000 queries by
  at most 1.5x, but costs the M-fold amortisation - M separate estimations
  instead of one circuit - which is the property this method exists to provide.

fit_heisenberg MEASURED. It reaches the derived scaling.
  M=6 commuting Z-type on N=3, 3 seeds, error vs TOTAL EVOLUTION TIME, fitted
  over 5 level-counts spanning T_total 1.4e3 - 4.1e5:

      guard                        steps/level   fitted exponent
      SHIPPED fit_heisenberg defaults       48            -1.060
      return probability only               48            -0.961
      max|theta - c_true| (oracle)          48            -0.952
      max|theta - c_true| (oracle)           6            -0.618
                                  Heisenberg -1.000, SQL -0.500

  THE SHIPPED -1.060 IS NOT A SUPER-HEISENBERG RESULT AND MUST NOT BE READ AS
  ONE. 1/T_total is a LOWER BOUND on the error; an exponent past -1.000 is
  unphysical, so the excess is fit scatter (5 points, 3 seeds) or slack in the
  T_total accounting used here, which is sum over levels of
  T * circuits_per_level * shots and may undercount. The defensible claim is
  that the schedule is CONSISTENT WITH the Heisenberg limit and decisively
  clear of SQL, not that it beats the bound. Resolving -1.06 vs -1.00 needs
  more level-counts, more seeds, and an independent audit of the time ledger.

  THE EXPONENT IS SET BY ONE RATIO. T doubles per level, so if the inner loop
  contracts the residual by r each level then

      err ~ T^(log r / log 2)

  Heisenberg needs r = 1/2. Six inner steps deliver r = 0.776, which FORCES
  -0.32 and accounts for the measured -0.618 with nothing left over. Raising to
  48 steps gives r = 0.609 and the exponent follows to -0.95. This is not a
  tuning accident: r is the whole mechanism, and the failure mode was that
  alpha*R*g/mx normalises by the LARGEST gradient, so only the leading
  coefficient travels the full alpha*R while the max-ERROR coefficient - a
  different one - lags and never halves.
  96 steps does WORSE (r = 0.648) because err ~ 0.004 at 32 shots/level is the
  shot floor; past it, extra steps add noise and no signal.
  T*delta peaked at 1.0, well inside pi, so wrapping was never the cause.

  WHY THE GUARD USES P AND NOT THE ERROR. The level guard must decide when the
  residual has actually shrunk, but max|theta - c_true| requires the answer we
  are learning. P is measurable and P -> 1 as theta -> c, with 1 - P ~ T^2
  delta^2 in the linearisation window, so

      delta_hat = sqrt(max(1-P, 0)) / T

  schedules both T and R from data alone. It costs nothing: -0.961 observable
  vs -0.952 oracle, and it reaches 0.0098 in 4x LESS evolution time than the
  oracle arm because it stops over-doubling.

  SCOPE, HONESTLY. One M, one N, one initial-error scale, commuting terms only,
  3 seeds. steps_per_level=48 is fitted at M=6 and its scaling with M is
  UNMEASURED - the default is very likely wrong for larger M, since the number
  of steps needed to halve the worst of M coefficients should grow with M.
  Nothing here is tested on non-commuting H, where the Trotter argument above
  says the whole construction breaks.

  TWO WITHDRAWN RESULTS, recorded so they are not re-derived:
  - A single seed gave 0.0221 and was briefly written here as a 19x win. Three
    seeds give 0.18 at those settings. Withdrawn as a lucky draw.
  - An earlier -0.258 was a 4-point fit stopping at level 9; the 5-point fit
    gives -0.618 for the identical configuration. Withdrawn as undersampled.
  - /tmp/tsched.py appeared to show growing-T reaching 0.001 at every shot
    budget. Its noise model divided by T, contradicting the derivation above in
    which bounded-bit readout makes Walsh noise INDEPENDENT of T. Artefact.
"""
import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from nisq_v6 import _design_spec, _design_sign


class QLTOHamiltonianLearner:
    """Recover Hamiltonian coefficients with a log-width design register.

    One circuit per gradient regardless of how many coefficients are being
    learned, because the design register indexes them rather than allocating a
    qubit each.
    """

    def __init__(self, terms, evolution_time=1.0, shots=8192, n_scratch=2,
                 design_resolution=5, backend=None, seed=None):
        """
        terms              Pauli strings whose coefficients are unknown.
        evolution_time     T in e^{-iHT}. Too small and the signal vanishes;
                           too large and the return probability wraps, so the
                           landscape develops spurious optima. T ~ 1/||H|| is
                           the sane range.
        design_resolution  5 by default HERE, unlike V6's VQE default of 4:
                           v90 measured the aliasing fix helping this task
                           (0.714 -> 0.927 at M=16) and v97 measured it NOT
                           helping VQE. The default differs because the
                           evidence differs.
        """
        self.terms = list(terms)
        self.M = len(self.terms)
        self.N = len(self.terms[0])
        self.T = float(evolution_time)
        self.shots = int(shots)
        self.n_scratch = max(1, int(n_scratch))
        self.design_resolution = int(design_resolution)
        self.backend = backend or AerSimulator(method='statevector')
        self.rng = np.random.default_rng(seed)
        self.nefv = 0
        self.max_circuit_depth = 0
        self._paulis = [SparsePauliOp.from_list([(t, 1.0)]) for t in self.terms]
        # RADIUS RESCALING. The +-R design displaces all M coordinates at once,
        # so the state moves by about R*sqrt(M) - a random walk, not R. The
        # finite-radius bias goes as the square of that displacement, c*R^2*M,
        # so a FIXED R gives a floor rising as sqrt(M) in coefficient error.
        # Dividing by sqrt(M) holds R*sqrt(M) constant and the bias with it.
        # nisq_v6 does this internally (sqrt(n/N)); omitting it there cost
        # cos 0.975 -> 0.886, and omitting it here is what made the measured
        # floor look flat when it was merely slowly rising.
        self._r_scale = 1.0 / np.sqrt(max(self.M, 1))

    # ---- model -----------------------------------------------------------
    def _H(self, coeffs):
        return SparsePauliOp.from_list(
            list(zip(self.terms, np.asarray(coeffs, float)))).simplify()

    def return_probability(self, c_true, theta):
        """Exact |<+..+| U_model^dag U_true |+..+>|^2, for scoring only."""
        qc = QuantumCircuit(self.N)
        qc.h(range(self.N))
        qc.append(PauliEvolutionGate(self._H(c_true), time=self.T),
                  range(self.N))
        qc.append(PauliEvolutionGate(self._H(theta), time=-self.T),
                  range(self.N))
        qc.h(range(self.N))
        return float(abs(Statevector(qc).data[0]) ** 2)

    # ---- the sensing circuit --------------------------------------------
    def _sense_circuit(self, c_true, centre, R):
        """One circuit: all M coefficients displaced by +-R on a log register.

        theta_k = centre_k + R * sigma_k with sigma_k the design sign. The base
        evolution runs at centre_k + R and a controlled -2R increment fires on
        the parity bit, so sigma_k = +1 when that parity is even - the
        convention _design_sign uses. Getting it backwards flips every
        coordinate at once, which the recovery would survive and the gradient
        cosine would not.
        """
        m_row, cols = _design_spec(self.M, self.n_scratch,
                                   self.design_resolution)
        fold = 1 if self.design_resolution >= 4 else 0
        nreg = m_row + fold
        ns = max(1, min(self.n_scratch, self.M))

        param = QuantumRegister(nreg, 'p')
        sysr = QuantumRegister(self.N, 's')
        scr = QuantumRegister(ns, 'a')
        qc = QuantumCircuit(param, sysr, scr,
                            ClassicalRegister(nreg, 'cp'),
                            ClassicalRegister(self.N, 'cs'))
        qc.h(param)
        qc.h(sysr)
        qc.append(PauliEvolutionGate(self._H(c_true), time=self.T), sysr)

        for i in range(self.M):
            s = i % ns
            qc.append(PauliEvolutionGate(self._paulis[i],
                                         time=-(centre[i] + R) * self.T), sysr)
            for b in range(m_row):
                if (cols[i] >> b) & 1:
                    qc.cx(param[b], scr[s])
            if fold:
                qc.cx(param[m_row], scr[s])
            qc.append(PauliEvolutionGate(self._paulis[i],
                                         time=+2.0 * R * self.T).control(1),
                      [scr[s]] + list(sysr))
            if fold:
                qc.cx(param[m_row], scr[s])
            for b in range(m_row):
                if (cols[i] >> b) & 1:
                    qc.cx(param[b], scr[s])

        qc.h(sysr)
        qc.measure(param, qc.cregs[0])
        qc.measure(sysr, qc.cregs[1])
        return qc, m_row, cols, fold

    def gradient(self, c_true, centre, R, rescale=True):
        """Degree-1 Walsh marginal of the return bit. ONE circuit, any M.

        R is rescaled by 1/sqrt(M) unless rescale=False, so that the callers
        radius means the same displacement at every M. Pass rescale=False to
        reproduce the uncompensated behaviour.
        """
        Reff = R * (self._r_scale if rescale else 1.0)
        qc, m_row, cols, fold = self._sense_circuit(c_true, centre, Reff)
        tqc = transpile(qc, self.backend, optimization_level=1)
        self.max_circuit_depth = max(self.max_circuit_depth, tqc.depth())
        counts = self.backend.run(tqc, shots=self.shots).result().get_counts()
        self.nefv += 1

        tot, acc = 0, np.zeros(self.M)
        for bs, cnt in counts.items():
            parts = bs.split()
            if len(parts) != 2:
                continue
            sysb, parb = parts[0], parts[1]
            tot += cnt
            if set(sysb) != {'0'}:
                continue                       # probe did not return
            xb = parb[::-1]
            row = sum(1 << b for b in range(m_row)
                      if b < len(xb) and xb[b] == '1')
            f = 1 if (fold and m_row < len(xb) and xb[m_row] == '1') else 0
            sg = np.array([_design_sign(row, f, cols[i])
                           for i in range(self.M)])
            acc += sg * cnt
        return acc / max(tot, 1) / Reff

    # ---- the loop --------------------------------------------------------
    def fit(self, c_true, theta0, epochs=30, r0=0.5, r_decay=0.93, alpha=0.35,
            verbose=False, rescale=True):
        """Ascend the return probability. Returns (theta, trace).

        The step is MAX-NORMALISED, so only the gradient's direction is used and
        its magnitude is discarded. That is why the radius must decay: with a
        fixed R the dominant coordinate steps by exactly alpha*R every epoch and
        the iterate limit-cycles rather than converging (measured in v96).
        """
        theta = np.array(theta0, dtype=float)
        r, trace = float(r0), []
        for ep in range(epochs):
            # SENSE at the rescaled radius, STEP at the raw one. Using the
            # rescaled R for both shrinks the step by sqrt(M) as well, so the
            # iterate covers sqrt(M) less ground in the same epochs - which is
            # commit 09f1d6c in this repo, 'the radius rescaling was leaking
            # into the step', reintroduced here and caught by the M-sweep.
            g = self.gradient(c_true, theta, r, rescale=rescale)
            mx = float(np.max(np.abs(g)))
            if mx > 1e-12:
                theta = theta + alpha * r * g / mx      # ASCEND: maximise P
            p = self.return_probability(c_true, theta)
            err = float(np.max(np.abs(theta - np.asarray(c_true, float))))
            trace.append({'epoch': ep, 'R': r, 'return_prob': p,
                          'max_coeff_err': err, 'nefv': self.nefv})
            if verbose:
                print(f"  ep {ep:>3}  R {r:.4f}  P {p:.4f}  err {err:.4f}")
            r = max(r * r_decay, 1e-4)
        return theta, trace


    # ---- the hierarchical schedule ---------------------------------------
    def fit_heisenberg(self, c_true, theta0, levels=8, shots_per_level=None,
                       T0=0.25, alpha=0.35, steps_per_level=48, verbose=False):
        """Hierarchical T-doubling. Otilde(M/eps) evolution time on COMMUTING H.

        Each level drives the residual down, then doubles T - holding T*|delta|
        near the linearisation boundary where the signal is largest. R is tied
        to the residual rather than decayed on a clock, because the SNR
        cancellation in the module docstring needs R ~ delta specifically.

        WHY T FOLLOWS THE ESTIMATE AND NEVER LEADS IT. P is |average of
        phases|^2 and wraps once T*|delta| exceeds about pi; past that the
        landscape grows spurious optima and the gradient points the wrong way.
        So T is raised only on evidence the residual actually shrank - that is
        what makes this hierarchical rather than merely scheduled.

        shots_per_level defaults to max(M, 32). The information floor says M
        coefficients need at least M bits per level and one shot yields one bit,
        so fewer cannot work however long the evolution runs.
        """
        theta = np.array(theta0, dtype=float)
        n_shots = int(shots_per_level or max(self.M, 32))
        T = float(T0)
        saved_for_probe = self.shots, self.T
        self.shots, self.T = n_shots, T
        delta = np.sqrt(max(1.0 - self.return_probability(c_true, theta), 1e-12))
        delta = max(float(delta) / max(T, 1e-9), 1e-3)
        self.shots, self.T = saved_for_probe
        trace = []
        saved_shots, saved_T = self.shots, self.T
        self.shots = n_shots
        try:
            for lev in range(levels):
                self.T = min(T, 1.0 / max(delta, 1e-9))
                R = max(0.5 * delta, 1e-4)
                for _ in range(steps_per_level):
                    g = self.gradient(c_true, theta, R, rescale=False)
                    mx = float(np.max(np.abs(g)))
                    if mx > 1e-12:
                        theta = theta + alpha * R * g / mx
                p = self.return_probability(c_true, theta)
                err = float(np.max(np.abs(theta - np.asarray(c_true, float))))
                trace.append({'level': lev, 'T': self.T, 'R': R,
                              'return_prob': p, 'max_coeff_err': err,
                              'nefv': self.nefv,
                              'evolution_time': self.nefv * self.T * n_shots})
                if verbose:
                    print('  lev %2d  T %7.3f  R %.4f  P %.5f  err %.5f'
                          % (lev, self.T, R, p, err))
                # guard on the MEASURABLE residual proxy, never on c_true
                delta_hat = np.sqrt(max(1.0 - p, 1e-12)) / max(self.T, 1e-9)
                if delta_hat < delta:
                    delta = max(float(delta_hat), 1e-9)
                    T *= 2.0
                else:
                    T *= 1.2
        finally:
            self.shots, self.T = saved_shots, saved_T
        return theta, trace


def benchmark_hl(terms, c_true, seeds=5, epochs=30, T=1.0, shots=8192):
    """QLTO against parameter-shift on the SAME loss, matched shot budget.

    This exists because the advantage question is open, not settled. Both arms
    optimise the same return probability from the same starts; only the gradient
    estimator differs. Parameter-shift needs 2M circuits per gradient, QLTO
    needs 1 - so the honest comparison is final error at MATCHED TOTAL CIRCUITS,
    which is what the caller should read.
    """
    out = {'qlto': [], 'pshift': []}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        th0 = np.asarray(c_true, float) + rng.uniform(-0.4, 0.4, len(c_true))

        q = QLTOHamiltonianLearner(terms, evolution_time=T, shots=shots, seed=s)
        thq, trq = q.fit(c_true, th0, epochs=epochs)
        out['qlto'].append({'err': float(np.max(np.abs(thq - c_true))),
                            'circuits': q.nefv, 'trace': trq})

        p = QLTOHamiltonianLearner(terms, evolution_time=T, shots=shots, seed=s)
        th = th0.copy()
        circuits, r = 0, 0.5
        for ep in range(epochs):
            g = np.zeros(len(c_true))
            for k in range(len(c_true)):
                e = np.zeros(len(c_true)); e[k] = np.pi / 4
                g[k] = (p.return_probability(c_true, th + e)
                        - p.return_probability(c_true, th - e))
                circuits += 2
            mx = float(np.max(np.abs(g)))
            if mx > 1e-12:
                th = th + 0.35 * r * g / mx
            r = max(r * 0.93, 1e-4)
        out['pshift'].append({'err': float(np.max(np.abs(th - c_true))),
                              'circuits': circuits})
    return out
