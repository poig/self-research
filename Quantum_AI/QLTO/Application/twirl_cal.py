"""Twirl-design device calibration - one compiled circuit, any number of terms.

Learns the coefficients c of an unknown device Hamiltonian H = sum_k c_k P_k
WITHOUT synthesising a model of it. qlto_hl builds e^{+iH(theta)T} on the model
side, which forces a product formula and carries Trotter bias; see that file's
docstring for how badly (P(theta=c_true) falls to 0.14 by T=16). Nothing here is
synthesised, so none of that arises.

THE CONSTRUCTION. A Pauli conjugation flips coefficient signs:

    Q P_k Q^dag = +P_k if [Q,P_k]=0, else -P_k
    ⟹  Q e^{-iHT} Q^dag = exp(-i sum_k sigma_k c_k P_k T)

so a twirl IS a design row, and the DEVICE supplies the evolution exactly in
every branch. Writing Q(a,b) = prod_i X^{a_i} Z^{b_i} and P_k in symplectic form,

    sigma_k(a,b) = (-1)^(a . z_k + b . x_k)

which is a PARITY of the register bits - i.e. the Walsh column for term k is
handed over by P_k's own symplectic vector. Nothing has to be designed.

FULL RANK IS A THEOREM. The columns are Walsh characters indexed by distinct
symplectic vectors, distinct characters are orthogonal, orthogonal implies
independent. So the design matrix has rank M for any set of distinct Paulis, at
any size. See supplement/v101 for the check to M=45.

WHY AN OBSERVABLE AND NOT A HADAMARD TEST. The return probability |<psi|U|psi>|^2
carries no degree-1 signal at order T - the linear part is purely imaginary and
squares away - so recovering it needs a Hadamard test, which needs a CONTROLLED
device evolution. An always-on chip Hamiltonian does not offer one. Measuring an
observable after the twirl does:

    <O>_sigma  ~  <O> + i T sum_k sigma_k c_k <[P_k, O]>
    degree-1 Walsh coefficient in sigma_k  ->  T c_k <i[P_k,O]>

so the device evolution stays uncontrolled and free.

  CORRECTED, supplement/v106. An earlier version of the paragraph above said the
  return probability has "NO first-order term in sigma" full stop. That is wrong:
  its degree-1 Walsh coefficient is nonzero and scales as T^2 (fitted slope 1.986,
  against the observable's 1.088), so it is suppressed, not absent. The conclusion
  is unchanged - T^2 against T is the wrong direction, since small T is where the
  bias is small - but the reason is.

QPE CANNOT BE SUBSTITUTED EITHER, and for a stronger reason than cost. The twirl
acts by CONJUGATION, so H_sigma = Q H Q^dag is ISOSPECTRAL to H at every sigma:
measured worst spectral deviation 1.4e-15 over all 64 register values, and a phase
readout's degree-1 coefficient is 2.1e-16 - machine zero, no signal at any T or
depth. V5's QPE path worked because its register changed the STATE, so energies
moved and a phase carried them; this register changes the Hamiltonian by
conjugation, so eigenvectors rotate and eigenvalues do not. Only a fixed-basis
expectation value sees it. supplement/v106.

THE DESIGN IS CONFOUNDED AT DEGREE 2, and unlike QLTO's register this cannot be
fixed by adding rows. sigma_j sigma_k is itself a Walsh character - the one indexed
by v_j + v_k - so whenever that sum equals some v_m already in the term set, a
degree-2 effect is indistinguishable from term m's degree-1 effect. For crosstalk
v_XX + v_YY = v_ZZ on every bond, and 12 such triples exist at N=3. The general
statement is worse: M distinct symplectic vectors live in GF(2)^2N, so M > 2N
FORCES dependencies, and here M = 4N-3. v101's full-rank theorem is correct and
does not cover this - it proves the M degree-1 columns independent, which says
nothing about a degree-2 product landing on one of them. Consequences: the T^2
bias below is aliasing rather than truncation, Richardson still removes it (it
cancels any T^2 term whatever its origin) but not for the stated reason, and the
columns ARE the Paulis so there is no _design_spec-style lever. Untested cures:
drop one term per triple and infer it, or choose probes nulling the aliased
contribution. supplement/v106.

ONE COMPILED CIRCUIT. The register is measured, so the superposition is doing
the same job as sampling twirls at random - the gain is that it is ONE circuit
structure, compiled and calibrated once, rather than a fresh circuit per design
row. That is the same argument as QLTO's design register, and it is what
"O(1) circuits" means here: circuit COUNT is one per (probe, observable) pair
and does not grow with M, against parameter-shift's 2M distinct circuits.

    register width   2N qubits
    Clifford gates   4N controlled Paulis
    device evolutions 1 per circuit
    circuits         2 * n_probes, independent of N AND of M   (grouped=True)
                     2N * n_probes                             (grouped=False)

The first row is the claim; the second is what this file did until v105 measured
it at exactly 8N for N=3..8. See estimate() for why two bases suffice.

SCOPE. First order in T, so there is a linearity window; supplement/v101
measures 0.13% relative error at T=0.1 on N=4 crosstalk and 3.2% by T=0.5 - both
on EXACT AMPLITUDES, no circuit and no shot noise (tier C under project rule R1).
Probe choice is load-bearing: a probe with <[P_k,O]> = 0 cannot see term k, and
|+..+> misses 10 of 13 crosstalk terms. Use several random product probes.

THE CIRCUIT NUMBERS, SEED-AVERAGED (supplement/v102). The 3.0% once quoted for
T=0.25 at 65536 shots was a single unseeded draw; the seed mean at that
configuration is 6.7% +- 1.0%, and 1.9% +- 0.15% at 524288 shots. The reading that
8x shots bought nothing and the estimator was "bias-limited" there is REFUTED -
the error falls 3.46x, so T=0.25 is shot-limited. That diagnosis does hold at
T >= 0.5 where truncation dominates, which is why it looked right.

device_reps IS A SIMULATION ARTEFACT, not a protocol parameter - on hardware the
evolution is the chip's own and exact. v102 measured the simulated device's
Trotter error at 1.4e-05 relative at reps=12, far below the estimator's own, with
accuracy flat across reps 1..24. But v103 measured reps=12 costing 3.4x under
depolarising noise at p2=1e-3 while reps=1 does not, so prefer 1 in any noisy run.

UNDER NOISE (supplement/v103) the failure mode is the recoverable one: across p2
from 0 to 1e-2 the cosine to c_true moves 0.0045 while the best global scale falls
0.983 -> 0.630. Noise contracts the estimate, it does not rotate it, and at
p2=1e-3 the shape error is statistically identical to noiseless. NOT shown: that
the scale is recoverable without ground truth, which needs its own protocol. NOT
tested: T1/T2 idle decay during the device evolution, the omission most likely to
flatter all of the above.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator


class TwirlCalibrator:
    """Estimate Hamiltonian coefficients from one compiled circuit per probe."""

    def __init__(self, terms, evolution_time=0.1, shots=1 << 15, seed=None,
                 device_reps=12, backend=None):
        self.terms = list(terms)
        self.M = len(self.terms)
        self.N = len(self.terms[0])
        self.T = float(evolution_time)
        self.shots = int(shots)
        self.device_reps = int(device_reps)
        self.backend = backend or AerSimulator(method='statevector')
        self.rng = np.random.default_rng(seed)
        self.ncircuits = 0
        # symplectic vectors: sigma_k(a,b) = (-1)^(a.z_k + b.x_k)
        self._z = np.array([Pauli(t).z.astype(int) for t in self.terms])
        self._x = np.array([Pauli(t).x.astype(int) for t in self.terms])

    # ---- the circuit -----------------------------------------------------
    def _circuit(self, c_true, probe_angles, obs_basis):
        """register |+>, controlled twirl, device evolution, untwirl, measure.

        obs_basis picks the measurement basis per system qubit: 'Z', 'X' or 'Y'.
        The device evolution is UNCONTROLLED - no Hadamard test, so no
        controlled-e^{-iHT} is ever required.
        """
        N, M = self.N, self.M
        reg = QuantumRegister(2 * N, 'r')
        sys = QuantumRegister(N, 's')
        creg = ClassicalRegister(2 * N, 'cr')
        csys = ClassicalRegister(N, 'cs')
        qc = QuantumCircuit(reg, sys, creg, csys)

        qc.h(reg)
        for q, (th, ph) in enumerate(probe_angles):
            qc.u(th, ph, 0.0, sys[q])

        # Q(a,b) = prod_i X^{a_i} Z^{b_i}, controlled off the register
        for i in range(N):
            qc.cx(reg[i], sys[i])
            qc.cz(reg[N + i], sys[i])

        H = SparsePauliOp.from_list(
            list(zip(self.terms, np.asarray(c_true, float)))).simplify()
        qc.append(PauliEvolutionGate(
            H, time=self.T,
            synthesis=SuzukiTrotter(order=2, reps=self.device_reps)), sys)

        for i in range(N):                      # Paulis are self-inverse
            qc.cz(reg[N + i], sys[i])
            qc.cx(reg[i], sys[i])

        for q, b in enumerate(obs_basis):       # rotate into the measured basis
            if b == 'X':
                qc.h(sys[q])
            elif b == 'Y':
                qc.sdg(sys[q])
                qc.h(sys[q])
        qc.measure(reg, creg)
        qc.measure(sys, csys)
        return qc

    # ---- one circuit, decoded --------------------------------------------
    def _walsh(self, c_true, probe_angles, obs_pauli):
        """Degree-1 Walsh coefficients of <O> over the twirl signs."""
        # Qiskit Pauli strings are little-endian: qubit q is obs_pauli[N-1-q]
        n = self.N
        basis = [obs_pauli[n - 1 - q] if obs_pauli[n - 1 - q] in 'XYZ' else 'Z'
                 for q in range(n)]
        support = [q for q in range(n) if obs_pauli[n - 1 - q] in 'XYZ']
        qc = self._circuit(c_true, probe_angles, basis)
        tqc = transpile(qc, self.backend, optimization_level=1)
        counts = self.backend.run(tqc, shots=self.shots).result().get_counts()
        self.ncircuits += 1

        acc = np.zeros(self.M)
        tot = 0
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            sysb, regb = parts[0][::-1], parts[1][::-1]
            o = 1.0
            for q in support:                   # eigenvalue of the Pauli product
                if sysb[q] == '1':
                    o = -o
            a = np.array([int(regb[i]) for i in range(self.N)])
            b = np.array([int(regb[self.N + i]) for i in range(self.N)])
            sig = (-1.0) ** ((self._z @ a + self._x @ b) % 2)
            acc += sig * o * cnt
            tot += cnt
        return acc / max(tot, 1)

    # ---- the estimator ---------------------------------------------------
    def _walsh_basis(self, c_true, probe_angles, basis_letter):
        """ONE circuit; every single-qubit observable in that basis decoded from it.

        This is what makes the circuit count constant. _walsh builds its basis as
        `obs_pauli[n-1-q] if in XYZ else 'Z'`, so the N single-qubit Z observables
        all produce the SAME circuit and differ only in which qubit's parity the
        decode reads. Running them separately spent N circuits to obtain N numbers
        that one set of counts already contains. Here the basis is fixed first and
        every observable diagonal in it is read off the same bitstrings.

        It also improves accuracy at matched total shots, for the same reason the
        design register does: every shot now contributes to N observables'
        marginals instead of one. Measured in supplement/v105 - N=4, 2.10e6 total
        shots either way, mean rel err 0.1238 ungrouped against 0.0814 grouped.
        """
        n = self.N
        qc = self._circuit(c_true, probe_angles, [basis_letter] * n)
        tqc = transpile(qc, self.backend, optimization_level=1)
        counts = self.backend.run(tqc, shots=self.shots).result().get_counts()
        self.ncircuits += 1

        acc = np.zeros((n, self.M))
        tot = 0
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            sysb, regb = parts[0][::-1], parts[1][::-1]
            a = np.array([int(regb[i]) for i in range(n)])
            b = np.array([int(regb[n + i]) for i in range(n)])
            sig = (-1.0) ** ((self._z @ a + self._x @ b) % 2)
            for q in range(n):
                acc[q] += sig * (-1.0 if sysb[q] == '1' else 1.0) * cnt
            tot += cnt

        out = {}
        for q in range(n):
            s = ['I'] * n
            s[n - 1 - q] = basis_letter
            out[''.join(s)] = acc[q] / max(tot, 1)
        return out

    def estimate(self, c_true, n_probes=4, observables=None, probe_seed=0,
                 grouped=True):
        """Direct estimate of every coefficient. No optimisation loop.

        c_true is the DEVICE - it is used only to build the evolution gate that
        the hardware would supply for free, never read by the estimator.

        GROUPED=TRUE IS THE DEFAULT AND COSTS 2*n_probes CIRCUITS, flat in N and
        in M. The ungrouped path costs 2*N*n_probes - measured at exactly 8N in
        supplement/v104 (24, 32, 40, 48, 56, 64 at N=3..8), which is why the
        original 'O(1) circuits' claim did not hold as written. It is kept
        reachable with grouped=False as the comparison arm, not as a fallback.

        TWO BASES SUFFICE FOR ANY PAULI HAMILTONIAN. In basis b the measurable
        observables are the Paulis matching b on a subset, and P_k has a nonzero
        commutator with one of them iff P_k disagrees with b somewhere it is not
        identity. A term invisible to all-Z lies in {I,Z}^N; invisible to all-X
        lies in {I,X}^N; the intersection is the identity alone. Coverage is
        therefore complete at two bases at any N - verified M/M for N=3..8 in
        supplement/v105. What stays family-dependent is n_probes, since
        [P_k,O] != 0 is an operator condition while <i[P_k,O]> != 0 is a
        statement about the probe. v105 measures coverage complete even at one
        probe but the CONDITIONING poor (N=6, one probe: mean rel err 2.58);
        it plateaus by 3-4.
        """
        N, M = self.N, self.M
        pr = np.random.default_rng(probe_seed)
        probes = []
        for _ in range(n_probes):
            probes.append([(float(np.arccos(1 - 2 * pr.random())),
                            float(2 * np.pi * pr.random())) for _ in range(N)])

        num = np.zeros(M)
        den = np.zeros(M)

        def absorb(psi, ob, g):
            resp = self._response(psi, ob)          # <i[P_k,O]>, classical
            w = resp ** 2
            est = np.where(np.abs(resp) > 1e-6,
                           g / (self.T * np.where(np.abs(resp) > 1e-6,
                                                  resp, 1.0)), 0.0)
            return w * est, w

        if grouped and observables is None:
            for ang in probes:
                psi = self._probe_state(ang)
                for letter in ('Z', 'X'):
                    for ob, g in self._walsh_basis(c_true, ang, letter).items():
                        dn, dd = absorb(psi, ob, g)
                        num += dn
                        den += dd
        else:
            if observables is None:
                observables = [''.join('Z' if q == i else 'I' for q in range(N))
                               for i in range(N)]
                observables += [''.join('X' if q == i else 'I' for q in range(N))
                                for i in range(N)]
            for ang in probes:
                psi = self._probe_state(ang)
                for ob in observables:
                    dn, dd = absorb(psi, ob, self._walsh(c_true, ang, ob))
                    num += dn
                    den += dd

        self.coverage = int(np.sum(den > 1e-12))
        return num / np.maximum(den, 1e-30)

    # ---- classical helpers (probe is chosen, so these are known) ---------
    def _probe_state(self, angles):
        # Qiskit is little-endian: qubit 0 is the LAST kron factor
        v = np.array([1.0 + 0j])
        for th, ph in reversed(list(angles)):
            v = np.kron(v, np.array([np.cos(th / 2),
                                     np.exp(1j * ph) * np.sin(th / 2)]))
        return v

    def _response(self, psi, obs_pauli):
        """<i[P_k, O]> for every k - the sensitivity of O to each coefficient."""
        O = Pauli(obs_pauli).to_matrix()
        out = np.empty(self.M)
        for k, t in enumerate(self.terms):
            P = Pauli(t).to_matrix()
            comm = 1j * (P @ O - O @ P)
            out[k] = float(np.real(psi.conj() @ (comm @ psi)))
        return out


def crosstalk_terms(N):
    def put(d):
        s = ['I'] * N
        for i, ch in d.items():
            s[i] = ch
        return ''.join(s)
    t = []
    for i in range(N - 1):
        t += [put({i: 'Z', i + 1: 'Z'}), put({i: 'X', i + 1: 'X'}),
              put({i: 'Y', i + 1: 'Y'})]
    t += [put({i: 'Z'}) for i in range(N)]
    return t


def crosstalk_coeffs(N, seed=7):
    rng = np.random.default_rng(seed)
    c = []
    for _ in range(N - 1):
        c += [rng.uniform(0.01, 0.05), rng.uniform(0.1, 0.3),
              rng.uniform(0.1, 0.3)]
    c += list(rng.uniform(-0.2, 0.2, N))
    return np.round(np.array(c), 4)


if __name__ == '__main__':
    N = 3
    terms = crosstalk_terms(N)
    c = crosstalk_coeffs(N)
    print('TWIRL CALIBRATION on a real Qiskit circuit, N=%d, M=%d' % (N, len(terms)))
    print()
    for T, shots in ((0.1, 1 << 16), (0.2, 1 << 16)):
        cal = TwirlCalibrator(terms, evolution_time=T, shots=shots, seed=0)
        chat = cal.estimate(c, n_probes=3)
        rel = np.abs(chat - c) / np.abs(c)
        print('  T=%.2f shots=%d  circuits=%d  coverage=%d/%d'
              % (T, shots, cal.ncircuits, cal.coverage, len(terms)))
        print('     max rel err %.4f   mean rel err %.4f'
              % (np.max(rel), np.mean(rel)))
        print('     true  ', np.round(c, 3))
        print('     est   ', np.round(chat, 3))
        print()
