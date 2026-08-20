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
has NO first-order term in sigma - the linear part is purely imaginary and
squares away - so recovering it needs a Hadamard test, which needs a CONTROLLED
device evolution. An always-on chip Hamiltonian does not offer one. Measuring an
observable after the twirl does:

    <O>_sigma  ~  <O> + i T sum_k sigma_k c_k <[P_k, O]>
    degree-1 Walsh coefficient in sigma_k  ->  T c_k <i[P_k,O]>

so the device evolution stays uncontrolled and free.

ONE COMPILED CIRCUIT. The register is measured, so the superposition is doing
the same job as sampling twirls at random - the gain is that it is ONE circuit
structure, compiled and calibrated once, rather than a fresh circuit per design
row. That is the same argument as QLTO's design register, and it is what
"O(1) circuits" means here: circuit COUNT is one per (probe, observable) pair
and does not grow with M, against parameter-shift's 2M distinct circuits.

    register width   2N qubits
    Clifford gates   4N controlled Paulis
    device evolutions 1 per circuit
    circuits         n_probes * n_observables, INDEPENDENT of M

SCOPE. First order in T, so there is a linearity window; supplement/v101
measures 0.13% relative error at T=0.1 on N=4 crosstalk and 3.2% by T=0.5.
Probe choice is load-bearing: a probe with <[P_k,O]> = 0 cannot see term k, and
|+..+> misses 10 of 13 crosstalk terms. Use several random product probes.
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
    def estimate(self, c_true, n_probes=4, observables=None, probe_seed=0):
        """Direct estimate of every coefficient. No optimisation loop.

        c_true is the DEVICE - it is used only to build the evolution gate that
        the hardware would supply for free, never read by the estimator.
        """
        N, M = self.N, self.M
        if observables is None:
            observables = [''.join('Z' if q == i else 'I' for q in range(N))
                           for i in range(N)]
            observables += [''.join('X' if q == i else 'I' for q in range(N))
                            for i in range(N)]
        pr = np.random.default_rng(probe_seed)
        probes = []
        for _ in range(n_probes):
            probes.append([(float(np.arccos(1 - 2 * pr.random())),
                            float(2 * np.pi * pr.random())) for _ in range(N)])

        num = np.zeros(M)
        den = np.zeros(M)
        for ang in probes:
            psi = self._probe_state(ang)
            for ob in observables:
                resp = self._response(psi, ob)      # <i[P_k,O]>, classical
                g = self._walsh(c_true, ang, ob)
                w = resp ** 2
                est = np.where(np.abs(resp) > 1e-6,
                               g / (self.T * np.where(np.abs(resp) > 1e-6,
                                                      resp, 1.0)), 0.0)
                num += w * est
                den += w
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
