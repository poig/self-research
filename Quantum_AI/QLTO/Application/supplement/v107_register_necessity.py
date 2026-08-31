"""Is the 2N-qubit twirl register doing any WORK, or only saving compilation?

The single largest gap between this construction and the control-free frontier is
qubits. arXiv:2606.19486 proves Theta(Lambda/eps^2 log(Lambda/eps)) OPTIMAL for
control-free Hamiltonian learning using ZERO ancillas; twirl_cal spends 2N on the
twirl register, so a 100-qubit chip would need 300 qubits.

twirl_cal's own docstring already concedes the mechanism:

    "the register is measured, so the superposition is doing the same job as
     sampling twirls at random - the gain is that it is ONE circuit structure,
     compiled and calibrated once, rather than a fresh circuit per design row"

If that is the whole story, the register buys COMPILATION CONVENIENCE and not
INFORMATION, and an ancilla-free version exists: choose a Pauli frame Q classically,
apply it, evolve, un-apply, measure. Regress the same degree-1 Walsh coefficients
over the frames you chose. Zero ancillas, same estimator, same shots.

WHY THIS MATTERS MORE THAN IT SOUNDS. On real hardware a Pauli frame is FREE -
randomised compiling absorbs it into the classical control, no extra gates - so the
"one compiled circuit" advantage is worth much less on a device than in a
simulator. And if the ancilla-free arm matches, then:

    - the qubit gap to the frontier protocol CLOSES (3N -> N)
    - the register's idle-decoherence exposure, the biggest untested risk in v103,
      disappears entirely because there is no register to decohere
    - "O(1) circuits" becomes "O(1) circuit STRUCTURES", which is a weaker but
      still real claim, since every frame is the same template

THE TWO ARMS DIFFER AS EXPERIMENT DESIGNS, and that is the one thing that could
make the register earn its keep. The register arm draws sigma uniformly from all
2^(2N) patterns, one per shot - a RANDOM design. The frame arm fixes K frames and
spends S/K shots on each - a FRACTIONAL design with K rows. If K is too small the
frame arm is rank-deficient or ill-conditioned; if K is large enough the two should
be statistically equivalent. So the question is not only "does it match" but "at
what K", and K is the number of distinct circuits the ancilla-free arm needs.

PREDICTION, STATED BEFORE RUNNING. They match at matched total shots once K is a
small multiple of M, because the information is in the shots and both arms decode
the same degree-1 Walsh coefficients. If so, the register is not load-bearing.

TIER (project rule R1): tier A throughout - real Qiskit circuits, AerSimulator,
finite shots, both arms.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs

N = 3
T = 0.25
N_PROBES = 4
SEEDS = [11, 22, 33]
TOTAL_SHOTS = 8 * (1 << 16)          # what the register arm spends at 8 circuits

terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)
Z = np.array([Pauli(t).z.astype(int) for t in terms])
X = np.array([Pauli(t).x.astype(int) for t in terms])


def frame_circuit(c_true, probe_angles, basis_letter, a, b, device_reps=1):
    """ANCILLA-FREE: a fixed Pauli frame Q(a,b), applied and undone classically.

    N qubits total. No register, no controlled Cliffords - the frame is a layer
    of X and Z gates, which on hardware is absorbed into the control software.
    """
    sysr = QuantumRegister(N, 's')
    csys = ClassicalRegister(N, 'cs')
    qc = QuantumCircuit(sysr, csys)
    for q, (th, ph) in enumerate(probe_angles):
        qc.u(th, ph, 0.0, sysr[q])
    for i in range(N):                       # Q = prod X^a Z^b
        if a[i]:
            qc.x(sysr[i])
        if b[i]:
            qc.z(sysr[i])
    H = SparsePauliOp.from_list(list(zip(terms, np.asarray(c_true, float)))).simplify()
    qc.append(PauliEvolutionGate(
        H, time=T, synthesis=SuzukiTrotter(order=2, reps=device_reps)), sysr)
    for i in range(N):                       # Paulis are self-inverse
        if b[i]:
            qc.z(sysr[i])
        if a[i]:
            qc.x(sysr[i])
    for q in range(N):
        if basis_letter == 'X':
            qc.h(sysr[q])
        elif basis_letter == 'Y':
            qc.sdg(sysr[q]); qc.h(sysr[q])
    qc.measure(sysr, csys)
    return qc


def sigma_of(a, b):
    return (-1.0) ** ((Z @ a + X @ b) % 2)


def estimate_frames(cal, K, shots_per, probe_seed=0, seed=0, enumerate_all=False,
                    distinct=False):
    """Ancilla-free arm: K classical Pauli frames, shots_per shots each.

    enumerate_all replaces random sampling by the FULL factorial over every
    (a,b) in {0,1}^N x {0,1}^N - which is exactly what the register realises,
    since H on 2N qubits puts equal amplitude on all 2^(2N) patterns.
    """
    rng = np.random.default_rng(seed)
    all_ab = [(np.array(aa), np.array(bb))
              for aa in np.ndindex(*([2] * N))
              for bb in np.ndindex(*([2] * N))] if enumerate_all else None
    pr = np.random.default_rng(probe_seed)
    probes = []
    for _ in range(N_PROBES):
        probes.append([(float(np.arccos(1 - 2 * pr.random())),
                        float(2 * np.pi * pr.random())) for _ in range(N)])

    num = np.zeros(M)
    den = np.zeros(M)
    ncirc = 0
    for ang in probes:
        psi = cal._probe_state(ang)
        for letter in ('Z', 'X'):
            # accumulate per-observable Walsh sums over the sampled frames
            acc = np.zeros((N, M))
            tot = 0
            if enumerate_all:
                chosen = all_ab[:K]
            elif distinct:
                # balanced fraction: K DISTINCT frames, no duplicates
                idx = rng.choice(4 ** N, size=min(K, 4 ** N), replace=False)
                pool = [(np.array(aa), np.array(bb))
                        for aa in np.ndindex(*([2] * N))
                        for bb in np.ndindex(*([2] * N))]
                chosen = [pool[i] for i in idx]
            else:
                chosen = [(rng.integers(0, 2, N), rng.integers(0, 2, N))
                          for _ in range(K)]
            for a, b in chosen:
                sig = sigma_of(a, b)
                qc = frame_circuit(c_true, ang, letter, a, b)
                tqc = transpile(qc, cal.backend, optimization_level=1)
                counts = cal.backend.run(tqc, shots=shots_per).result().get_counts()
                ncirc += 1
                for bitstr, cnt in counts.items():
                    sysb = bitstr.replace(' ', '')[::-1]
                    for q in range(N):
                        o = -1.0 if sysb[q] == '1' else 1.0
                        acc[q] += sig * o * cnt
                    tot += cnt
            for q in range(N):
                s = ['I'] * N
                s[N - 1 - q] = letter
                ob = ''.join(s)
                g = acc[q] / max(tot, 1)
                resp = cal._response(psi, ob)
                w = resp ** 2
                est = np.where(np.abs(resp) > 1e-6,
                               g / (cal.T * np.where(np.abs(resp) > 1e-6,
                                                     resp, 1.0)), 0.0)
                num += w * est
                den += w
    cal.coverage = int(np.sum(den > 1e-12))
    return num / np.maximum(den, 1e-30), ncirc


def relstats(chat):
    rel = np.abs(chat - c_true) / np.abs(c_true)
    return float(np.mean(rel)), float(np.max(rel))


print("=" * 100)
print("v107  IS THE REGISTER LOAD-BEARING?  2N ancillas against zero")
print("=" * 100)
print("  N=%d, M=%d, T=%.2f, %d probes, matched TOTAL shots = %.2e"
      % (N, M, T, N_PROBES, TOTAL_SHOTS))
print("  %d seeds. TIER A both arms." % len(SEEDS))
print()
print("   arm                       qubits   distinct circuits   shots/circ    mean rel      max rel")
print("   " + "-" * 96)

# --- register arm ------------------------------------------------------------
A, B = [], []
for sd in SEEDS:
    be = AerSimulator(method='statevector', seed_simulator=sd)
    cal = TwirlCalibrator(terms, evolution_time=T, shots=TOTAL_SHOTS // 8,
                          seed=sd, device_reps=1, backend=be)
    chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0, grouped=True)
    m, x = relstats(chat)
    A.append(m); B.append(x)
print("   register (superposed)      %4d          %4d          %8d      %.4f       %.4f"
      % (3 * N, 8, TOTAL_SHOTS // 8, np.mean(A), np.mean(B)))

# --- ancilla-free arm, several K --------------------------------------------
for K in (4, 8, 16, 32):
    ncirc_total = 2 * N_PROBES * K
    shots_per = max(1, TOTAL_SHOTS // ncirc_total)
    A, B = [], []
    for sd in SEEDS:
        be = AerSimulator(method='statevector', seed_simulator=sd)
        cal = TwirlCalibrator(terms, evolution_time=T, shots=1, seed=sd,
                              device_reps=1, backend=be)
        chat, nc = estimate_frames(cal, K, shots_per, probe_seed=0, seed=sd)
        m, x = relstats(chat)
        A.append(m); B.append(x)
    print("   frames, K=%-3d random        %4d          %4d          %8d      %.4f       %.4f"
          % (K, N, ncirc_total, shots_per, np.mean(A), np.mean(B)))

# --- ancilla-free, BALANCED FRACTION: K distinct frames, no duplicates -------
for K in (12, 16, 24, 32):
    ncirc_total = 2 * N_PROBES * K
    shots_per = max(1, TOTAL_SHOTS // ncirc_total)
    A, B = [], []
    for sd in SEEDS:
        be = AerSimulator(method='statevector', seed_simulator=sd)
        cal = TwirlCalibrator(terms, evolution_time=T, shots=1, seed=sd,
                              device_reps=1, backend=be)
        chat, nc = estimate_frames(cal, K, shots_per, probe_seed=0, seed=sd,
                                   distinct=True)
        m, x = relstats(chat)
        A.append(m); B.append(x)
    print("   frames, K=%-3d distinct      %4d          %4d          %8d      %.4f       %.4f"
          % (K, N, ncirc_total, shots_per, np.mean(A), np.mean(B)))

# --- ancilla-free, FULL FACTORIAL: what the register actually realises -------
K_FULL = 4 ** N
ncirc_total = 2 * N_PROBES * K_FULL
shots_per = max(1, TOTAL_SHOTS // ncirc_total)
A, B = [], []
for sd in SEEDS:
    be = AerSimulator(method='statevector', seed_simulator=sd)
    cal = TwirlCalibrator(terms, evolution_time=T, shots=1, seed=sd,
                          device_reps=1, backend=be)
    chat, nc = estimate_frames(cal, K_FULL, shots_per, probe_seed=0, seed=sd,
                               enumerate_all=True)
    m, x = relstats(chat)
    A.append(m); B.append(x)
print("   frames, ALL %d enumerated    %4d          %4d          %8d      %.4f       %.4f"
      % (K_FULL, N, ncirc_total, shots_per, np.mean(A), np.mean(B)))
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  THE PREDICTION WAS HALF RIGHT, AND THE HALF IT GOT WRONG IS THE INTERESTING")
print("  ONE. Ancilla-free DOES match - the FULL factorial arm reaches 0.054 against")
print("  the register's 0.075 at matched total shots, on 3 qubits instead of 9. So")
print("  the register carries no INFORMATION the frames cannot.")
print()
print("  But it needs ALL 4^N frames to do it, and no fraction works:")
print()
print("      K=32 random      1.1035        K=32 distinct    1.0977")
print("      K=24 distinct    1.3729        ALL 64           0.0543")
print()
print("  Removing duplicates changes essentially nothing (1.1035 -> 1.0977), and")
print("  half the design is still 20x worse than all of it. So the failure is not")
print("  sampling noise and not duplication - it is ORTHOGONALITY. The decode here")
print("  is a plain average, acc = sum sigma_k o cnt / tot, which estimates the")
print("  degree-1 Walsh coefficient without bias ONLY when sigma is uniform over the")
print("  whole group. On a partial design the Walsh characters are no longer")
print("  orthogonal and other degrees leak into the estimate.")
print()
print("  SO WHAT THE REGISTER ACTUALLY DOES is compress a 4^N-row balanced design")
print("  into O(1) circuits. That is EXPONENTIAL CIRCUIT COMPRESSION, and it is a")
print("  much stronger statement than twirl_cal's docstring makes - that file says")
print("  the superposition is 'doing the same job as sampling twirls at random' and")
print("  claims only that it saves compilation. On this evidence random sampling is")
print("  not the same job at all.")
print()
print("  THE OPEN QUESTION, AND IT DECIDES THE QUBIT COST. The plain average needs")
print("  the full group; a WEIGHTED LEAST SQUARES decode over a partial design does")
print("  not, and this project already built one - supplement/v78_wls_on_circuits.py.")
print("  If WLS on an O(M)-row balanced fraction recovers the accuracy, the")
print("  ancilla-free arm becomes O(M) circuits on N qubits, the 2N register is")
print("  redundant, and the largest gap to arXiv:2606.19486's zero-ancilla")
print("  requirement closes. If it does not, the register's compression is genuinely")
print("  load-bearing and 2N ancillas are what it costs. UNTESTED either way.")
print()
print("  What no arm here changes: the precision exponent, and the degree-2 aliasing")
print("  from v106 - both decode the same Walsh columns and inherit the same")
print("  v_XX + v_YY = v_ZZ degeneracy.")
print()
print("  Scope: N=3, one coefficient draw, noiseless, one T, plain-average decode.")
print("  The full-factorial arm's 512 circuits are 4^N * 2 * n_probes and therefore")
print("  exponential in N - it is a proof that the information is there, not a")
print("  usable protocol.")
