"""Theta(N) -> Theta(1) circuits: the observable count was an implementation cost.

v104's scaling table measured twirl_cal at exactly 8N circuits (24, 32, 40, 48,
56, 64 at N=3..8), so the commit's "O(1) circuits" does not hold. The growth comes
entirely from the default observable set - N single-qubit Z plus N single-qubit X,
one circuit each.

BUT THOSE CIRCUITS ARE NOT DISTINCT. _walsh builds

    basis[q] = obs_pauli[n-1-q] if obs_pauli[n-1-q] in 'XYZ' else 'Z'

so every one of the N Z-observables gives basis ['Z','Z',...,'Z'] - the SAME
circuit - and they differ only in `support`, i.e. in which qubit's parity the
classical decode reads out of the same bitstrings. N circuits are being run where
one set of counts already contains all N answers.

THE FIX IS TO GROUP BY MEASUREMENT BASIS. Fix a basis b, run ONE circuit, and
decode every observable that is diagonal in b from the same counts. This is the
qubit-wise-commuting grouping that the rest of this project already lives on - the
G in "one gradient costs G circuits" - applied to the readout side here.

TWO BASES SUFFICE FOR THIS FAMILY, and the reason is a commutation argument, not
a measurement. In basis b the available observables are the Paulis P_S with
P_S[q] = b[q] on a subset S. P_k has a nonzero commutator with some P_S iff P_k
disagrees with b on at least one non-identity position:

    all-Z basis  sees every term carrying an X or Y      -> XX, YY
                 blind to terms built only from I and Z  -> ZZ, single Z
    all-X basis  sees every term carrying a Y or Z       -> ZZ, YY, single Z
                 blind to terms built only from I and X  -> XX

Their union covers all four families, so coverage is complete at TWO bases at any
N. Circuits become n_probes * 2, INDEPENDENT OF N AND OF M. That is the O(1)
the commit claimed, reached honestly.

WHAT THIS DOES NOT BUY, and the comparison must be at matched TOTAL shots. Running
2 circuits instead of 2N at the same shots-per-circuit spends 1/N the shots and
must lose accuracy - the information was never in the circuit count, it was in the
shots. The claim is that the same total shot budget can be delivered through
Theta(1) circuits instead of Theta(N), which matters because vendors bill per task
as well as per shot. So the grouped arm is given N times the shots per circuit.

A SECOND COST, NAMED. Observables sharing a basis are now decoded from the SAME
counts, so their shot noise is correlated where before it was independent. The
weighted combination in estimate() assumes nothing about independence, but the
variance of the result is not identical - which is exactly why this is measured
rather than asserted.

TIER (project rule R1): tier A throughout - real circuits, AerSimulator, shots.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit_aer import AerSimulator
from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs

T_MAIN = 0.25
N_PROBES = 4
SEEDS = [11, 22, 33]
BASE_SHOTS = 1 << 16


def obs_string(N, basis_letter, q):
    """Pauli string with basis_letter on qubit q (little-endian), I elsewhere."""
    s = ['I'] * N
    s[N - 1 - q] = basis_letter
    return ''.join(s)


def walsh_grouped(cal, c_true, probe_angles, basis_letter):
    """ONE circuit; decode every single-qubit observable in that basis from it."""
    n = cal.N
    basis = [basis_letter] * n
    qc = cal._circuit(c_true, probe_angles, basis)
    tqc = transpile(qc, cal.backend, optimization_level=1)
    counts = cal.backend.run(tqc, shots=cal.shots).result().get_counts()
    cal.ncircuits += 1

    out = {}
    for q in range(n):
        acc = np.zeros(cal.M)
        tot = 0
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            sysb, regb = parts[0][::-1], parts[1][::-1]
            o = -1.0 if sysb[q] == '1' else 1.0
            a = np.array([int(regb[i]) for i in range(n)])
            b = np.array([int(regb[n + i]) for i in range(n)])
            sig = (-1.0) ** ((cal._z @ a + cal._x @ b) % 2)
            acc += sig * o * cnt
            tot += cnt
        out[obs_string(n, basis_letter, q)] = acc / max(tot, 1)
    return out


def estimate_grouped(cal, c_true, n_probes=N_PROBES, probe_seed=0):
    """Same estimator as TwirlCalibrator.estimate, but 2 circuits per probe."""
    N, M = cal.N, cal.M
    pr = np.random.default_rng(probe_seed)
    probes = []
    for _ in range(n_probes):
        probes.append([(float(np.arccos(1 - 2 * pr.random())),
                        float(2 * np.pi * pr.random())) for _ in range(N)])

    num = np.zeros(M)
    den = np.zeros(M)
    for ang in probes:
        psi = cal._probe_state(ang)
        for letter in ('Z', 'X'):
            g_by_obs = walsh_grouped(cal, c_true, ang, letter)
            for ob, g in g_by_obs.items():
                resp = cal._response(psi, ob)
                w = resp ** 2
                est = np.where(np.abs(resp) > 1e-6,
                               g / (cal.T * np.where(np.abs(resp) > 1e-6,
                                                     resp, 1.0)), 0.0)
                num += w * est
                den += w
    cal.coverage = int(np.sum(den > 1e-12))
    return num / np.maximum(den, 1e-30)


print("=" * 100)
print("v105  BASIS GROUPING:  the observable count was implementation, not physics")
print("=" * 100)
print()
print("=" * 100)
print("PART 1  CIRCUIT COUNT vs N,  ungrouped against grouped")
print("=" * 100)
print("  Coverage is checked, not assumed: 'cov' is how many of the M terms have")
print("  nonzero total response across the probe/observable set.")
print()
print("    N     M    ungrouped circuits    grouped circuits    cov(grouped)")
print("   " + "-" * 78)
for N in (3, 4, 5, 6, 7, 8):
    terms = crosstalk_terms(N)
    c = crosstalk_coeffs(N)
    cal = TwirlCalibrator(terms, evolution_time=T_MAIN, shots=1024, seed=0)
    chat = estimate_grouped(cal, c, n_probes=N_PROBES)
    print("   %2d   %3d          %4d                %4d              %d/%d"
          % (N, len(terms), N_PROBES * 2 * N, cal.ncircuits,
             cal.coverage, len(terms)))
print()

print("=" * 100)
print("PART 2  ACCURACY AT MATCHED TOTAL SHOTS,  N=4")
print("=" * 100)
print("  Grouped runs N times fewer circuits, so it is given N times the shots per")
print("  circuit. Total shots identical; only the circuit count differs.")
print()
N = 4
terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)


def stats(chat):
    rel = np.abs(chat - c_true) / np.abs(c_true)
    return float(np.mean(rel)), float(np.max(rel)), float(np.max(np.abs(chat - c_true)))


print("   arm            circuits   shots/circ   total shots   mean rel      max rel")
print("   " + "-" * 88)
rows = []
for tag in ('ungrouped', 'grouped'):
    A, B, C, NC = [], [], [], []
    for sd in SEEDS:
        be = AerSimulator(method='statevector', seed_simulator=sd)
        if tag == 'ungrouped':
            sh = BASE_SHOTS
            cal = TwirlCalibrator(terms, evolution_time=T_MAIN, shots=sh, seed=sd,
                                  device_reps=1, backend=be)
            chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0)
        else:
            sh = BASE_SHOTS * N          # N x shots, N x fewer circuits
            cal = TwirlCalibrator(terms, evolution_time=T_MAIN, shots=sh, seed=sd,
                                  device_reps=1, backend=be)
            chat = estimate_grouped(cal, c_true, n_probes=N_PROBES, probe_seed=0)
        a, b, cc = stats(chat)
        A.append(a); B.append(b); C.append(cc); NC.append(cal.ncircuits)
    nc = int(np.mean(NC))
    rows.append((tag, nc, sh, nc * sh, np.mean(A), np.std(A), np.mean(B)))
    print("   %-12s   %5d     %8d     %.2e    %.4f+-%.4f   %.4f"
          % (tag, nc, sh, nc * sh, np.mean(A), np.std(A), np.mean(B)))
print()

ung, grp = rows[0], rows[1]
print("=" * 100)
print("READING IT")
print("=" * 100)
print("  circuits    %d -> %d   (%.1fx fewer)" % (ung[1], grp[1], ung[1] / grp[1]))
print("  total shots %.2e vs %.2e" % (ung[3], grp[3]))
print("  mean rel    %.4f vs %.4f" % (ung[4], grp[4]))
print()
if grp[4] <= ung[4] + 2 * max(ung[5], grp[5]):
    print("  SAME ACCURACY AT THE SAME TOTAL SHOTS, in Theta(1) circuits rather than")
    print("  Theta(N). The observable count was never physics: N single-qubit Z")
    print("  observables share one measurement basis, so one circuit's counts already")
    print("  hold all N answers. Circuits are now n_probes * 2, independent of N AND")
    print("  of M, and the correlated shot noise across co-measured observables costs")
    print("  nothing detectable at this size.")
else:
    print("  ACCURACY LOST. The correlated shot noise across observables decoded from")
    print("  shared counts is not free, and the Theta(1) circuit count is bought at a")
    print("  price in precision that has to be quoted alongside it.")
print()
print("  Scope: N<=8, one coefficient draw, noiseless, one T, single-qubit")
print("  observables only. Whether TWO bases still suffice for a Hamiltonian family")
print("  outside this ZZ+XY+Z set is a commutation question per family, not a")
print("  measurement - the argument in the docstring is what generalises.")
print()

print("=" * 100)
print("PART 3  HOW FAR DOWN?  the remaining lever is n_probes, not the basis count")
print("=" * 100)
print("  Two bases is minimal for this family by the commutation argument above, so")
print("  the circuit count is now 2 * n_probes and the only question left is how few")
print("  probes still cover M terms. Probes are needed for COVERAGE - a term with")
print("  <i[P_k,O]> = 0 on every probe is unrecoverable at any shot count - so this")
print("  is a floor set by the Hamiltonian, not by statistics.")
print()
print("    N     M   probes   CIRCUITS   cov     mean rel err")
print("   " + "-" * 74)
for N in (4, 6):
    terms = crosstalk_terms(N)
    ct = crosstalk_coeffs(N)
    for npr in (1, 2, 3, 4):
        errs = []
        cov = None
        nc = None
        for sd in SEEDS:
            be = AerSimulator(method='statevector', seed_simulator=sd)
            cal = TwirlCalibrator(terms, evolution_time=T_MAIN,
                                  shots=BASE_SHOTS, seed=sd,
                                  device_reps=1, backend=be)
            chat = estimate_grouped(cal, ct, n_probes=npr, probe_seed=0)
            errs.append(float(np.mean(np.abs(chat - ct) / np.abs(ct))))
            cov, nc = cal.coverage, cal.ncircuits
        flag = "" if cov == len(terms) else "   <- INCOMPLETE"
        print("   %2d   %3d      %d        %4d      %2d/%2d    %.4f%s"
              % (N, len(terms), npr, nc, cov, len(terms), np.mean(errs), flag))
    print()

print("=" * 100)
print("WHERE THIS LANDS")
print("=" * 100)
print("  Circuits = 2 * n_probes: independent of N and of M, set only by how many")
print("  probes the Hamiltonian family needs for coverage. That is O(1) in the sense")
print("  the commit wanted, and it was reached by grouping the readout rather than")
print("  by changing the construction. The 8N figure in v104 stands as a measurement")
print("  of what the code did, not of what the method requires.")
