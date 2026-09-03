"""twirl_cal on the literature's axis: total evolution time to reach precision eps.

Every Hamiltonian-learning result in the field is quoted as TOTAL EVOLUTION TIME
as a function of accuracy. twirl_cal has never been measured that way, which is
why it has never actually been compared to anything. This file measures it.

THE COMPARISON POINTS, from Table 1 of arXiv:2606.19486:

    Huang et al. [24]     Local & KNOWN structure   discrete control, no ancilla   O(1/eps)
    Bakshi et al. [26]    Local & bounded-degree    discrete control, no ancilla   O(log n / eps)
    Sinha & Tong,
      Abbas et al.        ansatz-free               discrete control, ancilla      ~O(M/eps)
    arXiv:2606.19486      ansatz-free               NO control, NO ancilla         Theta(Lambda/eps^2)
                          + a matching lower bound: optimal for control-free

WHICH ROW twirl_cal IS IN, and it is not the flattering one. TwirlCalibrator
takes `terms` as an argument - it cannot discover a Pauli string that was not
named - so it solves the KNOWN-STRUCTURE problem, Huang et al.'s row, where the
frontier is HEISENBERG-LIMITED O(1/eps) with no ancilla. twirl_cal is first order
in T, so SQL at best, and spends 2N ancillas. It is therefore solving an easier
problem with more resources, and the only open question is whether it at least
achieves the SQL rate.

That is what this file answers. Total evolution time for twirl_cal is exactly

    T_total = n_circuits * shots_per_circuit * T

since every shot evolves the device for time T once, uncontrolled. Sweeping both
shots and T and taking the best eps at each T_total traces the achievable
frontier, and its slope is the exponent to compare.

    slope -1/2  -> SQL, matching Theta(Lambda/eps^2). The rate is competitive and
                   what remains against the frontier is the ancilla count and the
                   fact that this needs the term set in advance.
    slope worse -> twirl_cal does not even reach the control-free optimum, and the
                   O(1) circuit count is the only axis it can claim.
    slope -1    -> Heisenberg-limited, which would contradict its own first-order
                   construction and should be disbelieved before it is believed.

eps IS ABSOLUTE, max_k |chat_k - c_k|, to match the theorem's "estimates every
coefficient up to eps". Relative error is not what the bound is about.

WHAT IS NOT BENCHMARKED HERE, and why. A head-to-head against Algorithm 2 of
arXiv:2606.19486 would need its Appendix C kernel - G = ||L||^-1 sign(L(T)) Z
with L a Whittaker-Kotelnikov-Shannon derivative kernel under a smooth cutoff,
plus constants C_0 and gamma_1. Implementing that wrong silently breaks the
estimator's unbiasedness and would produce a strawman baseline, which is worse
than no baseline. The scaling comparison below is against the PROVEN bound
instead, which needs no reimplementation and cannot be strawmanned.

TIER (project rule R1): tier A - real Qiskit circuits, AerSimulator, finite shots.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs

N = 3
N_PROBES = 4
SEEDS = [11, 22, 33]
T_LIST = [0.10, 0.25, 0.50, 1.00]
SHOTS_LIST = [1 << 12, 1 << 14, 1 << 16, 1 << 18]

terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)
H = SparsePauliOp.from_list(list(zip(terms, c_true))).simplify()
LAM = float(np.linalg.norm(H.to_matrix(), 2))
NCIRC = 2 * N_PROBES                       # grouped: 2 bases x n_probes


def run(T, shots, sd):
    be = AerSimulator(method='statevector', seed_simulator=sd)
    cal = TwirlCalibrator(terms, evolution_time=T, shots=shots, seed=sd,
                          device_reps=1, backend=be)
    chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0, grouped=True)
    return float(np.max(np.abs(chat - c_true))), cal.ncircuits


print("=" * 100)
print("v113  TOTAL EVOLUTION TIME TO PRECISION eps")
print("=" * 100)
print("  N=%d crosstalk, M=%d, ||H||_2 = Lambda = %.4f, %d probes -> %d circuits."
      % (N, M, LAM, N_PROBES, NCIRC))
print("  eps is ABSOLUTE max_k |chat_k - c_k|, matching the theorem's statement.")
print("  T_total = circuits * shots * T. %d seeds per point. TIER A." % len(SEEDS))
print()
print("      T      shots/circ    T_total        eps (abs)")
print("  " + "-" * 66)

pts = []
for T in T_LIST:
    for sh in SHOTS_LIST:
        es = [run(T, sh, sd)[0] for sd in SEEDS]
        ttot = NCIRC * sh * T
        pts.append((T, sh, ttot, float(np.mean(es))))
        print("   %5.2f   %10d   %.4e     %.5f" % (T, sh, ttot, np.mean(es)))
    print()

print("=" * 100)
print("THE ACHIEVABLE FRONTIER  (best eps at each total evolution time)")
print("=" * 100)
print("  For each T_total, the lowest eps over all T that reach it - which is what")
print("  an experimenter free to choose T would get.")
print()
# bucket by total evolution time, keep the best eps in each bucket
buckets = {}
for T, sh, ttot, e in pts:
    k = round(np.log10(ttot), 1)
    if k not in buckets or e < buckets[k][1]:
        buckets[k] = (ttot, e, T)
ks = sorted(buckets)
print("     T_total        best eps     at T")
print("  " + "-" * 52)
for k in ks:
    ttot, e, T = buckets[k]
    print("   %.4e     %.5f      %.2f" % (ttot, e, T))
print()

x = np.log(np.array([buckets[k][0] for k in ks]))
y = np.log(np.array([buckets[k][1] for k in ks]))
slope = float(np.polyfit(x, y, 1)[0])

print("=" * 100)
print("READING IT")
print("=" * 100)
print("   fitted   eps ~ T_total^(%.3f)" % slope)
print()
print("   SQL / arXiv:2606.19486  Theta(Lambda/eps^2)  ->  slope -0.500")
print("   Heisenberg-limited      O(1/eps)             ->  slope -1.000")
print()
if slope > -0.40:
    print("   WORSE THAN SQL. twirl_cal does not reach the control-free optimum on")
    print("   its own axis, so the O(1) circuit count is the only claim it can make.")
elif slope > -0.62:
    print("   AT SQL, matching the control-free optimum's RATE. What remains against")
    print("   arXiv:2606.19486 is then not the rate but the resources: it needs the")
    print("   term set named in advance (their protocol does not) and spends 2N")
    print("   ancillas (their protocol spends none). Solving an easier problem with")
    print("   more hardware at the same rate is not a win.")
else:
    print("   BETTER THAN SQL, which contradicts a first-order-in-T construction and")
    print("   should be disbelieved until the fit is checked over a wider range.")
print()
print("   Against Huang et al.'s O(1/eps) for KNOWN structure - twirl_cal's actual")
print("   problem class - anything at SQL is a full factor of 1/eps behind.")
print()
print("  Scope: N=3, M=%d, one coefficient draw, noiseless, %d seeds, %d points"
      % (M, len(SEEDS), len(ks)))
print("  spanning several decades of T_total. A slope fitted over that range is")
print("  indicative, not a scaling law; the bias floor at large T bends it.")
