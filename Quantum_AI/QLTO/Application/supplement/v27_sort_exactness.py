"""Is term sorting exact on the FULL sensing circuit, or only on one stage?

v24 compared a single standalone PauliEvolutionGate and measured 0.00e+00. On the
strength of that I shipped sort_terms=True as a default. v26 then found the
sorted arm 2.2 sigma worse on MaxCut N=4 and 3.3 sigma worse on Heisenberg N=4 -
which an exact rewrite cannot cause. One of the two results is wrong, and shipping
a default on the strength of the weaker test was the mistake pattern these notes
already record for the Suzuki step-floor rule.

Three candidates:

  (a) SORTING IS NOT EXACT HERE. v24 tested ONE stage at ONE rep count. The full
      circuit runs kappa stages at different evolution times and different rep
      counts, and Suzuki-2 is a symmetric formula whose error depends on which
      terms are ADJACENT. Reordering changes adjacency, so the error constant can
      change even when a single stage happens to come out exact.
  (b) SORTING IS EXACT AND v26 IS NOISE. The seeds fix only the initial
      parameters - sampling is unseeded - so paired arms are still independent
      draws. These notes record two results at 2.5-3 sigma on few seeds that
      reversed on replication.
  (c) SOMETHING ELSE IS ORDER-DEPENDENT. H0_norm, tau0, or dead-block detection
      reading term order.

(a) and (b) are separated by a single deterministic measurement that needs no
seeds at all: build the full sensing circuit both ways, strip the measurements,
and compare unitaries. Exact means exact - if the spectral distance is zero the
optimiser difference cannot be caused by sorting and must be sampling noise.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Operator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def strip_measure(qc):
    out = qc.copy_empty_like()
    for instr in qc.data:
        if instr.operation.name not in ('measure', 'barrier'):
            out.append(instr)
    return out


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4))]

print("=" * 88)
print("IS SORTING EXACT ON THE FULL SENSING CIRCUIT?")
print("=" * 88)
print("  Same problem, same centre, same R. sort_terms False vs True, measurements")
print("  stripped, unitaries compared directly. No seeds, no shots, no sampling.")
print()
print(f"  {'problem':<18}{'kappa':>6}{'H0_norm same':>14}{'tau0 same':>11}"
      f"{'spectral dist':>15}{'verdict':>10}")
print("  " + "-" * 74)

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)
    for kappa in (4, 2):
        qa = Q(ansatz, H, shot_budget=1024, num_ancillas=kappa, sort_terms=False)
        qb = Q(ansatz, H, shot_budget=1024, num_ancillas=kappa, sort_terms=True)
        act = [b['params'] for b in qa.layers if b['params']][0]

        same_norm = abs(qa.H0_norm - qb.H0_norm) < 1e-10
        same_tau = abs(qa.tau0 - qb.tau0) < 1e-14

        Ua = Operator(strip_measure(qa._build_qpe_sensing_circuit(c, 0.6, act))).data
        Ub = Operator(strip_measure(qb._build_qpe_sensing_circuit(c, 0.6, act))).data
        d = float(np.linalg.norm(Ua - Ub, 2))
        print(f"  {name:<18}{kappa:>6}{str(same_norm):>14}{str(same_tau):>11}"
              f"{d:>15.3e}{'EXACT' if d < 1e-9 else 'DIFFERS':>10}", flush=True)

print()
print("  Zero spectral distance => sorting cannot have caused v26's 2.2 and 3.3")
print("  sigma, and those are sampling noise on six unseeded draws - keep the")
print("  default and record the false alarm.")
print("  Nonzero => Suzuki-2's error constant is adjacency-dependent, v24's")
print("  single-stage 0.00e+00 did not generalise, and sort_terms must be")
print("  reverted to False until the accuracy cost is characterised.")
