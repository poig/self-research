"""Why does the free log have an offset when the exact log does not?

The two are not the same measurement, and the difference is the whole answer.

    exact log   <psi|H|psi>, computed directly
    free log    the MEAN of QPE-decoded eigenvalue samples

Those agree only if QPE's mean is an unbiased estimator of the phase. It is not.
For a true phase phi, QPE returns bin m with

    P(m) = sin^2(2^k pi d) / (4^k sin^2(pi d)),      d = phi - m/2^k

whose MODE is the nearest bin - that is the textbook guarantee - but whose TAILS
FALL ONLY AS 1/d^2. The mean therefore weights outcomes by d * P(d) ~ 1/d, whose
sum is logarithmically divergent: the mean is dominated by rare far-off bins, not
by the peak. On top of that the wrap into [-1/2, 1/2) sends a tail that belongs
at phi = 0.51 to -0.49, so it contributes with the WRONG SIGN rather than merely
the wrong magnitude.

    QPE ESTIMATES A MODE WELL AND A MEAN BADLY.

Three candidate contributions, and they are separable by sweeping one knob each:

    binning / tails   sweep kappa      more bits -> narrower peak, but the 1/d^2
                                       tail is scale-free so this may not help
    wrap              sweep qpe_margin larger margin pushes the spectrum away
                                       from the +-0.5 boundary
    Trotter           sweep reps       changes the effective Hamiltonian

AND A FIX FALLS OUT. The gradient must use the mean - T2's unbiasedness is
exactly the linearity of an empirical mean, and a median would forfeit it. But
the LOG is a single scalar with no such requirement, so a median or trimmed mean
can be used there and should shed the tail contamination.

Everything here runs at R = 0, where the W gate maps every vertex to theta_c and
the hypercube smearing vanishes identically - so any residual offset is the
readout alone, not the O(R^2) curvature term.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def decoded_samples(q, center, R, act):
    """Every shot's decoded energy, as a flat array (with multiplicity)."""
    counts = q._run(q._build_qpe_sensing_circuit(center, R, act))
    k = q.num_ancillas
    vals, wts = [], []
    for bitstr, cnt in counts.items():
        parts = bitstr.split()
        if len(parts) != 2:
            continue
        m = int(parts[0], 2)
        phi = m / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        vals.append(-2.0 * np.pi * phi / (q.tau0 + 1e-12))
        wts.append(cnt)
    v = np.array(vals); w = np.array(wts, dtype=float)
    order = np.argsort(v)
    return v[order], w[order]


def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))


def wmedian(v, w):
    c = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(c, 0.5)])


def wtrimmed(v, w, frac=0.1):
    c = np.cumsum(w) / np.sum(w)
    keep = (c >= frac) & (c <= 1 - frac)
    if not keep.any():
        return wmean(v, w)
    return wmean(v[keep], w[keep])


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4))]
SHOTS = 65536

print("=" * 96)
print("WHY THE FREE LOG IS OFFSET — at R=0, so smearing is identically zero")
print("=" * 96)
print("  Any residual here is the QPE READOUT, not the O(R^2) hypercube curvature.")
print("  mean is what the free log uses; median/trimmed shed the 1/d^2 tails.")
print()
print(f"  {'problem':<17}{'kappa':>6}{'exact':>10}{'mean':>10}{'median':>10}"
      f"{'trim10%':>10}{'bias mean':>11}{'bias med':>10}")
print("  " + "-" * 84)

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)
    for kappa in (3, 4, 5, 6):
        q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=kappa, sim_seed=3)
        q.reset_shot_stream()
        act = [b['params'] for b in q.layers if b['params']][0]
        ex = float(np.real(Statevector(
            ansatz.assign_parameters(c)).expectation_value(q.H_sense)))
        v, w = decoded_samples(q, c, 0.0, act)
        mu, md, tr = wmean(v, w), wmedian(v, w), wtrimmed(v, w)
        print(f"  {name if kappa == 3 else '':<17}{kappa:>6}{ex:>10.4f}{mu:>10.4f}"
              f"{md:>10.4f}{tr:>10.4f}{mu - ex:>+11.4f}{md - ex:>+10.4f}", flush=True)

print()
print("  If |bias med| << |bias mean| the diagnosis is confirmed: the offset is")
print("  tail contamination of the MEAN, and the log should use a median or")
print("  trimmed mean. The GRADIENT must keep the mean - T2's unbiasedness at any")
print("  shots-per-vertex IS the linearity of an empirical mean, and a median is")
print("  not linear, so this fix applies to the scalar log ONLY.")
print()
print("  If both biases persist and shrink with kappa, it is binning. If neither")
print("  moves with kappa, it is Trotter error in the effective Hamiltonian.")
