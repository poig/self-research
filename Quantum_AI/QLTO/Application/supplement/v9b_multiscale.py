"""Multi-bit encoding as a FREE multi-scale gradient probe.

The proposal was multi-ancilla, one per precision level. That conflates two
registers: ancillas resolve the ENERGY (k QPE bits), param qubits resolve the
PARAMETERS. Adding ancillas cannot refine theta.

But the underlying goal - know the coarse AND fine structure while walking, rather
than stepping blind - falls out of a multi-bit param encoding for free. With

    theta_i = c_i + R*s_i0 + (R/2)*s_i1

the degree-1 Walsh coefficient PER BIT comes from the same shot record (T2):

    Ehat({i0}) ~ R   * d_iE       coarse scale
    Ehat({i1}) ~ R/2 * d_iE       fine scale

Two claims to check:

  RATIO      Ehat({i0}) / Ehat({i1}) should be exactly 2 where the landscape is
             locally linear in coordinate i, and deviate where it is not. That
             deviation is a free "is my step size still in the linear regime?"
             diagnostic - the thing that tells you whether you are walking blind.

  BIAS       Ehat({i1})/(R/2) differences coordinate i at HALF the span, so it
             should track the true gradient better than Ehat({i0})/R, at the cost
             of a smaller signal. A bias-variance dial from ONE circuit.

Exact enumeration of the 2^(2n) grid - no circuits, no shot noise, so any
deviation from 2 is real structure rather than sampling.
"""
import sys, os, contextlib, io, itertools
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


def two_bit_walsh(ansatz, H, c, R, act):
    """Exact per-bit degree-1 Walsh coefficients for a 2-bit-per-param encoding."""
    n = len(act)
    combos = list(itertools.product([-1.0, 1.0], repeat=2 * n))
    E = np.empty(len(combos))
    S0 = np.empty((len(combos), n)); S1 = np.empty((len(combos), n))
    for v, s in enumerate(combos):
        s0 = np.array(s[:n]); s1 = np.array(s[n:])
        p = c.copy()
        p[act] = c[act] + R * s0 + (R / 2.0) * s1
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
        S0[v] = s0; S1[v] = s1
    w0 = np.array([np.mean(E * S0[:, i]) for i in range(n)])
    w1 = np.array([np.mean(E * S1[:, i]) for i in range(n)])
    return w0, w1


def exact_grad(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                    .expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                      .expectation_value(H))))
    return g


ansatz, H, _ = B.get_heisenberg_problem(4)
q = Q(ansatz, H, shot_budget=8192)
act = q.layers[0]['params']
print("=" * 86)
print("Multi-scale gradient from ONE multi-bit circuit")
print("=" * 86)
print(f"  Heisenberg N=4, block of {len(act)} params, 2 bits each "
      f"(2^{2*len(act)} = {2**(2*len(act))} vertices), 3 centres")
print()
print(f"  {'R':>7}{'ratio w0/w1':>14}{'(exact 2 if linear)':>22}"
      f"{'cos(w0,g)':>11}{'cos(w1,g)':>11}{'|w1/(R/2)|/|g|':>16}")
print("  " + "-" * 81)
for R in (0.2, 0.4, 0.6, 1.0, 1.5):
    rats, c0, c1, mag1, mag0 = [], [], [], [], []
    for seed in (3, 11, 17):
        c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                ansatz.num_parameters)
        w0, w1 = two_bit_walsh(ansatz, H, c, R, act)
        g = exact_grad(ansatz, H, c, act)
        keep = np.abs(w1) > 1e-6
        if keep.any():
            rats.append(np.median(w0[keep] / w1[keep]))
        cs = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        c0.append(cs(w0, g)); c1.append(cs(w1, g))
        mag0.append(np.linalg.norm(w0 / R) / np.linalg.norm(g))
        mag1.append(np.linalg.norm(w1 / (R / 2.0)) / np.linalg.norm(g))
    print(f"  {R:>7.2f}{np.mean(rats):>14.4f}{'':>22}"
          f"{np.mean(c0):>11.5f}{np.mean(c1):>11.5f}"
          f"{np.mean(mag1):>16.4f}", flush=True)

print()
print("  ratio -> 2.0 at small R and drifting at large R means the coarse/fine")
print("  disagreement IS a usable nonlinearity signal, available from the same")
print("  shots that already produce the gradient.")
print("  cos(w1,g) > cos(w0,g) means the fine bit tracks the true gradient")
print("  direction better - the bias half of the bias-variance dial.")
