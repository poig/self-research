"""Does Richardson move V6 off the T^(-1/3) exponent, and what does it cost?

The one thing V6 does not fix is the finite-radius bias cR^2, which forces
R* ~ S^(-1/6) and error ~ T^(-1/3), against parameter-shift's unbiased T^(-1/2).
That exponent is why parameter-shift wins above the crossover, measured in v81 as
0.9877 against 0.9751 at T = 295k, and it is the only remaining gap that is a
SCALING gap rather than a constant.

Two radii cancel the leading bias:

    g(R)       = g + cR^2   + O(R^4)
    g(R/sqrt2) = g + cR^2/2 + O(R^4)
    2g(R/sqrt2) - g(R) = g + O(R^4)

which re-optimises to R* ~ S^(-1/10) and error ~ T^(-2/5).

A CLAIM I MADE AND WITHDREW BEFORE IMPLEMENTING IT. I proposed multiplexing both
radii into the design register on the grounds that V6's log-width register makes
extra structure cost a qubit rather than a circuit. Writing the displacement out,

    d = -R + 2R s + R(1 - 1/sqrt2) r + 2R(1/sqrt2 - 1) s r

the s*r term needs a DOUBLY-CONTROLLED rotation per parameter, roughly a Toffoli
each. At M=36 that is about 288 extra CX per circuit, so 3 circuits at ~422 gates
= 1266, against simply binding two radii to the same cached template: 6 circuits
at ~134 = 804. Multiplexing is WORSE than running the template twice. The
implementation therefore does the obvious thing.

SHOT ALLOCATION IS DERIVED. Minimising sum_i a_i^2 v_i / T_i at fixed total gives
T_i proportional to |a_i| sqrt(v_i); with a = (-1, 2) and v ~ 1/R^2, so
v(R/sqrt2) = 2 v(R), the split is 26% / 74% rather than even.

WHAT IS MEASURED. Error against the exact gradient over a shot sweep, with the
radius chosen per budget from a grid in BOTH arms so neither is handicapped by a
fixed R - the mistake that made v14 report a plateau that v69 later showed was an
artefact of frozen R. Then the exponent is fitted.

WHAT WOULD CONFIRM IT: the Richardson exponent steeper than plain V6's, moving
from about -2/3 toward -4/5 in 1-cos, which tracks error squared.
WHAT WOULD KILL IT: equal or shallower exponents, meaning the variance inflation
from the combination eats the bias cancellation at reachable budgets. That is a
real possibility, since the combination carries 4 v_hi + v_lo, and it would mean
Richardson belongs at large T only.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v6


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def exact_grad(ansatz, Hm, theta):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            g[i] += s * float(np.real(np.conj(v) @ (Hm @ v))) / 2
    return g


REPS = 2
BUDGETS = (2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18)
# Opened upward deliberately. Cancelling the bias is what LETS Richardson take a
# larger radius, and a grid capped at 0.6 forbids the move it exists to make: in
# the first run R* for the Richardson arm sat at the grid maximum for the two
# largest budgets, so the comparison was handicapping it.
RGRID = (1.6, 1.2, 0.9, 0.6, 0.45, 0.3, 0.2)

print("=" * 100)
print("RICHARDSON ON V6:  does the T^(-1/3) exponent move?")
print("=" * 100)
print("  Both arms get the best R from a grid at every budget, so neither is")
print("  handicapped by a frozen radius. Richardson spends its budget 26/74 across")
print("  the two radii, and costs 2G circuit executions against G.")
print()
print(f"  {'N':>3}{'T total':>10}{'circuits':>10}{'1-cos plain':>13}"
      f"{'R* plain':>10}{'1-cos rich':>12}{'R* rich':>9}{'gain':>7}")
print("  " + "-" * 74)

for N in (4,):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)
    rows = {'plain': [], 'rich': []}

    for T in BUDGETS:
        best = {}
        for tag, rich in (('plain', False), ('rich', True)):
            ncirc_mult = 2 if rich else 1
            per = max(1, T // (3 * ncirc_mult))
            bb, br = 2.0, None
            for Rv in RGRID:
                cs = []
                for s in range(REPS):
                    with contextlib.redirect_stdout(io.StringIO()):
                        q = nisq_v6.QLTOv6(ansatz, H, shot_budget=per,
                                           sim_seed=700 + s, richardson=rich)
                    gh = np.zeros(M)
                    for act in [b['params'] for b in q.layers if b['params']]:
                        gi, _ = q.sense(theta, Rv, act)
                        gh += gi
                    cs.append(cosine(gh, g_ex))
                e = max(1 - float(np.mean(cs)), 1e-12)
                if e < bb:
                    bb, br = e, Rv
            best[tag] = (bb, br)
        rows['plain'].append(best['plain'][0])
        rows['rich'].append(best['rich'][0])
        print(f"  {N:>3}{T:>10}{'3 / 6':>10}{best['plain'][0]:>13.5f}"
              f"{best['plain'][1]:>10.2f}{best['rich'][0]:>12.5f}"
              f"{best['rich'][1]:>9.2f}"
              f"{best['plain'][0] / best['rich'][0]:>7.2f}", flush=True)

    lt = np.log(np.array(BUDGETS, dtype=float))
    ap = float(np.polyfit(lt, np.log(np.array(rows['plain'])), 1)[0])
    ar = float(np.polyfit(lt, np.log(np.array(rows['rich'])), 1)[0])
    print(f"       fitted 1-cos ~ T^beta:   plain {ap:+.3f}   richardson {ar:+.3f}")
    print(f"       predicted:               plain -0.667      richardson -0.800")
    print("  " + "." * 74)

print()
print("  1-cos tracks error SQUARED, so the predicted exponents are twice those on")
print("  the error itself: -2/3 for the T^(-1/3) estimator and -4/5 after the bias")
print("  is cancelled. 'gain' is plain error over Richardson error at equal TOTAL")
print("  shots, already charging Richardson its doubled circuit count.")
