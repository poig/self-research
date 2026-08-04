"""What map does the walk actually implement? Measured, not argued.

These notes contain many measurements ABOUT the walk and no statement of what it
COMPUTES. It ties a classical Boltzmann decode; its parameter-register dynamics has
DLA su(2)^n and so is separable; k_steps acts as a step-size multiplier with
sum_step s = k/2; zeroing the gradient costs 4.32 Hartree while random drift is
worse than none; the step is bounded by R; it survives ansatz depth. Every one of
those is a property. None of them says what the walk does to its input.

That gap is why three explanations in a row failed - each proposed a mechanism
before anyone had measured the input-output relation.

So measure it. Feed the walk a SYNTHETIC gradient vector, record the resulting
parameter displacement, and ask two questions with no theory attached:

  SEPARABILITY   is delta_theta_i a function of g_i alone, or does it depend on the
                 other components? The DLA argument says the former, but that
                 argument is about the generator algebra, not about the decode,
                 and _decode_walk takes a weighted mean over sampled corners which
                 could couple coordinates through the shot record.

  TRANSFER       what function is it? A bounded, sign-preserving, saturating map is
                 what the pieces suggest - phase proportional to g_i, mixed by CRX,
                 decoded to a weighted mean of +-R corners - but "suggest" is how
                 the last three explanations started.

If delta_theta_i collapses onto a single curve in g_i, the walk is a classical
bounded update and can be written down in closed form. That would be an accurate
account of the mechanism, and it would also say precisely what the quantum circuit
is contributing, which at present nobody can state.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


N, R, DT, KS, SHOTS = 4, 0.6, 0.5, 15, 65536
H = heis(N)
ansatz = efficient_su2(N, reps=1)
M = ansatz.num_parameters
q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=17)
act = [b['params'] for b in q.layers if b['params']][0]
n = len(act)
centre = np.random.RandomState(7).uniform(-np.pi, np.pi, M)


def step(gvec, reps=4):
    """Mean parameter displacement produced by the walk for a given gradient."""
    out = []
    for r in range(reps):
        q.reset_shot_stream()
        g = np.zeros(M)
        g[act] = gvec
        p = q._execute_walk(centre, KS, DT, R, act, g)
        out.append(p[act] - centre[act])
    return np.mean(out, axis=0)


print("=" * 92)
print("WALK TRANSFER FUNCTION — what map does it implement?")
print("=" * 92)
print(f"  N={N}, R={R}, dt={DT}, k_steps={KS}, {SHOTS} shots, 4 repeats per point.")

print("\n  (1) SEPARABILITY — vary g_0 with the others FIXED at three settings.")
print("      If d_theta_0 depends only on g_0, the three columns agree.")
print(f"  {'g_0':>8}{'others=0':>12}{'others=+0.5':>14}{'others=-0.5':>14}"
      f"{'spread':>9}")
print("  " + "-" * 57)
grid = [-1.5, -0.8, -0.3, -0.1, 0.0, 0.1, 0.3, 0.8, 1.5]
for g0 in grid:
    row = []
    for other in (0.0, 0.5, -0.5):
        gv = np.full(n, other)
        gv[0] = g0
        row.append(step(gv)[0])
    print(f"  {g0:>8.2f}{row[0]:>12.5f}{row[1]:>14.5f}{row[2]:>14.5f}"
          f"{max(row) - min(row):>9.5f}", flush=True)

print("\n  (2) TRANSFER — d_theta_0 against g_0, with the others zero.")
print("      R bounds the step, so look for saturation and for the sign convention.")
print(f"  {'g_0':>9}{'d_theta_0':>12}{'d/R':>9}{'ratio to g':>12}")
print("  " + "-" * 42)
fine = [-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.15, -0.05,
        0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
xs, ys = [], []
for g0 in fine:
    gv = np.zeros(n); gv[0] = g0
    d = step(gv)[0]
    xs.append(g0); ys.append(d)
    ratio = d / g0 if abs(g0) > 1e-9 else float('nan')
    print(f"  {g0:>9.2f}{d:>12.5f}{d / R:>9.4f}{ratio:>12.4f}")

xs, ys = np.array(xs), np.array(ys)
lin = xs[np.abs(xs) <= 0.3]
slope = float(np.polyfit(lin, ys[np.abs(xs) <= 0.3], 1)[0]) if len(lin) > 1 else np.nan
print(f"\n  small-g slope       : {slope:>10.4f}")
print(f"  max |d_theta|       : {np.max(np.abs(ys)):>10.4f}   (R = {R})")
print(f"  saturates within R? : {'yes' if np.max(np.abs(ys)) <= R + 1e-9 else 'NO'}")
print(f"  sign convention     : d_theta {'opposes' if slope < 0 else 'follows'} g")

print("\n  (3) A CLASSICAL SURROGATE. Fit d = -A*R*tanh(B*g) and report the residual.")
from scipy.optimize import curve_fit
try:
    popt, _ = curve_fit(lambda g, A, B: -A * R * np.tanh(B * g), xs, ys,
                        p0=[1.0, 1.0], maxfev=20000)
    pred = -popt[0] * R * np.tanh(popt[1] * xs)
    rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
    print(f"      A = {popt[0]:.4f}   B = {popt[1]:.4f}   rms residual = {rms:.5f}"
          f"   ({rms / max(np.max(np.abs(ys)), 1e-12) * 100:.2f}% of range)")
except Exception as e:
    print("      fit failed:", e)

print()
print("  A tight fit means the walk circuit is reproducible by a two-parameter")
print("  classical update, which would explain why a Boltzmann decode ties it and")
print("  would say exactly what the quantum step is contributing: nothing beyond")
print("  the sensing, at these settings. A poor fit means there is structure in the")
print("  walk that no one has characterised, and it is worth finding before")
print("  redesigning the mixer.")
