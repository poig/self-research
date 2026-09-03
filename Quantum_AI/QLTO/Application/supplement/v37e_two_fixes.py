"""Two candidate fixes for the wrap. Which one, and what does each actually buy?

v36 found the walk's transfer function non-monotonic and non-separable. v37b/c
traced both to the algebra and validated the derivation against the simulator at
0.00241. Two defects, and they have DIFFERENT causes:

  WRAP            one ancilla controls all k steps, so they compose as a product
                  of ROTATIONS and the drift angle adds without bound. Periodic
                  in g => 10 sign crossings over |g|<=1.6.
  NON-SEPARABLE   the anc=1 post-selection is GLOBAL, so |(I - tensor U_i)|psi>|^2
                  does not factorise however small the angles are.

Two fixes are on the table and they are not equally priced:

  rescale   divide the drift by the accumulated-angle factor so the total stays
            under pi. ONE CONSTANT in the existing code. No new gates, no
            mid-circuit measurement, nothing to ask of hardware.
  reset     fresh ancilla each step. The k steps then compose as CHANNELS,
            rho -> (rho + V rho V)/2, which contract instead of rotating. Costs a
            mid-circuit reset per step, and on the real circuit an energy imprint
            per step - k controlled evolutions instead of 1.

The rescale should fix the WRAP only, since the post-selection is untouched. The
reset should fix BOTH, because discarding intermediate ancillas removes the
conditioning that correlated the coordinates. If that is what the numbers say,
the choice is not "which fix" but "is separability worth k imprints" - and that
is a question about what the walk is FOR, which the notes have never had to
answer because nobody knew the coupling was there.

Bare model throughout: no W gate, no energy imprint. v37c measured what those add
(0.063 and 0.325) and confirmed the bare model carries the STRUCTURE while the
imprint carries the VALUES. Structure is what is at issue here.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def expm2(a, b):
    th = float(np.hypot(a, b))
    if th < 1e-15:
        return I2.copy()
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * ((a / th) * Z + (b / th) * X)


def kron_all(mats):
    out = np.ones((1, 1), dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def marginals(rho, n, R):
    p = np.real(np.diag(rho))
    p = p / p.sum()
    idx = np.arange(2 ** n)
    return np.array([R * (2 * float(p[((idx >> (n - 1 - i)) & 1) == 1].sum()) - 1)
                     for i in range(n)])


def run(gvec, k, dt, R, mode, scale=1.0):
    """mode in {'base', 'rescale', 'reset'}. scale divides the drift only."""
    n = len(gvec)
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    sc = scale if mode == 'rescale' else 1.0

    Vs = []
    for step in range(k):
        s = (step + 0.5) / k
        be = (1.0 - s) * np.pi * dt
        Vs.append(kron_all([expm2(g * (s * np.pi * dt) * 0.5 * np.pi * gain * sc, be)
                            for g in gvec]))

    plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho = kron_all([plus] * n)

    if mode == 'reset':
        for V in Vs[:-1]:
            rho = 0.5 * (rho + V @ rho @ V.conj().T)
        last = Vs[-1]
    else:
        last = np.eye(2 ** n, dtype=complex)
        for V in Vs:
            last = V @ last

    K = (np.eye(2 ** n) - last) / 2
    new = K @ rho @ K.conj().T
    tr = float(np.real(np.trace(new)))
    if tr < 0.05:
        K = (np.eye(2 ** n) + last) / 2
        new = K @ rho @ K.conj().T
        tr = float(np.real(np.trace(new)))
        return 0.3 * marginals(new / tr, n, R) if tr > 1e-12 else np.zeros(n)
    return marginals(new / tr, n, R)


N_ACT, R, DT, KS = 4, 0.6, 0.5, 15
tot = np.pi * DT * KS / 2 * 0.5 * np.pi / np.sqrt(R)
SCALE = np.pi / tot          # total drift angle = pi at |g| = 1

MODES = [('base', 'base', 1.0), ('rescale', 'rescale', SCALE),
         ('reset', 'reset', 1.0)]

print("=" * 92)
print("TWO FIXES FOR THE WRAP — rescale the drift, or reset the ancilla")
print("=" * 92)
print(f"  n_active={N_ACT}, R={R}, dt={DT}, k={KS}")
print(f"  shipped accumulated angle = g * {tot:.2f};  rescale factor = {SCALE:.4f}")

grid = [-2.0, -1.5, -1.0, -0.6, -0.3, -0.15, -0.05,
        0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]
print(f"\n  (1) TRANSFER FUNCTION")
print(f"  {'g_0':>8}" + "".join(f"{m:>13}" for m, _, _ in MODES))
print("  " + "-" * (8 + 13 * len(MODES)))
cols = {m: [] for m, _, _ in MODES}
for g0 in grid:
    gv = np.zeros(N_ACT); gv[0] = g0
    row = []
    for nm, md, sc in MODES:
        v = run(gv, KS, DT, R, md, sc)[0]
        cols[nm].append(v); row.append(v)
    print(f"  {g0:>8.2f}" + "".join(f"{v:>13.5f}" for v in row))

xs = np.array(grid)
print(f"\n  {'mode':>10}{'crossings':>11}{'turns':>7}{'corr(d,g)':>12}"
      f"{'|d| max':>10}{'slope':>10}")
print("  " + "-" * 60)
fine = np.linspace(-1.6, 1.6, 321)
for nm, md, sc in MODES:
    y = np.array(cols[nm])
    d = np.diff(y)
    turns = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    v = np.array([run(np.pad([g], (0, N_ACT - 1)), KS, DT, R, md, sc)[0]
                  for g in fine])
    s = np.sign(v)
    cr = int(np.sum(s[:-1] * s[1:] < 0))
    lin = np.abs(xs) <= 0.3
    print(f"  {nm:>10}{cr:>11}{turns:>7}"
          f"{float(np.corrcoef(xs, y)[0, 1]):>12.4f}{np.max(np.abs(y)):>10.4f}"
          f"{float(np.polyfit(xs[lin], y[lin], 1)[0]):>10.4f}")

print("\n  (2) SEPARABILITY — d_theta_0 as the OTHER components move.")
print("      The rescale leaves the global post-selection in place; the reset")
print("      discards it. This is where the two fixes should part company.")
print(f"  {'mode':>10}{'g_0':>7}{'others=0':>12}{'others=+0.5':>14}"
      f"{'others=-0.5':>14}{'spread':>9}")
print("  " + "-" * 66)
worst = {}
for nm, md, sc in MODES:
    w = 0.0
    for g0 in (-1.5, -0.6, -0.3, 0.3):
        row = []
        for other in (0.0, 0.5, -0.5):
            gv = np.full(N_ACT, other); gv[0] = g0
            row.append(run(gv, KS, DT, R, md, sc)[0])
        sp = max(row) - min(row)
        w = max(w, sp)
        print(f"  {nm:>10}{g0:>7.2f}{row[0]:>12.5f}{row[1]:>14.5f}"
              f"{row[2]:>14.5f}{sp:>9.5f}")
    worst[nm] = w
    print("  " + "." * 66)

print(f"\n  worst-case spread:  " +
      "   ".join(f"{k} {v:.4f}" for k, v in worst.items()))

print("\n  (3) WHERE THE KNEE SITS relative to the gradients actually measured.")
print("      The benchmark's sensed |g| is 0.58-0.97. A fix that saturates well")
print("      below that range is sign-descent: correct, but magnitude-blind.")
print(f"  {'mode':>10}{'|d| @0.1':>11}{'@0.3':>9}{'@0.6':>9}{'@1.0':>9}"
      f"{'@2.0':>9}{'knee':>8}")
print("  " + "-" * 65)
probe = [0.1, 0.3, 0.6, 1.0, 2.0]
for nm, md, sc in MODES:
    vals = []
    for g0 in probe:
        gv = np.zeros(N_ACT); gv[0] = -g0
        vals.append(abs(run(gv, KS, DT, R, md, sc)[0]))
    dense = np.linspace(0.01, 3.0, 300)
    dv = np.array([abs(run(np.pad([-g], (0, N_ACT - 1)), KS, DT, R, md, sc)[0])
                   for g in dense])
    half = dv.max() / 2
    knee = float(dense[np.argmax(dv >= half)]) if dv.max() > 1e-9 else float('nan')
    print(f"  {nm:>10}" + "".join(f"{v:>11.4f}" if i == 0 else f"{v:>9.4f}"
                                  for i, v in enumerate(vals))
          + f"{knee:>8.3f}")

print()
print("  'knee' is where the response reaches half its maximum. If it sits far")
print("  below 0.58, the walk is a bounded SIGN step and the gradient's magnitude")
print("  is discarded by construction - which would finally explain, exactly, the")
print("  notes' recurring observation that direction survives and magnitude does")
print("  not matter. Moving the knee into the operating range is then a separate,")
print("  and cheaper, tuning question than either fix above.")
