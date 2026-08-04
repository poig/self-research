"""Put the gradient in the AXIS only, not the angle. Closed form, no simulator.

The merged walk builds each step as exp(-i(al Z + be X)/2) and parametrises it as

    th = hypot(al, be)     rotation ANGLE, proportional to |g|, UNBOUNDED
    ph = atan2(be, al)     axis TILT, an arctangent, BOUNDED

The gradient enters both. v37b/c/d showed the angle channel accumulates over k
steps to ~23.9 g and wraps, destroying the magnitude. The axis channel cannot
wrap: atan2 is bounded by construction, so a scale error in g moves it by a
bounded increment. That is exactly the recorded behaviour - "direction survives,
magnitude does not matter", 158x Trotter error changing nothing, per-block scale
errors of 2x never stopping convergence - and it is the polar decomposition of a
controlled rotation rather than a robustness accident.

It also says why the rescale fix was energy-neutral. Scaling al down shrinks BOTH
channels: the wrap goes away and so does the step, which is precisely what v37f
measured (|d| at |g|=0.6 falling 0.417 -> 0.055) and why v37g found it identical
to base at matched |move|.

THE VARIANT UNDER TEST decouples them:

    axis_only:   th = TH0 (a constant),  ph = atan2(be, al)

so the accumulated angle is k*TH0 regardless of the gradient - no wrap - while
the direction still rides the unbounded-in-g but bounded-in-value arctangent.
This is not the rescale: the step size is set by TH0 and is independent of |g|,
where the rescale made the step proportional to a shrunken |g|.

Evaluated with the closed form validated against the simulator at 0.00241
(v37c, arm A), so this costs nothing and is exact. Reported against the shipped
walk and against the rescale, on the two properties that matter:

    crossings   sign changes over |g| <= 1.6; a monotone map has at most 1
    knee        where the response reaches half its maximum, against the
                measured operating range |g| = 0.58-0.97
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rot(nz, nx, th):
    """exp(-i th (nz Z + nx X)/2) with (nz,nx) a unit vector."""
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * (nz * Z + nx * X)


def step_op(step, k, dt, R, g, mode, th0, scale):
    s = (step + 0.5) / k
    al = g * (s * np.pi * dt) * 0.5 * np.pi / np.sqrt(max(R, 1e-9))
    be = (1.0 - s) * np.pi * dt
    if mode == 'rescale':
        al *= scale
    h = float(np.hypot(al, be))
    if h < 1e-15:
        return I2.copy()
    nz, nx = al / h, be / h
    th = th0 if mode == 'axis_only' else h
    return rot(nz, nx, th)


def kron_all(ms):
    o = np.ones(1, dtype=complex)          # 1-D: these are state vectors
    for m in ms:
        o = np.kron(o, m)
    return o


def decode(gvec, k, dt, R, mode, th0=0.35, scale=1.0):
    n = len(gvec)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ups = []
    for g in gvec:
        U = I2.copy()
        for s in range(k):
            U = step_op(s, k, dt, R, g, mode, th0, scale) @ U
        ups.append(U @ plus)
    psi = kron_all([plus] * n)
    upsi = kron_all(ups)
    amp = (psi - upsi) / 2.0
    p = np.abs(amp) ** 2
    tot = p.sum()
    damp = 1.0
    if tot < 0.05:                     # base's activation fallback
        p = np.abs((psi + upsi) / 2.0) ** 2
        tot = p.sum(); damp = 0.3
        if tot < 1e-15:
            return np.zeros(n)
    p /= tot
    idx = np.arange(2 ** n)
    return np.array([damp * R * (2 * float(p[((idx >> (n - 1 - i)) & 1) == 1].sum()) - 1)
                     for i in range(n)])


N_ACT, R, DT, KS = 4, 0.6, 0.5, 15
TOT = np.pi * DT * KS / 2 * 0.5 * np.pi / np.sqrt(R)
SCALE = np.pi / TOT
MODES = [('base', dict(mode='base')),
         ('rescale', dict(mode='rescale', scale=SCALE)),
         ('axis_only', dict(mode='axis_only', th0=0.35))]

print("=" * 92)
print("GRADIENT IN THE AXIS ONLY — decoupling direction from step size")
print("=" * 92)
print(f"  n_active={N_ACT}, R={R}, dt={DT}, k={KS}. Closed form (validated 0.00241).")
print(f"  base: th=hypot(al,be) wraps at ~23.9 g.  axis_only: th fixed at 0.35,")
print(f"  direction carried entirely by ph=atan2(be,al), which cannot wrap.")

grid = [-2.0, -1.5, -1.0, -0.6, -0.3, -0.15, -0.05,
        0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]
print(f"\n  (1) TRANSFER FUNCTION")
print(f"  {'g_0':>8}" + "".join(f"{m:>13}" for m, _ in MODES))
print("  " + "-" * (8 + 13 * len(MODES)))
cols = {m: [] for m, _ in MODES}
for g0 in grid:
    gv = np.zeros(N_ACT); gv[0] = g0
    row = []
    for m, kw in MODES:
        v = decode(gv, KS, DT, R, **kw)[0]
        cols[m].append(v); row.append(v)
    print(f"  {g0:>8.2f}" + "".join(f"{v:>13.5f}" for v in row))

xs = np.array(grid)
fine = np.linspace(-1.6, 1.6, 321)
print(f"\n  {'mode':>11}{'crossings':>11}{'turns':>7}{'corr':>9}{'|d|max':>9}"
      f"{'knee':>8}{'|d|@0.6':>10}{'|d|@1.0':>10}")
print("  " + "-" * 75)
for m, kw in MODES:
    y = np.array(cols[m])
    d = np.diff(y)
    turns = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    v = np.array([decode(np.pad([g], (0, N_ACT - 1)), KS, DT, R, **kw)[0]
                  for g in fine])
    sg = np.sign(v)
    cr = int(np.sum(sg[:-1] * sg[1:] < 0))
    dense = np.linspace(0.01, 3.0, 200)
    dv = np.array([abs(decode(np.pad([-g], (0, N_ACT - 1)), KS, DT, R, **kw)[0])
                   for g in dense])
    knee = float(dense[np.argmax(dv >= dv.max() / 2)]) if dv.max() > 1e-9 else np.nan
    a6 = abs(decode(np.pad([-0.6], (0, N_ACT - 1)), KS, DT, R, **kw)[0])
    a10 = abs(decode(np.pad([-1.0], (0, N_ACT - 1)), KS, DT, R, **kw)[0])
    print(f"  {m:>11}{cr:>11}{turns:>7}{float(np.corrcoef(xs, y)[0, 1]):>9.4f}"
          f"{np.max(np.abs(y)):>9.4f}{knee:>8.3f}{a6:>10.4f}{a10:>10.4f}")

print(f"\n  (2) TH0 SWEEP — the step size is now an explicit knob, not a")
print(f"      consequence of |g|. Which TH0 keeps the response monotone AND")
print(f"      large enough to move? Operating range is |g| = 0.58-0.97.")
print(f"  {'th0':>8}{'crossings':>11}{'corr':>9}{'|d|max':>9}{'|d|@0.6':>10}"
      f"{'knee':>8}")
print("  " + "-" * 55)
for th0 in (0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0):
    kw = dict(mode='axis_only', th0=th0)
    y = np.array([decode(np.pad([g], (0, N_ACT - 1)), KS, DT, R, **kw)[0]
                  for g in grid])
    v = np.array([decode(np.pad([g], (0, N_ACT - 1)), KS, DT, R, **kw)[0]
                  for g in fine])
    sg = np.sign(v)
    cr = int(np.sum(sg[:-1] * sg[1:] < 0))
    dense = np.linspace(0.01, 3.0, 200)
    dv = np.array([abs(decode(np.pad([-g], (0, N_ACT - 1)), KS, DT, R, **kw)[0])
                   for g in dense])
    knee = float(dense[np.argmax(dv >= dv.max() / 2)]) if dv.max() > 1e-9 else np.nan
    a6 = abs(decode(np.pad([-0.6], (0, N_ACT - 1)), KS, DT, R, **kw)[0])
    print(f"  {th0:>8.2f}{cr:>11}{float(np.corrcoef(np.array(grid), y)[0, 1]):>9.4f}"
          f"{np.max(np.abs(y)):>9.4f}{a6:>10.4f}{knee:>8.3f}")

print()
print("  axis_only with 0 crossings and |d|@0.6 comparable to base's 0.417 would")
print("  be the first variant that removes the wrap WITHOUT shrinking the step -")
print("  which is exactly what the rescale failed to do, and why v37g found the")
print("  rescale identical to base at matched |move|. TH0 then replaces the")
print("  accidental k-coupling with an explicit, tunable step size.")
