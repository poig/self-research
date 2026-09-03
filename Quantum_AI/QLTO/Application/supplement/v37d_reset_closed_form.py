"""Does resetting the ancilla remove the wrap? Answered in closed form.

v37c validated v37b's derivation: the bare walk's decode, computed from 2x2
products with a global post-selection, matches the simulator to 0.00241 - shot
noise. So the algebra is trustworthy, and the reset variant can be evaluated the
same way instead of waiting on a mid-circuit-measurement simulation, which Aer
must run shot-by-shot.

THE RESET VARIANT IN CLOSED FORM. Each step gets a fresh ancilla in |+>, applies
controlled-V, rotates with H, and is measured. When that outcome is DISCARDED the
param register undergoes

    rho -> (I+V)/2 rho (I+V)^dag/2 + (I-V)/2 rho (I-V)^dag/2  =  (rho + V rho V^dag)/2

a dephasing channel about V's rotation axis - NOT a rotation. This is the whole
difference. In the shipped walk the k steps compose as a PRODUCT of rotations, so
the angle adds and the Bloch response is periodic in g. Under reset they compose
as a product of CHANNELS, which contracts the Bloch vector toward V's axis and
cannot overshoot. Only the final step is conditioned, exactly as the shipped
decode conditions on its one ancilla.

    base    rho -> U rho U^dag             U = prod_s V_s        angle adds, WRAPS
    reset   rho -> E_k( ... E_1(rho))      E_s(rho)=(rho+V_s rho V_s)/2, CONTRACTS

Both then apply the same conditioned last step and the same decode,
d_theta_i = R(2 P(x_i=1) - 1), so any difference in the transfer function is the
reset and nothing else.

This is the BARE model - no W gate, no energy imprint. v37c measured what those
add: W contributes 0.063 and the imprint 0.325, and arm C reproduced v36's
shipped-walk numbers. The imprint therefore dominates the VALUES. But the wrap
lives in the drift accumulation, which is precisely what the bare model captures,
so the bare model is the right instrument for the monotonicity question and the
wrong one for predicting absolute displacements. Both claims are checked below.
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


def step_angles(step, k, dt, R, g_i):
    s = (step + 0.5) / k
    al = g_i * (s * np.pi * dt) * 0.5 * np.pi / np.sqrt(max(R, 1e-9))
    be = (1.0 - s) * np.pi * dt
    return al, be


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


def _condition(rho, V, n, R):
    """Final Hadamard + anc=1 post-selection, with base's damping fallback."""
    K = (np.eye(2 ** n) - V) / 2
    new = K @ rho @ K.conj().T
    tr = float(np.real(np.trace(new)))
    if tr < 0.05:                                   # base branches on activation
        K = (np.eye(2 ** n) + V) / 2
        new = K @ rho @ K.conj().T
        tr = float(np.real(np.trace(new)))
        if tr < 1e-12:
            return np.zeros(n)
        return 0.3 * marginals(new / tr, n, R)
    return marginals(new / tr, n, R)


def run(gvec, k, dt, R, reset):
    n = len(gvec)
    plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho = kron_all([plus] * n)
    Vs = [kron_all([expm2(*step_angles(s, k, dt, R, g)) for g in gvec])
          for s in range(k)]

    if not reset:
        # ONE ancilla controls every step, so the anc=0 branch is the identity and
        # the whole product sits inside the conditioning: K = (I - prod V_s)/2.
        U = np.eye(2 ** n, dtype=complex)
        for V in Vs:
            U = V @ U
        return _condition(rho, U, n, R)

    # Fresh ancilla per step. Discarded outcomes give a channel; only the last
    # step is conditioned, mirroring base's single-bit decode.
    for V in Vs[:-1]:
        rho = 0.5 * (rho + V @ rho @ V.conj().T)
    return _condition(rho, Vs[-1], n, R)


N_ACT, R, DT, KS = 4, 0.6, 0.5, 15
grid = [-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.15, -0.05,
        0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]

print("=" * 92)
print("ANCILLA RESET IN CLOSED FORM — coherent product vs product of channels")
print("=" * 92)
print(f"  n_active={N_ACT}, R={R}, dt={DT}, k={KS}. Bare model: no W gate, no imprint.")
print("  Validated against the simulator by v37c at 0.00241 for the base arm.")

print(f"\n  (1) TRANSFER FUNCTION")
print(f"  {'g_0':>8}{'base':>13}{'reset':>13}")
print("  " + "-" * 34)
yb, yr = [], []
for g0 in grid:
    gv = np.zeros(N_ACT); gv[0] = g0
    b = run(gv, KS, DT, R, False)[0]
    r = run(gv, KS, DT, R, True)[0]
    yb.append(b); yr.append(r)
    print(f"  {g0:>8.2f}{b:>13.5f}{r:>13.5f}")

xs = np.array(grid)
print(f"\n  {'arm':>8}{'turns':>8}{'monotone':>10}{'|d| max':>10}{'corr(d,g)':>12}"
      f"{'small-g slope':>15}")
print("  " + "-" * 63)
for nm, y in (('base', np.array(yb)), ('reset', np.array(yr))):
    d = np.diff(y)
    turns = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
    lin = np.abs(xs) <= 0.3
    slope = float(np.polyfit(xs[lin], y[lin], 1)[0])
    print(f"  {nm:>8}{turns:>8}{str(mono):>10}{np.max(np.abs(y)):>10.4f}"
          f"{float(np.corrcoef(xs, y)[0, 1]):>12.4f}{slope:>15.4f}")

print("\n  (2) SIGN CHANGES over |g| <= 1.6, finely sampled — a monotone map has 1.")
fine = np.linspace(-1.6, 1.6, 321)
for nm, rs in (('base', False), ('reset', True)):
    v = np.array([run(np.pad([g], (0, N_ACT - 1)), KS, DT, R, rs)[0] for g in fine])
    s = np.sign(v)
    c = fine[:-1][s[:-1] * s[1:] < 0]
    print(f"      {nm:>6}: {len(c):>3} crossings"
          + (f"   at {', '.join(f'{x:+.2f}' for x in c[:8])}" if len(c) else ""))

print("\n  (3) SEPARABILITY — vary g_0, others fixed. The anc=1 conditioning is")
print("      global in BOTH arms, so neither is expected to be separable.")
print(f"  {'arm':>8}{'g_0':>8}{'others=0':>12}{'others=+0.5':>14}{'spread':>9}")
print("  " + "-" * 51)
for nm, rs in (('base', False), ('reset', True)):
    for g0 in (-1.5, -0.3, 0.3):
        row = []
        for other in (0.0, 0.5):
            gv = np.full(N_ACT, other); gv[0] = g0
            row.append(run(gv, KS, DT, R, rs)[0])
        print(f"  {nm:>8}{g0:>8.2f}{row[0]:>12.5f}{row[1]:>14.5f}"
              f"{abs(row[1] - row[0]):>9.5f}")

print("\n  (4) STEP-COUNT DEPENDENCE — base's angle grows with k, reset's saturates.")
print(f"  {'k':>5}{'base @g=-0.6':>15}{'reset @g=-0.6':>16}")
print("  " + "-" * 36)
for k in (3, 5, 10, 15, 25, 40):
    gv = np.zeros(N_ACT); gv[0] = -0.6
    print(f"  {k:>5}{run(gv, k, DT, R, False)[0]:>15.5f}"
          f"{run(gv, k, DT, R, True)[0]:>16.5f}")

print()
print("  If reset shows 1 crossing where base shows many, the reset removes the")
print("  wrap and the walk becomes a usable update rule. Section (4) is the")
print("  independent check: base's response should keep oscillating as k grows")
print("  while reset's should settle, because channels contract and rotations do")
print("  not. That is the same statement as (2) from a different direction.")
