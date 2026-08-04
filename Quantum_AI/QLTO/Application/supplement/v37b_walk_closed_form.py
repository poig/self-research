"""The walk's decode in closed form. No simulator, no shots - 2x2 matrices.

Every explanation of the walk in these notes has been a story about what it ought
to do. The circuit is small enough to write down exactly, so write it down.

DERIVATION. In the merged walk each step is

    ry(-ph) ; crz(th) ; ry(ph)      th = hypot(al, beta), ph = atan2(beta, al)
    al_s = g_i * gamma_s * 0.5 pi / sqrt(R)      gamma_s = s pi dt
    beta_s = (1 - s) pi dt                       s = (step + 0.5)/k

The two RY are UNCONTROLLED, so in the anc=0 branch the CRZ is the identity and
RY(ph)RY(-ph) cancels within every step: THE anc=0 BRANCH IS EXACTLY THE IDENTITY.
In the anc=1 branch,

    RY(ph) RZ(th) RY(-ph) = exp(-i th/2 (Z cos ph + X sin ph))
                          = exp(-i (al Z + beta X)/2)

so the walk unitary factorises over param qubits, U = tensor_i U_i with

    U_i = prod_{s=k-1..0} exp(-i (al_s^i Z + beta_s X) / 2).

The circuit is h(param) -> controlled-U -> h(anc) -> measure. Before the final
Hadamard the state is (|0>|psi> + |1>U|psi>)/sqrt2 with |psi> = |+>^n, so

    anc = 1  projects param onto  (I - U)|psi> / 2.

_weighted_vertices maps bit=1 to centre+R and bit=0 to centre-R, so the decoded
displacement is exactly

    d_theta_i = R * (2 P(x_i = 1) - 1)   under the anc=1-conditioned distribution.

TWO CONSEQUENCES, both of which v36 measured and no prior explanation predicted.

  ALIASING      U_i is a product of rotations, hence a rotation by an angle that
                GROWS LINEARLY IN g_i. Its Bloch response is periodic, so
                d_theta_i oscillates in g_i instead of saturating. Total drift
                angle = g * (pi dt k / 2) * 0.5 pi / sqrt(R) ~ 23.9 g at shipped
                settings, first wrap at |g| = 0.131.

  NON-SEPARABLE Although U factorises, the anc=1 CONDITIONING IS GLOBAL:
                |(I - tensor U_i)|psi>|^2 does not factorise. Post-selection on a
                single ancilla correlates all n coordinates. This is why v36 found
                d_theta_0 shifting by 0.29 when the OTHER components changed, and
                it is a property of the decode, not of the generator algebra - the
                DLA argument was answering a different question.

This file evaluates that closed form and compares it against v36's measured base
column. Agreement means the walk is fully explained at fixed centre; the residual
is then attributable to the parts this model omits - the W gate's param-sys
entanglement and the controlled H_sense imprint, which is where the energy
dependence actually enters.
"""
import itertools
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def expm2(a, b):
    """exp(-i (a Z + b X)/2) in closed form."""
    th = float(np.hypot(a, b))
    if th < 1e-15:
        return I2.copy()
    nz, nx = a / th, b / th
    return np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * (nz * Z + nx * X)


def walk_unitary(g_i, k, dt, R):
    """U_i for one param qubit, and the total accumulated drift angle."""
    U = I2.copy()
    tot = 0.0
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    for step in range(k):
        s = (step + 0.5) / k
        al = g_i * (s * np.pi * dt) * 0.5 * np.pi * gain
        be = (1.0 - s) * np.pi * dt
        U = expm2(al, be) @ U
        tot += al
    return U, tot


def decode(gvec, k, dt, R):
    """Exact d_theta for the anc=1 branch, all n coordinates."""
    n = len(gvec)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ups = [walk_unitary(g, k, dt, R)[0] @ plus for g in gvec]

    psi = np.ones(1, dtype=complex)
    upsi = np.ones(1, dtype=complex)
    for i in range(n):
        psi = np.kron(psi, plus)
        upsi = np.kron(upsi, ups[i])
    amp = (psi - upsi) / 2.0
    p = np.abs(amp) ** 2
    tot = p.sum()
    if tot < 1e-18:                      # ancilla never fires: base damps by 0.3
        p = np.abs((psi + upsi) / 2.0) ** 2
        tot = p.sum()
        damp = 0.3
    else:
        damp = 1.0
    p = p / tot

    # qubit 0 is the FIRST kron factor => most significant index bit
    out = np.zeros(n)
    idx = np.arange(2 ** n)
    for i in range(n):
        bit = (idx >> (n - 1 - i)) & 1
        p1 = float(p[bit == 1].sum())
        out[i] = damp * R * (2 * p1 - 1)
    return out


N_ACT, R, DT, KS = 4, 0.6, 0.5, 15
tot_angle = np.pi * DT * KS / 2 * 0.5 * np.pi / np.sqrt(R)

print("=" * 92)
print("WALK DECODE IN CLOSED FORM — exact, no simulator")
print("=" * 92)
print(f"  n_active={N_ACT}, R={R}, dt={DT}, k={KS}")
print(f"  predicted drift angle = g * {tot_angle:.2f}   =>  first wrap |g| = "
      f"{np.pi / tot_angle:.4f}")

_, a1 = walk_unitary(1.0, KS, DT, R)
print(f"  measured from the product: sum al at g=1 is {a1:.2f}"
      f"   ({a1 / (2 * np.pi):.2f} full turns)")

print("\n  (1) TRANSFER — closed form vs v36's MEASURED base column (65536 shots).")
print(f"  {'g_0':>8}{'closed form':>14}{'v36 measured':>15}{'diff':>9}")
print("  " + "-" * 46)
v36 = {-3.0: 0.36210, -2.0: 0.22578, -1.5: 0.05614, -1.0: -0.23170,
       -0.6: 0.33577, -0.3: 0.02379, -0.15: 0.16126, -0.05: 0.06872,
       0.05: -0.06834, 0.15: -0.16109, 0.3: -0.01945, 0.6: -0.33566,
       1.0: 0.24172, 1.5: -0.05587, 2.0: -0.22585, 3.0: -0.36213}
errs = []
for g0 in sorted(v36):
    gv = np.zeros(N_ACT); gv[0] = g0
    d = decode(gv, KS, DT, R)[0]
    errs.append(abs(d - v36[g0]))
    print(f"  {g0:>8.2f}{d:>14.5f}{v36[g0]:>15.5f}{d - v36[g0]:>9.5f}")
rng = max(abs(v) for v in v36.values()) * 2
print(f"\n  mean |error| = {np.mean(errs):.5f}   max = {np.max(errs):.5f}"
      f"   ({np.mean(errs) / rng * 100:.1f}% of range)")

print("\n  (2) WRAP LOCATION — where the closed form changes sign, finely sampled.")
fine = np.linspace(-1.6, 1.6, 641)
vals = np.array([decode(np.pad([g], (0, N_ACT - 1)), KS, DT, R)[0] for g in fine])
sgn = np.sign(vals)
cross = fine[:-1][sgn[:-1] * sgn[1:] < 0]
print(f"      sign changes at g = "
      f"{', '.join(f'{c:+.3f}' for c in cross[:12])}"
      f"{' ...' if len(cross) > 12 else ''}")
if len(cross) > 2:
    sp = np.diff(cross)
    print(f"      spacing: mean {np.mean(sp):.4f}, predicted pi/{tot_angle:.1f} = "
          f"{np.pi / tot_angle:.4f}")
print(f"      total sign changes over |g|<=1.6: {len(cross)}"
      f"   (a monotone map has at most 1)")

print("\n  (3) SEPARABILITY — vary g_0 with the others fixed, closed form only.")
print(f"  {'g_0':>8}{'others=0':>12}{'others=+0.5':>14}{'others=-0.5':>14}"
      f"{'spread':>9}")
print("  " + "-" * 57)
for g0 in (-1.5, -0.3, 0.3, 1.5):
    row = []
    for other in (0.0, 0.5, -0.5):
        gv = np.full(N_ACT, other); gv[0] = g0
        row.append(decode(gv, KS, DT, R)[0])
    print(f"  {g0:>8.2f}{row[0]:>12.5f}{row[1]:>14.5f}{row[2]:>14.5f}"
          f"{max(row) - min(row):>9.5f}")

print("\n  (4) WHAT A NON-WRAPPING SCHEDULE LOOKS LIKE — same circuit, drift")
print("      rescaled so the total angle is pi at |g|=1 instead of 23.9.")
print(f"  {'g_0':>8}{'shipped':>12}{'unwrapped':>12}")
print("  " + "-" * 32)
scale = np.pi / tot_angle
for g0 in (-1.5, -1.0, -0.6, -0.3, 0.3, 0.6, 1.0, 1.5):
    gv = np.zeros(N_ACT); gv[0] = g0
    a = decode(gv, KS, DT, R)[0]
    gv2 = np.zeros(N_ACT); gv2[0] = g0 * scale
    b = decode(gv2, KS, DT, R)[0]
    print(f"  {g0:>8.2f}{a:>12.5f}{b:>12.5f}")

print()
print("  If column (1) tracks v36, the walk at fixed centre is fully described by")
print("  a 2x2 product and a global post-selection - no quantum advantage is")
print("  hiding in it, and the aliasing is arithmetic rather than noise.")
