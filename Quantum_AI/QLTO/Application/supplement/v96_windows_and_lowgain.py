"""Are v93's chaotic bands real, and is the cascade hiding below gain 0.10?

v93's full sweep shows periodic windows NARROWING as gain rises - period 2 over
0.10-0.30 (5 wide), 0.50-0.65 (4 wide), 0.85-0.90 (2 wide), 1.00 (1 wide) -
separated by aperiodic bands. That is not a period-doubling cascade. It is what
a map looks like ABOVE its accumulation point: chaos with periodic windows
embedded in it. Two consequences follow and neither was tested.

  (A) SOME 'APERIODIC' LABELS MAY BE UNCONVERGED TRANSIENTS. The detector used
      tol = 1e-7 after only 400 transient steps. Compare the attractor spreads:
      gain 0.40 is aperiodic with spread 1.4e-1, a genuine chaotic band, while
      gain 0.70 is aperiodic with spread 2.9e-3 - far too small for chaos and
      much more like an orbit still drifting toward a cycle. Re-running the
      ambiguous gains with a 12x longer transient separates the two.

  (B) THE CASCADE MAY BE BELOW 0.10. If gain 0.10 is already past accumulation,
      the 2 -> 4 -> 8 structure sits at smaller gain and v93 never looked.
      Scanning 0.005 - 0.10 settles it.

SPEED. Both tests need far more iterations than v93, and 256 hypercube points
per gradient at ~1e-4 s per qiskit Statevector call puts a 5000-step transient
at over two minutes PER GAIN. So the energy is evaluated by a direct numpy
simulator built from the circuit's OWN decomposed gate list - faithful by
construction rather than by hand-transcription - and verified against qiskit
before use. If that check fails the script stops rather than reporting fast
wrong numbers.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp

N, REPS, R = 2, 1, 0.6
ANSATZ = efficient_su2(N, reps=REPS).decompose()
M = ANSATZ.num_parameters
HM = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)]).to_matrix()
DIM = 2 ** N

# ---- build a gate program from the circuit's own instruction list ----------
PLIST = list(ANSATZ.parameters)
PIDX = {p: i for i, p in enumerate(PLIST)}
PROG = []
for inst in ANSATZ.data:
    name = inst.operation.name
    qs = [ANSATZ.find_bit(q).index for q in inst.qubits]
    if name in ('ry', 'rz', 'rx'):
        expr = inst.operation.params[0]
        idx = PIDX[next(iter(expr.parameters))]
        PROG.append((name, qs[0], idx, 0.0))
    elif name == 'r':
        # r(theta, phi): rotation by theta about cos(phi) X + sin(phi) Y.
        # efficient_su2 emits phi = pi/2, i.e. RY, but phi is read rather than
        # assumed so a different decomposition cannot silently change the map.
        expr, phi = inst.operation.params
        idx = PIDX[next(iter(expr.parameters))]
        PROG.append(('r', qs[0], idx, float(phi)))
    elif name == 'p':
        expr = inst.operation.params[0]
        idx = PIDX[next(iter(expr.parameters))]
        PROG.append(('p', qs[0], idx, 0.0))
    elif name == 'cx':
        PROG.append(('cx', (qs[0], qs[1]), None, 0.0))
    elif name in ('barrier', 'id'):
        continue
    else:
        raise RuntimeError(f"unhandled gate {name}")


def apply_1q(psi, U, q):
    psi = psi.reshape([2] * N)
    psi = np.moveaxis(psi, q, 0).reshape(2, -1)
    psi = U @ psi
    psi = np.moveaxis(psi.reshape([2] * N), 0, q)
    return psi.reshape(-1)


def apply_cx(psi, c, t):
    psi = psi.reshape([2] * N).copy()
    sl_c1 = [slice(None)] * N
    sl_c1[c] = 1
    blk = psi[tuple(sl_c1)]
    blk = np.flip(blk, axis=t if t < c else t - 1)
    psi[tuple(sl_c1)] = blk
    return psi.reshape(-1)


def energy(th):
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    for name, q, idx, phi in PROG:
        if name == 'cx':
            psi = apply_cx(psi, q[0], q[1])
            continue
        if name == 'p':
            U = np.array([[1.0, 0.0], [0.0, np.exp(1j * th[idx])]], dtype=complex)
            psi = apply_1q(psi, U, q)
            continue
        a = th[idx] / 2.0
        c, s = np.cos(a), np.sin(a)
        if name == 'ry':
            U = np.array([[c, -s], [s, c]], dtype=complex)
        elif name == 'rx':
            U = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        elif name == 'rz':
            U = np.array([[np.exp(-1j * a), 0], [0, np.exp(1j * a)]])
        else:                                   # generic r(theta, phi)
            U = np.array([[c, -1j * np.exp(-1j * phi) * s],
                          [-1j * np.exp(1j * phi) * s, c]], dtype=complex)
        psi = apply_1q(psi, U, q)
    return float(np.real(np.conj(psi) @ (HM @ psi)))


# ---- verify against qiskit before trusting it ------------------------------
rng = np.random.default_rng(0)
err = 0.0
for _ in range(20):
    th = rng.uniform(0, 2 * np.pi, M)
    v = Statevector(ANSATZ.assign_parameters(th)).data
    ref = float(np.real(np.conj(v) @ (HM @ v)))
    err = max(err, abs(ref - energy(th)))
print(f"  numpy simulator vs qiskit, max |diff| over 20 draws: {err:.2e}")
if err > 1e-10:
    print("  MISMATCH - refusing to run on an unverified simulator.")
    sys.exit(1)
print("  verified.\n")

SIGNS = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                  for v in range(2 ** M)])


def ghat(th):
    E = np.array([energy(th + R * s) for s in SIGNS])
    g = (SIGNS * E[:, None]).mean(axis=0) / R
    mx = float(np.max(np.abs(g)))
    return g / mx if mx > 1e-14 else np.zeros_like(g)


def run(p0, gain, n_trans, n_samp=120):
    p = p0.copy()
    for _ in range(n_trans):
        p = p - gain * ghat(p)
    out = []
    for _ in range(n_samp):
        p = p - gain * ghat(p)
        out.append(energy(p))
    return np.array(out)


def period_of(traj, tol=1e-7):
    tail = traj[-60:]
    for k in (1, 2, 4, 8, 16, 32):
        if len(tail) <= k:
            break
        if np.max(np.abs(tail[:-k] - tail[k:])) < tol:
            return k
    return 0


p0 = np.random.RandomState(7).uniform(0, 2 * np.pi, M)

print("=" * 92)
print("TEST A.  Do the 'aperiodic' labels survive a 12x longer transient?")
print("=" * 92)
print(f"  {'gain':>7}{'v93 (400)':>12}{'now (5000)':>12}"
      f"{'spread':>12}{'verdict':>22}")
print("  " + "-" * 65)
V93 = {0.35: 0, 0.40: 0, 0.45: 0, 0.50: 2, 0.70: 0, 0.75: 0, 0.80: 0, 0.95: 0}
for gain in sorted(V93):
    traj = run(p0, gain, 5000)
    per = period_of(traj)
    spread = float(traj.max() - traj.min())
    old = 'aperiodic' if V93[gain] == 0 else str(V93[gain])
    new = 'aperiodic' if per == 0 else str(per)
    if V93[gain] == 0 and per != 0:
        verdict = "was TRANSIENT"
    elif V93[gain] == 0 and per == 0:
        verdict = "chaotic, confirmed"
    else:
        verdict = "unchanged"
    print(f"  {gain:>7.2f}{old:>12}{new:>12}{spread:>12.3e}{verdict:>22}",
          flush=True)

print()
print("=" * 92)
print("TEST B.  Is the cascade below gain 0.10?")
print("=" * 92)
print(f"  {'gain':>7}{'period':>10}{'spread':>12}{'E mean':>12}")
print("  " + "-" * 41)
seen = []
for gain in np.arange(0.005, 0.101, 0.005):
    traj = run(p0, float(gain), 3000)
    per = period_of(traj)
    seen.append((float(gain), per))
    tag = 'aperiodic' if per == 0 else str(per)
    print(f"  {gain:>7.3f}{tag:>10}{traj.max() - traj.min():>12.3e}"
          f"{traj.mean():>12.6f}", flush=True)

print()
print("=" * 92)
print("READING IT")
print("=" * 92)
ps = sorted({p for _, p in seen if p != 0})
print(f"  periods seen below 0.10: {ps}")
if 1 in ps:
    print("  A FIXED POINT exists at low gain, so the map does settle and the")
    print("  period-2 at 0.10 is a genuine first bifurcation above it.")
if any(p in (4, 8, 16) for p in ps):
    print("  Intermediate periods present -> the cascade IS below 0.10 and v93")
    print("  was scanning entirely above the accumulation point.")
elif ps and set(ps) <= {1, 2}:
    print("  Only periods 1 and 2 below 0.10: no cascade there either. The")
    print("  window/chaos structure is then the whole story and this map does")
    print("  not reach chaos by period doubling at any gain tested.")
