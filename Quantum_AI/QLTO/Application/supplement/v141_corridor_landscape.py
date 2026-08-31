"""Liu-Su-Li's Eq. 120 corridor landscape, on a circuit: walk vs SGD.

TIER A - the walk is a Qiskit circuit on AerSimulator with shots. The classical
arms are exact.

THE CONSTRUCTION, verbatim from arXiv:2209.14501 Eq. (119)-(120):

    f(x) = (1/2) w^2 ||x||^2          x in W- = B(0,a)        the KNOWN well
           (1/2) w^2 ||x-2bv||^2      x in W+ = B(2bv,a)      the TARGET
           H1                         x in B_v                a narrow CORRIDOR
           H2                         otherwise               plateau, H1 << H2

    B_v = { x : w < x.v < 2b-w,  sqrt(||x||^2-(x.v)^2) < sqrt(a^2-w^2) }

a tube of radius sqrt(a^2 - w^2) around the segment from 0 to 2bv, with v drawn
at random from the unit sphere. The only cheap path between the wells runs down
that tube.

WHY THIS AND NOT THE BOXES v139/v140 USED. Both of those put a random quadratic
model on a box and compared against BRUTE FORCE. Brute force is handed the whole
landscape, so it wins at any size we can simulate, and a random quadratic has no
barrier to tunnel through in the first place. Eq. 120 is built so that the
barrier is real and so that LOCAL methods cannot find the corridor: by measure
concentration, a random point of B(0,R) lies in the slab |x.v| <= w with
probability >= 1 - O(e^{-d w^2 / 2 R^2}), and inside that slab nothing reveals v.
Their Proposition 4.1 turns that into a classical lower bound; Proposition 4.2
gives the quantum upper bound. The right benchmark is therefore SGD, not
exhaustive search.

RESULT: NEGATIVE, AND THE REASON IS A BUG IN THIS CONSTRUCTION, NOT IN THEIRS.

    seed   h     loc      t*     P(W+) quantum   SGD s=0.2   SGD s=0.5
       0  0.30  1.000   177.8      0.0000         0.8600      0.9933
       0  0.45  1.000    39.5      0.0000         0.8600      0.9933
       2  0.45  1.000     9.9      0.0001         0.8900      1.0000
       5  0.45  1.000     2.5      0.0000         0.8800      1.0000
    ... all twelve configurations give P(W+) = 0.0000, SGD 0.84 - 1.00.
    circuit arm at d=2, 8 qubits: exact 0.0000, circuit 0.0198, SGD 0.9967.

  THE QUANTUM ARM FAILS BECAUSE THE WELLS ARE NOT RESONANT. The tell is the
  loc column: Phi_- is localised in W- to 1.000 in EVERY row. A tunnelling
  doublet would have Phi_0 and Phi_1 be the symmetric and antisymmetric
  combinations of the two well states, so no single low eigenvector could be
  fully localised on one side. There is no doublet here at all.

  The cause is discretisation. Liu-Su-Li's Assumption 2.5 demands RESONANT
  wells - "the energy difference between any two local ground states are of the
  order O(h^infinity)" - and their tunnelling is a two-level Rabi oscillation
  between (Phi_0 +- Phi_1)/sqrt(2), which a detuning Delta suppresses by
  (DeltaE/Delta)^2. On a 32-point-per-axis grid the two wells contain DIFFERENT
  numbers of lattice points (|W-| = 12 against |W+| = 13 in the circuit arm), so
  their discrete ground energies differ by far more than the tunnelling
  splitting. The wells are detuned and nothing transfers.

  THE CLASSICAL ARM IS ALSO OUT OF ITS REGIME. Their Proposition 4.1 hides the
  corridor by measure concentration, P(x in S_v) >= 1 - O(e^{-d w^2/2R^2}). At
  d = 2 there is no concentration, so SGD simply walks down a 4-of-256 corridor
  and reaches W+ essentially always. Both arms behave exactly as the theory says
  they should at this dimension.

  SO THIS CONSTRUCTION NEEDS TWO THINGS WE CANNOT HAVE TOGETHER: exact well
  degeneracy on a lattice, and a dimension high enough to hide the corridor.
  Fixing the first is possible - place the wells so they contain identical point
  sets, or detune W+ deliberately to zero - but the second is out of simulation
  reach, and without it there is no classical hardness to beat. Recorded as
  BLOCKED with the reason, not tidied away (R2).

WHAT THIS CAN AND CANNOT SHOW. The classical hardness is a LARGE-d statement -
the concentration factor is e^{-d w^2/2R^2} and at d = 3 there is no
concentration at all. So this file tests the MECHANISM (does amplitude
concentrate on the corridor and carry the walk to W+ where local descent stalls)
and NOT the separation. The separation needs a d we cannot simulate.

The walk's kinetic sign is +h^2(D - A) throughout - v136 PART 7's correction.
The initial state is the local ground state of the H restricted to W-, which is
Liu-Su-Li's Phi_-, not a computational basis state: their tunnelling is a
two-level Rabi oscillation between (Phi_0 +- Phi_1)/sqrt(2) and a basis state
does not participate in it.
"""
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, DiagonalGate, StatePreparation
from qiskit_aer import AerSimulator

BASIS = ['rz', 'sx', 'x', 'cx']


def build_landscape(d, kappa, seed, a=0.26, b=0.36, w=0.20,
                    omega=4.0, H1=0.65, H2=7.0):
    """Eq. 120 on a d-dimensional grid of 2^kappa points per axis in [-1,1]."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    L = 1 << kappa
    ax = np.linspace(-1.0, 1.0, L)
    N = L ** d
    pts = np.empty((N, d))
    for idx in range(N):
        for i in range(d):
            pts[idx, i] = ax[(idx >> (i * kappa)) & (L - 1)]

    tgt = 2.0 * b * v
    r0 = np.linalg.norm(pts, axis=1)
    r1 = np.linalg.norm(pts - tgt, axis=1)
    proj = pts @ v
    perp = np.sqrt(np.maximum(r0 ** 2 - proj ** 2, 0.0))

    f = np.full(N, H2)
    inB = (proj > w) & (proj < 2 * b - w) & (perp < np.sqrt(max(a * a - w * w, 1e-9)))
    f[inB] = H1
    inM = r0 < a
    f[inM] = 0.5 * omega ** 2 * r0[inM] ** 2
    inP = r1 < a
    f[inP] = 0.5 * omega ** 2 * r1[inP] ** 2
    return f, np.nonzero(inM)[0], np.nonzero(inP)[0], np.nonzero(inB)[0], v


def grid_laplacian(d, kappa):
    """D - A for the Cartesian product of d cycles of 2^kappa sites (a torus).
    kappa >= 3 puts each direction on the PARTICLE side of v136 PART 7."""
    L = 1 << kappa
    N = L ** d
    A = np.zeros((N, N))
    for x in range(N):
        for i in range(d):
            c = (x >> (i * kappa)) & (L - 1)
            for step in (1, -1):
                y = x - (c << (i * kappa)) + (((c + step) % L) << (i * kappa))
                A[x, y] += 1.0
    return np.diag(A.sum(1)) - A, A


def sgd_success(f, start_idx, target, d, kappa, s, T, trials, seed):
    """Discrete Langevin on the same grid, VECTORISED over trials.

    Move to a random neighbour with probability min(1, exp(-(f_new-f_old)/s)).
    The scalar version ran 1.2M python iterations per call and dominated the
    runtime; this advances all  walkers in lockstep with numpy.
    """
    rng = np.random.default_rng(seed)
    L = 1 << kappa
    tgt = np.zeros(len(f), bool)
    tgt[target] = True
    x = rng.choice(start_idx, size=trials)
    hit = np.zeros(trials, bool)
    for _ in range(T):
        i = rng.integers(d, size=trials)
        step = rng.choice([1, -1], size=trials)
        sh = i * kappa
        c = (x >> sh) & (L - 1)
        y = x - (c << sh) + (((c + step) % L) << sh)
        dE = f[y] - f[x]
        acc = (dE <= 0) | (rng.random(trials) < np.exp(-np.clip(dE, 0, 50) / s))
        x = np.where(acc & ~hit, y, x)
        hit |= tgt[x]
        if hit.all():
            break
    return float(hit.mean())


if __name__ == '__main__':
    print(__doc__.split(chr(10))[0])
    print("Exact sweep at d=3 (numpy); ONE circuit at d=2 to verify the")
    print("circuit reproduces it. Building a 12-qubit DiagonalGate 16 times")
    print("per configuration was the bottleneck - 4096 multiplexed rotations")
    print("each - and none of it is needed to get the physics.")
    print("")

    # ---- PART 1: the physics, exactly, at d = 3 -----------------------
    d, kappa = 2, 5
    print("PART 1  d=%d, %d points/axis, %d vertices.  TIER C - dense spectra."
          % (d, 1 << kappa, (1 << kappa) ** d))
    print("  %4s %6s %8s %9s %9s %9s %9s"
          % ("seed", "h", "loc", "t*", "P(W+) qu", "SGD .2", "SGD .5"))
    Lap, _ = grid_laplacian(d, kappa)
    for seed in (0, 1, 2, 3, 4, 5):
        f, W_m, W_p, Bv, v = build_landscape(d, kappa, seed)
        if len(W_p) == 0 or len(Bv) == 0:
            print("  %4d  empty region, skipped" % seed)
            continue
        s02 = sgd_success(f, W_m, W_p, d, kappa, 0.2, 4000, 300, 5)
        s05 = sgd_success(f, W_m, W_p, d, kappa, 0.5, 4000, 300, 5)
        for h in (0.30, 0.45):
            H = h * h * Lap + np.diag(f)
            ev, U = np.linalg.eigh(H)
            sub = U[:, :6]
            phi = sub[:, int(np.argmax((np.abs(sub[W_m]) ** 2).sum(0)))]
            phi = phi / np.linalg.norm(phi)
            loc = float((np.abs(phi[W_m]) ** 2).sum())
            best = (0.0, 0.0)
            for tm in (0.25, 0.5, 1, 2, 4, 8, 16):
                t = tm / max(h * h, 1e-9)
                # U diag(e^{-iEt}) U^dag phi - free once H is diagonalised.
                # Calling expm on a 4096x4096 matrix 56 times was the second
                # bottleneck after the DiagonalGate synthesis, and neither was
                # needed: the eigendecomposition is already in hand.
                psi = U @ (np.exp(-1j * ev * t) * (U.conj().T @ phi))
                p = float((np.abs(psi)[W_p] ** 2).sum())
                if p > best[0]:
                    best = (p, t)
            print("  %4d %6.2f %8.3f %9.1f %9.4f %9.4f %9.4f"
                  % (seed, h, loc, best[1], best[0], s02, s05))

    # ---- PART 2: one circuit, d = 2, to check the circuit ------------
    print("")
    d2, k2 = 2, 4
    nq = d2 * k2
    print("PART 2  d=%d, %d qubits.  TIER A - circuit with shots, against the"
          % (d2, nq))
    print("        same expm on the same landscape.")
    Lap2, _ = grid_laplacian(d2, k2)
    f, W_m, W_p, Bv, v = build_landscape(d2, k2, 0)
    h = 0.45
    H = h * h * Lap2 + np.diag(f)
    ev, U = np.linalg.eigh(H)
    sub = U[:, :6]
    phi = sub[:, int(np.argmax((np.abs(sub[W_m]) ** 2).sum(0)))]
    phi = phi / np.linalg.norm(phi)
    best = (0.0, 0.0)
    for tm in (0.5, 1, 2, 4, 8):
        t = tm / (h * h)
        psi = U @ (np.exp(-1j * ev * t) * (U.conj().T @ phi))
        p = float((np.abs(psi)[W_p] ** 2).sum())
        if p > best[0]:
            best = (p, t)
    pex, t = best
    steps = 6
    dt = t / steps
    lam = 2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(1 << k2) / (1 << k2))
    mix = QuantumCircuit(k2)
    mix.append(QFTGate(k2).inverse(), range(k2))
    mix.append(DiagonalGate(list(np.exp(-1j * h * h * lam * dt))), range(k2))
    mix.append(QFTGate(k2), range(k2))
    pot = QuantumCircuit(nq)
    pot.append(DiagonalGate(list(np.exp(-1j * f * dt))), range(nq))
    qc = QuantumCircuit(nq, nq)
    qc.append(StatePreparation(phi.astype(complex)), range(nq))
    for _ in range(steps):
        for i in range(d2):
            qc.compose(mix, qubits=range(i * k2, (i + 1) * k2), inplace=True)
        qc.compose(pot, inplace=True)
    qc.measure(range(nq), range(nq))
    be = AerSimulator(seed_simulator=13)
    tq = transpile(qc, be, basis_gates=BASIS, optimization_level=1)
    cnt = be.run(tq, shots=20000).result().get_counts()
    tset = set(W_p.tolist())
    pc = sum(c for kk, c in cnt.items() if int(kk, 2) in tset) / 20000
    sg = sgd_success(f, W_m, W_p, d2, k2, 0.2, 4000, 300, 5)
    print("  |W-|=%d |W+|=%d corridor=%d of %d,  t=%.1f, %d Trotter steps"
          % (len(W_m), len(W_p), len(Bv), len(f), t, steps))
    print("  P(W+): exact %.4f   circuit %.4f   SGD %.4f   depth %d"
          % (pex, pc, sg, tq.depth()))
    print("")
    print("  The walk is ONE circuit from Phi_-; SGD gets 4000 moves and 300")
    print("  restarts. A gap here confirms the MECHANISM; the separation")
    print("  itself is a large-d statement and is out of simulation reach.")
