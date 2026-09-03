"""The COMPLETE walk model - mixer, W gate and imprint together. Validated or not.

Every model of the walk so far has been partial, and each partial model produced a
design that failed:

  v37b   2x2 products with a global post-selection. Matched the simulator to
         0.00241 - but only for the BARE arm, with no W gate and no imprint.
         v37c measured those contributing 0.063 and 0.325.
  v47    designed an "optimal" phase against the bare model's P ~ sin^2(phi/2).
         Produced a UNIFORM distribution in the real circuit.
  v48    added the imprint as P ~ 2 - 2 Re[e^{i phi} c_x]. Also wrong, because it
         assumed the walk acts as a pure phase on param, while CRX moves
         populations, so the walk unitary is not diagonal.

THE EXACT FORM. Write |psi_x> for the ansatz state at vertex x, U_t = e^{-i H_s t}
for the imprint, and V for the walk unitary on the PARAM register (a product of
2x2 rotations, so V is 2^n x 2^n and cheap). The circuit is

    |+>_a (x) 2^{-n/2} sum_x |x>|psi_x>
      -> controlled imprint -> controlled walk -> H on anc -> post-select anc=1

which gives, exactly,

    sum_y |y> ( |psi_y> - sum_x V_{yx} U_t |psi_x> )
    P(y) ~ || |psi_y> - sum_x V_{yx} U_t |psi_x> ||^2 .

THE PIECE EVERY EARLIER MODEL DROPPED is that the vertex states are NOT
ORTHOGONAL: <psi_x'|psi_x> is a full Gram matrix, not delta. That overlap
structure IS the param-sys entanglement, and it is what the bare model was
missing by 0.325.

This is computed by direct construction on the joint param (x) sys space - 2^n *
2^N amplitudes, which is 256 at n=N=4 and 4096 at n=N=6 - so it is exact, with no
free parameters and no fitting.

VALIDATION. Predicted P(y) is compared against the simulator by total variation
distance, and - the part that matters - against the SHOT-NOISE FLOOR measured as
the TVD between two independent simulator runs at different seeds. A prediction
is only validated if TVD(pred, meas) is comparable to TVD(meas1, meas2). Reporting
TVD against uniform as well, so a model that is merely "not obviously wrong" on a
near-uniform distribution cannot pass.

If this validates, the walk finally has a complete model and phase design can be
done against the right objective. If it does not, the remaining gap is the
Trotterisation of the imprint (the circuit uses LieTrotter reps=1 while this uses
exact expm), which is checkable by matching the synthesis.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector, Operator
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v43_phase_offset import OffsetWalk

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rz(a):
    return np.array([[np.exp(-1j * a / 2), 0], [0, np.exp(1j * a / 2)]],
                    dtype=complex)


def rx(a):
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * X


def walk_unitary(gvec, k, dt, R, n):
    """V on the param register: the anc=1 branch of the k-step walk."""
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    Vs = [I2.copy() for _ in range(n)]
    for step in range(k):
        s = (step + 0.5) / k
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        for i in range(n):
            al = gvec[i] * gamma * 0.5 * np.pi * gain
            # circuit order is crz then crx, so crx multiplies on the left
            Vs[i] = rx(beta) @ rz(al) @ Vs[i]
    # Qiskit orders qubit 0 as the LEAST significant bit, so it is the RIGHTMOST
    # kron factor. Building kron(V_0, ..., V_{n-1}) instead puts qubit 0 in the
    # most significant position and silently permutes the whole distribution -
    # which is exactly what a large TVD with mixed-sign correlation looks like.
    V = np.ones((1, 1), dtype=complex)
    for i in reversed(range(n)):
        V = np.kron(V, Vs[i])
    return V


def tvd(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("COMPLETE WALK MODEL — mixer + W gate + imprint, exact construction")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. P(y) ~ || |psi_y> - sum_x V_yx")
print(f"  U_t |psi_x> ||^2, built on the joint param(x)sys space. No fitting.")
print(f"  'floor' is TVD between two simulator runs at different seeds - the")
print(f"  prediction is validated only if TVD pred is comparable to it.")
print()
print(f"  {'problem':>15}{'blk':>4}{'n':>3}{'TVD pred':>10}{'floor':>9}"
      f"{'TVD unif':>10}{'corr':>8}{'ratio':>8}")
print("  " + "-" * 67)

for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        idx = np.array([int(''.join('1' if s[i] > 0 else '0'
                                    for i in range(n))[::-1], 2) for s in sig])

        with contextlib.redirect_stdout(io.StringIO()):
            q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                           merged_walk=False)
        q.reset_shot_stream()
        g1, _ = sense_deg12(q, centre, R, act)

        # |psi_x> for every vertex, indexed by bitmask
        dS = 2 ** N
        psis = np.empty((2 ** n, dS), dtype=complex)
        for kk, sv in enumerate(sig):
            p = centre.copy()
            p[act] = p[act] + R * sv
            psis[idx[kk]] = Statevector(ansatz.assign_parameters(p)).data

        # U_t built with the SAME synthesis the circuit uses, not exact expm
        ev = PauliEvolutionGate(q.H_sense, time=DT * np.pi,
                                synthesis=LieTrotter(reps=1))
        Ut = np.asarray(Operator(ev).data)

        V = walk_unitary(g1, KS, DT, R, n)
        UtPsi = psis @ Ut.T                       # rows: U_t|psi_x>
        branch = psis - V @ UtPsi                 # rows indexed by y
        pred = np.sum(np.abs(branch) ** 2, axis=1)
        pred = pred / max(pred.sum(), 1e-18)

        def measure(seed):
            with contextlib.redirect_stdout(io.StringIO()):
                qq = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=seed,
                                merged_walk=False)
            qq.reset_shot_stream()
            counts = qq.walk(centre, KS, DT, R, act, g1,
                             np.zeros((n, n)), False, 0.0)
            m = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    m[int(parts[1].replace(" ", ""), 2)] += c
            return m / max(m.sum(), 1)

        m1, m2 = measure(17), measure(9001)
        unif = np.ones(2 ** n) / 2 ** n
        tp, fl = tvd(pred, m1), tvd(m1, m2)
        cc = float(np.corrcoef(pred, m1)[0, 1]) if m1.std() > 1e-12 else 0.0
        print(f"  {name:>15}{bi:>4}{n:>3}{tp:>10.4f}{fl:>9.4f}"
              f"{tvd(unif, m1):>10.4f}{cc:>8.4f}"
              f"{tp / fl if fl > 1e-9 else np.inf:>8.2f}", flush=True)

print()
print("  ratio near 1 means the prediction is as close to the simulator as the")
print("  simulator is to itself - a complete model. Large ratio with high corr")
print("  means the SHAPE is right and something scales it. Large ratio with low")
print("  corr means the model is still missing structure, and the next suspect is")
print("  the walk unitary's ordering or the anc=1 branch algebra rather than the")
print("  Gram overlaps, since those are now included exactly.")
