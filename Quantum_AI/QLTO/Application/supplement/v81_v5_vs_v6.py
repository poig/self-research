"""V5 against V6 head to head, and both against parameter-shift.

V6 is now its own class rather than a set of flags on V5, so this compares them the
way a reader would: construct each with its own defaults, hand both the same
ansatz, Hamiltonian, theta and TOTAL shot budget, and score against the exact
gradient.

THE POINT OF SEPARATING THEM. Every log in this repository from v60 to v80 was
produced by V5's defaults. V6 changes three of them at once - log-width register,
one global block, parallel parity scratch - and any of the three could carry the
result. Keeping V5 untouched means the baseline cannot drift while V6 moves.

V6 ALSO RESCALES THE RADIUS, and that has to be checked rather than trusted. A
block of n parameters displaces the state by about sqrt(n)*R, so V5's radius
over-displaces V6's wider block. V6 divides by sqrt(n/N) internally so that the
SAME R works for both. The scale_radius=False arm below is what happens without
it, and it is the difference between 0.98 and 0.83 - a silent regression that
would read as a bug elsewhere.

FOUR ARMS at matched TOTAL shots:
    V5                      one-hot register, layered blocks
    V6                      design register, global block, radius rescaled
    V6, scale_radius=False  the trap, kept visible
    parameter-shift         2*M*G circuits, exact per component, the baseline both
                            are trying to beat

WHAT WOULD MAKE V6 THE BETTER DEFAULT: equal accuracy on fewer circuits with no
axis badly worse. Depth is expected to be worse and is reported so the trade is
visible rather than buried.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5
import nisq_v6

BASIS = ['rz', 'sx', 'x', 'cx']
R, REPS = 0.45, 3
BUDGET = 294912


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


def ps_sampled(ansatz, gmats, theta, shots, rng):
    """Parameter-shift with honest shot noise, variance exact from the state."""
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            e = 0.0
            for Hg, Hg2 in gmats:
                m1 = float(np.real(np.conj(v) @ (Hg @ v)))
                m2 = float(np.real(np.conj(v) @ (Hg2 @ v)))
                e += m1 + rng.normal(0.0, np.sqrt(max(m2 - m1 * m1, 0.0)
                                                  / max(shots, 1)))
            g[i] += s * e / 2
    return g


print("=" * 104)
print("V5 vs V6 vs PARAMETER-SHIFT, matched total shots")
print("=" * 104)
print(f"  Same ansatz, Hamiltonian, theta and total budget T = {BUDGET}.")
print(f"  R = {R} handed to BOTH; V6 rescales it internally by sqrt(N/n).")
print(f"  {REPS} seeds, transpiled to {BASIS} for the gate columns.")
print()
print(f"  {'N':>3}{'M':>4}{'method':>24}{'circuits':>10}{'shots/circ':>11}"
      f"{'reg q':>7}{'depth':>7}{'2q/grad':>9}{'cos':>9}")
print("  " + "-" * 84)

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)
    gmats = None

    def arm(cls, tag, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            q0 = cls(ansatz, H, shot_budget=1024, sim_seed=700, **kw)
        blocks = [b['params'] for b in q0.layers if b['params']]
        ncirc = len(q0.groups) * len(blocks)
        per = max(1, BUDGET // ncirc)
        t, _, _ = q0._direct_template(blocks[0], q0.groups[0])
        tt = transpile(t, basis_gates=BASIS, optimization_level=1)
        two = int(tt.count_ops().get('cx', 0))
        nb = max(len(b) for b in blocks)
        # V5 spends one register qubit per parameter; V6 spends log2(n)+1 for the
        # design row plus its scratch wires.
        if isinstance(q0, nisq_v6.QLTOv6):
            reg = int(np.ceil(np.log2(nb + 1))) + 1 + min(q0.n_scratch, nb)
        else:
            reg = nb
        cs = []
        for s in range(REPS):
            with contextlib.redirect_stdout(io.StringIO()):
                q = cls(ansatz, H, shot_budget=per, sim_seed=700 + s, **kw)
            gh = np.zeros(M)
            for act in [b['params'] for b in q.layers if b['params']]:
                gi, _ = q.sense(theta, R, act)
                gh += gi
            cs.append(cosine(gh, g_ex))
        print(f"  {N:>3}{M:>4}{tag:>24}{ncirc:>10}{per:>11}{reg:>7}"
              f"{tt.depth():>7}{two * ncirc:>9}{float(np.mean(cs)):>9.4f}")
        return q0

    q5 = arm(nisq_v5.QLTOv5, 'V5')
    arm(nisq_v6.QLTOv6, 'V6')
    arm(nisq_v6.QLTOv6, 'V6 (no radius scaling)', scale_radius=False)

    G = len(q5.groups)
    gmats = [(g.to_matrix(), (g @ g).simplify().to_matrix()) for g in q5.groups]
    ncirc_ps = 2 * M * G
    per_ps = max(1, BUDGET // ncirc_ps)
    cs = []
    for s in range(REPS):
        rng = np.random.RandomState(9000 + s)
        cs.append(cosine(ps_sampled(ansatz, gmats, theta, per_ps, rng), g_ex))
    tt = transpile(ansatz, basis_gates=BASIS, optimization_level=1)
    two = int(tt.count_ops().get('cx', 0))
    print(f"  {N:>3}{M:>4}{'parameter-shift':>24}{ncirc_ps:>10}{per_ps:>11}"
          f"{0:>7}{tt.depth():>7}{two * ncirc_ps:>9}"
          f"{float(np.mean(cs)):>9.4f}")
    print("  " + "." * 84)

print()
print("  'reg q' is register qubits ON TOP of the N system qubits; parameter-shift")
print("  needs none. 'depth' and '2q/grad' come from a transpile with NO COUPLING")
print("  MAP, so they assume all-to-all connectivity and understate every arm that")
print("  entangles a register with the system - which is both QLTO arms and not")
print("  parameter-shift. Treat the gate columns as a lower bound on the QLTO cost.")
