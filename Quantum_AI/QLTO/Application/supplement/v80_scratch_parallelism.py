"""Does parallelising the parity scratch close the design encoding's depth gap?

After Gray ordering, design/global beat one-hot/layered on circuits (3 against 18)
and on two-qubit gates per gradient (372 against 396) at equal accuracy, on 7
register qubits against 36. One axis still lost badly: DEPTH PER CIRCUIT, 229
against 34.

That is not gate count, which Gray ordering already fixed. It is serialisation.
Every parity was computed on the SAME scratch wire, so no two of them could
overlap and the block's depth became their sum, while one-hot's controlled
rotations sit on different system qubits and partly schedule together.

THE FIX UNDER TEST. Round-robin the parities over several scratch qubits. The
parity CNOT for the next parameter then touches a different wire from the current
parameter's controlled rotation, the two are disjoint, and the transpiler puts
them in the same layer.

WHY IT WILL NOT SCALE LINEARLY, stated in advance so the result is not read as a
disappointment: the register qubits are shared CONTROLS and a qubit takes part in
only one two-qubit gate at a time, so the register itself serialises some of it.
It also costs gates, because parameters sharing a scratch wire are k apart in the
Gray sequence and their columns then differ in more than one bit, so each update
is more than the single CNOT that Gray ordering bought.

So the expected shape is depth falling and two-qubit count rising, with an
interior optimum. The question is whether the optimum lands near one-hot's depth
while keeping the gate and circuit advantages.

CORRECTNESS IS CHECKED AT EVERY SCRATCH COUNT. The number of scratch wires must
not change the gradient at all: it changes which wire holds a parity, never the
parity. A drift in the cosine would mean the round-robin bookkeeping is wrong,
and a wrong sign is silent rather than loud.
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

BASIS = ['rz', 'sx', 'x', 'cx']
SHOTS, REPS = 16384, 3
R_GLOBAL = {4: 0.10, 6: 0.18}          # per-width radii established in v79


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


def build(ansatz, H, enc, blk, seed, shots, nscr=1):
    with contextlib.redirect_stdout(io.StringIO()):
        return nisq_v5.QLTOv5(ansatz, H, shot_budget=shots,
                              gradient_mode='direct', sim_seed=seed,
                              encoding=enc, block_mode=blk, n_scratch=nscr)


print("=" * 104)
print("SCRATCH PARALLELISM:  depth, gates and correctness against scratch count")
print("=" * 104)
print(f"  design/global at the per-width radius from v79 (N=4: {R_GLOBAL[4]},")
print(f"  N=6: {R_GLOBAL[6]}), matched total shots, {REPS} seeds, basis {BASIS}.")
print("  The one-hot/layered row is the reference every column is trying to reach.")
print()
print(f"  {'N':>3}{'config':>22}{'scratch':>9}{'circuits':>10}{'depth/circ':>12}"
      f"{'2q/circ':>9}{'2q per grad':>13}{'cos':>9}")
print("  " + "-" * 87)

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    ref = build(ansatz, H, 'onehot', 'layered', 700, SHOTS)
    n_lay = len(ref.groups) * len([b for b in ref.layers if b['params']])
    total = n_lay * SHOTS

    def run(enc, blk, nscr, Rv):
        q0 = build(ansatz, H, enc, blk, 700, SHOTS, nscr)
        blocks = [b['params'] for b in q0.layers if b['params']]
        ncirc = len(q0.groups) * len(blocks)
        per = max(1, total // ncirc)
        t, _, _ = q0._direct_template(blocks[0], q0.groups[0])
        tt = transpile(t, basis_gates=BASIS, optimization_level=1)
        two = int(tt.count_ops().get('cx', 0))
        cs = []
        for s in range(REPS):
            q = build(ansatz, H, enc, blk, 700 + s, per, nscr)
            gh = np.zeros(M)
            for act in [b['params'] for b in q.layers if b['params']]:
                gi, _ = q.sense(theta, Rv, act)
                gh += gi
            cs.append(cosine(gh, g_ex))
        return ncirc, tt.depth(), two, float(np.mean(cs))

    c, d, t2, cs = run('onehot', 'layered', 1, 0.45)
    print(f"  {N:>3}{'onehot/layered':>22}{'-':>9}{c:>10}{d:>12}{t2:>9}"
          f"{t2 * c:>13}{cs:>9.4f}")
    for nscr in (1, 2, 3, 4, 6):
        c, d, t2, cs = run('design', 'global', nscr, R_GLOBAL[N])
        print(f"  {N:>3}{'design/global':>22}{nscr:>9}{c:>10}{d:>12}{t2:>9}"
              f"{t2 * c:>13}{cs:>9.4f}")
    print("  " + "." * 87)

print()
print("  The cosine column must be FLAT across scratch counts. It is the same")
print("  circuit with parities on different wires, so any drift is a bookkeeping")
print("  error rather than a trade.")
print()
print("  Depth should fall and 2q should rise. Worth taking only if some scratch")
print("  count gets depth near the one-hot row WITHOUT giving back the 6x circuit")
print("  advantage or the two-qubit advantage.")
