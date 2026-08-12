"""The log-qubit design encoding on real circuits, and what it unlocks.

The one-hot register spends ONE QUBIT PER PARAMETER. That caps block width n, and
since the advantage over parameter-shift is exactly 2n and circuits per gradient
are G*M/n, the cap is what has held the method at 2N.

The design encoding indexes rows of a resolution-IV Hadamard design on
ceil(log2(n+1)) + 1 qubits, so a block of n parameters costs LOG register width.
v77 measured it accuracy-neutral on a synthetic landscape. This runs it on the
actual sensing circuits, where the sign convention has to be right and the parity
CNOTs are real gates.

THREE THINGS, in order of what they would invalidate.

  1. CORRECTNESS FIRST. If the encoding's sign convention is wrong the gradient is
     silently wrong, not obviously wrong, which is the failure mode this project
     has hit repeatedly. Both encodings are scored against the same exact
     parameter-shift gradient at the same theta. They must agree with it, and with
     each other, before any efficiency number is worth reading.

  2. WHAT IT UNLOCKS. block_mode='global' puts all M parameters in ONE block, so
     L = 1 and circuits per gradient fall from G*L to G. Under one-hot that needs
     M register qubits: 24 at N=4 and 36 at N=6, the latter beyond simulation with
     the system register. Under the design it needs 6 and 7. The comparison is
     therefore not "which is better" but "one of these can be run at all".

  3. WHAT IT COSTS. The control qubit per parameter is replaced by parity CNOTs,
     roughly 2*popcount(c_j) of them per parameter plus the foldover. Transpiled
     depth and two-qubit count are reported for both, because a register saving
     paid for in depth is not obviously a saving, and this project has already
     been burned once by assuming a gate cost instead of measuring it.

WHAT WOULD KILL IT: the design gradient disagreeing with exact where one-hot
agrees, or the two-qubit count per gradient rising by more than the circuit-count
reduction gains.
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


def build(ansatz, H, enc, blk, dec, seed, shots):
    with contextlib.redirect_stdout(io.StringIO()):
        return nisq_v5.QLTOv5(ansatz, H, shot_budget=shots,
                              gradient_mode='direct', sim_seed=seed,
                              encoding=enc, block_mode=blk, decoder=dec)


def measure(q, theta, R, M):
    blocks = [b['params'] for b in q.layers if b['params']]
    gh = np.zeros(M)
    for act in blocks:
        gi, _ = q.sense(theta, R, act)
        gh += gi
    return gh, len(blocks)


R, SHOTS, REPS = 0.45, 16384, 3
BASIS = ['rz', 'sx', 'x', 'cx']

print("=" * 104)
print("(1)  CORRECTNESS:  does the design encoding reproduce the gradient?")
print("=" * 104)
print(f"  Scored against exact parameter-shift at the same theta. R = {R},")
print(f"  {SHOTS} shots per circuit, {REPS} seeds, layered blocks for both.")
print()
print(f"  {'N':>3}{'M':>4}{'encoding':>10}{'reg qubits':>12}{'blocks':>8}"
      f"{'cos vs exact':>14}")
print("  " + "-" * 51)

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    for enc in ('onehot', 'design'):
        cs, L = [], 0
        for s in range(REPS):
            q = build(ansatz, H, enc, 'layered', 'marginal', 700 + s, SHOTS)
            gh, L = measure(q, theta, R, M)
            cs.append(cosine(gh, g_ex))
        blocks = [b['params'] for b in q.layers if b['params']]
        nb = max(len(b) for b in blocks)
        reg = nb if enc == 'onehot' else int(np.ceil(np.log2(nb + 1))) + 1
        print(f"  {N:>3}{M:>4}{enc:>10}{reg:>12}{L:>8}"
              f"{float(np.mean(cs)):>14.4f}")
    print("  " + "." * 51)

print()
print("  The two encodings must land on the same cosine. A gap means the sign")
print("  convention is wrong, and a wrong sign is silent rather than loud.")

print()
print("=" * 104)
print("(2)  WHAT IT UNLOCKS:  all M parameters in ONE block")
print("=" * 104)
print("  block_mode='global' sets L = 1, so circuits per gradient fall from G*L to")
print("  G. Register need under one-hot is M qubits; under the design it is")
print("  ceil(log2(M+1)) + 1.")
print()
print("  Compared at MATCHED TOTAL SHOTS. Global runs 6x fewer circuits, so it gets")
print("  6x the shots on each; comparing at equal shots-per-circuit would hand")
print("  layered six times the budget and mean nothing.")
print()
print(f"  {'N':>3}{'M':>4}{'config':>18}{'reg':>6}{'circuits':>10}"
      f"{'shots/circ':>12}{'T total':>10}{'cos vs exact':>14}")
print("  " + "-" * 82)
for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    ref = build(ansatz, H, 'onehot', 'layered', 'marginal', 700, SHOTS)
    n_lay = len(ref.groups) * len([b for b in ref.layers if b['params']])
    total = n_lay * SHOTS

    for enc, blk, dec in (('onehot', 'layered', 'marginal'),
                          ('design', 'global', 'marginal'),
                          ('design', 'global', 'wls')):
        probe = build(ansatz, H, enc, blk, dec, 700, SHOTS)
        ncirc = len(probe.groups) * len([b for b in probe.layers if b['params']])
        per = max(1, total // ncirc)
        blocks = [b['params'] for b in probe.layers if b['params']]
        nb = max(len(b) for b in blocks)
        reg = nb if enc == 'onehot' else int(np.ceil(np.log2(nb + 1))) + 1
        cs = []
        for s in range(REPS):
            q = build(ansatz, H, enc, blk, dec, 700 + s, per)
            gh, _ = measure(q, theta, R, M)
            cs.append(cosine(gh, g_ex))
        print(f"  {N:>3}{M:>4}{enc + '/' + blk + '/' + dec:>18}{reg:>6}"
              f"{ncirc:>10}{per:>12}{ncirc * per:>10}"
              f"{float(np.mean(cs)):>14.4f}")
    print("  " + "." * 82)
print()
print("  One-hot at N=6 would need 36 register qubits plus 6 system, which is not")
print("  simulable here; that row exists only because of the encoding.")

print()
print("=" * 104)
print("(2b) IS THE GLOBAL-BLOCK DEFICIT BIAS, AND IS R THE CAUSE?")
print("=" * 104)
print("  A block of n parameters displaces the state by ~sqrt(n)*R, so the radius")
print("  that is right for a 6-parameter block over-displaces a 36-parameter one by")
print("  about 2.4x and the linearisation E ~ E0 + R sum_j g_j sigma_j degrades.")
print("  WLS cannot help with that: it removes cross terms, which is VARIANCE, and")
print("  this would be BIAS. If sweeping R recovers the accuracy, the global deficit")
print("  is a radius that was never re-tuned for the wider block, not a property of")
print("  wide blocks. If it does not, wide blocks genuinely cost accuracy.")
print()
print(f"  {'N':>3}{'R':>8}{'cos global':>12}{'cos layered @ R=0.45':>22}")
print("  " + "-" * 45)
for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)
    ref = build(ansatz, H, 'onehot', 'layered', 'marginal', 700, SHOTS)
    n_lay = len(ref.groups) * len([b for b in ref.layers if b['params']])
    total = n_lay * SHOTS
    cs_ref = []
    for s in range(REPS):
        q = build(ansatz, H, 'onehot', 'layered', 'marginal', 700 + s, SHOTS)
        gh, _ = measure(q, theta, 0.45, M)
        cs_ref.append(cosine(gh, g_ex))
    ref_cos = float(np.mean(cs_ref))
    for Rg in (0.05, 0.10, 0.18, 0.30, 0.45):
        cs = []
        for s in range(REPS):
            q = build(ansatz, H, 'design', 'global', 'marginal', 700 + s,
                      max(1, total // 3))
            gh, _ = measure(q, theta, Rg, M)
            cs.append(cosine(gh, g_ex))
        print(f"  {N:>3}{Rg:>8.2f}{float(np.mean(cs)):>12.4f}{ref_cos:>22.4f}")
    print("  " + "." * 45)

print()
print("=" * 104)
print("(3)  WHAT IT COSTS:  transpiled depth and two-qubit gates per GRADIENT")
print("=" * 104)
print(f"  Common hardware basis {BASIS}, so neither arm is flattered by a")
print("  simulator basis that keeps cry as one gate.")
print()
print(f"  {'N':>3}{'config':>22}{'circuits':>10}{'depth/circ':>12}"
      f"{'2q/circ':>10}{'2q per gradient':>17}")
print("  " + "-" * 73)
for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    for enc, blk in (('onehot', 'layered'), ('design', 'layered'),
                     ('design', 'global')):
        q = build(ansatz, H, enc, blk, 'marginal', 700, SHOTS)
        blocks = [b['params'] for b in q.layers if b['params']]
        t, _, _ = q._direct_template(blocks[0], q.groups[0])
        tt = transpile(t, basis_gates=BASIS, optimization_level=1)
        ops = tt.count_ops()
        two = int(ops.get('cx', 0))
        circuits = len(q.groups) * len(blocks)
        print(f"  {N:>3}{enc + '/' + blk:>22}{circuits:>10}{tt.depth():>12}"
              f"{two:>10}{two * circuits:>17}")
    print("  " + "." * 73)
print()
print("  '2q per gradient' is the number that matters: circuits times two-qubit")
print("  gates each. The design trades a control qubit per parameter for parity")
print("  CNOTs, so per circuit it should cost MORE, and only wins if the circuit")
print("  count falls faster than the per-circuit cost rises.")
