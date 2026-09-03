"""Grover needs ORACLE-DIFFUSER ALTERNATION. v39 tested the diffuser without it.

v39 swapped the product mixer for the global reflection exp(-i beta |s><s|) and
measured NO CHANGE: 14/40 sign errors for both arms. That result stands, but it
does not test the hypothesis, because the circuit it tested is not Grover's.

The shipped walk applies the energy imprint ONCE, before the k-step loop:

    imprint ; [drift ; mix] x k

Grover alternates: mark, diffuse, mark, diffuse. One marking followed by k
diffusions cannot amplify anything - without re-marking, diffusion is a rotation
in a fixed plane that returns amplitude where it came from. So v39 measured a
diffuser with nothing to diffuse, and it behaved exactly as it must.

This is also, in hindsight, why reset_full was the best arm in v37: it was the
only variant that re-imprinted every step, and so the only one with any
alternating structure at all.

ARMS, all with identical drift (merged_walk=False) and identical schedule:

    once_prod    imprint once, product CRX mixer          <- SHIPPED
    step_prod    imprint EVERY step, product CRX mixer    <- alternation only
    step_glob    imprint EVERY step, global reflection    <- true Grover structure

Total evolution time is held equal across arms: the per-step imprint runs at
dt*pi/k so that k of them match the single dt*pi of the shipped path. Without
that the arms differ in how much energy information enters, not just in how it
is used.

WHAT WOULD SETTLE IT. v38 enumerated the hypercube and showed the product mixer
converges to the sign pattern of the degree-1 Walsh coefficients, wrong on 7/16
blocks with regret to 0.889, while a degree-<=2 target was exact everywhere. If
step_glob reduces the sign errors on those blocks, the reachability defect is the
mixer's locality COMBINED with the missing alternation. If it does not, then the
defect is elsewhere - and the first place to look is the decode, which takes a
WEIGHTED MEAN over sampled corners and would wash out concentration even if the
amplitude did concentrate.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


class AltWalk(nisq_v3.QLTOv3):
    per_step_imprint = True
    global_mixer = True

    def _execute_walk(self, center_params, k_steps, delta_t, radius,
                      active_indices, grad):
        n = len(active_indices)
        gl = grad[active_indices]
        gain = 1.0 / np.sqrt(max(radius, 1e-9))

        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n, 'param')
        sysr = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cp = ClassicalRegister(n, 'c_param')
        ca = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sysr, cp, ca)

        qc.h(anc)
        qc.h(param)
        qc.append(self.build_w_gate(param, sysr, center_params, radius,
                                    active_indices), list(param) + list(sysr))

        t = delta_t * np.pi / k_steps if self.per_step_imprint else delta_t * np.pi
        imprint = PauliEvolutionGate(self.H_sense, time=t,
                                     synthesis=LieTrotter(reps=1)).control(1)
        if not self.per_step_imprint:
            qc.append(imprint, [anc[0]] + list(sysr))

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t
            beta = (1.0 - s) * np.pi * delta_t
            if self.per_step_imprint:
                qc.append(imprint, [anc[0]] + list(sysr))   # MARK
            for i in range(n):
                qc.crz(gl[i] * gamma * 0.5 * np.pi * gain, anc[0], param[i])
            if self.global_mixer:                            # DIFFUSE
                qc.h(param); qc.x(param)
                if n == 1:
                    qc.cp(-2.0 * beta, anc[0], param[0])
                else:
                    qc.mcp(-2.0 * beta, [anc[0]] + list(param[:-1]), param[-1])
                qc.x(param); qc.h(param)
            else:
                for i in range(n):
                    qc.crx(beta, anc[0], param[i])

        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        counts = self._run(qc)
        blk = self._decode_walk(counts, center_params, active_indices, radius)
        out = center_params.copy()
        out[active_indices] = blk
        return out


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def maxcut(N):
    return SparsePauliOp.from_list(
        [("".join("Z" if q in (i, i + 1) else "I" for q in range(N)), 1.0)
         for i in range(N - 1)])


def h2():
    return SparsePauliOp.from_list([("II", -1.0523), ("IZ", 0.3979),
                                    ("ZI", -0.3979), ("ZZ", -0.0113),
                                    ("XX", 0.1809)])


def E(a, H, p):
    return float(np.real(Statevector(a.assign_parameters(p)).expectation_value(H)))


def mk(cls, a, H, imp=None, glob=None, **kw):
    kw.setdefault('merged_walk', False)
    with contextlib.redirect_stdout(io.StringIO()):
        q = cls(a, H, **kw)
    if imp is not None:
        q.per_step_imprint = imp
    if glob is not None:
        q.global_mixer = glob
    return q


R, DT, KS, SHOTS = 0.6, 0.5, 15, 16384
ARMS = [('once_prod', False, False), ('step_prod', True, False),
        ('step_glob', True, True)]
PROBLEMS = [("H2", h2()), ("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 100)
print("ORACLE-DIFFUSER ALTERNATION — the structure v39 was missing")
print("=" * 100)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Total evolution time equal in")
print(f"  every arm (per-step imprint runs at dt*pi/k). Drift identical throughout.")
print(f"  Sign errors are against the TRUE hypercube argmin, enumerated exactly.")
print()
hdr = "".join(f"{a:>12}" for a, _, _ in ARMS)
print(f"  {'problem':>15}{'blk':>4}{'n':>3}{'deg1':>8}{'deg2':>8}{hdr}")
print("  " + "-" * (38 + 12 * len(ARMS)))

tot = {a: 0 for a, _, _ in ARMS}
ntot = 0
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    qs = {a: mk(AltWalk, ansatz, H, imp, gl, shot_budget=SHOTS, sim_seed=17)
          for a, imp, gl in ARMS}
    BLK = [b['params'] for b in qs['once_prod'].layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.array([E(ansatz, H, np.where(
            np.isin(np.arange(M), act),
            centre + R * np.array([sv[list(act).index(j)] if j in act else 0
                                   for j in range(M)]), centre))
            for sv in sig])
        x_true = sig[int(np.argmin(vals))]
        d1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        p1w = float(np.sum(d1 ** 2))
        p2w = float(np.sum([np.mean(vals * sig[:, i] * sig[:, j]) ** 2
                            for i in range(n) for j in range(i + 1, n)]))
        cells = []
        for a, _, _ in ARMS:
            q = qs[a]
            q.reset_shot_stream()
            g = q.sense_gradient(centre, R, act)
            p1 = q._execute_walk(centre, KS, DT, R, act, g)
            xs = np.where(p1[act] - centre[act] >= 0, 1.0, -1.0)
            ham = int(np.sum(xs != x_true))
            tot[a] += ham
            cells.append(ham)
        ntot += n
        print(f"  {name:>15}{bi:>4}{n:>3}{p1w:>8.4f}{p2w:>8.4f}"
              + "".join(f"{c:>12}" for c in cells), flush=True)

print("  " + "-" * (38 + 12 * len(ARMS)))
print(f"  {'TOTAL sign errors':>30}" + "".join(f"{tot[a]:>12}" for a, _, _ in ARMS)
      + f"   of {ntot}")
print()
print("  step_glob below once_prod would mean the missing piece was the")
print("  ALTERNATION, not the diffuser alone. All three equal would mean the")
print("  bottleneck is downstream of the walk entirely - most likely the decode,")
print("  which averages over sampled corners and cannot report a concentration")
print("  even if one exists. That is the next thing to test, and it is cheap:")
print("  read the raw param distribution instead of its weighted mean.")
