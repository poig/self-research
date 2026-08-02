"""How coherent is the walk in x, really?

The walk IS coherent - CRX genuinely mixes different |x> and the final H
interferes the ancilla branches. But after CRX moves |x> -> |x'>, the system
register still carries |psi_x> tied to the ORIGINAL x, so tracing it out weights
every cross-term by the overlap:

    P(x') = sum_{x,x''} A*(x->x') A(x''->x') <psi_x|psi_x''>

Coherence between two vertices therefore survives only in proportion to
|<psi_x|psi_x''>|. This measures that Gram matrix directly.

WHY IT DECIDES THE HSP QUESTION. A Simon-style extraction needs coherence between
x and x XOR s. For any s of high Hamming weight those vertices are ANTIPODAL on
the hypercube - maximally separated - so the antipodal overlap is the quantity
that gates the whole idea. If it collapses with R, no amount of coherent walking
recovers a hidden period, and designing an ansatz with psi_x = psi_{x XOR s} is
the only route left (which needs s known in advance, i.e. circular).

Reports, per radius:
    mean |<psi_x|psi_x'>| over ALL pairs
    mean over ADJACENT pairs (Hamming distance 1) - what the walk's local mixing
        actually uses
    mean over ANTIPODAL pairs (distance n) - what an HSP extraction would need
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def gram(ansatz, c, R, act):
    n = len(act)
    verts = list(itertools.product([-1.0, 1.0], repeat=n))
    states = []
    for s in verts:
        p = c.copy(); p[act] = c[act] + R * np.array(s)
        states.append(Statevector(ansatz.assign_parameters(p)).data)
    S = np.array(states)
    G = np.abs(S.conj() @ S.T)
    return verts, G


ansatz, H, _ = B.get_heisenberg_problem(4)
q = Q(ansatz, H, shot_budget=8192)
act = q.layers[0]['params']
n = len(act)
verts = list(itertools.product([-1.0, 1.0], repeat=n))
ham = np.array([[int(sum(a != b for a, b in zip(u, v))) for v in verts]
                for u in verts])

print("=" * 78)
print("Walk coherence in x: overlap |<psi_x|psi_x'>| across the hypercube")
print("=" * 78)
print(f"  Heisenberg N=4, block of {n} params, {2**n} vertices, 3 centres")
print()
print(f"  {'R':>7}{'all pairs':>12}{'adjacent':>12}{'antipodal':>12}"
      f"{'min pair':>11}")
print("  " + "-" * 54)
for R in (0.1, 0.3, 0.6, 1.0, np.pi / 2):
    allm, adjm, antm, mn = [], [], [], []
    for seed in (3, 11, 17):
        c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                ansatz.num_parameters)
        _, G = gram(ansatz, c, R, act)
        off = ~np.eye(len(verts), dtype=bool)
        allm.append(G[off].mean())
        adjm.append(G[ham == 1].mean())
        antm.append(G[ham == n].mean())
        mn.append(G[off].min())
    print(f"  {R:>7.3f}{np.mean(allm):>12.4f}{np.mean(adjm):>12.4f}"
          f"{np.mean(antm):>12.4f}{np.mean(mn):>11.4f}", flush=True)

print()
print("  ADJACENT overlap is what the walk's local CRX mixing rides on.")
print("  ANTIPODAL overlap is what a Simon-style hidden-period extraction would")
print("  need, since x and x XOR s are maximally separated for high-weight s.")
print("  If antipodal collapses while adjacent survives, the walk is coherent")
print("  LOCALLY and cannot support global interference - which is exactly the")
print("  boundary between 'the walk works' and 'HSP is reachable'.")
