"""V6 against Bowles, counted from both implementations rather than from abstracts.

The literature figure for Bowles et al. (arXiv:2306.14962) is 2B-1 circuits for a
B-block commuting circuit, and the notes here record it as "BEATS QLTO's 2N". That
figure does not apply to a hardware-efficient ansatz, and commute_gradient_paper.py
says so itself:

    Pure commuting circuits (no CNOT in W): 2B-1   (one circuit per block)
    EfficientSU2 / CNOT-entangled:          O(M)   (one circuit per param)

because W~ = G_j W G_j is qubit-specific once CNOTs couple generator qubits to
their neighbours. Its get_nefv_cost() reports both, so the real number is read off
the implementation rather than estimated.

CIRCUIT COUNT IS ONLY HALF OF IT. Each intermediate-block circuit is an LCU
carrying a CONTROLLED W and a CONTROLLED W~, i.e. two controlled copies of the
future ansatz. V6's circuits carry the plain ansatz plus parity CNOTs and
controlled single-qubit rotations. Comparing counts alone would flatter Bowles, so
both arms are transpiled to the same hardware basis and the per-circuit cost is
measured too.

WHAT IS REPORTED: circuits per gradient, depth and two-qubit gates per circuit, and
the product, for V6, Bowles-as-implemented, Bowles' theoretical 2B-1, and
parameter-shift. No accuracy is measured here; this is a cost count, and the two
estimators target different objectives anyway (Bowles is exact for grad E, V6 is
exact for the smoothed grad E_R).
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
import nisq_v6
from commute_gradient_paper import CommutingBlockGradient

BASIS = ['rz', 'sx', 'x', 'cx']


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


print("=" * 100)
print("V6 vs BOWLES, counted from the implementations")
print("=" * 100)
print(f"  Both transpiled to {BASIS}. Circuit counts read from Bowles'")
print("  own get_nefv_cost(), not from the paper's abstract.")
print()

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)

    with contextlib.redirect_stdout(io.StringIO()):
        cg = CommutingBlockGradient(ansatz, H)
    cost = cg.get_nefv_cost()

    with contextlib.redirect_stdout(io.StringIO()):
        q6 = nisq_v6.QLTOv6(ansatz, H, shot_budget=1024, sim_seed=1)
    G = len(q6.groups)
    blocks6 = [b['params'] for b in q6.layers if b['params']]
    t6, _, _ = q6._direct_template(blocks6[0], q6.groups[0])
    tt6 = transpile(t6, basis_gates=BASIS, optimization_level=1)
    d6, g6 = tt6.depth(), int(tt6.count_ops().get('cx', 0))

    # One Bowles LCU circuit, priced the same way. Build the intermediate-block
    # construction directly: it carries controlled W and controlled W~.
    d_b = g_b = None
    try:
        layer = cg.layers[0]
        end_idx = layer['end_index']
        bound = cg.ansatz.assign_parameters(cg._to_work_params(theta))
        total_ins = len(bound.decompose().data)
        u_past = cg._slice_ansatz(0, end_idx, theta)
        u_future = cg._slice_ansatz(end_idx + 1, total_ins - 1, theta)
        W_gate = u_future.to_gate(label='W')
        Wt = cg._build_W_tilde_gate(layer['type'], layer['gen_qubits'][0], W_gate)
        lcu = cg._build_lcu_circuit(u_past, W_gate, Wt, False)
        ttb = transpile(lcu, basis_gates=BASIS, optimization_level=1)
        d_b, g_b = ttb.depth(), int(ttb.count_ops().get('cx', 0))
    except Exception as e:
        print(f"  [LCU pricing failed at N={N}: {repr(e)[:70]}]")

    # parameter-shift: the bare ansatz, 2MG circuits
    ttp = transpile(ansatz, basis_gates=BASIS, optimization_level=1)
    dp, gp = ttp.depth(), int(ttp.count_ops().get('cx', 0))

    print(f"  N={N}, M={M}, G={G}, Bowles blocks B={cost['num_blocks']}")
    print(f"  {'method':>34}{'circuits':>10}{'depth/c':>9}{'2q/c':>7}"
          f"{'2q total':>10}")
    print("  " + "-" * 70)
    print(f"  {'V6':>34}{G:>10}{d6:>9}{g6:>7}{g6 * G:>10}")
    if d_b is not None:
        print(f"  {'Bowles, as implemented':>34}{cost['actual_with_cnot']:>10}"
              f"{d_b:>9}{g_b:>7}{g_b * cost['actual_with_cnot']:>10}")
        print(f"  {'Bowles, theoretical 2B-1':>34}"
              f"{cost['theoretical_2B_minus_1']:>10}{d_b:>9}{g_b:>7}"
              f"{g_b * cost['theoretical_2B_minus_1']:>10}")
    print(f"  {'parameter-shift':>34}{2 * M * G:>10}{dp:>9}{gp:>7}"
          f"{gp * 2 * M * G:>10}")
    print()
    if d_b is not None:
        print(f"    V6 circuits vs Bowles-as-implemented : "
              f"{cost['actual_with_cnot'] / G:.1f}x fewer")
        print(f"    V6 2q total vs Bowles-as-implemented : "
              f"{(g_b * cost['actual_with_cnot']) / (g6 * G):.1f}x fewer")
    print("  " + "." * 70)

print()
print("  'Bowles, theoretical 2B-1' is the figure the paper quotes and the notes")
print("  repeated. It requires no CNOTs in the future block, which EfficientSU2")
print("  violates by construction, so the row above it is the applicable one.")
print()
print("  The LCU circuits carry two CONTROLLED copies of the future ansatz, which")
print("  is why their per-circuit cost is not comparable to a bare ansatz run.")
