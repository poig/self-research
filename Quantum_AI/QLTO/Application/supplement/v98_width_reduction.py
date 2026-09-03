"""Can V6 be made NARROWER for free? Two levers, both untested in the benchmark.

v97 settled the direction: resolution 5 buys nothing on VQE (three ties and one
marginal loss over 5 paired seeds) and costs 20-52% depth. If MORE design
resolution does not help, the interesting question is whether LESS costs
anything - because width is the one axis where V6 loses (2.25x-5x the ansatz),
and a qubit returned is worth more than a marginal energy gain.

TWO LEVERS.

  FOLDOVER (resolution 4 -> 3). The fold bit makes every parameter's sign carry
  (-1)^f, so a main effect and a 2-factor interaction differ in f-parity and
  cannot alias. Dropping it saves ONE register qubit and reintroduces that
  aliasing: main effect i is confounded with {j,k} exactly when c_i = c_j ^ c_k,
  and v90 measured such triples to exist in the shipped Gray columns.
  Whether that matters depends on how much degree-2 content the loss carries -
  and v92 fitted an effective degree of only ~5, so possibly very little.

  SCRATCH WIRES. n_scratch = 3 by default, fixed, never swept in the benchmark.
  That is THREE qubits - at Heisenberg N=8, 3 of 18, twice what the foldover
  buys. v80 studied scratch parallelism for DEPTH; its effect on converged
  energy was never measured. Fewer wires serialise the parity chains, so the
  trade is width against depth.

Paired on identical seeds, same r0, same epochs, so the seed-to-seed variance
that made v90's single-draw comparison unreadable is differenced out. Width and
depth are reported alongside energy because a win on energy that costs depth is
not a win on the axis this is about.

r0 IS NOT RE-TUNED for any arm. It was selected against resolution 4 with
n_scratch 3, which biases toward the incumbent - so any arm that ties is
genuinely interesting, not merely unharmed.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

import benchmark as B

PROBS = [B.get_heisenberg_problem(4), B.get_heisenberg_problem(6),
         B.get_h2_problem()]
SEEDS, EPOCHS = 5, 20
ARMS = [("res4 ns3  (shipped)", 4, 3),
        ("res3 ns3  (no fold)", 3, 3),
        ("res4 ns2", 4, 2),
        ("res4 ns1", 4, 1),
        ("res3 ns1  (both)", 3, 1)]


def run(ansatz, H, seed, res, ns):
    M = ansatz.num_parameters
    np.random.seed(seed)
    p = np.random.uniform(-np.pi, np.pi, M)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = B.QLTOv6_Wrapper(ansatz, H, r0=B.TUNED['QLTO V6'],
                               n_scratch=ns, design_resolution=res)
        B._mult(opt, 1)
        es = []
        for _ in range(EPOCHS):
            p = opt.step(p)
            es.append(B.report_energy(ansatz, H, p))
    m_row = max(1, int(np.ceil(np.log2(M + 1))))
    width = m_row + (1 if res >= 4 else 0) + ansatz.num_qubits + max(1, min(ns, M))
    return min(es), width, opt.max_circuit_depth


print("=" * 98)
print("WIDTH REDUCTION:  is a narrower design free?")
print("=" * 98)
print(f"  {SEEDS} paired seeds x {EPOCHS} epochs, r0 = {B.TUNED['QLTO V6']}"
      f" (tuned for res4/ns3, so biased toward the incumbent).")
print("  'diff' is paired against the shipped arm; POSITIVE = worse.")
print()

for ansatz, H, name in PROBS:
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    print(f"  {name}   exact {exact:.4f}   ansatz {ansatz.num_qubits}q"
          f"  M={ansatz.num_parameters}")
    print(f"    {'arm':<22}{'mean':>10}{'diff':>10}{'sem':>9}"
          f"{'width':>7}{'depth':>7}")
    print("    " + "-" * 65)
    base = None
    for label, res, ns in ARMS:
        vals, w, d = [], 0, 0
        for s in range(SEEDS):
            e, w, d = run(ansatz, H, 42 + s, res, ns)
            vals.append(e)
        vals = np.array(vals)
        if base is None:
            base = vals
            print(f"    {label:<22}{vals.mean():>10.4f}{'--':>10}{'--':>9}"
                  f"{w:>7}{d:>7}", flush=True)
        else:
            diff = vals - base
            sem = float(np.std(diff) / np.sqrt(max(len(diff), 1)))
            print(f"    {label:<22}{vals.mean():>10.4f}{diff.mean():>10.4f}"
                  f"{sem:>9.4f}{w:>7}{d:>7}", flush=True)
    print()

print("=" * 98)
print("READING IT")
print("=" * 98)
print("  An arm whose paired diff is inside ~2 SEM is FREE: it gives the qubits")
print("  back at no measurable cost in energy, and width is the axis V6 loses.")
print("  An arm that is free AND shallower is strictly better than the shipped")
print("  default and should become it. An arm that is free but DEEPER is a")
print("  genuine trade and belongs as a knob, like resolution.")
print("  Note r0 favours the incumbent, so a tie understates the alternative.")
