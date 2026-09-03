"""Walk vs gradstep on the FULL benchmark suite. The last gap.

Everything decided so far is at Heisenberg N=4 and N=6. The suite has eight
problems including LiH and Heisenberg N=8, and nothing in this session has
touched them. This runs the A/B on all of them.

STATE OF THE QUESTION going in - seven of eight measurements favour gradstep:

  shipped R, N=4 (four independent runs, two implementations, both merged_walk
  settings): walk -5.79/-5.05/-4.54/-5.15 against gradstep -5.81/-5.58/-5.39.
  Never ahead.

  wide R (v53, R0=pi/2 where v9_globalgrid puts the box at 1.7 -> 3.3 minima -
  the walk's own claim, a stochastic bounded step through a multi-modal box):
      N   shots   walk      boltz     gradstep   walk-grad
      4     256  -5.7042   -5.3651   -5.5346    -0.170 (1.4s)   <- only walk win
      4    1024  -5.6594   -5.7465   -5.7793    +0.120 (0.7s)
      6     256  -7.7775   -7.9072   -8.2025    +0.425 (1.2s)
      6    1024  -8.7440   -8.9257   -8.7769    +0.033 (0.3s)
  The one walk win did not replicate at 4x the shots, and boltz beat the walk in
  three of four rows despite running below its own validity threshold.

NO ROW CLEARS 3 SIGMA either way, and this project's own calibration says that
matters: v27 showed the harness returns 2.2 and 3.3 sigma on a PROVABLY NULL
comparison. So the case rests on direction being consistent, not on any single
decisive row - which is exactly why the full suite is worth running rather than
another N=4 variant.

The mechanism predicts what to expect: the walk discards the gradient's magnitude
channel to wrapping and keeps only direction, so it PLATEAUS while arms that use
magnitude convert shots into energy. v53 saw that directly - 4x the shots moved
the walk by -0.04 while gradstep gained +0.24.

THIS MIRRORS run_benchmark_with_stats EXACTLY - same problems, same 20 epochs,
same seeds 42+t, same report_energy, same optimizer_circuits accounting, same
_mult(1) for V3 rows - but runs only the two rows in question, since the full
suite with all six optimizers is hours.

DECISION RULE, fixed here rather than after the numbers: gradstep costs ONE
circuit per block-epoch against the walk's two, so PARITY IS A WIN for gradstep.
Flip the default unless the walk is ahead by more than the harness's null scale
on a clear majority of problems.
"""
import sys, os, contextlib, io, time, gc
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
import benchmark as B
from qiskit_aer import AerSimulator

N_TRIALS = 3
EPOCHS = 20

PROBLEMS = [
    B.get_maxcut_problem(4, seed=101),
    B.get_maxcut_problem(6, seed=102),
    B.get_h2_problem(),
    B.get_lih_problem(),
    B.get_heisenberg_problem(4),
    B.get_heisenberg_problem(6),
    B.get_heisenberg_problem(8),
]

ARMS = {
    'walk': lambda a, h, bk: B.QLTO_Wrapper(
        a, h, k_step=B.TUNED['QLTO V3 QPE (k=3)'], bits_per_param=1, layer=True,
        backend=bk, walk_gradient=True, v3_ancillas=3, decoder='walk'),
    'gradstep': lambda a, h, bk: B.QLTO_Wrapper(
        a, h, k_step=B.TUNED['QLTO V3 QPE (k=3)'], bits_per_param=1, layer=True,
        backend=bk, walk_gradient=True, v3_ancillas=3, decoder='gradstep'),
}

print("=" * 100)
print("BENCHMARK A/B — walk vs gradstep, full suite")
print("=" * 100)
print(f"  {N_TRIALS} trials, {EPOCHS} epochs, seeds 42+t, same harness accounting.")
print(f"  gradstep costs 1 circuit/block-epoch, walk 2 - so PARITY IS A WIN.")
print()
print(f"  {'problem':>16}{'exact':>10}{'walk E':>10}{'grad E':>10}{'diff':>9}"
      f"{'walk C':>9}{'grad C':>9}{'C ratio':>9}{'sec':>7}")
print("  " + "-" * 91)

wins = {'walk': 0, 'gradstep': 0, 'tie': 0}
for ansatz, H, prob_name in PROBLEMS:
    t0 = time.time()
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    bk = AerSimulator(method='matrix_product_state')
    res, circ = {}, {}
    for arm, factory in ARMS.items():
        finals, nefvs = [], []
        for t in range(N_TRIALS):
            np.random.seed(42 + t)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    opt = factory(ansatz, H, bk)
                B._mult(opt, 1)                      # V3: one circuit per energy
                params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
            except Exception as e:
                print(f"  {prob_name:>16}  {arm} construction failed: {e}")
                break
            E = None
            for ep in range(EPOCHS):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        params = opt.step(params)
                    E = B.report_energy(ansatz, H, params)
                except Exception:
                    break
            if E is not None:
                finals.append(E)
                nefvs.append(B.optimizer_circuits(opt))
            del opt
            gc.collect()
        res[arm] = float(np.mean(finals)) if finals else float('nan')
        circ[arm] = float(np.mean(nefvs)) if nefvs else float('nan')

    d = res['walk'] - res['gradstep']               # negative => walk better
    if abs(d) < 0.09:                               # harness null scale
        wins['tie'] += 1
    elif d < 0:
        wins['walk'] += 1
    else:
        wins['gradstep'] += 1
    print(f"  {prob_name:>16}{exact:>10.4f}{res['walk']:>10.4f}"
          f"{res['gradstep']:>10.4f}{d:>+9.4f}{circ['walk']:>9.0f}"
          f"{circ['gradstep']:>9.0f}"
          f"{circ['walk'] / max(circ['gradstep'], 1):>9.2f}"
          f"{time.time() - t0:>7.0f}", flush=True)

print("  " + "-" * 91)
print(f"  walk better: {wins['walk']}   gradstep better: {wins['gradstep']}"
      f"   tie (<0.09): {wins['tie']}   of {len(PROBLEMS)}")
print()
print("  POSITIVE diff means gradstep reached lower energy. 'C ratio' is the")
print("  circuit saving. Since gradstep costs half the circuits, ties count in")
print("  its favour - the walk must WIN on a clear majority to justify keeping")
print("  the default. Read the 0.09 threshold as this harness's own null scale,")
print("  measured by v27 on a provably identical pair of unitaries.")
