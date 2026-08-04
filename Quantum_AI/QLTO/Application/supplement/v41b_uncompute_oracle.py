"""Does W-dagger connect the energy oracle? The one-line change nobody tested for this.

v41 removed the degree-1 drift and swept the energy imprint up to 10x. The
parameter distribution went EXACTLY UNIFORM (H/Hmax = 1.000) and stayed there.
The imprint marks nothing.

The reason is structural. The controlled evolution acts on the SYSTEM register:

    sum_x |x>|psi_x>   ->   |1> sum_x |x> e^{-iHt}|psi_x>

and e^{-iHt}|psi_x> is not e^{-iE(x)t}|psi_x> unless |psi_x> is an eigenstate. The
energy phase exists only in the OVERLAP <psi_x|e^{-iHt}|psi_x>, and recovering it
on the parameter register requires uncomputing W. With uncompute_w=False - the
shipped default - the system register stays entangled with param and acts as a
which-path marker, so the imprint DECOHERES the parameter register instead of
phasing it.

RESEARCH_NOTES states the requirement and then defaults it off:

    "The coherent readout sum_x <psi_x|e^{-iHt}|psi_x> |x>, WHICH NEEDS W^dag and
     is already implemented, makes the marginal an AMPLITUDE rather than a sample
     mean"

and, separately, "W-dagger REMOVAL ... correct but nearly worthless" - judged on
depth and on the decoded marginals, never on whether it connects the oracle. This
also corrects a claim made earlier from v37c: the imprint moves the decoded step
by 0.325, but as decoherence, not as marking. Those are different effects.

So run the same sweep with uncompute_w=True. With the drift at zero, ANY
structure in the parameter distribution can only have come from the energy, so
this is a clean test of whether the oracle exists at all once it is connected.

    drift 0, uncompute False  ->  uniform, by v41
    drift 0, uncompute True   ->  ?

If structure appears, the walk has had a correct all-degree oracle since it was
written and has never been wired to it, and the degree-1 targeting v38/v39c
measured is a consequence of the missing W-dagger rather than of the drift's
dominance. If it stays uniform, the phase spread t*(max E - min E) is simply too
small and the fix is evolution time, which costs depth.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import BalancedWalk, heis, maxcut, E

R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
# (drift_scale, imprint_scale, uncompute_w)
GRID = [(1.0, 1.0, False), (1.0, 1.0, True),
        (0.0, 1.0, False), (0.0, 1.0, True),
        (0.0, 2.0, True), (0.0, 4.0, True), (0.0, 8.0, True),
        (0.3, 4.0, True)]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 100)
print("DOES W-DAGGER CONNECT THE ENERGY ORACLE?")
print("=" * 100)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. With drift=0 the ONLY possible")
print(f"  source of structure is the energy, so any enhance != 1 is the oracle.")
print(f"  v41 established: drift=0, uncompute=False gives H/Hmax = 1.000 exactly.")

for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64, merged_walk=False)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        if bi > 2:
            continue
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for k, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[k] = E(ansatz, H, p)
        x_true = sig[int(np.argmin(vals))]
        d1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        x_d1 = np.where(d1 <= 0, 1.0, -1.0)
        agree = bool(np.all(x_d1 == x_true))

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true, i_d1 = idx_of(x_true), idx_of(x_d1)
        # correlation of P(x) against -E(x) is the sharpest oracle test: a real
        # energy oracle makes low-energy vertices MORE likely, monotonically
        order = np.array([idx_of(s) for s in sig])
        e_by_idx = np.empty(2 ** n)
        e_by_idx[order] = vals

        print(f"\n  {name} block {bi}  (n={n}, degree-1 target "
              f"{'CORRECT' if agree else 'WRONG'})")
        print(f"  {'drift':>7}{'imprint':>9}{'W-dag':>7}{'enhance':>9}"
              f"{'e_deg1':>9}{'corr(P,-E)':>12}{'H/Hmax':>9}")
        print("  " + "-" * 62)
        for ds, ims, unc in GRID:
            with contextlib.redirect_stdout(io.StringIO()):
                q = BalancedWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                                 merged_walk=False, uncompute_w=unc)
            q.drift_scale, q.imprint_scale = ds, ims
            q.reset_shot_stream()
            g = q.sense_gradient(centre, R, act)
            cap = {}
            orig = q._run
            def spy(qc, _o=orig, _c=cap):
                r = _o(qc); _c['last'] = r; return r
            q._run = spy
            q._execute_walk(centre, KS, DT, R, act, g)
            q._run = orig

            sel = np.zeros(2 ** n)
            for bs, c in cap['last'].items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            qq = sel[sel > 0]
            ent = float(-np.sum(qq * np.log2(qq))) / n
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])
            print(f"  {ds:>7.1f}{ims:>9.1f}{str(unc):>7}"
                  f"{sel[i_true] * 2 ** n:>9.3f}{sel[i_d1] * 2 ** n:>9.3f}"
                  f"{cc:>12.4f}{ent:>9.3f}", flush=True)

print()
print("  corr(P,-E) is the cleanest single number: positive means low-energy")
print("  vertices are more likely, which is what an energy oracle must do. With")
print("  drift=0 that correlation can ONLY come from the imprint, so a jump from")
print("  ~0 at W-dag False to something positive at True is the whole result.")
