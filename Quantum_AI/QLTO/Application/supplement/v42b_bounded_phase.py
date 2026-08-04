"""Degree-2 drift with the phase BOUNDED. The two findings of the day, combined.

v42 added the degree-2 Walsh terms to the drift and the oracle got 4x WORSE:
mean corr(P,-E) 0.4535 -> 0.1170, with enhancement collapsing (MaxCut blk0
4.784 -> 0.943). T7's verdict was right; its instrument just could not say why.

The why is the wrap. The degree-2 terms went into the SAME unbounded phase
channel at the SAME accumulated scale of ~23.9. With |E_hat_ij|/R ~ 0.1-0.5 that
is 2-12 rad per pair, times n(n-1)/2 = 6 pairs: 14-72 radians of total phase,
aliased many times. Six more wrapping terms is not more information.

THE FIX FOLLOWS FROM THE DERIVATION, not from tuning. The walk marks vertex x by

    phi(x) = ACC * [ sum_i g1_i x_i + sum_{i<j} g2_ij x_i x_j ],
    ACC = 0.5 pi / sqrt(R) * pi * dt * k/2  = 23.9 at shipped settings

and which vertex is marked depends only on the RELATIVE values of phi, never on
its overall scale - as long as phi does not wrap. So normalise:

    max_x |phi(x)| <= ACC * ( sum_i |g1_i| + sum_{i<j} |g2_ij| )   (triangle bound)
    norm = PHI_MAX / that,     applied to g1 and g2 TOGETHER

This preserves the phase's shape exactly while guaranteeing no aliasing, and it
is not the v37f rescale: that shrank the drift by a fixed constant and shrank the
STEP with it. Here the mixer beta is untouched, so the step size is unchanged;
only the marking is made faithful.

With PHI_MAX <= pi the oracle becomes a monotone function of the degree-<=2
truncation of E, and v38 proved by enumeration that the argmin of that truncation
is the TRUE argmin on all 16 blocks, regret2 = 0.000.

SWEPT: PHI_MAX over pi/2 .. 8pi, for degree-1 and degree-1+2 drifts, against the
unnormalised shipped drift as the control. If bounded deg1+2 beats bounded deg1
and both beat the control, then the degree-2 information was always usable and
the channel was always the obstruction.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import efficient_su2
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import Deg2Walk, sense_deg12

R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
ACC = 0.5 * np.pi / np.sqrt(R) * np.pi * DT * KS / 2.0
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]
CONFIGS = ([('shipped-deg1', False, None), ('shipped-deg1+2', True, None)]
           + [(f'bd1  PHI={p:.2f}', False, p) for p in
              (np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi)]
           + [(f'bd12 PHI={p:.2f}', True, p) for p in
              (np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi)])

print("=" * 96)
print("BOUNDED-PHASE DRIFT — the degree-2 oracle without the wrap")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Accumulated scale ACC = {ACC:.2f}.")
print(f"  Normalisation touches only the DRIFT coefficients; beta and therefore")
print(f"  the step size are untouched, which is what separates this from v37f.")

summ = {}
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
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        x_true = sig[int(np.argmin(vals))]
        d1e = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        d1ok = bool(np.all(np.where(d1e <= 0, 1.0, -1.0) == x_true))

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true = idx_of(x_true)
        e_by_idx = np.empty(2 ** n)
        e_by_idx[np.array([idx_of(s) for s in sig])] = vals

        print(f"\n  {name} block {bi}  (n={n}, degree-1 target "
              f"{'CORRECT' if d1ok else 'WRONG'})")
        print(f"  {'config':>16}{'corr(P,-E)':>12}{'enh_true':>10}"
              f"{'mode=x*':>9}{'phi_max':>10}")
        print("  " + "-" * 57)

        for tag, use2, phimax in CONFIGS:
            with contextlib.redirect_stdout(io.StringIO()):
                q = Deg2Walk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                             merged_walk=False)
            q.use_deg2 = use2
            q.reset_shot_stream()
            g1, g2 = sense_deg12(q, centre, R, act)
            l1 = float(np.sum(np.abs(g1)))
            l2 = float(np.sum(np.abs(np.triu(g2, 1)))) if use2 else 0.0
            raw = ACC * (l1 + l2)
            if phimax is not None and raw > 1e-12:
                nrm = phimax / raw
                g1, g2 = g1 * nrm, g2 * nrm
                shown = phimax
            else:
                shown = raw
            counts = q.walk(centre, KS, DT, R, act, g1, g2)

            sel = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])
            summ.setdefault(tag, []).append((cc, sel[i_true] * 2 ** n,
                                             int(np.argmax(sel)) == i_true))
            print(f"  {tag:>16}{cc:>12.4f}{sel[i_true] * 2 ** n:>10.3f}"
                  f"{str(int(np.argmax(sel)) == i_true):>9}{shown:>10.2f}",
                  flush=True)

print("\n" + "=" * 96)
print(f"  {'config':>16}{'mean corr':>12}{'min':>9}{'frac>0':>9}"
      f"{'mean enh':>10}{'mode hits':>11}")
print("  " + "-" * 67)
for tag, _, _ in CONFIGS:
    a = np.array([s[0] for s in summ[tag]])
    b = np.array([s[1] for s in summ[tag]])
    m = np.array([s[2] for s in summ[tag]])
    print(f"  {tag:>16}{a.mean():>12.4f}{a.min():>9.4f}{np.mean(a > 0):>9.1%}"
          f"{b.mean():>10.3f}{f'{m.sum()}/{len(m)}':>11}")

print()
print("  If bounded deg1+2 tops the table, the degree-2 information was always")
print("  usable and the unbounded phase channel was the whole obstruction - which")
print("  would make T7 a correct measurement of an incorrectly built circuit.")
print("  If bounded deg1 wins instead, the degree-2 coefficients are too noisy at")
print("  this shot budget, and that is a variance question with a known answer.")
