"""Finish v7: is the uniform beta simply mistuned? Now answerable exactly.

THIS IS NOT A NEW IDEA. The mixer has been worked on repeatedly and this file
completes one specific question that was left open for lack of seeds:

  merged_walk   SHIPPED. Fuses CRZ and CRX into one tilted-axis controlled
                rotation, -37% walk depth (162->102 at N=4, 246->156 at N=6),
                paired at 12 seeds, -0.0032 +- 0.0101, better on 7/12. Not an
                equivalent rewrite: 0.813 in operator norm at the angles used,
                so it is different dynamics at lower depth.
  v7_mixer      UNFINISHED. Non-uniform beta per coordinate,
                beta_i = beta (1 + lambda (1 - |g_i|/max|g|)). ALL EIGHT
                nonzero-lambda cells beat uniform. But BOTH SIGNS HELPED EQUALLY
                - lambda=-0.5 and lambda=+1.0 both gave -0.041 - which kills the
                "explore the flat directions" mechanism, and the note's own read
                was: "more likely the uniform beta is simply mistuned and any
                perturbation of the AVERAGE mixing amount helps." It could not be
                closed because the effect equalled cross-run reproducibility;
                it needed 20 seeds with an interleaved control.
  v39, v39b     global reflection / Grover diffuser, this session. No change.

WHAT IS ACTUALLY NEW is not the question but the instrument. v49b validated the
complete walk model to TVD 0.00000, and that model is DETERMINISTIC - no shot
noise, no cross-run spread, no seeds. The blocker that stopped v7 does not exist
here. "Is uniform beta mistuned?" becomes an exact optimisation.

v50 optimised the drift coefficients and found the shipped energy truncation 2.42x
off its own optimum, but held beta_s = (1-s) pi dt fixed. That ramp was never
derived from anything - it is the anneal ramp, and v44b showed the anneal reading
yields no handle - so there is no argument that it is right, and v7's data already
points at it being wrong.

FOUR ARMS, all on the exact model (v49b, TVD 0.00000), so this costs milliseconds:

    shipped      drift = energy truncation, beta = shipped ramp
    opt drift    drift optimised, beta = shipped ramp          (this is v50)
    opt mixer    drift = energy truncation, beta_s free
    opt both     both free

beta_s is optimised FREELY - all k of them, no functional form imposed - because
v7 already tried a one-parameter family and could not separate its effect from
noise. A free schedule bounds what ANY parametrisation could reach, so a small
gain here closes v7 in the negative for good.

If 'opt mixer' is close to 'shipped' the ramp is already near optimal, v7's 8/8
was the mistuning it suspected but a negligible one, and the mixer is not where
the headroom is. If it is close to 'opt drift' then the mixer is worth as much as
the phase - and v39's "the Grover diffuser changes nothing" was a comparison
between two mixer families on a schedule that suited neither, which would make
that verdict unsafe too.

Reported as P(G_m) * 2^n / m: 1.0 is uniform, 2^n/m is a perfect hit.
"""
import sys, os, contextlib, io, itertools
import numpy as np
from scipy.optimize import minimize

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector, Operator
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v50_design_on_true_model import rx, kron_qiskit

R, DT, KS = 0.6, 0.5, 15
MS = [1, 2, 4]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]


def make_V(g1, betas, R, n, signs, k, dt):
    """V = prod_s M(beta_s) D(phi_s), with beta_s supplied rather than assumed."""
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    V = np.eye(2 ** n, dtype=complex)
    for step in range(k):
        s = (step + 0.5) / k
        sc = (s * np.pi * dt) * 0.5 * np.pi * gain
        D = np.diag(np.exp(1j * 0.5 * (signs @ (g1 * sc))))
        V = kron_qiskit([rx(betas[step])] * n) @ D @ V
    return V


print("=" * 92)
print("OPTIMISING THE MIXER — the schedule nobody derived")
print("=" * 92)
print(f"  R={R}, dt={DT}, k={KS}. Exact model, no shots. beta_s free (all {KS}).")
print(f"  Shipped ramp is beta_s = (1-s) pi dt, which was never derived.")
print()
print(f"  {'problem':>15}{'blk':>4}{'m':>3}{'shipped':>9}{'opt drift':>11}"
      f"{'opt mixer':>11}{'opt both':>10}{'2^n/m':>8}")
print("  " + "-" * 71)

agg = {m: {k: [] for k in ('s', 'd', 'x', 'b')} for m in MS}
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=65536, sim_seed=17,
                           merged_walk=False)
    BLK = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)
    Ut = np.asarray(Operator(PauliEvolutionGate(
        q.H_sense, time=DT * np.pi, synthesis=LieTrotter(reps=1))).data)
    beta_ship = np.array([(1.0 - (s + 0.5) / KS) * np.pi * DT for s in range(KS)])

    for bi, act in enumerate(BLK):
        n = len(act)
        signs = np.array([[1.0 if (y >> i) & 1 else -1.0 for i in range(n)]
                          for y in range(2 ** n)])
        wq = QuantumCircuit(QuantumRegister(n, 'param'), QuantumRegister(N, 'sys'))
        wq.h(range(n))
        wq.append(q.build_w_gate(wq.qregs[0], wq.qregs[1], centre, R, act),
                  list(range(n + N)))
        wpsi = Statevector(wq).data
        psis = np.zeros((2 ** n, 2 ** N), dtype=complex)
        for i, a in enumerate(wpsi):
            psis[i & (2 ** n - 1), i >> n] = a
        psis *= np.sqrt(2 ** n)
        UtPsi = psis @ Ut.T

        e_by = np.empty(2 ** n)
        for y in range(2 ** n):
            p = centre.copy(); p[act] = p[act] + R * signs[y]
            e_by[y] = E(ansatz, H, p)
        rank = np.argsort(e_by)
        q.reset_shot_stream()
        g_ship, _ = sense_deg12(q, centre, R, act)

        def dist(g1, betas):
            V = make_V(g1, betas, R, n, signs, KS, DT)
            p = np.sum(np.abs(psis - V @ UtPsi) ** 2, axis=1)
            t = p.sum()
            return p / t if t > 1e-18 else np.ones(2 ** n) / 2 ** n

        for m in MS:
            mask = np.zeros(2 ** n, dtype=bool)
            mask[rank[:m]] = True
            f = 2 ** n / m
            rng = np.random.RandomState(bi * 13 + m)

            def score(g1, betas):
                return f * float(dist(g1, betas)[mask].sum())

            ship = score(g_ship, beta_ship)
            best = {'d': ship, 'x': ship, 'b': ship}
            for r in range(5):
                x0 = g_ship * (1 + 0.4 * rng.randn(n)) if r == 0 \
                    else rng.randn(n) * 0.8
                best['d'] = max(best['d'], -minimize(
                    lambda v: -score(v, beta_ship), x0, method='BFGS',
                    options={'maxiter': 350}).fun)
                b0 = beta_ship * (1 + 0.4 * rng.randn(KS)) if r == 0 \
                    else rng.randn(KS) * 0.8
                best['x'] = max(best['x'], -minimize(
                    lambda v: -score(g_ship, v), b0, method='BFGS',
                    options={'maxiter': 350}).fun)
                j0 = np.concatenate([x0, b0])
                best['b'] = max(best['b'], -minimize(
                    lambda v: -score(v[:n], v[n:]), j0, method='BFGS',
                    options={'maxiter': 500}).fun)
            for k_, v in (('s', ship), ('d', best['d']),
                          ('x', best['x']), ('b', best['b'])):
                agg[m][k_].append(v)
            print(f"  {name if m == MS[0] else '':>15}{bi if m == MS[0] else '':>4}"
                  f"{m:>3}{ship:>9.3f}{best['d']:>11.3f}{best['x']:>11.3f}"
                  f"{best['b']:>10.3f}{f:>8.1f}", flush=True)
        print("  " + "." * 71)

print(f"\n  {'m':>4}{'shipped':>10}{'opt drift':>11}{'opt mixer':>11}"
      f"{'opt both':>10}{'mix/ship':>10}{'both/drift':>12}")
print("  " + "-" * 64)
for m in MS:
    a = {k: np.mean(agg[m][k]) for k in agg[m]}
    print(f"  {m:>4}{a['s']:>10.3f}{a['d']:>11.3f}{a['x']:>11.3f}{a['b']:>10.3f}"
          f"{a['x'] / a['s']:>10.2f}{a['b'] / a['d']:>12.2f}")

print()
print("  mix/ship prices the mixer schedule on its own - what the hand-chosen")
print("  ramp gives up. both/drift says whether optimising the mixer adds")
print("  anything ONCE the drift is already optimal, which is the question that")
print("  decides whether every mixer result this session was measured on an")
print("  arbitrary schedule or on a near-optimal one.")
