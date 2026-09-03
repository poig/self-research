"""The walk's distribution in closed form WITH the imprint. The missing c_x.

v37b derived the walk's decode from 2x2 matrices and matched the simulator to
0.00241 - but that arm had NO energy imprint. Every later attempt to design a
phase used the resulting pure-phase model,

    P(x) ~ 4 sin^2( phi(x) / 2 ),

and v47 then found that writing the phase which is OPTIMAL under that model
produces a UNIFORM distribution in the real circuit (0.92-0.99 against the
shipped 2.02). So the model is wrong once the imprint is on, and the design built
on it could not have worked.

THE CORRECT FORM. After W and the controlled evolution, post-selecting anc=1,

    sum_x |x> ( |psi_x> - e^{i phi(x)} e^{-iHt} |psi_x> )

so, writing c_x = <psi_x| e^{-iHt} |psi_x>,

    P(x) ~ 2 - 2 Re[ e^{i phi(x)} c_x ]
         = 2 - 2 |c_x| cos( phi(x) + arg c_x ).

The SYSTEM overlap enters twice: arg c_x shifts the phase per vertex, and |c_x|
sets the contrast. Two consequences follow immediately and both match earlier
measurements that had no explanation:

  at phi = 0,  P ~ 2 - 2 Re c_x ~ t^2 <H^2>_x  -  second order and NOT tracking
  <H> in sign, which is the quadrature behaviour v41/v41b found and could not
  place. arg c_x ~ -E_x t is the energy phase, so the imprint does NOT contribute
  "nothing" as this file's predecessors concluded; it contributes the RIGHT phase
  in the WRONG quadrature.

CHECKED HERE against the simulator, three ways:

  (1) predicted P(x) from the formula vs measured, at the shipped settings
  (2) the same with the drift off, isolating the imprint term
  (3) the same with the imprint off, which must reduce to v37b's sin^2 model

c_x is computed exactly by statevector, so this is a prediction with no free
parameters. If it matches, the walk finally has a complete analytic model and the
optimal-phase design can be recomputed against the RIGHT objective:
maximise P(G) under P ~ 2 - 2|c_x| cos(phi(x) + arg c_x), which is a different
optimisation from the one v47 solved.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v43_phase_offset import OffsetWalk

R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
ACC = 0.5 * np.pi / np.sqrt(R) * np.pi * DT * KS / 2.0
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]


def tvd(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


print("=" * 92)
print("FULL CLOSED FORM — P(x) ~ 2 - 2|c_x| cos(phi(x) + arg c_x)")
print("=" * 92)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. c_x = <psi_x|e^{{-iHt}}|psi_x>")
print(f"  computed exactly by statevector. No free parameters.")
print()
print(f"  {'problem':>15}{'blk':>4}{'arm':>12}{'TVD pred':>10}{'TVD unif':>10}"
      f"{'corr':>8}{'|c| mean':>10}")
print("  " + "-" * 69)

for name, H in PROBLEMS:
    N = H.num_qubits
    Hm = H.to_matrix()
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        if bi > 1:
            continue
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        idx = np.array([int(''.join('1' if s[i] > 0 else '0'
                                    for i in range(n))[::-1], 2) for s in sig])

        with contextlib.redirect_stdout(io.StringIO()):
            q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                           merged_walk=False)
        q.reset_shot_stream()
        g1, g2 = sense_deg12(q, centre, R, act)

        # c_x, exactly. H_sense is the traceless shifted operator the walk uses.
        Hs = q.H_sense.to_matrix()
        for tag, tscale, gvec in (('shipped', 1.0, g1),
                                  ('no drift', 1.0, np.zeros(n)),
                                  ('no imprint', 0.0, g1)):
            t = DT * np.pi * tscale
            Ut = expm(-1j * Hs * t)
            cx = np.empty(2 ** n, dtype=complex)
            phi = np.empty(2 ** n)
            for kk, sv in enumerate(sig):
                p = centre.copy()
                p[act] = p[act] + R * sv
                psi = Statevector(ansatz.assign_parameters(p)).data
                cx[idx[kk]] = np.vdot(psi, Ut @ psi)
                phi[idx[kk]] = ACC * float(np.dot(gvec, sv))
            pred = 2.0 - 2.0 * np.real(np.exp(1j * phi) * cx)
            pred = np.clip(pred, 0, None)
            pred = pred / max(pred.sum(), 1e-18)

            # measured
            q.reset_shot_stream()
            counts = q.walk(centre, KS, DT, R, act, gvec,
                            np.zeros((n, n)), False, 0.0) if tscale == 1.0 \
                else None
            if counts is None:
                # imprint off: rebuild without the controlled evolution
                import types
                qc_backup = q.walk
                counts = q.walk(centre, KS, 1e-9, R, act, gvec * (DT / 1e-9),
                                np.zeros((n, n)), False, 0.0)
            meas = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    meas[int(parts[1].replace(" ", ""), 2)] += c
            meas = meas / max(meas.sum(), 1)
            unif = np.ones(2 ** n) / 2 ** n
            cc = float(np.corrcoef(pred, meas)[0, 1]) if meas.std() > 1e-12 else 0.0
            print(f"  {name if tag == 'shipped' else '':>15}"
                  f"{bi if tag == 'shipped' else '':>4}{tag:>12}"
                  f"{tvd(pred, meas):>10.4f}{tvd(unif, meas):>10.4f}"
                  f"{cc:>8.4f}{float(np.mean(np.abs(cx))):>10.4f}", flush=True)
        print("  " + "." * 69)

print()
print("  TVD pred well below TVD unif, with corr near 1, means the formula is the")
print("  walk's actual distribution and the model is complete. The optimal-phase")
print("  design can then be recomputed against")
print("      maximise P(G) under P ~ 2 - 2|c_x| cos(phi(x) + arg c_x)")
print("  which is NOT the objective v47 solved, and is why v47 returned uniform.")
