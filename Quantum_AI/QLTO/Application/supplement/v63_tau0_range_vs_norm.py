"""tau0 = pi/(margin*||H0||) or pi/(margin*range)? V3 and V5 disagree.

V3 line 371:   self.tau0 = pi / (qpe_margin * self.H0_norm + 1e-12)
V5 line 237:   self.tau0 = pi / (qpe_margin * self.H_range + 1e-12)

Both lines sit next to arguments in the SAME file that support them, which is why
the disagreement survived. They are answering different questions:

  V3's comment (lines 355-369) is about ALIASING: "its phase phi = -E tau0/2pi
  must stay inside one turn, so |E| tau0 <= pi". The bound on |E| is max|lambda|,
  which is the SPECTRAL NORM. The comment even names the failure - at margin=1
  the extreme eigenvalues sit on the wrap boundary and a state with weight near
  the spectrum edge decodes to +2.99 when the truth is -3.00.

  _sensing_hamiltonian's docstring is about SIGNAL: "tau must scale with the
  spectral RANGE, not the spectral norm: only the variation of H across the
  search window carries gradient information, and an identity term inflates
  ||H|| without contributing any." Its evidence is LiH, c = -7.883 against a
  range of 1.783.

The signal argument is about an IDENTITY TERM, and H_sense is already traceless -
_sensing_hamiltonian strips the identity into h_offset before returning. So the
LiH ratio 8.950/1.783 is a property of the ORIGINAL H, not of H_sense. Applied to
a traceless H0 the argument has nothing left to bite on, and the two quantities
collapse to within a factor of two:

    lambda_max >= 0 >= lambda_min   (traceless)
    ==>  range/2  <=  ||H0||_2 = max(|lambda_max|,|lambda_min|)  <=  range

so tau0_V5 is between 1x and 2x SMALLER than tau0_V3, always. V5 can therefore
never alias where V3 does not - it is strictly the safer choice - but it pays for
that in resolution, because the k-ancilla readout spans +-pi/tau0 in 2^k bins:

    bin width = 2 * margin * S / 2^k,    S = ||H0|| (V3) or range (V5)

For a SYMMETRIC spectrum - Heisenberg, MaxCut, TFIM are all traceless and
near-symmetric - range/||H0|| ~ 2, so V5 throws away a full bit of a 3-bit
readout. That is a plausible cause of V5-qpe's weak smoke test (E = -5.8055).

MEASURED HERE, per problem: the two S values and their ratio; then the actual
QPE energy readout error and gradient cosine under each convention at matched
ancillas and shots, plus an explicit aliasing check (does any eigenvalue exceed
the +-pi/tau0 window?). If V3's convention wins on accuracy without aliasing, V5
line 237 is a regression and should be changed to match.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
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


def maxcut(N):
    o = []
    for i in range(N):
        j = (i + 1) % N
        s = ["I"] * N
        s[i] = s[j] = "Z"
        o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def tfim(N):
    o = []
    for i in range(N - 1):
        s = ["I"] * N
        s[i] = s[i + 1] = "Z"
        o.append(("".join(s), 1.0))
    for i in range(N):
        s = ["I"] * N
        s[i] = "X"
        o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def h2():
    return SparsePauliOp.from_list([
        ("II", -1.0523), ("IZ", 0.3979), ("ZI", -0.3979),
        ("ZZ", -0.0113), ("XX", 0.1809)])


PROBS = [('Heis N=4', lambda: (efficient_su2(4, reps=2), heis(4))),
         ('Heis N=6', lambda: (efficient_su2(6, reps=2), heis(6))),
         ('MaxCut N=4', lambda: (efficient_su2(4, reps=2), maxcut(4))),
         ('MaxCut N=6', lambda: (efficient_su2(6, reps=2), maxcut(6))),
         ('TFIM N=4', lambda: (efficient_su2(4, reps=2), tfim(4))),
         ('H2', lambda: (efficient_su2(2, reps=2), h2()))]

MARGIN, KA, SHOTS = 2.0, 3, 4096

print("=" * 100)
print("tau0 SCALE: SPECTRAL NORM (V3) vs SPECTRAL RANGE (V5)")
print("=" * 100)
print("  H_sense is traceless, so range/2 <= ||H0||_2 <= range and V5's tau0 is")
print("  always 1x-2x smaller. Cost of the smaller tau0 is readout resolution.")
print()
print(f"  {'problem':>12}{'||H0||':>10}{'range':>10}{'ratio':>8}"
      f"{'lam_min':>10}{'lam_max':>10}{'bin V3':>10}{'bin V5':>10}")
print("  " + "-" * 80)

rows = []
for name, mk in PROBS:
    ansatz, H = mk()
    H0, off, rng = nisq_v5.QLTOv5._sensing_hamiltonian(H)
    ev = np.linalg.eigvalsh(H0.to_matrix())
    nrm = float(max(abs(ev[0]), abs(ev[-1])))
    binv3 = 2 * MARGIN * nrm / 2 ** KA
    binv5 = 2 * MARGIN * rng / 2 ** KA
    print(f"  {name:>12}{nrm:>10.4f}{rng:>10.4f}{rng / nrm:>8.3f}"
          f"{ev[0]:>10.4f}{ev[-1]:>10.4f}{binv3:>10.4f}{binv5:>10.4f}")
    rows.append((name, mk, nrm, rng, ev))

print()
print("  ALIASING CHECK: the QPE window is |E| <= pi/tau0 = margin*S. Aliasing")
print("  happens iff max|lambda| > margin*S, i.e. iff ||H0|| > margin*S.")
print(f"  {'problem':>12}{'window V3':>12}{'window V5':>12}{'max|lam|':>11}"
      f"{'alias V3':>10}{'alias V5':>10}")
print("  " + "-" * 67)
for name, mk, nrm, rng, ev in rows:
    wv3, wv5 = MARGIN * nrm, MARGIN * rng
    print(f"  {name:>12}{wv3:>12.4f}{wv5:>12.4f}{nrm:>11.4f}"
          f"{'YES' if nrm > wv3 else 'no':>10}{'YES' if nrm > wv5 else 'no':>10}")

print()
print("=" * 100)
print("  READOUT AND GRADIENT UNDER EACH CONVENTION")
print("=" * 100)
print(f"  num_ancillas={KA}, margin={MARGIN}, shots={SHOTS}, R=0.6, block 0,")
print(f"  seeds 0-4. 'E err' is |E_qpe - E_exact| at the centre; 'cos' is the")
print(f"  cosine between the block's sensed gradient and the exact gradient.")
print()
print(f"  {'problem':>12}{'E err V3':>11}{'E err V5':>11}{'cos V3':>10}"
      f"{'cos V5':>10}{'winner':>10}")
print("  " + "-" * 64)

R = 0.6
for name, mk, nrm, rng, ev in rows:
    res = {}
    for tag, S in (('V3', nrm), ('V5', rng)):
        eerr, coss = [], []
        for sd in range(5):
            ansatz, H = mk()
            with contextlib.redirect_stdout(io.StringIO()):
                q = nisq_v5.QLTOv5(ansatz, H, shot_budget=SHOTS,
                                   gradient_mode='qpe', num_ancillas=KA,
                                   qpe_margin=MARGIN, sim_seed=sd)
            q.tau0 = np.pi / (MARGIN * S + 1e-12)     # the line under test
            act = q.layers[0]['params']
            p = np.random.RandomState(sd).uniform(-np.pi, np.pi,
                                                  ansatz.num_parameters)
            g, e = q.sense(p, R, act)

            sv = Statevector(ansatz.assign_parameters(p))
            e_ex = float(np.real(sv.expectation_value(H)))
            eerr.append(abs(e - e_ex))

            gx = np.zeros(len(act))
            for i, idx in enumerate(act):
                for sgn in (+1, -1):
                    pp = p.copy()
                    pp[idx] += sgn * np.pi / 2
                    ee = float(np.real(Statevector(
                        ansatz.assign_parameters(pp)).expectation_value(H)))
                    gx[i] += sgn * ee / 2
            gs = np.array([g[idx] for idx in act])
            d = np.linalg.norm(gs) * np.linalg.norm(gx)
            coss.append(float(gs @ gx / d) if d > 0 else 0.0)
        res[tag] = (float(np.mean(eerr)), float(np.mean(coss)))
    w = 'V3' if res['V3'][1] > res['V5'][1] else 'V5'
    if abs(res['V3'][1] - res['V5'][1]) < 0.01:
        w = 'tie'
    print(f"  {name:>12}{res['V3'][0]:>11.4f}{res['V5'][0]:>11.4f}"
          f"{res['V3'][1]:>10.4f}{res['V5'][1]:>10.4f}{w:>10}", flush=True)

print()
print("  If V3 wins on E err and cos with no aliasing flagged, V5 line 237 is a")
print("  resolution regression and should read H0_norm. If they tie, the choice is")
print("  free and V5's is the safer default. Either way the two files should stop")
print("  disagreeing silently.")
