"""Does the walk concentrate AT ALL? Read the raw distribution, not its mean.

Three hypotheses have now failed in sequence:

  v38  the product mixer converges to the degree-1 argmin, wrong on 7/16 blocks
       with regret to 0.889, while a degree-<=2 target is exact everywhere.
       CONFIRMED by exact enumeration.
  v39  so swap in Grover's diffuser.            14/40 -> 14/40. NO CHANGE.
  v39b so add the oracle-diffuser alternation.  14/40 -> 14/40. NO CHANGE.
       (product mixer with alternation: 12/40, marginal.)

Every one of those measured the DECODED step, which is a weighted mean over the
sampled corners. A mean cannot report a concentration. If the walk's output
distribution were sharply peaked on the true corner, the mean would still sit
near the centre of the hypercube whenever the peak is not overwhelming, and every
test above would read the same regardless.

So stop decoding and look at the distribution itself:

    p_true      P(x_true) from the walk's param marginal
    enhance     p_true / 2^-n, the enhancement over uniform. Grover's whole
                content is that this grows with step count.
    argmax==?   is the MODE of the distribution the true corner
    H/Hmax      normalised Shannon entropy; 1.0 is uniform, 0 is a point mass

This separates two very different situations that every previous test conflated:

  concentration EXISTS, decode discards it   -> the fix is the decoder, and the
      notes already have the candidates (argmin, top-m, Boltzmann; v4_argmin
      found hard argmin LOSES and Boltzmann TIES, which would be evidence
      against concentration, so this is a real cross-check)
  concentration DOES NOT EXIST               -> the walk is not doing anything
      Grover-like, and the reason has to be found upstream of both the mixer and
      the decode

Also reported: the same statistics for the ancilla-conditioned and unconditioned
marginals, since the anc=1 post-selection is what makes the decode non-separable
and could itself be destroying a peak.
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
    from v39b_alternating_oracle import AltWalk, heis, maxcut, h2, E, mk


def marginals(counts, n):
    """P(x) over param strings, conditioned on anc=1 and unconditioned."""
    sel = np.zeros(2 ** n)
    allc = np.zeros(2 ** n)
    for bs, c in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        a, x = parts[0][-1], parts[1].replace(" ", "")
        idx = int(x, 2)
        allc[idx] += c
        if a == '1':
            sel[idx] += c
    return (sel / max(sel.sum(), 1)), (allc / max(allc.sum(), 1))


def stats(p, i_true, n):
    q = p[p > 0]
    ent = float(-np.sum(q * np.log2(q))) / n if n else 0.0
    return float(p[i_true]), int(np.argmax(p)), ent


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
ARMS = [('once_prod', False, False), ('step_prod', True, False),
        ('step_glob', True, True)]
PROBLEMS = [("H2", h2()), ("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 104)
print("DOES THE WALK CONCENTRATE? Raw param distribution, not the decoded mean.")
print("=" * 104)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. 'enhance' = P(x_true)/2^-n;")
print(f"  Grover's entire content is that this grows with step count. 1.0 = uniform.")
print()
print(f"  {'problem':>15}{'blk':>4}{'n':>3}{'arm':>11}{'p_true':>9}"
      f"{'enhance':>9}{'mode=x*':>9}{'H/Hmax':>9}{'p_max':>9}")
print("  " + "-" * 78)

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
        vals = np.empty(len(sig))
        for k, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[k] = E(ansatz, H, p)
        x_true = sig[int(np.argmin(vals))]
        # counts index bit i of the param string is act[i]; little-endian in the
        # decode, so build the matching index for x_true
        bits = ''.join('1' if x_true[i] > 0 else '0' for i in range(n))[::-1]
        i_true = int(bits, 2)

        for a, _, _ in ARMS:
            q = qs[a]
            q.reset_shot_stream()
            g = q.sense_gradient(centre, R, act)
            cap = {}

            orig = q._run
            def spy(qc, _o=orig, _c=cap):
                r = _o(qc)
                _c['last'] = r
                return r
            q._run = spy
            q._execute_walk(centre, KS, DT, R, act, g)
            q._run = orig

            sel, allc = marginals(cap['last'], n)
            pt, mode, ent = stats(sel, i_true, n)
            print(f"  {name if a == ARMS[0][0] else '':>15}"
                  f"{bi if a == ARMS[0][0] else '':>4}"
                  f"{n if a == ARMS[0][0] else '':>3}{a:>11}{pt:>9.4f}"
                  f"{pt * (2 ** n):>9.3f}{str(mode == i_true):>9}"
                  f"{ent:>9.3f}{sel.max():>9.4f}", flush=True)
        print("  " + "." * 78)

print()
print("  enhance ~ 1.0 everywhere means NO concentration: the walk leaves the")
print("  hypercube distribution essentially uniform and the decode is averaging")
print("  noise. In that case neither the mixer nor the decoder is the bottleneck -")
print("  the marking itself is too weak, and the quantity to check next is the")
print("  PHASE SPREAD the imprint writes, (max E - min E) * t, against pi.")
print("  enhance >> 1 with mode == x* means the concentration is real and the")
print("  weighted-mean decode is throwing it away.")
