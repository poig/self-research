"""The X-X correlator as a connectivity witness. The roadmap's last unrun experiment.

research_roadmap_final.md retains this at low priority and specifies it exactly:
"Compute <X_i X_j> from the W-gate statevector for Heisenberg N=4. Compare against
the circuit topology. A few hours of statevector work."

BACKGROUND, and why only the OFF-DIAGONAL is left. The brainstorm proposed reading
the QFIM off the param register in the X basis, on the grounds that <X_i> encodes
the overlap <psi(theta_i - R)|psi(theta_i + R)>. The roadmap killed the diagonal
with an exact calculation: for a gate U_i = exp(-i theta_i G_i / 2) with G_i a
Pauli generator,

    <psi(theta_i)|psi(theta_i + 2R)> = <psi_eff| exp(-i R G_i) |psi_eff>
                                     = cos(R) - i sin(R) <G_i>_eff

so Re<...> = cos(R) EXACTLY, independent of the state, because G_i^2 = I forces
the overlap magnitude to be R-determined rather than state-determined. The QFIM
diagonal F_ii = 1 - <G_i>^2 is therefore invisible, structurally and not as a
protocol defect.

WHAT IS LEFT is the connected correlator

    C_ij = <X_i X_j> - <X_i><X_j>          ( = <X_i X_j> - cos^2(R) if the
                                              diagonal prediction holds )

which is NOT trivially zero when parameters i and j are joined by entanglers
downstream of the block. Whether it maps onto F_ij depends on circuit topology
rather than on the generator algebra, which is why it is a CONNECTIVITY WITNESS
and not a metric estimator.

TWO FALSIFIABLE PREDICTIONS, tested here:

  (1) <X_i> = cos(R) for every i, every centre, every block - to machine
      precision. If this fails, the roadmap's diagonal-is-dead calculation is
      wrong and the QFIM direction reopens.

  (2) C_ij is structured by the entangler topology. efficient_su2(N, reps=1)
      applies a CX chain 0-1, 1-2, ..., N-2:N-1 after the first rotation layers,
      so parameters on qubits further apart along the chain should show weaker
      correlation. A flat or noise-level C_ij means the witness carries nothing.

No shots: this is the exact statevector, so anything nonzero is real.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5

X = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def param_density(q, centre, R, act):
    """rho on the param register of the W-gate state, sys traced out."""
    n, N = len(act), q.N
    qc = QuantumCircuit(QuantumRegister(n, 'param'), QuantumRegister(N, 'sys'))
    qc.h(range(n))
    q._build_w(qc, qc.qregs[0], qc.qregs[1], centre, R, act)
    psi = Statevector(qc).data
    # param is qreg 0, so it occupies the LOW bits: index = p + 2^n * s
    Mx = psi.reshape(2 ** N, 2 ** n)
    return np.einsum('sp,sq->pq', Mx, Mx.conj())


def op_on(n, sites):
    """X on each site in `sites`, identity elsewhere; qubit 0 is the LSB."""
    out = np.ones((1, 1), dtype=complex)
    for k in reversed(range(n)):
        out = np.kron(out, X if k in sites else I2)
    return out


R = 0.6
PROBLEMS = [("Heisenberg N=4", heis(4)), ("Heisenberg N=6", heis(6))]

print("=" * 92)
print("X-X CORRELATOR AS CONNECTIVITY WITNESS")
print("=" * 92)
print(f"  R={R}, exact statevector, no shots. cos(R) = {np.cos(R):.6f}")
print(f"  Prediction 1: <X_i> = cos(R) exactly, for all i - the dead diagonal.")
print(f"  Prediction 2: C_ij = <X_iX_j> - <X_i><X_j> tracks entangler topology.")

for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
    blocks = [b['params'] for b in q.layers if b['params']]

    print(f"\n  ===== {name} | M={M} | {len(blocks)} blocks =====")
    for bi, act in enumerate(blocks):
        n = len(act)
        if n < 2:
            continue
        devs, mats = [], []
        for seed in (11, 12, 13):
            centre = np.random.RandomState(seed).uniform(-np.pi, np.pi, M)
            rho = param_density(q, centre, R, act)
            rho = rho / np.real(np.trace(rho))
            xs = np.array([float(np.real(np.trace(rho @ op_on(n, {i}))))
                           for i in range(n)])
            devs.append(np.max(np.abs(xs - np.cos(R))))
            C = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    xx = float(np.real(np.trace(rho @ op_on(n, {i, j}))))
                    C[i, j] = C[j, i] = xx - xs[i] * xs[j]
            mats.append(C)
        Cm = np.mean(mats, axis=0)

        print(f"\n  block {bi}  (n={n}, params {act[0]}..{act[-1]})")
        print(f"    max |<X_i> - cos(R)| over 3 centres : {max(devs):.3e}"
              f"   -> diagonal {'DEAD as predicted' if max(devs) < 1e-9 else 'ALIVE'}")
        print(f"    connected C_ij, mean over 3 centres:")
        for i in range(n):
            print("      " + "".join(f"{Cm[i, j]:>10.5f}" for j in range(n)))
        iu = np.triu_indices(n, 1)
        off = np.abs(Cm[iu])
        # chain distance between the qubits the two parameters act on
        dist = np.array([abs(i - j) for i in range(n) for j in range(i + 1, n)])
        print(f"    |C| by chain distance:")
        for d in sorted(set(dist)):
            m = dist == d
            print(f"      d={d}: mean |C| = {off[m].mean():.5f}"
                  f"   (n={int(m.sum())} pairs)")
        if off.max() > 1e-9 and len(set(dist)) > 1:
            cc = float(np.corrcoef(dist, off)[0, 1])
            print(f"    corr(chain distance, |C|) = {cc:+.4f}"
                  f"   -> {'decays with distance' if cc < -0.3 else 'no clear decay'}")

print()
print("  Prediction 1 failing would REOPEN the QFIM direction, since the whole")
print("  case for the diagonal being dead is that cos(R) is state-independent.")
print("  Prediction 2 holding gives a free connectivity readout from a circuit")
print("  already being run - useful for ansatz diagnostics, not for the metric.")
print("  |C| at noise level everywhere means the witness carries nothing and the")
print("  last surviving piece of the QFIM direction closes with it.")
