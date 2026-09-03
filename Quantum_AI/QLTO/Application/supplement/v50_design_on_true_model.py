"""Degree-2, redone against the VALIDATED model. Optimisation, not sweeps.

v49b validated the complete walk model to machine precision - TVD 0.00000, corr
1.0000, every block, every arm:

    P(y) ~ || |psi_y> - sum_x V_{yx} U_t |psi_x> ||^2

so the phase-design question can now be OPTIMISED against the real objective
instead of swept on the simulator. Every earlier design attempt optimised
P ~ sin^2(phi/2), a model that omits the Gram overlaps entirely, and v47's
"optimal phase returns uniform" is an artefact of that omission rather than a
fact about the circuit.

The degree-2 numbers from v42/v42b/v43 were measured ON THE CIRCUIT so they
stand, but the REASONING given for them was built on the bare model. This redoes
the question properly:

    shipped     g_i = E_hat({i})/R, the degree-1 energy truncation
    opt d1      g_i optimised to maximise P(G_m) under the TRUE model
    opt d2      g_i and g_ij optimised likewise

The distinction that matters and that this project has never drawn: the drift has
always been set to a TRUNCATION OF THE ENERGY, while the phase that maximises
concentration is a different object. With the true model in hand the difference is
measurable rather than assumed.

V is built exactly as the circuit builds it - per step, the diagonal drift phase
then the product mixer, with the kron ordering v49b established - and |psi_x> is
taken from the W GATE, not from assign_parameters, since controlled Z-rotations
carry an x-dependent global phase that becomes relative inside the superposition.

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

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def rx(a):
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * X


def kron_qiskit(mats):
    """kron with qubit 0 as the LEAST significant bit (v49b)."""
    out = np.ones((1, 1), dtype=complex)
    for m in reversed(mats):
        out = np.kron(out, m)
    return out


def make_V(g1, g2, k, dt, R, n, signs):
    """The walk unitary on param: per step, diagonal drift then product mixer."""
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    V = np.eye(2 ** n, dtype=complex)
    iu = np.triu_indices(n, 1)
    for step in range(k):
        s = (step + 0.5) / k
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        sc = gamma * 0.5 * np.pi * gain
        # crz(angle) puts phase exp(i*angle*x/2) on each vertex
        ph = 0.5 * (signs @ (g1 * sc))
        if g2 is not None:
            pair = signs[:, iu[0]] * signs[:, iu[1]]
            ph = ph + 0.5 * (pair @ (2.0 * g2[iu] * sc))
        D = np.diag(np.exp(1j * ph))
        Mx = kron_qiskit([rx(beta)] * n)
        V = Mx @ D @ V
    return V


R, DT, KS = 0.6, 0.5, 15
MS = [1, 2, 4]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("PHASE DESIGN ON THE VALIDATED MODEL — degree-2 asked properly")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}. Exact model, no shots. Reported as")
print(f"  P(G_m) * 2^n / m; 1.0 is uniform, 2^n/m is perfect.")
print()
print(f"  {'problem':>15}{'blk':>4}{'m':>3}{'shipped':>10}{'opt d1':>9}"
      f"{'opt d2':>9}{'d2/d1':>8}{'2^n/m':>8}")
print("  " + "-" * 66)

agg = {m: {'s': [], '1': [], '2': []} for m in MS}
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=65536, sim_seed=17,
                           merged_walk=False)
    BLK = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)
    ev = PauliEvolutionGate(q.H_sense, time=DT * np.pi,
                            synthesis=LieTrotter(reps=1))
    Ut = np.asarray(Operator(ev).data)

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

        # exact energies at each vertex, for ranking
        e_by = np.empty(2 ** n)
        for y in range(2 ** n):
            p = centre.copy()
            p[act] = p[act] + R * signs[y]
            e_by[y] = E(ansatz, H, p)
        rank = np.argsort(e_by)

        q.reset_shot_stream()
        g1_ship, _ = sense_deg12(q, centre, R, act)

        def dist(g1, g2):
            V = make_V(g1, g2, KS, DT, R, n, signs)
            br = psis - V @ UtPsi
            p = np.sum(np.abs(br) ** 2, axis=1)
            t = p.sum()
            return p / t if t > 1e-18 else np.ones(2 ** n) / 2 ** n

        iu = np.triu_indices(n, 1)
        for m in MS:
            mask = np.zeros(2 ** n, dtype=bool)
            mask[rank[:m]] = True
            f = 2 ** n / m

            def obj1(v):
                return -f * float(dist(v, None)[mask].sum())

            def obj2(v):
                g2 = np.zeros((n, n))
                g2[iu] = v[n:]
                return -f * float(dist(v[:n], g2)[mask].sum())

            rng = np.random.RandomState(bi * 10 + m)
            b1 = -obj1(g1_ship)
            for r in range(6):
                x0 = g1_ship * (1 + 0.5 * rng.randn(n)) if r == 0 \
                    else rng.randn(n) * 0.8
                b1 = max(b1, -minimize(obj1, x0, method='BFGS',
                                       options={'maxiter': 400}).fun)
            b2 = b1
            for r in range(6):
                x0 = np.concatenate([g1_ship, np.zeros(len(iu[0]))]) if r == 0 \
                    else rng.randn(n + len(iu[0])) * 0.8
                b2 = max(b2, -minimize(obj2, x0, method='BFGS',
                                       options={'maxiter': 500}).fun)
            ship = f * float(dist(g1_ship, None)[mask].sum())
            agg[m]['s'].append(ship); agg[m]['1'].append(b1); agg[m]['2'].append(b2)
            print(f"  {name if m == MS[0] else '':>15}{bi if m == MS[0] else '':>4}"
                  f"{m:>3}{ship:>10.3f}{b1:>9.3f}{b2:>9.3f}"
                  f"{b2 / b1 if b1 > 1e-9 else 0:>8.2f}{f:>8.1f}", flush=True)
        print("  " + "." * 66)

print(f"\n  {'m':>4}{'shipped':>10}{'opt d1':>9}{'opt d2':>9}"
      f"{'d1/ship':>10}{'d2/d1':>8}")
print("  " + "-" * 50)
for m in MS:
    s = np.mean(agg[m]['s']); a = np.mean(agg[m]['1']); b = np.mean(agg[m]['2'])
    print(f"  {m:>4}{s:>10.3f}{a:>9.3f}{b:>9.3f}{a / s:>10.2f}{b / a:>8.2f}")

print()
print("  d1/ship is what the shipped drift gives up by being an energy truncation")
print("  rather than an optimised phase - the distinction this project has never")
print("  drawn. d2/d1 is what the pairwise terms are worth once the objective is")
print("  right, which is the question v42/v42b/v43 could not ask.")
