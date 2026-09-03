"""Before engineering a way to GET curvature, is curvature worth having?

The k-fold Hadamard test (arXiv:2408.05406) computes the kth-order partial
derivative from a single circuit with k ancillas. That is attractive here because
QLTO's symmetric +-R design is STRUCTURALLY blind to the diagonal Hessian -
sigma_i^2 = 1, so the degree-2 Walsh coefficient Ehat({i,i}) is degenerate with
the constant term. The same blindness killed free-energy curvature, the QFIM
diagonal and the X-X witness. A k-fold HT would sidestep it without breaking the
symmetry that gives T1/T2 their O(R^2) cancellation.

BUT THAT IS AN ENGINEERING ANSWER TO A QUESTION NOBODY HAS ASKED YET. Getting the
diagonal Hessian is not actually hard - for a generator with G^2 proportional to
I the energy is f(theta) = a + b cos theta + c sin theta, so

    d^2 f / d theta_i^2  =  -[ f(theta) - f(theta + pi) ] / 2

is exact in TWO evaluations per parameter. The hard question is whether a
diagonally preconditioned step is any better than the plain one. If an ORACLE
Hessian - handed over free, exact, at zero circuit cost - does not beat plain
gradstep, then no estimator for it is worth building, k-fold or otherwise, and
this whole branch closes for the price of one cheap run.

THREE ARMS, all starting from the same parameters:

    sensed          QLTO's sensed gradient, plain bounded max-normalised step.
                    This is what ships.
    sensed + oracle diag  same gradient, step divided by |H_ii| + lambda with the
                    EXACT diagonal Hessian from the statevector. Free curvature.
    exact + oracle diag   exact gradient AND exact diagonal Hessian. The ceiling -
                    what preconditioning could deliver if sensing were perfect.

The third arm matters: if arm 3 does not beat arm 1 either, the limitation is the
LANDSCAPE, not the estimator, and curvature is worthless here regardless of how
it is obtained. That is the outcome the notes' own degree accounting predicts -
T6 found degree-1 plus degree-2 accounts for 99.6% of the landscape, with the
gradient direction already at cos 0.977, so there may simply be very little for
second-order information to correct.

The pi-shift Hessian is cross-checked against a central finite difference before
being used, because it is only exact for generators with G^2 proportional to I
and V5 decomposes the ansatz to RGate, which could invalidate it silently.
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


def mk_energy(ansatz, Hm):
    def E(t):
        v = Statevector(ansatz.assign_parameters(t)).data
        return float(np.real(np.conj(v) @ (Hm @ v)))
    return E


def exact_grad(E, t):
    g = np.zeros(len(t))
    for i in range(len(t)):
        for s in (+1, -1):
            u = t.copy()
            u[i] += s * np.pi / 2
            g[i] += s * E(u) / 2
    return g


def diag_hess_pishift(E, t):
    """-[f(theta) - f(theta+pi)]/2, exact when G^2 ~ I."""
    f0 = E(t)
    h = np.zeros(len(t))
    for i in range(len(t)):
        u = t.copy()
        u[i] += np.pi
        h[i] = -(f0 - E(u)) / 2.0
    return h


def diag_hess_fd(E, t, eps=1e-4):
    f0 = E(t)
    h = np.zeros(len(t))
    for i in range(len(t)):
        a, b = t.copy(), t.copy()
        a[i] += eps
        b[i] -= eps
        h[i] = (E(a) - 2 * f0 + E(b)) / eps ** 2
    return h


PROBS = [('Heis N=4', 4, heis), ('Heis N=6', 6, heis),
         ('MaxCut N=4', 4, maxcut)]
SEEDS = (42, 43, 44, 45)
EPOCHS, SHOTS, LAM = 20, 4096, 0.5

print("=" * 100)
print("IS CURVATURE WORTH HAVING? oracle diagonal Hessian, free of charge")
print("=" * 100)
print(f"  {EPOCHS} epochs, {SHOTS} shots, seeds {SEEDS}, damping lambda={LAM}.")
print(f"  The Hessian is EXACT and costs nothing - if it does not help here it")
print(f"  cannot help when it has to be paid for.")
print()

# validity check on the pi-shift formula before it is trusted anywhere
a4 = efficient_su2(4, reps=2)
E4 = mk_energy(a4, heis(4).to_matrix())
t4 = np.random.RandomState(1).uniform(-np.pi, np.pi, a4.num_parameters)
hp, hf = diag_hess_pishift(E4, t4), diag_hess_fd(E4, t4)
rel = float(np.max(np.abs(hp - hf)) / (np.max(np.abs(hf)) + 1e-12))
print(f"  pi-shift Hessian vs central finite difference: max rel dev {rel:.2e}"
      f"  -> {'VALID' if rel < 1e-3 else 'INVALID, results below are meaningless'}")
print()

print(f"  {'problem':>12}{'seed':>6}{'sensed':>12}{'sens+diag':>12}"
      f"{'exact+diag':>12}{'exact E':>11}")
print("  " + "-" * 65)

tally = {'sensed': [], 'sens+diag': [], 'exact+diag': []}
for name, N, mk in PROBS:
    ansatz = efficient_su2(N, reps=2)
    H = mk(N)
    Hm = H.to_matrix()
    M = ansatz.num_parameters
    E = mk_energy(ansatz, Hm)
    e_exact = float(np.min(np.linalg.eigvalsh(Hm)))

    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v5.QLTOv5(ansatz, H, shot_budget=SHOTS, gradient_mode='direct',
                           sim_seed=7)
    blocks = [b['params'] for b in q.layers if b['params']]

    for sd in SEEDS:
        p0 = np.random.RandomState(sd).uniform(-np.pi, np.pi, M)
        out = {}
        for arm in ('sensed', 'sens+diag', 'exact+diag'):
            p = p0.copy()
            for ep in range(EPOCHS):
                R = max(0.6 * (0.9 ** ep), 1e-4)
                if arm == 'exact+diag':
                    g = exact_grad(E, p)
                else:
                    g = np.zeros(M)
                    for act in blocks:
                        gi, _ = q.sense(p, R, act)
                        g += gi
                if arm != 'sensed':
                    g = g / (np.abs(diag_hess_pishift(E, p)) + LAM)
                mx = np.max(np.abs(g))
                if mx > 0:
                    p = p - R * g / mx
            out[arm] = E(p)
            tally[arm].append(E(p) - e_exact)
        print(f"  {name:>12}{sd:>6}{out['sensed']:>12.4f}"
              f"{out['sens+diag']:>12.4f}{out['exact+diag']:>12.4f}"
              f"{e_exact:>11.4f}", flush=True)
    print("  " + "." * 65)

print()
print(f"  mean gap above the exact ground state, all problems and seeds:")
for arm in ('sensed', 'sens+diag', 'exact+diag'):
    v = np.array(tally[arm])
    print(f"      {arm:>12}  {v.mean():>8.4f}  +- {v.std(ddof=1) / np.sqrt(len(v)):.4f}")
print()
print("  sens+diag ~ sensed means curvature does not help THIS optimiser, and the")
print("  k-fold Hadamard test has nothing to buy here. exact+diag ~ sensed too")
print("  means the ceiling is the LANDSCAPE, not the estimator, and no amount of")
print("  second-order information helps - which is what T6's degree accounting")
print("  (deg1+deg2 = 99.6%, gradient already at cos 0.977) would predict.")
print("  Only a clear win for sens+diag reopens the k-fold HT as worth building.")
