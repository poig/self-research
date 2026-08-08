"""A scan, not a prediction: which modifications break the symmetric interval?

Four hypotheses have failed - asymmetric generator spectra, depolarising the
state, the protocol's own multi-cycle mixing, and breaking time-reversal symmetry
of H. Each time the interval stayed symmetric to ~1e-17. Meanwhile independent
random Hermitian pairs give a normalised asymmetry of ~0.3, so the symmetry is
not a general fact about i[rho,H].

Rather than propose a fifth mechanism, this file scans the space and reports what
the pattern is.

WHAT THE ALGEBRA ALREADY GUARANTEES. Corollary 3 covers any PURE post-sensing
branch: rank(M11) <= 2 with eigenvalues +/-Delta_H, so the interval is symmetric
for every generator, EVERY Hamiltonian, and every sensing unitary - the sensing
step never enters. Proposition 1 adds only that Delta_H is tau-independent when
the sensing commutes with H. So no modification acting on a pure branch can
possibly help, which retro-explains three of the four failures.

That leaves mixedness as the only candidate, and the multi-cycle test says
mixedness alone is not sufficient either. So the question is precisely: WHICH
mixed branches give an asymmetric interval?

AXES SCANNED
  purity of the branch          pure / protocol-generated / random mixed
  reality of H                  real Paulis / complex (DM term)
  reality of the branch         real / complex
  sensing unitary               commuting with H / not commuting

The output is a table of which combinations break symmetry. Any mechanism is read
off afterwards.
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 3
d = 2 ** N
TAU, THETA = 1.042, 0.2
rng = np.random.RandomState(11)


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def op(t):
    return SparsePauliOp.from_list(t).to_matrix()


H_real = op([(lbl(N, **{str(i): "Z", str(j): "Z"}), 0.7)
             for i in range(N) for j in range(i + 1, N)] +
            [(lbl(N, **{str(i): "X"}), 0.4) for i in range(N)])
H_cplx = H_real + op([(lbl(N, **{str(i): "X", str(i + 1): "Y"}), 0.6)
                      for i in range(N - 1)])
Y_gen = sum(op([(lbl(N, **{str(i): "X"}), 1.0)]) for i in range(N))

plus = np.ones(d) / np.sqrt(d)


def sym_defect(M):
    ev = np.sort(np.linalg.eigvalsh(M))
    s = np.max(np.abs(ev))
    return float(np.max(np.abs(ev + ev[::-1])) / s) if s > 1e-13 else 0.0


def M11(branch, Hm):
    return 0.5j * (branch @ Hm - Hm @ branch)


def cycle(rho, Hm, tau=TAU, theta=THETA):
    Uv = expm(-1j * Hm * tau)
    W = np.vstack([np.eye(d) / np.sqrt(2), Uv / np.sqrt(2)])
    big = W @ rho @ W.conj().T
    U = expm(-1j * (theta / 2.0) * np.kron(np.diag([0.0, 1.0]), Y_gen))
    out = U @ big @ U.conj().T
    return out[:d, :d] + out[d:, d:]


def rand_mixed(k, real=False):
    """Mixture of k random pure states."""
    r = np.zeros((d, d), dtype=complex)
    for _ in range(k):
        v = rng.normal(size=d) + (0 if real else 1j * rng.normal(size=d))
        v = v / np.linalg.norm(v)
        r += np.outer(v, v.conj())
    return r / k


print("=" * 96)
print("SCAN — which branch/Hamiltonian combinations give an ASYMMETRIC interval?")
print("=" * 96)
print("  defect = max_k |lam_k + lam_(n-1-k)| / max|lam|.  0 = symmetric.")
print()
print(f"  {'branch (|1> block)':<38}{'H':<10}{'rank M11':>10}{'defect':>12}{'verdict':>12}")
print("  " + "-" * 82)

cases = []

# pure branches, several sensing unitaries - Corollary 3 says all symmetric
Uc = expm(-1j * H_real * TAU)                       # commutes with H_real
Unc = expm(-1j * (H_real + 0.9 * Y_gen) * TAU)      # does NOT commute
Uhaar = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))[0]
for tag, S in (("pure, sensing e^{-iHt}", Uc),
               ("pure, sensing NON-commuting", Unc),
               ("pure, sensing Haar random", Uhaar)):
    v = S @ plus
    cases.append((tag, "real", np.outer(v, v.conj()), H_real))

# protocol-generated mixed branches
rho = np.outer(plus, plus.conj())
for k in (1, 3, 6):
    for _ in range(1 if k == 1 else 2):
        rho = cycle(rho, H_real)
    br = expm(-1j * H_real * TAU) @ rho @ expm(1j * H_real * TAU)
    cases.append((f"protocol mixed, {k} cycles", "real", br, H_real))

rho = np.outer(plus, plus.conj())
for k in (1, 3, 6):
    for _ in range(1 if k == 1 else 2):
        rho = cycle(rho, H_cplx)
    br = expm(-1j * H_cplx * TAU) @ rho @ expm(1j * H_cplx * TAU)
    cases.append((f"protocol mixed, {k} cycles", "complex", br, H_cplx))

# externally prepared mixed branches
for k in (2, 4):
    cases.append((f"random mixed (real), k={k}", "real", rand_mixed(k, True), H_real))
    cases.append((f"random mixed (complex), k={k}", "real", rand_mixed(k, False), H_real))
    cases.append((f"random mixed (complex), k={k}", "complex", rand_mixed(k, False), H_cplx))

for tag, htag, br, Hm in cases:
    M = M11(br, Hm)
    rk = int(np.sum(np.abs(np.linalg.eigvalsh(M)) > 1e-10))
    dfc = sym_defect(M)
    print(f"  {tag:<38}{htag:<10}{rk:>10}{dfc:>12.2e}"
          f"{('symmetric' if dfc < 1e-9 else 'ASYMMETRIC'):>12}")

print()
print("  Read the pattern off the verdict column rather than assuming one. If every")
print("  protocol-generated row is symmetric while externally prepared mixed states")
print("  are not, the protocol's own dynamics is preserving something the general")
print("  case does not - and that invariant, not purity and not reality, is what a")
print("  cooling protocol has to break.")
