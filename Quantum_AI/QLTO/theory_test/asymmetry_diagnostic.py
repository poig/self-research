"""Is the work resource cheaply measurable BEFORE running the protocol?

Theorem 2 gives the first-order work as Tr(M_11 Y) with
M_11 = <1| i[rho, A] |1>, and M_11 vanishes exactly when rho commutes with
A = I_A (x) H, i.e. when the state carries no coherence between energy
eigenspaces. So the resource is ASYMMETRY, not correlation. That is a physics
statement; this file asks whether it is also an ENGINEERING one, by reducing the
resource to something measurable with an ordinary expectation value.

REDUCTION. The post-sensing state is

    |Psi_1> = (1/sqrt2) ( |0>|psi> + |1>|psi_1> ),   |psi_1> = e^{-iH tau}|psi>

and its (1,1) ancilla block is <1|rho|1> = (1/2)|psi_1><psi_1|. Hence

    M_11 = <1| i[rho, A] |1> = (i/2) [ |psi_1><psi_1| , H ]

and by cyclicity of the trace,

    Tr(M_11 Y) = (i/2) Tr( |psi_1><psi_1| [H, Y] ) = (i/2) <psi_1| [H,Y] |psi_1>

so that Theorem 2's first-order work collapses to

    W  =  (theta/4) <psi_1| i[H,Y] |psi_1>  +  O(theta^2).            (*)

i[H,Y] is Hermitian (a commutator of two Hermitians times i), so (*) is an
ORDINARY EXPECTATION VALUE on the post-sensing state. No ancilla, no Hadamard
test, no tomography of M_11.

WHY THAT IS CHEAP, AND IT IS THE WHOLE POINT. [H,Y] keeps only the terms of H
that FAIL to commute with Y. For a local generator and a local Hamiltonian that
is a small fraction of H, so the diagnostic costs fewer measurement settings than
one energy evaluation - it is cheaper than the thing it is deciding whether to
run.

WHAT THIS FILE CHECKS
  (a) (*) reproduces the exact W of the full protocol as theta -> 0, at the
      right order: the residual must fall as theta^2, not merely be small
  (b) it reports ZERO exactly when [H,Y] = 0, matching Corollary 1
  (c) the measurement cost: terms and qubit-wise-commuting groups of i[H,Y]
      against those of H itself
  (d) it tracks the resource across a sensing-time sweep, where the work changes
      sign - a go/no-go test has to get the sign right, not just the magnitude
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 4
TAU = 1.042


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def build_H(n, fam, seed=42):
    ops = []
    if fam == "sum-Z":
        for i in range(n):
            ops.append((lbl(n, **{str(i): "Z"}), 1.0))
    elif fam == "paper-fig1":
        rng = np.random.RandomState(seed)
        for i in range(n):
            for j in range(i + 1, n):
                ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}),
                            rng.uniform(-1, 1)))
        for i in range(n):
            ops.append((lbl(n, **{str(i): "X"}), rng.uniform(-0.5, 0.5)))
    elif fam == "heisenberg":
        for i in range(n - 1):
            for p in "XYZ":
                ops.append((lbl(n, **{str(i): p, str(i + 1): p}), 1.0))
    return SparsePauliOp.from_list(ops).simplify()


def sum_X(n):
    return SparsePauliOp.from_list(
        [(lbl(n, **{str(i): "X"}), 1.0) for i in range(n)])


def commutator(A, B):
    """i[A,B] as a SparsePauliOp - Hermitian, and usually far sparser than A."""
    return (SparsePauliOp(1j * (A @ B - B @ A).coeffs.astype(complex),
                          (A @ B - B @ A).paulis)
            if False else (1j * ((A @ B) - (B @ A))).simplify())


def protocol_work(Hm, Ym, n, tau, theta):
    """Exact W for one feedback step, no expansion in theta."""
    d = 2 ** n
    psi = np.ones(d) / np.sqrt(d)                    # |+>^n
    psi1 = expm(-1j * Hm * tau) @ psi
    Psi1 = np.zeros(2 * d, dtype=complex)            # ancilla (x) system
    Psi1[:d] = psi / np.sqrt(2)                      # |0> branch
    Psi1[d:] = psi1 / np.sqrt(2)                     # |1> branch

    A = np.kron(np.eye(2), Hm)
    P1 = np.diag([0.0, 1.0])
    K = np.kron(P1, Ym)
    U = expm(-1j * (theta / 2.0) * K)
    return float(np.real(Psi1.conj() @ (A - U.conj().T @ A @ U) @ Psi1))


def diagnostic(Hm, Ym, n, tau, theta):
    """(theta/4) <psi_1| i[H,Y] |psi_1> - one expectation value."""
    d = 2 ** n
    psi = np.ones(d) / np.sqrt(d)
    psi1 = expm(-1j * Hm * tau) @ psi
    C = 1j * (Hm @ Ym - Ym @ Hm)
    return float(np.real(psi1.conj() @ C @ psi1)) * theta / 4.0


print("=" * 92)
print("ASYMMETRY DIAGNOSTIC — is the work resource measurable before the run?")
print("=" * 92)
print("  W = (theta/4) <psi_1| i[H,Y] |psi_1> + O(theta^2),  |psi_1> = e^{-iH tau}|+>^n")

FAMILIES = ["sum-Z", "paper-fig1", "heisenberg"]

print()
print("  (c) MEASUREMENT COST — the diagnostic observable against H itself")
print(f"  {'family':<14}{'H terms':>9}{'H groups':>10}{'[H,Y] terms':>13}"
      f"{'[H,Y] groups':>14}{'cost ratio':>12}")
print("  " + "-" * 72)
for fam in FAMILIES:
    H = build_H(N, fam)
    Y = sum_X(N)
    C = (1j * ((H @ Y) - (Y @ H))).simplify()
    nz = SparsePauliOp.from_list(
        [(str(p), c) for p, c in zip(C.paulis, C.coeffs) if abs(c) > 1e-12]) \
        if np.any(np.abs(C.coeffs) > 1e-12) else None
    gH = len(H.group_commuting(qubit_wise=True))
    if nz is None:
        print(f"  {fam:<14}{len(H.paulis):>9}{gH:>10}{0:>13}{0:>14}{'0 (free)':>12}")
        continue
    gC = len(nz.group_commuting(qubit_wise=True))
    print(f"  {fam:<14}{len(H.paulis):>9}{gH:>10}{len(nz.paulis):>13}"
          f"{gC:>14}{gC / max(gH, 1):>12.2f}")

print()
print("  (a) ORDER CHECK — the residual must fall as theta^2")
print(f"  {'family':<14}{'theta':>8}{'exact W':>13}{'diagnostic':>13}"
      f"{'residual':>12}{'res/theta^2':>13}")
print("  " + "-" * 73)
for fam in FAMILIES:
    H = build_H(N, fam); Hm = H.to_matrix()
    Ym = sum_X(N).to_matrix()
    for th in (0.4, 0.2, 0.1, 0.05):
        w = protocol_work(Hm, Ym, N, TAU, th)
        d = diagnostic(Hm, Ym, N, TAU, th)
        r = abs(w - d)
        print(f"  {fam if th == 0.4 else '':<14}{th:>8.2f}{w:>13.6f}{d:>13.6f}"
              f"{r:>12.2e}{r / th ** 2:>13.4f}")

print()
print("  (b) VANISHING CHECK — Corollary 1: [H,Y]=0 must give exactly zero")
print(f"  {'case':<26}{'||[H,Y]||':>12}{'exact W':>13}{'diagnostic':>13}")
print("  " + "-" * 64)
n = N
Y = sum_X(n)
# a Hamiltonian built from Y itself commutes with it
H_commuting = SparsePauliOp.from_list([(str(p), float(np.real(c)))
                                       for p, c in zip(Y.paulis, Y.coeffs)])
for tag, H in (("H = sum_i X_i  ([H,Y]=0)", H_commuting),
               ("H = sum_i Z_i  ([H,Y]!=0)", build_H(n, "sum-Z"))):
    Hm = H.to_matrix(); Ym = Y.to_matrix()
    nrm = float(np.linalg.norm(Hm @ Ym - Ym @ Hm))
    print(f"  {tag:<26}{nrm:>12.2e}"
          f"{protocol_work(Hm, Ym, n, TAU, 0.2):>13.2e}"
          f"{diagnostic(Hm, Ym, n, TAU, 0.2):>13.2e}")

print()
print("  (d) SIGN TRACKING — sweep the sensing time, where W changes sign")
H = build_H(N, "paper-fig1"); Hm = H.to_matrix(); Ym = sum_X(N).to_matrix()
print(f"  {'tau':>7}{'exact W':>13}{'diagnostic':>13}{'sign match':>12}")
print("  " + "-" * 45)
agree = 0
taus = np.linspace(0.2, 3.0, 12)
for t in taus:
    w = protocol_work(Hm, Ym, N, t, 0.1)
    dg = diagnostic(Hm, Ym, N, t, 0.1)
    ok = (np.sign(w) == np.sign(dg)) or abs(w) < 1e-12
    agree += bool(ok)
    print(f"  {t:>7.2f}{w:>13.6f}{dg:>13.6f}{'yes' if ok else 'NO':>12}")
print(f"\n  sign agreement: {agree}/{len(taus)}")

print()
print("  A go/no-go test is only useful if it is cheaper than the run it gates.")
print("  Column (c) is that comparison: measurement settings for i[H,Y] against")
print("  settings for H. Anything below 1.00 means the diagnostic costs less than")
print("  a single energy evaluation of the protocol it is deciding about.")
