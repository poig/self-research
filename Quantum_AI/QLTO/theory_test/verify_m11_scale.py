"""Is spec(M_11) = +-Delta_H, or +-Delta_H/2?  Settled against the exact work.

A referee reports a factor of two in Corollary cor:variance: the paper states
that for a pure post-sensing branch M_11 has eigenvalues +-Delta_H, and the proof
writes

    M_11 propto i(|chi><phi| - |phi><chi|)

where that "propto" is exactly where a constant can be lost.

WHY THE EXACT WORK IS THE ARBITER. Eigenvalue conventions can be argued about;
the work cannot. W = <Psi_1| A - U^dag A U |Psi_1> is a number produced by the
protocol. If the first-order prediction (theta/2) Tr(M_11 Y) tracks it as
theta -> 0 then the chain from M_11 to W is right, and whichever closed form for
spec(M_11) reproduces THAT is the correct one. So this script computes:

  1. M_11 numerically from its own definition, and its eigenvalues.
  2. Delta_H on the relevant marginal.
  3. The exact W by building U and evaluating the identity.
  4. The first-order slope dW/dtheta at theta -> 0, against BOTH candidate
     closed forms for the pure-branch interval endpoint.

The referee's own check (n=1, H=Z, |phi>=|+>, Y=X) is included as the first case.
"""
import numpy as np
from itertools import product

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def build(n, H, psi, tau, Ygen, theta):
    """|Psi_1> = (|0>|psi> + |1>|phi>)/sqrt2 with |phi> = e^{-iH tau}|psi>."""
    d = 2 ** n
    evals, evecs = np.linalg.eigh(H)
    Ut = evecs @ np.diag(np.exp(-1j * evals * tau)) @ evecs.conj().T
    phi = Ut @ psi
    Psi = np.zeros(2 * d, dtype=complex)
    Psi[:d] = psi / np.sqrt(2)
    Psi[d:] = phi / np.sqrt(2)

    A = np.kron(I2, H)                       # I_A (x) H
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    K = np.kron(P1, Ygen)                    # P_1 (x) Y
    ev, evec = np.linalg.eigh(K)
    U = evec @ np.diag(np.exp(-1j * ev * theta / 2)) @ evec.conj().T
    W = float(np.real(Psi.conj() @ (A - U.conj().T @ A @ U) @ Psi))
    return Psi, A, W, phi


def m11_of(Psi, A, n):
    """M_11 = <1| i[rho, A] |1>, the |1><1| ancilla block."""
    d = 2 ** n
    rho = np.outer(Psi, Psi.conj())
    C = 1j * (rho @ A - A @ rho)
    return C[d:, d:]                          # ancilla-|1> diagonal block


CASES = [
    # (label, n, H, psi, tau, Y)
    ("referee check: H=Z, |phi>=|+>, Y=X", 1, Z,
     np.array([1, 0], dtype=complex), None, X),
    ("n=1  H=Z  Y=X  tau=0.7", 1, Z, np.array([1, 1], dtype=complex) / np.sqrt(2),
     0.7, X),
    ("n=2  H=ZZ+X1  Y=X1+X2", 2, kron(Z, Z) + kron(X, I2),
     None, 0.9, kron(X, I2) + kron(I2, X)),
    ("n=3  H=Heis chain  Y=sum X", 3,
     sum(kron(*[p if k in (i, i + 1) else I2 for k in range(3)])
         for i in range(2) for p in [X]) + 0 * kron(I2, I2, I2),
     None, 1.1, sum(kron(*[X if k == i else I2 for k in range(3)])
                    for i in range(3))),
]

print("=" * 96)
print("PART 1.  spec(M_11) against Delta_H")
print("=" * 96)
print(f"  {'case':<38}{'Delta_H':>10}{'max|eig M11|':>15}"
      f"{'ratio':>9}{'verdict':>14}")
print("  " + "-" * 84)

rng = np.random.default_rng(4)
for label, n, H, psi, tau, Ygen in CASES:
    d = 2 ** n
    if psi is None:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi = v / np.linalg.norm(v)
    else:
        psi = psi / np.linalg.norm(psi)
    if tau is None:
        # referee's case: |phi> = |+> directly, so pick tau making it so
        tau = np.pi / 2                      # e^{-iZ pi/2}|0> ~ |0>, use explicit
        phi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        Psi = np.zeros(2 * d, dtype=complex)
        Psi[:d] = psi / np.sqrt(2)
        Psi[d:] = phi / np.sqrt(2)
        A = np.kron(I2, H)
    else:
        Psi, A, _, phi = build(n, H, psi, tau, Ygen, 1e-6)

    M11 = m11_of(Psi, A, n)
    eig = np.linalg.eigvalsh(M11)
    mx = float(np.max(np.abs(eig)))

    # Delta_H on the |1> branch marginal, i.e. on |phi>
    ph = Psi[d:] * np.sqrt(2)
    ph = ph / np.linalg.norm(ph)
    eH = float(np.real(ph.conj() @ H @ ph))
    eH2 = float(np.real(ph.conj() @ H @ H @ ph))
    dH = np.sqrt(max(eH2 - eH ** 2, 0.0))

    ratio = mx / dH if dH > 1e-12 else float('nan')
    verdict = "Delta_H/2" if abs(ratio - 0.5) < 1e-6 else (
        "Delta_H" if abs(ratio - 1.0) < 1e-6 else "neither")
    print(f"  {label:<38}{dH:>10.6f}{mx:>15.6f}{ratio:>9.4f}{verdict:>14}")

print()
print("=" * 96)
print("PART 2.  trace norm ||M_11||_1")
print("=" * 96)
print(f"  {'case':<38}{'||M11||_1':>12}{'Delta_H':>10}{'2*Delta_H':>12}{'which':>12}")
print("  " + "-" * 84)

for label, n, H, psi, tau, Ygen in CASES[1:]:
    d = 2 ** n
    if psi is None:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi = v / np.linalg.norm(v)
    Psi, A, _, phi = build(n, H, psi, tau, Ygen, 1e-6)
    M11 = m11_of(Psi, A, n)
    tn = float(np.sum(np.abs(np.linalg.eigvalsh(M11))))
    ph = phi / np.linalg.norm(phi)
    eH = float(np.real(ph.conj() @ H @ ph))
    eH2 = float(np.real(ph.conj() @ H @ H @ ph))
    dH = np.sqrt(max(eH2 - eH ** 2, 0.0))
    which = "Delta_H" if abs(tn - dH) < 1e-8 else (
        "2 Delta_H" if abs(tn - 2 * dH) < 1e-8 else "neither")
    print(f"  {label:<38}{tn:>12.6f}{dH:>10.6f}{2 * dH:>12.6f}{which:>12}")

print()
print("=" * 96)
print("PART 3.  THE ARBITER: exact W against both candidate first-order slopes")
print("=" * 96)
print("  dW/dtheta at theta->0 must equal (1/2) Tr(M_11 Y). The competing")
print("  pure-branch endpoints are (theta/2) Delta_H * spread(Y)  [paper]")
print("  and (theta/4) Delta_H * spread(Y)  [referee].")
print()
print(f"  {'case':<30}{'exact dW/dth':>15}{'(1/2)Tr(M11 Y)':>17}"
      f"{'paper end':>12}{'referee end':>13}")
print("  " + "-" * 87)

for label, n, H, psi, tau, Ygen in CASES[1:]:
    d = 2 ** n
    if psi is None:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi = v / np.linalg.norm(v)
    th = 1e-5
    _, _, Wp, _ = build(n, H, psi, tau, Ygen, th)
    _, _, Wm, _ = build(n, H, psi, tau, Ygen, -th)
    slope = (Wp - Wm) / (2 * th)

    Psi, A, _, phi = build(n, H, psi, tau, Ygen, th)
    M11 = m11_of(Psi, A, n)
    pred = 0.5 * float(np.real(np.trace(M11 @ Ygen)))

    ph = phi / np.linalg.norm(phi)
    eH = float(np.real(ph.conj() @ H @ ph))
    eH2 = float(np.real(ph.conj() @ H @ H @ ph))
    dH = np.sqrt(max(eH2 - eH ** 2, 0.0))
    yv = np.linalg.eigvalsh(Ygen)
    spread = float(yv.max() - yv.min())
    print(f"  {label:<30}{slope:>15.8f}{pred:>17.8f}"
          f"{0.5 * dH * spread:>12.6f}{0.25 * dH * spread:>13.6f}")

print()
print("  The first two columns agreeing confirms the chain identity -> M_11 -> W.")
print("  The endpoint columns are the MAXIMUM |dW/dtheta| over the isospectral")
print("  orbit of Y, so the true slope must not exceed the correct one, and the")
print("  incorrect one will be exceeded by orbit-optimal configurations.")
