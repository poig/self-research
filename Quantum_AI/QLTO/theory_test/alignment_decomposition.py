"""
Does the gradient ceiling decompose the barren plateau into a measurable cause?

Robertson + Popoviciu bound the first-order energy response of any variational
parameter with generator G:

    |dE/dtheta|  <=  Delta_H * ( lam_max(G) - lam_min(G) )          (ceiling)

Define the alignment

    alpha = |dE/dtheta| / ( Delta_H * range(G) )   in [0, 1]

so that a vanishing gradient has exactly two possible causes, and they are
distinguishable from measurements the optimiser already takes:

    Delta_H small  ->  the state is near an eigenstate (converged, or trapped)
    alpha small    ->  the generator is nearly orthogonal to the descent
                       direction; the ceiling is fine, the frame is not

The question this script answers is whether the split is informative at scale.
If barren plateaus were a variance effect, Delta_H would collapse with N. If they
are an alignment effect, Delta_H stays polynomial while alpha collapses.

Measured over Haar-random states (the regime barren-plateau results describe) and
over states from a hardware-efficient ansatz at random parameters, for a local
Heisenberg chain. G = sum_i X_i, the feedback generator of the protocol, with
range(G) = 2N.

This is a measurement, not a theorem: the bound itself is textbook (Robertson
1929, Popoviciu 1935). What is being tested is whether the decomposition it
induces separates the two failure modes in practice.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

N_RANGE = range(2, 11)
N_SAMPLES = 200


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def heisenberg(n):
    ops = []
    for i in range(n - 1):
        for p in "XYZ":
            ops.append((lbl(n, **{str(i): p, str(i + 1): p}), 1.0))
    return SparsePauliOp.from_list(ops).to_matrix()


def sum_x(n):
    return sum(SparsePauliOp(lbl(n, **{str(i): "X"})).to_matrix() for i in range(n))


def haar_state(d, rng):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def hea_state(n, rng, layers=4):
    """Hardware-efficient ansatz at random parameters, applied to |0...0>."""
    d = 2 ** n
    v = np.zeros(d, dtype=complex)
    v[0] = 1.0
    for _ in range(layers):
        for q in range(n):                      # single-qubit RY, RZ
            for gate in ("Y", "Z"):
                th = rng.uniform(0, 2 * np.pi)
                P = SparsePauliOp(lbl(n, **{str(q): gate})).to_matrix()
                v = (np.cos(th / 2) * v) - 1j * np.sin(th / 2) * (P @ v)
        for q in range(n - 1):                  # CZ entangler (diagonal, cheap)
            mask = np.array([(i >> q) & 1 and (i >> (q + 1)) & 1
                             for i in range(d)])
            v = np.where(mask, -v, v)
    return v / np.linalg.norm(v)


def measure(v, Hm, G, rng_G):
    e1 = float(np.real(v.conj() @ (Hm @ v)))
    e2 = float(np.real(v.conj() @ (Hm @ (Hm @ v))))
    dH = float(np.sqrt(max(e2 - e1 ** 2, 0.0)))
    grad = abs(float(np.real(v.conj() @ ((1j * (Hm @ G - G @ Hm)) @ v))))
    ceiling = dH * rng_G
    return dH, grad, (grad / ceiling if ceiling > 1e-15 else 0.0)


def main():
    rng = np.random.default_rng(3)
    print("=" * 96)
    print("ALIGNMENT DECOMPOSITION OF THE GRADIENT CEILING")
    print("  H = Heisenberg chain,  G = sum_i X_i,  range(G) = 2N,  "
          f"{N_SAMPLES} samples per point")
    print("  ceiling = Delta_H * range(G);   alpha = |grad| / ceiling")
    print("=" * 96)

    for tag, maker in [("Haar-random", "haar"), ("hardware-efficient", "hea")]:
        print(f"\n  {tag} states")
        print(f"  {'N':>3} {'Delta_H':>12} {'|grad|':>13} {'ceiling':>12} "
              f"{'alpha':>12} {'alpha ratio':>12}")
        prev_alpha = None
        for n in N_RANGE:
            Hm = heisenberg(n)
            G = sum_x(n)
            rg = 2.0 * n
            dHs, grads, alphas = [], [], []
            for _ in range(N_SAMPLES):
                v = (haar_state(2 ** n, rng) if maker == "haar"
                     else hea_state(n, rng))
                a, b, c = measure(v, Hm, G, rg)
                dHs.append(a)
                grads.append(b)
                alphas.append(c)
            dH_m = float(np.mean(dHs))
            gr_m = float(np.mean(grads))
            al_m = float(np.mean(alphas))
            ratio = "" if prev_alpha is None else f"{prev_alpha / max(al_m, 1e-18):>12.2f}"
            print(f"  {n:>3} {dH_m:>12.4f} {gr_m:>13.3e} {dH_m * rg:>12.4f} "
                  f"{al_m:>12.3e} {ratio}")
            prev_alpha = al_m

    print("\n" + "=" * 96)
    print("READING: if Delta_H grows or stays flat while alpha falls geometrically,")
    print("the vanishing gradient is an alignment failure, not a variance failure --")
    print("the ceiling remains available and the frame fails to reach it.")
    print("=" * 96)


if __name__ == "__main__":
    main()
