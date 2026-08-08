"""
Does the ordered/chaotic work gap survive normalization by ||H||?

thermo_scrambling_crash.py already normalizes eta by the spectral norm.  This
applies the same normalization to the extracted work itself, to test whether
the raw ordered-vs-chaotic gap is a statement about the dynamics or about the
two families having different energy scales -- ordered couplings are uniform so
||H|| grows ~N^2, while chaotic couplings are zero-mean so ||H|| grows ~N.

Also normalizes the commutator expectation <i[sum X, H]>, which has the same
exposure.
"""

import numpy as np

from harmonized_sweep import (CHAOTIC_SEEDS, TAUS, THETA, build, feedback_generator,
                              lbl)
from harmonized_sweep import cycle as hs_cycle
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp

N_RANGE = range(3, 8)


def max_work_and_norms(n, fam, seed=42):
    H = build(n, fam, seed)
    Hm = H.to_matrix()
    K = feedback_generator(n)
    A = np.kron(Hm, np.eye(2))
    spec = np.linalg.eigh(Hm)
    u_fb = expm(-1j * (THETA / 2.0) * K)

    rows = np.array([hs_cycle(H, n, t, "plus", THETA, K, A, spec, u_fb)
                     for t in TAUS])
    max_w = np.abs(rows[:, 3]).max()

    h_norm = float(np.max(np.abs(np.linalg.eigvalsh(Hm))))

    # BUGFIX: this previously returned ||i[sum X, H]||, the operator norm, under
    # a column labelled as an expectation value.  Those are different objects and
    # only the expectation enters W -- the operator norm sitting near 2||H|| is
    # close to automatic and establishes nothing.  hs_cycle already returns
    # W_first_order = (theta/2) <Psi1| i[A, K] |Psi1>, so the expectation of the
    # feedback commutator on the actual post-sensing state is 2/theta times that.
    # Averaged over the tau grid, as the magnitude the mechanism actually uses.
    comm_exp = float(np.mean(np.abs(rows[:, 5]))) * 2.0 / THETA

    return max_w, h_norm, comm_exp


def main():
    print("=" * 96)
    print("WORK GAP UNDER ||H|| NORMALIZATION   (init |+>^n, exact evolution, "
          f"theta = {THETA})")
    print("=" * 96)
    print(f"{'N':>3} | {'||H|| ord':>9} {'||H|| chao':>10} | {'W ord':>8} {'W chao':>8} "
          f"{'raw':>6} | {'W/||H|| ord':>11} {'W/||H|| chao':>12} {'norm':>6}")
    for n in N_RANGE:
        wo, no, co = max_work_and_norms(n, "ordered")
        chao = [max_work_and_norms(n, "chaotic", s) for s in CHAOTIC_SEEDS]
        wc = np.mean([c[0] for c in chao])
        nc = np.mean([c[1] for c in chao])
        print(f"{n:>3} | {no:>9.2f} {nc:>10.2f} | {wo:>8.4f} {wc:>8.4f} "
              f"{wo / wc:>6.2f} | {wo / no:>11.4f} {wc / nc:>12.4f} "
              f"{(wo / no) / (wc / nc):>6.2f}")

    print("\n" + "=" * 96)
    print("COMMUTATOR EXPECTATION  <Psi1| i[A,K] |Psi1>, mean |.| over tau -- the quantity entering W")
    print("=" * 96)
    print(f"{'N':>3} | {'ord':>9} {'chao':>9} {'raw ratio':>10} | "
          f"{'ord/||H||':>10} {'chao/||H||':>11} {'norm ratio':>11}")
    for n in N_RANGE:
        _, no, co = max_work_and_norms(n, "ordered")
        chao = [max_work_and_norms(n, "chaotic", s) for s in CHAOTIC_SEEDS]
        cc = np.mean([c[2] for c in chao])
        nc = np.mean([c[1] for c in chao])
        print(f"{n:>3} | {co:>9.3f} {cc:>9.3f} {co / cc:>10.2f} | "
              f"{co / no:>10.3f} {cc / nc:>11.3f} {(co / no) / (cc / nc):>11.2f}")


if __name__ == "__main__":
    main()
