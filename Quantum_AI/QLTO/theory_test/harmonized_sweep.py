"""
Harmonized sweep for the v3 commutator result.

One tau grid, one gain, one N range, four Hamiltonian families, both initial
states.  Replaces the three separate sweeps (thermo_constitutive_law,
thermo_scrambling_crash, k_ancilla_bandwidth_test) whose rows could not be
placed in a single table because they used different parameters.

What this establishes, in order of importance to the paper:

  1. W is NOT a function of I(S:A).  Two families reach exactly 2.000 bits
     (the single-ancilla bound) and give work differing by ~30 orders of
     magnitude, with opposite sign on the non-degenerate one.

  2. W IS predicted by the feedback commutator.  The measured work is
     compared against the first-order expansion

         U_fb = exp(-i (theta/2) K),   K = sum_i |1><1|_A (x) X_i
         W    = <Psi1| A - U_fb^dag A U_fb |Psi1>,   A = I_A (x) H
              = (theta/2) <Psi1| i[A, K] |Psi1>  + O(theta^2)

     and the residual is shown to shrink linearly in theta (SCALING CHECK).

  3. The ordered arm has no fittable W-I relation at any N, so the efficiency
     eta reported in v1-v2 was a slope through noise.  R^2 never approaches
     the R2_MIN = 0.60 acceptance gate used in thermo_scrambling_crash.py.

Initial state matters and is reported explicitly.  If the init is an
eigenstate of H the controlled evolution produces only a global phase, so
I(S:A) = 0 and the row proves nothing.  That is why the commuting control
(H = sum X) must be run from |0>^n and NOT from |+>^n, and symmetrically why
the non-interacting control (H = sum Z) must be run from |+>^n.  Degenerate
rows are flagged rather than dropped.

E_before is not swept: rho_S after sensing is (1/2)(|psi><psi| +
e^{-iHt}|psi><psi|e^{iHt}) and e^{iHt} H e^{-iHt} = H, so Tr(rho_S H) is
exactly tau-independent.  Verified numerically in BASELINE CHECK.

Outputs harmonized_sweep.csv.
"""

import csv
import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import (DensityMatrix, SparsePauliOp, Statevector,
                                 entropy, partial_trace)
from scipy.linalg import expm
from scipy.stats import linregress

warnings.filterwarnings("ignore")

TAUS = np.linspace(0.05, 1.5, 20)
THETA = 0.2
N_RANGE = range(3, 8)
FAMILIES = ["paper-fig1", "ordered", "chaotic", "non-interacting", "commuting"]
INITS = ["plus", "zero"]

# Only the chaotic family carries randomness; ordered/non-interacting/commuting
# are deterministic by construction, so they are run once and reported without
# error bars rather than plotted with zero-width bands.
CHAOTIC_SEEDS = [0, 100, 200, 300, 400]

# Acceptance gate from thermo_scrambling_crash.py, reproduced here so the
# ordered arm's failure to clear it is visible in the same table.
R2_MIN = 0.60

# Rows whose mutual information never exceeds this are eigenstate-degenerate:
# the init commutes with H, no information is written, and the row is
# uninformative about work extraction either way.
DEGENERATE_I = 1e-9


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def build(n, fam, seed=42):
    rng = np.random.default_rng(seed)
    ops = []
    if fam == "paper-fig1":
        # Byte-for-byte reproduction of thermo_constitutive_law.py's Hamiltonian:
        # legacy np.random.seed(42), J ~ U(-1,1) over all pairs, then
        # h ~ U(-0.5, 0.5).  This is the instance behind Fig. 1, Table I, and
        # the abstract's eta = 0.1104 / R^2 = 0.8852 at N = 4.  Note the field
        # range differs from the 'chaotic' family below (U(-1,1)), which is why
        # the two are NOT interchangeable and both must appear in the table.
        # The seed argument is ignored: this is one fixed instance, not a family.
        np.random.seed(42)
        for i in range(n):
            for j in range(i + 1, n):
                ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}),
                            np.random.uniform(-1, 1)))
        for i in range(n):
            ops.append((lbl(n, **{str(i): "X"}), np.random.uniform(-0.5, 0.5)))
    elif fam in ("ordered", "chaotic"):
        for i in range(n):
            for j in range(i + 1, n):
                J = -1.0 if fam == "ordered" else rng.uniform(-1, 1)
                ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}), J))
        for i in range(n):
            h = 1.0 if fam == "ordered" else rng.uniform(-1, 1)
            ops.append((lbl(n, **{str(i): "X"}), h))
    elif fam == "non-interacting":
        for i in range(n):
            ops.append((lbl(n, **{str(i): "Z"}), 1.0))
    elif fam == "commuting":
        for i in range(n):
            ops.append((lbl(n, **{str(i): "X"}), 1.0))
    return SparsePauliOp.from_list(ops)


def feedback_generator(n):
    """K = sum_i |1><1|_anc (x) X_i, on the n+1 qubit register (anc = qubit 0).

    |1><1| = (I - Z)/2, so each term splits into X_i/2 - Z_anc X_i/2.
    """
    ops = []
    for i in range(n):
        ops.append((lbl(n + 1, **{str(i + 1): "X"}), 0.5))
        ops.append((lbl(n + 1, **{str(i + 1): "X", "0": "Z"}), -0.5))
    return SparsePauliOp.from_list(ops).to_matrix()


def controlled_evolution(evals, evecs, n, tau):
    """|0><0|_anc (x) I  +  |1><1|_anc (x) e^{-iH tau}, on n+1 qubits (anc = 0).

    Qiskit orders kron with the highest-index qubit leftmost, so the system
    factor sits on the left and the ancilla projector on the right.  Built by
    eigendecomposition rather than a fresh expm at every tau.
    """
    u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    return np.kron(np.eye(2 ** n), p0) + np.kron(u_sys, p1)


def cycle(H, n, tau, init, theta=THETA, K_mat=None, A_mat=None, spec=None,
          u_fb=None):
    """One sense-lock-actuate cycle.

    Returns (I, P1, E_before, W_measured, W_identity, W_first_order).

    W_identity is the closed form  <Psi1| A - U_fb^dag A U_fb |Psi1>  evaluated
    as a matrix element.  It is an identity, not an approximation, and should
    agree with the circuit-measured W to machine precision at every theta --
    this is a separate and stronger claim than the O(theta) convergence of
    W_first_order, and is reported separately.
    """
    qa = QuantumRegister(1, "anc")
    qs = QuantumRegister(n, "sys")
    qc = QuantumCircuit(qa, qs)
    if init == "plus":
        qc.h(qs)
    qc.h(qa)
    qc.append(UnitaryGate(controlled_evolution(*spec, n, tau)),
              [qa[0]] + list(qs))
    qc.h(qa)

    psi1 = Statevector(qc)
    rho = DensityMatrix(psi1)
    rS = partial_trace(rho, [0])
    rA = partial_trace(rho, range(1, n + 1))
    info = entropy(rS, base=2) + entropy(rA, base=2) - entropy(rho, base=2)
    p1 = float(np.real(rA.data[1, 1]))
    e_before = float(np.real(rS.expectation_value(H)))

    qc_fb = qc.copy()
    for i in range(n):
        qc_fb.crx(theta, qa[0], qs[i])
    rS_after = partial_trace(DensityMatrix(Statevector(qc_fb)), [0])
    work = e_before - float(np.real(rS_after.expectation_value(H)))

    v = np.asarray(psi1.data)

    # Exact closed form: W = <Psi1| A |Psi1> - <U_fb Psi1| A |U_fb Psi1>.
    uv = u_fb @ v
    work_id = float(np.real(np.vdot(v, A_mat @ v) - np.vdot(uv, A_mat @ uv)))

    # Leading term of that identity: W ~ (theta/2) <i[A, K]> + O(theta^2).
    comm = 1j * (A_mat @ K_mat - K_mat @ A_mat)
    work_fo = (theta / 2.0) * float(np.real(np.vdot(v, comm @ v)))

    return info, p1, e_before, work, work_id, work_fo


def sweep(H, n, init, theta=THETA):
    K_mat = feedback_generator(n)
    Hm = H.to_matrix()
    A_mat = np.kron(Hm, np.eye(2))
    spec = np.linalg.eigh(Hm)
    u_fb = expm(-1j * (theta / 2.0) * K_mat)
    rows = np.array([cycle(H, n, t, init, theta, K_mat, A_mat, spec, u_fb)
                     for t in TAUS])
    info, p1, e_pre, work, work_id, work_fo = (rows[:, k] for k in range(6))

    degenerate = info.max() < DEGENERATE_I
    if degenerate or np.ptp(info) < 1e-12:
        r2_i = slope_i = intercept_i = np.nan
        r2_p = slope_p = intercept_p = np.nan
    else:
        fit_i = linregress(info, work)
        fit_p = linregress(p1, work)
        r2_i, slope_i, intercept_i = fit_i.rvalue ** 2, fit_i.slope, fit_i.intercept
        r2_p, slope_p, intercept_p = fit_p.rvalue ** 2, fit_p.slope, fit_p.intercept

    denom = max(np.abs(work).max(), 1e-18)
    # DEFECT FIX: max_I and max_absW are independent maxima over the tau grid and
    # in general occur at DIFFERENT tau, so quoting them together states a
    # comparison that was never evaluated at a common sensing time. Any
    # matched-information claim must use w_at_max_I, which pairs the work with
    # the tau that actually attains max_I.
    i_arg = int(np.argmax(info))
    return dict(
        max_I=info.max(),
        max_absW=np.abs(work).max(),
        w_at_max_I=float(work[i_arg]),
        tau_at_max_I=float(TAUS[i_arg]),
        tau_at_max_absW=float(TAUS[int(np.argmax(np.abs(work)))]),
        r2_I=r2_i,
        slope_I=slope_i,
        intercept_I=intercept_i,
        r2_P=r2_p,
        slope_P=slope_p,
        intercept_P=intercept_p,
        id_abserr=np.abs(work - work_id).max(),
        fo_relerr=np.abs(work - work_fo).max() / denom,
        e_before_drift=np.ptp(e_pre),
        degenerate=degenerate,
    )


def agg(dicts):
    """Mean/std across seeds for the one family that has randomness."""
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        if k == "degenerate":
            out[k] = any(d[k] for d in dicts)
            continue
        vals = np.array([d[k] for d in dicts], dtype=float)
        out[k] = np.nanmean(vals)
        out[k + "_std"] = np.nanstd(vals)
    return out


def main():
    print("=" * 100)
    print("HARMONIZED SWEEP")
    print(f"  tau in [{TAUS[0]:.2f}, {TAUS[-1]:.2f}] x {len(TAUS)}   theta = {THETA}   "
          f"N = {min(N_RANGE)}-{max(N_RANGE)}   exact evolution (no Trotter)")
    print(f"  chaotic arm averaged over {len(CHAOTIC_SEEDS)} seeds; other families "
          f"are deterministic by construction")
    print("=" * 100)

    records = []
    for init in INITS:
        print(f"\n{'-' * 100}\nINITIAL STATE: |{'+' if init == 'plus' else '0'}>^n"
              f"\n{'-' * 100}")
        print(f"{'N':>3} {'family':>16} {'max I':>8} {'max|W|':>11} {'R2(W~I)':>9} "
              f"{'int_I':>9} {'R2(W~P1)':>9} {'int_P':>9} {'|W-W_id|':>10}  note")
        for n in N_RANGE:
            for fam in FAMILIES:
                if fam == "chaotic":
                    res = agg([sweep(build(n, fam, s), n, init) for s in CHAOTIC_SEEDS])
                else:
                    res = sweep(build(n, fam), n, init)

                note = ""
                if res["degenerate"]:
                    note = "DEGENERATE (init is eigenstate of H)"
                elif not np.isnan(res["r2_I"]) and res["r2_I"] < R2_MIN:
                    note = f"R2 < {R2_MIN} -> eta rejected"

                print(f"{n:>3} {fam:>16} {res['max_I']:>8.3f} {res['max_absW']:>11.3e} "
                      f"{res['r2_I']:>9.3f} {res['intercept_I']:>+9.4f} "
                      f"{res['r2_P']:>9.3f} {res['intercept_P']:>+9.4f} "
                      f"{res['id_abserr']:>10.2e}  {note}")

                records.append(dict(init=init, N=n, family=fam, **res))

    with open("harmonized_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in records for k in r}))
        w.writeheader()
        w.writerows(records)
    print(f"\n[Saved] harmonized_sweep.csv  ({len(records)} rows)")

    # ---------------------------------------------------------------- checks
    print("\n" + "=" * 100)
    print("BASELINE CHECK  --  E_before is tau-independent because [e^{-iHt}, H] = 0")
    print("=" * 100)
    for fam, init in [("chaotic", "plus"), ("ordered", "plus"), ("commuting", "zero")]:
        r = sweep(build(4, fam), 4, init)
        print(f"  N=4 {fam:>16} init |{'+' if init == 'plus' else '0'}>^n :  "
              f"E_before range over tau = {r['e_before_drift']:.3e}")

    print("\n" + "=" * 100)
    print("IDENTITY CHECK  --  W = <Psi1| A - U_fb^dag A U_fb |Psi1> is exact, "
          "not asymptotic")
    print("  max |W_circuit - W_identity| over the tau grid, at each theta")
    print("=" * 100)
    check_fams = ["paper-fig1", "chaotic", "ordered", "non-interacting"]
    print(f"{'theta':>8}" + "".join(f"{f:>18}" for f in check_fams))
    for th in [0.4, 0.2, 0.1, 0.05, 0.025]:
        cells = [sweep(build(4, f), 4, "plus", theta=th)["id_abserr"]
                 for f in check_fams]
        print(f"{th:>8.3f}" + "".join(f"{c:>18.2e}" for c in cells))

    print("\n" + "=" * 100)
    print("SCALING CHECK  --  the FIRST-ORDER form (theta/2)<i[A,K]> is the "
          "leading term of that identity")
    print("  relative residual should fall as O(theta), i.e. halve when theta halves")
    print("=" * 100)
    print(f"{'theta':>8}" + "".join(f"{f:>18}" for f in check_fams))
    for th in [0.4, 0.2, 0.1, 0.05, 0.025]:
        cells = [sweep(build(4, f), 4, "plus", theta=th)["fo_relerr"]
                 for f in check_fams]
        print(f"{th:>8.3f}" + "".join(f"{c:>18.3e}" for c in cells))

    print("\n" + "=" * 100)
    print("PAPER FIGURE 1 FAMILY  --  the instance behind eta = 0.1104, "
          "R^2 = 0.8852, Table I")
    print("  N = 4, legacy seed 42, J ~ U(-1,1), h ~ U(-0.5,0.5), exact evolution")
    print("=" * 100)
    p = sweep(build(4, "paper-fig1"), 4, "plus")
    print(f"  W ~ I(S:A)   :  R^2 = {p['r2_I']:.3f}   slope = {p['slope_I']:+.4f}   "
          f"intercept = {p['intercept_I']:+.4f}")
    print(f"  W ~ P(anc=1) :  R^2 = {p['r2_P']:.3f}   slope = {p['slope_P']:+.4f}   "
          f"intercept = {p['intercept_P']:+.4f}")
    print(f"  The P(anc=1) regression is the better model AND passes through the "
          f"origin;\n  the I(S:A) regression does not. Intercept ratio: "
          f"{abs(p['intercept_I']) / max(abs(p['intercept_P']), 1e-12):.0f}x")

    print("\n" + "=" * 100)
    print("MATCHED-INFORMATION COMPARISON  --  the row pair that ends dE <= eta*I")
    print("=" * 100)
    a = sweep(build(4, "non-interacting"), 4, "plus")
    b = sweep(build(4, "commuting"), 4, "zero")
    for nm, r in [("non-interacting (H = sum Z, init |+>^n)", a),
                  ("commuting       (H = sum X, init |0>^n)", b)]:
        print(f"  {nm}:")
        print(f"      max I  = {r['max_I']:.6f} bits at tau = {r['tau_at_max_I']:.3f},"
              f"  W there = {r['w_at_max_I']:+.4e}")
        print(f"      max|W| = {r['max_absW']:.3e}      at tau = "
              f"{r['tau_at_max_absW']:.3f}   <- different tau, do not pair this one")
    ratio = abs(a['w_at_max_I']) / max(abs(b['w_at_max_I']), 1e-300)
    print(f"\n  At matched information ({a['max_I']:.6f} vs {b['max_I']:.6f} bits, "
          f"differing by {abs(a['max_I'] - b['max_I']):.1e}):")
    print(f"      work differs by {ratio:.1e}x.")
    print("  NOTE: superseded by isospectral_family.py, which holds I(S:A) and E_N")
    print("        exactly constant by construction rather than finding a coincidence.")


if __name__ == "__main__":
    main()
