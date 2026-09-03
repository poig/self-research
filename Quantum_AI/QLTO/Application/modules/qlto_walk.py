"""QLTO-walk - a THREE-LEVEL design register: gradient AND Hessian, one circuit.

Standalone: numpy and qiskit only. V6 is the frozen log-register line and does
not move. This is the line that adds the second level.

WHY THREE LEVELS. Every parameter enters through a generator with P^2 = I, so
the landscape's Fourier support is exactly {-1,0,1}^M (v135: second harmonic
5.6e-16, no ansatz in use ties a parameter). For a shift u_j = R sigma_j,

    E(theta + R sigma) = sum_k A_k prod_{j: k_j != 0} g_{k_j}(sigma_j),
    g_kappa(sigma)     = 1 + sigma^2 (cos R - 1) + i kappa sigma sin R.

At q=2, sigma^2 == 1 identically, so g collapses to cos R + i kappa sigma sin R.
TWO CONSEQUENCES, and both are artifacts of the second level being absent:

  THE DIAGONAL CURVATURE IS INVISIBLE. d2E/dtheta_j^2 = -sum_{k_j != 0} A_k is
  degenerate with the constant term when sigma_j^2 is constant. v138 PART 2
  measures it appearing the moment a third level exists, and the q=2 case
  returns NaN by construction because sigma^2 - <sigma^2> is identically zero.

  THE SPECTATOR ATTENUATION IS FIXED AT cos R. V6's low-pass filter
  A: c_k -> cos(R)^{|k|-1} c_k comes from the sigma-free part of each spectator
  factor being cos R. With three levels it is

      alpha = p0 + 2 p1 cos R,      p0 + 2 p1 = 1

  so p0 - the fraction of rows that leave a coordinate ALONE - is a knob q=2
  does not have. v138 PART 4 measured the bias tracking 0.37 (1 - alpha^3) to
  5% across p0 = 0 .. 0.9, a 9.8x reduction at p0 = 0.9.

q >= 4 BUYS NOTHING. Per coordinate the landscape supplies exactly three
functions of the shift, sigma -> e^{i k_j R sigma} for k_j in {-1,0,1}:

    (1,1,1),  (e^{-iR}, 1, e^{iR}),  (e^{iR}, 1, e^{-iR})

whose determinant is nonzero for cos R != 1. Three levels span the space; a
fourth adds rows without adding information. This is a hard stop.

HOW THE THIRD LEVEL IS BUILT ON QUBITS. The register is binary, so a level in
{-1,0,+1} is decoded from TWO parities of the row index d:

    a_j = (-1)^popcount(d AND c_j),   b_j = (-1)^popcount(d AND e_j)
    sigma_j = (a_j + b_j) / 2         (+,+) -> +1   (-,-) -> -1   mixed -> 0

Rotations about one axis add, so R sigma_j = (R/2) a_j + (R/2) b_j and the whole
level is realised as

    RY(theta_j + R)   then   cRY(-R) from a_j   then   cRY(-R) from b_j

    a b  ->  angle            sigma
    0 0      theta + R         +1
    1 0      theta              0
    0 1      theta              0
    1 1      theta - R         -1

TWO controlled rotations per parameter against V6's one - the 2x gate cost - and
p0 = 1/2 by construction, giving alpha = (1 + cos R)/2 and about half V6's bias
at the same radius.

WHAT THIS SUPPLIES THAT V6 CANNOT, and all three come from the same shot record:

    1. NEWTON drift        alpha = -(H + mu I)^-1 g   instead of steepest descent
    2. lambda_max          h <~ kappa R^2 sqrt(lambda_max) - the walk's schedule
                           auto-tuned from measured curvature, not a fixed ramp
    3. the degree-2 POTENTIAL  sum a_ij Z_i Z_j - the only thing that gives a
                           walk a barrier at all, since a degree-1 potential has
                           exactly ONE minimum on any connected graph

AND THE THREE ARE NOT EQUALLY HARD. The Hessian's signal is sin^2 R against the
gradient's sin R, so per-entry SNR is R times worse. lambda_max is an aggregate
over ~M^2/2 entries and averages that down; the Newton solve is the fragile one.
The easiest of the three is the one the walk actually needs.

WHAT IS NOT HERE. The walk step itself. supplement/v136 built the cycle mixer
(F^dag . ladder . F, verified against expm at 1.4e-14) and measured that the
cycle register is a PARTICLE - DeltaE = e^{-S0/h}, slope -0.998 against
Liu-Su-Li Eq. 7 - while V3-V6's hypercube register is a SPIN, DeltaE = e^{-n S~},
linear in the PARAMETER COUNT to r = -0.99994. So hypercube tunnelling degrades
exponentially as the model grows and the cycle's does not. Joining that mixer to
the h computed here is the next build; `grad_step` is a regularised Newton move
in the meantime so the sensing can be tested alone.
"""
import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# op(a) op(b) = op(a+b): the one-parameter abelian group condition. Same as V6's
# and the same identity He 2023 Eq. 4 rests on.
_CTRL = {'rx': 'crx', 'ry': 'cry', 'rz': 'crz', 'p': 'cp', 'u1': 'cp'}
LEVELS = np.array([-1.0, 0.0, 1.0])


def resolution_v_columns(want, k):
    """Greedy set of  columns over k bits with RESOLUTION V:

        (a) no column equals the XOR of two others   -> no 3-term relation,
            so a two-factor interaction is never confounded with a main effect
        (b) all pairwise XORs are distinct           -> no 4-term relation,
            so two-factor interactions are separately identifiable

    These are the two conditions V6's  states. The first version of
    this file used plain Gray columns and violated (a) immediately -
    gray(1) XOR gray(2) = 1 XOR 3 = 2 = gray(3) - which aliased the {j,l}
    interaction onto parameter 3's main effect. The gradient and the DIAGONAL
    survived that (measured cos 0.99984, rel Hdiag 0.024) because neither is a
    two-factor contrast; the OFF-DIAGONAL did not, and its error grew as R fell
    (3.8 -> 14.7) because a constant alias divided by sin^2 R blows up.
    """
    chosen, pxor = [], set()
    v = 1
    while len(chosen) < want:
        if v >= (1 << k):
            raise ValueError("k=%d too small for %d resolution-V columns"
                             % (k, want))
        new = {v ^ c for c in chosen}
        if v not in pxor and not (new & pxor) and len(new) == len(chosen):
            pxor |= new
            chosen.append(v)
        v += 1
    return chosen


def level_cols(m, k):
    """(c, e): two column sets giving levels in {-1,0,+1} via two parities.

        a_j = (-1)^popcount(d AND c_j),  b_j = (-1)^popcount(d AND e_j)
        sigma_j = (a_j + b_j)/2

    Both sets are drawn from ONE resolution-V family of 2m columns, because the
    design's effects involve every XOR among {c_j, e_j} - c_j^c_l, c_j^e_l,
    e_j^c_l, e_j^e_l all appear in sigma_j sigma_l, and any of them colliding
    with a main-effect column aliases the Hessian onto the gradient.
    """
    cols = resolution_v_columns(2 * m, k)
    return cols[:m], cols[m:]


def sigma_of(d, c, e):
    """sigma_j(d) = (a_j + b_j)/2 for every j, from the measured row index."""
    a = 1.0 - 2.0 * np.array([bin(d & cj).count('1') & 1 for cj in c])
    b = 1.0 - 2.0 * np.array([bin(d & ej).count('1') & 1 for ej in e])
    return 0.5 * (a + b)


class QLTOWalk:
    def __init__(self, ansatz, hamiltonian, shot_budget=8192, backend=None,
                 sim_seed=None, k_extra=2, decode="ols"):
        self.decode_mode = decode
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.N = hamiltonian.num_qubits
        self.shots = int(shot_budget)
        self.k_extra = int(k_extra)
        self.backend = backend or AerSimulator(seed_simulator=sim_seed)
        self.groups = hamiltonian.group_commuting(qubit_wise=True)
        self._pidx = {p: i for i, p in enumerate(ansatz.parameters)}
        self._tmpl = {}

    def register_width(self, m):
        """k bits. Degree-2 identifiability needs ~2 m^2 rows (Rao, strength 4,
        3 levels); k_extra widens past that."""
        k = max(5, int(np.ceil(np.log2(2 * m * m + 1))) + self.k_extra)
        while True:                       # widen until 2m resolution-V columns fit
            try:
                resolution_v_columns(2 * m, k)
                return k
            except ValueError:
                k += 1

    # -- circuit ---------------------------------------------------------
    def _template(self, active, gi_group):
        key = (tuple(active), gi_group)
        if key in self._tmpl:
            return self._tmpl[key]
        group = self.groups[gi_group]
        m = len(active)
        k = self.register_width(m)
        c, e = level_cols(m, k)
        theta = list(self.ansatz.parameters)
        R = Parameter('R_%d' % m)
        pos = {p: i for i, p in enumerate(active)}

        reg = QuantumRegister(k, 'reg')
        sys = QuantumRegister(self.N, 'sys')
        scr = QuantumRegister(2, 'scr')          # running parities a and b
        qc = QuantumCircuit(reg, sys, scr,
                            ClassicalRegister(self.N, 'cs'),
                            ClassicalRegister(k, 'cr'))
        qc.h(reg)
        prev_c = prev_e = 0
        for inst in self.ansatz.data:
            op = inst.operation
            qs = [sys[self.ansatz.find_bit(b).index] for b in inst.qubits]
            prm = [p for p in op.params
                   if isinstance(p, ParameterExpression) and p.parameters]
            if not prm:
                qc.append(op, qs)
                continue
            j = self._pidx[next(iter(prm[0].parameters))]
            if j not in pos:
                qc.append(op.__class__(theta[j]), qs)
                continue
            if op.name not in _CTRL:
                raise ValueError("no controlled form of '%s'" % op.name)
            a = pos[j]
            # advance both running parities by XOR with the previous column
            for b_ in range(k):
                if (c[a] ^ prev_c) >> b_ & 1:
                    qc.cx(reg[b_], scr[0])
                if (e[a] ^ prev_e) >> b_ & 1:
                    qc.cx(reg[b_], scr[1])
            prev_c, prev_e = c[a], e[a]
            qc.append(op.__class__(theta[j] + R), qs)
            getattr(qc, _CTRL[op.name])(-R, scr[0], qs[0])
            getattr(qc, _CTRL[op.name])(-R, scr[1], qs[0])
        for b_ in range(k):                        # uncompute the parities
            if prev_c >> b_ & 1:
                qc.cx(reg[b_], scr[0])
            if prev_e >> b_ & 1:
                qc.cx(reg[b_], scr[1])
        self._basis(qc, sys, group)
        qc.measure(sys, qc.cregs[0])
        qc.measure(reg, qc.cregs[1])
        t = transpile(qc, self.backend, optimization_level=1)
        self._tmpl[key] = (t, theta, R, c, e, k)
        return self._tmpl[key]

    @staticmethod
    def _basis(qc, sys, group):
        """Rotate each qubit into the group's measurement basis.

        THIS READ ONLY group.paulis.to_labels()[0] - the FIRST term - and that
        is wrong for any group with more than one term. Qubit-wise commuting
        means the terms agree wherever they BOTH act, not that they act on the
        same qubits: the Heisenberg group at N=3 is {'XXI','IXX'}, whose first
        label carries 'I' on qubit 0, so qubit 0 was left in the Z basis while
        'IXX' needed X. Every qubit outside the first term's support was
        measured in the wrong basis.

        Invisible at N=2, where each group holds a single term and the first
        label covers every qubit - which is why qlto_walk's own self-check
        (N=2) reported cos 0.9996 while N=3 sensing ran at cos 0.82 against the
        true gradient. Tier B was unaffected throughout, since it reads
        energies by Statevector and never changes basis; the disagreement
        between the two tiers is what exposed it.

        The fix is to scan ALL labels and take the first non-identity Pauli on
        each qubit, which is well defined precisely because the group is
        qubit-wise commuting.
        """
        labs = group.paulis.to_labels()
        n = len(labs[0])
        for q in range(n):
            p = 'I'
            for lab in labs:
                ch = lab[n - 1 - q]
                if ch != 'I':
                    p = ch
                    break
            if p == 'X':
                qc.h(sys[q])
            elif p == 'Y':
                qc.sdg(sys[q])
                qc.h(sys[q])

    # -- sensing ---------------------------------------------------------
    def sense(self, centre, R, active, exact=False):
        """(gradient, Hessian, energy) from ONE shot record per group.

        exact=True reads E(row) by Statevector instead of sampling - tier B, for
        checking the decode against the design algebra without shot noise.
        """
        m = len(active)
        k = self.register_width(m)
        c, e = level_cols(m, k)
        Nrow = 1 << k
        acc = np.zeros(Nrow)
        cnt = np.zeros(Nrow)

        if exact:
            for d in range(Nrow):
                x = np.array(centre, float)
                x[active] += R * sigma_of(d, c, e)
                acc[d] = float(np.real(Statevector(
                    self.ansatz.assign_parameters(x)
                ).expectation_value(self.hamiltonian)))
                cnt[d] = 1.0
        else:
            for gi in range(len(self.groups)):
                t, theta, Rp, _, _, _ = self._template(active, gi)
                bind = {theta[i]: float(centre[i]) for i in range(len(theta))}
                bind[Rp] = float(R)
                counts = self.backend.run(
                    t.assign_parameters(bind, inplace=False),
                    shots=self.shots).result().get_counts()
                grp = self.groups[gi]
                labels = grp.paulis.to_labels()
                coeffs = np.real(grp.coeffs)
                supp = []
                for lab in labels:
                    s = 0
                    for q in range(len(lab)):
                        if lab[len(lab) - 1 - q] != 'I':
                            s |= 1 << q
                    supp.append(s)
                for key, v in counts.items():
                    parts = key.split()
                    if len(parts) != 2:
                        continue
                    d = int(parts[0], 2)             # register, created last
                    w = int(parts[1], 2)
                    ev = sum(co * (1.0 - 2.0 * (bin(w & s).count('1') & 1))
                             for co, s in zip(coeffs, supp))
                    acc[d] += ev * v
                    cnt[d] += v
            # per-group energies are summed, not averaged: E = sum_g E_g
            cnt = np.where(cnt > 0, cnt / max(len(self.groups), 1), 0.0)

        Ed = np.divide(acc, cnt, out=np.zeros(Nrow), where=cnt > 0)
        return self._decode(Ed, cnt, c, e, k, active, len(centre), R)

    def _decode(self, Ed, cnt, c, e, k, active, ntot, R):
        """Weighted least squares on the full quadratic design basis.

        THE OLD DECODE WAS THE FIXED CONTRAST, projecting Ed onto each design
        column and dividing by the column norm as if the observed rows were
        exactly orthogonal. Over a SAMPLED set of rows they are not: the
        empirical Gram is N I + O(sqrt(N)), and that fluctuation puts
        (||g||^2 - g_j^2)/N into the variance - numerically identical to SPSA's
        cross term, and it is what made v147 withdraw the sqrt(M) claim.

        v148 derived the Cramer-Rao floor for the scalar-oracle model,
        sigma^2/(N R^2), with NO M in it, and showed OLS on the SAME shots
        attains it while the contrast misses it by a factor growing as sqrt(M)
        (2.3x at M=16, 82x at M=65536). The fix is post-processing only: same
        circuits, same shots, empirical Gram instead of an assumed one.

        Regressing on the FULL basis - main effects, pure quadratics, and
        cross terms together - also removes the gradient/curvature aliasing
        that a sampled design otherwise carries, which the separate
        projections could not see.

        Rows are weighted by their SHOT COUNTS. The old code weighted observed
        rows equally regardless of how many shots each received, discarding
        that information too.
        """
        Nrow = 1 << k
        S = np.array([sigma_of(d, c, e) for d in range(Nrow)])   # Nrow x m
        m = len(active)
        obs = cnt > 0
        w = np.where(obs, cnt, 0.0).astype(float)

        s2 = float((w @ (S[:, 0] ** 2)) / max(w.sum(), 1e-300))
        cols = [S[:, a] for a in range(m)]                        # main
        cols += [S[:, a] ** 2 - s2 for a in range(m)]             # pure quad
        pair = [(a, b) for a in range(m) for b in range(a + 1, m)]
        cols += [S[:, a] * S[:, b] for a, b in pair]              # cross
        X = np.column_stack(cols)

        if self.decode_mode == "contrast":
            # the original: project onto each column, divide by its norm, i.e.
            # ASSUME the observed Gram is diagonal. Kept because v148's gain is
            # conditional - it appears only when the register is large relative
            # to the shot budget, and in the well-sampled regime (256 rows, 500+
            # shots per row) the two decodes agree to within shot noise.
            u = np.where(obs, 1.0, 0.0)
            u = u / max(u.sum(), 1e-300)
            beta = np.array([float(u @ (Ed * X[:, i]))
                             / max(float(u @ (X[:, i] ** 2)), 1e-300)
                             for i in range(X.shape[1])])
        else:
            sw = np.sqrt(w)
            Xw, yw = X * sw[:, None], Ed * sw
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

        sR, cR = np.sin(R), np.cos(R)
        g = np.zeros(ntot)
        H = np.zeros((ntot, ntot))
        for a in range(m):
            g[active[a]] = beta[a] / sR
            H[active[a], active[a]] = beta[m + a] / (1.0 - cR)
        for i, (a, b) in enumerate(pair):
            j, l = active[a], active[b]
            H[j, l] = H[l, j] = beta[2 * m + i] / (sR * sR)
        e0 = float((w @ Ed) / max(w.sum(), 1e-300))
        return g, H, e0

    # -- step ------------------------------------------------------------
    @staticmethod
    def suggest_h(H, R, active, kappa=1.0):
        """h <~ kappa R^2 sqrt(lambda_max). The ground state of -h^2 d2 +
        (lambda/2) x^2 has width ~ (h^2/lambda)^{1/4}; keeping it inside the
        trust region gives this. lambda_max is an AGGREGATE over ~m^2/2 Hessian
        entries, so it survives the per-entry noise that the Newton solve does
        not."""
        idx = np.asarray(active)
        Hi = H[np.ix_(idx, idx)]
        if not np.any(Hi):
            return None
        lam = float(np.max(np.abs(np.linalg.eigvalsh(Hi))))
        return float(kappa * R * R * np.sqrt(max(lam, 1e-12)))

    @staticmethod
    def grad_step(centre, g, H, R, active, newton=True):
        idx = np.asarray(active)
        gi = g[idx]
        if newton and H is not None and np.any(H[np.ix_(idx, idx)]):
            Hi = H[np.ix_(idx, idx)]
            lam = np.linalg.eigvalsh(Hi)
            mu = max(0.0, -lam.min()) + 1e-6 + np.linalg.norm(gi) / max(R, 1e-9)
            d = -np.linalg.solve(Hi + mu * np.eye(len(idx)), gi)
        else:
            d = -gi
        # BOX, not ball. The walk register spans [-R, +R] per COORDINATE, so a
        # 2-norm cap would hand the walk a step sqrt(len(active)) times larger
        # and any comparison would measure step size, not method. Clip to the
        # same infinity-norm the walk is confined to.
        d = np.clip(d, -R, R)
        out = np.array(centre, float)
        out[idx] += d
        return out



# -- the walk step -------------------------------------------------------

def _cycle_mixer_qc(kappa, h, dt):
    """e^{-i h^2 (D - A) dt} on one cycle of 2^kappa sites, as F^dag . diag . F.

    SIGN. The Schroedinger kinetic term is -h^2 Laplacian = +h^2 (D - A), so the
    cycle's spectrum enters as lambda_k = 2 - 2 cos(2 pi k / 2^kappa) >= 0 with
    its minimum at k = 0. v136's first two rounds used e^{-i A dt} - the wrong
    sign - which inverts the dispersion and makes the well's ground state the
    HIGHEST momentum state. Verified there against expm at 1.4e-14.
    """
    N = 1 << kappa
    lam = 2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(N) / N)
    qc = QuantumCircuit(kappa)
    qc.append(QFTGate(kappa).inverse(), range(kappa))
    qc.append(DiagonalGate(list(np.exp(-1j * h * h * lam * dt))), range(kappa))
    qc.append(QFTGate(kappa), range(kappa))
    return qc


def _quadratic_potential_qc(d, kappa, gsub, Hsub, R, dt):
    """e^{-i V dt} for V the measured local quadratic model, as RZ + RZZ.

    Coordinate i of the walk register is an integer x_i in [0, 2^kappa), and the
    displacement along parameter i is affine in it. Writing x_{i,m} = (1-z)/2,

        t_i = -a sum_m w_m z_{i,m},   a = 2R/(2^kappa - 1),  w_m = 2^{m-1}

    with NO constant term - the box is centred by construction. Then

        V = sum_i g_i t_i + (1/2) sum_ij H_ij t_i t_j

    is a quadratic form in the z's, so it is exactly RZ + RZZ. O((d kappa)^2)
    two-qubit gates and NOT a DiagonalGate, which would cost O(2^{d kappa}).

    The H_ij cross terms with i != j are what make this NON-SEPARABLE. Part VI's
    theorem is that a degree-1 potential factorises whatever the mixer; this is
    degree 2 by construction, and it is the measured Hessian rather than an
    assumed one.
    """
    a = 2.0 * R / ((1 << kappa) - 1)
    w = np.array([2.0 ** (m - 1) for m in range(kappa)])
    qc = QuantumCircuit(d * kappa)
    idx = lambda i, m: i * kappa + m
    for i in range(d):
        for m in range(kappa):
            ang = 2.0 * (-a * gsub[i] * w[m]) * dt
            if abs(ang) > 1e-12:
                qc.rz(ang, idx(i, m))
    for i in range(d):
        for m in range(kappa):
            for j in range(d):
                for n in range(kappa):
                    if (i, m) >= (j, n):
                        continue
                    coef = a * a * Hsub[i, j] * w[m] * w[n]
                    if i == j:
                        coef *= 1.0          # same coordinate, both orders once
                    ang = 2.0 * coef * dt
                    if abs(ang) > 1e-12:
                        qc.rzz(ang, idx(i, m), idx(j, n))
    return qc


def walk_step(q, centre, R, active, g, H, kappa=3, d_walk=None, steps=12,
              t_total=None, kappa_h=1.0, shots=8192, seed=None):
    """One coherent walk step: the measured model becomes the potential.

    Returns (new_centre, info). The walk register is d_walk cycles of 2^kappa
    sites - a TORUS, not one long cycle - because v136 PART 7 measured that a
    2-site-per-direction register is a SPIN (DeltaE = e^{-n S~}, degrading
    exponentially in the parameter count) while a genuine 1-D lattice is a
    PARTICLE (DeltaE = e^{-S0/h}, h free and flat in M). kappa >= 3 puts each
    direction on the particle side.

    The subspace is chosen by |g|, NOT by diagonalising H - the Hessian
    eigenbasis would make the quadratic form diagonal and hence the walk
    separable again, which is precisely the failure Part VI proves.
    """
    idx = np.asarray(active)
    order = np.argsort(-np.abs(g[idx]))
    d = min(d_walk or len(idx), len(idx))
    pick = idx[order[:d]]
    gsub = g[pick]
    Hsub = H[np.ix_(pick, pick)]

    h = QLTOWalk.suggest_h(H, R, list(pick), kappa=kappa_h) or 1.0
    if t_total is None:
        # enough time for the mixer to move one site: h^2 * t ~ 1
        t_total = 1.0 / max(h * h, 1e-9)

    nq = d * kappa
    qc = QuantumCircuit(nq, nq)
    for i in range(d):                     # centre of each cycle = current theta
        qc.h(range(i * kappa, (i + 1) * kappa))
    dt = t_total / steps
    # ANNEALED, and the first version of this function was not. With a FIXED h
    # the state stays delocalised for the whole evolution, so the measurement is
    # a sample from a spread distribution rather than a minimum - and a sampled
    # step EQUILIBRATES instead of converging. Measured in qlto_prototype: the
    # MSE fell 1.334 -> 0.341 and then bounced back to 0.448.
    #
    # V3's _execute_walk always had the schedule - beta = (1-s) pi dt ramping
    # the mixer DOWN, gamma = s pi dt ramping the potential UP - and dropping it
    # was the bug. At s -> 1 the Hamiltonian is pure potential, whose ground
    # state is the vertex we want.
    for st in range(steps):
        sch = (st + 0.5) / steps
        h_s = h * (1.0 - sch)
        if h_s > 1e-9:
            mix = _cycle_mixer_qc(kappa, h_s, dt)
            for i in range(d):
                qc.compose(mix, qubits=range(i * kappa, (i + 1) * kappa),
                           inplace=True)
        qc.compose(_quadratic_potential_qc(d, kappa, gsub, Hsub, R,
                                           dt * sch), inplace=True)
    qc.measure(range(nq), range(nq))

    be = AerSimulator(seed_simulator=seed)
    t = transpile(qc, be, basis_gates=['rz', 'sx', 'x', 'cx'],
                  optimization_level=1)
    counts = be.run(t, shots=shots).result().get_counts()

    # BEST measured vertex by the MODEL, not the most likely bitstring. The
    # model is classical and already in hand, so scoring every distinct outcome
    # is free - the walk is a PROPOSAL distribution and there is no reason to
    # take its mode rather than its best proposal.
    a = 2.0 * R / ((1 << kappa) - 1)
    bestv, bestm = None, np.inf
    for key in counts:
        bits = key[::-1]
        dsp = np.array([a * int(bits[i * kappa:(i + 1) * kappa][::-1], 2) - R
                        for i in range(d)])
        mval = float(gsub @ dsp + 0.5 * dsp @ Hsub @ dsp)
        if mval < bestm:
            bestm, bestv = mval, dsp
    disp = bestv
    out = np.array(centre, float)
    out[pick] += disp
    return out, {'h': h, 't': t_total, 'd': d, 'kappa': kappa,
                 'depth': t.depth(), 'cx': t.count_ops().get('cx', 0),
                 'qubits': nq, 'n_distinct': len(counts), 'model': bestm,
                 'disp': disp}


# -- self check ----------------------------------------------------------

if __name__ == '__main__':
    from qiskit.circuit import ParameterVector

    def ry_rz_ansatz(n, reps=1):
        """ry/rz/cx only. efficient_su2().decompose() emits generic  gates,
        which have no controlled form in _CTRL."""
        p = ParameterVector('t', 2 * n * (reps + 1))
        qc = QuantumCircuit(n)
        i = 0
        for _ in range(reps):
            for q in range(n):
                qc.ry(p[i], q); i += 1
                qc.rz(p[i], q); i += 1
            for q in range(n - 1):
                qc.cx(q, q + 1)
        for q in range(n):
            qc.ry(p[i], q); i += 1
            qc.rz(p[i], q); i += 1
        return qc

    def heis(n):
        t = []
        for i in range(n - 1):
            for p in ('XX', 'YY', 'ZZ'):
                lab = ['I'] * n
                lab[i], lab[i + 1] = p[0], p[1]
                t.append((''.join(reversed(lab)), 1.0))
        return SparsePauliOp.from_list(t)

    def exact_gH(anz, Hm, th, sub):
        E = lambda x: float(np.real(
            Statevector(anz.assign_parameters(x)).expectation_value(Hm)))
        s = np.pi / 2
        M = len(th)
        g = np.zeros(M)
        Hs = np.zeros((M, M))
        for j in sub:
            p, q = th.copy(), th.copy()
            p[j] += s; q[j] -= s
            g[j] = 0.5 * (E(p) - E(q))
        for j in sub:
            for l in sub:
                a, b, c, d = th.copy(), th.copy(), th.copy(), th.copy()
                a[j] += s; a[l] += s
                b[j] += s; b[l] -= s
                c[j] -= s; c[l] += s
                d[j] -= s; d[l] -= s
                Hs[j, l] = 0.25 * (E(a) - E(b) - E(c) + E(d))
        return g, Hs

    print(__doc__.split('\n')[0])
    print("=" * 74)
    n_sys = 2
    anz = ry_rz_ansatz(n_sys, reps=1)
    Hm = heis(n_sys)
    M = anz.num_parameters
    rng = np.random.default_rng(5)
    th = rng.uniform(-np.pi, np.pi, M)
    sub = [0, 1, 2, 3]
    g_ex, H_ex = exact_gH(anz, Hm, th, sub)
    off = np.array([[a != b for b in sub] for a in sub])

    q = QLTOWalk(anz, Hm, shot_budget=1 << 17, sim_seed=7)
    k = q.register_width(len(sub))
    print("m=%d active, register k=%d qubits, %d rows, p0=1/2, alpha=(1+cosR)/2"
          % (len(sub), k, 1 << k))
    print("")
    print("PART 0  TIER B - rows read by Statevector, no shots.")
    print("        does the CIRCUIT-REALISABLE 3-level design decode?")
    print("  %6s %11s %11s %11s %11s"
          % ("R", "cos(g)", "rel g", "rel Hdiag", "rel Hoff"))
    for R in (0.5, 0.35, 0.25, 0.15):
        g, H, e0 = q.sense(th, R, sub, exact=True)
        gs, Hs = g[sub], H[np.ix_(sub, sub)]
        ge, He = g_ex[sub], H_ex[np.ix_(sub, sub)]
        print("  %6.2f %11.6f %11.4e %11.4e %11.4e"
              % (R, gs @ ge / (np.linalg.norm(gs) * np.linalg.norm(ge)),
                 np.linalg.norm(gs - ge) / np.linalg.norm(ge),
                 np.linalg.norm(np.diag(Hs - He)) /
                 np.linalg.norm(np.diag(He)),
                 np.linalg.norm((Hs - He)[off]) / np.linalg.norm(He[off])))

    print("")
    print("PART 1  TIER A - the circuit, AerSimulator, %d shots per group, "
          "G=%d" % (q.shots, len(q.groups)))
    print("  %6s %11s %11s %11s %11s"
          % ("R", "cos(g)", "rel g", "rel Hdiag", "rel Hoff"))
    for R in (0.5, 0.35, 0.25):
        g, H, e0 = q.sense(th, R, sub)
        gs, Hs = g[sub], H[np.ix_(sub, sub)]
        ge, He = g_ex[sub], H_ex[np.ix_(sub, sub)]
        print("  %6.2f %11.6f %11.4e %11.4e %11.4e"
              % (R, gs @ ge / (np.linalg.norm(gs) * np.linalg.norm(ge)),
                 np.linalg.norm(gs - ge) / np.linalg.norm(ge),
                 np.linalg.norm(np.diag(Hs - He)) /
                 np.linalg.norm(np.diag(He)),
                 np.linalg.norm((Hs - He)[off]) / np.linalg.norm(He[off])))
        print("         suggest_h(R) = %s"
              % QLTOWalk.suggest_h(H, R, sub))

    t, _, _, _, _, _ = q._template(sub, 0)
    o = t.count_ops()
    print("")
    print("  circuit: %d qubits, depth %d, 2q %d, total %d"
          % (t.num_qubits, t.depth(), o.get('cx', 0), sum(o.values())))


# =====================================================================
# SECOND-ORDER PRIMITIVES FOR A COMPOSITE (QML) LOSS
#
# The design register's weight-2 Walsh coefficient gives the Hessian of an
# EXPECTATION VALUE. For a per-sample loss L = sum_x l(f_x, y_x) that is not
# enough: the curvature splits as
#
#   d2L/dti dtj = sum_x [ df_x/dti df_x/dtj + (f_x-y_x) d2f_x/dti dtj ]
#                        \____ GAUSS-NEWTON ___/  \______ residual ______/
#
# and a weighted data register returns the derivative of a weighted MEAN,
# which is the residual term only. Gauss-Newton is a MEAN OF PRODUCTS and no
# reweighting produces one - near a good fit J^T J dominates, so the weighted
# register's Hessian is uncorrelated with the true one (measured: rel 0.89-1.34).
#
# Both primitives below are TWO COPIES of the system, each with its own design
# register. What distinguishes them is whether the DATA register is shared:
#
#   shared data register     -> mean of products  -> J^T W J   (Gauss-Newton)
#   independent + SWAP test  -> fidelity curvature -> F        (QFIM)
#
# Verified exactly (Statevector): Gauss-Newton to 1e-11 against J^T W J with a
# control that collapses to outer(gbar) when the sharing is broken; QFIM to
# O(R^2) against the exact Fubini-Study metric with a control that corrupts the
# off-diagonal when the design is made degenerate.
#
# Cost: 2n + 2k + log|D| qubits, ONE circuit family. The standard cost is
# O(d^2) circuits for the QFIM (Gacon, Zoufal, Carleo & Woerner, arXiv
# 2103.09232) and no O(1) construction for Gauss-Newton.
# =====================================================================

def two_copy_design_state(anz, th, active, c, e, k, R, data_prep=None,
                          share_data=True):
    """|Psi> = sum_x sqrt(w_x)|x> (x) |d1>|psi(th+R sig(d1),x)> (x) |d2>|...>.

    data_prep(qc, dq, sq) encodes the data register onto a system copy; pass
    None for a data-free model. share_data=True gives the Gauss-Newton object,
    False gives two independent copies (the QFIM object, which additionally
    needs a SWAP test - see qfim_observable).
    """
    n = anz.num_qubits
    nd = 0 if data_prep is None else data_prep.num_data
    dq = QuantumRegister(nd, 'x') if nd else None
    dq2 = dq if (share_data or not nd) else QuantumRegister(nd, 'x2')
    sA, rA, cA = (QuantumRegister(n, 'sA'), QuantumRegister(k, 'rA'),
                  QuantumRegister(2, 'cA'))
    sB, rB, cB = (QuantumRegister(n, 'sB'), QuantumRegister(k, 'rB'),
                  QuantumRegister(2, 'cB'))
    regs = ([dq] if dq else []) + ([dq2] if (dq2 is not None and dq2 is not dq)
                                   else []) + [sA, rA, cA, sB, rB, cB]
    qc = QuantumCircuit(*regs)
    for dreg, sreg, rreg, creg in ((dq, sA, rA, cA), (dq2, sB, rB, cB)):
        qc.h(rreg)
        if data_prep is not None:
            data_prep(qc, dreg, sreg)
        _design_ansatz(qc, anz, th, active, c, e, k, R, sreg, rreg, creg)
    return qc, (sA, rA, sB, rB)


def _design_ansatz(qc, anz, th, active, c, e, k, R, sq, reg, scr):
    """The ansatz with theta_j -> theta_j + R sigma_j(d), two-parity form."""
    pidx = {p: i for i, p in enumerate(anz.parameters)}
    pos = {p: i for i, p in enumerate(active)}
    prev_c = prev_e = 0
    for inst in anz.data:
        op = inst.operation
        qs = [sq[anz.find_bit(b).index] for b in inst.qubits]
        prm = [p for p in op.params
               if isinstance(p, ParameterExpression) and p.parameters]
        if not prm:
            qc.append(op, qs)
            continue
        j = pidx[next(iter(prm[0].parameters))]
        if j not in pos:
            qc.append(op.__class__(float(th[j])), qs)
            continue
        a = pos[j]
        for b_ in range(k):
            if (c[a] ^ prev_c) >> b_ & 1:
                qc.cx(reg[b_], scr[0])
            if (e[a] ^ prev_e) >> b_ & 1:
                qc.cx(reg[b_], scr[1])
        prev_c, prev_e = c[a], e[a]
        qc.append(op.__class__(float(th[j]) + R), qs)
        getattr(qc, _CTRL[op.name])(-R, scr[0], qs[0])
        getattr(qc, _CTRL[op.name])(-R, scr[1], qs[0])
    for b_ in range(k):
        if prev_c >> b_ & 1:
            qc.cx(reg[b_], scr[0])
        if prev_e >> b_ & 1:
            qc.cx(reg[b_], scr[1])


def gauss_newton_scale(R):
    """(J^T W J)_jl = <sigma_Aj sigma_Bl O_A O_B> / this."""
    return (0.5 * np.sin(R)) ** 2


def qfim_scale(R):
    """F_pq = <sigma_Ap sigma_Bq X_anc> / this.  8 / sin^2 R."""
    return (np.sin(R) ** 2) / 8.0


def add_swap_test(qc, sA, sB):
    """Append a SWAP test; returns the ancilla whose Z reads |<psi_A|psi_B>|^2."""
    anc = QuantumRegister(1, 'anc')
    qc.add_register(anc)
    qc.h(anc[0])
    for i in range(len(sA)):
        qc.cswap(anc[0], sA[i], sB[i])
    qc.h(anc[0])
    return anc
