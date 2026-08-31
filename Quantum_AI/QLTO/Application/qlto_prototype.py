"""QLTO prototype - data register + 3-level sensing + walk step, end to end.

Standalone: numpy and qiskit only. Composes what the pieces already do:

    qlto_qml.py   weighted data register: THREE circuits per epoch, flat in |D|
                  and in M. prep_weights exact to 1.11e-16.
    qlto_walk.py  3-level design register: gradient AND Hessian from one shot
                  record. cos 0.9998, rel Hdiag 0.095 at tier A.
    v136          cycle mixer, verified against expm at 1.4e-14.

THE POINT OF RUNNING IT SMALL IS TO FIND WHAT BREAKS. Everything here is
instrumented against an exact classical reference so failures are visible rather
than absorbed. Three are already known and are printed every epoch:

  THE WEIGHTED REGISTER GIVES THE RESIDUAL HESSIAN, NOT THE MSE HESSIAN. For
  L = (1/S) sum_x (f_x - y_x)^2,

      d2L/dti dtj = (2/S) sum_x [ df_x/dti df_x/dtj  +  (f_x - y_x) d2f_x/dti dtj ]
                                 \\_____ Gauss-Newton _____/   \\____ residual ____/

  The weighted register returns sum_x w_x (derivative of f_x), so it gives the
  RESIDUAL term and misses Gauss-Newton entirely - and near a good fit
  Gauss-Newton is the dominant one. J^T J would need per-sample Jacobians, which
  is exactly the |D|-dependence the data register exists to avoid. Printed as
  `H res vs true` so the gap is on the record, not in a footnote.

  THE HESSIAN IS SHOT-LIMITED WHERE THE GRADIENT IS NOT. Its signal is sin^2 R
  against the gradient's sin R, so per-entry SNR is R times worse and it gets
  WORSE as R shrinks. qlto_walk measured rel Hdiag going 0.095 -> 0.125 as R fell
  0.5 -> 0.25 while the gradient improved. There is an optimal R for curvature
  and it is LARGER than the gradient's.

  THE WALK STEP IS UNPROVEN AT THIS SIZE. v139 measured it losing to brute force
  5 times in 6 over 4096 vertices, and v141's corridor test was blocked (wells
  detuned by discretisation, and no measure concentration at d=2). It is run
  here anyway, beside the Newton step, because small scale is where its bugs are
  cheap to find.

WHAT IS BEING CLAIMED, and it is only the sensing. Three circuits per epoch,
flat in |D| and M, returning gradient and curvature. The walk is logged as the
scaling path - v136 measured the cycle register under DeltaE = e^{-S0/h} with h
free and flat in M, against the hypercube's e^{-n S~} which degrades
exponentially in the parameter count - and that is a claim about big machines,
not about this one.
"""
import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from qlto_walk import (QLTOWalk, level_cols, sigma_of, walk_step, _CTRL)


def prep_weights(qc, p, dq):
    """|0> -> sum_x sqrt(p_x)|x> by Moettoenen uniformly-controlled RY.

    ENDIANNESS: the level-lvl rotation targets dq[d-1-lvl], not dq[lvl]. Qiskit
    is little-endian and treating qubit 0 as the MSB was the fourth endianness
    bug in this project (errors 0.30/0.55/0.11 at d=2,3,4). Verified exact to
    1.11e-16 in qlto_qml.py PART 0.
    """
    d = len(dq)
    p = np.asarray(p, float)
    p = p / max(p.sum(), 1e-300)
    for lvl in range(d):
        blk = 1 << (d - lvl - 1)
        thetas = []
        for j in range(1 << lvl):
            lo = j * (blk << 1)
            tot = p[lo:lo + (blk << 1)].sum()
            hi = p[lo + blk:lo + (blk << 1)].sum()
            r = hi / tot if tot > 1e-300 else 0.0
            thetas.append(2.0 * np.arcsin(np.sqrt(np.clip(r, 0.0, 1.0))))
        target = dq[d - 1 - lvl]
        if lvl == 0:
            qc.ry(float(thetas[0]), target)
        else:
            _mux_ry(qc, thetas, [dq[d - lvl + i] for i in range(lvl)], target)


def _mux_ry(qc, thetas, controls, target):
    k = len(controls)
    n = 1 << k
    A = np.empty((n, n))
    for i in range(n):
        gcode = i ^ (i >> 1)
        for j in range(n):
            A[i, j] = (-1.0) ** (bin(j & gcode).count('1'))
    alpha = (A @ np.asarray(thetas, float)) / n
    for i in range(n):
        qc.ry(float(alpha[i]), target)
        c = min(((i + 1) & -(i + 1)).bit_length() - 1, k - 1)
        qc.cx(controls[c], target)


def _cry(qc, a, c, t):
    """CX RY(-a/2) CX RY(a/2): identity on control 0, RY(a) on control 1."""
    qc.ry(a / 2.0, t)
    qc.cx(c, t)
    qc.ry(-a / 2.0, t)
    qc.cx(c, t)


class QLTOPrototype:
    def __init__(self, n_sys, n_data, alpha, core, obs, shots=1 << 15,
                 seed=None):
        self.n_sys = n_sys
        self.n_data = n_data
        self.D = 1 << n_data
        self.alpha = np.asarray(alpha, float)      # (n_sys, n_data), FIXED
        self.core = core
        self.obs = obs
        self.shots = int(shots)
        self.be = AerSimulator(seed_simulator=seed)
        self.M = core.num_parameters
        self._pidx = {p: i for i, p in enumerate(core.parameters)}

    # -- shared front end -------------------------------------------------
    def _front(self, qc, dq, sq, p=None):
        if p is None:
            qc.h(dq)                          # uniform: circuit 1
        else:
            prep_weights(qc, p, dq)           # weighted: circuits 2, 3
        for j in range(self.n_sys):
            for d in range(self.n_data):
                _cry(qc, float(self.alpha[j, d]), dq[d], sq[j])

    # -- circuit 1: every f_x at once -------------------------------------
    def residuals(self, theta):
        dq = QuantumRegister(self.n_data, 'd')
        sq = QuantumRegister(self.n_sys, 's')
        qc = QuantumCircuit(dq, sq, ClassicalRegister(self.n_sys, 'cs'),
                            ClassicalRegister(self.n_data, 'cd'))
        self._front(qc, dq, sq)
        qc.compose(self.core.assign_parameters(theta), qubits=list(sq),
                   inplace=True)
        qc.measure(sq, qc.cregs[0])
        qc.measure(dq, qc.cregs[1])
        t = transpile(qc, self.be, optimization_level=1)
        cnt = self.be.run(t, shots=self.shots).result().get_counts()
        num = np.zeros(self.D)
        den = np.zeros(self.D)
        for k, v in cnt.items():
            a, b = k.split()
            x = int(a, 2)                     # data reg, created last
            w = int(b, 2)
            num[x] += self._obs_val(w) * v
            den[x] += v
        return np.divide(num, den, out=np.zeros(self.D), where=den > 0)

    def _obs_val(self, w):
        return 1.0 - 2.0 * (w & 1)            # Z on system qubit 0

    # -- circuits 2 and 3: g and H on the weighted register ---------------
    def _rows(self, theta, R, active, wts, k, c, e):
        Nrow = 1 << k
        acc = np.zeros(Nrow)
        cnt = np.zeros(Nrow)
        for sgn in (+1.0, -1.0):
            m = (np.sign(wts) == sgn)
            if not m.any():
                continue
            p = np.where(m, np.abs(wts), 0.0)
            mass = p.sum()
            for d, (s, n) in self._design_run(theta, R, active, p,
                                              k, c, e).items():
                # s is ALREADY the mean over that row's shots, so it must be
                # re-weighted by n before being divided by the pooled count.
                # Adding the bare mean and dividing by sum(n) scaled Ed down by
                # the shots-per-row (~256 here), which is why |H measured| sat
                # at a noise floor two orders below |H true| - and why the
                # GRADIENT still looked correct: cos is scale-invariant, so a
                # uniform factor is invisible to it. Only the Hessian reports a
                # magnitude, and only the magnitude exposed this.
                acc[d] += sgn * mass * s * n
                cnt[d] += n
        return np.divide(acc, cnt, out=np.zeros(Nrow), where=cnt > 0), cnt

    def sense(self, theta, R, active, wts, R_hess=None):
        """TWO RADII, and this is the fix for the prototype's dead Hessian.

        The gradient's signal is sin R and its bias is O(R^2), so it wants R
        SMALL. The Hessian's signal is sin^2 R, so it wants R LARGE. A single
        radius starves one of them, and it starved the Hessian: |H measured|
        sat at a noise floor of ~0.003 while |H true| fell 0.59 -> 0.04.

        And the two radii are not even the same KIND of parameter. R for the
        gradient is a TRUST REGION - how far the linear model is trusted, so it
        must shrink as the optimiser converges. R for the Hessian is a
        MEASUREMENT setting - curvature is a property of the point, not of the
        step, so it should sit wherever the SNR is best and never shrink at all.
        Tying them together was a category error.

        Costs one extra design pass: 2G circuits instead of G, 5 per epoch
        instead of 3.
        """
        m = len(active)
        k = self._k(m)
        c, e = level_cols(m, k)
        Ed_g, cnt_g = self._rows(theta, R, active, wts, k, c, e)
        g, _ = self._decode(Ed_g, cnt_g, c, e, k, active, R)
        if R_hess is None or abs(R_hess - R) < 1e-12:
            _, H = self._decode(Ed_g, cnt_g, c, e, k, active, R)
        else:
            Ed_h, cnt_h = self._rows(theta, R_hess, active, wts, k, c, e)
            _, H = self._decode(Ed_h, cnt_h, c, e, k, active, R_hess)
        return g, H

    def _k(self, m):
        kk = max(5, int(np.ceil(np.log2(2 * m * m + 1))) + 2)
        while True:
            try:
                level_cols(m, kk)
                return kk
            except ValueError:
                kk += 1

    def _design_run(self, theta, R, active, p, k, c, e):
        dq = QuantumRegister(self.n_data, 'd')
        sq = QuantumRegister(self.n_sys, 's')
        rg = QuantumRegister(k, 'reg')
        sc = QuantumRegister(2, 'scr')
        qc = QuantumCircuit(dq, sq, rg, sc,
                            ClassicalRegister(self.n_sys, 'cs'),
                            ClassicalRegister(k, 'cr'))
        self._front(qc, dq, sq, p)
        qc.h(rg)
        pos = {q: i for i, q in enumerate(active)}
        pc = pe = 0
        for inst in self.core.data:
            op = inst.operation
            qs = [sq[self.core.find_bit(b).index] for b in inst.qubits]
            prm = [x for x in op.params
                   if isinstance(x, ParameterExpression) and x.parameters]
            if not prm:
                qc.append(op, qs)
                continue
            j = self._pidx[next(iter(prm[0].parameters))]
            if j not in pos:
                qc.append(op.__class__(float(theta[j])), qs)
                continue
            a = pos[j]
            for b_ in range(k):
                if (c[a] ^ pc) >> b_ & 1:
                    qc.cx(rg[b_], sc[0])
                if (e[a] ^ pe) >> b_ & 1:
                    qc.cx(rg[b_], sc[1])
            pc, pe = c[a], e[a]
            qc.append(op.__class__(float(theta[j]) + R), qs)
            getattr(qc, _CTRL[op.name])(-R, sc[0], qs[0])
            getattr(qc, _CTRL[op.name])(-R, sc[1], qs[0])
        for b_ in range(k):
            if pc >> b_ & 1:
                qc.cx(rg[b_], sc[0])
            if pe >> b_ & 1:
                qc.cx(rg[b_], sc[1])
        qc.measure(sq, qc.cregs[0])
        qc.measure(rg, qc.cregs[1])
        t = transpile(qc, self.be, optimization_level=1)
        cnt = self.be.run(t, shots=self.shots).result().get_counts()
        out = {}
        for key, v in cnt.items():
            a, b = key.split()
            d = int(a, 2)
            w = int(b, 2)
            s, n = out.get(d, (0.0, 0.0))
            out[d] = (s + self._obs_val(w) * v, n + v)
        return {d: (s / n, n) for d, (s, n) in out.items() if n > 0}

    def _decode(self, Ed, cnt, c, e, k, active, R):
        Nrow = 1 << k
        S = np.array([sigma_of(d, c, e) for d in range(Nrow)])
        w = np.where(cnt > 0, 1.0, 0.0)
        w = w / max(w.sum(), 1e-300)
        s2 = float(w @ (S[:, 0] ** 2))
        P2 = S ** 2 - s2
        nP1 = s2
        nP2 = float(w @ (P2[:, 0] ** 2))
        m = len(active)
        g = np.zeros(self.M)
        H = np.zeros((self.M, self.M))
        sR, cR = np.sin(R), np.cos(R)
        for a in range(m):
            j = active[a]
            g[j] = float(w @ (Ed * S[:, a])) / nP1 / sR
            if nP2 > 1e-14:
                H[j, j] = float(w @ (Ed * P2[:, a])) / nP2 / (1.0 - cR)
            for b in range(a + 1, m):
                l = active[b]
                v = float(w @ (Ed * S[:, a] * S[:, b])) / (nP1 * nP1)
                H[j, l] = H[l, j] = v / (sR * sR)
        return g, H

    # -- one epoch --------------------------------------------------------
    def epoch(self, theta, R, active, y, use_walk=False, kappa=3, seed=None):
        """residuals -> weights -> sense -> step. THREE circuits for the
        sensing (1 for the weights, 2 for the sign branches), independent of
        |D| and of M."""
        f = self.residuals(theta)
        w = f - y
        # R_hess FIXED: curvature is a property of the point, not the step.
        # sin^2 R peaks at pi/2; 0.9 trades a little signal for less bias.
        g, H = self.sense(theta, R, active, w, R_hess=0.9)
        g *= 2.0 / self.D                    # d/dtheta of (1/D) sum (f-y)^2
        H *= 2.0 / self.D
        h = QLTOWalk.suggest_h(H, R, active)
        if use_walk:
            new, info = walk_step(self, theta, R, active, g, H, kappa=kappa,
                                  d_walk=len(active), steps=8, shots=4096,
                                  seed=seed)
        else:
            new = QLTOWalk.grad_step(theta, g, H, R, active, newton=True)
            info = {}
        out = dict(g=g, H=H, h_sugg=h, f=f, mse=float(np.mean(w ** 2)))
        out.update(info)                     # walk_step also reports an h
        return new, out


# -- exact reference -----------------------------------------------------

def exact_f(proto, theta):
    """f_x for every sample, by Statevector - the reference the circuit is
    checked against, not a substitute for it."""
    out = np.zeros(proto.D)
    for x in range(proto.D):
        qc = QuantumCircuit(proto.n_sys)
        for j in range(proto.n_sys):
            ang = sum(proto.alpha[j, d] for d in range(proto.n_data)
                      if (x >> d) & 1)
            qc.ry(ang, j)
        qc.compose(proto.core.assign_parameters(theta), inplace=True)
        out[x] = float(np.real(Statevector(qc).expectation_value(proto.obs)))
    return out


def exact_gH(proto, theta, y, active):
    """Exact MSE gradient and FULL Hessian by parameter shift, including the
    Gauss-Newton term the weighted register cannot see."""
    s = np.pi / 2
    M = proto.M
    D = proto.D

    def F(t):
        return exact_f(proto, t)
    g = np.zeros(M)
    J = np.zeros((D, M))
    for j in active:
        p, q = np.array(theta), np.array(theta)
        p[j] += s; q[j] -= s
        J[:, j] = 0.5 * (F(p) - F(q))
    f = F(theta)
    r = f - y
    g = (2.0 / D) * (J.T @ r)
    H = np.zeros((M, M))
    for j in active:
        for l in active:
            a, b, c, d = (np.array(theta), np.array(theta),
                          np.array(theta), np.array(theta))
            a[j] += s; a[l] += s
            b[j] += s; b[l] -= s
            c[j] -= s; c[l] += s
            d[j] -= s; d[l] -= s
            d2f = 0.25 * (F(a) - F(b) - F(c) + F(d))
            H[j, l] = (2.0 / D) * (J[:, j] @ J[:, l] + r @ d2f)
    Hres = np.zeros((M, M))
    for j in active:
        for l in active:
            a, b, c, d = (np.array(theta), np.array(theta),
                          np.array(theta), np.array(theta))
            a[j] += s; a[l] += s
            b[j] += s; b[l] -= s
            c[j] -= s; c[l] += s
            d[j] -= s; d[l] -= s
            d2f = 0.25 * (F(a) - F(b) - F(c) + F(d))
            Hres[j, l] = (2.0 / D) * (r @ d2f)
    return g, H, Hres


if __name__ == '__main__':
    print(__doc__.split('\n')[0])
    print("=" * 74)
    n_sys, n_data = 2, 3
    rng = np.random.default_rng(4)
    p = ParameterVector('t', 2 * n_sys * 2)
    core = QuantumCircuit(n_sys)
    i = 0
    for q in range(n_sys):
        core.ry(p[i], q); i += 1
        core.rz(p[i], q); i += 1
    core.cx(0, 1)
    for q in range(n_sys):
        core.ry(p[i], q); i += 1
        core.rz(p[i], q); i += 1
    alpha = rng.uniform(-1.0, 1.0, (n_sys, n_data))
    obs = SparsePauliOp.from_list([('I' * (n_sys - 1) + 'Z', 1.0)])
    proto = QLTOPrototype(n_sys, n_data, alpha, core, obs,
                          shots=1 << 15, seed=3)

    th_star = rng.uniform(-np.pi, np.pi, proto.M)
    y = exact_f(proto, th_star)              # realizable labels
    active = list(range(proto.M))
    print("  |D| = %d samples, M = %d parameters, N_sys = %d"
          % (proto.D, proto.M, n_sys))
    print("  labels realizable (y = f(theta*)), MSE loss")
    print("")

    # SAME theta0 for both arms. The first run drew from an advanced RNG, so
    # Newton started at MSE 0.061 and the walk at 1.334 - not a comparison.
    th0 = rng.uniform(-np.pi, np.pi, proto.M)
    for use_walk in (False, True):
        th = np.array(th0)
        R = 0.5
        tag = 'WALK  ' if use_walk else 'NEWTON'
        print("  %s step" % tag)
        print("   %3s %9s %9s %9s %9s %9s %8s %8s"
              % ("ep", "MSE", "cos(g)", "H res", "H true", "h",
                 "|Hmeas|", "|Href|"))
        for ep in range(8):
            new, info = proto.epoch(th, R, active, y, use_walk=use_walk,
                                    seed=50 + ep)
            ge, He, Hres = exact_gH(proto, th, y, active)
            cg = float(info['g'] @ ge /
                       max(np.linalg.norm(info['g']) * np.linalg.norm(ge), 1e-12))
            rr = float(np.linalg.norm(info['H'] - Hres) /
                       max(np.linalg.norm(Hres), 1e-12))
            rt = float(np.linalg.norm(info['H'] - He) /
                       max(np.linalg.norm(He), 1e-12))
            print("   %3d %9.5f %9.5f %9.4f %9.4f %9.4f %8.4f %8.4f"
                  % (ep, info['mse'], cg, rr, rt,
                     info['h_sugg'] if info['h_sugg'] else float('nan'),
                     np.linalg.norm(info['H']), np.linalg.norm(Hres)))
            th = new
            R *= 0.85
        fin = exact_f(proto, th)
        print("   final exact MSE = %.5f" % float(np.mean((fin - y) ** 2)))
        print("")

    print("  H res  = |measured H - RESIDUAL Hessian| / |residual|")
    print("  H true = |measured H - FULL MSE Hessian|  / |full|")
    print("  The gap between them is the Gauss-Newton term J^T J, which the")
    print("  weighted register cannot see. If H res stays small while H true")
    print("  does not, the sensing is correct and the LOSS is the problem.")
