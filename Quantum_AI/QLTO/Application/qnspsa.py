"""QN-SPSA: Gacon, Zoufal, Carleo, Woerner, Quantum 5, 567 (2021), arXiv:2103.09232.

The one competing method that sits in V6's cost bracket AND carries the metric
preconditioning that beats V6 on MaxCut N=6. QNG in the suite reaches 0.0047
there against V6's 0.0904, but pays 1040 circuits against 20. QN-SPSA asks
whether that quality is available at constant cost - which is exactly the claim
that would take V6's niche.

THE ALGORITHM, following the paper's numbering.

  Eq. (10)  theta <- theta - eta g^-1 grad f            natural-gradient update
  Eq. (11)  g is the Fubini-Study metric
  Eq. (12)  the QFIM point-estimate, 2-SPSA applied to the FIDELITY rather than
            to the loss. With d1, d2 drawn from {-1,+1}^d and perturbation eps,

              dF = F(t, t+e d1+e d2) - F(t, t+e d1)
                 - F(t, t-e d1+e d2) + F(t, t-e d1)

              g_hat = -1/2 * dF/(2 eps^2) * (d1 d2^T + d2 d1^T)/2

            The leading -1/2 is because the Fubini-Study metric is minus one
            half the Hessian of the fidelity at coincident arguments; without it
            the estimate has the wrong sign and the update ASCENDS.
  Eq. (6)   exponential smoothing  g_bar_k = k/(k+1) g_bar_{k-1} + 1/(k+1) g_hat
  Eq. (7)   sqrt(g_bar g_bar) + beta I, whose eigenvalues are |lambda| + beta,
            imposing positive-definiteness and invertibility at once
  Sec 2     blocking: accept the step only if the loss does not increase beyond
            a tolerance, else resample. The paper suggests a tolerance of twice
            the loss standard deviation when the loss is sampled.

  The gradient itself is first-order SPSA: 2 loss evaluations.

COST, AND WHY IT IS NOT SIMPLY 6 CIRCUITS. The paper counts 6 FUNCTION
evaluations - 2 for the gradient, 4 for the metric. On a Pauli-sum Hamiltonian a
LOSS evaluation is not one circuit: it needs one per qubit-wise-commuting group,
so 2G. The four FIDELITY evaluations are different: the compute-uncompute
circuit U^dag(theta')U(theta)|0> is measured once in the computational basis and
the observable is the projector onto |0..0>, so each costs exactly ONE circuit
whatever H looks like. Hence

    NEFV per step = 2G + 4

which is 10 at Heisenberg (G=3) against V6's 3, and 6 at MaxCut (G=1) against
V6's 1. Billing it as 6G would overcharge it on every Heisenberg row and billing
it as 6 would undercharge it - the harness multiplies estimator-driven rows by
G, so this class counts its own circuits and sets counts_groups_internally.

The compute-uncompute overlap is the paper's own second option (Sec. 3.1): width
n rather than the swap test's 2n, at twice the depth. That trade is the right one
here because the suite bills circuits, not depth.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class QNSPSA:
    """Quantum Natural SPSA. NEFV: 2G + 4 per step, independent of M."""

    counts_groups_internally = True

    def __init__(self, ansatz, hamiltonian, lr=0.01, perturbation=0.1,
                 regularization=1e-3, blocking=True, shots=8192,
                 estimator=None, n_groups=1, seed=None):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.lr = lr
        self.eps = perturbation
        self.beta = regularization
        self.blocking = blocking
        self.shots = shots
        self.estimator = estimator
        self.G = max(1, int(n_groups))
        self.rng = np.random.default_rng(seed)

        self.d = ansatz.num_parameters
        self.g_bar = np.identity(self.d)
        self.k = 0
        self.nefv = 0
        self.max_circuit_depth = 0
        self._last_loss = None

        self._backend = AerSimulator()
        # compute-uncompute template, bound per call
        self._ov = None

    # ---- loss -------------------------------------------------------------
    def _loss(self, theta):
        job = self.estimator.run([(self.ansatz.assign_parameters(theta),
                                   self.hamiltonian)])
        return float(job.result()[0].data.evs)

    # ---- fidelity ---------------------------------------------------------
    def _fidelity(self, theta_a, theta_b):
        """|<psi(a)|psi(b)>|^2 by compute-uncompute, one computational-basis run."""
        qc = QuantumCircuit(self.ansatz.num_qubits)
        qc.compose(self.ansatz.assign_parameters(theta_a), inplace=True)
        qc.compose(self.ansatz.assign_parameters(theta_b).inverse(), inplace=True)
        qc.measure_all()
        tqc = transpile(qc, self._backend, optimization_level=1)
        self.max_circuit_depth = max(self.max_circuit_depth, tqc.depth())
        counts = self._backend.run(tqc, shots=self.shots).result().get_counts()
        zero = '0' * self.ansatz.num_qubits
        return counts.get(zero, 0) / max(sum(counts.values()), 1)

    # ---- one iteration ----------------------------------------------------
    def step(self, params):
        theta = np.asarray(params, dtype=float)
        self.k += 1
        e = self.eps

        d1 = self.rng.choice([-1.0, 1.0], size=self.d)
        d2 = self.rng.choice([-1.0, 1.0], size=self.d)

        # --- gradient: first-order SPSA, 2 loss evaluations (2G circuits)
        lp = self._loss(theta + e * d1)
        lm = self._loss(theta - e * d1)
        grad = (lp - lm) / (2.0 * e) * d1
        self.nefv += 2 * self.G

        # --- metric: 2-SPSA on the fidelity, 4 evaluations (4 circuits)
        f_pp = self._fidelity(theta, theta + e * d1 + e * d2)
        f_p = self._fidelity(theta, theta + e * d1)
        f_mp = self._fidelity(theta, theta - e * d1 + e * d2)
        f_m = self._fidelity(theta, theta - e * d1)
        self.nefv += 4

        dF = f_pp - f_p - f_mp + f_m
        rank_one = np.outer(d1, d2)
        g_hat = -0.5 * (dF / (2.0 * e * e)) * (rank_one + rank_one.T) / 2.0

        # --- Eq. (6) exponential smoothing
        self.g_bar = (self.k / (self.k + 1.0)) * self.g_bar \
            + (1.0 / (self.k + 1.0)) * g_hat

        # --- Eq. (7) sqrt(g g) + beta I, via eigen-decomposition
        w, V = np.linalg.eigh(self.g_bar)
        g_psd = V @ np.diag(np.abs(w) + self.beta) @ V.T

        try:
            nat = np.linalg.solve(g_psd, grad)
        except np.linalg.LinAlgError:
            nat = grad

        candidate = theta - self.lr * nat

        # --- blocking (Sec. 2): reject an uphill step rather than take it
        if self.blocking:
            if self._last_loss is None:
                self._last_loss = self._loss(theta)
                self.nefv += self.G
            new_loss = self._loss(candidate)
            self.nefv += self.G
            tol = 2.0 * abs(self._last_loss) / np.sqrt(max(self.shots, 1))
            if new_loss > self._last_loss + tol:
                return theta                     # reject, keep current point
            self._last_loss = new_loss
        return candidate
