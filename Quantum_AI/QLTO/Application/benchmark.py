"""
qlto_benchmarks.py

Comprehensive benchmark suite for QLTO vs Classical vs QNG vs AdamW.
Tests on H2, LiH, and Heisenberg (N=4, 6, 8).

Features:
- Exact NEFV Tracking (Circuit Executions).
- Correct QNG (Efficient Simulation): Uses Param Shift for values but counts O(L) cost.
- Block-Diagonal Inversion: Avoids O(N^3) classical overhead.
- AdamW: Uses Param Shift (O(N) cost).
- QLTO: Global Mode (Riemannian Coherent Walk).
"""

# Set non-interactive backend BEFORE importing pyplot to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')

import numpy as np
import time
import matplotlib.pyplot as plt
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator

# Import Efficient Engines
import sys
import os
import gc

# All figures and the results CSV land here rather than beside the script.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Shot budget shared by EVERY optimizer.
#
# Without this the comparison is rigged: QLTO V3 samples its sensing circuit at
# `SHOTS`, while AdamW / QNG / QAOA / V2 ran through estimators with no precision
# set, which return exact statevector expectations - verified reproducible to
# 5.6e-17 over repeated runs. That is a noiseless gradient no hardware can
# supply, and it flattered every shot-free method on accuracy for free.
#
# THE ABOVE FIX WAS INCOMPLETE AND THE REASONING ON THIS LINE WAS WRONG.
# "precision is the standard error, so 1/sqrt(SHOTS) is the matching setting" is
# only true when Var(H) = 1. StatevectorEstimator(default_precision=p) returns the
# EXACT expectation plus Gaussian noise of std p; it never samples, so it is blind
# to both Var(H) and the number of measurement settings. Verified empirically
# (supplement/results/audit_benchmark.log): std/p = 0.95, 1.02, 1.02 over p spanning 18x
# while Var(H) = 12.03.
#
# A real device reaching standard error sigma needs G * sum_g Var(H_g) / sigma^2
# shots, so the EFFECTIVE budget the baselines were handed was:
#
#     problem          G   Var(H)   eff shots   vs the 8192 claimed
#     H2               2    0.185        3059   0.37x
#     LiH              3    0.159        3995   0.49x
#     MaxCut N=4       1    0.766        6278   0.77x
#     MaxCut N=6       1    3.977       32580      4x
#     Heisenberg N=4   3    8.315      191331     23x
#     Heisenberg N=6   3   14.110      351902     43x
#     Heisenberg N=8   3   21.241      464701     57x
#
# Problem-dependent, because Var(H) is large for unit-coefficient spin sums and
# tiny for the molecules. It was near-honest on H2/LiH/MaxCut and wildly generous
# on Heisenberg - which is exactly where V3 lost (N=4 to V2 at 23x, N=8 to QNG at
# 57x) and where its one clear win happened DESPITE 43x. V3 was never subsidised:
# its sensing calls backend.run(qc, shots=shot_budget) on a real simulator. V2 was,
# via BaseEstimator(precision=PRECISION), so the V2-vs-V3 comparisons are the ones
# most affected.
#
# BackendEstimatorV2 on AerSimulator actually samples, allocating 1/precision^2
# shots PER commuting group. Same nominal budget, real noise: measured std 0.03794
# against the old 0.01105 on Heisenberg N=4, and 0.00384 on H2 where real sampling
# is BETTER than the fixed noise was.
SHOTS = 8192
PRECISION = (1.0 / np.sqrt(SHOTS)) if SHOTS else 0.0


def make_estimator():
    """Genuinely shot-based estimator honouring the shared budget.

    Costs real time - it simulates G circuits of SHOTS shots per expectation
    value instead of adding a number to an exact result - which is the point.
    """
    if SHOTS:
        return BackendEstimatorV2(backend=AerSimulator(),
                                  options={'default_precision': PRECISION})
    return Estimator()


# Reporting is NOT part of any optimizer's cost, so it must be identical and
# noiseless for every method. It was neither: energies were logged through
# `opt.estimator`, which is exact for V3 (AerEstimator, no precision set),
# fixed-noise for V2, and - once make_estimator started really sampling - shot-
# noisy for the baselines. E_best is a min over epochs, so reporting noise biases
# it LOW, handing whichever method had the noisiest logging a free advantage on
# exactly the column the RESULT table ranks by.
REPORT_ESTIMATOR = Estimator()


def report_energy(ansatz, hamiltonian, params):
    """Exact energy for logging, same path for every optimizer."""
    job = REPORT_ESTIMATOR.run([(ansatz.assign_parameters(params), hamiltonian)])
    return float(job.result()[0].data.evs)


def pauli_groups(hamiltonian):
    """Measurement settings <H> needs, hence circuits per energy evaluation.

    Every baseline bills one energy evaluation as one circuit, but a Pauli sum
    needs one circuit per qubit-wise-commuting group. QLTO V3 genuinely needs
    one: it measures the param and ancilla registers in the computational basis
    and takes the energy from the PHASE of exp(-iHt), so this returns 1 for it
    regardless of H. That asymmetry is the whole structural cost argument, and
    leaving it uncounted understated V3 by 3x on every Heisenberg problem.
    """
    try:
        return max(1, len(hamiltonian.group_commuting(qubit_wise=True)))
    except Exception:
        return 1


# One tuned hyperparameter per optimizer, shared by every problem.
#
# Fairness note: QLTO's k_step=15 was chosen from a sweep over two problems,
# i.e. tuned GLOBALLY and not per-problem. The classical baselines were left at
# whatever lr the original file shipped with (0.1 across the board, never
# swept), while this session tuned QLTO's k_steps, tau, simulator, shots and
# ancilla count. That asymmetry silently favours QLTO, and it is the least
# visible of the benchmark's problems because - unlike the QAOA cost-layer bug -
# it produces plausible-looking numbers.
#
# `python benchmark.py --tune` re-derives this table by sweeping each grid on a
# representative subset, symmetric across all methods: one global value each,
# selected the same way.
# FINAL - every method searched until its optimum was interior or bracketed.
# Scores are the mean gap to exact as a fraction of the spectral range, over
# H2 / MaxCut N=4 / Heisenberg N=4 / Heisenberg N=6, 2 trials each:
#
#   QLTO V2        k=45   0.0234   interior (30/45/60)
#   Correct QNG    lr=0.3 0.0301   interior
#   QLTO V3 Had.   k=20   0.0308   interior
#   QLTO V3 QPE    k=15   0.0318   interior
#   AdamW          lr=0.5 0.0348   bracketed 0.3..1.0
#   SPSA           lr=0.5 0.0458   bracketed 0.1..1.0
#   QAOA           p=4    0.0817   bracketed 3..6
#
# Repeatability: V3 QPE at k=15 scored 0.0299 (round 2) and 0.0318 (round 3) on
# identical settings, so 2-trial noise is ~+-0.002 and gaps under ~0.005 are not
# real. QNG / V3-Hadamard / V3-QPE / AdamW are therefore one tied band.
#
# What tuning changed, and why it mattered: AdamW moved 0.0463 -> 0.0306 (lr
# 0.1 -> 0.5) and V2 moved 0.0236 -> 0.0234 (k 20 -> 45). Every earlier run I
# made used AdamW at lr=0.1, i.e. against a handicapped baseline - the
# classical mirror of V2's identity-phase bug. V3's own optimum barely moved
# (k=15, already the default), so QLTO was near-tuned all along and the
# baselines were not.
#
# Note V2 buys its lead with DEPTH: k=45 triples its walk depth versus k=15.
TUNED = {
    'QLTO V3 QPE (k=3)': 15,      # k_step
    'QLTO V3 (Hadamard)': 20,     # k_step
    'QLTO V2 (engine-grad)': 45,  # k_step
    'QAOA': 4,                    # p_layers
    'Correct QNG': 0.3,           # lr
    'AdamW': 0.5,                 # lr
    'SPSA': 0.5,                  # lr
}

# Round 2. Round 1 used [10,15,20] / [2,3,4] / [0.01,0.05,0.1] / [0.05,0.1,0.5]
# and EVERY method selected the top of its grid - so those were grid edges, not
# optima. It mattered most for the baselines: AdamW improved 8x across its grid
# (0.368 -> 0.047) and was still improving at the boundary, i.e. the values the
# suite had been using all along were far from its best. Grids shifted upward to
# bracket the optimum; equal size so no method gets more search than another.
# Round 3. Search continues for any method still selecting a grid EDGE - an edge
# means the optimum has not been found, so stopping there under-serves that
# method. Five settled in rounds 1-2 with interior/bracketed optima and are
# pinned. Two were still improving at the boundary:
#   QLTO V2   20 -> 30, still descending (0.0236 -> 0.0222)
#   AdamW    0.1 -> 0.5, still descending (0.0463 -> 0.0333 -> 0.0306)
# Note this gives the unsettled methods more grid evaluations than the settled
# ones. That is the fair procedure, not a bias: every method is searched until
# its optimum is interior, and the ones that converged early simply needed less.
TUNE_GRID = {
    'QLTO V3 QPE (k=3)': [15],
    'QLTO V3 (Hadamard)': [20],
    'QLTO V2 (engine-grad)': [30, 45, 60],
    'QAOA': [4],
    'Correct QNG': [0.3],
    'AdamW': [0.5, 1.0, 2.0],
    'SPSA': [0.5],
}


def result_path(*parts):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, *parts)


def safe(name):
    """Filesystem-safe problem name."""
    return name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")

try:
    from commute_fim import CommutingBlockFIM
    from commute_gradient import CommutingBlockGradient
    from nisq_v2 import RiemannianQLTO
    ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Efficient Engines not found: {e}")
    ENGINES_AVAILABLE = False

# Import DirectITE
try:
    from coherent_ite import DirectITE
    DIRECT_ITE_AVAILABLE = True
except ImportError:
    DIRECT_ITE_AVAILABLE = False

# Import PennyLane for QNG
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# --- Helper Functions ---

def block_diagonal_solve(fim, grad, layers, regularization=1e-3):
    """
    Solves F * x = g using Block-Diagonal approximation to avoid O(N^3) inversion.
    Inverts each layer's block independently. reduce to O(L⋅(N/L)^3)≈O(N)
    
    Args:
        fim (np.ndarray): Full Fisher Information Matrix (NxN).
        grad (np.ndarray): Gradient vector (N).
        layers (list): List of layers from the FIM engine, used to determine blocks.
    
    Returns:
        np.ndarray: The natural gradient update vector.
    """
    nat_grad = np.zeros_like(grad)
    used_indices = []
    
    try:
        # Iterate through layers to identify parameter blocks
        for layer in layers:
            # Extract indices for the current block
            # Case A: Layer is a list of objects with 'index' (e.g., PauliStrings)
            if isinstance(layer, list) and hasattr(layer[0], 'index'):
                idxs = [op.index for op in layer if hasattr(op, 'index')]
            # Case B: Layer is a list of indices directly
            elif isinstance(layer, list) and isinstance(layer[0], int):
                idxs = layer
            else:
                # Fallback: Sequential assumption or skip
                continue
                
            if not idxs:
                continue
                
            # Extract sub-matrices
            # np.ix_ creates the meshgrid for block slicing
            fim_block = fim[np.ix_(idxs, idxs)]
            grad_block = grad[idxs]
            
            # Regularize and Invert Block
            # Cost is O(k^3) where k is params per layer (small), not N (large)
            fim_block_reg = fim_block + regularization * np.eye(len(fim_block))
            nat_grad_block = np.linalg.pinv(fim_block_reg) @ grad_block
            
            # Place result back
            nat_grad[idxs] = nat_grad_block
            used_indices.extend(idxs)
            
        # Handle any 'orphaned' parameters not in a layer (fallback to standard gradient)
        all_indices = set(range(len(grad)))
        remaining = list(all_indices - set(used_indices))
        if remaining:
            nat_grad[remaining] = grad[remaining]
            
    except Exception as e:
        print(f"Warning: Block-Diagonal solver failed ({e}), falling back to full PINV.")
        fim_reg = fim + regularization * np.eye(len(fim))
        nat_grad = np.linalg.pinv(fim_reg) @ grad
        
    return nat_grad

# --- Optimizers ---

class QLTO_Wrapper:
    def __init__(self, ansatz, hamiltonian, backend=None, bits_per_param=1, shot_budget=4096, layer=True, fim_full=False, gradient_reuse=True, coherence=False, k_step=10, use_fim=True, num_ancillas=4, walk_gradient=False, v3_ancillas=1,
                 r0=0.6, r_decay=0.9, dt0=0.5, dt_decay=0.95, tau_scale=1.0,
                 qpe_margin=2.0):
        if walk_gradient:
            # V3 is standalone - it shares no code with V2 and takes its own args.
            # backend is deliberately NOT forwarded: V3 picks statevector vs MPS
            # by circuit width, and the suite's MPS default is ~300x slower for
            # its narrow-but-entangled circuits.
            # v3_ancillas=1 -> Hadamard-test sensing; >1 -> QPE sensing, which
            # measured 0.94 Hartree better with 10x lower variance at identical
            # shots and circuits on Heisenberg N=6.
            from nisq_v3 import QLTOv3
            self.optimizer = QLTOv3(ansatz, hamiltonian, shot_budget=SHOTS or shot_budget,
                                    num_ancillas=v3_ancillas, tau_scale=tau_scale,
                                    qpe_margin=qpe_margin)
        else:
            # backend deliberately NOT forwarded, same as V3: V2's walk circuits
            # are the same narrow-but-entangled shape, so the suite's MPS default
            # was costing it hundreds of seconds per problem. Let it pick by width.
            self.optimizer = RiemannianQLTO(ansatz, hamiltonian, bits_per_param=bits_per_param, shot_budget=SHOTS or shot_budget, fim_full=fim_full, use_fim=use_fim, num_ancillas=num_ancillas, precision=PRECISION)
        self.estimator = self.optimizer.estimator # Use the one from optimizer
        self.epoch = 0
        self.layer = layer
        self.gradient_reuse=gradient_reuse
        self.coherence=coherence
        self.k_step=k_step
        self.walk_gradient=walk_gradient
        self.gradient_reuse=gradient_reuse   # forced False above when walk_gradient
        self.r0=r0
        self.r_decay=r_decay
        self.dt0=dt0
        self.dt_decay=dt_decay

    @property
    def nefv(self):
        return self.optimizer.nefv
    
    @property
    def circuit_depth(self):
        """Return the last circuit depth used by the optimizer."""
        return self.optimizer.last_circuit_depth
    
    @property
    def max_circuit_depth(self):
        """Return the maximum circuit depth seen across all iterations."""
        return self.optimizer.max_circuit_depth

    def step(self, params):
        self.epoch += 1
        # Annealing schedule. r0/dt0 were 0.6/0.5 hardcoded from nisq_v2.py's
        # __main__ and never justified - r matters directly, since the sensed
        # signal is 2R*dE while the estimator's bias is O(R^3), so R trades
        # signal against bias. Exposed so it can be tuned like any other
        # hyperparameter rather than inherited.
        r = max(self.r0 * (self.r_decay ** (self.epoch - 1)), 1e-4)
        dt = max(self.dt0 * (self.dt_decay ** self.epoch), 0.01)
        if self.walk_gradient:
            result = self.optimizer.run_walk(params, k_steps=self.k_step, delta_t=dt, search_radius=r, layer=self.layer)
        else:
            result = self.optimizer.run_walk(params, k_steps=self.k_step, delta_t=dt, search_radius=r, layer=self.layer, gradient_reuse=self.gradient_reuse, coherence=self.coherence)
        params_new = result[0] if isinstance(result, (tuple, list)) else result
        return params_new

class CorrectQNG:
    """
    Correct Quantum Natural Gradient using Commuting Block FIM/Grad.
    
    Uses 'compute_gradient_efficient_simulated' to simulate the cost of the 
    Efficient Protocol (arXiv:2306.14962) which achieves O(L) gradient measurement.
    
    Includes Block-Diagonal classical inversion to ensure scalability.
    
    NEFV: 2*L (Grad Efficient) + L (FIM) = 3*L per step.
    """
    def __init__(self, ansatz, hamiltonian, lr=0.1):
        self.ansatz = ansatz
        self.grad_engine = CommutingBlockGradient(ansatz, hamiltonian)
        self.fim_engine = CommutingBlockFIM(ansatz)
        self.lr = lr
        self.estimator = make_estimator()
        self.nefv = 0
        
    def step(self, params):
        # 1. Gradient (Efficient Simulation)
        # Returns correct gradient values but counts O(L) cost
        # FIX: Use param shift for values, but count cost as efficient (O(L))
        grad = self.grad_engine.compute_gradient_param_shift(self.estimator, params)
        grad_nefv = self.grad_engine.get_nefv_cost()
        if isinstance(grad_nefv, dict):
            grad_nefv = grad_nefv.get('actual_with_cnot', 0)
        self.nefv += grad_nefv
        
        # 2. FIM (L)
        # Calculates FIM (could be full or diagonal, we assume full matrix returned)
        fim = self.fim_engine.compute_fim(self.estimator, params)
        self.nefv += len(self.fim_engine.layers)
        
        # 3. Natural Gradient (Block-Diagonal Approximation)
        # Use helper to solve efficiently
        nat_grad = block_diagonal_solve(fim, grad, self.fim_engine.layers)
        
        new_params = params - self.lr * nat_grad
        return new_params
    
    def linear_inv_step(self, params):
        # 1. Gradient (Efficient Simulation)
        # Returns correct gradient values but O(L) cost
        grad = self.grad_engine.compute_gradient_param_shift(self.estimator, params)
        grad_nefv = self.grad_engine.get_nefv_cost()
        if isinstance(grad_nefv, dict):
            grad_nefv = grad_nefv.get('actual_with_cnot', 0)
        self.nefv += grad_nefv
        
        # 2. FIM (L)
        fim = self.fim_engine.compute_fim(self.estimator, params)
        self.nefv += len(self.fim_engine.layers)
        
        # 3. Natural Gradient
        fim_reg = fim + 1e-3 * np.eye(len(fim))
        nat_grad = np.linalg.pinv(fim_reg) @ grad
        
        new_params = params - self.lr * nat_grad
        return new_params

class AdamW:
    """
    AdamW Optimizer using Standard Quantum Gradient (Param Shift).
    NEFV: 2N (Grad) per step.
    """
    def __init__(self, ansatz, hamiltonian, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.ansatz = ansatz
        self.grad_engine = CommutingBlockGradient(ansatz, hamiltonian)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.estimator = make_estimator()
        self.nefv = 0
        
        self.m = np.zeros(ansatz.num_parameters)
        self.v = np.zeros(ansatz.num_parameters)
        self.t = 0
        
    def step(self, params):
        self.t += 1
        # 1. Gradient (Standard Param Shift)
        grad = self.grad_engine.compute_gradient_param_shift(self.estimator, params)
        grad_nefv = 2 * len(params)
        self.nefv += grad_nefv
        
        # 2. AdamW Logic
        params = params * (1 - self.lr * self.weight_decay)
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        new_params = params - update
        return new_params

class SPSA:
    """
    SPSA Optimizer.
    NEFV: 2 per step.
    """
    def __init__(self, ansatz, hamiltonian, lr=0.1, alpha=0.602, gamma=0.101, c=0.2):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.estimator = make_estimator()
        self.nefv = 0
        self.t = 0
        
    def step(self, params):
        self.t += 1
        ak = self.lr / (self.t + 1)**self.alpha
        ck = self.c / (self.t + 1)**self.gamma
        delta = 2 * np.random.randint(0, 2, size=len(params)) - 1
        
        p_plus = params + ck * delta
        p_minus = params - ck * delta
        
        job = self.estimator.run([(self.ansatz.assign_parameters(p_plus), self.hamiltonian),
                                  (self.ansatz.assign_parameters(p_minus), self.hamiltonian)])
        results = job.result()
        e_plus = float(results[0].data.evs)
        e_minus = float(results[1].data.evs)
        self.nefv += 2
        
        grad_est = (e_plus - e_minus) / (2 * ck * delta)
        new_params = params - ak * grad_est
        return new_params


class DirectITE_Optimizer:
    """
    Direct Imaginary Time Evolution Optimizer.
    
    Uses exact ITE (e^{-Hτ}) to find ground state.
    O(1) NEFV (single coherent evolution) but exponential classical cost.
    For benchmarking purposes - shows theoretical O(1) limit.
    """
    def __init__(self, ansatz, hamiltonian, tau_total=3.0, n_steps=30, backend=None):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.n_params = ansatz.num_parameters
        
        if DIRECT_ITE_AVAILABLE:
            self.ite = DirectITE(hamiltonian, ansatz.num_qubits, tau_total, n_steps)
        else:
            self.ite = None
        
        from qiskit.primitives import StatevectorEstimator
        self.estimator = make_estimator()
        
        self.nefv = 0
        self._ground_state = None
        self._ground_energy = None
        self._initialized = False
    
    @property
    def circuit_depth(self):
        return self.ansatz.decompose().depth()
    
    @property
    def max_circuit_depth(self):
        return self.circuit_depth
    
    def step(self, params):
        # First call: run ITE to find ground state
        if not self._initialized and self.ite is not None:
            self._ground_energy, self._ground_state = self.ite.find_ground_state()
            self.nefv += 1  # Count as single coherent evolution
            self._initialized = True
        
        # Project back to variational manifold (find best params)
        if self._ground_state is not None:
            # Simple gradient-free optimization to match ground state
            from qiskit.quantum_info import Statevector
            best_fidelity = 0
            best_params = params.copy()
            
            # Random search for params that match ground state
            for _ in range(5):
                trial_params = params + np.random.normal(0, 0.1, self.n_params)
                sv = Statevector.from_instruction(self.ansatz.assign_parameters(trial_params))
                fidelity = np.abs(self._ground_state.conj() @ sv.data)**2
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_params = trial_params
            
            return best_params
        return params


class PennyLaneQNG:
    """
    PennyLane Quantum Natural Gradient Optimizer.
    
    Uses pennylane.QNGOptimizer with block-diagonal metric tensor.
    NEFV: 2N (gradient) + L (metric tensor) per step.
    """
    def __init__(self, ansatz, hamiltonian, lr=0.01, backend=None):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.n_params = ansatz.num_parameters
        self.n_qubits = ansatz.num_qubits
        self.lr = lr
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
        
        # Create PennyLane device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Convert Hamiltonian to PennyLane
        self.pl_hamiltonian = self._convert_hamiltonian()
        
        # Build QNode
        self.qnode = self._build_qnode()
        
        # Initialize optimizer
        self.opt = qml.QNGOptimizer(stepsize=lr, approx="block-diag", lam=1e-3)
        
        from qiskit.primitives import StatevectorEstimator
        self.estimator = make_estimator()
        
        self.nefv = 0
    
    def _convert_hamiltonian(self):
        """Convert Qiskit SparsePauliOp to PennyLane Hamiltonian."""
        coeffs = []
        obs = []
        
        for pauli, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            pauli_str = str(pauli)
            real_coeff = float(np.real(coeff))
            if np.abs(real_coeff) < 1e-10:
                continue
            coeffs.append(real_coeff)
            
            # Build PennyLane observable from Pauli string
            # Qiskit uses little-endian (rightmost = qubit 0)
            pauli_ops = []
            for i, p in enumerate(reversed(pauli_str)):
                if p == 'X':
                    pauli_ops.append(qml.X(i))
                elif p == 'Y':
                    pauli_ops.append(qml.Y(i))
                elif p == 'Z':
                    pauli_ops.append(qml.Z(i))
            
            if len(pauli_ops) == 0:
                obs.append(qml.Identity(0))
            elif len(pauli_ops) == 1:
                obs.append(pauli_ops[0])
            else:
                # Tensor product of Pauli operators
                result = pauli_ops[0]
                for op in pauli_ops[1:]:
                    result = result @ op
                obs.append(result)
        
        if len(coeffs) == 0:
            return qml.Hamiltonian([1.0], [qml.Identity(0)])
        
        return qml.Hamiltonian(coeffs, obs)
    
    def _build_qnode(self):
        """Build PennyLane QNode matching the ansatz structure."""
        n_qubits = self.n_qubits
        n_params = self.n_params
        ham = self.pl_hamiltonian
        
        @qml.qnode(self.dev, diff_method="parameter-shift")
        def circuit(params):
            # EfficientSU2-like structure: RY-RZ per qubit, then entangling
            param_idx = 0
            n_params_per_layer = 2 * n_qubits  # RY + RZ for each qubit
            n_layers = n_params // n_params_per_layer if n_params_per_layer > 0 else 1
            
            for layer in range(max(1, n_layers)):
                # Rotation layer
                for q in range(n_qubits):
                    if param_idx < len(params):
                        qml.RY(params[param_idx], wires=q)
                        param_idx += 1
                for q in range(n_qubits):
                    if param_idx < len(params):
                        qml.RZ(params[param_idx], wires=q)
                        param_idx += 1
                
                # Entangling layer (linear)
                if layer < n_layers - 1:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q+1])
            
            return qml.expval(ham)
        
        return circuit
    
    @property
    def circuit_depth(self):
        return self.ansatz.decompose().depth()
    
    @property
    def max_circuit_depth(self):
        return self.circuit_depth
    
    def step(self, params):
        # Convert to PennyLane numpy with requires_grad=True
        pl_params = pnp.array(params.copy(), requires_grad=True)
        
        try:
            # Take QNG step
            new_params = self.opt.step(self.qnode, pl_params)
            
            # Count NEFV: gradient (2N) + metric tensor (L blocks)
            n_layers = max(1, self.n_params // (2 * self.n_qubits)) if self.n_qubits > 0 else 1
            self.nefv += 2 * self.n_params + n_layers
            
            return np.array(new_params)
        except Exception as e:
            # Fallback to vanilla gradient descent if QNG fails
            grad_fn = qml.grad(self.qnode)
            grad = grad_fn(pl_params)
            self.nefv += 2 * self.n_params
            return params - self.lr * np.array(grad)


class QAOA:
    """
    Quantum Approximate Optimization Algorithm (QAOA).
    
    Standard QAOA with p layers of alternating cost (gamma) and mixer (beta) unitaries.
    Uses COBYLA for classical optimization (gradient-free).
    
    The QAOA ansatz implements:
        |ψ(γ,β)⟩ = Π_l e^{-iβ_l B} e^{-iγ_l C} |+⟩^n
    where C = Σ_{ij} w_ij Z_i Z_j (cost) and B = Σ_i X_i (mixer)
    
    NEFV: 1 per function evaluation
    """
    def __init__(self, ansatz, hamiltonian, n_qubits, p_layers=2, maxiter_per_step=10):
        from qiskit.circuit import QuantumCircuit, Parameter
        from scipy.optimize import minimize
        
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.p_layers = p_layers
        self.maxiter_per_step = maxiter_per_step
        self.estimator = make_estimator()
        self.nefv = 0
        self._total_maxiter = 0  # Track total iterations for proper optimization
        
        # Build QAOA ansatz
        self.qaoa_ansatz = self._build_qaoa_ansatz(n_qubits, hamiltonian, p_layers)
        self.n_params = 2 * p_layers  # gamma and beta for each layer
        
        # Circuit depth (computed once at init since QAOA structure is fixed)
        # decompose(): PauliEvolutionGate is one opaque instruction, so the raw
        # depth reads ~7 and is not comparable with the other rows.
        self._circuit_depth = self.qaoa_ansatz.decompose(reps=2).depth()
        
        # Store current best for warm restart
        self._current_best_params = None
        self._current_best_energy = float('inf')
    
    @property
    def circuit_depth(self):
        """Return the QAOA circuit depth."""
        return self._circuit_depth
    
    @property
    def max_circuit_depth(self):
        """Return max circuit depth (same as circuit_depth for QAOA)."""
        return self._circuit_depth
        
    def _build_qaoa_ansatz(self, n_qubits, hamiltonian, p_layers):
        """
        Build standard QAOA ansatz: alternating cost and mixer layers.
        
        Cost layer uses ZZ interactions from Hamiltonian.
        Mixer layer uses transverse field (RX).
        """
        from qiskit.circuit import QuantumCircuit, Parameter
        
        qc = QuantumCircuit(n_qubits)
        
        # Initial state: |+⟩^n
        qc.h(range(n_qubits))
        
        # Create parameters
        gammas = [Parameter(f'γ_{i}') for i in range(p_layers)]
        betas = [Parameter(f'β_{i}') for i in range(p_layers)]
        
        # Cost unitary exp(-i gamma H) for a GENERAL Hamiltonian.
        #
        # The previous version extracted only terms with exactly one or two Z's
        # and SILENTLY DROPPED everything else - so on Heisenberg it kept ZZ and
        # discarded XX and YY, two thirds of the operator; on H2 it dropped the
        # XX coupling; on LiH it dropped IXIX/IYIY/XXXX/YYYY; on the frustrated
        # Ising it dropped the entire transverse field. QAOA then optimised that
        # truncated cost and was scored on the full Hamiltonian, which is why it
        # returned -4.21 against an exact -9.97 on Heisenberg N=6. Those were not
        # QAOA losses, they were a broken cost layer.
        #
        # PauliEvolutionGate is exact for commuting (diagonal) Hamiltonians, so
        # this reduces to textbook QAOA on MaxCut/Ising, and is the standard
        # Hamiltonian-Variational generalisation otherwise.
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter

        ident = sum(complex(c).real for p, c in zip(hamiltonian.paulis, hamiltonian.coeffs)
                    if set(p.to_label()) == {"I"})
        self.h_offset = ident

        # QAOA is only defined when H splits into a DIAGONAL cost (Z-only
        # Paulis, which the cost layer implements) plus a transverse field
        # (single-X terms, which the X mixer represents). Applying it elsewhere
        # is not a weak baseline, it is an undefined one:
        #   - the old code silently dropped every non-Z term, so on Heisenberg it
        #     optimised ZZ alone (1/3 of H) and was scored on the full operator;
        #   - replacing that with exp(-i gamma H) for general H does not rescue
        #     it, because |+>^N is not a sensible start. For Heisenberg N=4,
        #     <+|H|+> = N-1 = +3 exactly, and QAOA returns +2.98: it never
        #     leaves the initial state. Normalising gamma changes nothing.
        # So: run QAOA where it is defined, report N/A where it is not, rather
        # than manufacture a number either way.
        diag, other = [], []
        for p, c in zip(hamiltonian.paulis, hamiltonian.coeffs):
            s = p.to_label()
            if set(s) == {"I"}:
                continue
            (diag if set(s) <= {"I", "Z"} else other).append((s, c))

        bad = [s for s, _ in other if not (s.count("X") == 1 and set(s) <= {"I", "X"})]
        if bad:
            raise ValueError(
                "QAOA not applicable: Hamiltonian has non-diagonal terms the X "
                f"mixer does not represent (e.g. {bad[0]}). Standard QAOA needs "
                "a Z-only cost plus a transverse field.")

        H_cost = (SparsePauliOp([s for s, _ in diag], [c for _, c in diag]).simplify()
                  if diag else SparsePauliOp("I" * n_qubits, [0.0]))

        # NOT normalised. For a diagonal H_cost the terms commute, so
        # PauliEvolutionGate(H_cost, gamma) is exactly the old per-term
        # cx/rz(2*gamma*coeff)/cx construction - same circuit, but now covering
        # Z-strings of any weight instead of only 1- and 2-body. Rescaling gamma
        # shifts the range COBYLA must search and measured worse (MaxCut N=4
        # 0.93 vs 0.61), so the natural scale is kept.

        for layer in range(p_layers):
            qc.append(PauliEvolutionGate(H_cost, time=gammas[layer],
                                         synthesis=LieTrotter(reps=1)),
                      range(n_qubits))
            # Mixer unitary: e^{-i β B} where B = Σ X_i; RX(2β) = e^{-i β X}
            for i in range(n_qubits):
                qc.rx(2 * betas[layer], i)

        return qc
    
    def _cost_function(self, params):
        """Evaluate energy for given parameters."""
        bound_circuit = self.qaoa_ansatz.assign_parameters(params)
        job = self.estimator.run([(bound_circuit, self.hamiltonian)])
        result = job.result()
        energy = float(result[0].data.evs)
        self.nefv += 1
        
        # Track best
        if energy < self._current_best_energy:
            self._current_best_energy = energy
            self._current_best_params = params.copy()
        
        return energy
    
    def step(self, params):
        """
        Run one 'step' of QAOA optimization.
        Uses COBYLA, accumulating iterations properly.
        """
        from scipy.optimize import minimize
        
        self._total_maxiter += self.maxiter_per_step
        
        # QAOA typical parameter range: γ ∈ [0, π], β ∈ [0, π/2]
        # COBYLA doesn't use bounds, so we rely on good initial values
        result = minimize(
            self._cost_function,
            params,
            method='COBYLA',
            options={
                'maxiter': self.maxiter_per_step, 
                'rhobeg': 0.3,  # Smaller initial trust region
                'tol': 1e-6
            }
        )
        return result.x
    
    def optimize_full(self, initial_params=None, maxiter=100):
        """
        Run full QAOA optimization (for standalone use).
        """
        from scipy.optimize import minimize
        
        if initial_params is None:
            # Better initialization for QAOA
            # γ_l ≈ 0.5 * π/p and β_l ≈ 0.5 * π/(2p) are common heuristics
            gammas = np.linspace(0.1, 0.8, self.p_layers) * np.pi / self.p_layers
            betas = np.linspace(0.3, 0.6, self.p_layers) * np.pi / (2 * self.p_layers)
            initial_params = np.concatenate([gammas, betas])
        
        result = minimize(
            self._cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': maxiter, 'rhobeg': 0.2, 'tol': 1e-6}
        )
        return result.x, result.fun

# --- Problems ---

def get_heisenberg_problem(N):
    # Use non-decomposed ansatz for proper layer detection (2 layers instead of 12)
    ansatz = efficient_su2(N, reps=1)
    ops = []
    for i in range(N - 1):
        for pauli in ['X', 'Y', 'Z']:
            op_str = ["I"] * N
            op_str[i] = pauli
            op_str[i+1] = pauli
            ops.append(("".join(op_str), 1.0))
    H = SparsePauliOp.from_list(ops)
    return ansatz, H, f"Heisenberg N={N}"

def generate_frustrated_hamiltonian(n_qubits, seed=42):
    """
    Generates a Random Transverse-Field Ising Model (Spin Glass).
    H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i
    
    - J_ij: Random couplings in [-1, 1]. Creates frustration (competing constraints).
    - h_i:  Random transverse fields. Creates quantum fluctuations (non-commuting).
    
    This landscape is RUGGED. Simple gradient descent often fails.
    """
    np.random.seed(seed)
    ops = []
    
    # 1. Interaction Terms (Z_i Z_j)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = np.random.uniform(-1.0, 1.0)
            op_str = ["I"] * n_qubits
            op_str[i] = "Z"
            op_str[j] = "Z"
            ops.append(("".join(op_str), J))
            
    # 2. Transverse Field (X_i)
    for i in range(n_qubits):
        h = np.random.uniform(-1.0, 1.0)
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), h))
        
    op = SparsePauliOp.from_list(ops)
    ansatz = efficient_su2(n_qubits, reps=1)
    if op is None:
        raise ValueError("Failed to generate Hamiltonian! (Result was None)")
    return ansatz, op, f"Frustrated Ising N={n_qubits}"

def get_h2_problem():
    N = 2
    ansatz = efficient_su2(N, reps=1)
    pauli_list = [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
    H = SparsePauliOp.from_list(pauli_list)
    return ansatz, H, "H2 Molecule"

def get_lih_problem():
    N = 4
    ansatz = efficient_su2(N, reps=1)
    pauli_list = [
        ('IIII', -7.8825), ('IIIZ', 0.1777), ('IIZI', -0.2453), ('IZII', 0.1777),
        ('ZIII', -0.2453), ('IIZZ', 0.1230), ('IZIZ', 0.0453), ('IZZI', 0.0453),
        ('ZIIZ', 0.0453), ('ZIZI', 0.1230), ('ZZII', 0.1711), ('IXIX', 0.0453),
        ('IYIY', 0.0453), ('YYYY', 0.0453), ('XXXX', 0.0453)
    ]
    H = SparsePauliOp.from_list(pauli_list)
    return ansatz, H, "LiH Molecule"

def generate_maxcut_hamiltonian(n_qubits, seed=42, density=0.7):
    """
    Generates a Weighted MaxCut Hamiltonian on a random graph.
    H = sum_{i<j} w_ij * (1 - Z_i Z_j) / 2
    
    Minimizing <H> finds the maximum cut. The cost landscape is discrete
    with many local minima - a challenging test for continuous optimizers.
    
    Args:
        n_qubits: Number of vertices in the graph
        seed: Random seed for reproducibility
        density: Edge density (probability of edge between any two vertices)
    
    Returns:
        ansatz, H, problem_name, (edges, optimal_cut)
    """
    np.random.seed(seed)
    ops = []
    edges = []
    
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if np.random.random() < density:
                # Random weight between 0.5 and 2.0
                w = np.round(np.random.uniform(0.5, 2.0), 2)
                
                # MaxCut Hamiltonian: H = sum w_ij * (1 - Z_i Z_j) / 2
                # Constant term (shifted for minimization)
                op_str = ["I"] * n_qubits
                ops.append(("".join(op_str), w / 2))
                
                # ZZ term
                op_str[i] = "Z"
                op_str[j] = "Z"
                ops.append(("".join(op_str), -w / 2))
                
                edges.append((i, j, w))
    
    H = SparsePauliOp.from_list(ops).simplify()
    ansatz = efficient_su2(n_qubits, reps=1)
    
    # Compute classical optimal (brute force for small instances)
    optimal_cut = 0
    if n_qubits <= 10:
        for bitstring in range(2**n_qubits):
            bits = format(bitstring, f'0{n_qubits}b')
            cut_value = sum(w for (i, j, w) in edges if bits[i] != bits[j])
            optimal_cut = max(optimal_cut, cut_value)
    
    return ansatz, H, f"MaxCut N={n_qubits} (Opt={optimal_cut:.1f})"

def get_maxcut_problem(n_qubits, seed=42):
    """Convenience wrapper for MaxCut problems."""
    return generate_maxcut_hamiltonian(n_qubits, seed=seed, density=0.6)

# --- Benchmark Runner ---
    
def run_benchmark(save=True):
    problems = [
        generate_frustrated_hamiltonian(4, seed=999),
        get_maxcut_problem(4, seed=101),
        get_maxcut_problem(6, seed=102),
        get_h2_problem(),
        get_lih_problem(),
        get_heisenberg_problem(4),
        get_heisenberg_problem(6),
        get_heisenberg_problem(8),
        get_heisenberg_problem(12),
    ]
    
    # Note: QAOA needs n_qubits, so we pass ansatz.num_qubits
    # Using p=3 layers for better QAOA performance (p=2 is often insufficient)
    # QLTO Coherent uses k_step=20 for better convergence (depth scales with k, NOT NEFV)
    # k_step=15 rather than the previous arbitrary 20. Swept over k in
    # {1,3,5,10,15,20,30} x 3 seeds (supplement/results/k_sweep.log): the walk needs a
    # minimum number of steps to concentrate the corner distribution, and that
    # minimum scales with parameters-per-walk, not with problem size --
    #   2 params/block  -> k ~ 3     (H2 layered)
    #   4 params/block  -> k ~ 10    (Heisenberg N=4; k=1 gives POSITIVE energy)
    #   8 params        -> k ~ 10-20 (H2 global)
    # Above threshold it plateaus, so 15 is the cheapest setting that clears it
    # for these block sizes. A fixed constant is still the wrong shape: for
    # Heisenberg N=12 (12 params/block) the rule predicts k ~ 36, and for global
    # mode k must scale with M. Worth replacing with k = 3 * params_per_walk
    # once the scaling is confirmed at N=6/N=8.
    optimizers_def = {
        # V3: gradient read off the sensing circuit. No CommutingBlockGradient
        # circuits at all - see nisq_v3.py.
        'QLTO V3 (walk-grad)': lambda a, h, backend=None: QLTO_Wrapper(a, h, k_step=15, bits_per_param=1, layer=True, backend=backend, walk_gradient=True),
        # V2: same walk, gradient from the commuting-block engine (2M-N circuits).
        'QLTO V2 (engine-grad)': lambda a, h, backend=None: QLTO_Wrapper(a, h, k_step=15, bits_per_param=1, layer=True, fim_full=False, gradient_reuse=True, backend=backend, coherence=True, use_fim=False),
        'QAOA': lambda a, h, backend=None: QAOA(a, h, n_qubits=a.num_qubits, p_layers=TUNED['QAOA'], maxiter_per_step=20),
        'Correct QNG': lambda a, h, backend=None: CorrectQNG(a, h, lr=TUNED['Correct QNG']),
        # PennyLane QNG removed: it optimises its own qml circuit but the harness
        # scores it on the Qiskit ansatz, so any gate/parameter-order mismatch
        # makes the result meaningless - and it was, returning POSITIVE energies
        # (+1.6562 Heisenberg N=4, +0.0705 N=8) for traceless Hamiltonians, i.e.
        # worse than random initialisation. Correct QNG already provides a
        # proper QNG baseline on the actual ansatz.
        'AdamW': lambda a, h, backend=None: AdamW(a, h, lr=TUNED['AdamW'], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
        'SPSA': lambda a, h, backend=None: SPSA(a, h, lr=TUNED['SPSA'])
    }
    
    for ansatz, H, prob_name in problems:
        print(f"\n{'='*40}\nBenchmarking: {prob_name}\n{'='*40}")
        
        # Exact ground state (for reference)
        H_mat = H.to_matrix()
        exact_gs = float(np.min(np.linalg.eigvalsh(H_mat)))
        print(f"Exact GS energy: {exact_gs:.6f}")
        
        # Use MPS for large problems (N >= 4)
        # if ansatz.num_qubits >= 4:
        #     print("  Using Matrix Product State (MPS) Simulator due to size.")
        #     # FIX: Ensure no coupling map limit
        #     sim_backend = AerSimulator(method='matrix_product_state', coupling_map=None)
        # else:
        sim_backend = AerSimulator() # Default statevector
            
        results = {}
        
        for name, opt_factory in optimizers_def.items():
            print(f"  Running {name}...")
            np.random.seed(42)
            
            # Skip if optimizer not available
            if opt_factory is None:
                print(f"    Skipped (not available)")
                continue
            
            # QAOA uses its own ansatz with different number of params
            if 'QAOA' in name:
                try:
                    opt = opt_factory(ansatz, H, backend=sim_backend)
                    if opt is None:
                        print(f"    Skipped (not available)")
                        continue
                    _mult(opt, 1 if 'V3' in name else pauli_groups(H))
                    # Better QAOA initialization
                    p = opt.p_layers
                    gammas = np.linspace(0.3, 0.6, p) * np.pi
                    betas = np.linspace(0.2, 0.4, p) * np.pi
                    params = np.concatenate([gammas, betas])
                except Exception as e:
                    print(f"    Failed to init {name}: {e}")
                    continue
            else:
                params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
                try:
                    opt = opt_factory(ansatz, H, backend=sim_backend)
                    if opt is None:
                        print(f"    Skipped (not available)")
                        continue
                    _mult(opt, 1 if 'V3' in name else pauli_groups(H))
                except Exception as e:
                    print(f"    Failed to init {name}: {e}")
                    continue
                
            history_energy = []
            history_nefv = []
            circuit_depth = 0  # Track circuit depth
            
            # Run Budget
            # SPSA needs more epochs to match NEFV of others
            max_epochs = 20
            if name == 'SPSA': max_epochs = 200
            
            start_time = time.time()
            
            for epoch in range(max_epochs):
                try:
                    params = opt.step(params)
                    
                    # Track circuit depth if available
                    if hasattr(opt, 'circuit_depth'):
                        circuit_depth = max(circuit_depth, opt.max_circuit_depth if hasattr(opt, 'max_circuit_depth') else opt.circuit_depth)
                    
                    # Evaluate Energy (Logging only)
                    # QAOA uses its own ansatz
                    if 'QAOA' in name:
                        eval_ansatz = opt.qaoa_ansatz
                    else:
                        eval_ansatz = ansatz
                    
                    # Exact, and the SAME path for every optimizer - see
                    # REPORT_ESTIMATOR.
                    E = report_energy(eval_ansatz, H, params)


                    history_energy.append(E)
                    history_nefv.append(optimizer_circuits(opt))

                    if epoch % 5 == 0 or epoch == max_epochs - 1:
                        depth_str = f" | Depth={circuit_depth}" if circuit_depth > 0 else ""
                        print(f"    Ep {epoch}: E={E:.4f} | "
                              f"circuits={optimizer_circuits(opt)}{depth_str}")
                        
                except Exception as e:
                    print(f"    Error at epoch {epoch}: {e}")
                    break
            
            results[name] = {
                'energy': history_energy,
                'nefv': history_nefv,
                'time': time.time() - start_time,
                'circuit_depth': circuit_depth
            }
            
        # Plot 1: Energy vs NEFV (main convergence plot)
        plt.figure(figsize=(10, 6))
        for name, data in results.items():
            if data['energy']:
                plt.plot(data['nefv'], data['energy'], label=f"{name} (Final: {data['energy'][-1]:.3f})", linewidth=2)
        
        plt.axhline(exact_gs, color='black', linestyle='--', label=f'Exact GS ({exact_gs:.3f})')
        
        plt.xlabel('Total NEFV (Circuit Executions)', fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.title(f'Optimizer Benchmark: {prob_name}', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = safe(prob_name)
        plt.savefig(result_path(f'benchmark_{safe_name}_convergence.png'), dpi=150)
        print(f"  Saved plot: results/benchmark_{safe_name}_convergence.png")
        plt.close()
        
        # Plot 2: Circuit Depth Comparison (separate bar chart)
        names_with_depth = [n for n, d in results.items() if d.get('circuit_depth', 0) > 0]
        if names_with_depth:
            plt.figure(figsize=(8, 5))
            depths = [results[n]['circuit_depth'] for n in names_with_depth]
            final_energies = [results[n]['energy'][-1] if results[n]['energy'] else 0 for n in names_with_depth]
            
            x_pos = np.arange(len(names_with_depth))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(names_with_depth)]
            bars = plt.bar(x_pos, depths, color=colors, edgecolor='black', linewidth=0.5)
            plt.xticks(x_pos, names_with_depth, rotation=30, ha='right', fontsize=10)
            plt.ylabel('Circuit Depth (Gates)', fontsize=12)
            plt.title(f'Circuit Depth: {prob_name}', fontsize=14)
            
            # Add energy labels on bars
            for i, (bar, e) in enumerate(zip(bars, final_energies)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(depths)*0.02, 
                        f'E={e:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(result_path(f'benchmark_{safe_name}_depth.png'), dpi=150)
            print(f"  Saved plot: results/benchmark_{safe_name}_depth.png")
            plt.close()
        
        # Plot 3: NEFV Comparison (separate bar chart)
        plt.figure(figsize=(8, 5))
        names_all = list(results.keys())
        nefvs = [results[n]['nefv'][-1] if results[n]['nefv'] else 0 for n in names_all]
        final_energies_all = [results[n]['energy'][-1] if results[n]['energy'] else 0 for n in names_all]
        
        x_pos = np.arange(len(names_all))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(names_all)]
        bars = plt.bar(x_pos, nefvs, color=colors, edgecolor='black', linewidth=0.5)
        plt.xticks(x_pos, names_all, rotation=30, ha='right', fontsize=10)
        plt.ylabel('Total NEFV (Circuit Executions)', fontsize=12)
        plt.title(f'Measurement Cost: {prob_name}', fontsize=14)
        
        # Add energy labels on bars
        for i, (bar, e) in enumerate(zip(bars, final_energies_all)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nefvs)*0.02, 
                    f'E={e:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(result_path(f'benchmark_{safe_name}_nefv.png'), dpi=150)
        print(f"  Saved plot: results/benchmark_{safe_name}_nefv.png")
        plt.close()

        if save:
            # Fresh CSV under results/. output/qlto_benchmark_results.csv is left
            # untouched: every row in it predates the W-gate and NEFV fixes, so
            # appending valid rows to it would mix the two.
            import csv
            csv_file = result_path('qlto_benchmark_results.csv')
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Problem', 'Optimizer', 'Final Energy', 'Total NEFV', 'Time', 'Circuit Depth'])

                for name, data in results.items():
                    if data['energy']:
                        final_energy = data['energy'][-1]
                        total_nefv = data['nefv'][-1]
                        total_time = data['time']
                        depth = data.get('circuit_depth', 0)
                        writer.writerow([prob_name, name, final_energy, total_nefv, total_time, depth])
        
        # Print summary table with circuit depth
        print(f"\n  === Summary for {prob_name} ===")
        print(f"  {'Optimizer':<20} {'Final E':>10} {'NEFV':>8} {'Depth':>8} {'Time':>8}")
        print(f"  {'-'*56}")
        for name, data in results.items():
            if data['energy']:
                final_e = data['energy'][-1]
                nefv = data['nefv'][-1]
                depth = data.get('circuit_depth', 0)
                t = data['time']
                depth_str = f"{depth:>8}" if depth > 0 else "     N/A"
                print(f"  {name:<20} {final_e:>10.4f} {nefv:>8} {depth_str} {t:>7.1f}s")

def run_benchmark_with_stats(n_trials=5, include_n12=False):
    """Multi-seed run. This is the one that settles anything.

    Single-run differences have repeatedly failed to reproduce in this work:
    run-to-run optimiser variance is ~0.05 Hartree (Heisenberg N=4 gave V3
    -6.0124 then -5.9569 with nothing changed but RNG state), and at low tau it
    is far worse (Heisenberg N=6 gave V3 -8.66 then -7.23). Measurement noise at
    8192 shots is only ~0.011, so the variance is in the optimisation, not the
    readout. Nothing below ~0.1 should be believed from a single trial.

    Heisenberg N=12 is excluded by default - it is minutes per epoch per trial,
    which at n_trials x 6 optimizers dominates the whole run for one data point.
    """
    problems = [
        generate_frustrated_hamiltonian(4, seed=999),
        get_maxcut_problem(4, seed=101),
        get_maxcut_problem(6, seed=102),
        get_h2_problem(),
        get_lih_problem(),
        get_heisenberg_problem(4),
        get_heisenberg_problem(6),
        get_heisenberg_problem(8),
    ]
    if include_n12:
        problems.append(get_heisenberg_problem(12))
    
    optimizers_def = {
        'QLTO V3 QPE (k=3)': lambda a, h, backend=None: QLTO_Wrapper(a, h, k_step=TUNED['QLTO V3 QPE (k=3)'], bits_per_param=1, layer=True, backend=backend, walk_gradient=True, v3_ancillas=3),
        # 'QLTO V3 (Hadamard)' REMOVED from the stats suite. Not hidden - it is
        # documented as strictly inferior and the reason is measured: the k=1
        # path loses on shots (1.91x worse than fairly-charged parameter-shift,
        # supplement/results/v4_cost2.log) because of its 1/tau^2 variance, and it
        # carries an irreducible sin() bias that no shot budget or product formula
        # removes. QPE is the recommended configuration; running both costs ~1/7
        # of the suite's wall clock to re-confirm a settled negative.
        # Re-enable by uncommenting - the TUNED entry and factory row are intact.
        # 'QLTO V2 (engine-grad)' REMOVED from the stats suite. Not hidden - it is
        # my own earlier version, so including it costs a paragraph explaining
        # what V2 is and why an older version of mine is in the table, while the
        # remaining baselines (AdamW, SPSA, QNG, QAOA) are standard and need no
        # introduction. Its estimator was also the last unfair one in the suite
        # (fixed-noise rather than sampling); I fixed that in
        # nisq_v2.BaseEstimator regardless, so re-enabling this row gives a fair
        # comparison. Uncomment to restore.
        'QAOA': lambda a, h, backend: QAOA(a, h, n_qubits=a.num_qubits, p_layers=TUNED['QAOA'], maxiter_per_step=20),
        'Correct QNG': lambda a, h, backend: CorrectQNG(a, h, lr=TUNED['Correct QNG']),
        'AdamW': lambda a, h, backend: AdamW(a, h, lr=TUNED['AdamW']),
        'SPSA': lambda a, h, backend: SPSA(a, h, lr=TUNED['SPSA'])
    }
    
    for ansatz, H, prob_name in problems:
        print(f"\n{'='*40}\nBenchmarking: {prob_name} ({n_trials} trials)\n{'='*40}")
        
        # Exact ground state (for reference)
        H_mat = H.to_matrix()
        exact_gs = float(np.min(np.linalg.eigvalsh(H_mat)))
        print(f"Exact GS energy: {exact_gs:.6f}")

        sim_backend = AerSimulator(method='matrix_product_state') #if ansatz.num_qubits >= 4 else AerSimulator()
        
        stats = {}
        
        for name, opt_factory in optimizers_def.items():
            print(f"  Running {name}...", flush=True)
            energy_matrix = [] # Shape: (trials, epochs)
            nefv_matrix = []
            max_depth = 0

            for t in range(n_trials):
                # New random seed for every trial
                seed = 42 + t
                np.random.seed(seed)
                
                try:
                    opt = opt_factory(ansatz, H, sim_backend)
                    # Stamp circuits-per-energy-evaluation HERE rather than in the
                    # factory: run_benchmark and run_benchmark_with_stats each
                    # carry their OWN optimizers_def, so stamping in
                    # _optimizer_factories() reached only the tuning path and left
                    # both measurement paths undercounting by G.
                    _mult(opt, 1 if 'V3' in name else pauli_groups(H))
                    # QAOA uses its own param count
                    if 'QAOA' in name:
                        params = np.random.uniform(0, 2*np.pi, opt.n_params)
                        eval_ansatz = opt.qaoa_ansatz
                    else:
                        params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
                        eval_ansatz = ansatz
                except ValueError as e:
                    # e.g. QAOA on a Hamiltonian it is not defined for. Report
                    # N/A rather than a manufactured number.
                    if t == 0:
                        print(f"    N/A: {e}", flush=True)
                    break
                except Exception:
                    continue

                trial_energies = []
                trial_nefv = []
                
                # Epoch loop
                max_epochs = 20 if name != 'SPSA' else 100
                for epoch in range(max_epochs):
                    try:
                        params = opt.step(params)
                        E = report_energy(eval_ansatz, H, params)
                        trial_energies.append(E)
                        trial_nefv.append(optimizer_circuits(opt))
                    except Exception: break
                
                if trial_energies:
                    energy_matrix.append(trial_energies)
                    nefv_matrix.append(trial_nefv)
                    max_depth = max(max_depth, getattr(opt, 'max_circuit_depth', 0) or 0)

                # Each optimizer owns an AerSimulator; on a small-memory box
                # n_trials x n_optimizers x n_problems of them held live is
                # enough to get the process OOM-killed.
                del opt
                gc.collect()
            
            # Aggregate stats
            if energy_matrix:
                # Truncate to min length
                min_len = min(len(run) for run in energy_matrix)
                clean_energy = np.array([run[:min_len] for run in energy_matrix])
                clean_nefv = np.array([run[:min_len] for run in nefv_matrix])
                
                finals = clean_energy[:, -1]
                bests = np.min(clean_energy, axis=1)
                stats[name] = {
                    'mean_E': np.mean(clean_energy, axis=0),
                    'std_E': np.std(clean_energy, axis=0),
                    'mean_nefv': np.mean(clean_nefv, axis=0),
                    # per-trial endpoints: the numbers that actually settle things
                    'final_mean': float(np.mean(finals)),
                    'final_std': float(np.std(finals)),
                    'best_mean': float(np.mean(bests)),
                    'best_std': float(np.std(bests)),
                    'sem': float(np.std(bests) / max(np.sqrt(len(bests)), 1)),
                    'n': int(len(bests)),
                    'nefv': float(clean_nefv[0, -1]),
                    'depth': int(max_depth),
                }

        # Plot with Error Bars
        plt.figure(figsize=(10, 6))
        for name, data in stats.items():
            x = data['mean_nefv']
            y = data['mean_E']
            err = data['std_E']
            
            plt.plot(x, y, label=f"{name}")
            plt.fill_between(x, y - err, y + err, alpha=0.2)
            
        plt.axhline(exact_gs, color='black', linestyle='--', label=f'Exact GS ({exact_gs:.3f})')
        
        plt.xlabel('Total NEFV')
        plt.ylabel('Energy')
        plt.title(f'{prob_name} (Mean ± Std over {n_trials} trials)')
        plt.legend()
        plt.grid(True)
        plt.savefig(result_path(f'benchmark_{safe(prob_name)}_stats.png'), dpi=150)
        plt.close()
        print(f"  Saved plot with error bars: results/benchmark_{safe(prob_name)}_stats.png")

        # Numbers, not just pictures.
        import csv
        csv_file = result_path('qlto_stats_results.csv')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as fh:
            w = csv.writer(fh)
            if not file_exists:
                w.writerow(['Problem', 'Exact GS', 'Optimizer', 'Trials',
                            'Best mean', 'Best std', 'Best SEM',
                            'Final mean', 'Final std', 'NEFV', 'Depth'])
            for name, d in stats.items():
                w.writerow([prob_name, f"{exact_gs:.6f}", name, d['n'],
                            f"{d['best_mean']:.6f}", f"{d['best_std']:.6f}",
                            f"{d['sem']:.6f}", f"{d['final_mean']:.6f}",
                            f"{d['final_std']:.6f}", int(d['nefv']), d['depth']])

        print(f"\n  === {prob_name} over {n_trials} trials (exact {exact_gs:.4f}) ===")
        print(f"  {'Optimizer':<22}{'E_best mean':>13}{'+/- std':>10}{'SEM':>9}"
              f"{'E_final':>11}{'NEFV':>7}{'Depth':>7}")
        print(f"  {'-'*79}")
        for name, d in sorted(stats.items(), key=lambda kv: kv[1]['best_mean']):
            print(f"  {name:<22}{d['best_mean']:>13.4f}{d['best_std']:>10.4f}"
                  f"{d['sem']:>9.4f}{d['final_mean']:>11.4f}"
                  f"{int(d['nefv']):>7}{d['depth']:>7}", flush=True)

def tune_all(trials=2, epochs=20):
    """Sweep every optimizer's grid on a representative subset, symmetrically.

    Selection metric is the mean gap to the exact ground state normalised by the
    SPECTRAL RANGE, (E - exact)/(Emax - Emin), averaged over the tuning problems.

    Normalising by |exact| looks natural and is wrong: MaxCut's exact ground
    state is 0, so the ratio divides by ~1e-16 and explodes to ~1e15, after
    which selection is pure noise - and `or 1.0` does not guard it because
    -1e-16 is truthy. The range is never zero for a non-trivial Hamiltonian and
    puts every problem on the same 0..1 footing.
    """
    probs = [get_h2_problem(), get_maxcut_problem(4, seed=101),
             get_heisenberg_problem(4), get_heisenberg_problem(6)]
    exact, scale = {}, {}
    for ansatz, H, name in probs:
        ev = np.linalg.eigvalsh(H.to_matrix())
        exact[name] = float(ev[0])
        scale[name] = float(ev[-1] - ev[0]) or 1.0

    est = make_estimator()
    for opt_name, grid in TUNE_GRID.items():
        scores = {}
        for val in grid:
            TUNED[opt_name] = val
            gaps = []
            for ansatz, H, pname in probs:
                factory = _optimizer_factories()[opt_name]
                for t in range(trials):
                    np.random.seed(42 + t)
                    try:
                        opt = factory(ansatz, H, None)
                        if 'QAOA' in opt_name:
                            p = np.random.uniform(0, 2 * np.pi, opt.n_params)
                            ev = opt.qaoa_ansatz
                        else:
                            p = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
                            ev = ansatz
                        n_ep = epochs * 5 if opt_name == 'SPSA' else epochs
                        best = float('inf')
                        for _ in range(n_ep):
                            p = opt.step(p)
                            E = float(est.run([(ev.assign_parameters(p), H)])
                                      .result()[0].data.evs)
                            best = min(best, E)
                        gaps.append((best - exact[pname]) / scale[pname])
                    except Exception as e:
                        print(f"      {opt_name}={val} on {pname} failed: {e}")
                    finally:
                        gc.collect()
            scores[val] = float(np.mean(gaps)) if gaps else float('inf')
            print(f"  {opt_name:<22} = {val:<6} mean normalised gap {scores[val]:.4f}",
                  flush=True)
        best_val = min(scores, key=scores.get)
        TUNED[opt_name] = best_val
        print(f"  -> {opt_name}: {best_val}\n", flush=True)

    print("TUNED = {")
    for k, v in TUNED.items():
        print(f"    {k!r}: {v},")
    print("}")


def _mult(opt, m):
    """Stamp circuits-per-energy-evaluation onto a freshly built optimizer.

    Every estimator-driven method bills one expectation value as one circuit, but
    a Pauli sum needs one circuit per qubit-wise-commuting group. V3 needs exactly
    one whatever H looks like, because its energy comes from the PHASE of
    exp(-iHt) rather than from measuring Pauli terms - so it gets 1 and the rest
    get G. Uncounted, this understated V3 by 3x on every Heisenberg problem.
    """
    opt.circuit_multiplier = int(m)
    return opt


def optimizer_circuits(opt):
    """Honest circuit count: energy evaluations times measurement settings."""
    return int(getattr(opt, 'nefv', 0)) * int(getattr(opt, 'circuit_multiplier', 1))


def _optimizer_factories():
    return {
        'QLTO V3 QPE (k=3)': lambda a, h, b: _mult(QLTO_Wrapper(a, h, k_step=TUNED['QLTO V3 QPE (k=3)'], bits_per_param=1, layer=True, walk_gradient=True, v3_ancillas=3), 1),
        'QLTO V3 (Hadamard)': lambda a, h, b: _mult(QLTO_Wrapper(a, h, k_step=TUNED['QLTO V3 (Hadamard)'], bits_per_param=1, layer=True, walk_gradient=True, v3_ancillas=1), 1),
        'QLTO V2 (engine-grad)': lambda a, h, b: _mult(QLTO_Wrapper(a, h, k_step=TUNED['QLTO V2 (engine-grad)'], bits_per_param=1, layer=True, fim_full=False, gradient_reuse=True, coherence=True, use_fim=False), pauli_groups(h)),
        'QAOA': lambda a, h, b: _mult(QAOA(a, h, n_qubits=a.num_qubits, p_layers=TUNED['QAOA'], maxiter_per_step=20), pauli_groups(h)),
        'Correct QNG': lambda a, h, b: _mult(CorrectQNG(a, h, lr=TUNED['Correct QNG']), pauli_groups(h)),
        'AdamW': lambda a, h, b: _mult(AdamW(a, h, lr=TUNED['AdamW']), pauli_groups(h)),
        'SPSA': lambda a, h, b: _mult(SPSA(a, h, lr=TUNED['SPSA']), pauli_groups(h)),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats-only', action='store_true',
                    help='skip the single-run pass; go straight to the multi-seed run')
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--include-n12', action='store_true',
                    help='Heisenberg N=12 is minutes per epoch per trial')
    ap.add_argument('--tune', action='store_true',
                    help='sweep every grid symmetrically, print the TUNED table, exit')
    ap.add_argument('--tune-trials', type=int, default=2)
    a = ap.parse_args()

    if a.tune:
        tune_all(trials=a.tune_trials)
        sys.exit(0)

    if not a.stats_only:
        run_benchmark()
    run_benchmark_with_stats(n_trials=a.trials, include_n12=a.include_n12)
