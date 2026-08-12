"""QLTO V6 - the log-register gradient: one block, one circuit per group.

V5 spends ONE REGISTER QUBIT PER PARAMETER. That caps block width n, and since the
advantage over parameter-shift is exactly 2n while circuits per gradient are
G*M/n, the register is what held the method at 2N.

V6 changes three things and nothing else. It subclasses V5 deliberately: the
circuit builder and the decoder are shared, not copied. Two copies of the same
construction drifting apart is the documented failure mode of this project, and it
already cost a published number once.

    encoding    'design'    rows of a resolution-IV Hadamard design indexed on
                            ceil(log2(n+1)) + 1 qubits instead of one per
                            parameter, columns Gray-ordered so a running parity
                            advances by a single CNOT
    block_mode  'global'    all M parameters in ONE block, so L = 1 and circuits
                            per gradient fall from G*L to G
    n_scratch   3           parities dealt round-robin over three scratch wires,
                            each with its own Gray slice of the column space, so
                            the next parity overlaps the current rotation

MEASURED, Heisenberg N=6, M=36, matched total shots, basis rz/sx/x/cx:

                            V5 (onehot/layered)   V6 (design/global, 3 scratch)
    accuracy, cos vs exact               0.9796                          0.9797
    circuits per gradient                    18                               3
    two-qubit gates per gradient            396                             402
    register qubits                          36                              10
    depth per circuit                        34                             102

Equal accuracy on a sixth of the circuits and under a third of the register, at
three times the depth and a percent more gates. That is a TRADE, not a strict
improvement: it wins outright when the cost is billed per circuit execution, which
is how vendors bill (v21), and loses when the limit is coherence.

WHAT V6 DOES NOT CHANGE. The error still falls as T^(-1/3) against
parameter-shift's T^(-1/2), because the finite-radius bias cR^2 is untouched;
parameter-shift still wins at large budgets and the crossover still recedes as
M^3. Richardson extrapolation on R is the identified fix and is not implemented.
G is also untouched, and at global block width G IS the circuit count, so it is now
the whole quantum cost. For chemistry v30 measured G ~ N^4.24.

THE RADIUS IS RESCALED AUTOMATICALLY, and this is not cosmetic. A block of n
parameters displaces the state by about sqrt(n)*R, so a radius chosen for an
N-wide block over-displaces an M-wide one and the linearisation degrades. Passing
V5's R = 0.45 straight into a 36-parameter block gives cos 0.83 instead of 0.98 -
a silent twenty percent regression that looks like a bug somewhere else. V6
therefore scales R by sqrt(N/n) internally, so the SAME radius a caller would give
V5 is correct here. Measured best radii were 0.18 at N=6/M=36 against the
predicted 0.184, and 0.10 at N=4/M=24 against a predicted 0.184 that still reached
0.978.

USAGE is identical to V5:

    q = QLTOv6(ansatz, hamiltonian, shot_budget=8192)
    for block in [b['params'] for b in q.layers if b['params']]:
        grad, energy = q.sense(theta, R, block)
        theta = q.grad_step(theta, R, block, grad)

with one block rather than several.
"""
import numpy as np

from nisq_v5 import QLTOv5


class QLTOv6(QLTOv5):
    """V5 with the log-width register, a single global block, and parallel parity.

    Every behavioural difference is a default. Passing the V5 defaults explicitly
    reproduces V5 exactly, which is what makes the two comparable.
    """

    def __init__(self, ansatz, hamiltonian, shot_budget=8192, alpha=0.9,
                 sim_seed=None, backend=None, gradient_mode='direct',
                 num_ancillas=3, qpe_margin=2.0, block_mode='global',
                 decoder='marginal', encoding='design', n_scratch=3,
                 scale_radius=True):
        super().__init__(ansatz, hamiltonian, shot_budget=shot_budget,
                         alpha=alpha, sim_seed=sim_seed, backend=backend,
                         gradient_mode=gradient_mode, num_ancillas=num_ancillas,
                         qpe_margin=qpe_margin, block_mode=block_mode,
                         decoder=decoder, encoding=encoding, n_scratch=n_scratch)
        # Off only for measuring the unscaled behaviour; leaving it off with a
        # global block is the trap described in the module docstring.
        self.scale_radius = bool(scale_radius)

    def _radius(self, R, n):
        """Radius for a block of width n, from one quoted for an N-wide block.

        Displacement grows as sqrt(n)*R, so the radius has to shrink as the block
        widens or the first-order picture the estimator rests on stops holding.
        Blocks no wider than N are left alone.
        """
        if not self.scale_radius or n <= self.N:
            return R
        return float(R) * float(np.sqrt(self.N / float(n)))

    def sense(self, centre, R, active):
        return super().sense(centre, self._radius(R, len(active)), active)

    def grad_step(self, centre, R, active, grad):
        # The step must use the SAME radius the gradient was measured at, or the
        # bounded step size and the smoothing scale disagree.
        return super().grad_step(centre, self._radius(R, len(active)), active,
                                 grad)
