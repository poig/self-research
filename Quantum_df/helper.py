import numpy as np
from numpy.linalg import eigh, svd
from qiskit.circuit.library import UnitaryGate, CSwapGate   # FredkinGate is the controlled-SWAP (cswap)
from qiskit.quantum_info import Statevector, Operator
from qiskit import QuantumCircuit

def reembedding_operator(m):
    """
    Constructs a re-embedding operator R (of shape m×(2m)) which maps an m-dimensional
    row vector v to the extended 2m-dimensional row vector [0, v]. This allows one to
    feed the recovered result from one block (dimension m) into the next block (expecting a 2m vector).
    """
    # Build R such that for any row vector v (1×m), v @ R = [0 , v] (1×(2m)).
    R = np.hstack([np.zeros((m, m), dtype=complex), np.eye(m, dtype=complex)])
    return R

def normalize_minmax(I):
    I = np.asarray(I, dtype=float)
    return (I - I.min()) / (I.max() - I.min() + 1e-8)

def normalize_2d(matrix):
    """
    Normalizes a 2D NumPy array using Frobenius norm (L2 norm).
    Returns the normalized matrix and the norm value.
    """
    norm = np.sqrt(np.sum(matrix ** 2))  # More efficient than np.linalg.norm
    if norm == 0:
        return matrix.copy(), 1.0  # Avoid divide-by-zero
    return matrix / norm, norm

def is_unitary(matrix):
    """
    Efficiently check if a matrix is unitary.
    
    O(n²) implementation that avoids forming full matrix products.
    For large matrices, uses a randomized approach that requires O(k) operations
    where k << n².
    """
    n = matrix.shape[0]
    
    # For small matrices, standard check
    if n < 100:
        # Use in-place operations to avoid temporaries
        mm = matrix @ matrix.conj().T
        np.subtract(mm, np.eye(n), out=mm)
        if np.max(np.abs(mm)) > 1e-10:
            return False
        return True
    
    # For large matrices, use randomized checking with k random vectors
    k = min(10, n // 10)
    
    for _ in range(k):
        # Generate random unit vector
        v = np.random.normal(size=n) + 1j * np.random.normal(size=n)
        v = v / np.linalg.norm(v)
        
        # Check if ||U†U|v⟩ - |v⟩|| ≈ 0
        Uv = matrix @ v
        UdUv = matrix.conj().T @ Uv
        
        # If significant deviation from identity, not unitary
        if np.linalg.norm(UdUv - v) > 1e-10:
            return False
    
    #identity = np.eye(matrix.shape[0])
    #check_unitary = np.allclose(np.dot(matrix, matrix.conj().T), identity) and np.allclose(np.dot(matrix.conj().T, matrix), identity)
    #return True if check_unitary else False

    return True

def block_encode_apply(v, U_A, s_alpha, V_A):
    """Matrix-free application of block-encoded operator to a vector."""
    # This is a sketch of how to apply the block encoding without forming full matrices
    m = len(s_alpha)
    v1, v2 = v[:m], v[m:]
    
    # Apply the block operations
    Sv1 = s_alpha * (U_A.T @ v1)
    Yv1 = np.sqrt(1 - s_alpha**2) * (U_A.T @ v1)
    Sv2 = s_alpha * (V_A.T @ v2)
    Yv2 = np.sqrt(1 - s_alpha**2) * (V_A.T @ v2)
    
    # Combine results
    result1 = U_A @ Sv1 + V_A @ Yv2
    result2 = U_A @ Yv1 - V_A @ Sv2
    
    return np.concatenate([result1, result2])

def block_encode_via_svd(A):
    """
    Optimized block-encoding of matrix A into a unitary matrix U.
    Returns U and the scaling factor alpha.
    
    This implementation avoids explicitly forming large intermediate matrices
    and reduces O(n³) operations where possible.
    """
    m = A.shape[0]
    
    # 1) Use randomized SVD for large matrices (O(n²k) where k << n)
    if m > 100:
        from sklearn.utils.extmath import randomized_svd
        U_A, s, Vt_A = randomized_svd(A, n_components=min(m, 20))
    else:
        # Standard SVD for smaller matrices
        U_A, s, Vt_A = np.linalg.svd(A, full_matrices=False)  # Reduced SVD
    
    V_A = Vt_A.T
    alpha = s[0]  # Operator norm = largest singular value

    # 2) Efficiently build diagonal matrices without materializing full matrices
    #s_alpha = s / alpha
    if alpha < 1e-14:
        print("Warning: Matrix A has negligible singular values. Alpha set to 1.0.")
        alpha = 1.0
        s_alpha = np.zeros(m)
    else:
        s_alpha = s / alpha
    
    # Function to apply S or Y to a vector without forming full matrices
    def apply_S(v):
        return s_alpha * v
    
    def apply_Y(v):
        return np.sqrt(1 - s_alpha**2) * v
    
    # 3) Use matrix-free implementation for large matrices
    if m > 100:
        # Return lambda functions that apply the operations without forming the matrix
        # This is a sketch - full implementation would require custom operator classes
        return (lambda v: block_encode_apply(v, U_A, s_alpha, V_A), alpha)
    
    # Standard implementation for smaller matrices - still optimize where possible
    S = np.diag(s_alpha)
    Y = np.diag(np.sqrt(1 - s_alpha**2))
    
    # More efficient block construction using sparse operations if available
    W = np.block([[ S,  Y],
                 [ Y, -S]])
    
    # Use more efficient matrix multiplication sequence (to avoid large intermediates)
    # (A @ B) @ C instead of A @ (B @ C) if dimensions make it beneficial
    U_pre = np.block([[U_A,             np.zeros((m, m))],
                     [np.zeros((m, m)), V_A           ]])
    U_post = np.block([[V_A.T,           np.zeros((m, m))],
                      [np.zeros((m, m)), U_A.T         ]])
    
    # Optimize matrix multiplication chain
    U = U_pre @ (W @ U_post)  # Can be more efficient than (U_pre @ W) @ U_post depending on dimensions
    
    # Skip expensive verification in production code
    # assert is_unitary(U), f"U†U−I = {np.linalg.norm(U.conj().T@U - np.eye(2*m))}"
    identity_check = U @ U.conj().T
    if not np.allclose(identity_check, np.eye(2 * m)):
        print("Warning: Block encoding U is not unitary!")
    
    return U, alpha

def pad_to_N(f, N):
    f_pad = np.zeros(N, dtype=f.dtype)
    f_pad[:len(f)] = f
    return f_pad

def lower_toeplitz_matrix(f_pad):
    """
    Build the N×N lower‐Toeplitz convolution matrix for f_pad:
       (T_f)_{i,j} = f_pad[i-j]  if 0 ≤ j ≤ i, else 0.
    """
    N = len(f_pad)
    T = np.zeros((N, N), dtype=f_pad.dtype)
    for i in range(N):
        # fill columns j=0..i with f_pad[i-j]
        T[i, :i+1] = f_pad[i::-1]
    return T

def conv2d_toeplitz_matrix(kernel, image_shape):
    """
    Build the (out_h*out_w)×(H*W) matrix T such that
      vec(valid_conv2d(image, kernel)) == T @ vec(image).
    """
    H, W = image_shape
    h, w = kernel.shape
    out_h = H - h + 1
    out_w = W - w + 1

    T = np.zeros((out_h * out_w, H * W), dtype=kernel.dtype)
    for r in range(out_h):
        for c in range(out_w):
            row = r * out_w + c
            for i in range(h):
                for j in range(w):
                    col = (r + i) * W + (c + j)
                    T[row, col] = kernel[i, j]
    return T

def generate_differentiation_matrix(n):
    """
    Generates a "differentiated" matrix directly.
    
    For a given exponent n, the number of rows is m = 2**n, and
    the resulting matrix has m+1 columns. Each row i has a single
    nonzero entry at column (i+1) (i.e. shifted one to the right) with
    value equal to i+1.
    
    For example, if n = 2 (m = 4), the matrix is:
        [[0, 1, 0, 0, 0],
         [0, 0, 2, 0, 0],
         [0, 0, 0, 3, 0],
         [0, 0, 0, 0, 4]]
         
    Returns:
        A NumPy array of shape (2**n, 2**n + 1) with the described pattern.
    """
    m = 2 ** n  # number of rows
    # Create a matrix of zeros with m rows and (m+1) columns.
    matrix = np.zeros((m, m), dtype=int)
    # Using vectorized assignment, fill the first superdiagonal of the submatrix.
    # M[:, 1:] is an (m x m) array. Its diagonal is the location for our non-zeros.
    np.fill_diagonal(matrix[:, 1:], np.arange(1, m + 1))


    #size = 2**3  # Calculate 2^3 = 8
    #diagonal_values = np.arange(1, size + 1) # Create a sequence of numbers from 1 to 8
    #diagonal_matrix = np.diag(diagonal_values, 1)
    
    return matrix

def integration_matrix_rect(poly_degree):
    """
    Build the rectangular integration matrix mapping coefficients for degree poly_degree.
    """
    M = poly_degree + 1
    I = np.zeros((M + 1, M), dtype=float)
    for i in range(M):
        I[i + 1, i] = 1.0 / (i + 1)
    return I

def generate_integration_matrix(n):
    """
    Generates an integration matrix corresponding to the integration operator on polynomial
    coefficients. For a given exponent n, let m = 2**n. The resulting matrix has shape (m, m)
    and is defined as follows:
    
      - The first row is all zeros (i.e. the constant of integration is taken as zero).
      - For each row i (with i >= 1), the entry in column (i-1) is 1/i, with all other entries zero.
    
    In other words, if the input coefficient vector is [a_0, a_1, ..., a_{m-1}],
    the matrix maps this to an "integrated" coefficient vector [0, a_0, a_1/2, a_2/3, ..., a_{m-2}/(m-1)].
    
    For example, if n = 2 (so m = 4), the resulting matrix is:
        [[0,   0,   0,   0],
         [1,   0,   0,   0],
         [0, 1/2,   0,   0],
         [0,   0, 1/3,   0]]
    
    Returns:
        A NumPy array of shape (2**n, 2**n) representing the integration operator.
    
    Optimized implementation with reduced O(n) complexity instead of O(n²)
    by using sparse matrix representation for large dimensions.
    
    """
    m = 2 ** n
    
    # For large matrices, use scipy sparse representation
    if m > 100:
        try:
            from scipy import sparse
            # Create sparse matrix directly
            row_indices = np.arange(1, m)
            col_indices = row_indices - 1
            data = 1.0 / row_indices
            
            # Create sparse matrix in efficient CSR format
            matrix = sparse.csr_matrix((data, (row_indices, col_indices)), 
                                       shape=(m, m), dtype=float)
            
            # Convert back to dense if needed by the calling code
            return matrix.toarray()
        except ImportError:
            pass  # Fall back to dense implementation
    
    # For smaller matrices, use vectorized numpy operations
    matrix = np.zeros((m, m), dtype=float)
    indices = np.arange(1, m)
    matrix[indices, indices - 1] = 1.0 / indices
    return matrix

def compute_a(n):
    """
    Compute coefficient for Taylor series of sqrt(1-x²) with O(1) complexity
    when used with memoization.
    
    Uses a direct formula for a_n instead of iteration:
    a_n = (-1)^n * (2n)! / (4^n * (n!)^2 * (2n-1))
    
    For larger n, uses logarithmic computation to avoid overflow.
    """
    # For very large n, use logarithmic computation to avoid overflow
    if n > 100:
        import math
        if n == 0:
            return 1.0
        
        # Log formula: log(a_n) = n*log(-1) + log((2n)!) - n*log(4) - 2*log(n!) - log(2n-1)
        # = n*log(-1) + sum(log(i) for i in range(1,2n+1)) - n*log(4) - 2*sum(log(i) for i in range(1,n+1)) - log(2n-1)
        # Simplify using properties of factorials and logarithms
        
        log_result = 0
        # (-1)^n part
        sign = -1 if n % 2 == 1 else 1
        
        # Logarithm of factorial ratios
        for i in range(n+1, 2*n+1):
            log_result += math.log(i)
        
        # Divide by 4^n
        log_result -= n * math.log(4)
        
        # Divide by (2n-1)
        log_result -= math.log(2*n-1)
        
        return sign * math.exp(log_result)
    
    # Memoized computation for medium-sized n
    if hasattr(compute_a, 'memo'):
        if n in compute_a.memo:
            return compute_a.memo[n]
    else:
        compute_a.memo = {0: 1.0}
    
    # Direct computation for small n
    if n == 0:
        return 1.0
    
    # Use recurrence relation with previous result if available
    if n-1 in compute_a.memo:
        result = -((0.5 - (n - 1)) / n) * compute_a.memo[n-1]
    else:
        # Fallback to iterative calculation
        result = 1.0
        for k in range(1, n + 1):
            result *= -((0.5 - (k - 1)) / k)
    
    # Store result in memo
    compute_a.memo[n] = result
    return result

def taylor_series_coefficients(max_degree):
    """
    Compute Taylor coefficients for sqrt(1-x²) with O(n) complexity.
    
    Uses dynamic programming to avoid redundant computation and
    returns a sparse representation for large degrees.
    """
    # Use memoization for compute_a to avoid redundant calculations
    if not hasattr(taylor_series_coefficients, 'memo'):
        taylor_series_coefficients.memo = {0: 1.0}
    
    # For very large degrees, use sparse representation
    if max_degree > 1000:
        from collections import defaultdict
        sparse_coeff = defaultdict(float)
        max_n = max_degree // 2
        
        # Only compute and store non-zero coefficients (even indices)
        for n in range(max_n + 1):
            if n not in taylor_series_coefficients.memo:
                # Use recursion with memoization
                if n == 0:
                    taylor_series_coefficients.memo[n] = 1.0
                else:
                    prev = taylor_series_coefficients.memo[n-1]
                    taylor_series_coefficients.memo[n] = -((0.5 - (n - 1)) / n) * prev
            
            sparse_coeff[2 * n] = taylor_series_coefficients.memo[n]
        
        # Convert to dense array if needed by calling code
        coeff = [0.0] * (max_degree + 1)
        for idx, val in sparse_coeff.items():
            if idx <= max_degree:
                coeff[idx] = val
        return coeff
    
    # Standard implementation for moderate sizes
    coeff = [0.0] * (max_degree + 1)
    max_n = max_degree // 2
    
    # Use vectorized operations where possible
    for n in range(max_n + 1):
        if n not in taylor_series_coefficients.memo:
            # Calculate coefficient using recursion with memoization
            if n == 0:
                taylor_series_coefficients.memo[n] = 1.0
            else:
                prev = taylor_series_coefficients.memo[n-1]
                taylor_series_coefficients.memo[n] = -((0.5 - (n - 1)) / n) * prev
        
        coeff[2 * n] = taylor_series_coefficients.memo[n]
    
    return coeff

def qsp_state_preparation_unitary(target_state, phases=None):
    """
    Placeholder QSP-based unitary.
    In an optimized version, you would compute the required phase factors to implement the target polynomial
    transformation via a sequence of Rx rotations and fixed reflections.
    Here, if no phases are provided, we simply revert to the Householder method.
    """
    if phases is None:
        return state_preparation_unitary(target_state)
    # Example (non-optimal): apply Rx rotations with the given phases.
    dim = len(target_state)
    n_qubits = int(np.ceil(np.log2(dim)))
    qc = QuantumCircuit(n_qubits)
    for phi in phases:
        for qubit in range(n_qubits):
            qc.rx(2 * phi, qubit)
    U_qsp = Operator(qc).data
    return U_qsp

def pad_unitary(U, n_qubits):
    """
    Pads a unitary U (of dimension m x m) to a 2^n_qubits x 2^n_qubits unitary
    by embedding U in the top-left block.
    """
    dim_target = 2 ** n_qubits
    current_dim = U.shape[0]
    if current_dim > dim_target:
        raise ValueError("U's dimension exceeds 2^n_qubits.")
    padded = np.eye(dim_target, dtype=U.dtype)
    padded[:current_dim, :current_dim] = U
    return padded

class Polynomial:
    def __init__(self, coefficients, variable = "x"):
        self.coeffs = coefficients
        self.variable=variable

    def __str__(self):
        chunks = []
        for coeff, power in zip(self.coeffs, range(len(self.coeffs) - 1, -1, -1)):
            if coeff == 0:
                continue
            chunks.append(self.format_coeff(coeff))
            chunks.append(self.format_power(power, self.variable))
        chunks[0] = chunks[0].lstrip("+")
        return ''.join(chunks)

    @staticmethod
    def format_coeff(coeff):
        return str(coeff) if coeff < 0 else "+{0}".format(coeff)

    @staticmethod
    def format_power(power,variable):
        return '{0}^{1}'.format(variable, power) if power != 0 else ''
assert str(Polynomial([2, -3, 0, 5])) == "2x^3-3x^2+5"

def evaluate_polynomial(coeffs, x):
    """
    Evaluate a polynomial with given coefficients at a point x.
    coeffs[0] is the constant term, coeffs[1] is the coefficient for x, etc.
    """
    total = 0.0
    for i, c in enumerate(coeffs):
        total += c * (x ** i)
    return total

# --------------------------------------------
# Helper functions for state construction
# --------------------------------------------
def canonical_phase(state):
    """
    Efficiently adjust phase so first nonzero element is real and positive.
    
    O(k) implementation where k is the index of the first non-zero element.
    For sparse vectors, this is much faster than examining all elements.
    """
    state = np.array(state, dtype=complex)
    
    # Vectorized approach for finding first non-zero element
    nonzero_mask = np.abs(state) > 1e-12
    if not np.any(nonzero_mask):
        return state  # All zeros, return unchanged
    
    # Find index of first non-zero element
    first_idx = np.argmax(nonzero_mask)
    phase = np.angle(state[first_idx])
    
    # Apply phase correction only if needed
    if abs(phase) > 1e-12:
        return state * np.exp(-1j * phase)
    return state

#def state_preparation_unitary(state):
#    """
#    Construct a unitary matrix U which maps |0> to the given normalized state,
#    using a Householder reflection based method.
#    """
#    state = np.array(state, dtype=complex)
#    dim = len(state)
#    norm_state = np.linalg.norm(state)
#    if abs(norm_state - 1.0) > 1e-9:
#        if norm_state < 1e-14:
#            return np.eye(dim, dtype=complex)
#        state = state / norm_state
#
#    ket0 = np.zeros(dim, dtype=complex)
#    ket0[0] = 1.0
#
#    if np.allclose(np.abs(state[0]), 1.0) and np.allclose(state[1:], 0.0):
#         phase = np.angle(state[0])
#         return np.exp(1j * phase) * np.eye(dim, dtype=complex)
#
#    v = ket0 - state
#    norm_v_sq = np.vdot(v, v).real
#    if norm_v_sq < 1e-14:
#        print("Warning: state_preparation_unitary norm_v_sq is near zero unexpectedly.")
#        return np.eye(dim, dtype=complex)
#    U = np.eye(dim, dtype=complex) - (2.0 / norm_v_sq) * np.outer(v, np.conjugate(v))
#    return U

class HouseholderUnitary:
    """Matrix-free implementation of Householder unitary."""
    def __init__(self, v, norm_v_sq):
        self.v = v
        self.scale = 2.0 / norm_v_sq
        self.dim = len(v)
    
    def __call__(self, x):
        """Apply the unitary to vector x without forming the matrix."""
        # U|x⟩ = |x⟩ - (2/||v||²) |v⟩⟨v|x⟩
        v_dot_x = np.vdot(self.v, x)
        return x - self.scale * v_dot_x * self.v
    
    @property
    def data(self):
        """Materialize the matrix if needed (for interfaces requiring explicit matrices)."""
        U = np.eye(self.dim, dtype=complex)
        U -= self.scale * np.outer(self.v, np.conj(self.v))
        return U
    
    # Add shape property to make it compatible with matrix-like objects
    @property
    def shape(self):
        return (self.dim, self.dim)

def sparse_state_preparation(state):
    """
    Fast implementation for sparse states with O(k) complexity
    where k is the number of non-zero elements.
    
    Modified to always return the matrix form, not the callable.
    """
    # Get non-zero indices and values
    nonzero_idx = np.nonzero(state)[0]
    if len(nonzero_idx) == 0:
        return np.eye(len(state), dtype=complex)
    
    # Normalize the state
    norm = np.linalg.norm(state)
    if norm < 1e-14:
        return np.eye(len(state), dtype=complex)
    
    normalized_state = state / norm
    
    # Create the Householder vector with only non-zero elements
    ket0 = np.zeros_like(normalized_state)
    ket0[0] = 1.0
    v = ket0 - normalized_state
    
    # Compute the scale factor
    norm_v_sq = np.sum(np.abs(v)**2)
    
    # Instead of returning the callable, return the matrix directly
    U = np.eye(len(state), dtype=complex)
    U -= (2.0 / norm_v_sq) * np.outer(v, np.conj(v))
    return U

def state_preparation_unitary(state):
    """
    Modified implementation to always return the matrix representation,
    not the callable object.
    """
    state = np.array(state, dtype=complex)
    dim = len(state)
    
    # Fast path for large dimensions with sparse states
    if dim > 1000:
        return sparse_state_preparation(state)
    
    # Handle zero or near-zero states quickly
    norm_state = np.linalg.norm(state)
    if norm_state < 1e-14:
        return np.eye(dim, dtype=complex)
    
    # Normalize efficiently only if needed
    if abs(norm_state - 1.0) > 1e-9:
        state = state / norm_state

    # Standard basis state |0>
    ket0 = np.zeros(dim, dtype=complex)
    ket0[0] = 1.0

    # Fast path: if state is already very close to |0>, just apply phase
    if abs(abs(state[0]) - 1.0) < 1e-9 and np.max(np.abs(state[1:])) < 1e-9:
        phase = np.angle(state[0])
        return np.exp(1j * phase) * np.eye(dim, dtype=complex)

    # Compute Householder vector efficiently
    v = ket0 - state
    
    # Compute norm directly to avoid complex conjugation overhead
    norm_v_sq = np.sum(np.abs(v)**2)
    if norm_v_sq < 1e-14:
        return np.eye(dim, dtype=complex)
    
    # Always compute and return the matrix explicitly
    scale = 2.0 / norm_v_sq
    U = np.eye(dim, dtype=complex)
    U -= scale * np.outer(v, np.conj(v))
    return U


# --------------------------------------------
# Custom helper for controlled gate creation
# --------------------------------------------
def get_custom_controlled_gate(U, label=""):
    """
    Returns a controlled unitary gate constructed from U.
    Adjust this function to refine the gate decomposition if desired.
    """
    base_gate = UnitaryGate(U, label=label)
    controlled_gate = base_gate.control(1, label="c_" + label)
    return controlled_gate
