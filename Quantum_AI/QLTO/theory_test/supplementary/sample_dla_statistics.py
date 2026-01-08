import numpy as np
import matplotlib.pyplot as plt
import os

# Simple simulation of Dynamical Lie Algebra (DLA) operator growth
# This toy model mimics the "operator sparsity" metric described in the plan.
# It randomly samples nested commutators and records the number of Pauli terms.

def random_pauli_term(num_qubits):
    """Return a random Pauli string as a list of characters (I, X, Y, Z)."""
    paulis = ['I', 'X', 'Y', 'Z']
    return [np.random.choice(paulis) for _ in range(num_qubits)]

def multiply_terms(term1, term2):
    """Multiply two Pauli strings term‑wise, ignoring global phases.
    The product of two Paulis is another Pauli (up to a phase).
    """
    result = []
    for p1, p2 in zip(term1, term2):
        if p1 == 'I':
            result.append(p2)
        elif p2 == 'I':
            result.append(p1)
        elif p1 == p2:
            result.append('I')  # X*X = I, etc.
        else:
            # X*Y = iZ, Y*Z = iX, Z*X = iY (phase ignored)
            result.append({"X", "Y", "Z"}.difference({p1, p2}).pop())
    return result

def simulate_operator_growth(num_qubits, depth, trials=1000):
    """Simulate growth of a single operator by applying random nested commutators.
    Returns a list of term counts (size of the operator expressed as a sum of Pauli strings).
    """
    counts = []
    for _ in range(trials):
        # start with a single random Pauli term
        current_terms = [random_pauli_term(num_qubits)]
        for _ in range(depth):
            gen = random_pauli_term(num_qubits)
            new_terms = []
            for term in current_terms:
                prod = multiply_terms(term, gen)
                new_terms.append(prod)          # A*B
                new_terms.append(term)          # -B*A approximated by keeping original
            # deduplicate
            unique = {tuple(t) for t in new_terms}
            current_terms = [list(t) for t in unique]
        counts.append(len(current_terms))
    return counts

def main():
    num_qubits = 7
    depth_ordered = 3   # shallow depth → polynomial growth
    depth_chaotic = 7   # deeper → exponential growth (matched to N)
    trials = 5000       # increased for better resolution

    ordered_counts = simulate_operator_growth(num_qubits, depth_ordered, trials)
    chaotic_counts = simulate_operator_growth(num_qubits, depth_chaotic, trials)

    # Plot histograms
    plt.figure(figsize=(8, 5))
    max_bin = max(max(ordered_counts), max(chaotic_counts))
    bins = np.arange(0, max_bin + 2) - 0.5
    plt.hist(ordered_counts, bins=bins, alpha=0.7, label='Ordered (Polynomial)', color='tab:blue')
    plt.hist(chaotic_counts, bins=bins, alpha=0.7, label='Chaotic (Exponential)', color='tab:red')
    plt.xlabel('Number of Pauli Terms in Operator')
    plt.ylabel('Frequency')
    plt.title(f'DLA Operator Sparsity Statistics (N = {num_qubits} qubits)')
    plt.legend()
    out_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..', 'Quantum_AI', 'QLTO', 'paper'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dla_statistics.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print('=== DLA STRUCTURE STATISTICS (RMT TEST) ===')
    print('System Size N =', num_qubits)
    print('Mean Pauli Terms (Ordered): %.2f' % np.mean(ordered_counts))
    print('Mean Pauli Terms (Chaotic): %.2f' % np.mean(chaotic_counts))
    print(f"[Output] Saved statistics to '{out_path}'")

if __name__ == '__main__':
    main()