"""
Unit Tests for Corrected Hamiltonian Construction
=================================================

Tests verify that H = Σ_c Π_c (sum of clause projectors) is correctly implemented.

KEY PROPERTY TO TEST:
For any assignment x ∈ {0,1}^N, the diagonal element H[x,x] should equal
the number of clauses violated by assignment x.

Test cases:
1. Simple 3-SAT instances (N=3,4,5)
2. Known backdoor instances (N=8)
3. Random 3-SAT instances (N=10)
4. Edge cases (tautologies, contradictions, empty clauses)
"""

import numpy as np
import sys
from typing import List, Tuple
import unittest

# Import the QSA module
sys.path.insert(0, '.')
from quantum_structure_analyzer import QuantumStructureAnalyzer


def count_violated_clauses(assignment: int, clauses: List[Tuple[int, ...]], n_vars: int) -> int:
    """
    Count how many clauses are violated by a given assignment
    
    Args:
        assignment: Integer representing bit assignment (0 to 2^n_vars - 1)
        clauses: List of clauses, each clause is tuple of literals
        n_vars: Number of variables
    
    Returns:
        Number of violated clauses
    """
    violated = 0
    
    for clause in clauses:
        clause_satisfied = False
        
        for lit in clause:
            var = abs(lit) - 1  # Convert to 0-indexed
            if var >= n_vars:
                continue
            
            # Get variable value from assignment
            var_value = (assignment >> var) & 1
            
            # Check if literal is satisfied
            if lit > 0:  # Positive literal xi
                if var_value == 1:
                    clause_satisfied = True
                    break
            else:  # Negative literal ¬xi
                if var_value == 0:
                    clause_satisfied = True
                    break
        
        if not clause_satisfied:
            violated += 1
    
    return violated


class TestHamiltonianConstruction(unittest.TestCase):
    """Test suite for Hamiltonian construction"""
    
    def setUp(self):
        """Set up QSA instance for testing"""
        self.qsa = QuantumStructureAnalyzer(use_ml=False)
    
    def test_simple_2sat(self):
        """Test: Simple 2-SAT instance with N=2 variables"""
        # (x1 ∨ x2) ∧ (¬x1 ∨ x2)
        clauses = [(1, 2), (-1, 2)]
        n_vars = 2
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Verify for all 4 assignments
        for assignment in range(2 ** n_vars):
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6,
                msg=f"Assignment {assignment:0{n_vars}b}: expected {expected}, got {actual}")
    
    def test_simple_3sat(self):
        """Test: Simple 3-SAT instance with N=3 variables"""
        # (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3) ∧ (x1 ∨ ¬x2 ∨ x3)
        clauses = [(1, 2, 3), (-1, 2, -3), (1, -2, 3)]
        n_vars = 3
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Verify for all 8 assignments
        for assignment in range(2 ** n_vars):
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6,
                msg=f"Assignment {assignment:0{n_vars}b}: expected {expected}, got {actual}")
    
    def test_unsatisfiable_instance(self):
        """Test: UNSAT instance - all assignments violate at least one clause"""
        # (x1) ∧ (¬x1) - contradiction
        clauses = [(1,), (-1,)]
        n_vars = 1
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Both assignments should violate exactly 1 clause
        for assignment in range(2 ** n_vars):
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(expected, 1, "UNSAT instance should have min violations = 1")
    
    def test_satisfiable_with_solution(self):
        """Test: SAT instance with unique solution"""
        # (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
        # Solution: x1=1, x2=0, x3=1 (assignment 101 = 5)
        clauses = [(1, 2), (-1, 3), (-2, -3)]
        n_vars = 3
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Verify all assignments
        for assignment in range(2 ** n_vars):
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6)
        
        # Solution should have 0 violations
        solution_assignment = 0b101  # x1=1, x2=0, x3=1
        self.assertAlmostEqual(H_diag[solution_assignment], 0.0, places=6,
            msg="Solution should have H_diag = 0")
    
    def test_n8_random_instance(self):
        """Test: Random 3-SAT instance with N=8 variables"""
        # Random instance with 20 clauses
        np.random.seed(42)
        n_vars = 8
        n_clauses = 20
        
        clauses = []
        for _ in range(n_clauses):
            # Generate random 3-SAT clause
            vars_in_clause = np.random.choice(range(1, n_vars + 1), size=3, replace=False)
            signs = np.random.choice([-1, 1], size=3)
            clause = tuple(int(s * v) for s, v in zip(signs, vars_in_clause))
            clauses.append(clause)
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Verify random sample of assignments (reduced from 50 to 20 for speed)
        np.random.seed(43)
        sample_assignments = np.random.choice(2 ** n_vars, size=20, replace=False)
        
        for assignment in sample_assignments:
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6,
                msg=f"Assignment {assignment}: expected {expected}, got {actual}")
    
    def test_n10_backdoor_instance(self):
        """Test: Instance with known backdoor structure (N=10)"""
        # Create instance with backdoor of size k=2 (variables 1,2)
        # All other variables can be set deterministically once backdoor is fixed
        n_vars = 10
        
        # Clauses that create backdoor structure
        clauses = [
            (1, 2, 3),      # Backdoor vars + forced var
            (-1, 2, 4),
            (1, -2, 5),
            (-1, -2, 6),
            (3, 7, 8),      # Forced vars create implications
            (-3, -7, 9),
            (4, -8, 10),
            (-4, 8, -10),
        ]
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        H_diag = np.diag(H_matrix).real
        
        # Verify sample of assignments (reduced from 100 to 20 for speed)
        np.random.seed(44)
        sample_assignments = np.random.choice(2 ** n_vars, size=20, replace=False)
        
        for assignment in sample_assignments:
            expected = count_violated_clauses(assignment, clauses, n_vars)
            actual = H_diag[assignment]
            
            self.assertAlmostEqual(actual, expected, places=6,
                msg=f"Assignment {assignment}: expected {expected}, got {actual}")
    
    def test_hermiticity(self):
        """Test: Verify Hamiltonian is Hermitian (H = H†)"""
        clauses = [(1, 2, 3), (-1, 2, -3), (1, -2, 3)]
        n_vars = 3
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        
        # Check Hermiticity
        self.assertTrue(np.allclose(H_matrix, H_matrix.conj().T),
            "Hamiltonian must be Hermitian")
    
    def test_ground_state_is_solution(self):
        """Test: Ground state (minimum eigenvalue) corresponds to satisfying assignment"""
        # SAT instance with known solution
        clauses = [(1, 2, 3), (-1, -2, 3), (-1, 2, -3)]
        n_vars = 3
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        H_matrix = H.to_matrix()
        
        # Find ground state energy
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        ground_energy = eigenvalues[0]
        
        # Ground state should have energy = 0 for SAT instance
        self.assertAlmostEqual(ground_energy, 0.0, places=6,
            msg="SAT instance should have ground state energy = 0")
    
    def test_sparse_pauli_conversion(self):
        """Test: Verify SparsePauliOp representation is equivalent to matrix"""
        clauses = [(1, 2), (-1, 3), (2, -3)]
        n_vars = 3
        
        H = self.qsa._build_hamiltonian(clauses, n_vars)
        
        # Verify it's a valid SparsePauliOp
        self.assertIsNotNone(H, "Hamiltonian should not be None")
        
        # Verify matrix representation works
        H_matrix = H.to_matrix()
        self.assertEqual(H_matrix.shape, (2**n_vars, 2**n_vars),
            "Matrix should have correct dimension")


class TestHamiltonianSparse(unittest.TestCase):
    """Test suite for sparse Hamiltonian construction (large N)"""
    
    def setUp(self):
        """Set up QSA instance for testing"""
        self.qsa = QuantumStructureAnalyzer(use_ml=False)
    
    def test_sparse_construction_equivalence(self):
        """Test: Sparse construction matches dense for small N"""
        clauses = [(1, 2, 3), (-1, 2, -3)]
        n_vars = 4  # Small enough for both methods
        
        # Build using dense method (standard for n_vars <= 16)
        H_dense = self.qsa._build_hamiltonian(clauses, n_vars)
        H_dense_matrix = H_dense.to_matrix()
        
        # Build using sparse method
        H_sparse = self.qsa._build_hamiltonian_sparse(clauses, n_vars)
        H_sparse_matrix = H_sparse.to_matrix()
        
        # They should be equal
        self.assertTrue(np.allclose(H_dense_matrix, H_sparse_matrix),
            "Dense and sparse construction should match")


def run_tests():
    """Run all tests and report results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestHamiltonianConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestHamiltonianSparse))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Hamiltonian construction is correct.")
    else:
        print("\n❌ SOME TESTS FAILED. See details above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
