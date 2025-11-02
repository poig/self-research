"""
Verify QAOA Solution Correctness
================================

QAOA found: {3: True, 2: False, 1: True}
Expected:   {1: True, 2: True, 3: True}

Let's check if QAOA's solution is VALID (just different)
"""

def evaluate_clause(clause, assignment):
    """Check if clause is satisfied"""
    for lit in clause:
        var = abs(lit)
        val = assignment.get(var, False)
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return False


# Problem: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
clauses = [
    (1, 2),      # x1 ∨ x2
    (-1, 3),     # ¬x1 ∨ x3
    (-2, -3)     # ¬x2 ∨ ¬x3
]

# QAOA solution
qaoa_solution = {1: True, 2: False, 3: True}

# Expected solution
expected_solution = {1: True, 2: True, 3: True}

print("="*80)
print("SOLUTION VERIFICATION")
print("="*80)

print("\nQAOA Solution: x1=T, x2=F, x3=T")
print("-" * 40)
for i, clause in enumerate(clauses, 1):
    satisfied = evaluate_clause(clause, qaoa_solution)
    print(f"Clause {i} {clause}: {'✅ SAT' if satisfied else '❌ UNSAT'}")
    
    # Show evaluation
    if clause == (1, 2):
        print(f"  x1={qaoa_solution[1]} ∨ x2={qaoa_solution[2]}")
        print(f"  {qaoa_solution[1]} ∨ {qaoa_solution[2]} = {qaoa_solution[1] or qaoa_solution[2]}")
    elif clause == (-1, 3):
        print(f"  ¬x1={not qaoa_solution[1]} ∨ x3={qaoa_solution[3]}")
        print(f"  {not qaoa_solution[1]} ∨ {qaoa_solution[3]} = {not qaoa_solution[1] or qaoa_solution[3]}")
    elif clause == (-2, -3):
        print(f"  ¬x2={not qaoa_solution[2]} ∨ ¬x3={not qaoa_solution[3]}")
        print(f"  {not qaoa_solution[2]} ∨ {not qaoa_solution[3]} = {not qaoa_solution[2] or not qaoa_solution[3]}")

qaoa_valid = all(evaluate_clause(c, qaoa_solution) for c in clauses)

print("\n" + "="*80)
print(f"QAOA Solution: {'✅ VALID' if qaoa_valid else '❌ INVALID'}")
print("="*80)

print("\n\nExpected Solution: x1=T, x2=T, x3=T")
print("-" * 40)
for i, clause in enumerate(clauses, 1):
    satisfied = evaluate_clause(clause, expected_solution)
    print(f"Clause {i} {clause}: {'✅ SAT' if satisfied else '❌ UNSAT'}")
    
    # Show evaluation
    if clause == (1, 2):
        print(f"  x1={expected_solution[1]} ∨ x2={expected_solution[2]}")
        print(f"  {expected_solution[1]} ∨ {expected_solution[2]} = {expected_solution[1] or expected_solution[2]}")
    elif clause == (-1, 3):
        print(f"  ¬x1={not expected_solution[1]} ∨ x3={expected_solution[3]}")
        print(f"  {not expected_solution[1]} ∨ {expected_solution[3]} = {not expected_solution[1] or expected_solution[3]}")
    elif clause == (-2, -3):
        print(f"  ¬x2={not expected_solution[2]} ∨ ¬x3={not expected_solution[3]}")
        print(f"  {not expected_solution[2]} ∨ {not expected_solution[3]} = {not expected_solution[2] or not expected_solution[3]}")

expected_valid = all(evaluate_clause(c, expected_solution) for c in clauses)

print("\n" + "="*80)
print(f"Expected Solution: {'✅ VALID' if expected_valid else '❌ INVALID'}")
print("="*80)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if qaoa_valid and expected_valid:
    print("✅ Both solutions are VALID!")
    print("   SAT problems can have MULTIPLE valid solutions.")
    print("   QAOA found a DIFFERENT but CORRECT solution.")
    print("\n   This is NOT an error - it's expected behavior!")
elif qaoa_valid:
    print("✅ QAOA solution is VALID")
    print("❌ Expected solution is INVALID")
elif expected_valid:
    print("❌ QAOA solution is INVALID")
    print("✅ Expected solution is VALID")
else:
    print("❌ Both solutions are INVALID")
