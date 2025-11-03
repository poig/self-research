#!/usr/bin/env python
"""
check_quantum_setup.py

Verify all dependencies for quantum SAT certification
"""

import sys
import importlib

def check_package(package_name, import_path=None, display_name=None):
    """Check if package is installed"""
    display = display_name or package_name
    try:
        if import_path:
            exec(f"from {package_name} import {import_path}")
        else:
            mod = importlib.import_module(package_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {display:<30} version: {version}")
            return True
    except ImportError as e:
        print(f"❌ {display:<30} {e}")
        return False
    except Exception as e:
        print(f"⚠️  {display:<30} Error: {e}")
        return False
    
    print(f"✅ {display:<30}")
    return True

print("="*70)
print("     QUANTUM SAT CERTIFICATION - DEPENDENCY CHECK")
print("="*70)
print()

results = {}

# Core Python packages
print("1. Core Scientific Packages:")
results['numpy'] = check_package('numpy')
results['scipy'] = check_package('scipy')
results['networkx'] = check_package('networkx')
results['sklearn'] = check_package('sklearn', display_name='scikit-learn')
print()

# Quantum Computing
print("2. Quantum Computing Frameworks:")
results['qiskit'] = check_package('qiskit')
results['qiskit_aer'] = check_package('qiskit_aer', display_name='qiskit-aer')
results['qiskit.quantum_info'] = check_package('qiskit.quantum_info', 'Statevector')
print()

# Quantum Information Theory
print("3. Quantum Information Theory:")
results['cvxpy'] = check_package('cvxpy')
results['toqito'] = check_package('toqito')
if results.get('toqito', False):
    results['toqito.state_props'] = check_package('toqito.state_props', 'is_separable')
print()

# Local QLTO/QAADO
print("4. Local Quantum Optimization Libraries:")
sys.path.insert(0, 'C:/Users/junli/self-research/Quantum_AI/QLTO')
sys.path.insert(0, 'C:/Users/junli/self-research/Quantum_AI/qaado')

results['qlto_nisq'] = check_package('qlto_nisq')
if results.get('qlto_nisq', False):
    try:
        from qlto_nisq import run_qlto_nisq_optimizer, get_vqe_ansatz
        print(f"   → run_qlto_nisq_optimizer: ✅")
        print(f"   → get_vqe_ansatz: ✅")
    except ImportError as e:
        print(f"   → Function import failed: {e}")
        results['qlto_nisq'] = False

results['qaado'] = check_package('qng_qaado_fusion', display_name='qaado (qng_qaado_fusion)')
print()

# Classical SAT Framework
print("5. Classical SAT Framework:")
try:
    from sat_decompose import SATDecomposer, create_test_sat_instance
    print(f"✅ sat_decompose.py              Available")
    results['sat_decompose'] = True
except ImportError as e:
    print(f"❌ sat_decompose.py              {e}")
    results['sat_decompose'] = False

try:
    from sat_undecomposable import HardnessCertificate
    print(f"✅ sat_undecomposable.py         Available (classical)")
    results['sat_undecomposable'] = True
except ImportError as e:
    print(f"❌ sat_undecomposable.py         {e}")
    results['sat_undecomposable'] = False
print()

# Summary
print("="*70)
print("                          SUMMARY")
print("="*70)

total = len(results)
passed = sum(results.values())
percent = 100 * passed / total

print(f"\nPassed: {passed}/{total} ({percent:.0f}%)")
print()

# Determine what's possible
quantum_full = all([
    results.get('qiskit', False),
    results.get('qiskit_aer', False),
    results.get('qlto_nisq', False),
    results.get('toqito', False),
    results.get('cvxpy', False)
])

quantum_partial = all([
    results.get('qiskit', False),
    results.get('qiskit_aer', False),
    results.get('qlto_nisq', False)
])

classical_only = results.get('sat_decompose', False)

if quantum_full:
    print("🎉 STATUS: FULL QUANTUM CERTIFICATION READY!")
    print()
    print("You can run:")
    print("   python sat_undecomposable_quantum.py")
    print()
    print("Features enabled:")
    print("   ✅ QLTO-VQE ground state optimization")
    print("   ✅ Quantum entanglement analysis (Qiskit)")
    print("   ✅ Separability testing (toqito)")
    print("   ✅ Provably correct certification (99.9% confidence)")
    
elif quantum_partial:
    print("🔬 STATUS: PARTIAL QUANTUM CERTIFICATION")
    print()
    print("You can run quantum certification without toqito:")
    print("   python sat_undecomposable_quantum.py")
    print()
    print("Features enabled:")
    print("   ✅ QLTO-VQE ground state optimization")
    print("   ✅ Quantum entanglement analysis (Qiskit)")
    print("   ⚠️  Separability testing (toqito) - NOT AVAILABLE")
    print()
    print("To enable full features:")
    print("   pip install toqito cvxpy")
    
elif classical_only:
    print("📊 STATUS: CLASSICAL CERTIFICATION ONLY")
    print()
    print("Quantum libraries not available. Install with:")
    print("   pip install qiskit qiskit-aer toqito cvxpy")
    print()
    print("You can still run classical certification:")
    print("   python sat_undecomposable.py")
    
else:
    print("❌ STATUS: MISSING CRITICAL DEPENDENCIES")
    print()
    print("Please install:")
    print("   pip install qiskit qiskit-aer toqito cvxpy numpy scipy networkx scikit-learn")

print("="*70)

# Missing packages
missing = [k for k, v in results.items() if not v]
if missing:
    print()
    print("Missing packages:")
    for pkg in missing:
        if pkg == 'qlto_nisq':
            print(f"   • {pkg} - Check: C:/Users/junli/self-research/Quantum_AI/QLTO/")
        elif pkg == 'qaado':
            print(f"   • {pkg} - Check: C:/Users/junli/self-research/Quantum_AI/qaado/")
        elif pkg == 'sat_decompose':
            print(f"   • {pkg} - Should be in current directory")
        else:
            print(f"   • {pkg} - Install: pip install {pkg.replace('_', '-')}")
    print("="*70)

print()
