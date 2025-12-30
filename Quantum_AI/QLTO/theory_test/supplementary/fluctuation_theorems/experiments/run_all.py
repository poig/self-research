"""
Run All Paper 4 Experiments (Fixed)
====================================

Master script to run all 6 corrected experiments.
"""

import time

Max_N = 8

def run_all():
    """Run all experiments."""
    print("╔" + "═" * 60 + "╗")
    print("║" + "  PAPER 4: EXPERIMENTAL SUITE (FIXED)  ".center(60) + "║")
    print("║" + "  Quantum Thermodynamics of VQA  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    
    total_start = time.time()
    
    experiments = [
        ("1: N-Sweep (Efficiency Collapse)", "exp01_n_sweep", "run_n_sweep", {"N_range": (2, Max_N)}),
        ("2: Jarzynski Equality", "exp02_jarzynski", "run_jarzynski_test", {"N_values": [i for i in range(2, Max_N + 1)]}),
        ("3: Holevo Bottleneck (Honest)", "exp03_holevo", "run_holevo_test", {"N_values": [i for i in range(2, Max_N + 1)]}),
        ("4: Constitutive Breakdown", "exp04_free_energy", "run_breakdown_test", {"N_values": [i for i in range(3, Max_N + 1)]}),
        ("5: Scrambling Rate", "exp05_mss", "run_scrambling_test", {"N_values": [i for i in range(3, Max_N + 1)]}),
        ("6: Second Law", "exp06_noise", "run_second_law_test", {"N_values": [i for i in range(3, Max_N + 1)]}),
    ]
    
    results = {}
    
    for name, module_name, func_name, kwargs in experiments:
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print("="*60)
        
        start = time.time()
        
        try:
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            results[name] = func(**kwargs)
            
            elapsed = time.time() - start
            print(f"\n✓ {name} completed in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║" + "  ALL EXPERIMENTS COMPLETE  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"\nTotal time: {total_time:.1f}s")
    
    print("\nFigures saved:")
    print("  01_n_sweep.png    - Efficiency collapse with transition marker")
    print("  02_jarzynski.png  - Jarzynski verification (clean)")
    print("  03_holevo.png     - Information bottleneck (honest DLA)")
    print("  04_breakdown.png  - W vs I scatter showing law breakdown")
    print("  05_scrambling.png - Scrambling rate λ vs efficiency η")
    print("  06_second_law.png - Second law verification")
    
    return results


if __name__ == "__main__":
    run_all()
