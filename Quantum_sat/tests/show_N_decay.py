# show_N_decay.py

import numpy as np

def show_N_decay():
    """
    Show how success rate decays with N (and n_partitions).
    """
    print("SUCCESS RATE DECAY vs. N and n_partitions")
    print("="*50)

    k_star = 10 # A tractable k*

    depth = 15
    n_basins = 10
    n_iterations = 1000
    
    search_volume = depth * n_basins * n_iterations
    
    print(f"Fixing k* = {k_star}")
    print(f"With search volume = {search_volume}")
    print("-" * 50)
    print("N (vars) | Partitions | Prob/Partition | Overall Success")
    print("-" * 50)

    for n_vars in range(100, 13001, 500):
        n_partitions = max(1, n_vars // (k_star + 1))
        
        problem_complexity = k_star * n_vars
        
        exponent = -2.0 * search_volume / problem_complexity
        prob_per_partition = 1.0 - np.exp(exponent)
        
        overall_success = prob_per_partition ** n_partitions
        
        print(f"{n_vars:<8} | {n_partitions:<10} | {prob_per_partition*100:<14.2f}% | {overall_success*100:.2f}%")

if __name__ == "__main__":
    show_N_decay()
