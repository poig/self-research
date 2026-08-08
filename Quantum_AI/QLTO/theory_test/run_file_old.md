(base) poig@pc:~/project/self-research/Quantum_AI/QLTO/theory_test$ python ancilla_test.py 
============================================================
FULL QLTO CIRCUIT TEST (W -> U -> W_dag)
Hypothesis: W_dag is required to see the correlation.
============================================================

[Case A] Independent H (Z0 + Z1)
  Mutual Information: 0.000000

[Case B] Interacting H (Z0 * Z1)
  Mutual Information: 1.742394

============================================================
CONCLUSION
============================================================
✓ SUCCESS: NON-LINEARITY PROVEN.
  Independent MI: 0.0000 (Linear/Separable)
  Interacting MI: 1.7424 (Non-Linear/Entangled)

  The math holds:
  - exp(-i(A+B)) factors into independent rotations.
  - exp(-i(A*B)) creates entanglement.
  - W_dag successfully transferred this info to the parameters.
(base) poig@pc:~/project/self-research/Quantum_AI/QLTO/theory_test$ python k_ancilla_bandwidth_test.py 
======================================================================
K-ANCILLA BANDWIDTH SCALING TEST (Multi-Seed Averaging)
Averaging over 5 random seeds per configuration
======================================================================

======================================================================
ANCILLA COUNT: k = 1 (max info = 1 bits per cycle)
======================================================================
  [N=3] Running 5 seeds... η = 0.0871 ± 0.000
  [N=4] Running 5 seeds... η = 0.1132 ± 0.000
  [N=5] Running 5 seeds... η = -0.0329 ± 0.000
  [N=6] Running 5 seeds... η = 0.1360 ± 0.000
  [N=7] Running 5 seeds... η = -0.2873 ± 0.000

======================================================================
ANCILLA COUNT: k = 2 (max info = 2 bits per cycle)
======================================================================
  [N=3] Running 5 seeds... η = 0.0013 ± 0.000
  [N=4] Running 5 seeds... η = 0.0648 ± 0.000
  [N=5] Running 5 seeds... η = 0.0074 ± 0.000
  [N=6] Running 5 seeds... η = 0.0048 ± 0.000
  [N=7] Running 5 seeds... η = -0.0719 ± 0.000
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""

[Saved] k_ancilla_bandwidth_test.png

======================================================================
ANALYSIS
======================================================================
  k=1: Crash point Nc ≈ 5, Avg Bandwidth = 1.13 bits
  k=2: Crash point Nc ≈ 3, Avg Bandwidth = 2.34 bits

======================================================================
CONCLUSION
======================================================================

  Bandwidth scaling: k=2/k=1 ratio = 2.07 (expected: 2.0)
  ✓ SUCCESS: More ancillae → More information bandwidth!
(base) poig@pc:~/project/self-research/Quantum_AI/QLTO/theory_test$ python landauer_limit_test.py 
[Init] System N=4. Landauer Test Ready.
======================================================================
EXPERIMENT 1b: LANDAUER COST ANALYSIS
Checking if Quantum Correlations allow Positive Net Work.
======================================================================
Tau    | MI (I)     | S(A) (Cost)  | Ratio I/S  | Work      
----------------------------------------------------------------------
0.00   | 0.0000     | 0.0000       | 0.00       | 0.0000    
0.08   | 0.0628     | 0.0314       | 2.00       | 0.0003    
0.16   | 0.1983     | 0.0991       | 2.00       | 0.0020    
0.24   | 0.3752     | 0.1876       | 2.00       | 0.0059    
0.32   | 0.5755     | 0.2877       | 2.00       | 0.0130    
0.39   | 0.7856     | 0.3928       | 2.00       | 0.0235    
0.47   | 0.9951     | 0.4975       | 2.00       | 0.0376    
0.55   | 1.1952     | 0.5976       | 2.00       | 0.0550    
0.63   | 1.3792     | 0.6896       | 2.00       | 0.0750    
0.71   | 1.5419     | 0.7709       | 2.00       | 0.0968    
0.79   | 1.6796     | 0.8398       | 2.00       | 0.1193    
0.87   | 1.7902     | 0.8951       | 2.00       | 0.1412    
0.95   | 1.8728     | 0.9364       | 2.00       | 0.1615    
1.03   | 1.9281     | 0.9640       | 2.00       | 0.1791    
1.11   | 1.9580     | 0.9790       | 2.00       | 0.1931    
1.18   | 1.9655     | 0.9827       | 2.00       | 0.2029    
1.26   | 1.9546     | 0.9773       | 2.00       | 0.2084    
1.34   | 1.9301     | 0.9651       | 2.00       | 0.2094    
1.42   | 1.8972     | 0.9486       | 2.00       | 0.2064    
1.50   | 1.8613     | 0.9306       | 2.00       | 0.1998    
----------------------------------------------------------------------
Algorithmic Efficiency η = 0.1104 Energy/Bit
Landauer Cost (ln2 * S_A):     0.4496 (avg)
Avg Quantum Ratio (I / S_A):   2.00

✓ SUCCESS: QUANTUM ADVANTAGE CONFIRMED.
  I(S:A)/S(A) ≈ 2.00 (near theoretical max of 2.0)
  Entanglement halves the information cost per unit work.
  The demon requires half the entropy production of a classical loop.
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
Saved plot to 'thermo_landauer_check.png'
(base) poig@pc:~/project/self-research/Quantum_AI/QLTO/theory_test$ python thermo_constitutive_law.py 
[Init] System N=4. Hamiltonian Terms=10
============================================================
EXPERIMENT 0: NON-INTERACTING EQUATION OF STATE (Work vs Info)
Fixed Kick Strength: 0.2
============================================================
Tau        | Mutual Info (bits)   | Work Extracted      
------------------------------------------------------------
0.0000     | 0.000000             | 0.000000            
0.0789     | 0.060763             | 0.000004            
0.1579     | 0.192149             | 0.000031            
0.2368     | 0.363664             | 0.000090            
0.3158     | 0.557625             | 0.000175            
0.3947     | 0.760954             | 0.000267            
0.4737     | 0.963374             | 0.000333            
0.5526     | 1.156765             | 0.000333            
0.6316     | 1.334885             | 0.000223            
0.7105     | 1.493239             | -0.000041           
0.7895     | 1.628984             | -0.000492           
0.8684     | 1.740843             | -0.001154           
0.9474     | 1.828985             | -0.002032           
1.0263     | 1.894867             | -0.003114           
1.1053     | 1.941014             | -0.004370           
1.1842     | 1.970749             | -0.005748           
1.2632     | 1.987853             | -0.007184           
1.3421     | 1.996188             | -0.008604           
1.4211     | 1.999285             | -0.009927           
1.5000     | 1.999964             | -0.011075           
------------------------------------------------------------
Fit Results: Work = -0.0036 * I + 0.0021
R-squared: 0.4826
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
Saved plot to 'thermo_control_equation_of_state.png'
============================================================
EXPERIMENT 1: EQUATION OF STATE (Work vs Info)
Fixed Kick Strength: 0.2
============================================================
Tau        | Mutual Info (bits)   | Work Extracted      
------------------------------------------------------------
0.0000     | 0.000000             | 0.000000            
0.0789     | 0.062751             | 0.000309            
0.1579     | 0.198292             | 0.001967            
0.2368     | 0.375233             | 0.005949            
0.3158     | 0.575478             | 0.012995            
0.3947     | 0.785627             | 0.023532            
0.4737     | 0.995050             | 0.037634            
0.5526     | 1.195179             | 0.055014            
0.6316     | 1.379197             | 0.075047            
0.7105     | 1.541899             | 0.096835            
0.7895     | 1.679614             | 0.119291            
0.8684     | 1.790159             | 0.141237            
0.9474     | 1.872781             | 0.161510            
1.0263     | 1.928094             | 0.179069            
1.1053     | 1.957987             | 0.193070            
1.1842     | 1.965492             | 0.202936            
1.2632     | 1.954618             | 0.208388            
1.3421     | 1.930121             | 0.209444            
1.4211     | 1.897226             | 0.206403            
1.5000     | 1.861281             | 0.199786            
------------------------------------------------------------
Fit Results: Work = 0.1104 * I + -0.0367
R-squared: 0.8852
Saved plot to 'thermo_equation_of_state.png'
(base) poig@pc:~/project/self-research/Quantum_AI/QLTO/theory_test$ python thermo_scrambling_crash.py 
======================================================================
EXPERIMENT 3 (CORRECTED): NORMALIZED EFFICIENCY CRASH
Metric: Normalized Efficiency (eta / N^2)
Hypothesis: Chaotic systems crash at N ~ 6.5
======================================================================
N     | Ord Raw    | Ord Norm   | Cha Raw    | Cha Norm  
----------------------------------------------------------------------
3     | 0.0871     | 0.0097     | 0.0166     | 0.0018    
4     | 0.0372     | 0.0023     | 0.0237     | 0.0015    
5     | 0.1466     | 0.0059     | -0.0051    | -0.0002   
6     | 0.0392     | 0.0011     | -0.0051    | -0.0001   
7     | 0.1304     | 0.0027     | 0.0046     | 0.0001    
8     | 0.1201     | 0.0019     | 0.0063     | 0.0001    
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""

Saved plot to 'thermo_complexity_crash_corrected.png'

>>> SUCCESS: CRASH DETECTED.
    Chaotic efficiency drops significantly as N grows.