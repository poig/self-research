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
/home/poig/project/self-research/Quantum_AI/QLTO/theory_test/k_ancilla_bandwidth_test.py:18: SyntaxWarning: invalid escape sequence '\l'
  state has the upper bound I(S:A) \le 2k (bits) when A is k qubits.
======================================================================
K-ANCILLA BANDWIDTH SCALING TEST (Multi-Seed Averaging)
Averaging over 5 random seeds per configuration
======================================================================

======================================================================
ANCILLA COUNT: k = 1 (max I(S:A) = 2 bits per cycle)
======================================================================
  [N=3] Running 5 seeds... η = 0.0871 ± 0.000
  [N=4] Running 5 seeds... η = 0.1132 ± 0.000
  [N=5] Running 5 seeds... η = -0.0329 ± 0.000
  [N=6] Running 5 seeds... η = 0.1360 ± 0.000
  [N=7] Running 5 seeds... η = -0.2873 ± 0.000

======================================================================
ANCILLA COUNT: k = 2 (max I(S:A) = 4 bits per cycle)
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
  k=1: Crash point Nc ≈ 5, Avg Bandwidth (2·S(A)) = 1.13 bits, Avg η = +0.003
  k=2: Crash point Nc ≈ 3, Avg Bandwidth (2·S(A)) = 2.34 bits, Avg η = +0.001

======================================================================
CONCLUSION
======================================================================

  Bandwidth proxy scaling: k=2/k=1 ratio = 2.07 (ideal ≈ 2.00)
  Avg η: k=1: +0.0032 | k=2: +0.0013
  Avg work proxy (η×2S(A)): k=1: +0.0072 | k=2: -0.0069

  NOTE: Bandwidth increased, but work proxy did not improve (efficiency is the bottleneck, not bandwidth).
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
Landauer Cost (kBT·ln2·S_A):   0.4496 (avg)
Assumed k_B T:                1.000
Avg Quantum Ratio (I / S_A):   2.00

Note: S(SA) ≈ 0 in statevector simulation, so I(S:A) = 2 S(A) holds as an identity.
      Treat I/S(A) ≈ 2 as a regime check (pure joint state), not an empirical advantage claim.
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
0.0789     | 0.108775             | -0.000194           
0.1579     | 0.331156             | -0.001525           
0.2368     | 0.602478             | -0.004989           
0.3158     | 0.884902             | -0.011314           
0.3947     | 1.152415             | -0.020860           
0.4737     | 1.387923             | -0.033564           
0.5526     | 1.582007             | -0.048918           
0.6316     | 1.731819             | -0.066004           
0.7105     | 1.839752             | -0.083570           
0.7895     | 1.911834             | -0.100141           
0.8684     | 1.956003             | -0.114161           
0.9474     | 1.980484             | -0.124142           
1.0263     | 1.992518             | -0.128816           
1.1053     | 1.997621             | -0.127266           
1.1842     | 1.999411             | -0.119029           
1.2632     | 1.999898             | -0.104155           
1.3421     | 1.999990             | -0.083224           
1.4211     | 2.000000             | -0.057309           
1.5000     | 2.000000             | -0.027896           
------------------------------------------------------------
Fit Results: Work = -0.0568 * I + 0.0208
R-squared: 0.6922
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
EXPERIMENT 3 (FIXED v2): COMPLEXITY PHASE TRANSITION
η accepted only when R² ≥ 0.6  |  Significance p < 0.05
Normalization: η_norm = η / ‖H‖  (intensive efficiency)
======================================================================
N     | Ord η±σ            | Ord R²   | Cha η±σ            | Cha R²   | p-val    | Sig?
--------------------------------------------------------------------------------
3     | +nan ± nan   | 0.005   | +0.0017 ± 0.0050   | 0.894   | nan   | n/a (too few valid)
4     | +nan ± nan   | 0.143   | -0.0032 ± 0.0092   | 0.796   | nan   | n/a (too few valid)
5     | +nan ± nan   | 0.324   | +0.0011 ± 0.0061   | 0.619   | nan   | n/a (too few valid)
6     | +nan ± nan   | 0.300   | +0.0039 ± 0.0054   | 0.615   | nan   | n/a (too few valid)
7     | +nan ± nan   | 0.262   | -0.0055 ± 0.0138   | 0.623   | nan   | n/a (too few valid)
