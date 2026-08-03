"""What the hardware vendors actually charge for, and what that does to V3 vs V4.

Every cost verdict in these notes so far assumed a currency without checking it.
I charged QLTO for circuit DEPTH on the grounds that a deep circuit occupies the
processor longer - true in physics, but the vendors do not all bill physics. So:
read the billing models out of the SDKs and price one gradient under each.

The finding that motivated this file, from IBM's own execution docs: circuits are
executed COLUMN-WISE over a matrix of circuits x shots, and the inter-circuit
rep_delay "is inserted between circuits, [so] each shot of the execution
encounters this delay". Total QPU time is therefore

    n_circuits * shots * (rep_delay + circuit_duration + readout)

and rep_delay is HUNDREDS of microseconds against a circuit duration of ones to
hundreds. So circuit COUNT multiplies the dominant term and depth is nearly free
- the opposite of what I assumed when I called V3 unshippable.

Braket is a second model entirely: per-task plus per-shot, with NO time or depth
term at all. IonQ is a third: gate-shots, which charges gate COUNT but not depth.
Three models, three different answers, so the honest output is a table rather
than a verdict.
"""
import inspect
import re
import numpy as np

print("=" * 92)
print("PART 1 — billing models as implemented in the vendor SDKs")
print("=" * 92)

# ---------------------------------------------------------------- Braket
try:
    from braket.tracking import tracker as BT
    src = inspect.getsource(BT)
    print("\n  AWS Braket (braket.tracking.tracker) — the qBraid cost tracker delegates here")
    keys = set()
    for ln in src.split("\n"):
        if re.search(r"price|shot|task", ln, re.I) and re.search(r"=|return|\[", ln):
            s = ln.strip()
            if len(s) < 120 and not s.startswith("#"):
                keys.add(s)
    for s in sorted(keys)[:18]:
        print("    ", s)
except Exception as e:                                   # pragma: no cover
    print("  braket tracker unavailable:", e)

# ---------------------------------------------------------------- IonQ
try:
    from qbraid.runtime.ionq import job as IJ
    src = inspect.getsource(IJ)
    print("\n  IonQ (qbraid.runtime.ionq.job) — cost fields exposed on the job")
    for ln in src.split("\n"):
        if re.search(r"cost|usd|credit|gate.?shot", ln, re.I):
            s = ln.strip()
            if s and not s.startswith("#") and len(s) < 120:
                print("    ", s)
except Exception as e:                                   # pragma: no cover
    print("  ionq job unavailable:", e)

# ---------------------------------------------------------------- IBM via qiskit
try:
    from qiskit_ibm_runtime import QiskitRuntimeService     # noqa: F401
    print("\n  IBM (qiskit-ibm-runtime): billed in QPU SECONDS. rep_delay is a backend")
    print("    configuration field; the relevant attributes are")
    print("    backend.configuration().rep_delay_range / default_rep_delay,")
    print("    and dt (gate time granularity). Requires credentials to query, so the")
    print("    numbers below are the published defaults rather than a live read.")
except Exception as e:
    print("\n  qiskit-ibm-runtime not installed:", e)


print()
print("=" * 92)
print("PART 2 — one Heisenberg N=6 gradient priced under each model")
print("=" * 92)

# measured in v16/v18/v19. CORRECTED: the QLTO arms were short by the per-epoch
# point-energy circuit. The shipped accounting is 2B+1 per epoch - B sensing, B
# walk, 1 energy log - which is the 180-circuits-for-20-epochs figure in the
# README (B=4 -> 9/epoch). V4 is B*G sensing + B walk + 1 log = B(G+1)+1.
# The log circuit is listed under free savings as removable; it is counted here
# because it is currently spent.
B_BLK, G_GRP = 4, 3
CIRCUITS = {'parameter-shift': 144,                     # 2*M*G, M=24 G=3
            'QLTO V3 (QPE)': 2 * B_BLK + 1,             # 9
            'QLTO V4 (direct)': B_BLK * (G_GRP + 1) + 1}  # 17
DEPTH    = {'parameter-shift': 14,  'QLTO V3 (QPE)': 1976, 'QLTO V4 (direct)': 21}
GATES2Q  = {'parameter-shift': 5,   'QLTO V3 (QPE)': 949, 'QLTO V4 (direct)': 17}
# CORRECTED. My first pass put a G-fold shot multiplier on the direct-readout
# arms ON TOP of already giving them G times as many circuits, double-charging
# them 3x. At S shots per circuit all three reach the SAME per-evaluation
# variance Var(H)/S: QPE samples eigenvalues directly, while direct readout gets
# sum_g Var(H_g)/S from G separate circuits of S shots each. So the G penalty is
# entirely a CIRCUIT-COUNT effect and is already in the row above. Multiplier 1.
SHOT_MULT = {'parameter-shift': 1.0, 'QLTO V3 (QPE)': 1.0, 'QLTO V4 (direct)': 1.0}
S_BASE = 4096

T_GATE   = 70e-9      # Heron 2q gate, seconds
T_READ   = 4e-6       # readout + reset, inside IBM's "circuit length"
REP_DELAY = 250e-6    # IBM DOCUMENTED DEFAULT, incurred per circuit PER SHOT
IBM_OVERHEAD = 2.0    # IBM documented per-sub-job overhead, seconds
IBM_USD_PER_SEC = 96.0 / 60.0        # Pay-As-You-Go $96/min

# AWS Braket, verified: $0.30/task on every on-demand QPU, and the per-shot
# price "doesn't change based on how many gates you use or what type they are" -
# i.e. ZERO depth term, the cleanest case for the argument.
BRAKET_PER_TASK = 0.30
BRAKET_SHOT = {'Rigetti Ankaa': 0.00090, 'IQM Garnet': 0.00145,
               'IonQ Aria': 0.03000}

print(f"\n  Heisenberg N=6: M=24, G=3, base {S_BASE} shots/circuit")
print("  IBM documented formula: overhead + (rep_delay + circuit_length) * circuits * shots")
print(f"    overhead {IBM_OVERHEAD}s, rep_delay {REP_DELAY*1e6:.0f}us, ${IBM_USD_PER_SEC:.2f}/s")
print(f"  Braket: ${BRAKET_PER_TASK}/task + per-shot, NO gate or depth term")
print()
print(f"  {'method':<20}{'circuits':>9}{'shots':>10}{'circ len':>10}{'QPU s':>9}"
      f"{'IBM $':>9}" + "".join(f"{d.split()[0]+' $':>14}" for d in BRAKET_SHOT))
print("  " + "-" * 105)

rows = {}
for m in CIRCUITS:
    nc = CIRCUITS[m]
    shots = S_BASE * SHOT_MULT[m]
    clen = DEPTH[m] * T_GATE + T_READ
    nexec = nc * shots
    qpu_s = IBM_OVERHEAD + (REP_DELAY + clen) * nexec
    ibm = qpu_s * IBM_USD_PER_SEC
    brk = {d: nc * BRAKET_PER_TASK + nexec * p for d, p in BRAKET_SHOT.items()}
    rows[m] = (qpu_s, ibm, brk)
    print(f"  {m:<20}{nc:>9}{int(nexec):>10}{clen*1e6:>9.1f}us{qpu_s:>9.1f}{ibm:>9.2f}"
          + "".join(f"{brk[d]:>14.2f}" for d in BRAKET_SHOT))

print()
base = rows['parameter-shift']
for m in CIRCUITS:
    if m == 'parameter-shift':
        continue
    r = rows[m]
    br = "  ".join(f"{d.split()[0]} {base[2][d]/r[2][d]:.1f}x" for d in BRAKET_SHOT)
    print(f"  {m:<20} vs p-shift:  IBM {base[1]/r[1]:>5.1f}x   {br}")
v3, v4 = rows['QLTO V3 (QPE)'], rows['QLTO V4 (direct)']
br = "  ".join(f"{d.split()[0]} {v4[2][d]/v3[2][d]:.1f}x" for d in BRAKET_SHOT)
print(f"  {'V3 vs V4':<20}              IBM {v4[1]/v3[1]:>5.1f}x   {br}"
      f"      (>1 = V3 cheaper)")

print()
print("=" * 92)
print("PART 3 — where the crossover sits, as a function of rep_delay")
print("=" * 92)
print("  V3 wins over V4 when its G-fold shot saving beats its extra circuit time:")
print("      G  >  (depth_V3*t_gate + t_read + rep) / (depth_V4*t_gate + t_read + rep)")
print()
print(f"  {'rep_delay':>12}{'V3 per-shot':>14}{'V4 per-shot':>14}{'ratio':>9}"
      f"{'G needed':>10}{'G=3 verdict':>14}")
print("  " + "-" * 73)
for rep in (0.0, 1e-6, 10e-6, 50e-6, 250e-6, 500e-6):
    d3 = DEPTH['QLTO V3 (QPE)'] * T_GATE + T_READ + rep
    d4 = DEPTH['QLTO V4 (direct)'] * T_GATE + T_READ + rep
    ratio = d3 / d4
    print(f"  {rep*1e6:>10.0f}us{d3*1e6:>13.1f}us{d4*1e6:>13.1f}us{ratio:>9.1f}"
          f"{ratio:>10.1f}{'V3 wins' if 3.0 > ratio else 'V4 wins':>14}")

print()
print("  The whole verdict turns on rep_delay. At rep=0 the deep circuit pays its")
print("  full 26x and V4 wins; at IBM's 250us default the delay dominates both, the")
print("  ratio collapses toward 1, and V3's G-fold circuit saving carries it.")


print()
print("=" * 92)
print("PART 4 — projection to sizes no simulator reaches")
print("=" * 92)
print("  Everything above is measured at N<=8, which is not where the question")
print("  lives. Past ~30 qubits a highly entangled circuit cannot be simulated at")
print("  all, so the ONLY way to compare is the analytic cost model - which needs")
print("  no simulator, because circuit count and depth are counting arguments.")
print()
print("  Fitted from v19 (Heisenberg, efficient_su2 reps=1, kappa FIXED at 4):")
print("     M = 4N     G = 3     T = 3(N-1)")
print("     depth  p-shift ~ N+8     V4 ~ N+15     V3 ~ 408N-472")
print("     circuits p-shift = 2MG = 24N    V3 = 2B = 8    V4 = B(G+1) = 16")
print()
print(f"  {'N':>5}{'M':>6}{'p-shift circ':>14}{'V3 depth':>10}{'V3 dur':>10}"
      f"{'IBM: p-shift':>14}{'V3':>10}{'V4':>10}{'V3/V4':>8}")
print("  " + "-" * 87)

for N in (8, 10, 20, 30, 50, 100):
    M = 4 * N
    G = 3
    nc = {'ps': 24 * N, 'v3': 2 * B_BLK + 1, 'v4': B_BLK * (G_GRP + 1) + 1}
    dep = {'ps': N + 8, 'v3': 408 * N - 472, 'v4': N + 15}
    cost = {}
    for kk in nc:
        clen = dep[kk] * T_GATE + T_READ
        cost[kk] = (IBM_OVERHEAD + (REP_DELAY + clen) * nc[kk] * S_BASE) * IBM_USD_PER_SEC
    dur3 = dep['v3'] * T_GATE * 1e6
    print(f"  {N:>5}{M:>6}{nc['ps']:>14}{dep['v3']:>10}{dur3:>9.0f}us"
          f"{cost['ps']:>14.0f}{cost['v3']:>10.1f}{cost['v4']:>10.1f}"
          f"{cost['v4']/cost['v3']:>8.2f}")

print()
print("  CORRECTION: the table above holds SHOTS FIXED at 4096 for every N, which")
print("  is wrong and would make V4's cost constant in N - an exponential speedup,")
print("  which is the tell. T4 gives Var(g_i) = (1/S)[a + b(n-1)] with b > 0 for any")
print("  readout that returns an ENERGY (both V3 and V4; only the bounded +-1")
print("  Hadamard bit has b = 0). And Var(H) is EXTENSIVE - the benchmark audit")
print("  measured 8.315 / 14.110 / 21.241 at N = 4/6/8, i.e. Theta(N). Holding")
print("  precision fixed therefore needs S ~ Theta(N).")
print()
print("  Per FULL RUN (20 epochs), shots scaled as S(N) = 4096 * Var(N)/Var(6):")
print(f"  {'N':>5}{'Var(H)':>8}{'shots':>9}{'p-shift circ':>13}{'V4 circ':>9}"
      f"{'p-shift $':>12}{'V3 $':>10}{'V4 $':>10}{'V4 saving':>11}")
print("  " + "-" * 87)
EPOCHS = 20
VAR6 = 3.23 * 6 - 4.6            # fitted from the audit table, Var ~ 3.23N - 4.6
for N in (8, 10, 20, 30, 50, 100):
    varN = 3.23 * N - 4.6
    S_N = S_BASE * varN / VAR6
    nc = {'ps': 24 * N, 'v3': 2 * B_BLK + 1, 'v4': B_BLK * (G_GRP + 1) + 1}
    dep = {'ps': N + 8, 'v3': 408 * N - 472, 'v4': N + 15}
    cost = {}
    for kk in nc:
        clen = dep[kk] * T_GATE + T_READ
        cost[kk] = (IBM_OVERHEAD
                    + (REP_DELAY + clen) * nc[kk] * EPOCHS * S_N) * IBM_USD_PER_SEC
    print(f"  {N:>5}{varN:>8.1f}{int(S_N):>9}{nc['ps']*EPOCHS:>13}"
          f"{nc['v4']*EPOCHS:>9}{cost['ps']:>12.0f}{cost['v3']:>10.0f}"
          f"{cost['v4']:>10.0f}{cost['ps']/cost['v4']:>10.0f}x")

print()
print("=" * 92)
print("PART 5 — the cost I never charged: FIDELITY")
print("=" * 92)
print("  Every table above prices TIME and treats depth as nearly free because")
print("  rep_delay dominates it. That is true of the BILL and false of the PHYSICS.")
print("  Depth is what fidelity is denominated in, and a circuit that returns noise")
print("  is not cheap at any price.")
print()
print("  Per-circuit two-qubit counts scale as (v16, confirmed at N=6):")
print("      parameter-shift  C_a           ~ N        (5 at N=6)")
print("      QLTO V4          C_a + 2n      ~ 3N       (17 at N=6)  <- the W gate")
print("      QLTO V3          + 2^k*r*T     ~ 158N     (949 at N=6)")
print()
print("  The W gate is the thing V4 pays for its shallowness: one CONTROLLED")
print("  rotation per active parameter, so 2n extra two-qubit gates that")
print("  parameter-shift never pays. Unmitigated survival is (1-p)^cx; zero-noise")
print("  extrapolation costs a shot overhead ~ exp(2*p*cx).")
print()
P2Q = 5e-3
print(f"  two-qubit error p = {P2Q}")
print(f"  {'N':>5}{'p-shift cx':>12}{'V4 cx':>8}{'V3 cx':>9}"
      f"{'p-shift surv':>14}{'V4 surv':>10}{'V3 surv':>10}"
      f"{'ZNE p-shift':>13}{'ZNE V4':>9}")
print("  " + "-" * 90)
for N in (6, 20, 50, 100, 300):
    cxp, cx4, cx3 = N, 3 * N, 158 * N
    sp, s4, s3 = (1-P2Q)**cxp, (1-P2Q)**cx4, (1-P2Q)**cx3
    zp, z4 = np.exp(2*P2Q*cxp), np.exp(2*P2Q*cx4)
    print(f"  {N:>5}{cxp:>12}{cx4:>8}{cx3:>9}{sp:>14.3f}{s4:>10.3f}{s3:>10.2e}"
          f"{zp:>13.1f}{z4:>9.1f}")

print()
print("  V3 IS ALREADY DEAD AT N=6 UNMITIGATED - survival 8.6e-03, i.e. the circuit")
print("  returns noise. That was the coherence objection I raised and then walked")
print("  back on billing grounds; the billing argument was right about the INVOICE")
print("  and wrong about whether the answer is usable.")
print()
print("  AND V4'S ADVANTAGE HAS A CEILING. Its circuit-count edge over")
print("  parameter-shift grows ~1.4N, but its mitigation overhead grows as")
print("  exp(2p*3N)/exp(2p*N) = exp(4pN) - EXPONENTIALLY faster. Break-even:")
for N in (100, 200, 300, 400, 500):
    edge = 1.4 * N
    penalty = np.exp(4 * P2Q * N)
    print(f"      N={N:>4}   circuit edge {edge:>7.0f}x   fidelity penalty "
          f"{penalty:>8.1f}x   net {edge/penalty:>7.1f}x")
print()
print("  So the Theta(N) advantage is real only while error is CORRECTED rather")
print("  than mitigated. Unmitigated it peaks near N~300 and inverts after. Under")
print("  fault tolerance the currency changes again to T-count / spacetime volume,")
print("  where the W gate's 2n rotations are the term that matters - UNCOSTED HERE.")
print()
print("  SO NOTHING HERE IS CONSTANT, AND NOTHING IS EXPONENTIAL. Counting orders:")
print("      parameter-shift  Theta(N) circuits x Theta(N) shots x O(1) dur = Theta(N^2)")
print("      QLTO V3          O(1) circuits x Theta(N) shots x Theta(N) dur = Theta(N^2)")
print("      QLTO V4          O(1) circuits x Theta(N) shots x O(1) dur     = Theta(N)")
print("  V4's advantage over parameter-shift is a factor of N - GROWING, POLYNOMIAL,")
print("  and it comes entirely from circuits being O(1) while depth stays under")
print("  rep_delay. V3 loses its edge because its duration grows with N as well.")

print()
print("  V3/V4 > 1 means V3 is cheaper. It falls through 1 where V3's circuit")
print("  DURATION overtakes rep_delay - depth*70ns > 250us, i.e. depth > 3571,")
print("  i.e. N ~ 10 on this fit. Past that V3 pays its depth in full and the")
print("  G-fold circuit saving no longer covers it.")
print()
print("  CAVEAT, and it cuts against V3 harder than the table shows: this holds")
print("  kappa FIXED at 4. QPE resolution is 2*margin*||H||/2^kappa and ||H|| is")
print("  extensive, so holding kappa constant means the gradient degrades as N")
print("  grows. Scaling kappa honestly makes V3 depth ~N*T = O(N^2) and brings the")
print("  crossover in sooner still. The V3 column is therefore optimistic.")
print()
print("  Note also NEITHER arm is tuned or optimised here - no transpiler")
print("  optimisation_level 3, no pulse-level compression, no rep_delay tuning,")
print("  no block-width optimum (T10 says n* ~ 0.65M, i.e. B ~ 1.5 against the 4")
print("  shipped, which is ~2.7x of circuits left on the table for BOTH V3 and V4).")
