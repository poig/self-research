"""Does the coherent walk step beat the alternatives - including brute force?

TIER A - qlto_walk sensing and the walk step are both real circuits on
AerSimulator with shots. The classical arms are exact.

THE JOIN. This is the first end-to-end run of

    sense (3-level design, one shot record)  ->  g, H
    h = kappa R^2 sqrt(lambda_max)           ->  the walk's schedule
    walk: cycle mixer + measured H as a degree-2 potential
    read a vertex                            ->  the step

The cycle mixer is v136's construction (F^dag . diag . F, verified against expm
at 1.4e-14) at the SIGN v136 PART 7 established, +h^2(D-A). The potential is
RZ + RZZ, verified against the direct quadratic form at 2.6e-16 - not a
DiagonalGate, which would cost O(2^{d kappa}).

FOUR CONTROLS, and the last one is the one that decides anything.

  newton   -(H + mu I)^-1 g, box-clipped
  grad     -g, box-clipped
  rand     uniform random vertex of the same box, 64 draws averaged
  brute    EXHAUSTIVE minimisation of the SAME measured quadratic model over
           the SAME 2^{d kappa} vertices

  The box clip is infinity-norm for every arm. A 2-norm cap would hand the walk
  a step sqrt(d) times larger and the comparison would measure step size. The
  first run of this file had exactly that bug and the walk won 6/6; matched, it
  wins 5/6 against grad and 6/6 against newton, and rand loses every time - so
  the win is not step size.

  BUT BRUTE FORCE WINS 5/6. At d*kappa = 12 qubits the box has 4096 vertices and
  exhaustive search is trivial, so the walk is a WORSE solver of a classically
  easy subproblem. That is the expected and correct result at this size, and it
  is what keeps the claim honest: this file measures PLUMBING, not advantage.

WHAT WOULD MEASURE ADVANTAGE, and this file does not. Two conditions must hold
together and neither does here:

  d*kappa >~ 40, so the box has ~10^12 vertices and brute force is infeasible;
  and a landscape with TALL THIN BARRIERS, which is the precondition of
  Liu-Su-Li's e^{S0/h} vs SGD's e^{2 H_f/s}. A random degree-2 model over a
  small box is a QUBO with mild structure - there may be nothing to tunnel
  through at all.

WHAT IS ENCOURAGING ANYWAY. The walk reached the exact brute-force optimum on
one seed and came within 3% on another, at P(best vertex) = 0.009 against a
uniform 0.00024 - 37x concentration. And the cost is honest: depth 1117, 2736
cx for ONE step on 12 qubits, dominated by the QFT mixer over 12 Trotter steps.
"""
import sys; sys.path.insert(0,'/home/poig/project/self-research/Quantum_AI/QLTO/Application')
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qlto_walk import QLTOWalk, walk_step

def heis(n):
    t=[]
    for i in range(n-1):
        for p in ('XX','YY','ZZ'):
            l=['I']*n; l[i],l[i+1]=p[0],p[1]; t.append((''.join(reversed(l)),1.0))
    return SparsePauliOp.from_list(t)

def anz_ry_rz(n,reps=1):
    p=ParameterVector('t',2*n*(reps+1)); qc=QuantumCircuit(n); i=0
    for _ in range(reps):
        for q in range(n): qc.ry(p[i],q); i+=1; qc.rz(p[i],q); i+=1
        for q in range(n-1): qc.cx(q,q+1)
    for q in range(n): qc.ry(p[i],q); i+=1; qc.rz(p[i],q); i+=1
    return qc

n=2; anz=anz_ry_rz(n); Hm=heis(n); M=anz.num_parameters
E=lambda x: float(np.real(Statevector(anz.assign_parameters(x)).expectation_value(Hm)))
sub=[0,1,2,3]
q=QLTOWalk(anz,Hm,shot_budget=1<<16,sim_seed=11)

print("SENSE -> h -> WALK, against Newton and plain descent")
print("  %5s %9s %8s %10s %10s %10s %10s %10s %8s"%("seed","E0","h","E walk","E newton","E grad","E rand","E brute","best"))
rng=np.random.default_rng(0)
wins={'walk':0,'newton':0,'grad':0,'rand':0,'brute':0}
for s in range(6):
    th=rng.uniform(-np.pi,np.pi,M); R=0.5
    g,H,_=q.sense(th,R,sub)
    e0=E(th)
    nw,info=walk_step(q,th,R,sub,g,H,kappa=3,d_walk=4,steps=12,shots=8192,seed=100+s)
    e_w=E(nw)
    e_n=E(QLTOWalk.grad_step(th,g,H,R,sub,newton=True))
    e_g=E(QLTOWalk.grad_step(th,g,H,R,sub,newton=False))
    # CONTROL: a uniformly random vertex of the same box. If this also beats
    # Newton, the walk is winning on step size and not on interference.
    rr=np.random.default_rng(900+s)
    e_r=np.mean([E(np.array(th,float)+np.pad(rr.uniform(-R,R,len(sub)),
                 (0,M-len(sub)))) for _ in range(64)])
    # THE CONTROL THAT MATTERS: the walk minimises the MEASURED quadratic model
    # over the box. So can brute force, over the same 2^(d*kappa) vertices. If
    # brute force wins, the walk is a worse solver of a classically easy
    # subproblem and the comparison against one Newton step was never the point.
    kap=3; a=2.0*R/((1<<kap)-1)
    grid=np.array([a*np.arange(1<<kap)-R])[0]
    import itertools
    Hs=H[np.ix_(sub,sub)]; gs=g[sub]
    bestv=None; bestm=np.inf
    for v in itertools.product(range(1<<kap),repeat=len(sub)):
        t=grid[list(v)]
        mv=gs@t+0.5*t@Hs@t
        if mv<bestm: bestm, bestv = mv, t
    xb=np.array(th,float); xb[sub]+=bestv; e_b=E(xb)
    best=min([('walk',e_w),('newton',e_n),('grad',e_g),('rand',e_r),('brute',e_b)],key=lambda t:t[1])[0]
    wins[best]+=1
    print("  %5d %9.4f %8.4f %10.4f %10.4f %10.4f %10.4f %10.4f %8s"%(s,e0,info['h'],e_w,e_n,e_g,e_r,e_b,best))
print("\n  wins:",wins)
print("  walk circuit: %d qubits, depth %d, cx %d, P(best vertex) %.3f"
      %(info['qubits'],info['depth'],info['cx'],info['p_best']))
