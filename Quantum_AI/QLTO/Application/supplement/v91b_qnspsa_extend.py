"""Extend the QN-SPSA sweep below lr=0.03, because the optimum hit the boundary.

The first sweep's best was lr=0.03 eps=0.05, the SMALLEST lr on the grid. TUNED's
own protocol requires an optimum that is interior or bracketed, and every entry
in that table records which it was. Declaring QN-SPSA beaten on an unbracketed
edge would be the same error as benchmarking AdamW at an untuned lr=0.1 - worse
here, because QN-SPSA was added specifically as the strongest competitor and a
rigged loss for it is worth nothing.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, "/home/poig/project/self-research/Quantum_AI/QLTO/Application")
os.chdir("/home/poig/project/self-research/Quantum_AI/QLTO/Application")
import benchmark as B

PROBS = [B.get_h2_problem(), B.get_maxcut_problem(4),
         B.get_heisenberg_problem(4)]
LRS = (0.001, 0.003, 0.01, 0.03)
EPS = (0.02, 0.05, 0.1)
TRIALS, EPOCHS = 2, 20

meta = []
for ansatz, H, name in PROBS:
    ev = np.linalg.eigvalsh(H.to_matrix())
    meta.append((ansatz, H, name, float(ev.min()), float(ev.max() - ev.min())))

print(f"  extension below the boundary. V6 reference = 0.0254 at nefv 60.")
print(f"  first sweep best was lr=0.03 eps=0.05 score=0.1707 (edge).")
print()
print(f"  {'lr':>8}{'eps':>7}{'score':>10}{'nefv':>8}   per-problem gaps")
print("  " + "-" * 66)

best = None
for lr in LRS:
    for eps in EPS:
        gaps, nef = [], 0
        for ansatz, H, name, exact, span in meta:
            M = ansatz.num_parameters
            g = []
            for t in range(TRIALS):
                np.random.seed(42 + t)
                p = np.random.uniform(-np.pi, np.pi, M)
                with contextlib.redirect_stdout(io.StringIO()):
                    opt = B.QNSPSA_Wrapper(ansatz, H, lr=lr, perturbation=eps,
                                           seed=42 + t)
                    B._mult(opt, 1)
                es = []
                for _ in range(EPOCHS):
                    p = opt.step(p)
                    es.append(B.report_energy(ansatz, H, p))
                g.append((min(es) - exact) / span)
                nef = B.optimizer_circuits(opt)
            gaps.append(float(np.mean(g)))
        sc = float(np.mean(gaps))
        flag = ''
        if best is None or sc < best[0]:
            best = (sc, lr, eps)
            flag = '  <-- best'
        print(f"  {lr:>8.3f}{eps:>7.2f}{sc:>10.4f}{nef:>8}   "
              + " ".join(f"{v:.4f}" for v in gaps) + flag, flush=True)

print()
print(f"  BEST OVERALL: lr={best[1]}  eps={best[2]}  score={best[0]:.4f}")
if best[1] == min(LRS):
    print("  STILL ON THE EDGE - the optimum is not bracketed and the")
    print("  comparison remains provisional.")
else:
    print("  INTERIOR or bracketed: the optimum is located and the")
    print("  comparison against V6 is now on the same footing as TUNED.")
