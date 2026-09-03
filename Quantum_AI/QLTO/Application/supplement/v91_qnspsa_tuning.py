"""Sweep QN-SPSA's lr the way TUNED was derived for everything else.

The smoke test had QN-SPSA descending far slower than V6, at the DEFAULT lr=0.1.
That is exactly the trap this file's own header records: every earlier run used
AdamW at lr=0.1 and called it a baseline, and tuning moved it 0.0463 -> 0.0306.
Reporting QN-SPSA against V6 before sweeping would repeat that error against a
method chosen precisely because it is the strongest competitor.

Same protocol as TUNED: score is the mean gap to exact as a fraction of the
spectral range, over a representative subset, 2 trials each, one global lr. The
perturbation eps is swept alongside because the natural-gradient step and the
metric estimate are coupled through it - a value good for the gradient can be bad
for the QFIM finite difference.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, "/home/poig/project/self-research/Quantum_AI/QLTO/Application")
os.chdir("/home/poig/project/self-research/Quantum_AI/QLTO/Application")
import benchmark as B

PROBS = [B.get_h2_problem(), B.get_maxcut_problem(4),
         B.get_heisenberg_problem(4)]
LRS = (0.03, 0.1, 0.3, 1.0, 3.0)
EPS = (0.05, 0.1, 0.3)
TRIALS, EPOCHS = 2, 20

meta = []
for ansatz, H, name in PROBS:
    ev = np.linalg.eigvalsh(H.to_matrix())
    meta.append((ansatz, H, name, float(ev.min()), float(ev.max() - ev.min())))

print(f"  score = mean gap/(spectral range), {TRIALS} trials, {EPOCHS} epochs")
print(f"  lower is better.  V6 reference computed alongside.")
print()
print(f"  {'lr':>7}{'eps':>7}{'score':>10}{'nefv':>8}   per-problem gaps")
print("  " + "-" * 70)

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
            flag = '  <-- best so far'
        print(f"  {lr:>7.2f}{eps:>7.2f}{sc:>10.4f}{nef:>8}   "
              + " ".join(f"{v:.4f}" for v in gaps) + flag, flush=True)

print()
print(f"  BEST: lr={best[1]}  eps={best[2]}  score={best[0]:.4f}")

gaps = []
v6_nefv = 0
for ansatz, H, name, exact, span in meta:
    M = ansatz.num_parameters
    g = []
    for t in range(TRIALS):
        np.random.seed(42 + t)
        p = np.random.uniform(-np.pi, np.pi, M)
        with contextlib.redirect_stdout(io.StringIO()):
            opt = B.QLTOv6_Wrapper(ansatz, H, r0=B.TUNED['QLTO V6'])
            B._mult(opt, 1)
        es = []
        for _ in range(EPOCHS):
            p = opt.step(p)
            es.append(B.report_energy(ansatz, H, p))
        g.append((min(es) - exact) / span)
        v6_nefv = B.optimizer_circuits(opt)
    gaps.append(float(np.mean(g)))
print(f"  V6 same protocol: score={float(np.mean(gaps)):.4f}"
      f"  nefv={v6_nefv}   per-problem " + " ".join(f"{v:.4f}" for v in gaps))
