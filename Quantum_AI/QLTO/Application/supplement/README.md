# Supplement: investigation scripts and their logs

Every script here answers one question and writes one log in `results/`. The
answers are folded into `../nisq_v3.py`'s module docstring, which cites these logs
by path; this file is the reverse index — log to question to verdict.

Each script is standalone: it inserts the parent `Application/` directory on
`sys.path` and imports `nisq_v3` / `benchmark` from there, so it runs from
anywhere. Run as e.g. `python supplement/v5_walsh.py`.

`../benchmark.py` and its outputs (`../results/`) are deliberately NOT here — that
is the headline suite, not a supplement.

## Verdicts

| script | question | verdict |
|---|---|---|
| `anomaly_and_wdag.py` | Is the ‖g‖ scale anomaly explained by the R-smeared target? Is W† removable? | Smearing does **not** explain it. W† removable (TVD 0.010 vs 0.046 noise floor) but saves 0% depth, 6–19% gates |
| `anomaly_c.py` | Noise floor or genuine bias? | **Bias.** Converged at 1e6 shots; QPE blk1 at 2.139 ± 0.014, 80σ from 1 |
| `anomaly_e.py` | Is it QPE quantisation? | **No.** 16× finer bins (3.232→0.202), bias unmoved. Also demotes adaptive-k |
| `anomaly_f.py` | Is it Trotter error? Reps at fixed τ | **Yes for Z blocks** (1.616→1.009). Y blocks flat → second mechanism |
| `anomaly_g.py` | Is the residual the sin() nonlinearity? | Killed for time; mechanism established two other ways (Suzuki-4 floor, QPE having no floor) |
| `ansatz_ceiling.py` | How much of the gap is the ansatz? | reps=1 caps at −6.1231 vs reps=3 −6.4641; all methods within 1–2% of the ceiling |
| `audit_benchmark.py` | Is `benchmark.py` fair? | **Three defects, all favouring baselines.** Subsidy 23–57× on Heisenberg |
| `v4_candidates.py` | Which sensing fix is best per unit depth? | Suzuki-2 wins; **Richardson rejected** (same bias, 2× noise, 2× circuits) |
| `v4_frontier.py` | Optimal QPE reps schedule | **suz2 at 2^a/2**: 2.8× less bias for +11% depth. Applied to `nisq_v3.py` |
| `v4_schedule.py` | Does k_steps double as a step-size knob? | Yes — Σs = k/2 exactly, \|move\| grows 0.306→0.860 |
| `v4_schedule2.py` | Is normalising it an improvement, dt re-tuned? | **No.** current −4.561 vs normalised −4.415. The coupling is load-bearing |
| `v4_walk_trotter.py` | Does the walk's own Trotter error matter? | **No.** 158× more error than sensing, every variant within 0.0–0.2σ |
| `v4_argmin.py` | Is argmin better than the marginal? Is Grover worth building? | **argmin loses** at both sizes; it can never take a small step. Grover closed |
| `v4_softmin.py` | Which decoder, with the shot-parity control? | Nothing beats the walk. Boltzmann T=0.1 **ties** at half the circuits. top-4's earlier "win" did **not** replicate |
| `v4_cost.py` | Cost vs block width | Var flat in n (b/a = −0.004); signal attenuates, so the optimum is **interior** (n≈M/2), not global |
| `v4_cost2.py` | Is QLTO cheaper, baseline charged honestly? | **Yes**: 3.2× fewer shots, 48× fewer circuits at n=8. Hadamard **loses** on shots |
| `v5_walsh.py` | What does the Walsh spectrum hold? | deg1+deg2 = **99.6%+**; deg2 exceeds deg1 on 2 of 4 blocks; measurable at SNR 3–4 |
| `v5_deg2walk.py` | Does degree-2 drift help? | **No.** Monotonic degradation past gain 0.25; variance grows with gain |
| `v5_merge.py` | Can CRZ+CRX merge to one tilted-axis rotation? | Identity exact to 4e−16. **Depth −37%**; energy contradictory across sizes, unresolved |
| `v5_locality.py` | Why is the landscape exactly quadratic? | **Locality, and it's a theorem.** deg3 = 1e−32 (exact zero) on blocks with no entangler after them, 1e−3 on those before a CX |
| `v5_moments.py` | Are all moments of H free from the QPE shots? | **Yes.** deg1 of e² at cos 0.995. Best at k=4–5, degrades at k=6–7; far more `qpe_margin`-sensitive than the first moment |
| `v6_hamlearn.py` | Does the Hamiltonian-learning pivot work? | **Yes — first pivot validated end to end.** Gradient cos 0.9993 vs the *true* gradient; recovery to 0.034 worst coefficient in 30 circuits vs 300. No ancilla, no QPE |
| `v6_multiprobe.py` | Does multi-probe help, or is one probe enough? | **Helps.** At fixed total shots, error halves from P=1 to P=4 — probe diversity beats shot precision, which justifies the shared-param parallel register |
| `v7_mixer.py` | Does non-uniform mixing help? | **Undecided.** 8/8 nonzero λ beat uniform, but both signs help equally (wrong for the proposed mechanism) and the effect equals the baseline's cross-run reproducibility. Needs 20 seeds with an interleaved control |
| `v7_bitsperparam.py` | Would a finer grid find basins the ±R corners miss? | **No at R≤1.2 — but see v9, this metric was wrong.** 1–2 minima either way, direction mixed; the coarse grid even invents one. Legs are shots, not grid points |
| `v8_attenuation.py` | Does the batching advantage survive to large M? | **Yes, it grows.** Signal decays exponentially in n but the rate falls as 1/N, so n\* ≈ 0.65M and circuits/gradient stays ≈1.5. Predicted n\*(N=8)=20.8 before running; measured 20.9 |
| `v8b_dense.py` | Does that survive **dense** coupling? | **Yes — my prediction failed.** All-to-all spin glass also shows cR²∝1/N, so the mechanism is the fraction n/M of coupled partners smeared, not locality. Dense fit noisier (9.6% vs 2.2%) |
| `v9_globalgrid.py` | Do extra bits help **wide-range** search? | **Yes — partly retracts v7.** At R=π/2, b=1 reaches −2.31 where b=3 reaches −4.31, and the box is multi-modal (1.7→3.3 minima). v7 tested only R≤1.2 and counted minima instead of best-point-found |
| `v9b_multiscale.py` | Can one circuit report two scales at once? | **Yes, as a diagnostic.** Per-bit Walsh ratio = 2.02 (linear) → 15.1 (broken), tracking cos(w,g) 0.9998 → 0.366. Measurable where gradient quality isn't → adaptive R. Fine bit gives smaller magnitude, **not** better direction |
| `v10_merge_paired.py` | Merged rotation, **paired** at 12 seeds | Re-test of v5_merge with both arms from identical initial params, cancelling the run-to-run drift that swamped the original |
| `v11_coherence.py` | How coherent is the walk in x? | **Locally yes, globally no.** Adjacent overlap 0.83 at R=0.6, antipodal 0.46 → 0.085 by R=1.0. Coherence and range pull opposite ways — closes HSP quantitatively |

## What is actually implemented in `nisq_v3.py`

Most entries above are *findings*, not code. The code changes are:

| change | status |
|---|---|
| `uncompute_w=False` (W† removal) | shipped, default |
| Suzuki-2 at `reps=2^a/2` in QPE sensing | shipped, default |
| `sense_moment_gradients` / `folded_spectrum_gradient` | shipped, opt-in |
| `boltzmann_step` / `decoder='boltzmann'` | shipped, opt-in, **guarded** (raises when 2ⁿ > shots/8) |
| `probe_linearity` | shipped, diagnostic only — **not** wired into the R schedule |

Everything else stayed documentation because it was negative, unresolved, or needed no code.

## Two traps these logs document

**Baselines that get free precision.** `StatevectorEstimator(default_precision=p)`
returns the exact expectation plus fixed Gaussian noise of std `p` — it never
samples, so it is blind to Var(H) and to Pauli grouping. This inverted a cost
verdict (`v4_cost.log` → `v4_cost2.log`) and silently subsidised the whole
benchmark by 23–57× on Heisenberg (`audit_benchmark.log`). Any comparison
involving an estimator: confirm it samples.

**Sub-2σ results on few seeds.** These runs are stochastic beyond the seed — the
seed fixes only the initial parameters. A 1.1σ effect on 4 seeds reversed on 6
(`v4_argmin.log` → `v4_softmin.log`), and `diag_sqrt`'s 0.9σ would have become
"the QFIM helps" had all four metric variants not been run. Replicate before
recording, and quote σ with the seed count beside it.
