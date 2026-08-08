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
| `v4_cost2.py` | Is QLTO cheaper, baseline charged honestly? | ~~Yes: 3.2× fewer shots~~ — **shot claim withdrawn by `v14`**, it normalised each method by its own target. 48× fewer circuits stands. Hadamard **loses** on shots |
| `v14_oracle_curve.py` | Gradient quality vs cost, both against the *same* target | QLTO **plateaus at cos ≈ 0.98** — bias, not variance. Parameter-shift has no floor and overtakes near **8k shots**. Circuit count is the only win |
| `v15_classical_overhead.py` | Is the circuit saving repaid in local CPU? | **40–100× worse** — but decode is *cheaper* (1.1 vs 3.4 ms); it is all rebuild+transpile, fixable with a parameterised template |
| `v16_hardware_currency.py` | Circuits, depth and 2q gates per gradient | Circuits **0.03–0.13×**, depth **19–141×**, 2q gates **5–40×**. The saving is bought, not free |
| `v17_depth_vs_ancillas.py` | Is the depth structural? | **Yes** — doubles per ancilla, the (2^k−1)·τ₀ ladder. k=1 is cost-competitive (1.2–3.6× 2q gates) but that is the Hadamard test, which loses on variance |
| `v18_v4_direct_readout.py` | Does the circuit-count win need QPE at all? | **No.** Direct Pauli readout: same win, depth **19–141× → 1.5×**, total 2q **0.28–0.46×** of parameter-shift, cos equal or better. QPE only ever bought G-independence |
| `v19_complexity_growth.py` | Whose cost *grows* slower? | Fitted α: total 2q **p-shift 2.20, V3 1.26, V4 1.06**; circuits **0.00** for both QLTO vs 1.00. V3 depth α=1.27 — grows faster than the problem |
| `v20_walk_shot_tolerance.py` | Does the walk earn its circuit at low shots? | **One row only** (job stopped). N=4/256 shots: walk beats Boltzmann by 2.5σ (the nonlinear collapse T2 predicts) but **loses to a classical step by 0.4σ** |
| `v21_billing_models.py` | What do vendors actually charge for? | **Circuits, not depth.** IBM: `(rep_delay + circ_len)×circuits×shots`, rep_delay 250 µs ≫ circuit duration. Braket: per-task + per-shot, **no depth term**. Reverses the depth verdict |
| `v22_v4_accuracy.py` | Does V4 optimise, or only look good on cosines? | **3 ties, 1 loss.** MaxCut N=4 at 3.0σ — but see `v27`: the harness produces 3.3σ on a null, so **this result is withdrawn as evidence** |
| `v22b_decode_sanity.py` | Is the V4 decode right, or a plausible-looking bug? | **Right.** At R=0 measured ⟨H⟩ matches exact within 1σ shot noise; **signed** cos +0.978–0.998, which is what tests bin ordering (a swap flips the sign, not \|cos\|) |
| `v23_v3_depth_source.py` | Is V3's Θ(N) depth the shared ancilla? | **Inconclusive, not refuted** — controlled and uncontrolled both grew, because the term-ordering artefact dominated both arms and masked the effect |
| `v24_term_ordering.py` | Is the depth just the term ORDER? | **Yes.** Chain order serialises every consecutive bond. Layer-sorting: depth **N^1.25 → N^0.00**, flat at 225, *fewer* gates, unitary error 0.00e+00 |
| `v24b_sorted_sensing.py` | Same, on the full sensing circuit | **N^1.22 → N^0.64**, 1.4–2.7×. Residual is the ancilla critical path + the ansatz CX chain. cx unchanged → **money fix, not fidelity fix** |
| `v25_ancilla_fanout.py` | Can fan-out flatten the residual? | **Yes, exactly.** Depth **N^0.55 → N^0.03**, unitary error 0.00e+00, for +7.6% gates and N/2 helper qubits. *Not yet wired into `nisq_v3`* |
| `v26_fix_validation.py` | Do sorting and κ reduction survive end-to-end? | **κ=3 is free** — every arm inside noise at **half** the gates, survival 0.009→0.098 at Heis N=6. κ=2 tempting (0.26× gates) but Heis N=6 regresses +0.29. **Now the default is κ=3** |
| `v27_sort_exactness.py` | Sorting was 2.2σ/3.3σ "worse" in v26 — real? | **No — and it calibrates the harness.** All 6 configs **0.000e+00: identical unitaries**. So v26's sorted arm was a NULL experiment that returned 0.2/2.2/3.3/1.9σ. **Two of four exceeded 2σ with nothing to detect** |
| `v28_seeded_null.py` | Does `sim_seed` make the null return *exactly* zero? | Acceptance test for the seeded sampler — criterion is bit-exact 0, not "smaller σ" |
| `v29_fanout_wired.py` | Does fan-out deliver once wired into `nisq_v3`? | **No — negative result.** 1–21% depth for **4–16% more gates**. The prototype isolated one stage; sorting and κ=3 had already taken the parallelism. **Stays off by default** |
| `v30_chemistry_scaling.py` | Is chemistry really T=Θ(N⁴), G=Θ(N³)? | **G was wrong.** Measured **T~N^4.61, G~N^4.24, T/G~N^0.37** — not N^1.0. Qiskit's greedy QWC compresses only ~3×, so **V3 wins chemistry 6.3× at N=12** and widening |
| `v35_reps_scaling.py` | Does more ansatz depth help QLTO? | **WITHDRAWN — the comparison is not sound.** Its "ceiling" used exact statevector energies and 600 BFGS iterations from 6 restarts while QLTO ran at 8192 shots for 20 epochs, so the "optimiser gap" mixes shot noise, budget and blindness. Its `deg3+ = 0.238` is a 400-sample regression artefact; exact enumeration gives 0.6–6%. Superseded by `v35b`/`v35c` |
| `v36_walk_transfer_function.py` | What map does the walk implement? | **The question nobody had asked.** Non-monotonic (at g=−1.0 the step goes the *wrong way*), non-separable (spread 0.29 on a step bounded by R=0.6), tanh surrogate fails at **46% residual**. Motivated the whole `v37` family |
| `v37b_walk_closed_form.py` | Write the walk down exactly | **anc=0 branch is exactly the identity**; anc=1 projects onto (I−U)\|ψ⟩/2 with U=⊗Uᵢ a product of 2×2 rotations. Drift angle ≈ 23.9·g ⇒ **periodic in g**, 10 sign crossings over \|g\|≤1.6. Wrap spacing **0.343**, not the naive π/23.9 = 0.131 — β tilts the axis |
| `v37c_walk_ablation.py` | Is that derivation right, or just plausible? | **Right: 0.00241 vs the simulator** on the identical circuit. Ladder: bare 0.00241 → +W gate 0.06289 → +energy imprint **0.32490**. The imprint carries the *values* and **flips signs**; the bare model carries the *structure*. Arm C reproduces `v36` to ~0.007 |
| `v37d_reset_closed_form.py` | Does resetting the ancilla remove the wrap? | **Yes.** Shared ancilla ⇒ steps compose as a product of **rotations** (angle adds, wraps); reset ⇒ product of **channels** ρ→(ρ+VρV†)/2 (contracts, cannot overshoot). Crossings **10 → 0**; base also aliases in *k* (−0.411, +0.426, −0.414, +0.417, +0.428, −0.421) where reset converges |
| `v37e_two_fixes.py` | Rescale the drift, or reset the ancilla? | Both kill the wrap; they buy different things. rescale: 0 crossings, 0 turns, corr **0.98**, knee **1.28**, costs *one constant*. reset: 0 crossings, best separability (0.067 vs 0.147), but knee **0.26** — below the operating range \|g\|=0.58–0.97, so it is bounded **sign descent**. Neither model settles the descent *sign* (the imprint flips it) |
| `v37_ancilla_reset.py` | Reset the ancilla each step, on the real circuit | **Splits.** Reset is **0.52 ahead at epoch 3** (null scale is 0.03–0.09) then plateaus and ends 0.08 behind (inside it). Fast start, no fine-tuning — the knee-at-0.26 signature. *Step size not controlled — see `v37g`* |
| `v37f_drift_scale_sweep.py` | Does rescaling the drift help? | **WITHDRAWN as a test of the fix.** Monotone collapse below scale 0.25, but scaling the drift down also shrinks the step 7.6×, so this compared "same schedule, smaller steps" over a fixed epoch budget. The `OPEN/schedule` entry warned about this confound *in these exact words* and the run reproduced it. Superseded by `v37g` |
| `v37g_step_matched.py` | Both fixes with dt swept and `\|move\|` reported | **The wrap is performance-neutral.** At `\|move\|` matched to 0.0004, rescale returns −5.9875 vs base −5.9871 — identical, so `v4_schedule2`'s "lost 0.146" was step size all along. What survives: at `\|move\|` matched to **0.0001**, reset is **0.567 ahead at epoch 3** but finishes worse with **9× the variance**. Fast, coarse, unreliable endgame = bounded sign descent. *`reset_full` not yet matched* |
| `v38_degree1_target.py` | What corner can a product mixer reach? | **The degree-1 argmin — wrong on 7/16 blocks**, regret to **0.889** of the hypercube range. The degree-≤2 target is exact everywhere (`regret2 = 0.000`). A *reachability* failure, so outside what Cerezo & Coles forbids |
| `v39_global_mixer.py` | Does Grover's diffuser fix it? | **No.** 14/40 → 14/40 sign errors. *And the circuit wasn't Grover* — the oracle was applied once, not alternated |
| `v39b_alternating_oracle.py` | Add the oracle–diffuser alternation | **Still no.** 14/40. Product mixer with alternation: 12/40, marginal |
| `v39c_raw_distribution.py` | Does the walk concentrate at all? | **Yes — up to 6.6× over uniform.** And it separates *perfectly* on v38's verdict: enhance 1.5–6.6 with mode = x\* where degree-1 is right, ≤1.55 and mode ≠ x\* where it's wrong. The walk amplifies the **degree-1 argmin** |
| `v41_oracle_balance.py` | Drift or imprint — which is the oracle? | **The drift.** Remove it and the distribution is *exactly* uniform (H/Hmax = 1.000); 10× imprint doesn't move it. corr(P,−E) = 0.45 with drift, 0.07 without |
| `v41b_uncompute_oracle.py` | Does W† connect the energy oracle? | **No, and provably not** — W is controlled on param, hence block-diagonal there, so it cannot move param populations. Identical to 4 decimals |
| `v41c_walk_quadrature.py` | Is the walk's missing `sdg` the cause? | **No.** corr 0.0673 → 0.0690. (Still a real inconsistency with the sensing path) |
| `v42_degree2_drift.py` | T7 remeasured on the oracle metric | **Worse, 4×**: corr 0.4535 → 0.1170. Degree-2 coefficients come from the *same shot record* — no extra circuits — but feeding them in degrades the oracle |
| `v42b_bounded_phase.py` | Was the unbounded phase channel the cause? | **Only for degree-1.** Bounding the span to π lifts degree-1 **0.4535 → 0.5352** (100% sign-consistent) but costs enhancement (2.036 → 1.177). Degree-2 never recovers |
| `v42c_deg2_estimator_quality.py` | Are the degree-2 coefficients just noisy? | **No.** cos(sampled, exact) = **0.98–0.9997** on every block with appreciable degree-2 weight |
| `v43_phase_offset.py` | Does `sin²` folding explain it? | **No.** Best 0.4957, below v42b's 0.5352. **Seventh consecutive falsified hypothesis**; the shipped config still has the best enhancement and mode hits of anything tested |
| `phase_degree_bound.py` *(theory_test)* | What can a degree-d phase concentrate? | **Exact bound: `max enhancement(n,d) = Σⱼ≤d C(n,j)`** — the dimension of the degree-≤d polynomial space. Degree-1 caps at **n+1** against 2ⁿ. A phase *proportional to energy* caps near n even at full degree, because `sin²(∝E)` is a Boltzmann reweighting — which is why a Boltzmann decode ties the walk |
| `v45_threshold_degree.py` | Which problems have low-degree good-set indicators? | **Two regimes.** m=1 is a *mathematical identity* — `1[x=x*]` has a flat Walsh spectrum for every landscape (`4/15`, `6/63`, `8/255` verified), so marking a unique optimum is universally hard. At **m>1** physical landscapes carry **4–5× more degree-1 weight than random**, `d90` one lower — the first quantitative evidence these landscapes are non-generic |
| `v46_set_target.py` | Score the walk against a *reachable* target | **The walk is at 41–67% of its own degree-1 ceiling** (2.035 vs 5.000 at m=1), rising with set size. So there is ~2.5× headroom and none of the nine interventions found it — "it was saturating a bound" is **not** the explanation for those failures |
| `v47_optimal_phase.py` | Write the optimal degree-1 phase instead of the energy truncation | **Returns uniform (0.92–0.99).** The pure-phase model it was designed against is wrong once the imprint is on |
| `v48_full_closed_form.py` | Add the imprint as `P ∝ 2−2Re[e^{iφ}c_x]` | **Also wrong** — assumes the walk is a pure phase on param, but CRX moves populations. Did measure `\|c_x\| ≈ 0.2`: the interference contrast is only ~20% at the shipped evolution time, independently explaining why the shortest anneal gave the best oracle |
| `v49_complete_model.py` | The complete model — mixer + W + imprint | **Validates exactly on some blocks: ratio 1.00 / 0.84, corr 0.9998 / 0.9999** against TVD-from-uniform of 0.31–0.34. The missing piece was the Gram structure `⟨ψ_x'\|ψ_x⟩` — the vertex states are *not* orthogonal. Remaining discrepancy is block-dependent and shape-preserving (corr 0.99, ratio 8.4), pointing at the walk unitary's magnitude. *Note: the kron order must be `V_{n-1}⊗…⊗V_0` — Qiskit puts qubit 0 as LSB* |
| `v49b_exact_statevector.py` | Validate the complete model against the exact unitary | **TVD 0.00000, corr 1.0000, every block, every arm.** `P(y) ∝ ‖\|ψ_y⟩ − Σₓ V_yx U_t\|ψₓ⟩‖²`. Three fixes were needed: the **Gram overlaps** `⟨ψ_x'\|ψₓ⟩ ≠ δ`; **kron order** `V_{n−1}⊗…⊗V_0`; and `\|ψₓ⟩` taken **from the W gate**, since controlled Z-rotations carry an x-dependent global phase that becomes relative inside the superposition — which is why RY blocks matched and RZ blocks didn't |
| `v50_design_on_true_model.py` | Degree-2, asked properly against the validated model | **The drift is 2.42× off its own optimum** (shipped 2.545 vs opt-d1 6.164 at m=1) — the energy truncation is a suboptimal phase, a distinction never drawn before. **Degree-2 adds only 4–6%**, which settles T7 for a defensible reason. *Caveats:* opt-d1 exceeds the `Σ C(n,j)` ceiling, so that bound doesn't apply to this circuit; and computing the optimum needs classical simulation of the ansatz, so it's an in-principle result, not a deployable design |
| `v51_optimise_mixer.py` | Finish `v7`: is the uniform β mistuned? | **Yes, by 85%.** Optimising `β_s` alone is worth **1.85×** with the drift untouched — `v7`'s suspicion confirmed, using the deterministic model so no seeds are needed. And the knobs **interact**: `both/drift = 1.19`, so drift and mixer can't be tuned one at a time. Total **2.83×** over shipped. *Makes `v39`'s "diffuser changes nothing" unsafe — it compared two mixer families on a ramp 85% off optimum* |
| `v53_walk_necessity.py` | Does the walk earn its circuit at **wide R**? | **No — its last regime is gone.** At R₀=π/2 where `v9_globalgrid` puts the box multi-modal (1.7→3.3 minima), `gradstep` wins **3 of 4** rows; the one walk win (1.4σ) failed to replicate at 4× shots. Even unguarded `boltz` beat the walk in 3 of 4. Also measured: **the walk plateaus** — 4× shots moved it −0.04 while `gradstep` gained +0.24 |
| `v53b_reconcile_v20.py` | Does v20's logged row still reproduce? | **No.** Walk arm off by **+0.739 (3.7 SEM)**; boltz/gradstep ~+0.2. `supplement/results/` is written against a moving `nisq_v3` — re-measure before quoting |
| `v53c_merged_walk_low_shots.py` | Is `merged_walk` the cause? | **No, exonerated.** −0.168/−0.074/+0.054 at 256/1024/8192 shots, all <2σ, sign flips. Cause of the drift still unidentified |
| `v54_benchmark_ab.py` | walk vs gradstep on the **full suite** | **Walk wins 0 of 7.** `gradstep` wins LiH (+0.108) and Heisenberg N=8 (+0.254) — the two largest — ties the rest, at **1.8× fewer circuits**. Decisive: parity already favours `gradstep` since it costs half |
| `v55_level2_walk.py` | The 2×2 factorial: correlated drift × correlated mixer | **T7 falsified.** Arm D (both halves) reaches corr 0.37 against Arm A's 0.70 — the correlated mixer does **not** rescue degree-2 drift. "Level 2 is a single upgrade with two halves" was the prediction; supplying both halves doesn't recover it. Arm C (mixer alone) gives the best correlation seen (0.77) at lower enhancement — the same fidelity/strength tradeoff as v42b |
| `v56_system_cooling.py` | Filtered quantum cooling on the system register | **Failed.** Energy pinned at 3.0000, fidelity 0.0000, every step and every M. `CumP_succ` decays 1.7e-2 → 5.7e-15 — the exponential post-selection death |
| `v57/b/c_pulse_control.py` | Path B — model-free pulse calibration | **Failed.** QLTO vs finite-difference gradient at **cos −0.41** (anti-correlated); fidelity 0.934 → 0.471. The log's closing "8.0× circuit reduction" is not a result — the run went backwards. Third file in the series, so the sign was already suspected and the fix didn't take |
| `v58_qng_geometry.py` | Quantum natural gradient vs Euclidean | **QNG loses everywhere.** "Euclidean faster" on every epoch of every problem, and the gap **widens with N** (Frustrated N=6: −3.74 vs −2.96). Confirms `WHY NO QFIM` empirically |
| `v61_ae_signal_amplitude.py` | Roadmap action item 3 — does Porter-Thomas kill AE's signal? | **No, the caveat was misaimed.** `P(good)` holds at **0.21–0.30** across reps 1–5 and N=4,6, while single-bitstring probabilities sit at 1e-2→1e-3 and fall with N. Bitstring probabilities concentrate; **two-bit marginals do not**. Doesn't revive AE — the T2 forfeiture is the reason it was demoted |
| `v59_xx_correlator.py` | The roadmap's last unrun experiment — X–X as a connectivity witness | **Dead, with a proof.** `⟨Xᵢ⟩ = cos(R)` to **1.1e-16** (diagonal dead, as the roadmap's calculation says) and the connected `C_ij` is **exactly 0** everywhere. Two reasons: downstream circuitry cancels (`V†V = I`, so the overlap cannot see entanglers at all), and the ±R sign symmetry averages `⟨GᵢGⱼ⟩` to zero over the four sign combinations. **Closes the QFIM direction entirely** |
| `v60_pathA_claims.py` | Path A's three quantitative claims, measured | **Claim 1 false as stated** — `L = 2(r+1)`, so circuits/epoch is `G·M/N`: constant in **N**, linear in reps, not "≈1.5G constant in M". The 1.5 describes T10's optimal blocking, which `_layers()` doesn't implement. **Claim 3 holds for `direct`** (ansatz + 7…11), fails for `qpe` (274→658). **Claim 2 caught a G-factor bug in v5** — `sense()` averaged over commuting groups where the energy is a sum, pinning relative error at 2/3. Fixed; `err·√S` now flat at 6.6–7.2 |
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
| `grad_step` / `decoder='gradstep'` | shipped, opt-in. **Never measured behind the walk** — see `v20`, `v53`, `v53b`, `v53c` — at half the circuits, and immune to `anomaly_c`'s per-block scale bias because it normalises by `max\|g\|` within the block. **Not the default**: wide R and N≥6 are untested, and the headline benchmark was run with the walk |
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

**…and the threshold is far above 2σ — now measured, not guessed.** `v26` ran
`base` against `term-sorted`, which `v27` then proved are **the same unitary to
0.000e+00** at every problem and every κ. That arm was therefore a **null
experiment**, and it returned:

| problem | "effect" on identical circuits |
|---|---|
| H2 | 0.2σ |
| MaxCut N=4 | **2.2σ** |
| Heisenberg N=4 | **3.3σ** |
| Heisenberg N=6 | 1.9σ |

**Two of four exceeded 2σ with nothing to detect.** The cause is that "paired
seeds" pin only the *initial parameters* while the sampling stays unseeded, so
the pairing never removes the dominant variance and every σ here is understated.
Consequences: treat 2–3σ at six seeds as consistent with zero; `v22`'s MaxCut
3.0σ is withdrawn as evidence; and **seeding the sampler** would make every
future A/B in this project roughly an order of magnitude more sensitive — it is
worth more than any single result in this table.
