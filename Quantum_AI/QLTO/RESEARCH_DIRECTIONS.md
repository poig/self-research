# A hierarchy of operations on parameter space

A forward programme. The record is
[`Application/RESEARCH_NOTES.md`](Application/RESEARCH_NOTES.md) and the theory
paper; this file says what can be built, what is closed, and why.

Every claim is tagged:

- **[measured]** — a number in this repo, log named
- **[proven]** — derived, no fitting
- **[literature]** — from a paper, with the paper named
- **[proposed]** — mine, unverified

The tagging is load-bearing. Several scalings quoted from memory in developing
this turned out wrong when measured — the chemistry group count `G`, the mapping
of the v9b per-bit ratio onto a two-circuit ratio, the fan-out prototype's depth
win. Treat **[proposed]** as a hypothesis with a named way to fail.

---

## 1. The organising frame

The W gate is a *controlled state preparation*: `|x⟩|0⟩ → |x⟩|ψ(θ_x)⟩`. Once that
exists, everything downstream is a question of **what operation you perform on the
parameter register**, and those operations form a hierarchy — each level needs
strictly more circuit structure, and each demands strictly more prior knowledge.
Like the Chomsky hierarchy, you cannot fake a level from below.

| level | operation on parameter space | must already know | structure | status |
|---|---|---|---|---|
| **0** | sample the landscape via marginals | nothing | superposition + measurement | **shipped** |
| **1** | phase drift + product mixer | the gradient (measured) | **separable** — `su(2)^⊗n`, classically simulable | **shipped**, and this is *why* it ties a classical decode |
| **2** | correlated drift + correlated mixer | degree-2 Walsh coefficients (measured) | non-separable, DLA beyond `su(2)^⊗n` | **untested** — T7 predicts both halves are needed together |
| **3** | eigenstate filtering / cooling | **the Hamiltonian only** | asymmetric reachable set via a filtered coupling | **not built** — the paper names it as the repair |
| **4** | amplitude amplification | how to *recognise* the answer | marking oracle | **closed** — circular for optimisation |
| **5** | period finding | a *deterministic* function with hidden structure | reversible arithmetic oracle | **closed** — expectation values are not deterministic |

Levels 4 and 5 are closed for different reasons, and both are proofs rather than
engineering gaps (§4). Level 3 is the frontier (§6). Level 1 is where the code is.

---

## 2. Level 1 is classical, and that explains a standing puzzle

The walk's drift writes phases `Σᵢ gᵢσᵢ` — degree-1, hence **separable**. Its
mixer is a *product* of independent single-qubit CRX rotations. Both terms are
single-qubit, so the parameter-register dynamics has DLA `su(2)^⊗n`, dimension
`3n` — **n independent qubits doing 1D walks**, generating no entanglement in
parameter space **[proven]**.

That is why Boltzmann decode *ties* the walk at half the circuits **[measured]**
(`v4_softmin.log`). Not a tuning failure: the walk ties a classical decode because
at level 1 it **is** classical. T11's measured interference (0.83 adjacent overlap)
is real and cannot pay, because a separable dynamics has nothing for it to buy.

**This also explains T7's unexplained negative.** Degree-2 drift alone did not
help because a product mixer structurally cannot convert pairwise phase
correlations into correlated population motion **[measured + reasoning in T7]**.
Level 2 is not "drift plus optionally a mixer" — it is a single upgrade with two
halves.

---

## 3. Where the field's boundary actually is

**Cerezo, Verdon, Huang, Cincio & Coles**, [arXiv:2303.09491](https://arxiv.org/abs/2303.09491)
(*Challenges and Opportunities in QML*) **[literature]**:

| advantage | status |
|---|---|
| exponential in **sample** complexity, **quantum data** | **proven, and un-dequantizable** — "impossible to find improved classical algorithms"; demonstrated on Sycamore |
| exponential in **time** complexity | *"more subtle... no rigorous proofs yet"* |
| ground-state properties of geometrically local gapped H | **"in the worst case, there is no exponential quantum advantage"** |
| classical data | *"no known exponential advantage"* — polynomial/quadratic only |

QLTO computes **ground-state properties of local Hamiltonians from classical
parameter data** — the intersection of rows 3 and 4, i.e. the two places the field
has *proven* there is nothing exponential to find.

Supporting bounds, both verified against the papers:

- **Larocca et al.**, [arXiv:2105.14377](https://arxiv.org/abs/2105.14377)
  (*Quantum* 6, 824): gradient scaling ↔ DLA dimension, and **no-go results for
  ground states of controllable systems**. The trainable/simulable trap is a
  theorem, not an inference **[literature]**.
- **Cerezo & Coles**, [arXiv:2008.07454](https://arxiv.org/abs/2008.07454)
  (*QST* 6, 035006): Hessian and all higher derivatives are exponentially
  suppressed on a plateau **[literature]**. **A correlated mixer cannot rescue a
  barren plateau** — there is no signal at any order. Parseval closes it from
  inside this framework too: `Var(E) = Σ_{S≠∅} Ê(S)²`, so a flat landscape means
  every Walsh degree is small, not just degree-1.

---

## 4. The borderline, stated mechanically

QLTO contains the whole advantage toolkit — superposition over configurations,
controlled state preparation, phase kickback, QPE, interference, Fourier readout —
and produces no advantage. The reason is two *separate* missing requirements.

**(a) Fourier-family advantage needs a deterministic function oracle.**
Shor, Simon and Bernstein–Vazirani all have `U_f: |x⟩|y⟩ → |x⟩|y⊕f(x)⟩` with `f` a
definite bit string. QLTO's objective is `⟨ψ|H|ψ⟩` — a *statistic of a
non-eigenstate*:

```
eigenstate      |E_j⟩|0⟩ → |E_j⟩|Ẽ_j⟩          deterministic, oracle-like, Heisenberg 1/T
superposition   Σc_j|E_j⟩|0⟩ → Σc_j|E_j⟩|Ẽ_j⟩  energy register entangled with eigenbasis
```

Reading the energy register on a superposition **collapses the state and returns a
sample from the spectrum, not the mean** — standard quantum limit, `1/√S`. VQE's
premise is that `ψ(θ)` is *not* an eigenstate. This is the Born rule, not
engineering **[proven]**, and the notes already state its consequence: Jordan
(PRL 95 050501) is not the citation, because no circuit writes an expectation
value into a register.

It also explains three measured things at once: QPE beating the Hadamard test at
3.1× depth ↔ 16× shots; the degree-0 energy carrying an irreducible bias floor
while the degree-1 *difference* does not; and the ε^−3 trade existing at all.

**(b) Learning advantage needs quantum data end to end.** The proven,
un-dequantizable separation requires quantum sensor → quantum memory → quantum
computer, with the information never becoming classical. QLTO is classical
parameters in, classical gradient out — the quantum part is a subroutine between
two classical interfaces.

**The Θ(1) is circuits, not information.** Each shot returns at most `n + κ`
classical bits (Holevo); M coefficients at precision ε need `M·log(1/ε)`. Nothing
is free, and nothing is violated. The circuit count is genuinely constant and the
**shots pay** — which is the standing limit on every direction below.

---

## 5. The constraint-relocation lens

**Yamaguchi, Rullkötter, Shehzad, Wagner, Tutschku & Kempf**,
[arXiv:2602.10695](https://arxiv.org/abs/2602.10695) (Feb 2026, IBM Heron-R2, up
to 154 qubits) **[literature]** — *encrypted cloning*: any number of **perfect**
clones, deterministically, if encrypted with a **single-use decryption key**.

It does not violate no-cloning. It **relocates where the constraint lives**.
Monogamy is intact — only one clone is ever decryptable, because the key is
consumed. What changes is the reading of the theorem:

> *quantum information can be spread at will, in theory and in practice, without
> dilution or degradation, if encrypted or obscured. **The actual constraint is
> that the decryption mechanism must be single-use.***

And the experimental finding sharpens it into a resource statement: degradation is
*"governed not by the number of encrypted clones but by the circuit depth"*.
Clones grow exponentially, the key grows linearly, and the cost is **depth**.

**THE TRANSFERABLE LESSON: a no-go theorem usually forbids an ACCESS PATTERN, not
an information structure. The productive move is to find which resource is
actually scarce and trade against it.**

### Application A — amplitude estimation on the marginal [proposed, buildable]

§4 identified QLTO's limit as `⟨H⟩` being a statistic of a non-eigenstate:
sampling, `1/√S`. Relocated, that is not a prohibition but a choice of estimator.

**Amplitude estimation gives `1/S`** — the standard-quantum-limit → Heisenberg-limit
quadratic **[literature]**, requiring coherent access to the state-preparation
unitary, its inverse, and controlled versions. **The W gate is exactly that**, and
W† is already implemented and documented as *required* for any interference-based
readout.

This project has already measured the trade in miniature: QPE beating the Hadamard
test at **3.1× depth ↔ 16× shots** **[measured]** is the same exchange applied to
the *energy readout*. What has never been done is applying it to the **marginal
estimation** itself — the marginal is currently a sample mean (`1/√S`), and the
coherent Loschmidt readout `Σ_x ⟨ψ_x|e^{−iHt}|ψ_x⟩|x⟩` would make it an amplitude.

It does not escape measurement theory. It moves the cost into **depth**, which is
where the hardware roadmap is spending — and depth is nearly free under the
billing model measured in `v21`.

**Way it fails:** the Loschmidt amplitude is `≈ 1 − it⟨H⟩` at small `t`, so the
signal is `O(t)` and may reproduce the Hadamard test's `1/τ²` variance problem,
eating the quadratic gain. That is the first thing to check, and it is checkable
analytically.

### Application B — the plateau theorem is about a MEASURE [proposed, needs literature check]

The barren-plateau result states that gradient **variance over uniformly random
parameters** vanishes exponentially. That is a statement about a *measure on
parameter space*, not about the landscape as an object.

Change the measure — structured initialisation, restricted manifold, problem-aware
ansatz — and the conclusion changes. This is presumably why *"problem-aware
ansätze and local cost functions"* are the standard mitigations, and why HVA and
warm starts work at all.

So the constraint may be relocatable exactly as no-cloning was: not *"the
landscape is flat"* but *"the uniform measure on a controllable manifold is
flat"* — naming a resource rather than a prohibition.

**CAUTION.** This looks obvious, which usually means it is either already in the
literature or already refuted there. Larocca et al.'s input-state dependence
result is adjacent and may cover it. **Check before treating it as an opening.**

---

## 6. The one theory opening: T8 supplies a condition the SOTA lacks

**Barthe et al.**, [arXiv:2506.17089](https://arxiv.org/abs/2506.17089)
(*Quantum Advantage in Learning Quantum Dynamics via Fourier coefficient
extraction*) **[literature]** — the same object QLTO computes, with a proven
separation.

Their **Corollary 1** is the analogue of T1: for a Pauli-encoded circuit and Pauli
observable, `f(α) = ⟨0|U(α)†PU(α)|0⟩` has a **finite** Fourier representation
`f(α) = Σ_{l∈L} b_l e^{iπα·l}`, `L = [−2L,+2L]^d`.

Mechanically they add a **frequency register** (`d⌈log L⌉+1` qubits), convert
data-encoding rotations into controlled increment/decrement on it, post-select the
all-zero ancilla to amplitude-encode the coefficients, and read individual `b_l`
by Hadamard test in `O(ε^{-2})` executions.

**The separation requires `d = O(log n)`** so the spectrum `(4L+1)^d` stays
polynomial. At `d = poly(n)` — QLTO's regime — efficient learning would break
**ring-LWE hardness**. So the Fourier→QML-advantage route is closed for
poly-parameter circuits, by complexity theory.

**But their Appendix G proves:** *poly-sparse frequency support ⟹ efficient
quantum PAC learning, with a classical separation.* And their only instance is
explicitly labelled **artificial** — contrived cancellations `Rz(αs)Y Rz(αs)`
leaving log-many effective parameters.

**T8 supplies that sparsity from physics.** For a Pauli-encoded gate `α_i` enters
as `e^{iπPα_i}`, so a frequency vector `l` with `|supp(l)| = j` **is** a degree-`j`
interaction — the support size *is* the Walsh degree. T8 proves a k-local
effective observable has degree ≤ k **exactly**, measured at `1e-32` — exact zero
in double precision — on blocks with no entangler after them **[measured]**
(`v5_locality.log`). Hence

```
support = Σ_{j≤k} C(d,j)·2^j = O(d^k)      polynomially sparse for constant k
```

**And the two conditions can be met together [proposed].** Their hardness lives in
the **fixed** prefix (`x` describes the fixed gates; Theorem 3 needs it BQP-hard);
the sparsity is a property of the **parameter** dependence. T8's mechanism
separates them: the effective observable is H conjugated by everything *after* the
parameter, so **parameters placed at the end of an arbitrarily deep BQP-hard
circuit** have effective observable H itself — k-local, support `O(d^k)` — while
the concept stays BQP-hard.

If it holds, that is an explicit physically-motivated family realising the
separation their kernel method needs and currently lacks.

**Three ways it fails, in order of likelihood:**
1. **The dequantization trap again.** Anything sparse enough to be tractable is
   often simulable enough to be classical. The proposed escape is that hardness
   lives in the fixed prefix, not the parameters — that must survive contact with
   their actual definitions.
2. **Appendix G's hypotheses.** I have the statement, not the proof. "Polynomially
   large support" may carry conditions the light-cone argument does not meet.
3. **The frequency/Walsh identification** assumes single-occurrence encoding. With
   reuploading (`L > 1`) a parameter recurs and frequencies extend to `[−2L,+2L]`;
   the identification needs redoing.

---

## 7. Level 3: the buildable synthesis

Every piece exists in the project and none of them have been assembled.

```
1. SENSE   one circuit → degree-1 AND degree-2 Walsh coefficients      free, by T2
2. BUILD   H_param = Σᵢ Ê({i})Zᵢ + Σᵢⱼ Ê({i,j})ZᵢZⱼ                    the landscape's own Ising model
3. COOL    filtered single-ancilla protocol on H_param                 Ding–Chen–Lin, not the current kick
4. READ    measure the parameter register
```

**Why it is not circular.** Level 4 (Grover) needs an oracle that *recognises* the
answer, which for optimisation requires already having it. Cooling needs only the
**Hamiltonian**, whose ground state is the answer. No marking, no circularity.

**Step 2 is measured.** Degree-1 + degree-2 is 99.6%+ of the local landscape, and
on two of four blocks **degree-2 exceeds degree-1**, at SNR 3–4 **[measured]**
(`v5_walsh.log`).

**Step 3 is what the paper proves the current walk is not.** An unfiltered
conditional kick has a reachable interval **symmetric about zero** `[−W*,+W*]`,
because the feedback generator's spectrum is symmetric and nothing breaks the sign
degeneracy **[proven]** — so it heats as readily as it cools. Ding–Chen–Lin's jump
operator `K = ∫f(s)e^{iHs}Ae^{-iHs}ds` with `f̂(ω)=0` for `ω ≥ 0` forbids energy
increments and makes the ground state an exact fixed point **[literature]**.

**This unifies three loose ends into one repair:** the paper's symmetric-interval
defect, T7's unexplained failure, and the walk's inability to beat a classical
decode all have the same cause and the same fix. A filtered coupling *is* a
non-product, energy-selective mixer — level 2 and level 3 are the same upgrade.

**Two ways it fails:**
- The filter is an integral over Heisenberg-evolved coupling operators and may
  cost more depth than the entire current sensing circuit.
- Couplings measured at SNR 3–4 may be too imprecise to cool against. A
  mis-specified Hamiltonian cools to the **wrong ground state**, and that failure
  is silent.

Whether this *outperforms* classical optimisation is the quantum-annealing
advantage question — genuinely contested. But unlike levels 4–5 it is **not closed
by an obstruction already measured**, and unlike the QML route it is not closed by
a complexity-theoretic no-go.

---

## 8. The engineering direction: model-free optimal control

**The review's own nomination for the nearest-term advantage demonstration**
**[literature]**:

> *"variational optimization at the continuous time optimal control level on
> analogue quantum simulators"*

— chosen independently by the authors who proved the bounds closing the
alternatives.

GRAPE and CRAB compute `∂F/∂u_j` by backpropagating through a **device model**.
The model is wrong — that is why one calibrates. This primitive measures
`∂⟨O⟩/∂u_j` for all `j` **on the device**.

| optimal control needs | this gives | evidence |
|---|---|---|
| loose precision, ε ~ 1e-2–1e-3 | wins exactly when ε > 1/M | **[proven]** |
| many parameters | Θ(1) circuits, advantage grows with M | **[measured]** T10 |
| no exact gradient | O(R²) bias tolerable | **[proven]** T3 |
| hardware truth | measured, not simulated | — |

**It is the only target where the method's weakness is irrelevant.** Chemistry
wants tight precision — the wrong side of ε > 1/M. Calibration wants loose
precision and many parameters — the right side.

**Already validated end to end**: the Hamiltonian-learning pivot is the same
construction with device parameters in the register — **no ancilla, no QPE, no
Trotterised sensing evolution** — at `cos(ĝ,∇_true) = 0.9993`, recovery to 0.034
worst coefficient in **30 circuits against 300** **[measured]**
(`v6_hamlearn.log`). The return-probability objective keeps the per-shot readout a
bounded bit, so T4's Bernoulli bound applies and the cross-coordinate variance term
is structurally zero **[proven]**.

**To build:** swap the W gate's encoding to pulse amplitudes (controlled
multi-qubit Pauli rotations decompose as `V; CRZ(2θ); V†` with `V` uncontrolled
**[proven]**, already built for the degree-2 test); keep the objective a **single
expectation** — a sum of squares is nonlinear and forfeits T2 **[proven]**.

**Check first:** the claim that GRAPE needs Θ(M) *device* gradients is
**[literature]** and unverified. Many implementations use adjoint methods on a
model, which is Θ(1) in simulation cost — so the honest framing is *model-based vs
device-measured*, not a complexity win.

---

## 9. What not to do

**Do not build the THC block-encoding.** It was proposed to rescue V3's chemistry
cost; that cost was never the problem, the arithmetic was. Measured directly,
`T ~ N^4.61`, `G ~ N^4.24`, so `T/G ~ N^0.37`, not `N^1.0` **[measured]**
(`v30_chemistry_scaling.log`). V3 already wins chemistry ~6× at N=12. THC would
still cut gates Θ(N⁴) → Θ(N²) **[literature]**, which helps *fidelity* — a weaker
justification than "it decides who wins."

**Do not claim Θ(1) circuits as novel.** Bowles et al.
([arXiv:2306.14962](https://arxiv.org/abs/2306.14962)) achieve `2B−1` circuits
**exactly** for commuting-block circuits **[literature]**, and SPSA is Θ(1) in M
as well. The defensible claim is narrower: **this stays Θ(1) in the presence of
entanglers**, where the exact protocol degrades to Θ(M) because `W̃ⱼ = GⱼWGⱼ`
becomes qubit-specific — bought by accepting an O(R²) bias.

**Do not chase QML advantage directly.** Closed twice over: by the two proven
no-gos in §3, and by ring-LWE hardness at `d = poly(n)` in §5. The contribution
available is a *theory* one — supplying someone else's sparsity condition.

**Do not build on the walk's necessity until `v20` settles it.** A classical step
on the same sensed gradient has never been shown to lose; the one row obtained had
the walk behind by 0.4σ.

---

## 10. The theorem this field actually needs

Every negative result read here is the same theorem in different clothes:

| result | statement |
|---|---|
| Larocca et al. | controllable ⟹ untrainable (DLA dimension) |
| Cerezo & Coles | no derivative order escapes a plateau |
| Barthe et al. | `log`-many parameters → separation; `poly`-many → breaks ring-LWE |
| Cerezo et al. review | classical data: no known exponential advantage |
| Bowles et al. | exact Θ(1) gradients — but only without entanglers |

**Enough expressivity to be classically hard costs you accessibility.** And every
one of these is **asymptotic and qualitative**. None of them prices the trade.

> **The theorem to solve: characterise the manifolds and measures on which a
> quantum model is simultaneously classically hard and variationally accessible —
> quantitatively, at finite resources.**

Larocca *starts* it — the DLA governs the trade — and stops at the asymptotic
statement. What is missing is the finite version: **how much trainability is
surrendered per unit of reachability.**

This project holds one exact finite-resource result that belongs to that
programme: the one-step reachable set is a **closed-form interval**, with
`W = ⟨Ψ₁|A − U†AU|Ψ₁⟩` verified to `3e-15` and vanishing identically when
`[G,H]=0` **[proven]**. That is a finite-time reachability statement where the
field has only asymptotics.

**The cheap first step remains:** extend the one-step interval to **two steps
analytically**. If the two-step set is not closed-form, this programme stalls, and
that is a day's work to establish rather than a quarter's.

And §5's lens applies here too: if the plateau results are measure-dependent
rather than landscape-dependent, then "quantitative trade-off" means *characterise
which measures are flat*, which is a strictly more tractable question than
characterising which landscapes are.

---

## 11. Ranking

| | direction | level | why | risk |
|---|---|---|---|---|
| **1** | model-free optimal control | 0–1 | primitive validated; regime cooperates; the review's own nomination | encoding may not fit pulse parameters |
| **2** | filtered cooling on `H_param` | 3 | unifies three loose ends into one repair | filter depth; SNR 3–4 couplings may be too coarse |
| **3** | T8 → Barthe sparsity | theory | supplies a condition the SOTA explicitly lacks | dequantization trap |
| **4** | correlated mixer + degree-2 drift | 2 | subsumed by (2) — a filter *is* a correlated mixer | do not build separately |

Direction 1 first: it does not depend on resolving V3 vs V4, chemistry scaling, or
whether the walk earns its circuit.

---

## 12. Standing caveat

Three results in this project were withdrawn to one cause: a harness whose "paired
seeds" pinned only the initial parameters while sampling stayed unseeded,
producing up to **3.3σ on a provably null comparison** **[measured]**
(`v26`/`v27`). Seeding is implemented but verified on only two of three problems —
Heisenberg still desynchronises.

Before any direction here produces a claim, run its harness on **two things that
cannot differ** and see what it reports. A comparison is only as good as its null.
