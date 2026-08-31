Repository: self-research
=========================

[![DOI](https://zenodo.org/badge/1079381087.svg)](https://doi.org/10.5281/zenodo.17834056)

This repository will contain multiple research projects (one per subdirectory). See the per-project README files for details.

Current projects:

- `Quantum_AI/` - Include all my Quantum AI related project (see `Quantum_AI/README.md`)
- `Quantum_df/` - Calculating Pi on Quantum Computing (see `Quantum_df/README.md`)

How to read this repository
---------------------------

Two rules govern the work, and both are stated in full in [`CLAUDE.md`](CLAUDE.md):

**R1 — Circuits, not matrices.** Every quantum construction is a Qiskit
`QuantumCircuit` run through a sampler with finite shots. NumPy/SciPy analytic
evaluation is a labelled fallback and never the source of a headline number.
Experiments are classified as tier A (circuit + shots, may support any claim),
tier B (circuit built, exact amplitudes — mechanism only), or tier C (no circuit —
scoping only, labelled `NO CIRCUIT`). The rule was written because it was measured
twice: the same construction gave 0.13% on classical amplitudes and 3.0% once
built as a circuit, and building it exposed two endianness bugs that dense
matrices hide.

**R2 — Withdrawals stay in the record.** Claims retracted after testing remain
documented beside the claim, with what refuted them. Several results here were
withdrawn by their own author — the theory strand's `ΔE ≤ η·I(S:A)`, the DLA
efficiency transition, four separate claims about the device-calibration line —
and those withdrawals are the reason the rest is trustworthy. **Read them before
citing anything.**

Current work is the device-calibration line in `Quantum_AI/QLTO/Application/`;
`supplement/` holds ~106 numbered experiments, each with its committed log, and
every number in the READMEs cites one.

Permissions & license
---------------------

This repository contains self-research projects created by the author (Tan Jun Liang). There are no external contributors. Reuse, redistribution, or republishing of any portion of the work (code, notebooks, or narrative) requires explicit permission from the author.

To request permission or discuss reuse, please contact the author via the contact in github profile page.


