# Roadmap

The harness works and is tested. What is missing is everything that turns a working harness into a benchmark other
people can trust: public data, a frozen scoring protocol, and a way to publish results that nobody has to retype.

This document is the plan for that, in the order the work unblocks other work. Status markers are honest:
**done**, **in progress**, **next**, **later**.

## Anti-goals

Worth stating first, because they shape every decision below.

- **Not a general MLIP accuracy benchmark.** Near-equilibrium energies and forces are measured well elsewhere. This
  repository asks one question — what happens beyond the cutoff — and declines to average that into a single score
  with unrelated quantities.
- **Not a single-number leaderboard.** The headline quantities are diagnostics with known correct values
  (tail fraction 1, ε 0, ΔΔE slope 1). Collapsing them into one ranking would destroy exactly the information that
  makes them useful. Tables stay multi-column.
- **Not a dataset repository.** Reference data is published as versioned datasets with their own DOIs and licences;
  this repository holds the code that scores against them, plus one toy set so the tests can run anywhere.
- **Not a model zoo.** Adapters wrap *released* models so they can be compared fairly. Weights stay with their
  authors, under their own licences.

## Phase 0 — a stranger can run it (now)

- [x] Self-contained toy set (`lrbench toy`), so install → first number needs no download and no GPU
- [x] Metrics, normalised errors, conformal calibration, equivariance probes, timing, chemiscope export
- [x] Model adapters for released long-range models (SO3LR, SO3LR-Tauri, MACE-POLAR, MACE-MH, MACE-OMOL, MACE-LES, CACE-LES)
- [x] Multipole oracle: the achievable ceiling, so a model's failure can be told from a metric's
- [x] BSD-3 code / CC0 data split, docs, contributing guide
- [ ] **next** — Release the canonical sets as a public HuggingFace dataset (CC0) with a dataset card, per-set
      provenance and a Zenodo DOI. Further sets to follow from the CSCS runs.
- [x] Converters for the public external sets: GSCDB137 noncovalent (45 subsets), NCI Atlas, the BFDB biodimers and
      the dimer overlays, each recording its own licence, reference level and checksums
- [x] `lrbench priors`: twenty pass/fail checks of physical, model and data assumptions, with planted-defect tests
- [ ] **next** — `lrbench fetch <set>`: wrap those converters so the download, the checksum and the conversion are
      one command instead of a paragraph of instructions
- [x] Make the level-of-theory refusal fire for reference-only sets: records referenced by a group with a `ref` count,
      not only records carrying an `energy`, so `ext_*` cannot be scored cross-level without the explicit flag
- [ ] **next** — PyPI release (`pip install longrangebench`) and tagged versions
- [ ] **in progress** — CI on 3.11–3.13 plus a weekly job that checks every adapter still imports against current
      upstream releases (adapters rot silently otherwise)
- [ ] **later** — A docs site built from `docs/` rather than a folder of Markdown

## Near term — checks a model cannot pass by accident (next)

**Why.** In November 2026 an LR model with a link-state mixture (PET-OMol-S plus a long-range head whose Coulomb
term is masked by a backbone link-state mixture; internal name FC3b) passed every check this suite runs on it, yet
it had defects found only outside the suite:

- **Wrong-sign mean force on NH₄⁺···Gdm⁺.** Under thermal sampling at 8.5 Å centre-of-mass separation, the mean
  force is −0.25 ± 0.09 and −0.15 ± 0.08 eV/Å for two training seeds. Bare Coulomb is +0.20 eV/Å. All 8 of 8
  replicas have the wrong sign.
- **The checks it passed:** `conservativeness` (median 0.010), `seam`, `energy_force`, `nve_short`, and its own
  single-orientation rigid scan (+0.19 eV/Å at that separation).
- **Cause: a static, orientation-dependent error.** In a 200-orientation rigid scan, a backbone link joins the two
  cations in 68 % of orientations at 8.5 Å, where the link switches on. Linked orientations average −0.13 /
  −0.06 eV/Å and unlinked ones +0.19. About half of all orientations have the wrong sign. The error is the
  derivative of the link mask acting on the masked Coulomb energy, so it is a property of the energy surface, not
  of the dynamics.
- **One orientation and a median cannot see it.** The fine scans use one orientation and score medians. Every
  median, of the conservativeness ratio and of the scan force, lands on the unlinked majority or on a scan that
  happens to stay unlinked.

**The same models show defects that no check here exercises:**

- **A localised seam in a model without the mixture (FTq13).** Its NH₄⁺···Gdm⁺ rigid scan at 11–11.5 Å
  centre-of-mass separation has f·r²/(k q_A q_B) jumping to −12.6 and +12.7, against 1 for Coulomb. Thermal
  integration there leaves the plotted range at 10.5 and 12.75 Å. A median over a scan does not see a spike
  this narrow.
- **Code path and topology.**
  - In a 5k-atom open droplet, the energy jumps by ±0.49 eV whenever a molecule splits off. The cause: the
    model's general path sums the energy in fp32 and its cached path in double. They differ by up to 0.19 eV on
    the same 3,000-atom geometry, while the forces agree.
  - Two Na⁺(H₂O)ₙ ··· Cl⁻(H₂O)ₙ droplets (902 atoms) drift by +0.75 [0.66, 0.85] meV/ps/atom (FTq13). The drift
    arrives as repeated ±0.1 eV single steps.
  - Nothing in this suite evaluates more than two fragments, the same structure through two code paths, or a
    fragment leaving a cluster.
- **Condensed phase.** In water at 1.00 and 0.30 g/cm³, in 1 M NaCl(aq) and in rock salt, the link graph
  percolates and the model's long-range term is exactly zero. No set here would notice that.

**Proposals.** Each has its own issue. They run in the order **A → B → C → D**, while the Phase 0/1 release work
continues alongside.

- [ ] **A (1 + 5)**, **next**: **oracle-scored orientation sampling, plus hardening of the existing checks**
  ([#28](https://github.com/EricBoittier/LongRangeBench/issues/28), [#29](https://github.com/EricBoittier/LongRangeBench/issues/29)).
  - **Orientation sets.**
    - Sample hundreds of rigid orientations per dimer pair at centre-of-mass separations of 10–12 Å and
      beyond, scored against `oracle:multipole`. The oracle gives a reference for any orientation without new
      DFT.
    - Add thermal snapshots.
  - **Metrics.** Wrong-sign fraction of the inter-fragment force, and the q95 and maximum error, not only the
    median.
  - **Adversarial search.**
    - Maximise |F_model − F_oracle| over rigid-body orientation and distance.
    - Output the worst geometry as extxyz, so a failure arrives with its witness.
  - **8–10 Å.** The oracle is validated against DFT on a subset first. DFT is added only where the two disagree.
  - **Hardening (5).**
    - Report q95 and max per class beside every median in `conservativeness`, `seam` and `virial_tail`.
    - `nve_short` with molecular ions, several orientations, and launches through switching windows. It scores
      the largest single-step energy jump as well as the drift.
    - `energy_force` and `seam` on the proposal-1 orientation sets.
- [ ] **B (3)**: **reference-free topology and many-fragment checks**
  ([#30](https://github.com/EricBoittier/LongRangeBench/issues/30)).
  - **Dilute ion gas.** k = 2…20 charged fragments, each at least 12 Å from the others. The exact energy is the sum
    of the monomer energies plus the Coulomb sum.
  - **Split/merge continuity.** A fragment pulled out of a cluster, and two droplets merging. The energy must be
    continuous and the work must close.
  - **Code-path consistency.** The same structure must give the same answer whether evaluated alone, with a
    distant spectator, batched or permuted.
- [ ] **C (2)**: **a thermal mean-force task** ([#31](https://github.com/EricBoittier/LongRangeBench/issues/31)).
  - **Systems.** NH₄⁺···Gdm⁺ plus one opposite-charge pair.
  - **Protocol.** 7 constrained windows (6.6, 7.2, 7.9, 8.5, 9.2, 12.75 and 15 Å centre-of-mass separation),
    1 ps each, forward and backward sweeps, 2 seeds.
  - **Scored.** Mean force against Coulomb, sweep hysteresis, path-work residual and temperature.
  - **Cost.** About 20 minutes per model. It is the test that found the defect above, and it needs no DFT.
- [ ] **D (4)**: **condensed phase with known physics** ([#32](https://github.com/EricBoittier/LongRangeBench/issues/32)).
  - **Finite-size slope.** The energy of a charged periodic system, against −ξq²/(2εL).
  - **Stillinger–Lovett.** The small-k behaviour of S_ZZ(k).
  - **Internal consistency of ε.** The finite-size slope against dipole fluctuations.
  - **Charged slab.** The field across it.
- [ ] **Protocol v1, the public data release and the contamination declaration (Phase 0/1)**
  ([#33](https://github.com/EricBoittier/LongRangeBench/issues/33)).
  - **v1 freeze.** Freeze v1 once A and B have landed, so that their thresholds are part of v1 rather than the
    first reason for a v2.
  - **Data release.** The HuggingFace dataset and DOI do not depend on A–D and can go first.

## Phase 1 — results are comparable (next)

A benchmark whose metric definitions drift is a benchmark whose numbers cannot be compared across papers. The fix is
a protocol that is explicitly frozen and explicitly versioned.

- [ ] **next** — **Protocol v1**: pin the set revisions, the bin edges, the 8 Å / 12 Å thresholds, the bootstrap
      scheme and the normalisation prior. Stamp `protocol: "lrbench-v1"` into every result JSON. Changing any of it
      means `lrbench-v2`, with both reported side by side for one release
- [x] Committed result rows (`results/<set>/<model>.json`, schema `lrbench-row/1`), set cards with data revisions,
      `lrbench results check` / `lrbench validate`, and tables generated by `lrbench results render`. CI validates
      every row and fails on a stale generated table ([docs/results.md](docs/results.md))
- [ ] **next** — `lrbench validate` for a raw result JSON as well as for rows, once the `protocol` stamp exists
- [ ] **next** — Mandatory contamination declaration: a submission states its training data, and the harness
      publishes the held-out-only and unseen-only tables next to the headline. A model trained on these pairs should
      not be able to win quietly
- [ ] **later** — `lrbench verify`: re-run a declared subset of a submission from its recorded command and
      environment, and report whether the numbers reproduce
- [ ] **later** — Environment capture (lockfile, driver, GPU) in the result JSON beyond what is already recorded

## Phase 2 — a leaderboard that does not rot (later)

Every hand-maintained leaderboard dies. This one should be generated from files that CI validates.

- [ ] Submissions by pull request, as `results/<set>/<model>.json` rows. These already carry the command, the
      weights hash, the dataset hashes, the commit, and the licence through the registry. CI already validates the
      schema and the generated tables. What is still to come is the protocol stamp, and third-party models that are
      not in the registry. No number is ever typed by hand
- [ ] Generated leaderboard (GitHub Pages or a HuggingFace Space) with diagnostic columns, not a single score, and
      with two fixed reference rows in every table: the multipole **oracle** (ceiling) and a **no-long-range
      control** (floor). A model's number means little without both
- [ ] "Reported, not verified" badge for submissions whose weights cannot be rerun by a third party
- [ ] Per-submission chemiscope link, so a reviewer can look at the structures where a model fails rather than only
      at its summary row

## Phase 3 — new physics, not just more models (later)

The current sets cover vacuum dimers, fixed-charge wires, trimers, charge parity and electrostatic properties. The
obvious gaps, each of which needs its own oracle and pathology audit before it is worth scoring:

- [ ] Condensed phase: periodic boxes where truncation and Ewald disagree, and the dielectric response that follows.
      The first, reference-free part of it is proposal **D** above: finite-size slope, Stillinger–Lovett, ε
      consistency, and the field across a charged slab
- [ ] Interfaces and slabs; solvated ions beyond the first shell
- [ ] Many-body beyond three fragments; cooperative polarisation. The reference-free dilute ion gas and the
      split/merge checks are proposal **B** above; referenced many-body sets follow them
- [ ] Dynamics-level consequences: NVE energy drift, PMFs for ion pairing from short MD. The thermal mean-force
      task is proposal **C** above. Longer PMFs, solvated ion pairs and drift at scale follow from it
- [ ] Element and charge-state coverage: transition metals, multiply charged species, open shells
- [ ] Response to an external field, beyond the finite-field polarisabilities already in `props`

## Phase 4 — governance (later)

- [ ] A short governance note: who may merge a protocol change, how a metric is deprecated, how a disputed finding is
      resolved
- [ ] `MAINTAINERS.md`, and a quarterly protocol review that either confirms or bumps the version
- [ ] Paper describing the sets and metrics; Zenodo DOI per release

## Where help is most useful

- **Adapters for models we do not have.** One small function, one table row — [docs/adding-a-model.md](docs/adding-a-model.md).
  If your model takes a total charge or carries an explicit Coulomb term, it belongs here.
- **Adversarial review of a metric.** If a diagnostic gives a model a score you can demonstrate is wrong, that is a
  bug in the benchmark. Those issues are the most valuable ones we can receive.
- **A new prior, with the defect that breaks it.** `lrbench priors` takes a check plus a synthetic model that
  violates it; both are small, and the pair is what makes the check trustworthy.
- **Reference data.** A long-range set with a defensible level of theory, a documented protocol and a licence that
  permits redistribution — especially condensed-phase.
- **Independent reruns.** Taking a published submission and reproducing it on your own hardware.
