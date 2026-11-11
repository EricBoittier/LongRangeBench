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
- [ ] **next** — make the level-of-theory refusal fire for reference-only sets (count records referenced by groups,
      not only records carrying an `energy`), so `ext_*` cannot be scored cross-level without the explicit flag
- [ ] **next** — PyPI release (`pip install longrangebench`) and tagged versions
- [ ] **in progress** — CI on 3.11–3.13 plus a weekly job that checks every adapter still imports against current
      upstream releases (adapters rot silently otherwise)
- [ ] **later** — A docs site built from `docs/` rather than a folder of Markdown

## Phase 1 — results are comparable (next)

A benchmark whose metric definitions drift is a benchmark whose numbers cannot be compared across papers. The fix is
a protocol that is explicitly frozen and explicitly versioned.

- [ ] **next** — **Protocol v1**: pin the set revisions, the bin edges, the 8 Å / 12 Å thresholds, the bootstrap
      scheme and the normalisation prior. Stamp `protocol: "lrbench-v1"` into every result JSON. Changing any of it
      means `lrbench-v2`, with both reported side by side for one release
- [ ] **next** — Result-schema version + `lrbench validate result.json`, so a submission can be checked mechanically
- [ ] **next** — Mandatory contamination declaration: a submission states its training data, and the harness
      publishes the held-out-only and unseen-only tables next to the headline. A model trained on these pairs should
      not be able to win quietly
- [ ] **later** — `lrbench verify`: re-run a declared subset of a submission from its recorded command and
      environment, and report whether the numbers reproduce
- [ ] **later** — Environment capture (lockfile, driver, GPU) in the result JSON beyond what is already recorded

## Phase 2 — a leaderboard that does not rot (later)

Every hand-maintained leaderboard dies. This one should be generated from files that CI validates.

- [ ] Submissions by pull request: `submissions/<model>/<date>/result.json` plus the exact command, the environment
      and the licence of the weights. CI validates the schema, the protocol stamp and the dataset hashes, then
      regenerates the tables. No number is ever typed by hand
- [ ] Generated leaderboard (GitHub Pages or a HuggingFace Space) with diagnostic columns, not a single score, and
      with two fixed reference rows in every table: the multipole **oracle** (ceiling) and a **no-long-range
      control** (floor). A model's number means little without both
- [ ] "Reported, not verified" badge for submissions whose weights cannot be rerun by a third party
- [ ] Per-submission chemiscope link, so a reviewer can look at the structures where a model fails rather than only
      at its summary row

## Phase 3 — new physics, not just more models (later)

The current sets cover vacuum dimers, fixed-charge wires, trimers, charge parity and electrostatic properties. The
obvious gaps, each of which needs its own oracle and pathology audit before it is worth scoring:

- [ ] Condensed phase: periodic boxes where truncation and Ewald disagree, and the dielectric response that follows
- [ ] Interfaces and slabs; solvated ions beyond the first shell
- [ ] Many-body beyond three fragments; cooperative polarisation
- [ ] Dynamics-level consequences: NVE energy drift, PMFs for ion pairing from short MD
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
