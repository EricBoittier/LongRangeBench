# Changelog

Notable changes to LongRangeBench. Newest first.

## Unreleased

### Added

- `results/`: committed result rows (`results/<set>/<model>.json`, schema `lrbench-row/1`). Each row carries:
  - the model's registry name;
  - the weights sha256, or the package version;
  - the model and reference levels, plus a mismatch flag;
  - seeds, giving screen or claim;
  - `n_failed`;
  - the date, the code commit and the data sha256;
  - the headline metrics.

  Each row has its directories stripped. `results/sets.json` holds the set cards, with a licence, a public or private
  flag and every scored data revision. `results/private/` is git-ignored and holds rows for unreleased models and
  internal-only data. See [docs/results.md](docs/results.md).
- `lrbench results add | add-set | check | render`, plus `lrbench validate` as an alias of `check`.
  - `render` regenerates `results/SUMMARY.md` and the generated blocks of `docs/models.md`, including the
    named-model table, which had fallen behind the registry.
  - A test fails if any of them is stale.
- `lrbench eval` records `code_commit`, and hashes the checkpoint an `ase:` adapter spec names (`path=` / `model=`).

## 0.1.0 — 2026-11-11

First public release of the harness. The metrics, adapters and CLI were developed and used privately before this
point; this is the extraction, not the beginning of the work.

### Added

- `lrbench eval` — scoring on canonical long-range sets: E_int by class × distance, **tail fraction**, offset-free
  error at 8 Å, |E_int| beyond 12 Å, fixed-charge ΔΔE slope and MAE, ΔE series, three-body E3 and ΔE3, forces,
  fragment-charge parity and dipoles.
- Physics-normalised errors (`--normalise`): signed ε against a multipole + induction prior, k_BT errors, normalised
  ΔΔE and ΔE3, Coulomb-prior check for gated-Coulomb models.
- `--conformal` / ensembles: Mondrian split-conformal and CQR calibration, grouped by scan.
- `--equivariance`: rotation, improper rotation, translation, permutation, wrap and periodic-image probes.
- `oracle:multipole` — the achievable-tail ceiling from the fragments' own DFT moments, truncated by rank.
- Backends: mlip.cpp GGUF (`mlipcpp-eval`, C API, Python bindings), metatomic / TorchScript, any ASE calculator by
  import path, and seed ensembles.
- Adapters and a named-model registry for released long-range models: SO3LR, SO3LR-Tauri S/M/L, MACE-POLAR-1 M/L,
  MACE-MH-1, MACE-OMOL-0, MACELES-OFF, CACE-LES, each with its training level, licence and charge/spin handling.
- `lrbench suite` — one eval per model, each in its own Python environment, with a combined table; local or Slurm.
- `lrbench perf` — force-call and MD-step timing, µs/atom/step, ns/day, peak GPU memory, generated water boxes.
- `lrbench viz` — chemiscope export with model errors as properties, following `tools/explorer/PROPERTIES.md`.
- `lrbench predict`, `lrbench emulation`, `lrbench models`, `lrbench sets`.
- `lrbench toy` and `lrbench.toy` — a self-contained Lennard-Jones set, so install → first number needs no download
  and no GPU, and the test suite can assert tail fraction 1.000 for an exact model and 0.000 for a truncated one.
- GSCDB137 support (`lrbench.gscdb`): manifest, export, ORCA point lists and scoring against a pinned upstream
  commit, with surrogate fidelity and end-to-end targets kept separate.
- `props`: dipoles, quadrupoles and polarisabilities against CP-SCF references; vertical IP / EA tables.
- Level-of-theory enforcement (never pooled, mismatches refused unless forced), per-element offset fitting, DFT
  pathology exclusion with the model's values still reported, contamination flags and held-out-only rescoring.
- Documentation: metrics, datasets, models, output schema, timing, properties, and guides for adding a model or a
  dataset.

### Added (folded in from the development workspace, 2026-11-11)

- **`lrbench priors`** (`lrbench.priors`): twenty pass/fail checks of physical priors and of model and data
  assumptions — far-field exponent, ε, seams, size consistency, work-energy consistency, bare-ion NVE, invariance,
  IP/EA convexity, separated-charge offset by Q, fragment charges, periodic↔open consistency, locality, charge
  conditioning and caching, determinism, precision, contamination, level of theory, CT/unbound labelling,
  offset-free degeneracy and distribution shift. `eval` output is unchanged. `--list` prints the checks and their
  rules. Validated by thirteen synthetic planted-defect models, one check each, in `tests/test_priors.py`.
- **External public QM set converters** (`tools/ext_*.py`): GSCDB137 noncovalent (45 subsets, 5,711 records), NCI
  Atlas (19,565), the BFDB biodimers (29,783, with the 30 Å reference cell's r² term handled and like-charge curves
  corrected), and the tier-A open-boundary relabel `ext_biodimers_wb97mv` (976 E_int on 121 curves). No benchmark
  data is redistributed: the converters read your own download and record its checksums.
- Probe builders for the priors suite (`tools/priors_ipea.py`, `tools/priors_sepframes.py`) and a check × model
  table (`tools/priors_table.py`); `ext_summary.py` and `ext_biodimer_score.py` for the external sets.
- `MlipcppEval` can return the projected physical charges alongside the latent ones (`charges_phys`).

### Known limitations

- The canonical reference sets are not yet public ([docs/datasets.md](docs/datasets.md)); the toy set and the
  external-set converters are the paths that work for everyone today.
- The scoring protocol is **not frozen**: metric definitions may still change before a `protocol: lrbench-v1` stamp
  exists ([ROADMAP.md](ROADMAP.md) Phase 1). Pin a commit if you publish numbers.
- GSCDB137: the 3BHET, IHB100x2 and NBC10 scoring paths are untested, and conformal calibration is not implemented
  for that set.
- The `ext_*` sets do not trigger the level-of-theory refusal, because set levels are inferred from records that
  carry an `energy` and theirs live on the groups. Pass `--allow-level-mismatch` deliberately and state the
  mismatch. The fix is to count records referenced by groups.
- The `ext_*` converters are smoke-run only: no unit tests and no `--help`.
- Conformal calibration needs cells of roughly 19 points or more; some GSCDB ion-pair cells (n = 16) are too small.
- `seen_by` contamination flags are not yet written into the `ext_*` records, though the overlap tables exist.
- Third-party slab/interface and multivalent-ion references are pointers only, with no loader: they are not ours to
  redistribute.
- `tools/` contains study-specific scripts that reference campaign output not included here (see
  [tools/README.md](tools/README.md)).
