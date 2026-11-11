# Changelog

Notable changes to LongRangeBench. Newest first.

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

### Known limitations

- The canonical reference sets are not yet public ([docs/datasets.md](docs/datasets.md)); the toy set and the
  external-set converters are the paths that work for everyone today.
- The scoring protocol is **not frozen**: metric definitions may still change before a `protocol: lrbench-v1` stamp
  exists ([ROADMAP.md](ROADMAP.md) Phase 1). Pin a commit if you publish numbers.
- GSCDB137: the 3BHET, IHB100x2 and NBC10 scoring paths are untested, and conformal calibration is not implemented
  for that set.
- `tools/` contains study-specific scripts that reference campaign output not included here (see
  [tools/README.md](tools/README.md)).
