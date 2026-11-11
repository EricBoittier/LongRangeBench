# The result JSON

`lrbench eval --out results.json` writes one self-describing file. It is the unit of exchange: `lrbench compare`
takes several of them, and a benchmark submission is one of them plus the command that produced it.

## Top level

| key | meaning |
|---|---|
| `lrbench_version` | the version that produced the file |
| `created`, `host`, `argv`, `wall_s` | when, where, the exact command line, and how long it took |
| `code_commit` | the git commit of the lrbench source that ran, with `+dirty` if it had uncommitted changes; `null` from an installed wheel |
| `label` | what to call this model in tables (`--label`, else the spec) |
| `members` | one entry per ensemble member: spec, path, **sha256** of the weights (also for an `ase:` adapter spec whose `path=` / `model=` names a file), backend, whether it could be passed formal charges / charge+spin, and `external_model` (name, version, **licence**, how charge/spin enter, the long-range term) for adapters |
| `level` | `model_level`, `model_level_source` (given or inferred), `label_levels` per set, and `mismatch` |
| `element_offsets` | `mode` (`fit` / file / `none`), the fitted offsets per level in eV, and the residual MAE per atom |
| `normalise` | the normalisation settings in force (floor, temperature, units, gates) |
| `sets` | the scores, one block per set — below |
| `equivariance` | with `--equivariance`: max and mean \|ΔE\| and \|F(Tx) − T·F(x)\| per transformation, the member and sample used, and whether periodic checks ran |
| `data` | per set: the files read, their **sha256**, and the dataset revision |
| `notes` | anything the run wants a reader to know first, e.g. a level mismatch |

Three of these exist specifically so a number can be audited later: the weight `sha256`, the dataset `sha256` and
`argv`. Keep them.

The file itself is not committed, because `host`, `path` and the data paths name the machine it ran on. To commit a
result, `lrbench results add` turns it into a row ([results.md](results.md)). A row keeps the hashes and the
command, and drops every directory.

## A set block (`sets.<set>`)

| key | meaning |
|---|---|
| `n_records`, `n_groups`, `group_kinds` | what was loaded |
| `n_pathology_records`, `n_pathology_groups`, `pathology` | what was excluded as a known-bad reference, **with the model's values for it** |
| `levels`, `n_groups_excluded_level`, `groups_excluded_level` | the levels present and what was dropped for mixing them |
| `source` | the files and hashes this block was computed from |
| `energy_force` | `n_energy`, `n_forces`, `by_class_distance`, `all` — absolute energy/force errors (same-level models only) |
| `E_int` | `n`, `n_failed`, `bins`, `by_distance`, `tail`, `offset_free_8A`, `abs_ge12`, `inside_median` |
| `ddE` / `dE` | `slope`, `mae`, `zero_model_mae`, `n_pairs`, `n_composition_changing`, `dft_median_abs`, `by_family`, `rows` |
| `E3` | absolute and geometry-relative three-body errors |
| `zwitterion` | the ΔE_z(m) series and local-zero score |
| `charges` | fragment-charge parity, per-scheme MAE, dipoles — or `{"skipped": ...}` with the reason |
| `normalised` | `dimer_tail`, `ddE`, `E3`, `coulomb_prior` — see [metrics.md](metrics.md#normalised-errors) |
| `calibration` | with an ensemble or `--conformal`: split-conformal and CQR coverage, widths and worst bin |
| `prediction_errors`, `n_prediction_errors` | terms the backend failed on, with the reason (up to 50 listed) |

Every estimate that has an interval carries it inline, with `n` and what `n` counts (points, scans, pairs, ddE
quartets) — never a bare number whose sample size you have to guess.

Blocks that could not be computed say **why** rather than being absent: `{"skipped": "model returns no charges"}`.
A metric that silently disappears is a metric that gets misread as "fine".

## Stability

The schema is additive in 0.x: keys get added, and a consumer should ignore unknown ones. It is not yet frozen —
[ROADMAP.md](../ROADMAP.md) Phase 1 adds a `protocol` stamp, a schema version and `lrbench validate`, after which
changing a definition bumps the version instead of editing it in place.

## Reading one without this document

```sh
python -c "import json,sys; J=json.load(open(sys.argv[1])); \
  print(J['label'], J['level']['model_level']); \
  print(json.dumps(J['sets']['dimers']['E_int']['tail'], indent=1))" results.json
```

`lrbench compare results.json` prints the same content as a table, which is usually what you want.
