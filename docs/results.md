# Committed results

A number in a table here is never typed. It comes from an `lrbench eval` result JSON, through `lrbench results add`,
into one **row file**, which a validator checks and from which every table is regenerated.

```
results/
  sets.json                 set cards: reference level, licence, public or not, every data revision scored
  <set>/<model>.json        one row: one model on one set, with its provenance        (committed)
  SUMMARY.md                generated tables, one section per set                     (committed, never edited)
  private/                  same layout, for unreleased models and internal-only data (git-ignored)
```

`docs/models.md` has two generated blocks: the named-model table and the list of committed rows. Both are generated
by the same command. CI fails if any generated file is out of date, or if any row or card does not validate.

## Adding a row

```sh
lrbench eval --model mace-polar --sets dimers --data ./lr-sets --normalise --out runs/mace-polar.json
lrbench results add runs/mace-polar.json            # -> results/dimers/mace-polar.json, validated before writing
lrbench results render                              # -> results/SUMMARY.md, docs/models.md blocks
lrbench results check                               # what CI runs (also: lrbench validate)
```

Commit the row and the regenerated files together. `--set` limits `add` to some of the result's sets, and `--model`
names the row when the eval label is not the registry name.

A result produced before lrbench recorded its own commit needs `--commit <sha>`: the commit of the code that
*produced* the numbers, not the one you are adding them from.

## What a row holds (`lrbench-row/1`)

| key | meaning |
|---|---|
| `set`, `model`, `label` | the set, the registry name (also the file name), the display label |
| `members` | per seed: the spec (directories stripped), the weights `sha256`, backend, package + version, checkpoint, charge/spin handling. `sha256` is null for checkpoints a package downloads itself (MACE, orb, SO3LR, ...), which are identified by package version and checkpoint name instead |
| `model_level`, `reference_level`, `level_mismatch` | the level the model was trained at, the set's label level(s), and whether they differ |
| `seeds`, `evidence` | number of members; `claim` at 4 or more seeds, else `screen` |
| `n_failed`, `n_scored` | the largest failure count the set block reports (backend errors, groups with no finite value, a backend's own `failures`), and E_int groups scored |
| `date`, `code_commit`, `lrbench_version` | when the eval ran and from which clean commit |
| `data` | jsonl and groups `sha256` (and HF revision) of the data scored |
| `argv` | the eval command, with directories stripped |
| `metrics` | the headline rows of `lrbench compare` for this set: metric, value [95 % CI], n, note |
| `notes` | the result's notes, with directories stripped, e.g. a level mismatch or partial failure |

## What the validator refuses

- **Unknown or missing keys**, including `host`, and any absolute or home-relative path anywhere in a row or card.
- **A row with no metrics.**
- **A row whose set has no card**, or whose data hashes match no carded revision. The data scored must be data
  someone recorded.
- **Inconsistent caveats.** The validator refuses:
  - a model level that differs from the reference level unless `level_mismatch` is true;
  - `n_failed > 0` without a note saying what failed;
  - an `evidence` value that does not follow from `seeds`.
- **A dirty or missing `code_commit`.** Numbers from uncommitted code cannot be reproduced.
- **In the public tree only:** these rows belong in `private/` instead:
  - a model that is not in the named-model registry (`lrbench models`), or whose members were not run with that
    entry's spec, so a row cannot relabel another model's result;
  - a member with neither a weights sha256 nor a package version, which is what `--no-hash` produces;
  - a set whose card is not public.

Each table flags these conditions in the row itself: **LEVEL MISMATCH**, **n failed**, **screen (k seeds)**, and
**STALE** for a row scored on an earlier revision of its set.

## Adding a set

A set is first made as [adding-a-dataset.md](adding-a-dataset.md) describes. Before its first row is added, it gets
a card:

```sh
lrbench results add-set myset --data ./lr-sets --licence CC0-1.0 --public --description "what the curves are"
```

The card records the reference level read from the records. Pass `--level` for a reference-only set. The card also
records the licence and the first revision: the jsonl and groups sha256 and the record and group counts. Use
`--public` only if the licence permits publishing derived numbers.

Data that may not be redistributed takes `--private`. Its card and every row on it go under `results/private/`, so
no number derived from it can be committed by accident. Sets that must never go in the public tree include:
- group-internal references;
- sets under a no-redistribution agreement.

A set is public or private, never both.

## Adding curves to a set

New scans, relabelled points or a corrected flag change the set's hashes. Record that as a new revision:

```sh
lrbench results add-set myset --data ./lr-sets --licence CC0-1.0 --public --note "+12 ion-pair scans (Cl-, Br-)"
```

Rows scored on the old revision stay valid, because the old revision is still on the card. The tables mark them
**STALE** until they are re-scored and re-added. A row scored on data that was never carded is refused.

Changing a set's reference level makes it a different set, not a new revision. Give it a new name.

## Unreleased models and internal data

Rows for unreleased models go under `results/private/`, which git ignores; use `lrbench results add --private`. The
same applies to rows on internal-only sets. `lrbench results check --private` validates them with the same rules,
except the public-only ones. `render` never reads them. Weights are never committed anywhere in this repository.
