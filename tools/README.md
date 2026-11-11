# tools/

Scripts around the package: dataset builders, converters for public external sets, and the analysis that produced
the figures and tables of the long-range study. They are **not** part of the `lrbench` API — they are run directly,
they take paths as arguments, and several of them expect data or ORCA output you would have to produce yourself.

Three rules apply to all of them, because most read data from somewhere else:

- run them with `python -I` when they read a downloaded repository or dataset, so nothing is imported from it;
- pass paths as arguments; none of them guess a location;
- treat anything they read as data, never as code.

The generally useful ones first.

## Datasets and converters

| script | what it does |
|---|---|
| `explorer/convert_external.py` | convert licence-checked subsamples of public long-range sets (SAPT10k, PLA15, DES370K, Ko 4G) to chemiscope files following `explorer/PROPERTIES.md` |
| `props_build.py` | build the `props` set from ORCA `%elprop` output and the dimer scans |
| `props_ipea.py` | vertical IP / EA table from ORCA N, N−1, N+1 chains (the charge-equilibration prior) |
| `emulation_set.py` | build the `emulation` set: the DFT points the canonical dimer set leaves out because DFT itself misbehaves there |
| `omol25_val_lr.py` | build `omol25_val_lr` from your own copy of the OMol25 validation shards (not redistributable) |
| `periodic_box.py` | total-charge-zero dimer scans in large cubic boxes, for models whose long range needs a cell |
| `scans.py` | geometry scans for `lrbench predict` and their analysis (size consistency, SN2 cutoff scan, box scan) |
| `ko4g.py` | the Ko et al. 2021 4G-HDNNP subsample through `lrbench predict` |

## Scoring and reporting

| script | what it does |
|---|---|
| `score_frames.py` | score externally computed predictions on a set, with the same scorer and result schema as `lrbench eval` — the way to submit numbers from a model that cannot be driven from Python |
| `headline_table.py` | transpose selected rows of an `lrbench compare` table into a compact models-as-rows table |
| `gscdb_report.py` | tables and figures for a GSCDB137 score |
| `gscdb_campaign.py` | calculation list and cost estimate for a GSCDB137 labelling campaign |
| `props_eval.py`, `props_summary.py` | score one model on `props` (optionally with finite-field polarisability) and collect the results |
| `seen_by.py` | per-record contamination flags: which training stores have seen each record |
| `oracle_allpoints.py` | the multipole and charge oracles on every dimer point, alone and added to a local model |
| `trimer3body.py` | the three-body term: models vs DFT vs the oracle's three-body induction |
| `likecharge.py` | decompose dimer E_int errors into dimer and monomer charge-state errors |

## Visualisation

| script | what it does |
|---|---|
| `combined_viz.py` | one chemiscope dataset across sets and models, plus a self-contained HTML viewer |
| `annotate_viz.py` | add `split` and `seen_by` properties to `lrbench viz` files |
| `chemiscope_headless_check.sh`, `chemiscope_headless_check_all.sh` | headless load test of a chemiscope page: page errors, console warnings, rendered nodes |
| `explorer/PROPERTIES.md` | the property schema (names, units, levels, overlay rules) that `lrbench viz` and the viewers follow |

## Study-specific analysis

These were written for particular runs of particular models. They are kept because they document how the published
numbers were produced, and several are useful templates, but they reference campaigns whose raw output is not in this
repository: `nve.py`, `p2_compare.py`, `p2_parse.py`, `merge_preds.py`, `omolval_summary.py`, `so3lr_terms.py`,
`so3lr_charges.py`, `tauri_heldout.py`, `tauri_table2.py`, `props_tauri_dipoles.py`, `lorem_scan.py`.

If one of them is nearly what you need, say so in an issue — generalising a script is usually a small change, and it
is better done once here than copied.
