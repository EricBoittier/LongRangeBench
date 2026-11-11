# Datasets

LongRangeBench is a harness: it scores a model against **records** (structures with reference values) and **groups**
(the differences that are actually scored). Any data in that schema works, wherever it comes from.

## Where data comes from

| `--data` value | meaning |
|---|---|
| a directory | `<dir>/canonical/<set>.jsonl` and `<dir>/canonical/<set>.groups.json` (also found directly in `<dir>`) |
| `hf` | the canonical HuggingFace dataset; set `$LRBENCH_HF_REPO` to use a different repo id, `--revision` to pin |

`lrbench sets --data <where>` lists what is available. Every result JSON records the file paths, their **sha256** and
the revision, so a table can always be traced back to the exact bytes it was computed from.

## Status of the reference data

| set family | contents | status |
|---|---|---|
| **toy** | 2 dimer scans, 3 ΔΔE quartets, 1 flagged pathology; references from a Lennard-Jones calculator | **ships with the repo** (`lrbench toy`), CC0. Used by the tests and CI |
| **canonical long-range sets** | `dimers`, `families_full`, `families_corrected`, `families_fixedq`, `trimers`, `charge_parity`, `pathology_h2o2p`, `pulled_apart/*`, `pbe0mbd_subset`, `props` | reference data at ωB97M-V/def2-TZVPD (OMol25 protocol). **Access-gated while unpublished**; being prepared for release as a public HuggingFace dataset (CC0), with further sets to follow from the CSCS runs |
| **external public QM sets** | GSCDB137, DES370K, SAPT10k, PLA15, Ko et al. 4G-HDNNP | converters included; you fetch the upstream data under its own licence |

Until the canonical sets are public, a stranger can still run the whole harness on the toy set and on the external
sets below. See [ROADMAP.md](../ROADMAP.md) Phase 0 for the release plan.

## Record schema

One JSON object per line in `canonical/<set>.jsonl`:

| field | type | meaning |
|---|---|---|
| `id` | str | unique within the set; the key groups refer to |
| `species` | list[str] | element symbols |
| `positions` | list[[x,y,z]] | Å |
| `cell`, `pbc` | 3×3 or null, list[bool] | periodic cell when there is one |
| `charge`, `spin` | int, int | total charge; spin multiplicity (2S+1) |
| `energy` | float | reference total energy, eV |
| `forces` | list[[fx,fy,fz]] | eV/Å, optional |
| `level` | str | **level of theory of the reference values** — required; groups never mix levels |
| `formal_charges` | list[int] | per-atom formal charges, passed to backends that accept them |
| `fragment_id` | list[int] | which fragment each atom belongs to; enables fragment slicing (`<id>#frag=k`) |
| `fragment_charges`, `fragment_spins` | list[int] | per fragment, for the sliced subsystems |
| `cls` | str | class, e.g. `ion_pair`, `same_sign`, `ion+polar`, `wire` — metrics are reported per class |
| `group`, `role` | str | the scan or family this record belongs to, and its role in it (`AB`, `s0`, …) |
| `distance` | float or null | the canonical distance of the point, Å (`d_min`) |
| `flags` | list[str] | `dft_pathology_do_not_score`, `energy_do_not_score`, `partner_copy`, `seen_by:*` |
| `refs` | dict | physical priors on the same geometry: `E_mp`, `E_mp_ind`, `E_es_exact`, … (meV) |
| `extra` | dict | anything else; `extra.strata` is used for stratified reporting |

## Group schema

A list of objects in `canonical/<set>.groups.json`. A group is a linear combination of records (or fragment slices of
records) with a reference value, which is what makes the metrics offset-free:

```json
{"name": "E_int:AB:o0:d8.0", "kind": "E_int", "ref": -0.0123,
 "terms": [["AB:o0:d8.0", 1.0], ["AB:o0:d8.0#frag=0", -1.0], ["AB:o0:d8.0#frag=1", -1.0]],
 "meta": {"cls": "ion_pair", "dmin": 8.0, "pair": "AB"}}
```

- `kind` — `E_int` (interaction energy), `dE`, `ddE` (a difference of differences), `zwitterion_dE`, `E3`
- `terms` — `[term, coefficient]`; a term is a record id, optionally with a fragment selector `#frag=k` or
  `#frags=k1,k2`. Each distinct term is evaluated **once** per model
- `ref` — the reference value of the combination, eV
- `meta` — `cls`, `dmin`, `series`, `family`, `flags`, … used for per-class and per-series reporting

Why groups rather than raw energies: in a `ddE` quartet the composition and the total charge are identical in all
four terms, so per-element offsets, charge-state offsets and any constant shift cancel exactly. What is left is the
long-range response, which is the thing being measured.

## External public sets

Each converter writes the canonical schema, carries the set's **own reference level** (never pooled with another) and
records its licence in the output.

| set | what it is | reference level | licence |
|---|---|---|---|
| **GSCDB137** | 137 reaction-energy subsets; `src/lrbench/gscdb.py` builds a manifest, exports extxyz, scores two targets separately (surrogate fidelity vs end-to-end) against a pinned upstream commit | per subset, as published | upstream repository's terms |
| **DES370K** | dimer interaction energies, 641 points / 30 scans in the converted example | CCSD(T)-class | DESRES Data Sets License |
| **SAPT10k** | 600 dimers with SAPT decompositions | SAPT2+(3)(CCD) | CC BY 4.0 (via the SO3LR Tauri Zenodo record) |
| **PLA15** | 15 protein–ligand dimers | CCSD(T)-class | CC BY 4.0 (same record) |
| **Ko et al. 4G-HDNNP** | 980 structures of the 4G-HDNNP charge-transfer systems | PBE | CC BY 4.0 |

Conversion entry points: `python -I -m lrbench.gscdb manifest|export|points|score` for GSCDB137, and
`tools/explorer/convert_external.py` for the dimer sets. Run them with `python -I` and treat a downloaded repository
as data: nothing in it is imported or executed.

OMol25 subsets are **not redistributed** (gated, custom licence); `tools/omol25_val_lr.py` builds the
`omol25_val_lr` set from your own local copy of the validation shards.

## Licensing of data you produce

Result JSONs, tables and figures produced by this repository are CC0 ([LICENSE-DATA](../LICENSE-DATA)). That does not
override the licence of the underlying reference data or model weights — those are recorded in each result JSON
(`members[].external_model`, `data.<set>`) so a downstream user can see what applies.

## Adding a set

[adding-a-dataset.md](adding-a-dataset.md).
