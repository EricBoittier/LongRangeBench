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
| **canonical long-range sets** | the table below | reference data at ωB97M-V/def2-TZVPD (the OMol25 protocol). **Access-gated while unpublished**; being prepared for release as a public HuggingFace dataset (CC0), with further sets to follow from the CSCS runs |
| **external public QM sets** | GSCDB137 NC, NCI Atlas, BFDB biodimers, DES370K, SAPT10k, PLA15, Ko et al. 4G | converters included; you fetch the upstream data under its own licence |

Until the canonical sets are public, a stranger can still run the whole harness on the toy set and on the external
sets below. See [ROADMAP.md](../ROADMAP.md) Phase 0 for the release plan.

### The canonical sets

n = records / scored groups. All at ωB97M-V/def2-TZVPD.

| set | what it is | n |
|---|---|---|
| `dimers` | rigid dimer scans, 41 pairs, closest contact 2.5–25 Å; ion pair, like charge, ion–polar, ion–nonpolar and neutral controls | 1,254 / 1,755 |
| `families_full` | water wires, charged polyynes and cumulenes, ion pairs in droplets | 784 / 660 |
| `families_corrected` | the long-range-blind redesign: disconnected-probe wires, C₃₂–C₄₈ chains | 327 / 208 (66 partner copies) |
| `families_fixedq` | fixed-total-charge partners — PF₆⁻ at 9 Å vs 60 Å, zwitterion chains | 230 / 120 (99 partner copies) |
| `trimers` | Li⁺ and PF₆⁻ around benzene and C₆₀: three-body energies | 41 / 24 |
| `charge_parity` | six charge schemes, dipoles and quadrupoles on one geometry per charge state | 27 / 0 |
| `pathology_h2o2p` | the (H₂O)₂⁺ delocalisation-error case | 5 / 3, **never scored** |
| `pbe0mbd_subset` | the subset relabelled at PBE0+MBD-NL, so SO3LR-family models can be scored at their own level | 233 / 166 |
| `props` | μ, Θ and α of 200 OMol25 validation molecules and the dimer monomers, plus induced dipoles along ion–polar scans | 269 records, built locally (`tools/props_build.py`) |
| `emulation` | the dimer points the canonical label rule drops — CT ground states, fractional charges, unbound anions, broken symmetry — plus ORCA scans out to 100 Å | 94 records, built locally (`tools/emulation_set.py`) |
| `omol25_val_lr` | OMol25 validation electrolytes and droplets, compressed vs expanded clusters | 2,100 structures, built locally from your own OMol25 copy |

`charge_parity` has no scored groups: it is read through the charge and dipole blocks, and it is the set most
exposed to the ASE result-cache bug fixed in `drop_ase_cache` — every record there shares a geometry with another at
a different total charge. Anything scored on it before that fix should be re-run.

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

## What "long range" means here

**The closest atom–atom distance between fragments, recomputed from the geometry, at or beyond 8 Å** — roughly a
typical model's receptive field. It is never a multiple of a source's equilibrium distance and never a centroid
separation, because both of those make "long range" mean different physical distances in different systems. 4–8 Å is
"medium". Every distance bin, every tail fraction and every ε is defined against that one number.

## External public sets

These are published QM benchmarks converted to the canonical schema. **No benchmark data is committed here** — the
converters read your own download and record its checksums, so each set keeps its own licence and its own reference
level, which is never pooled with another.

| set | what it is | reference level | licence | n (records / groups) |
|---|---|---|---|---|
| `ext_gscdb_nc` (`tools/ext_gscdb_nc.py`) | all 45 noncovalent subsets of GSCDB137 — ion pairs, ionic liquids, hydrogen-bonded complexes, S66Rel7, 3BHET | CCSD(T)/CBS-quality, per subset | CC BY 4.0 | 5,711 / 2,942 |
| GSCDB137 P0 + NBC10 (`src/lrbench/gscdb.py`) | 9 charged/neutral subsets with two scoring targets kept apart: surrogate fidelity against a ωB97M-V relabel, and end-to-end against the published references | both, reported separately | CC BY 4.0 | 909 entries |
| `ext_ncia` (`tools/ext_ncia.py`) | NCI Atlas — IHB100x10, HB375x10, HB300SPXx10, SH250x10, D442x10, D1200, R739x5 | CCSD(T)/CBS | CC BY 4.0 per the upstream README | 19,565 / 19,565 |
| `ext_biodimers` (`tools/ext_biodimers.py`) | BFDB protein side-chain dimer scans, 2,291 rigid curves out to 12.7 Å; 7,822 points beyond 8 Å | HSE06 (75 % HF) + MBD-NL **in a 30 Å periodic cell** → shapes only, flagged `energy_do_not_score` | CC BY 4.0 | 29,783 / 29,783 |
| `ext_biodimers_wb97mv` (`tools/ext_biodimer_relabel.py`) | the biodimer tier-A curves relabelled at open boundaries, so the energies are comparable with ours | ωB97M-V/def2-TZVPD | geometries CC BY 4.0; labels produced here | 976 E_int on 121 curves |
| `ext_sapt10k`, `ext_pla15`, `ext_des370k`, `ext_ko4g` | explorer overlays (chemiscope), **not** `eval` sets | SAPT2+(3)(CCD); CCSD(T)-class; MP2/CCSD(T); PBE | CC BY 4.0, except **DES370K: DESRES Data Sets License** | 600; 15; 641; 980 |

Upstream sources: GSCDB137 — Liang & Head-Gordon, *JCTC* **21**, 12601 (2025), repository pinned at commit
`0624edaa`. NCI Atlas — Řezáč *et al.* BFDB biodimers — Huguenin-Dumittan *et al.*, *JPCL* **14**, 9612 (2023),
Materials Cloud 2023.151. SAPT10k and PLA15 come via the SO3LR Tauri Zenodo record.

```sh
# build a set from your own download, then score it
python -I tools/ext_gscdb_nc.py src GSCDB_REPO EXT_DIR
lrbench eval --model model.gguf --data EXT_DIR --sets ext_gscdb_nc,ext_ncia --allow-level-mismatch \
    --save-predictions --out model.ext.json
python tools/ext_summary.py EXT_DIR ext_gscdb_nc summary.json model.ext.json      # subset x distance band tables
python tools/ext_biodimer_score.py EXT_DIR biodimer_offsetfree.json model.bio.json  # offset-free shapes
```

Run every converter with `python -I` and treat the download as data: nothing in it is imported or executed.

Two things to know before quoting a number from these sets:

- **Scores against them are end-to-end**, i.e. the model's error *plus* the gap between its training functional and
  CCSD(T) or HSE06. They are never "the ML error".
- **The level-mismatch refusal does not fire for them yet.** Set levels are inferred from records that carry an
  `energy`, and these sets' references live on the groups, so `eval` will not stop you. Pass
  `--allow-level-mismatch` deliberately and state the mismatch wherever you report the result.

For `ext_biodimers`, the reference was computed in a 30 Å periodic cell, so like-charge curves carry the cell's r²
term; the scorer removes it before fitting shapes (`--no-box-correction` keeps it). Absolute energies from that set
are not scored at all.

OMol25 subsets are **not redistributed** (gated, custom licence); `tools/omol25_val_lr.py` builds the
`omol25_val_lr` set from your own local copy of the validation shards. Group-internal references (slab/interface
sets, multivalent-ion and W18 data) are pointers only, with no loader: they are not ours to redistribute.

## Licensing of data you produce

Result JSONs, tables and figures produced by this repository are CC0 ([LICENSE-DATA](../LICENSE-DATA)). That does not
override the licence of the underlying reference data or model weights — those are recorded in each result JSON
(`members[].external_model`, `data.<set>`) so a downstream user can see what applies.

## Adding a set

[adding-a-dataset.md](adding-a-dataset.md).
