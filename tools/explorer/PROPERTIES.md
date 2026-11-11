# Chemiscope property naming spec (omol-longrange-explorer / `lrbench viz`)

Version 2, 2026-11-06 (v2: reference columns renamed `/DFT` -> `/REF`; `level` added; overlay rules). Followed by `lrbench viz` and by the group-entry explorer build, so
files from either tool can be loaded, overlaid and merged in the Space viewer.

## File
- One chemiscope JSON per dataset view, gzip-compressed (`<name>.json.gz`), written with `chemiscope.write_input`
  (chemiscope 1.1.0 schema). `meta.name` = human title, `meta.description` = what one entry is + units + data source.
- `meta.references` holds the dataset / model repo links.
- Entries are **structures** (one per record) except the *families* views, where an entry is one **group**
  (a dE / ddE / zwitterion difference from `<config>.groups.json`); the structure shown is the group's anchor record
  (see `anchor_id`). This is stated in `meta.description` and in the `entry` property.

## Conventions
- **Units in names are fixed:** energies in **meV** (converted from the canonical eV), distances in **Å**, charges in
  **e**, forces in **eV/Å**. Every property's `description` repeats the unit.
- **Missing values:** numbers -> `NaN` (chemiscope draws NaN points greyed / skips them), strings -> `"none"`.
  Booleans are written as strings `"true"` / `"false"` / `"none"` (categorical in chemiscope).
- **Reference columns:** `<quantity>/REF` = the reference value at the record's `level` (our sets: wB97M-V/def2-TZVPD
  DFT; external sets: whatever the source used, e.g. SAPT2+(3), CCSD(T)/CBS, PBE).
- **Model columns:** `<quantity>/<model>`. Signed errors: `<quantity>_err/<model>` = model - REF (meV). Absolute values are left to the user (chemiscope can't take abs, so `|err|` is also given as
  `<quantity>_abserr/<model>` for colour scales).
- **Seeds:** a model trained with several seeds (Ectl, Fa, FQ: s13, s21, s42, s63) gets the **seed mean** in
  `<q>/<model>` and the seed standard deviation (ddof=1) in `<q>_sd/<model>`. Single-checkpoint models (Z, FB) have no
  `_sd` column. Per-seed values are not written (they are in the source json files listed in the Space README).
- **Model tags** (also the `--model` labels in `lrbench viz`):
  | tag | checkpoint |
  |---|---|
  | `Z` | PET-OMol-S zero-shot (`zero-shot/model.gguf`) |
  | `Ectl` | short-range fine-tune control, no LR term (4 seeds) |
  | `Fa` | LES fine-tune, arm a (4 seeds) |
  | `FQ` | LES fine-tune with charge supervision (4 seeds) |
  | `FB` | fragment-charge model sig2 FB (formal unit charges as input; 1 checkpoint) |

## Structure properties (target `structure`)
| name | type | unit | meaning |
|---|---|---|---|
| `entry` | str | | `structure` or `group:<kind>` |
| `id` | str | | canonical record id (families: anchor record id) |
| `set` | str | | canonical config (external: dataset name) |
| `level` | str | | **required.** Level of theory of every `/REF` value and reference force on this entry: ours `wB97M-V/def2-TZVPD` (OMol25 level); QCell / SO3LR level-0 sets `PBE0+MBD-NL`; SO3LR level-1 sets `PBE0+MBD`; SO3LR OMol25/OPoly26 subsets `wB97M-V/def2-TZVPD (SO3LR OMol25 subset; mixed stored conventions, never trained on; energies not comparable)`; external benchmarks their own reference (e.g. `SAPT2+(3)(CCD)/aug-cc-pVTZ`, `CCSD(T)/CBS (DES370K)`, `PBE (FHI-aims, light)`). `unknown` if not known. |
| `class` | str | | canonical `cls` (dimer class / family name) |
| `group` | str | | canonical `group` (scan / pair / system key) |
| `role` | str | | canonical `role` |
| `charge` | int | e | total charge |
| `spin` | int | | multiplicity 2S+1 |
| `n_atoms` | int | | |
| `locality` | str | | `within_rf` / `lr_blind` / `none` (PET-OMol-S receptive field 8 Å × 3 layers) |
| `q_fixed_in_pair` | str | | `true` / `false` / `none` (every difference this record enters keeps Q fixed) |
| `flags` | str | | `;`-joined canonical flags, `none` if empty |
| `partner_copy` | str | | `true` if the record is a copy from another config |
| `distance` | float | Å | canonical `distance` (dimers: d_min; trimers: min inter-fragment distance) |
| `E_int/REF` | float | meV | interaction energy E_AB - E_A - E_B (dimers) / E_ABC - ΣE_mono (trimers) |
| `E_mp` | float | meV | multipole-expansion E_int (dimers, d ≥ 8 Å) |
| `E_int/<m>`, `E_int_err/<m>`, `E_int_abserr/<m>`, `E_int_sd/<m>` | float | meV | model E_int, error, abs error, seed sd |
| `tail/REF` | float | meV | offset-free tail ΔE = E_AB(d) - E_AB(d_ref), d ≥ 8 Å, d_ref = scan's longest valid point (same Q) |
| `tail/<m>`, `tail_err/<m>`, `tail_abserr/<m>`, `tail_sd/<m>` | float | meV | the same for the model ("offset-free error") |
| `E3/REF`, `E3/<m>`, `E3_err/<m>`, ... | float | meV | trimers: three-body energy (on ABC records) |
| `dE3_vs_far/REF`, `.../<m>` | float | meV | trimers: E3(geometry) - E3(far reference), fixed-Q brackets |
| `gap` | float | eV | HOMO-LUMO gap (families) |
| `dipole_norm` | float | e·Å | |μ| about the COM (suite-v2 records) |
| `kind` | str | | families: group kind (`dE`, `ddE`, `zwitterion_dE`) |
| `series` | str | | families: group `meta.series` |
| `group_q_fixed` | str | | families: group `meta.q_fixed` |
| `group_locality` | str | | families: group `meta.locality` |
| `value/REF` | float | meV | families: group value Σ c_i E_i (reference) |
| `value/<m>`, `value_err/<m>`, `value_abserr/<m>`, `value_sd/<m>` | float | meV | families: model group value, error, ... |
| `terms` | str | | families: `+1 id; -1 id; ...` |
| `anchor_id` | str | | families: record whose geometry is shown |
| `E_rel/REF` | float | meV | total energy relative to a stated per-group minimum (external sets without interaction energies) |
| `F_rms/REF` | float | eV/Å | RMS reference force |
| `charge_A`, `charge_B` | float | e | monomer charges (external dimer sets) |
| `source` | str | | added by the viewer when two files are merged |

## Atom properties (target `atom`)
| name | type | unit | meaning |
|---|---|---|---|
| `q_lowdin`, `q_iao`, `q_mulliken`, `q_esp`, `q_hirshfeld`, `q_mbis` | float | e | DFT partial charges (whichever schemes the record has; others NaN) |
| `q_formal` | float | e | formal unit charge on the unit's first atom (mlip.cpp `formal_charges` convention) |
| `fragment_id` | int | | interaction-energy subsystem label (NaN if none) |
| `unit_id` | int | | bonded unit |
| `q_latent/<m>` | float | e | model latent (LES) charges, seed mean; sum = total charge (Z has none) |
| `F_err/<m>` | float | eV/Å | |F_model - F_REF| per atom, seed mean of the model force (where reference forces exist) |
| `F_norm/REF` | float | eV/Å | |F_REF| per atom |

When an atom property exists, `environments` covers every atom (cutoff 3.5 Å) so per-atom colouring works.

## Default settings
- `settings.map.{x,y,color}.property` set per view (see the Space README); `color.palette` = `brg` for categorical
  props is avoided; numeric colour uses `viridis` or `seismic` for signed errors with symmetric min/max.
- `settings.structure[0]`: `atomLabels` off, `unitCell` off, environments off.

## Energy units recognised as level-dependent
A property is treated as an energy (never pooled across levels) when its `units` is one of
`meV`, `eV`, `kcal/mol`, `kJ/mol`, `Ha`, `Eh`, `hartree`, `meV/atom`, `eV/atom`. Use these exact unit strings.

## Overlay / merge rules (viewer, and `lrbench viz --compare`)
1. Structures and properties are concatenated (A then B); a structure property `source` (file name) is added, and
   `source_atom` / `level_atom` for atom targets.
2. Missing properties are filled with `NaN` (numbers) or `"none"` (strings); a property that is a string in either file
   becomes a string.
3. Atom properties are indexed by environments; a file without environments gets one environment per atom (3.5 Å).
4. **Levels.** If the merged entries carry more than one `level`, every energy property is split into
   `<name>@<level>` columns (values of other levels are NaN), so energies at different levels never share a column.
   The default map then uses level-independent quantities (`distance`, `n_atoms`, `charge`, ...), colour = `level`,
   symbol = `source`, and a visible note says energies are not pooled. If all entries share one level, energies keep
   their names and colour = `source`.
5. Structure colouring by property is dropped in overlays (element colours), and `pinned` is reset.
