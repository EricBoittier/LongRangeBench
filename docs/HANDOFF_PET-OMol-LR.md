# Handoff from the PET-OMol-LR workspace (2026-11-11)

This note is for whoever maintains LongRangeBench. It lists what the development workspace has that this repository
still lacks. That workspace is the `lrbench/` tree of a private fork, where the code lived before the fold-in
(1495184) and PR #1. The note does not port anything itself.

The two trees were compared file by file, mapping `lrbench/{src,tests,tools,configs}` → `{src,tests,tools,configs}`,
at workspace head da09f8c against this repository at 3405e7a.

## (a) What to port

### Already here: nothing to do

The fold-in was taken after the last workspace code commit, so every biodimer fix is already present. Only private
paths and wording differ.

| workspace change | file here | status |
|---|---|---|
| tier-A ORCA parse + level-matched set builder | `tools/ext_biodimer_relabel.py` | present (docstring wording only) |
| reassign swapped source charges by electron parity, flag `source_charges_swapped` | `tools/ext_biodimers.py` | present (cost-fit path scrubbed to `$LRBENCH_COST_FIT`) |
| use the charge-corrected monomer calcs when present (trailing `q` takes precedence) | `tools/ext_biodimer_relabel.py` | present |
| `ext_summary.py` `level_note` (ωB97M-V relabel vs end-to-end) | `tools/ext_summary.py` | byte-identical |
| periodic-cell analysis, like-charge r² correction, provisional `ct_candidate` / `unbound_anion_candidate` | `tools/ext_biodimer_{pbc_check,score}.py`, `tools/ext_biodimers.py` | present |
| priors suite, probe builders, `priors_table.py` | `src/lrbench/priors/`, `tools/priors_*.py` | present |

**Do not back-port `src/` or `tests/` from the workspace.** This repository is ahead there: it has
`drop_ase_cache`, `oracle:refs`, ten more adapters, `toy.py`, `diagnostics.py` and the scrubbed docstrings. Every
`src/` difference is either one of those or a private path.

### Missing: worth porting

| workspace file | target here | what it adds | notes |
|---|---|---|---|
| `runs/trained-models/run_frame_ipea.sh` | `tools/ipea_relabel/run_frame.sh` (suggested) | the ORCA ΔSCF protocol behind the `charge_state` IP/EA probes. It reproduces the stored label (r0), then runs UKS stability rounds until a round is stable. Its single retry uses SlowConv KDIIS SOSCF, because ORCA refuses TRAH with self-consistent VV10 (EnGrad). | Keep `ORCA_BIN` and the positional args. It is the documented way to make the probes `tools/priors_ipea.py` and `tools/props_ipea.py` consume. |
| `runs/trained-models/run_chunk_ipea.sbatch` | `tools/ipea_relabel/run_chunk.sbatch` (suggested) | Slurm array driver for the above | It has site-specific modules and partition names. Port it as a template next to `configs/slurm-gpu.sbatch-header`, or skip it. |
| `lrbench/README.md` § Conventions (*CT-ambiguous and unbound points*) | `docs/metrics.md` | `scf_wrong_state` points (`emulation`): the SCF landed above DFT's lowest state, so these are bad labels and are excluded from both emulation scores, while CT/unbound points are kept | not in `docs/` yet |
| `lrbench/README.md` § Conventions (*Uncertainty*) | `docs/metrics.md` | CQR intervals are asymmetric and are reported as such, next to the Mondrian split-conformal ones | one sentence |
| `lrbench/README.md` § External QM benchmark sets | `docs/datasets.md` § External public sets | the data facts in (b) below, and the biodimer converter command lines (`ext_biodimers.py LRBENCH_SRC bio_dimers.xyz BAS EXT_DIR`, `ext_biodimer_relabel.py build PARSED POINTS BIODIMERS EXT_DIR`) | `docs/datasets.md` already has 976 / 121 and the cell r² handling |
| `lrbench/CHANGELOG.md` § Open | `CHANGELOG.md` Known limitations / `ROADMAP.md` | the open items listed in (e) that are not yet here | |

### Missing: do not port (study-specific)

These depend on private data locations or one-off studies, so they do not belong in a public harness. Each could be
replaced by a sentence in `tools/README.md`.

- `tools/upet_dication_check.py`: a one-off check of upet on Ca²⁺/Mg²⁺–H₂O, with hard-coded paths. Its finding is
  already covered by `charge_conditioning` and the `drop_ase_cache` fix.
- `tools/explorer/build_explorer.py` and `tools/explorer/build_space.py`: build the private explorer Space from
  workspace result trees.
- `configs/clariden.sbatch-header` and `configs/lyra.sbatch-header`: site headers. `configs/slurm-gpu.sbatch-header`
  already serves as the template.

## (b) Data facts the benchmark must encode

All of these concern the BFDB biodimers (Huguenin-Dumittan *et al.*, JPCL 14, 9612 (2023)) and the tier-A
open-boundary relabel. Where a fact is a converter behaviour, the converter already does it. The point here is that
the **docs and flags** should state it.

1. **Swapped source charges.** In the published set, 248 of the 477 ion–neutral curves put chargeA/chargeB on the
   wrong side of `indexB`, so each monomer is odd-electron at its published charge. `ext_biodimers` reassigns the
   charges by electron parity and flags `source_charges_swapped`. Offset-free shape scores are unaffected, because
   the rigid-monomer terms are constant along a curve. Absolute E_int and any relabel are affected.
   - **Docs:** state this in `docs/datasets.md`.
2. **Tier A is level-matched: ωB97M-V/def2-TZVPD with open boundaries.**
   - **Size:** 976 E_int points on 121 curves (40 ion-pair, 16 like-charge, 40 ion–neutral and 25 neutral), built
     from 1,218 ORCA points (dimer points ≥ 5 Å plus the rigid monomers). None are excluded.
   - **Monomers:** they are their own calcs. The 17 swapped ion–neutral curves in tier A use monomers recomputed at
     the corrected charges.
3. **No CT at ≥ 9 Å, at this level, on what was checked.** Löwdin fragment charges at ≥ 9 Å equal the formal charges
   to 1e-4 e on every ion-pair, like-charge and ion–neutral point of the first 104 curves. The 17 recomputed
   ion–neutral curves were not re-checked, so the claim covers 104 of 121 curves.
4. **The 120 `ct_candidate` points stay labelled.** These are the tier-A ion-pair points ≥ 9 Å; the full source set
   has 516.
   - **Keep the label** and the label's own row, even though (3) found no CT. The label is provisional and comes
     from charges only.
   - **Change the flag note** in `ext_biodimers.py`. It still says the points "may sit on a CT state"; it could say
     that tier-A ωB97M-V found none on the checked curves.
5. **The source labels are not a substitute for a level-matched label.** Beyond 8 Å, the source labels' offset-free
   shapes differ from the ωB97M-V shapes by these medians; the source shapes are cell-corrected HSE06 (75 % HF) +
   MBD-NL:

   | class | median shape difference | coverage |
   |---|---|---|
   | ion pairs | 51 meV, 22 % of the 230 meV mean \|dE\| | all tier-A ion-pair curves |
   | like-charge | 32 meV | all tier-A like-charge curves |
   | ion–neutral | 0.9 meV | first 23 curves only |
   | neutral | 0.1 meV | all tier-A neutral curves |

   One model's ion-pair tail fraction beyond 9.5 Å read 1.21 against the source and 0.91 against ωB97M-V on the same
   curves.
   - **Docs:** `docs/datasets.md` should say that `ext_biodimers` shape scores are end-to-end in the functional, not
     only in the cell.

## (c) Conventions (already the house rules; restated so the port keeps them)

- **Offset-free shapes:** dE(d) = E_int(d) − E_int(d_last). The tail fraction is the slope through 0 of model dE
  on reference dE at d ≥ 8 Å. The offset-free MAE is always quoted next to the flat-model MAE. A periodic-cell
  reference (`ext_biodimers`) is scored on shapes only.
- **Level matching:** never pool levels. Scores against CCSD(T) or HSE06 references are end-to-end, never "ML
  error". Tier A is the only biodimer set on which an OMol25-level model's error is the model's own. The level
  refusal still does not fire for `ext_*` sets (see (e)), so pass `--allow-level-mismatch` deliberately and say so in
  the result.
- **CT-ambiguous and unbound points are kept and labelled**, in their own rows (`ct_candidate`, `ct_ambiguous`,
  `unbound_anion_candidate`). The only exclusions are `dft_pathology_do_not_score` and `scf_wrong_state`.
- **Bootstrap by curve:** hierarchical, resampling curves and then points within each curve. Points on one curve are
  never independent samples. Four seeds make a claim; one or two seeds make a screen.

## (d) Redistribution limits

Some group-internal references are **not for redistribution**:
- an interface/slab set;
- the multivalent-ion and W18 sets.

A loader may hold a pointer to the owner's location and must read from there. No data, derived numbers or file paths
from these sets go into this repository. `CHANGELOG.md` already states this in general terms; keep it that general.

## (e) Open items from the workspace changelog

Not yet recorded here:
- **External sets: run `ext_ncia` / `ext_gscdb_nc` at the end of every model release.** An earlier run returned 40
  non-finite points, all on structures where a lone atom sits ≥ 8 Å from the rest (RG10N, D442x10). The cause was a
  backend (mlip.cpp) error, not the benchmark. The NaN guard in the scorer is worth keeping.
- **Not staged:** Dip146 / Pol130 (need models that output dipoles; `props`) and the MACE-POLAR-1 ml-peg long-range
  protocols.
- **Optional relabels, not started:**
  - the full stratified biodimer relabel (≈ 3.5 k allocated core-h);
  - the NCI Atlas far points, to measure the CCSD(T) − ωB97M-V gap at 8 Å for dispersion pairs (≈ 16 core-h).
- **SO3LR(-T) on `omol25_val_lr`:** the ASE path recompiles for every atom count, so use a padded route.

Already recorded here, nothing to add:
- the level refusal for reference-only sets;
- third-party sets as pointers only;
- no unit tests or `--help` for `tools/ext_*.py`;
- `seen_by` not written into ext records;
- conformal cells with n < 19.

## (f) PET fp32 cutoff staircase

For PET models only: [metatensor/metatrain#1339](https://github.com/metatensor/metatrain/issues/1339), with the fix
in PR [#1340](https://github.com/metatensor/metatrain/pull/1340); both are open as of this note.

- **Mechanism:** PET's bump cutoff enters attention as log f_c. Evaluated as `0.5 (1 + tanh y)` in fp32, f_c is
  quantised near the pair cutoff. Energy then moves in **steps** as a pair crosses it. On pet-omol-s v1.0.0 (CPU,
  metatrain main) the fp32 curve has five steps and deviates from fp64 by up to 84 meV; the forces on the flat parts
  are tens of eV/Å. In fp64 the curve is smooth.
- **Why it matters here:** any PET model evaluated in fp32 near its cutoff picks up these steps. The `seam`,
  `energy_force` and `nve_short` checks will see them, along with any 6–9 Å band of a scan. They come from the
  arithmetic, not from the model's learned physics.

**Suggested check.** Run the bare-ion or ion–neutral closest-contact scan (6–9 Å, 0.01 Å steps) for one PET model
twice, in fp32 and fp64, and add `precision`'s fp64 reference to the `seam` report. A seam failure that disappears
in fp64 belongs to this issue, and the result should say so. Once #1340 is released, record the metatrain version
in the model metadata.
