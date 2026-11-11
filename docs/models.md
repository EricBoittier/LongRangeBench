# Models

Nothing about a model is special-cased in the metrics: a model is anything that can return an energy (and ideally
forces, and ideally per-atom charges) for a structure with a given total charge and multiplicity. How it does that is
the backend's problem.

## Model specs

`--model` takes any of these:

| spec | backend |
|---|---|
| `ase:module:Factory?kw=v&kw2=v2` | any ASE calculator by import path. Values are parsed as Python literals; quote strings (`?model='polar-1-m'`) |
| `model.pt` | metatomic / TorchScript export, through `metatomic-ase`. No level is implied by a bare path, so pass `--model-level` |
| `ase:lrbench.adapters:metatomic?path='model.pt'` | the same export, but with its level inferred from the file name and its charge handling read off the model — prefer this |
| `model.gguf` | mlip.cpp, via the `mlipcpp-eval` binary; else the C API (`--mlipcpp-lib` or `$MLIPCPP_C_LIB`); else the Python bindings, which pass no charge or spin and are flagged in the result |
| `mlipcpp-capi:model.gguf` | force the C API path |
| `oracle:multipole?order=2&ind=1&moments=frags.json` | the multipole oracle, not a model — see [metrics.md](metrics.md#the-multipole-oracle) |
| a name from `lrbench models` | a released model, with its environment, level and licence |

`--ensemble a.gguf,b.gguf,...` scores a seed ensemble: predictions are averaged for scoring and the member spread
feeds the normalised conformal intervals.

### Metatomic exports, and what "takes a charge" is allowed to mean

Any model exported to metatomic TorchScript — `mtt export`, i.e. most of the metatensor/metatomic ecosystem — can be
scored here, and that is the widest door into the benchmark. `metatomic-ase` reads the total charge from
`atoms.info["charge"]` and the multiplicity from `atoms.info["spin_multiplicity"]`, which is exactly where this
benchmark puts them, so a conditioned model is told the charge structure by structure through the same path as every
other adapter.

Prefer the `metatomic` adapter over a bare `model.pt` path, for two reasons:

```sh
lrbench eval --model "ase:lrbench.adapters:metatomic?path='checkpoints/pet-omol-s.pt'" --sets all --data ./lr-sets
```

**The level is inferred from the file name.** A bare `model.pt` carries no level, so its energies are scored but
never level-checked, and a quiet mismatch is the one error this benchmark most wants to avoid. The adapter reads the
dataset token out of a `upet`-style name (`pet-omol-*` → OMol25 `wB97M-V/def2-TZVPD`, `pet-spice-*` → SPICE,
`pet-mad-*` → PBEsol/PAW, …), longest token first so `pet-omatpes-l` is not read as OMat24. Pass `level=` for a
checkpoint whose name does not say.

**Whether the model takes a charge is read off the model, not guessed.** `AtomisticModel.requested_inputs()` is the
model's own declaration of what it consumes, and it sets `charge_spin`. A model that requests `charge` is handed one
and scored on what it does with it. A model that requests nothing is recorded as *unable to be told* a charge, and is
never handed one — so "ignored the charge" keeps meaning only what it says, and is never confused with "was never
asked". Of the released PET checkpoints only the `pet-omol-*` family requests `charge` and `spin_multiplicity`;
`pet-spice-*`, `pet-mols-*` and `pet-omad-*` request nothing, which is verifiable in one line: the three energies for
Q = 0, −1, +1 come back bit-identical.

This route needs `metatrain >= 2026.4` to read checkpoint version 16, which is what the current `upet` checkpoints
are; 2025.12 reads only up to v9 and fails with a version error.

### mlip.cpp GGUFs and the build that reads them

A GGUF from a long-range branch of mlip.cpp only means what it was trained to mean when it is run by **that
branch's build**. The extra terms live in metadata keys (`pet.plato_lr.lrc_qproj`, `lrc_links`, `lrc_states`, ...);
a build that predates them loads the file without an error and ignores the keys, and the energies it returns belong
to a different model. Charge-head models also name companion GGUFs (`lrc_qmodel = q.gguf`, `lrc_fmodel`), resolved
next to the model file.

`configs/gguf_models.json` (or `$LRBENCH_GGUF_REGISTRY`) names such models and binds each to its mlip.cpp commit,
with the sha256 of the weights and of every companion. Paths are written against `$LRBENCH_GGUF_ROOT`
(`<name>/model.gguf` plus companions) and `$LRBENCH_MLIPCPP_BUILDS` (`<commit>/bin/mlipcpp-eval`, or
`<commit>-cuda/...` for `--device cuda`, each with a `COMMIT` file). A registered name then works anywhere a model
name does:

```sh
lrbench eval --model ftq13 --sets all --data ./lr-sets --device cuda --out ftq13.json
python -m lrbench.gguf_registry check            # hashes, companions, build commit and metadata keys, every entry
python -m lrbench.gguf_registry smoke ftq        # one charged structure per model of a group, energy + forces
python -m lrbench.gguf_registry add NAME model.gguf --build COMMIT --binary .../mlipcpp-eval   # a new entry
```

Before it runs, the backend refuses a binary built from another commit, a binary that lacks a metadata key the
build of record contains, and a missing or changed companion. `LRBENCH_ALLOW_BINARY_MISMATCH=1` scores anyway and
records the mismatch in the member block (`gguf_registry.binary_check`). The key check is a string search of the
binary. Keys the build of record itself does not contain as literals (`keys_not_literal`, composed at run time) are
not held against another binary, so the commit check is the primary guard.

The registry holds identities, not weights: unreleased entries record a sha256 so that a result can name exactly
what was scored.

### Charges and spin

`--formal-charges auto` passes per-atom formal charges only to models that declare they use them; `from-data` always
passes them; `none` never does. Whether a backend *can* take a total charge, a multiplicity or formal charges is
recorded per member in the result JSON (`members[].backend_can_pass_formal_charges`, `charge_spin`) — a model that
silently ignores the charge on an ion pair is not quietly scored as if it had used it.

Why this matters: half the long-range failures in these sets are not approximation errors but plumbing. A model that
cannot be told that a fragment carries −1 cannot get an ion-pair tail right, no matter how good its architecture is.

### Precision

fp32 by default (`GGML_CUDA_NO_TF32=1`). `--tf32` sets `MLIPCPP_TF32=1`. `NVIDIA_TF32_OVERRIDE` is always removed
from the backend environment, so a stray export in a shell profile cannot silently change a published number.

## Levels of theory

Every model carries the level it was trained at, and every record the level of its reference. Mismatches are refused
unless `--allow-level-mismatch` is passed, and then flagged everywhere. See
[metrics.md](metrics.md#levels-of-theory) and [metrics.md](metrics.md#element-offsets) for the offset handling that
makes absolute energies comparable at all.

## Named models

The tables between the markers are generated by `lrbench results render` (from `lrbench models --md`, with local
checkpoint directories written as the variables that set them); CI fails if they are out of date.

<!-- BEGIN lrbench results render: models -->
| name | model | spec | energy level | env | checkpoint | licence | total Q / spin | long range |
|---|---|---|---|---|---|---|---|---|
| `so3lr` | SO3LR (original, v1) | `lrbench.adapters:so3lr?model='so3lr'` | PBE0+MBD | so3lr | bundled in the so3lr package (registry name 'so3lr') | code MIT; weights per the so3lr NOTICES | total charge + multiplicity | learned charges, Coulomb + dispersion, no cutoff in vacuum |
| `so3lr-tauri` | SO3LR Tauri M (2606) | `lrbench.adapters:so3lr?model='so3lr-tauri-m-2606'` | PBE0+MBD-NL | so3lr | bundled in the so3lr package (so3lr-tauri-m-2606) | code MIT; Tauri weights CC BY-NC 4.0 | total charge + multiplicity | equilibrated charges, erf Coulomb + dispersion, no cutoff in vacuum |
| `so3lr-tauri-s` | SO3LR Tauri S (2606) | `lrbench.adapters:so3lr?model='so3lr-tauri-s-2606'` | PBE0+MBD-NL | so3lr | bundled in the so3lr package | code MIT; Tauri weights CC BY-NC 4.0 | total charge + multiplicity | as so3lr-tauri |
| `so3lr-tauri-l` | SO3LR Tauri L (2606) | `lrbench.adapters:so3lr?model='so3lr-tauri-l-2606'` | PBE0+MBD-NL | so3lr | bundled in the so3lr package | code MIT; Tauri weights CC BY-NC 4.0 | total charge + multiplicity | as so3lr-tauri |
| `mace-h` | MACE-MH-1, OMol head | `lrbench.adapters:mace_mh?head='omol'` | wB97M-V/def2-TZVPD | mace | mace-torch download 'mh-1' (~/.cache/mace) | ASL (Academic Software License) | checked at load time; ignored if the model has no charge/spin embedding (recorded) | none (local message passing) |
| `mace-polar` | MACE-POLAR-1 M | `lrbench.adapters:mace_polar?model='polar-1-m'` | wB97M-V/def2-TZVPD | mace | mace-torch download 'polar-1-m' (~/.cache/mace) | ASL (Academic Software License) | total charge + spin (2S+1) | polarisable charge + spin densities, real-space Coulomb in vacuum |
| `mace-polar-l` | MACE-POLAR-1 L | `lrbench.adapters:mace_polar?model='polar-1-l'` | wB97M-V/def2-TZVPD | mace | mace-torch download 'polar-1-l' (~/.cache/mace) | ASL (Academic Software License) | total charge + spin (2S+1) | as mace-polar |
| `mace-omol` | MACE-OMOL-0 (extra large) | `lrbench.adapters:mace_omol` | wB97M-V/def2-TZVPD | mace | mace-torch download (~/.cache/mace) | ASL (Academic Software License) | total charge + spin embeddings | none (local message passing) |
| `maceles-off` | MACELES-OFF (Cheng group, MACE-LES) | `lrbench.adapters:maceles?model='$LRBENCH_LES_MODELS/MACELES-OFF_small_converted.model'` | wB97M-D3(BJ)/def2-TZVPPD | mace | MACELES-OFF/MACELES-OFF_small_converted.model, github.com/ChengUCB/les_fit (CC BY-NC 4.0); set LRBENCH_LES_MODELS to the download dir | CC BY-NC 4.0 | none: neutral closed-shell only (charged classes N/A); elements H C N O F P S Cl Br I | Latent Ewald Summation, vacuum |
| `caceles-water` | CACE-LES water (Cheng group) | `lrbench.adapters:cace?model='$LRBENCH_LES_MODELS/caceles-r-4.5-nl-1-nu-2_best_model.pth'` | RPBE-D3 (water, CACE-LES training set) | mace | MLIPs/CACE-LES/water/caceles-r-4.5-nl-1-nu-2/best_model.pth, github.com/ChengUCB/les_fit (CC BY-NC 4.0); set LRBENCH_LES_MODELS to the download dir | CC BY-NC 4.0 | none: neutral only; H and O only | Latent Ewald Summation |
| `orbmol` | OrbMol-v2 (learnable electrostatics) | `lrbench.adapters:orbmol?model='orbmol-v2'` | wB97M-V/def2-TZVPD | orb | orb-models download (orbmol-v2-*.ckpt, Orbital Materials S3) | Apache-2.0 | atoms.info charge + spin (2S+1) | latent charges constrained to the total charge + Coulomb (bare 1/r in vacuum, PME in a cell) |
| `orbmol-v1` | OrbMol v1 conservative | `lrbench.adapters:orbmol?model='orbmol-v1'` | wB97M-V/def2-TZVPD | orb | orb-models download (orb-v3-conservative-omol-*.ckpt) | Apache-2.0 | atoms.info charge + spin (2S+1) | none (local message passing, 6 A cutoff) |
| `dpa4-plus` | DPA4-Plus, OMol25 100M (8.8 M params) | `lrbench.adapters:deepmd?model='$LRBENCH_DPA4_DIR/DPA4-Plus-OMol25-100M-v20260820.pt'&name='DPA4-Plus-OMol25-100M'` | wB97M-V/def2-TZVPD | deepmd | huggingface.co/deepmodelingcommunity/DPA4-OMol | CC BY-NC 4.0 | atoms.info charge_spin = [Q, 2S+1], FiLM-conditioned descriptor | none (local, 6 A cutoff); the total charge only conditions the descriptor |
| `dpa4-air` | DPA4-Air, OMol25 100M (5.1 M params) | `lrbench.adapters:deepmd?model='$LRBENCH_DPA4_DIR/DPA4-Air-OMol25-100M-v20260820.pt'&name='DPA4-Air-OMol25-100M'` | wB97M-V/def2-TZVPD | deepmd | huggingface.co/deepmodelingcommunity/DPA4-OMol | CC BY-NC 4.0 | as dpa4-plus | as dpa4-plus |
| `uma-s` | UMA-S 1.2.1, omol task | `lrbench.adapters:fairchem?model='uma-s-1p2p1'&task='omol'` | wB97M-V/def2-TZVPD | fairchem | huggingface.co/facebook/UMA (gated) | Meta FAIR Chemistry model licence | atoms.info charge + spin (2S+1), as an embedding | none (local message passing) |
| `esen-omol` | eSEN-SM conserving, all of OMol25 | `lrbench.adapters:fairchem?model='esen-sm-conserving-all-omol'` | wB97M-V/def2-TZVPD | fairchem | fairchem-core download (gated on Hugging Face) | Meta FAIR Chemistry model licence | atoms.info charge + spin (2S+1), as an embedding | none (local message passing) |
| `aimnet2` | AIMNet2 (Coulomb, no cutoff) | `lrbench.adapters:aimnet2?lrcoulomb='simple'` | wB97M-D3/def2-TZVPPD | aimnet | fetched by the aimnet package on first use | MIT (code); see the model card for the weights | set_charge / set_mult on the calculator, not atoms.info; charges sum to the total charge | predicted partial charges + Coulomb over all pairs, no cutoff (vacuum) |
| `aimnet2-dsf` | AIMNet2, DSF Coulomb (15 A cutoff) | `lrbench.adapters:aimnet2?lrcoulomb='dsf'&cutoff=15.0` | wB97M-D3/def2-TZVPPD | aimnet | as aimnet2 | MIT (code); see the model card for the weights | as aimnet2 | damped-shifted-force Coulomb, 15 A cutoff: truncates the tails this benchmark measures, and is reported as its own row rather than silently substituted |
| `pet-mad` | PET-MAD (universal, MAD) | `lrbench.adapters:pet_mad?version='latest'` | PBEsol/PAW (MAD) | pet-mad | huggingface.co/lab-cosmo/pet-mad (fetched on first use) | BSD-3-Clause | none: no total-charge or spin input | none (local message passing) |
| `pet-omol-s` | PET-OMol-S (charge-conditioned) | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-omol-s.pt'&name='PET-OMol-S'` | wB97M-V/def2-TZVPD | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-omol-s-v1.0.0) | BSD-3-Clause | atoms.info charge + spin_multiplicity (declared by the model) | none (local message passing) |
| `pet-omol-m` | PET-OMol-M (charge-conditioned) | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-omol-m.pt'&name='PET-OMol-M'` | wB97M-V/def2-TZVPD | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-omol-m-v1.0.0) | BSD-3-Clause | as pet-omol-s | none (local message passing) |
| `pet-omol-l` | PET-OMol-L (charge-conditioned) | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-omol-l.pt'&name='PET-OMol-L'` | wB97M-V/def2-TZVPD | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-omol-l-v1.0.0) | BSD-3-Clause | as pet-omol-s | none (local message passing) |
| `pet-spice-s` | PET-SPICE-S | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-spice-s.pt'&name='PET-SPICE-S'` | wB97M-D3(BJ)/def2-TZVPPD | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-spice-s-v0.2.0) | BSD-3-Clause | none: the model requests no charge input | none (local message passing) |
| `pet-spice-l` | PET-SPICE-L | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-spice-l.pt'&name='PET-SPICE-L'` | wB97M-D3(BJ)/def2-TZVPPD | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-spice-l-v0.2.0) | BSD-3-Clause | as pet-spice-s | none (local message passing) |
| `pet-mols-s` | PET-MOLS-S (mixed molecular) | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-mols-s.pt'&name='PET-MOLS-S'` | mixed | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-mols-s-v1.1.0) | BSD-3-Clause | as pet-spice-s | none (local message passing) |
| `pet-omad-s` | PET-OMAD-S (OMat24 + MAD) | `lrbench.adapters:metatomic?path='$LRBENCH_UPET_DIR/pet-omad-s.pt'&name='PET-OMAD-S'` | mixed | metatomic | huggingface.co/lab-cosmo/upet, exported with `mtt export` (pet-omad-s-v1.0.0) | BSD-3-Clause | as pet-spice-s | none (local message passing) |
| `grace-2l` | GRACE-2L-OMAT (two-layer semi-local) | `lrbench.adapters:grace?model='GRACE-2L-OMAT'` | PBE/PAW (OMat24) | grace | grace_models download / tensorpotential auto-download (~/.cache/grace) | Academic Software License | none: no total-charge or spin input | none (local, 6 A cutoff) |
| `grace-1l` | GRACE-1L-OMAT (one-layer local) | `lrbench.adapters:grace?model='GRACE-1L-OMAT'` | PBE/PAW (OMat24) | grace | tensorpotential auto-download | Academic Software License | none: no total-charge or spin input | none (local, 6 A cutoff) |

| env | python | install | notes |
|---|---|---|---|
| so3lr | >=3.12,<3.14 | `uv pip install "so3lr[cuda12] @ git+https://github.com/general-molecular-simulations/so3lr@2a7480cb38cb0c75bc89784f32f8e253c1a70153"`<br>`uv pip install -e ".[test]"` | JAX < 0.11 with CUDA 12 (jaxlib 0.11.1 hangs in the PME scatter on CPU). Checkpoints ship inside the package (no download). x64 is switched on by the adapter (float64). |
| mace | >=3.10 | `uv pip install "mace-torch[polar]>=0.3.16"`<br>`uv pip install -e ".[test]"` | torch with CUDA matching the node. Checkpoints are downloaded by mace-torch on first use to ~/.cache/mace: pre-fetch on a login node (compute nodes may have no internet), or set XDG_CACHE_HOME to shared storage. |
| orb | >=3.10 | `uv pip install orb-models`<br>`uv pip install -e ".[test]"` | torch with CUDA matching the node. Checkpoints are downloaded from Orbital Materials' public S3 bucket on first use (~100 MB each) and cached by orb-models. |
| deepmd | >=3.10 | `uv pip install "deepmd-kit[torch]" ase`<br>`uv pip install -e ".[test]"` | torch with CUDA matching the node. The DPA4 OMol25 checkpoints are downloaded separately from huggingface.co/deepmodelingcommunity/DPA4-OMol; set LRBENCH_DPA4_DIR to where they are. |
| fairchem | >=3.10 | `uv pip install fairchem-core`<br>`uv pip install -e ".[test]"` | The weights are gated: accept the licence on huggingface.co/facebook/UMA and log in (hf auth login) before the first run. Checkpoints are cached in ~/.cache/fairchem (uma-s is ~2.3 GB). |
| aimnet | >=3.11,<3.13 | `uv pip install "aimnet[ase]"`<br>`uv pip install -e ".[test]"` | torch with CUDA matching the node. Checkpoints are fetched on first use. The total charge is set on the calculator rather than read from atoms.info, which the adapter does for every structure; the docs warn that a wrong charge is silently wrong, so that plumbing matters. |
| pet-mad | >=3.11 | `uv pip install pet-mad`<br>`uv pip install -e ".[test]"` | torch, metatomic and metatrain come with it. The checkpoint is fetched from huggingface.co/lab-cosmo/pet-mad on first use. |
| metatomic | >=3.11 | `uv pip install "metatrain[pet]>=2026.4" metatomic-ase`<br>`uv pip install -e ".[test]"` | metatrain 2026.4 or newer is required to read checkpoint version 16, which is what the current upet checkpoints are; 2025.12 reads only up to v9 and fails with a version error. The models enter the benchmark as exported TorchScript, so each checkpoint is converted once:
    mtt export models/pet-omol-s-v1.0.0.ckpt -o $LRBENCH_UPET_DIR/pet-omol-s.pt
metatomic-ase reads the total charge from atoms.info['charge'] and the multiplicity from atoms.info['spin_multiplicity'], so a model that declares those inputs is told them -- which is what lets the charge-conditioned PET models be scored here at all, with no mlip.cpp in the path. |
| grace | >=3.10 | `uv pip install tensorpotential`<br>`uv pip install -e ".[test]"` | TensorFlow; set TF_FORCE_GPU_ALLOW_GROWTH=true so it does not take the whole device. Models are downloaded on first use to ~/.cache/grace (GRACE_CACHE overrides). |
<!-- END lrbench results render: models -->

### What has actually been run

A registry row means the adapter exists and declares the model's level, licence and charge handling. It does not
mean somebody has scored it. As of this writing:

- **Run on the canonical sets here:** `mace-polar`, `mace-polar-l`, `mace-h`, `mace-omol`, `so3lr`,
  `so3lr-tauri{,-s,-l}`, `orbmol`, `orbmol-v1`, `dpa4-plus`, `dpa4-air`, `aimnet2`, `grace-2l`.
- **Written but unverified:** `uma-s`, `esen-omol`. The weights are gated on Hugging Face and the account used here
  is not on the authorised list, so the adapter has never been exercised against a real checkpoint. Treated as
  unverified until someone with access runs it.
- **Declared out of domain:** `grace-2l` / `grace-1l` are the ACE family's foundation models, trained on periodic
  materials data at PBE/PAW with no charge input. They are a floor, not a competitor: an eval is refused without
  `--allow-level-mismatch`, and `charge_spin=False` makes every result say the model could not be *told* the charge
  rather than appearing to have ignored one. Same idea for `maceles-off` and `caceles-water`, which refuse charged
  structures outright.

Numbers with provenance live in [results/](../results/SUMMARY.md), one validated row per model and set:

<!-- BEGIN lrbench results render: results -->
Committed result rows ([results/SUMMARY.md](../results/SUMMARY.md)):

- **charge_parity:** `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen)
- **dimers:** `aimnet2-dsf` (screen), `aimnet2` (screen), `caceles-water` (screen), `dpa4-air` (screen), `dpa4-plus` (screen), `grace-2l` (screen), `mace-h` (screen), `mace-omol` (screen), `mace-polar-l` (screen), `mace-polar` (screen), `maceles-off` (screen), `orbmol-v1` (screen), `orbmol` (screen), `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen), `so3lr-tauri-l` (screen), `so3lr-tauri-s` (screen), `so3lr-tauri` (screen), `so3lr` (screen)
- **ext_biodimers_wb97mv:** `aimnet2-dsf` (screen), `aimnet2` (screen), `dpa4-air` (screen), `dpa4-plus` (screen), `grace-2l` (screen), `mace-h` (screen), `mace-omol` (screen), `mace-polar-l` (screen), `mace-polar` (screen), `orbmol-v1` (screen), `orbmol` (screen), `pet-mad` (screen), `pet-mols-s` (screen), `pet-omol-s` (screen), `pet-spice-s` (screen), `so3lr-tauri-s` (screen), `so3lr-tauri` (screen), `so3lr` (screen)
- **ext_gscdb_nc:** `aimnet2-dsf` (screen), `aimnet2` (screen), `caceles-water` (screen), `dpa4-air` (screen), `dpa4-plus` (screen), `grace-2l` (screen), `mace-h` (screen), `mace-omol` (screen), `mace-polar-l` (screen), `mace-polar` (screen), `maceles-off` (screen), `orbmol-v1` (screen), `orbmol` (screen), `pet-mad` (screen), `pet-mols-s` (screen), `pet-omol-s` (screen), `pet-spice-s` (screen), `so3lr-tauri-s` (screen), `so3lr-tauri` (screen), `so3lr` (screen)
- **families_corrected:** `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen)
- **families_fixedq:** `aimnet2-dsf` (screen), `aimnet2` (screen), `dpa4-air` (screen), `dpa4-plus` (screen), `grace-2l` (screen), `mace-h` (screen), `mace-omol` (screen), `mace-polar-l` (screen), `mace-polar` (screen), `maceles-off` (screen), `orbmol-v1` (screen), `orbmol` (screen), `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen), `so3lr-tauri-l` (screen), `so3lr-tauri-s` (screen), `so3lr-tauri` (screen), `so3lr` (screen)
- **families_full:** `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen)
- **trimers:** `aimnet2-dsf` (screen), `aimnet2` (screen), `dpa4-air` (screen), `dpa4-plus` (screen), `grace-2l` (screen), `mace-h` (screen), `mace-omol` (screen), `mace-polar-l` (screen), `mace-polar` (screen), `maceles-off` (screen), `orbmol-v1` (screen), `orbmol` (screen), `pet-mols-s` (screen), `pet-omad-s` (screen), `pet-omol-l` (screen), `pet-omol-m` (screen), `pet-omol-s` (screen), `pet-spice-l` (screen), `pet-spice-s` (screen), `so3lr-tauri-l` (screen), `so3lr-tauri-s` (screen), `so3lr-tauri` (screen), `so3lr` (screen)
<!-- END lrbench results render: results -->

The `orbmol-v1` / `orbmol` pair is the most informative comparison in the table: one architecture, trained on the
same data, with and without a latent-charge head and a Coulomb module. Differences between those two rows are the
long-range term and nothing else.

Groups usable with `lrbench suite --models <group>`: `external`, `lr-comparison`, `all-external`. Names are
case-insensitive and aliases work (`tauri`, `mace-les`, …).

Each model family needs its own environment — JAX for SO3LR, PyTorch for MACE — and LongRangeBench itself needs only
numpy, scipy and ase. `lrbench suite` runs one eval per model with that environment's Python and writes a single
combined table; a model whose environment cannot import its package is **skipped with the reason** and the table is
still produced.

```sh
source configs/external_envs.sh      # LRBENCH_PY_SO3LR, LRBENCH_PY_MACE
lrbench suite --models external --sets all --data ./lr-sets --outdir runs/external -- --normalise --conformal
# plan only: writes run_<model>.sh and a Slurm array, executes nothing
lrbench suite --models external --sets all --data ./lr-sets --outdir runs/external \
    --sbatch-header configs/slurm-gpu.sbatch-header --dry-run -- --normalise
```

`--outdir` receives `suite.json` (per model: command, environment, status ok / failed / skipped / planned),
`<model>.json`, `<model>.log`, and `combined.md` / `combined.html` / `figs/`. Arguments after `--` go to every eval.

## Adding one

[adding-a-model.md](adding-a-model.md). An ASE calculator needs no code in this repository at all.
