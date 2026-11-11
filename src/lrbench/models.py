"""Named external long-range models, so every lrbench command takes a short name instead of a full ASE spec:

    lrbench eval --model mace-polar --sets all --out polar.json
    lrbench suite --models so3lr,so3lr-tauri,mace-h,mace-polar --sets all --outdir runs/external

Each entry says what the model needs to run (environment, packages, checkpoint, licence) and which level of theory its
energy head was trained at. The table in README.md ("Named models") is generated from the same data
(``python -m lrbench.models --md``). Names are case-insensitive, and ``_`` is accepted for ``-``.

The four models of the external comparison (``EXTERNAL``):
  - ``so3lr``: the original SO3LR (v1, "vanilla" in the so3lr registry), PBE0+MBD labels;
  - ``so3lr-tauri``: SO3LR Tauri M, release 2606 (pinned, not the moving ``so3lr-tauri-m`` alias), PBE0+MBD-NL;
  - ``mace-h``: MACE-MH-1, read through its OMol head (wB97M-V/def2-TZVPD). "MACE-H" is the user's name; MACE-MH-1
    is the released multi-head MACE whose name fits it. Other heads: ``mace-h:<head>``;
  - ``mace-polar``: MACE-POLAR-1 M (polarisable charge + spin densities), OMol25 labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

OMOL = "wB97M-V/def2-TZVPD"
SO3LR_ENV = "so3lr"
MACE_ENV = "mace"
ORB_ENV = "orb"
DEEPMD_ENV = "deepmd"
FAIRCHEM_ENV = "fairchem"
GRACE_ENV = "grace"
PETMAD_ENV = "pet-mad"
AIMNET_ENV = "aimnet"
METATOMIC_ENV = "metatomic"

# What each environment needs. One venv per environment: JAX (SO3LR) and torch (MACE) are not installed together.
ENVS: Dict[str, dict] = {
    SO3LR_ENV: {
        "probe": "so3lr",  # module whose import tells whether the environment can run these models
        "python": ">=3.12,<3.14",
        # There is no v0.2.0 *tag* upstream, only a release/v0.2.0 branch: install the pinned commit, not the tag.
        "install": ['uv pip install "so3lr[cuda12] @ git+https://github.com/general-molecular-simulations/so3lr'
                    '@2a7480cb38cb0c75bc89784f32f8e253c1a70153"',
                    'uv pip install -e ".[test]"'],
        "notes": "JAX < 0.11 with CUDA 12 (jaxlib 0.11.1 hangs in the PME scatter on CPU). Checkpoints ship inside "
                 "the package (no download). x64 is switched on by the adapter (float64).",
    },
    MACE_ENV: {
        "probe": "mace.calculators",
        "python": ">=3.10",
        "install": ['uv pip install "mace-torch[polar]>=0.3.16"', 'uv pip install -e ".[test]"'],
        "notes": "torch with CUDA matching the node. Checkpoints are downloaded by mace-torch on first use to "
                 "~/.cache/mace: pre-fetch on a login node (compute nodes may have no internet), or set "
                 "XDG_CACHE_HOME to shared storage.",
    },
    ORB_ENV: {
        "probe": "orb_models.forcefield.pretrained",
        "python": ">=3.10",
        "install": ['uv pip install orb-models', 'uv pip install -e ".[test]"'],
        "notes": "torch with CUDA matching the node. Checkpoints are downloaded from Orbital Materials' public S3 "
                 "bucket on first use (~100 MB each) and cached by orb-models.",
    },
    DEEPMD_ENV: {
        "probe": "deepmd.calculator",
        "python": ">=3.10",
        "install": ['uv pip install "deepmd-kit[torch]" ase', 'uv pip install -e ".[test]"'],
        "notes": "torch with CUDA matching the node. The DPA4 OMol25 checkpoints are downloaded separately from "
                 "huggingface.co/deepmodelingcommunity/DPA4-OMol; set LRBENCH_DPA4_DIR to where they are.",
    },
    FAIRCHEM_ENV: {
        "probe": "fairchem.core",
        "python": ">=3.10",
        "install": ['uv pip install fairchem-core', 'uv pip install -e ".[test]"'],
        "notes": "The weights are gated: accept the licence on huggingface.co/facebook/UMA and log in (hf auth "
                 "login) before the first run. Checkpoints are cached in ~/.cache/fairchem (uma-s is ~2.3 GB).",
    },
    AIMNET_ENV: {
        "probe": "aimnet.calculators",
        "python": ">=3.11,<3.13",
        "install": ['uv pip install "aimnet[ase]"', 'uv pip install -e ".[test]"'],
        "notes": "torch with CUDA matching the node. Checkpoints are fetched on first use. The total charge is "
                 "set on the calculator rather than read from atoms.info, which the adapter does for every "
                 "structure; the docs warn that a wrong charge is silently wrong, so that plumbing matters.",
    },
    PETMAD_ENV: {
        "probe": "pet_mad.calculator",
        "python": ">=3.11",
        "install": ['uv pip install pet-mad', 'uv pip install -e ".[test]"'],
        "notes": "torch, metatomic and metatrain come with it. The checkpoint is fetched from "
                 "huggingface.co/lab-cosmo/pet-mad on first use.",
    },
    METATOMIC_ENV: {
        "probe": "metatomic.torch",
        "python": ">=3.11",
        "install": ['uv pip install "metatrain[pet]>=2026.4" metatomic-ase',
                    'uv pip install -e ".[test]"'],
        "notes": "metatrain 2026.4 or newer is required to read checkpoint version 16, which is what the current "
                 "upet checkpoints are; 2025.12 reads only up to v9 and fails with a version error. The models "
                 "enter the benchmark as exported TorchScript, so each checkpoint is converted once:\n"
                 "    mtt export models/pet-omol-s-v1.0.0.ckpt -o $LRBENCH_UPET_DIR/pet-omol-s.pt\n"
                 "metatomic-ase reads the total charge from atoms.info['charge'] and the multiplicity from "
                 "atoms.info['spin_multiplicity'], so a model that declares those inputs is told them -- which is "
                 "what lets the charge-conditioned PET models be scored here at all, with no mlip.cpp in the path.",
    },
    GRACE_ENV: {
        "probe": "tensorpotential.calculator",
        "python": ">=3.10",
        "install": ['uv pip install tensorpotential', 'uv pip install -e ".[test]"'],
        "notes": "TensorFlow; set TF_FORCE_GPU_ALLOW_GROWTH=true so it does not take the whole device. Models are "
                 "downloaded on first use to ~/.cache/grace (GRACE_CACHE overrides).",
    },
}


@dataclass
class NamedModel:
    name: str
    spec: str            # ase:lrbench.adapters:... spec that lrbench's backends understand
    title: str
    level: str           # training level of the energy head that is read
    env: str             # key of ENVS
    checkpoint: str      # where the weights come from
    licence: str
    charge_spin: str
    long_range: str
    aliases: List[str] = field(default_factory=list)

    @property
    def same_level_as_ours(self) -> bool:
        return self.level == OMOL


MODELS: Dict[str, NamedModel] = {m.name: m for m in [
    NamedModel("so3lr", "ase:lrbench.adapters:so3lr?model='so3lr'", "SO3LR (original, v1)", "PBE0+MBD", SO3LR_ENV,
               "bundled in the so3lr package (registry name 'so3lr')", "code MIT; weights per the so3lr NOTICES",
               "total charge + multiplicity", "learned charges, Coulomb + dispersion, no cutoff in vacuum",
               aliases=["so3lr-v1", "so3lr-vanilla"]),
    NamedModel("so3lr-tauri", "ase:lrbench.adapters:so3lr?model='so3lr-tauri-m-2606'", "SO3LR Tauri M (2606)",
               "PBE0+MBD-NL", SO3LR_ENV, "bundled in the so3lr package (so3lr-tauri-m-2606)",
               "code MIT; Tauri weights CC BY-NC 4.0", "total charge + multiplicity",
               "equilibrated charges, erf Coulomb + dispersion, no cutoff in vacuum",
               aliases=["tauri", "so3lr-t", "so3lr-tauri-m"]),
    NamedModel("so3lr-tauri-s", "ase:lrbench.adapters:so3lr?model='so3lr-tauri-s-2606'", "SO3LR Tauri S (2606)",
               "PBE0+MBD-NL", SO3LR_ENV, "bundled in the so3lr package", "code MIT; Tauri weights CC BY-NC 4.0",
               "total charge + multiplicity", "as so3lr-tauri"),
    NamedModel("so3lr-tauri-l", "ase:lrbench.adapters:so3lr?model='so3lr-tauri-l-2606'", "SO3LR Tauri L (2606)",
               "PBE0+MBD-NL", SO3LR_ENV, "bundled in the so3lr package", "code MIT; Tauri weights CC BY-NC 4.0",
               "total charge + multiplicity", "as so3lr-tauri"),
    NamedModel("mace-h", "ase:lrbench.adapters:mace_mh?head='omol'", "MACE-MH-1, OMol head", OMOL, MACE_ENV,
               "mace-torch download 'mh-1' (~/.cache/mace)", "ASL (Academic Software License)",
               "checked at load time; ignored if the model has no charge/spin embedding (recorded)",
               "none (local message passing)", aliases=["mace-mh", "mace-mh-1"]),
    NamedModel("mace-polar", "ase:lrbench.adapters:mace_polar?model='polar-1-m'", "MACE-POLAR-1 M", OMOL, MACE_ENV,
               "mace-torch download 'polar-1-m' (~/.cache/mace)", "ASL (Academic Software License)",
               "total charge + spin (2S+1)", "polarisable charge + spin densities, real-space Coulomb in vacuum",
               aliases=["mace-polar-1", "mace-polar-m"]),
    NamedModel("mace-polar-l", "ase:lrbench.adapters:mace_polar?model='polar-1-l'", "MACE-POLAR-1 L", OMOL, MACE_ENV,
               "mace-torch download 'polar-1-l' (~/.cache/mace)", "ASL (Academic Software License)",
               "total charge + spin (2S+1)", "as mace-polar"),
    NamedModel("mace-omol", "ase:lrbench.adapters:mace_omol", "MACE-OMOL-0 (extra large)", OMOL, MACE_ENV,
               "mace-torch download (~/.cache/mace)", "ASL (Academic Software License)",
               "total charge + spin embeddings", "none (local message passing)"),
]}

LES_ENV = "mace"  # MACE-LES / CACE-LES run in the MACE env with `les` and `cace` added (see ENVS notes)
_LESFIT = "github.com/ChengUCB/les_fit (CC BY-NC 4.0); set LRBENCH_LES_MODELS to the download dir"
import os as _os
_LD = _os.environ.get("LRBENCH_LES_MODELS", _os.path.expanduser("~/.cache/lrbench/les"))
MODELS.update({m.name: m for m in [
    NamedModel("maceles-off", f"ase:lrbench.adapters:maceles?model='{_LD}/MACELES-OFF_small_converted.model'",
               "MACELES-OFF (Cheng group, MACE-LES)", "wB97M-D3(BJ)/def2-TZVPPD", LES_ENV,
               "MACELES-OFF/MACELES-OFF_small_converted.model, " + _LESFIT, "CC BY-NC 4.0",
               "none: neutral closed-shell only (charged classes N/A); elements H C N O F P S Cl Br I",
               "Latent Ewald Summation, vacuum", aliases=["maceles", "mace-les", "mace-les-off"]),
    NamedModel("caceles-water", f"ase:lrbench.adapters:cace?model='{_LD}/caceles-r-4.5-nl-1-nu-2_best_model.pth'",
               "CACE-LES water (Cheng group)", "RPBE-D3 (water, CACE-LES training set)", LES_ENV,
               "MLIPs/CACE-LES/water/caceles-r-4.5-nl-1-nu-2/best_model.pth, " + _LESFIT, "CC BY-NC 4.0",
               "none: neutral only; H and O only", "Latent Ewald Summation", aliases=["cace-les", "caceles"]),
]})

# OrbMol (Orbital Materials) and the released DeePMD DPA4 checkpoints: both trained on OMol25, i.e. the same level
# as our sets, and both take a total charge. OrbMol v1 -> v2 is the cleanest control we have: one architecture with
# and without an explicit Coulomb term.
_DPA4 = _os.environ.get("LRBENCH_DPA4_DIR", _os.path.expanduser("~/.cache/lrbench/dpa4"))
MODELS.update({m.name: m for m in [
    NamedModel("orbmol", "ase:lrbench.adapters:orbmol?model='orbmol-v2'", "OrbMol-v2 (learnable electrostatics)",
               OMOL, ORB_ENV, "orb-models download (orbmol-v2-*.ckpt, Orbital Materials S3)", "Apache-2.0",
               "atoms.info charge + spin (2S+1)",
               "latent charges constrained to the total charge + Coulomb (bare 1/r in vacuum, PME in a cell)",
               aliases=["orbmol-v2", "orb-mol", "orbmol2"]),
    NamedModel("orbmol-v1", "ase:lrbench.adapters:orbmol?model='orbmol-v1'", "OrbMol v1 conservative",
               OMOL, ORB_ENV, "orb-models download (orb-v3-conservative-omol-*.ckpt)", "Apache-2.0",
               "atoms.info charge + spin (2S+1)", "none (local message passing, 6 A cutoff)",
               aliases=["orbmol1", "orb-v3-omol"]),
    NamedModel("dpa4-plus", f"ase:lrbench.adapters:deepmd?model='{_DPA4}/DPA4-Plus-OMol25-100M-v20260820.pt'"
               "&name='DPA4-Plus-OMol25-100M'", "DPA4-Plus, OMol25 100M (8.8 M params)", OMOL, DEEPMD_ENV,
               "huggingface.co/deepmodelingcommunity/DPA4-OMol", "CC BY-NC 4.0",
               "atoms.info charge_spin = [Q, 2S+1], FiLM-conditioned descriptor",
               "none (local, 6 A cutoff); the total charge only conditions the descriptor",
               aliases=["dpa4", "deepmd", "dpa4-omol"]),
    NamedModel("dpa4-air", f"ase:lrbench.adapters:deepmd?model='{_DPA4}/DPA4-Air-OMol25-100M-v20260820.pt'"
               "&name='DPA4-Air-OMol25-100M'", "DPA4-Air, OMol25 100M (5.1 M params)", OMOL, DEEPMD_ENV,
               "huggingface.co/deepmodelingcommunity/DPA4-OMol", "CC BY-NC 4.0",
               "as dpa4-plus", "as dpa4-plus"),
]})

# Meta FAIR's OMol25 models: the same level as our sets, charge- and spin-aware, and local. They are the control
# that any long-range claim has to beat beyond the cutoff.
MODELS.update({m.name: m for m in [
    NamedModel("uma-s", "ase:lrbench.adapters:fairchem?model='uma-s-1p2p1'&task='omol'", "UMA-S 1.2.1, omol task",
               OMOL, FAIRCHEM_ENV, "huggingface.co/facebook/UMA (gated)",
               "Meta FAIR Chemistry model licence", "atoms.info charge + spin (2S+1), as an embedding",
               "none (local message passing)", aliases=["uma", "uma-small"]),
    NamedModel("esen-omol", "ase:lrbench.adapters:fairchem?model='esen-sm-conserving-all-omol'",
               "eSEN-SM conserving, all of OMol25", OMOL, FAIRCHEM_ENV,
               "fairchem-core download (gated on Hugging Face)", "Meta FAIR Chemistry model licence",
               "atoms.info charge + spin (2S+1), as an embedding", "none (local message passing)",
               aliases=["esen", "esen-sm"]),
]})

# AIMNet2: charge-aware, with an explicit Coulomb term over predicted charges that sum to the total charge.
MODELS.update({m.name: m for m in [
    NamedModel("aimnet2", "ase:lrbench.adapters:aimnet2?lrcoulomb='simple'", "AIMNet2 (Coulomb, no cutoff)",
               "wB97M-D3/def2-TZVPPD", AIMNET_ENV, "fetched by the aimnet package on first use",
               "MIT (code); see the model card for the weights",
               "set_charge / set_mult on the calculator, not atoms.info; charges sum to the total charge",
               "predicted partial charges + Coulomb over all pairs, no cutoff (vacuum)",
               aliases=["aimnet", "aimnet-2"]),
    NamedModel("aimnet2-dsf", "ase:lrbench.adapters:aimnet2?lrcoulomb='dsf'&cutoff=15.0",
               "AIMNet2, DSF Coulomb (15 A cutoff)", "wB97M-D3/def2-TZVPPD", AIMNET_ENV,
               "as aimnet2", "MIT (code); see the model card for the weights", "as aimnet2",
               "damped-shifted-force Coulomb, 15 A cutoff: truncates the tails this benchmark measures, and is "
               "reported as its own row rather than silently substituted"),
]})

# PET-MAD: the PET architecture as a universal *materials* model (MAD, PBEsol/PAW). Same family as the PET models
# this benchmark grew out of, but trained at another level and with no charge input, so it is declared as such.
MODELS.update({m.name: m for m in [
    NamedModel("pet-mad", "ase:lrbench.adapters:pet_mad?version='latest'", "PET-MAD (universal, MAD)",
               "PBEsol/PAW (MAD)", PETMAD_ENV, "huggingface.co/lab-cosmo/pet-mad (fetched on first use)",
               "BSD-3-Clause", "none: no total-charge or spin input",
               "none (local message passing)", aliases=["petmad", "pet"]),
]})

# The upet PET checkpoints (huggingface.co/lab-cosmo/upet, BSD-3-Clause), exported to metatomic TorchScript. These
# matter for two reasons. They are the same architecture as the long-range PET work this benchmark grew out of, and
# the pet-omol-* family is trained on OMol25 -- our own level -- *and* declares a total-charge input, so it is a
# charge-conditioned model scored against charged references with no level caveat. That combination is rare here.
#
# Whether a checkpoint consumes a charge is read off the model itself (AtomisticModel.requested_inputs), not assumed
# from its name: only pet-omol-* requests charge and spin_multiplicity. pet-spice-*, pet-mols-* and pet-omad-*
# request nothing, so they are recorded as unable to be *told* a charge, the same way pet-mad is -- which is the
# honest reading, and keeps "ignored the charge" meaning only what it says.
_UPET = _os.environ.get("LRBENCH_UPET_DIR", _os.path.expanduser("~/.cache/lrbench/upet"))
SPICE_LEVEL = "wB97M-D3(BJ)/def2-TZVPPD"   # SPICE, as in adapters.SPICE; models.py keeps its levels literal
_UPET_SRC = "huggingface.co/lab-cosmo/upet, exported with `mtt export`"
_UPET_LR = "none (local message passing)"
MODELS.update({m.name: m for m in [
    NamedModel("pet-omol-s", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-omol-s.pt'&name='PET-OMol-S'",
               "PET-OMol-S (charge-conditioned)", OMOL, METATOMIC_ENV, _UPET_SRC + " (pet-omol-s-v1.0.0)",
               "BSD-3-Clause", "atoms.info charge + spin_multiplicity (declared by the model)", _UPET_LR,
               aliases=["petomols", "pet-omol"]),
    NamedModel("pet-omol-m", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-omol-m.pt'&name='PET-OMol-M'",
               "PET-OMol-M (charge-conditioned)", OMOL, METATOMIC_ENV, _UPET_SRC + " (pet-omol-m-v1.0.0)",
               "BSD-3-Clause", "as pet-omol-s", _UPET_LR, aliases=["petomolm"]),
    NamedModel("pet-omol-l", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-omol-l.pt'&name='PET-OMol-L'",
               "PET-OMol-L (charge-conditioned)", OMOL, METATOMIC_ENV, _UPET_SRC + " (pet-omol-l-v1.0.0)",
               "BSD-3-Clause", "as pet-omol-s", _UPET_LR, aliases=["petomoll"]),
    NamedModel("pet-spice-s", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-spice-s.pt'&name='PET-SPICE-S'",
               "PET-SPICE-S", SPICE_LEVEL, METATOMIC_ENV, _UPET_SRC + " (pet-spice-s-v0.2.0)", "BSD-3-Clause",
               "none: the model requests no charge input", _UPET_LR, aliases=["petspices"]),
    NamedModel("pet-spice-l", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-spice-l.pt'&name='PET-SPICE-L'",
               "PET-SPICE-L", SPICE_LEVEL, METATOMIC_ENV, _UPET_SRC + " (pet-spice-l-v0.2.0)", "BSD-3-Clause",
               "as pet-spice-s", _UPET_LR, aliases=["petspicel"]),
    NamedModel("pet-mols-s", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-mols-s.pt'&name='PET-MOLS-S'",
               "PET-MOLS-S (mixed molecular)", "mixed", METATOMIC_ENV, _UPET_SRC + " (pet-mols-s-v1.1.0)",
               "BSD-3-Clause", "as pet-spice-s", _UPET_LR, aliases=["petmolss"]),
    NamedModel("pet-omad-s", f"ase:lrbench.adapters:metatomic?path='{_UPET}/pet-omad-s.pt'&name='PET-OMAD-S'",
               "PET-OMAD-S (OMat24 + MAD)", "mixed", METATOMIC_ENV, _UPET_SRC + " (pet-omad-s-v1.0.0)",
               "BSD-3-Clause", "as pet-spice-s", _UPET_LR, aliases=["petomads"]),
]})

# GRACE (graph ACE): a materials model at PBE/PAW with no charge input. It is here for the ACE family and as a
# floor row; its energies are not comparable with ours and its charge-blindness is recorded, not inferred.
MODELS.update({m.name: m for m in [
    NamedModel("grace-2l", "ase:lrbench.adapters:grace?model='GRACE-2L-OMAT'", "GRACE-2L-OMAT (two-layer semi-local)",
               "PBE/PAW (OMat24)", GRACE_ENV, "grace_models download / tensorpotential auto-download (~/.cache/grace)",
               "Academic Software License", "none: no total-charge or spin input",
               "none (local, 6 A cutoff)", aliases=["grace", "ace", "grace-2l-omat"]),
    NamedModel("grace-1l", "ase:lrbench.adapters:grace?model='GRACE-1L-OMAT'", "GRACE-1L-OMAT (one-layer local)",
               "PBE/PAW (OMat24)", GRACE_ENV, "tensorpotential auto-download", "Academic Software License",
               "none: no total-charge or spin input", "none (local, 6 A cutoff)"),
]})

# The four models of the external comparison, in table order.
EXTERNAL = ["so3lr", "so3lr-tauri", "mace-h", "mace-polar"]
# The models of the published long-range comparison (SN2 reaction-coordinate and fragment scans), so that
# `lrbench suite --models lr-comparison` evaluates the same three models on the lrbench sets.
LR_COMPARISON = ["so3lr", "mace-polar", "mace-polar-l"]
GROUPS = {"external": EXTERNAL, "lr-comparison": LR_COMPARISON, "all-external": list(MODELS)}

_ALIAS = {a: m.name for m in MODELS.values() for a in [m.name] + m.aliases}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def lookup(name: str) -> Optional[NamedModel]:
    """NamedModel for a name or alias (``mace-h:<head>`` builds the MACE-MH-1 entry for another head)."""
    n = _norm(name)
    if n.startswith(("mace-h:", "mace-mh:", "mace-mh-1:")):
        from .adapters import LEVELS
        head = name.split(":", 1)[1].strip()
        base = MODELS["mace-h"]
        return NamedModel(f"mace-h:{head}", f"ase:lrbench.adapters:mace_mh?head='{head}'",
                          f"MACE-MH-1, {head} head", LEVELS.get(f"mace_mh:{head}", "unknown"), MACE_ENV,
                          base.checkpoint, base.licence, base.charge_spin, base.long_range)
    return MODELS.get(_ALIAS.get(n, n))


def resolve(spec: str) -> str:
    """Model spec for a name (unchanged if it is not a name: paths, ase:/oracle:/mlipcpp-capi: specs)."""
    if not spec or spec.startswith(("ase:", "oracle:", "mlipcpp-capi:")) or spec.endswith((".gguf", ".pt")):
        return spec
    m = lookup(spec)
    if m:
        return m.spec
    from .gguf_registry import resolve as gguf_resolve  # mlip.cpp GGUFs bound to their build (configs/gguf_models.json)
    return gguf_resolve(spec) or spec


def expand(names: str) -> List[str]:
    """'external' / 'all-external' / comma list -> canonical names; unknown names raise with the list of known ones."""
    out = []
    for n in [x for x in names.split(",") if x.strip()]:
        if _norm(n) in GROUPS:
            out += GROUPS[_norm(n)]
            continue
        m = lookup(n)
        if m is None:
            raise ValueError(f"unknown model name {n!r}; known: {', '.join(MODELS)} (groups: {', '.join(GROUPS)})")
        out.append(m.name)
    return list(dict.fromkeys(out))


def requirements_md(names: Optional[List[str]] = None) -> str:
    names = names or list(MODELS)
    rows = ["| name | model | spec | energy level | env | checkpoint | licence | total Q / spin | long range |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        m = lookup(n)
        rows.append(f"| `{m.name}` | {m.title} | `{m.spec[4:]}` | {m.level} | {m.env} | {m.checkpoint} | "
                    f"{m.licence} | {m.charge_spin} | {m.long_range} |")
    rows += ["", "| env | python | install | notes |", "|---|---|---|---|"]
    for e in dict.fromkeys(lookup(n).env for n in names):
        E = ENVS[e]
        rows.append(f"| {e} | {E['python']} | " + "<br>".join(f"`{c}`" for c in E["install"]) + f" | {E['notes']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    print(requirements_md() if "--md" in sys.argv else "\n".join(f"{m.name}\t{m.spec}" for m in MODELS.values()))
