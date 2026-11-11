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

# What each environment needs. One venv per environment: JAX (SO3LR) and torch (MACE) are not installed together.
ENVS: Dict[str, dict] = {
    SO3LR_ENV: {
        "probe": "so3lr",  # module whose import tells whether the environment can run these models
        "python": ">=3.12,<3.14",
        "install": ['uv pip install "so3lr[cuda12] @ git+https://github.com/general-molecular-simulations/so3lr@v0.2.0"',
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
    return m.spec if m else spec


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
