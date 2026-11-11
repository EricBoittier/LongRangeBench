"""Adapters for external long-range-capable models, run through the generic ASE backend:

    lrbench eval --model "ase:lrbench.adapters:so3lr?model='so3lr-tauri-m-2606'" ...
    lrbench eval --model "ase:lrbench.adapters:mace_polar?model='polar-1-m'" ...
    lrbench eval --model "ase:lrbench.adapters:mace_mh?head='omol'" ...
    lrbench eval --model "ase:lrbench.adapters:mace_omol" ...

Each factory returns an :class:`Adapter`, an ASE calculator that
  - maps lrbench's ``atoms.info`` (``charge``, ``spin_multiplicity``) onto the keys the wrapped calculator reads;
  - returns per-atom ``charges`` and the total ``dipole`` (e Å, about the origin) when the model has them;
  - carries ``lrbench_level`` (training level of the energy head that is read), ``lrbench_meta`` (name, version,
    licence, how charge/spin enter, the long-range term) and ``charge_spin`` (False if total charge/spin are ignored).

``default_level(spec)`` gives the level for ``--model-level`` inference without building the model, so
``lrbench eval`` level-checks these specs like GGUFs instead of treating them as ``unknown``.

Gas phase: SO3LR's electrostatics and dispersion are evaluated with no long-range cutoff (``lr_cutoff=None`` and
``dispersion_energy_cutoff_lr_damping=None``: every pair, no switching), SO3LR's own infinite-cutoff mode. Its MD
default (12 Å, switched from 0.45 x cutoff, or PME in a box) would truncate the dimer tails. MACE-POLAR-1 uses
``pbc_handling='auto'``, i.e. real-space Coulomb for non-periodic structures, with a zero external field.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

OMOL = "wB97M-V/def2-TZVPD"
PBE0MBD = "PBE0+MBD-NL"
PBE0MBD_V1 = "PBE0+MBD"  # the original SO3LR (registry name 'so3lr') was trained on PBE0+MBD, not MBD-NL

SPICE = "wB97M-D3(BJ)/def2-TZVPPD"
RPBE_D3 = "RPBE-D3 (water, CACE-LES training set)"
LEVELS = {"maceles": SPICE, "cace": RPBE_D3, "so3lr": PBE0MBD, "mace_polar": OMOL, "mace_omol": OMOL, "kernel_lr": OMOL,
          # MACE-MH-1 heads (mace-foundations model card / arXiv 2510.25380). The omol head is listed there as
          # wB97M-VV10, i.e. OMol25's wB97M-V/def2-TZVPD labels (1 % subset).
          "mace_mh:omol": OMOL, "mace_mh:omat_pbe": "PBE/PAW (OMat24)", "mace_mh:spice_wB97M": "wB97M-D3(BJ)/def2-TZVPPD",
          "mace_mh:rgd1_b3lyp": "B3LYP-D3/6-31G*", "mace_mh:oc20_usemppbe": "RPBE/PAW (OC20)",
          "mace_mh:matpes_r2scan": "r2SCAN/PAW (MatPES)"}


def _kw(q: str) -> dict:
    out = {}
    for p in q.split("&"):
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v.strip("'\"")
    return out


def default_level(spec: str) -> str:
    """Training level implied by an ``[ase:]lrbench.adapters:<factory>?kw=v`` spec (no model is built)."""
    s = spec[4:] if spec.startswith("ase:") else spec
    path, _, q = s.partition("?")
    mod, _, name = path.rpartition(":")
    if mod != "lrbench.adapters":
        return "unknown"
    if name == "mace_mh":
        return LEVELS.get(f"mace_mh:{_kw(q).get('head', 'omol')}", "unknown")
    if name == "so3lr":
        return so3lr_level(_kw(q).get("model", "so3lr-tauri-m-2606"))
    return LEVELS.get(name, "unknown")


class Adapter(Calculator):
    """Wraps an ASE calculator: info-key mapping, charge/dipole extraction, metadata."""
    implemented_properties = ["energy", "forces", "charges", "dipole"]

    def __init__(self, inner, level: str, meta: dict, info_map=None, charge_spin: bool = True,
                 charges=None, dipole=None, extra_info=None, before=None):
        super().__init__()
        self.inner, self.lrbench_level, self.lrbench_meta = inner, level, dict(meta)
        self.info_map = dict(info_map or {})
        self.extra_info = dict(extra_info or {})
        self.charge_spin = charge_spin
        self._charges, self._dipole = charges, dipole  # callables(results dict) -> array | None
        self._before = before  # callable(inner, atoms) run before each call (e.g. cache resets)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        a.info = {k: v for k, v in atoms.info.items()}
        for src, dst in self.info_map.items():
            if src in a.info:
                a.info[dst] = a.info[src]
        for k, v in self.extra_info.items():
            a.info.setdefault(k, v)
        if not self.charge_spin:  # the model cannot see them: make that explicit rather than silently passing keys
            for k in ("charge", "spin", "spin_multiplicity", "multiplicity"):
                a.info.pop(k, None)
        if self._before:
            self._before(self.inner, a)
        a.calc = self.inner
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), float)
        R = dict(getattr(self.inner, "results", {}) or {})
        self.results = {"energy": e, "free_energy": e, "forces": f}
        q = self._charges(R) if self._charges else None
        if q is not None:
            q = np.asarray(q, float).reshape(len(a), -1)[:, 0]
            self.results["charges"] = q
        mu = self._dipole(R) if self._dipole else None
        if mu is None and q is not None:
            mu = (q[:, None] * a.positions).sum(0)
        if mu is not None:
            self.results["dipole"] = np.asarray(mu, float).reshape(-1)[:3]


def _first(R, *keys):
    for k in keys:
        if R.get(k) is not None:
            return R[k]
    return None


def so3lr_level(model: str) -> str:
    return PBE0MBD_V1 if model in ("so3lr", "so3lr-v1") else PBE0MBD


def so3lr(model: str = "so3lr-tauri-m-2606", lr_cutoff=None, charges: bool = True, dtype: str = "float64"):
    """SO3LR / SO3LR Tauri (github.com/general-molecular-simulations/so3lr; arXiv 2611.06237). The bundled Tauri
    checkpoints read their PBE0+MBD(-NL) energy head (``head_index`` from the model registry). Total charge from
    ``info['charge']``, spin from ``info['multiplicity']`` (2S+1)."""
    import so3lr as _s
    from so3lr import So3lrCalculator
    if dtype == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    kw = dict(model=model, lr_cutoff=lr_cutoff, dtype=getattr(np, dtype))
    if lr_cutoff is None:
        kw["dispersion_energy_cutoff_lr_damping"] = None  # required by SO3LR for the infinite cutoff
    if charges:  # intermediate observables come back in results["aux"] only with has_aux
        kw.update(observables=["partial_charges"], has_aux=True)
    calc = So3lrCalculator(**kw)
    meta = {"model": model, "package": "so3lr", "version": getattr(_s, "__version__", "?"),
            "licence": "code MIT; Tauri weights CC BY-NC 4.0", "head": f"{so3lr_level(model)} energy head",
            "charge_spin": "atoms.info charge / multiplicity (2S+1)", "lr_cutoff": lr_cutoff,
            "long_range": "erf-damped Coulomb on charge-equilibrated learned charges + learned-Hirshfeld "
                          "dispersion; " + ("no cutoff (vacuum)" if lr_cutoff is None else f"{lr_cutoff} A switched")}
    def resize(inner, atoms):  # the cached glp neighbour list assumes a fixed atom count
        if getattr(inner, "_lrbench_n", None) != len(atoms):
            inner.neighbors, inner.spatial_partitioning, inner._lrbench_n = None, None, len(atoms)

    return Adapter(calc, so3lr_level(model), meta, info_map={"spin_multiplicity": "multiplicity"}, before=resize,
                   charges=lambda R: (R.get("aux") or {}).get("partial_charges"))


def _mace_meta(name, extra):
    try:
        from importlib.metadata import version
        v = version("mace-torch")
    except Exception:
        v = "?"
    return dict({"model": name, "package": "mace-torch", "version": v}, **extra)


def _has_charge_spin_embedding(calc) -> bool:
    m = calc.models[0]
    spec = getattr(m, "embedding_specs", None) or {}
    keys = set(spec) if isinstance(spec, dict) else set()
    return bool(keys & {"total_charge", "charge", "total_spin", "spin"}) or hasattr(m, "joint_embedding")


def mace_polar(model: str = "polar-1-m", dtype: str = "float64", device: str = "cuda"):
    """MACE-POLAR-1 (mace-torch >= 0.3.16 with graph_longrange; arXiv 2602.19411): trained on OMol25 (wB97M-V/
    def2-TZVPD); learnable charge and spin densities, non-self-consistent polarisation, global charge equilibration
    to the total charge Q; real-space Coulomb in vacuum. ``info['charge']``, ``info['spin']`` (2S+1),
    ``info['external_field']`` set to zero."""
    from mace.calculators import mace_polar as mp
    calc = mp(model=model, device=device, default_dtype=dtype, pbc_handling="auto")
    meta = _mace_meta(f"MACE-POLAR-1 ({model})", {
        "licence": "ASL (Academic Software License)", "charge_spin": "atoms.info charge / spin (2S+1)",
        "long_range": "Fukui-equilibrated spin-resolved charge densities + non-self-consistent polarisation, "
                      "GTO-smeared Coulomb, real space in vacuum"})
    return Adapter(calc, OMOL, meta, info_map={"spin_multiplicity": "spin"},
                   extra_info={"external_field": [0.0, 0.0, 0.0]},
                   charges=lambda R: R["charges"] if R.get("charges") is not None else (
                       None if R.get("density_coefficients") is None else np.asarray(R["density_coefficients"])[:, 0]),
                   dipole=lambda R: R.get("dipole"))


def mace_omol(model: str = "extra_large", dtype: str = "float64", device: str = "cuda"):
    """MACE-OMOL-0 (mace-torch >= 0.3.14): full OMol25, local (no explicit long range), total charge/spin
    embeddings."""
    from mace.calculators import mace_omol as mo
    calc = mo(model=model, device=device, default_dtype=dtype)
    meta = _mace_meta(f"MACE-OMOL-0 ({model})", {"licence": "ASL (Academic Software License)",
                                                  "charge_spin": "atoms.info charge / spin (2S+1) embeddings",
                                                  "long_range": "none (local message passing)"})
    return Adapter(calc, OMOL, meta, info_map={"spin_multiplicity": "spin"})


def mace_mh(model: str = "mh-1", head: str = "omol", dtype: str = "float64", device: str = "cuda"):
    """MACE-MH-1 multi-head foundation model (mace-torch >= 0.3.15; arXiv 2510.25380), one head per level of theory.
    Whether total charge/spin enter is read from the loaded model and recorded (``charge_spin``)."""
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, device=device, default_dtype=dtype, head=head)
    cs = _has_charge_spin_embedding(calc)
    meta = _mace_meta(f"MACE-MH-1 ({head} head)", {
        "licence": "ASL (Academic Software License)", "head": head,
        "charge_spin": "atoms.info charge / spin" if cs else "IGNORED: no total-charge/spin input in the model",
        "long_range": "none (local message passing)"})
    return Adapter(calc, LEVELS.get(f"mace_mh:{head}", "unknown"), meta, info_map={"spin_multiplicity": "spin"},
                   charge_spin=cs)


class _NeutralOnly(Calculator):
    """Charge-blind models: refuse charged (or open-shell) structures instead of silently treating them as neutral,
    so lrbench reports those groups as not scored (N/A) rather than scoring a model on inputs it cannot represent;
    also refuses elements outside the model's training set."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, inner, elements=None):
        super().__init__()
        self.inner, self.elements = inner, None if elements is None else set(elements)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q, m = int(atoms.info.get("charge", 0)), int(atoms.info.get("spin_multiplicity", atoms.info.get("spin", 1)))
        if q != 0 or m != 1:
            raise ValueError(f"N/A: charge-blind model, structure has charge {q} / multiplicity {m}")
        if self.elements is not None and not set(atoms.numbers) <= self.elements:
            raise ValueError(f"N/A: elements {sorted(set(atoms.numbers) - self.elements)} not in the training set")
        a = atoms.copy(); a.info = {}
        a.calc = self.inner
        self.results = {"energy": float(np.ravel(a.get_potential_energy())[0]),  # CACE returns a (1,) array
                        "forces": np.asarray(a.get_forces(), float).reshape(len(a), 3)}
        self.results["free_energy"] = self.results["energy"]


def _tag(c, level, meta):
    """Not wrapped in Adapter: Adapter drops charge/spin from atoms.info for charge-blind models, and _NeutralOnly
    must see them to refuse charged structures."""
    c.lrbench_level, c.lrbench_meta, c.charge_spin = level, meta, False
    return c


def maceles(model: str, dtype: str = "float64", device: str = "cuda", name: str = "MACELES-OFF"):
    """MACE + Latent Ewald Summation from the Cheng group (MACELES-OFF: SPICE, wB97M-D3(BJ)/def2-TZVPPD; github.com/
    ChengUCB/les_fit, CC BY-NC 4.0; needs mace-torch main + the ``les`` package). No total-charge input: charged
    structures are refused (N/A), latent charges are not constrained to a total charge."""
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=model, device=device, default_dtype=dtype)
    Z = [int(z) for z in calc.models[0].atomic_numbers]
    meta = _mace_meta(f"{name} (Cheng group, MACE-LES)", {
        "checkpoint": model, "licence": "CC BY-NC 4.0 (les_fit)", "elements": Z,
        "charge_spin": "NONE: charge-blind; charged / open-shell structures refused (N/A)",
        "long_range": "Latent Ewald Summation (learned latent charges, Gaussian-smeared Coulomb), vacuum"})
    return _tag(_NeutralOnly(calc, Z), SPICE, meta)


def cace(model: str, device: str = "cuda", name: str = "CACE-LES water", level: str = RPBE_D3,
         elements=(1, 8)):
    """CACE (+LES) from the Cheng group (github.com/BingqingCheng/cace; checkpoints github.com/ChengUCB/les_fit,
    CC BY-NC 4.0). The released water models know H and O only and take no total charge: anything else is N/A."""
    import torch
    import builtins
    import cace.calculators.cace_calculator as _cc
    from cace.calculators import CACECalculator
    # cace 0.1.0 does float(<(1,) array>), a TypeError under numpy >= 2: give that module a size-1-tolerant float
    _cc.float = lambda x: builtins.float(np.ravel(np.asarray(x))[0]) if np.size(x) == 1 else builtins.float(x)
    torch.set_default_dtype(torch.float32)
    net = torch.load(model, map_location=device, weights_only=False)  # pickled nn.Module (torch >= 2.6 default fails)
    calc = CACECalculator(model_path=net, device=device, energy_key="CACE_energy", forces_key="CACE_forces")
    meta = {"model": f"{name} (Cheng group)", "package": "cace", "checkpoint": model, "licence": "CC BY-NC 4.0",
            "elements": list(elements), "charge_spin": "NONE: charge-blind; charged structures refused (N/A)",
            "long_range": "Latent Ewald Summation" if "les" in name.lower() or "lr" in name.lower() else "none"}
    return _tag(_NeutralOnly(calc, elements), level, meta)


class _KernelCalc(Calculator):
    """Energy-only ASE wrapper of a pickled kernel-lr model (kernel-lr/scripts/kmodel.py). Forces are not computed
    (NaN: only energies of these arms are scored) and the frozen-charge model's charges are returned."""
    implemented_properties = ["energy", "forces", "charges", "dipole"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        r = self.model.predict(a.numbers, a.positions, float(a.info.get("charge", 0)), float(a.info.get("spin_multiplicity", 1)))
        self.results = {"energy": r["E"], "free_energy": r["E"], "forces": np.full((len(a), 3), np.nan)}
        if r.get("charges") is not None:
            self.results["charges"] = q = np.asarray(r["charges"], float)
            mu = (q[:, None] * a.positions).sum(0)
            cm = self.model.cm
            if getattr(cm, "sat", False):  # octahedral arms: add the atomic dipoles carried by the satellites
                f = self.model.feat.many([(a.numbers, a.positions)])[0]
                mu = mu + cm.predict(a.numbers, a.positions, float(a.info.get("charge", 0)), f["kq"], f["ph"])[1].sum(0)
            self.results["dipole"] = mu


def kernel_lr(path: str, scripts: str = ""):
    """Kernel long-range baseline (SOAP-KRR, SOAP+LODE, KQEq, octahedral distributed charges) from a pickled
    kernel-lr model; ``scripts`` is the kernel-lr/scripts directory holding klr.py / kmodel.py (default: next to the
    pickle's ../../scripts). Trained on OMol25 (wB97M-V/def2-TZVPD) subset energies; inputs Z, R, Q, spin only."""
    import os
    import pickle
    import sys
    sys.path.insert(0, scripts or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))), "scripts"))
    model = pickle.load(open(path, "rb"))
    meta = {"model": model.label, "package": "kernel-lr", "charge_spin": "total charge Q and spin multiplicity",
            "long_range": "none" if model.cm is None else f"frozen charges, {model.lr}", "forces": "not computed"}
    return Adapter(_KernelCalc(model), OMOL, meta, charges=lambda R: R.get("charges"), dipole=lambda R: R.get("dipole"))
