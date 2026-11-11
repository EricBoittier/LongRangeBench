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
PBE_OMAT = "PBE/PAW (OMat24)"
PBESOL_MAD = "PBEsol/PAW (MAD)"
RPBE_D3 = "RPBE-D3 (water, CACE-LES training set)"
WB97M_D3_TZVPP = "wB97M-D3/def2-TZVPPD"
LEVELS = {"maceles": SPICE, "cace": RPBE_D3, "so3lr": PBE0MBD, "mace_polar": OMOL, "mace_omol": OMOL, "kernel_lr": OMOL,
          # OrbMol (OMol25; v2 adds OPoly26) and the released DPA4 checkpoints (OMol25 100M split)
          "orbmol": OMOL, "deepmd": OMOL, "aimnet2": WB97M_D3_TZVPP, "fairchem": OMOL,
          "grace": PBE_OMAT, "pet_mad": PBESOL_MAD,
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
    if name == "metatomic":
        kw = _kw(q)
        return kw.get("level") or metatomic_level(kw.get("path", ""))
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

    #: the charge/spin inputs that live in ``atoms.info`` and that ASE's own ``check_state`` does not look at
    INFO_KEYS = ("charge", "spin", "spin_multiplicity", "multiplicity", "charge_spin", "external_field")

    @classmethod
    def _info_state(cls, atoms):
        out = []
        for k in cls.INFO_KEYS:
            v = None if atoms is None else atoms.info.get(k)
            try:
                v = tuple(np.ravel(np.asarray(v, float)).tolist()) if hasattr(v, "__len__") else v
            except Exception:
                v = repr(v)
            out.append((k, v))
        return tuple(out)

    def check_state(self, atoms, tol=1e-15):
        """ASE's version compares geometry only, so a structure that differs from the last one *just* in total charge
        would be served the cached result. See :func:`lrbench.backends.drop_ase_cache`."""
        changes = super().check_state(atoms, tol)
        if not changes and self._info_state(atoms) != getattr(self, "_last_info", None):
            changes = ["info"]
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._last_info = self._info_state(atoms if atoms is not None else self.atoms)
        from .backends import drop_ase_cache
        drop_ase_cache(self.inner)  # the wrapped calculator caches on geometry too
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


def _pkg_meta(package: str, name: str, extra: dict) -> dict:
    try:
        from importlib.metadata import version
        v = version(package)
    except Exception:
        v = "?"
    return dict({"model": name, "package": package, "version": v}, **extra)


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


# ------------------------------------------------------------------------------------------- OrbMol (Orbital Materials)
ORBMOL = {"orbmol-v2": "orbmol_v2", "orbmol-v1": "orbmol_v1_conservative", "orbmol-v1-direct": "orbmol_v1_direct"}


def orbmol(model: str = "orbmol-v2", device: str = "cuda", precision: str = "float32-high"):
    """OrbMol (Orbital Materials, orb-models >= 0.7, Apache-2.0), trained on OMol25 (wB97M-V/def2-TZVPD; v2 also on
    OPoly26). ``atoms.info['charge']`` and ``atoms.info['spin']`` (multiplicity) are required inputs.

    v2 carries an explicit long-range term: a latent-charge head whose per-atom charges sum to the total charge, and
    a Coulomb module (bare 1/r in vacuum, PME in a cell). v1 is local -- 6 A cutoff, no electrostatic term -- so the
    pair is a controlled comparison of the same architecture with and without long range. Orbital Materials note
    that the v2 charges are not trained on, so they are scored as given, like other latent charges.
    """
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    fn = ORBMOL.get(model, model)
    orbff, atoms_adapter = getattr(pretrained, fn)(device=device, precision=precision)
    v2 = "v2" in fn
    meta = _pkg_meta("orb-models", f"OrbMol ({model})", {
        "licence": "Apache-2.0", "charge_spin": "atoms.info charge / spin (2S+1)",
        "precision": precision,
        "long_range": ("latent charges constrained to the total charge + Coulomb (bare 1/r in vacuum, PME in a "
                       "cell); charges not trained on" if v2 else "none (local message passing, 6 A cutoff)")})
    return Adapter(ORBCalculator(orbff, atoms_adapter, device=device), OMOL, meta,
                   info_map={"spin_multiplicity": "spin"},
                   charges=lambda R: R.get("charges"), dipole=lambda R: R.get("dipole"))


# ----------------------------------------------------------------------------------------------- DeePMD-kit (DPA4)
def deepmd(model: str, level: str = OMOL, name: str = "", licence: str = "CC BY-NC 4.0"):
    """A DeePMD-kit (>= 3.2) checkpoint through its ASE calculator. Written for the released
    ``DPA4-<size>-OMol25-100M`` models: OMol25 100M split (wB97M-V/def2-TZVPD), CC BY-NC 4.0.

    Total charge and spin multiplicity are passed as ``atoms.info['charge_spin'] = [Q, 2S+1]``, which conditions the
    descriptor through FiLM. That is global conditioning on the charge, **not** an electrostatic term: the model is
    local (6 A cutoff) and returns no per-atom charges, so the charge-parity and dipole blocks are skipped.
    """
    import os
    from deepmd.calculator import DP
    calc = DP(model=model)

    def _charge_spin(inner, a):
        a.info["charge_spin"] = np.array([int(a.info.get("charge", 0) or 0),
                                          int(a.info.get("spin_multiplicity", 1) or 1)])

    meta = _pkg_meta("deepmd-kit", name or f"DeePMD {os.path.basename(model)}", {
        "checkpoint": model, "licence": licence,
        "charge_spin": "atoms.info charge_spin = [Q, 2S+1], FiLM-conditioned descriptor",
        "long_range": "none (local, 6 A cutoff); total charge enters only as descriptor conditioning"})
    return Adapter(calc, level, meta, before=_charge_spin)


# ---------------------------------------------------------------------------------------------- AIMNet2 (Isayev lab)
def aimnet2(model: str = "aimnet2", lrcoulomb: str = "simple", cutoff: float = 15.0, dsf_alpha: float = 0.2,
            level: str = WB97M_D3_TZVPP, device: str = "cuda"):
    """AIMNet2 (``aimnet[ase]``, Isayev lab): charge-aware with an explicit Coulomb term over predicted partial
    charges, which are constrained to sum to the total charge.

    The total charge is **not** read from ``atoms.info``: it is set on the calculator (``set_charge`` / ``set_mult``),
    which this adapter does before every structure. The docs warn that a wrong charge gives wrong answers silently,
    so getting that plumbing right is the point of the adapter.

    ``lrcoulomb='simple'`` (the default here) sums Coulomb over every pair with no cutoff -- the right choice for
    vacuum dimer tails. ``'dsf'`` is the damped-shifted-force form AIMNet2 recommends for periodic systems; with its
    15 A cutoff it truncates exactly the tails this benchmark measures, so it is reported as a separate model rather
    than silently substituted.
    """
    from aimnet.calculators import AIMNet2ASE
    calc = AIMNet2ASE(model, charge=0, mult=1)
    if lrcoulomb and lrcoulomb != "simple":
        calc.set_lrcoulomb_method(lrcoulomb, cutoff=cutoff, dsf_alpha=dsf_alpha)
    if device and hasattr(calc, "to"):
        try:
            calc.to(device)
        except Exception:
            pass

    def _set_charge(inner, a):
        inner.set_charge(int(a.info.get("charge", 0) or 0))
        if hasattr(inner, "set_mult"):
            inner.set_mult(int(a.info.get("spin_multiplicity", 1) or 1))

    meta = _pkg_meta("aimnet", f"AIMNet2 ({model})", {
        "licence": "MIT (code); see the model card for the weights",
        "charge_spin": "set_charge / set_mult on the calculator (not atoms.info); charges sum to the total charge",
        "long_range": ("predicted partial charges + Coulomb over all pairs, no cutoff (vacuum)" if lrcoulomb ==
                       "simple" else f"predicted partial charges + {lrcoulomb.upper()} Coulomb, cutoff {cutoff} A")})
    return Adapter(calc, level, meta, before=_set_charge,
                   charges=lambda R: R.get("charges"), dipole=lambda R: R.get("dipole"))


# ------------------------------------------------------------------------------ fairchem (UMA, eSEN) -- Meta FAIR
def fairchem(model: str = "uma-s-1p2p1", task: str = "", device: str = "cuda", level: str = OMOL,
             inference_settings: str = "default"):
    """A fairchem-core v2 pretrained model (UMA, eSEN, AllSCAIP) through ``FAIRChemCalculator``.

    The OMol25 heads are trained at wB97M-V/def2-TZVPD -- the same level as our sets -- and take a total charge and
    spin multiplicity from ``atoms.info`` (``charge``, ``spin``). They carry **no** explicit electrostatic term: the
    charge enters as an embedding, so these are the strongest local models at the right level, which is exactly what
    makes them the control the long-range models have to beat beyond the cutoff.

    ``task`` is the UMA task name (``omol`` for molecules) and is required for the multi-task UMA checkpoints;
    single-task checkpoints such as ``esen-sm-conserving-all-omol`` take none, and it defaults to ``omol`` for a
    model whose name starts with ``uma``. Weights are gated on Hugging Face: accept the licence and log in first.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    unit = pretrained_mlip.get_predict_unit(model, device=device, inference_settings=inference_settings)
    task = task or ("omol" if model.startswith("uma") else "")
    calc = FAIRChemCalculator(unit, task_name=task or None)
    meta = _pkg_meta("fairchem-core", f"fairchem {model}" + (f" ({task} task)" if task else ""), {
        "licence": "Meta FAIR Chemistry model licence (gated on Hugging Face)",
        "charge_spin": "atoms.info charge / spin (2S+1), as an embedding",
        "long_range": "none (local message passing); the total charge enters only as an embedding"})
    return Adapter(calc, level, meta, info_map={"spin_multiplicity": "spin"},
                   charges=lambda R: R.get("charges"), dipole=lambda R: R.get("dipole"))


# ------------------------------------------------------------------------------------------- GRACE (ACE family, ICAMS)
def grace(model: str = "GRACE-2L-OMAT", level: str = PBE_OMAT, **kw):
    """A pretrained GRACE foundation model (graph ACE, ``tensorpotential``; Academic Software License).

    These are **materials** models: trained on OMat24/Alexandria/MPtrj at PBE(+U)/PAW, periodic, and with no total
    charge input at all. On the molecular ion-pair sets here they are therefore doubly out of domain -- wrong level
    *and* charge-blind -- which is why ``charge_spin=False`` is set, so every result records that the model could not
    be told the charge rather than appearing to have ignored it. Scored for the ACE family's sake and as a floor:
    read the dimensionless diagnostics, never the energies.
    """
    from tensorpotential.calculator import grace_fm
    calc = grace_fm(model, **kw)
    meta = _pkg_meta("tensorpotential", f"GRACE ({model})", {
        "licence": "Academic Software License",
        "charge_spin": "NONE: no total-charge or spin input (recorded, not silently ignored)",
        "long_range": "none (local, 6 A cutoff; semi-local message passing)"})
    return Adapter(calc, level, meta, charge_spin=False)


# ------------------------------------------------------------------------------------------ PET-MAD (lab-cosmo)
def pet_mad(version: str = "latest", device: str = "cuda", level: str = PBESOL_MAD, **kw):
    """PET-MAD (``pet-mad``, lab-cosmo; BSD-3-Clause), the universal PET trained on the MAD dataset.

    This is the same architecture as the PET models the long-range work is built on, which is why it belongs in
    the table — but it is a *materials* model: MAD is PBEsol/PAW in VASP, computed without spin polarisation and
    without dispersion, chosen for internal consistency rather than per-structure accuracy. It takes no total
    charge, so ``charge_spin=False`` records that the model could not be *told* the charge rather than leaving it
    to look as though it ignored one, and the level is declared so an eval against our wB97M-V/def2-TZVPD
    references is refused unless the mismatch is allowed explicitly.

    The charge-conditioned PET models (PET-OMol-S and its long-range fine-tunes) are a different matter: they are
    distributed as mlip.cpp GGUF, and running them needs a charge the published bindings cannot pass.
    """
    from pet_mad.calculator import PETMADCalculator
    calc = PETMADCalculator(version=version, device=device, **kw)
    meta = _pkg_meta("pet-mad", f"PET-MAD ({version})", {
        "licence": "BSD-3-Clause",
        "charge_spin": "NONE: no total-charge or spin input (recorded, not silently ignored)",
        "long_range": "none (local message passing)"})
    return Adapter(calc, level, meta, charge_spin=False)


# --------------------------------------------------------------------- metatomic (any exported .pt, incl. upet PET)
#: The training level of each ``upet`` checkpoint family, keyed by the dataset token in its name. The names on
#: huggingface.co/lab-cosmo/upet are ``pet-<dataset>-<size>-v<version>``, and the dataset fixes the labels:
#:
#:   omol    OMol25            wB97M-V/def2-TZVPD      -- the level of this benchmark's own references
#:   spice   SPICE             wB97M-D3(BJ)/def2-TZVPPD
#:   mols    mixed molecular   mixed (not a single level)
#:   mad     MAD               PBEsol/PAW, no spin polarisation, no dispersion
#:   omat    OMat24            PBE/PAW
#:   omad    OMat24 + MAD      mixed PBE/PAW
#:   oam     OMat24+Alexandria+MPtrj   mixed PBE/PAW
#:
#: Only the ``omol`` family shares our level, so only it is scored without ``--allow-level-mismatch``.
METATOMIC_LEVELS = {"omol": OMOL, "spice": SPICE, "mad": PBESOL_MAD, "omat": PBE_OMAT,
                    "omad": "mixed", "oam": "mixed", "omatpes": "mixed", "mols": "mixed"}


def metatomic_level(path: str) -> str:
    """Guess the training level from a ``upet``-style file name, or ``"unknown"``.

    Longest dataset token wins, so ``pet-omatpes-l`` is not read as ``omat``.
    """
    stem = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".pt") else stem
    best, best_len = "unknown", 0
    for token, level in METATOMIC_LEVELS.items():
        if token in stem and len(token) > best_len:
            best, best_len = level, len(token)
    return best


def metatomic(path: str, device: str = "cuda", level: str = "", name: str = "", licence: str = "BSD-3-Clause",
              long_range: str = "none (local message passing)"):
    """Any model exported to metatomic TorchScript (``mtt export`` / ``metatomic``), e.g. the ``upet`` PET family.

    This is the general door into the benchmark: ``metatomic-ase`` reads the total charge from ``atoms.info
    ["charge"]`` and the multiplicity from ``atoms.info["spin_multiplicity"]``, so a model that *declares* those
    inputs is told them, structure by structure, through the same path as every other adapter here.

    Whether it declares them is not guessed. ``AtomisticModel.requested_inputs()`` is the model's own statement of
    what it consumes, and it decides ``charge_spin``: a model that requests ``charge`` is given one and is scored on
    what it does with it, while a model that requests nothing is recorded as *unable to be told* -- the pet-mad
    distinction -- rather than left looking as though it ignored a charge it was handed. Of the released PET
    checkpoints only the ``pet-omol-*`` family requests ``charge`` and ``spin_multiplicity``; ``pet-spice-*``,
    ``pet-mols-*`` and ``pet-omad-*`` request nothing.

    ``level`` defaults to :func:`metatomic_level` on the file name; pass it explicitly for a checkpoint whose name
    does not carry its dataset, because an unknown level means energies are scored but not level-checked.
    """
    from .backends import metatomic_calculator
    calc = metatomic_calculator(path, device)
    level = level or metatomic_level(path)
    inputs = ()
    try:
        model = getattr(calc, "_model", None)
        if model is not None:
            try:
                inputs = tuple(sorted(model.requested_inputs(use_new_names=True).keys()))
            except TypeError:  # older metatomic: no use_new_names
                inputs = tuple(sorted(model.requested_inputs().keys()))
    except Exception:  # a model that cannot be asked is treated as not conditioned, and says so below
        inputs = ()
    conditioned = "charge" in inputs
    stem = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    meta = _pkg_meta("metatomic", name or stem, {
        "licence": licence,
        "checkpoint": stem,
        "requested_inputs": list(inputs),
        "charge_spin": ("atoms.info charge / spin_multiplicity (declared by the model)" if conditioned else
                        "NONE: the model requests no charge input (recorded, not silently ignored)"),
        "long_range": long_range})
    return Adapter(calc, level, meta, charge_spin=conditioned)
