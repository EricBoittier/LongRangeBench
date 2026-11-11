"""Model backends. Every backend maps a list of :class:`~lrbench.data.Structure` to a list of predictions
``{"energy": float, "forces": (N, 3) array | None, "charges": (N,) array | None, "error": str | None}``.

Model specs (``--model``):
  - ``path.gguf``                       mlip.cpp: ``mlipcpp-eval`` batch (default; the only path that passes formal
                                         charges), else the C API through ctypes (``MLIPCPP_C_LIB``; charge/spin, no
                                         formal charges), else the Python bindings (no charge/spin)
  - ``mlipcpp-capi:path.gguf``          force the C API path (the fast ragged path with ``MLIPCPP_RAGGED=1``)
  - ``path.pt``                         metatomic TorchScript via metatomic_ase.MetatomicCalculator
  - ``ase:module.path:Factory?kw=v``    any ASE calculator; kwargs are parsed as Python literals when possible
"""
from __future__ import annotations

import ast
import ctypes
import importlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import warnings
from typing import List, Optional

import numpy as np

from .data import Structure

FRAG_KEY = "pet.plato_lr.frag_charges"


def _clean_env(env: dict, tf32: bool) -> dict:
    env = dict(env)
    if env.pop("NVIDIA_TF32_OVERRIDE", None) is not None:  # defeats mlip.cpp's FP32 geometry pin (25x NVE drift)
        warnings.warn("NVIDIA_TF32_OVERRIDE removed from the backend environment (use MLIPCPP_TF32=1 instead)")
    if tf32:
        env["MLIPCPP_TF32"] = "1"
        env.pop("GGML_CUDA_NO_TF32", None)
    else:
        env.pop("MLIPCPP_TF32", None)
        env["GGML_CUDA_NO_TF32"] = "1"  # as the program's eval jobs (lrd.sbatch, dim_all.sbatch)
    return env


def gguf_metadata(path: str) -> dict:
    """Scalar GGUF metadata (key -> python value); {} if the gguf package is missing."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return {}
    r = GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            v = f.contents()
        except Exception:
            try:
                v = f.parts[f.data[0]].tolist() if f.data else None
            except Exception:
                v = None
        if isinstance(v, list) and len(v) > 64:
            v = f"<array {len(v)}>"
        out[k] = v
    return out


def has_frag_charges(path: str) -> bool:
    if not path.endswith(".gguf"):
        return False
    v = gguf_metadata(path).get(FRAG_KEY)
    return bool(v) and v not in (0, "0", False)


class Backend:
    name = "base"
    passes_formal_charges = False
    passes_charge_spin = True

    def predict(self, structs: List[Structure]) -> List[dict]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "backend_can_pass_formal_charges": self.passes_formal_charges,
                "charge_spin": self.passes_charge_spin}


def write_extxyz(structs: List[Structure], path: str, formal: bool):
    """Plain extxyz that both ASE and mlip.cpp's io::read_xyz parse (charge=, spin_multiplicity=, formal_charges)."""
    with open(path, "w") as f:
        for s in structs:
            fq = formal and s.formal_charges is not None
            props = "species:S:1:pos:R:3" + (":formal_charges:R:1" if fq else "")
            head = []
            if s.cell is not None and any(s.pbc):
                head.append('Lattice="' + " ".join(f"{x:.10f}" for x in np.asarray(s.cell).reshape(9)) + '"')
                head.append('pbc="' + " ".join("T" if p else "F" for p in s.pbc) + '"')
            else:
                head.append('pbc="F F F"')
            head += [f"Properties={props}", f"charge={int(s.charge)}", f"spin_multiplicity={int(s.spin)}"]
            f.write(f"{len(s.species)}\n{' '.join(head)}\n")
            for i, (e, x) in enumerate(zip(s.species, np.asarray(s.positions))):
                f.write(f"{e} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}" + (f" {s.formal_charges[i]:.1f}" if fq else "") + "\n")


class MlipcppEval(Backend):
    """Batch evaluation through the mlipcpp-eval binary (one process per call, all frames in one extxyz)."""
    name = "mlipcpp-eval"
    passes_formal_charges = True

    def __init__(self, model: str, binary: Optional[str] = None, device: str = "cpu", launcher: str = "",
                 formal: bool = False, tf32: bool = False, workdir: Optional[str] = None, chunk: int = 0):
        self.model, self.device, self.formal, self.tf32, self.chunk = model, device, formal, tf32, chunk
        self.binary = binary or os.environ.get("LRBENCH_MLIPCPP_EVAL") or shutil.which("mlipcpp-eval")
        if not self.binary:
            raise FileNotFoundError("mlipcpp-eval not found (--mlipcpp-bin, $LRBENCH_MLIPCPP_EVAL or PATH)")
        self.launcher = shlex.split(launcher or os.environ.get("LRBENCH_LAUNCHER", ""))
        self.workdir = workdir
        self.last_header: dict = {}

    def _run(self, structs):
        d = tempfile.mkdtemp(prefix="lrbench-", dir=self.workdir)
        xyz, out = os.path.join(d, "in.xyz"), os.path.join(d, "out.json")
        write_extxyz(structs, xyz, self.formal)
        cmd = self.launcher + [self.binary, self.model, xyz, out, "--backend", self.device]
        p = subprocess.run(cmd, env=_clean_env(os.environ, self.tf32), capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"mlipcpp-eval failed ({p.returncode}): {p.stderr[-2000:]}")
        J = json.load(open(out))
        shutil.rmtree(d, ignore_errors=True)
        self.last_header = {k: v for k, v in J.items() if k != "frames"}
        if len(J["frames"]) != len(structs):
            raise RuntimeError(f"mlipcpp-eval returned {len(J['frames'])} frames for {len(structs)}")
        res = []
        for s, f in zip(structs, J["frames"]):
            if "error" in f:
                res.append({"energy": None, "forces": None, "charges": None, "error": f["error"]})
                continue
            lq = f.get("latent_charges")
            res.append({"energy": float(f["energy"]), "forces": np.asarray(f["forces"], float).reshape(-1, 3),
                        "charges": None if lq is None else np.asarray(lq, float).reshape(len(s.species), -1)[:, 0],
                        "error": None})
        return res

    def predict(self, structs):
        if not structs:
            return []
        n = self.chunk or len(structs)
        return [r for i in range(0, len(structs), n) for r in self._run(structs[i:i + n])]

    def info(self):
        return dict(super().info(), binary=self.binary, device=self.device, launcher=" ".join(self.launcher),
                    precision="TF32 (MLIPCPP_TF32=1)" if self.tf32 else "fp32 (GGML_CUDA_NO_TF32=1)",
                    formal_charges_written=self.formal, **{"model_header": self.last_header})


class _ModelOpt(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("precision", ctypes.c_int), ("cutoff", ctypes.c_float)]


class _PredOpt(ctypes.Structure):  # mlipcpp_predict_options_t with the conditioning fields (feat/pet-plato-lr)
    _fields_ = [("forces", ctypes.c_bool), ("stress", ctypes.c_bool), ("nc", ctypes.c_bool),
                ("charge", ctypes.c_int32), ("spin", ctypes.c_int32)]


class MlipcppCAPI(Backend):
    """mlip.cpp C API through ctypes (port of hca2-ipi/scripts/mlipc.py). Charge/spin and cells; no formal charges.
    The ragged fast path is selected by MLIPCPP_RAGGED=1 in the environment, TF32 by MLIPCPP_TF32=1."""
    name = "mlipcpp-capi"

    def __init__(self, model: str, lib: Optional[str] = None, device: str = "cuda", tf32: bool = False):
        lib = lib or os.environ.get("MLIPCPP_C_LIB")
        if not lib:
            raise FileNotFoundError("C API library not given (--mlipcpp-lib or $MLIPCPP_C_LIB)")
        os.environ.update({k: v for k, v in _clean_env(os.environ, tf32).items() if k.startswith(("MLIPCPP", "GGML"))})
        for k in ("NVIDIA_TF32_OVERRIDE",) + (() if tf32 else ("MLIPCPP_TF32",)):
            os.environ.pop(k, None)
        self.lib_path, self.model, self.device, self.tf32 = lib, model, device, tf32
        L = self.lib = ctypes.CDLL(lib)
        L.mlipcpp_model_create.restype = ctypes.c_void_p
        L.mlipcpp_model_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.mlipcpp_predict_ptr_with_options.argtypes = [ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_void_p] * 6
        L.mlipcpp_result_get_energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        L.mlipcpp_result_get_forces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        L.mlipcpp_get_last_error.restype = ctypes.c_char_p
        L.mlipcpp_model_get_conditioning.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
        b = {"cpu": 1, "cuda": 2}[device]
        L.mlipcpp_set_backend(b)
        o = _ModelOpt()
        L.mlipcpp_model_options_default(ctypes.byref(o))
        o.backend = b
        self.m = L.mlipcpp_model_create(ctypes.byref(o))
        if L.mlipcpp_model_load(self.m, str(model).encode()) != 0:
            raise RuntimeError(L.mlipcpp_get_last_error().decode())
        c, qm, sm = ctypes.c_bool(), ctypes.c_int32(), ctypes.c_int32()
        L.mlipcpp_model_get_conditioning(self.m, ctypes.byref(c), ctypes.byref(qm), ctypes.byref(sm))
        self.conditioned = c.value

    def call(self, numbers, positions, cell=None, charge=0, spin=1):
        n = len(numbers)
        z = np.ascontiguousarray(numbers, dtype=np.int32)
        x = np.ascontiguousarray(positions, dtype=np.float32).reshape(n, 3)
        cp = pp = None
        if cell is not None:
            c = np.ascontiguousarray(cell, dtype=np.float32).reshape(9)
            pbc = (ctypes.c_bool * 3)(True, True, True)
            cp, pp = c.ctypes.data, ctypes.addressof(pbc)
        opt = _PredOpt(True, False, False, int(charge), int(spin))
        r = ctypes.c_void_p()
        if self.lib.mlipcpp_predict_ptr_with_options(self.m, n, x.ctypes.data, z.ctypes.data, cp, pp,
                                                     ctypes.byref(opt), ctypes.byref(r)) != 0:
            raise RuntimeError(self.lib.mlipcpp_get_last_error().decode())
        e = ctypes.c_float()
        self.lib.mlipcpp_result_get_energy(r, ctypes.byref(e))
        f = np.zeros((n, 3), dtype=np.float32)
        self.lib.mlipcpp_result_get_forces(r, f.ctypes.data, n)
        return float(e.value), f.astype(float)

    def predict(self, structs):
        from ase.data import atomic_numbers
        out = []
        for s in structs:
            try:
                e, f = self.call([atomic_numbers[x] for x in s.species], s.positions,
                                 s.cell if (s.cell is not None and any(s.pbc)) else None, s.charge, s.spin)
                out.append({"energy": e, "forces": f, "charges": None, "error": None})
            except RuntimeError as err:
                out.append({"energy": None, "forces": None, "charges": None, "error": str(err)})
        return out

    def ase_calculator(self, charge=0, spin=1):
        from ase.calculators.calculator import Calculator, all_changes
        be = self

        class _C(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                cell = atoms.cell.array if atoms.pbc.any() else None
                e, f = be.call(atoms.numbers, atoms.positions, cell, atoms.info.get("charge", charge),
                               atoms.info.get("spin_multiplicity", spin))
                self.results = {"energy": e, "forces": f}
        return _C()

    def info(self):
        return dict(super().info(), lib=self.lib_path, device=self.device,
                    ragged=os.environ.get("MLIPCPP_RAGGED", ""),
                    precision="TF32 (MLIPCPP_TF32=1)" if self.tf32 else "fp32")


def drop_ase_cache(calc) -> None:
    """Invalidate an ASE calculator's cached result.

    ``Calculator.check_state`` compares positions, numbers, cell and pbc only -- **not** ``atoms.info`` -- so two
    structures that differ only in total charge or spin multiplicity look identical to it and the second one is
    served the first one's energy. Our sets contain exactly such pairs (charge_parity, and every charge state of one
    geometry), and :meth:`ASEBackend.predict` groups structures by atom count, which puts them next to each other.
    So the cache is dropped before every structure. Calculators that cache internally on their own keys are
    unaffected, and energy/forces of one structure still come from a single evaluation.
    """
    for attr, empty in (("atoms", None), ("results", {})):
        if hasattr(calc, attr):
            try:
                setattr(calc, attr, dict(empty) if isinstance(empty, dict) else empty)
            except Exception:  # a calculator with a read-only property: nothing we can do, and nothing we break
                pass


class ASEBackend(Backend):
    """Any ASE calculator; charge/spin go through atoms.info (charge, spin_multiplicity, spin)."""
    name = "ase"
    passes_formal_charges = False

    def __init__(self, calc, label: str = "ase", formal: bool = False):
        self.calc, self.label, self.formal = calc, label, formal
        self.passes_formal_charges = formal  # a custom calculator may read atoms.arrays["formal_charges"]

    def predict(self, structs):
        # grouped by atom count (stable): shape-specialised JIT models (SO3LR/JAX) then recompile once per size
        order = sorted(range(len(structs)), key=lambda i: len(structs[i].species))
        res = self._predict([structs[i] for i in order])
        out = [None] * len(structs)
        for i, r in zip(order, res):
            out[i] = r
        return out

    def _predict(self, structs):
        out = []
        for s in structs:
            a = s.atoms(formal=self.formal)
            drop_ase_cache(self.calc)  # charge/spin live in atoms.info, which ASE's check_state ignores
            a.calc = self.calc
            try:
                e = float(a.get_potential_energy())
                f = np.asarray(a.get_forces(), float)
                R = getattr(self.calc, "results", {}) or {}  # read, never recompute, optional outputs
                q, mu = R.get("charges"), R.get("dipole")
                out.append({"energy": e, "forces": f, "charges": None if q is None else np.asarray(q, float),
                            "dipole": None if mu is None else np.asarray(mu, float).reshape(-1)[:3], "error": None})
            except Exception as err:  # a rejected charge/spin, an unsupported element...
                out.append({"energy": None, "forces": None, "charges": None, "error": f"{type(err).__name__}: {err}"})
        return out

    def info(self):
        meta = getattr(self.calc, "lrbench_meta", None)
        return dict(super().info(), calculator=self.label, **({"external_model": meta} if meta else {}))


def _literal(v: str):
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


def ase_from_spec(spec: str):
    """'module.path:Factory?kw=v&kw2=v2' -> calculator instance."""
    path, _, q = spec.partition("?")
    mod, _, attr = path.rpartition(":")
    if not mod:
        raise ValueError(f"ASE spec must be module.path:Factory, got {spec!r}")
    kw = {k: _literal(v) for k, v in (p.split("=", 1) for p in q.split("&") if p)}
    return getattr(importlib.import_module(mod), attr)(**kw)


def metatomic_calculator(path: str, device: str = "cpu"):
    try:
        from metatomic_ase import MetatomicCalculator
    except ImportError:
        from metatomic.torch.ase_calculator import MetatomicCalculator
    return MetatomicCalculator(path, device=None if device in ("auto", "") else device)


def make_backend(spec: str, device: str = "cpu", formal: bool = False, tf32: bool = False,
                 mlipcpp_bin: Optional[str] = None, mlipcpp_lib: Optional[str] = None, launcher: str = "",
                 chunk: int = 0) -> Backend:
    if spec.startswith("oracle:"):
        from .oracle import from_spec
        return from_spec(spec)
    if spec.startswith("ase:"):
        b = ASEBackend(ase_from_spec(spec[4:]), label=spec, formal=formal)
        b.passes_charge_spin = getattr(b.calc, "charge_spin", True)  # adapters say when a model ignores Q / spin
        return b
    if spec.startswith("mlipcpp-capi:"):
        if formal:
            raise ValueError("the C API does not take formal charges; use the mlipcpp-eval path")
        return MlipcppCAPI(spec[len("mlipcpp-capi:"):], mlipcpp_lib, device if device != "cpu" else "cpu", tf32)
    if spec.endswith(".pt"):
        if formal:
            raise ValueError("metatomic models get no formal charges here (--formal-charges none)")
        return ASEBackend(metatomic_calculator(spec, device), label=f"metatomic:{spec}")
    if spec.endswith(".gguf"):
        try:
            return MlipcppEval(spec, mlipcpp_bin, device, launcher, formal, tf32, chunk=chunk)
        except FileNotFoundError:
            if formal:
                raise
        if mlipcpp_lib or os.environ.get("MLIPCPP_C_LIB"):
            return MlipcppCAPI(spec, mlipcpp_lib, device, tf32)
        from mlipcpp.ase import MLIPCalculator  # bindings: no charge/spin, refuse charged structures below
        b = ASEBackend(MLIPCalculator(spec, backend=device), label=f"mlipcpp-python:{spec}")
        b.passes_charge_spin = False
        warnings.warn("mlip.cpp Python bindings pass no charge/spin: conditioned models see charge 0, spin 1")
        return b
    raise ValueError(f"cannot tell the backend of model spec {spec!r}")
