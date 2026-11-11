"""Timing on standard boxes:
force call = median of N calls after W warm-up calls, positions moved by a random 1e-3 A each call so nothing is
cached; MD = ASE VelocityVerlet at 0.5 fs, median step time after warm-up; peak GPU memory from nvidia-smi sampled
every 200 ms while the row runs. Every row carries its code path, precision and GPU model.

Boxes: any named structure file given with ``--box NAME=FILE``,
and generated water boxes ``water<N>`` (N atoms, 1.0 g/cm3 on a grid with random orientations; not equilibrated,
meant for timing only)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import threading
import time
from typing import List, Optional

import numpy as np

from .data import Structure

DT_FS = 0.5


def water_box(n_atoms: int, seed: int = 0, density: float = 1.0, periodic: bool = True) -> Structure:
    from .equivariance import random_rotation
    rng = np.random.default_rng(seed)
    nm = max(1, n_atoms // 3)
    vol = nm * 18.015 / (density * 0.6022140857)  # A^3
    L = vol ** (1 / 3)
    m = int(np.ceil(nm ** (1 / 3)))
    h = L / m
    g = np.array([(i, j, k) for i in range(m) for j in range(m) for k in range(m)][:nm], float) * h + h / 2
    th = np.deg2rad(104.52) / 2
    mol = np.array([[0, 0, 0], [0.9572 * np.sin(th), 0.9572 * np.cos(th), 0], [-0.9572 * np.sin(th), 0.9572 * np.cos(th), 0]])
    pos = np.concatenate([c + mol @ random_rotation(rng).T for c in g])
    sp = ["O", "H", "H"] * nm
    cell = np.eye(3) * L if periodic else None
    return Structure(f"water{3 * nm}", sp, pos, 0, 1, cell, (periodic,) * 3)


def read_box(path: str, name: str) -> Structure:
    from ase.io import read
    a = read(path)
    cell = np.array(a.cell[:]) if a.pbc.any() else None
    return Structure(name, list(a.get_chemical_symbols()), a.positions.copy(), int(a.info.get("charge", 0)),
                     int(a.info.get("spin_multiplicity", a.info.get("spin", 1))), cell, tuple(bool(p) for p in a.pbc))


def gpu_name() -> Optional[str]:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=20).stdout.strip().splitlines()[_gpu_index()]
    except Exception:
        return None


def _gpu_index() -> int:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return int(v) if v.isdigit() else 0


class GPUMem:
    """Peak memory.used (MiB) of the visible GPU while the block runs (whole device, as the protocol)."""

    def __enter__(self):
        self.peak, self.base, self._stop = 0, None, threading.Event()

        def run():
            while not self._stop.is_set():
                try:
                    v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                            "-i", str(_gpu_index())], capture_output=True, text=True,
                                           timeout=10).stdout.split()[0])
                    self.base = v if self.base is None else self.base
                    self.peak = max(self.peak, v)
                except Exception:
                    return
                self._stop.wait(0.2)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=15)


def _row(model, box: Structure, path, precision, extra=None):
    return dict(model=os.path.basename(model), box=box.key, n_atoms=len(box.species), code_path=path,
                precision=precision, gpu=gpu_name(), host=platform.node(), **(extra or {}))


def _finish(r, t_force=None, t_md=None):
    n = r["n_atoms"]
    if t_force is not None:
        r.update(force_ms=round(1e3 * t_force, 3), force_us_atom=round(1e6 * t_force / n, 3))
    if t_md is not None:
        r.update(md_ms_step=round(1e3 * t_md, 3), md_us_atom_step=round(1e6 * t_md / n, 3),
                 ns_day=round(DT_FS * 1e-6 * 86400 / t_md, 4), ns_day_basis="MD step")
    elif t_force is not None:
        r.update(ns_day=round(DT_FS * 1e-6 * 86400 / t_force, 4), ns_day_basis="force call only (no integrator)")
    return r


def time_calls(call, box: Structure, warm: int, n: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    ts = []
    for k in range(warm + n):
        x = box.positions + rng.normal(0, 1e-3, box.positions.shape)
        t0 = time.perf_counter()
        call(x)
        if k >= warm:
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def time_md(calc, box: Structure, warm: int, n: int) -> float:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase import units
    a = box.atoms()
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(0))
    md = VelocityVerlet(a, timestep=DT_FS * units.fs)
    md.run(warm)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        md.run(1)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def perf_capi(be, box: Structure, warm: int, n: int, md_steps: int, model: str):
    from ase.data import atomic_numbers
    z = [atomic_numbers[s] for s in box.species]
    cell = box.cell if box.cell is not None and any(box.pbc) else None
    path = f"mlip.cpp C API (ctypes){', ragged fast path' if os.environ.get('MLIPCPP_RAGGED') == '1' else ', padded graph'}"
    prec = "TF32 (MLIPCPP_TF32=1)" if be.tf32 else "fp32 (TF32 off)"
    r = _row(model, box, path, prec, {"warmup": warm, "n": n, "md_steps": md_steps})
    with GPUMem() as g:
        tf = time_calls(lambda x: be.call(z, x, cell, box.charge, box.spin), box, warm, n)
        tm = time_md(be.ase_calculator(box.charge, box.spin), box, min(warm, 5), md_steps) if md_steps else None
    r["peak_gpu_mib"] = g.peak or None
    return _finish(r, tf, tm)


def perf_ase(be, box: Structure, warm: int, n: int, md_steps: int, model: str):
    a = box.atoms()
    a.calc = be.calc

    def call(x):
        a.positions = x
        a.get_forces()
    r = _row(model, box, f"ASE calculator ({be.label})", os.environ.get("LRBENCH_PRECISION", "as configured"),
             {"warmup": warm, "n": n, "md_steps": md_steps})
    with GPUMem() as g:
        tf = time_calls(call, box, warm, n)
        tm = time_md(be.calc, box, min(warm, 5), md_steps) if md_steps else None
    r["peak_gpu_mib"] = g.peak or None
    return _finish(r, tf, tm)


def perf_eval(be, box: Structure, warm: int, n: int, model: str, formal: bool = False):
    """mlipcpp-eval --repeat n: one warm-up call, median of n calls on identical positions (the binary's own timer)."""
    from .backends import _clean_env, write_extxyz
    d = tempfile.mkdtemp(prefix="lrbench-perf-")
    xyz, out = os.path.join(d, "box.xyz"), os.path.join(d, "out.json")
    write_extxyz([box], xyz, formal)
    cmd = be.launcher + [be.binary, be.model, xyz, out, "--backend", be.device, "--repeat", str(n)]
    commit = os.path.join(os.path.dirname(be.binary), "COMMIT")
    commit = open(commit).read().strip() if os.path.exists(commit) else "?"
    graph = "ragged fast path (MLIPCPP_RAGGED=1)" if os.environ.get("MLIPCPP_RAGGED") == "1" else "padded graph"
    r = _row(model, box, f"mlipcpp-eval {commit} --repeat, {graph}, force call on identical positions"
             + (", formal_charges" if formal else ""),
             "TF32 (MLIPCPP_TF32=1)" if be.tf32 else "fp32 (GGML_CUDA_NO_TF32=1)", {"warmup": 1, "n": n})
    with GPUMem() as g:
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=_clean_env(os.environ, be.tf32), capture_output=True, text=True)
        wall = time.perf_counter() - t0
    if p.returncode:
        r["error"] = p.stderr[-1000:]
        return r
    f = json.load(open(out))["frames"][0]
    if "error" in f:
        r["error"] = f["error"]
        return r
    r["peak_gpu_mib"] = g.peak or None
    r["process_wall_s"] = round(wall, 2)
    return _finish(r, float(f["seconds"]))


def frag_cluster(n_atoms: int, seed: int = 0) -> Structure:
    """Open (non-periodic) water cluster with alternating +-1 formal charges on the O of successive molecules, so the
    O(N^2) host-side fragment Coulomb term has charged units to sum over (unphysical; timing only)."""
    s = water_box(n_atoms, seed, periodic=False)
    nm = len(s.species) // 3
    fq = np.zeros(len(s.species))
    fq[0::3] = [(1 if k % 2 == 0 else -1) for k in range(nm)]
    if nm % 2:
        fq[3 * (nm - 1)] = 0
    s.formal_charges = fq
    s.key = f"water-cluster{len(s.species)} (open, +-1 units)"
    return s


def run_perf(be, boxes: List[Structure], warm: int, n: int, md_steps: int, model: str, frag_sizes=()) -> dict:
    from .backends import ASEBackend, MlipcppCAPI, MlipcppEval
    rows = []
    for b in boxes:
        try:
            if isinstance(be, MlipcppCAPI):
                rows.append(perf_capi(be, b, warm, n, md_steps, model))
            elif isinstance(be, MlipcppEval):
                rows.append(perf_eval(be, b, warm, n, model))
            elif isinstance(be, ASEBackend):
                rows.append(perf_ase(be, b, warm, n, md_steps, model))
        except Exception as e:  # OOM on a big box should not lose the other rows
            rows.append(_row(model, b, be.name, "?", {"error": f"{type(e).__name__}: {e}"}))
        print(json.dumps(rows[-1]), flush=True)
    frag = []
    if frag_sizes:
        if not isinstance(be, MlipcppEval):
            frag.append({"error": "--frag needs the mlipcpp-eval backend (the only path that passes formal charges)"})
        else:
            for N in frag_sizes:
                c = frag_cluster(N)
                a = perf_eval(be, c, warm, n, model, formal=True)
                c0 = Structure(c.key, c.species, c.positions, c.charge, c.spin)
                b0 = perf_eval(be, c0, warm, n, model, formal=False)
                row = {"n_atoms": len(c.species), "with_formal_ms": a.get("force_ms"), "without_ms": b0.get("force_ms"),
                       "error": a.get("error") or b0.get("error")}
                if row["with_formal_ms"] and row["without_ms"]:
                    row["frag_term_ms"] = round(row["with_formal_ms"] - row["without_ms"], 3)
                frag.append(row)
                print(json.dumps(row), flush=True)
    return {"rows": rows, "frag_term": frag, "protocol": __doc__.strip()}
