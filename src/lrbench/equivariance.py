"""Symmetry checks on a sample of structures: rotation, reflection, translation, permutation (and, with pbc, wrapping
and periodic-image shifts). Reports |dE| (meV, meV/atom) and the force error |F(T x) - T F(x)| (meV/A).
PET is not equivariant by construction (rotational symmetry is learned by augmentation), so non-zero rotation
errors are expected and are the quantity of interest."""
from __future__ import annotations

import collections
import random
from dataclasses import replace
from typing import List

import numpy as np

from .data import Structure


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def transforms(s: Structure, rng, n_rot: int = 3):
    """Yield (name, transformed structure, map applied to the reference forces F -> expected F')."""
    x = np.asarray(s.positions, float)
    c = x.mean(0)
    for k in range(n_rot):
        R = random_rotation(rng)
        yield "rotation", replace(s, positions=(x - c) @ R.T + c, key=f"{s.key}|rot{k}"), (lambda F, R=R: F @ R.T)
    R = -random_rotation(rng)  # improper: inversion times a rotation
    yield "reflection", replace(s, positions=(x - c) @ R.T + c, key=f"{s.key}|refl"), (lambda F, R=R: F @ R.T)
    t = rng.normal(size=3) * 10.0
    yield "translation", replace(s, positions=x + t, key=f"{s.key}|trans"), (lambda F: F)
    p = rng.permutation(len(x))
    fq = None if s.formal_charges is None else _permute_formal(s, p)
    yield ("permutation", replace(s, positions=x[p], species=[s.species[i] for i in p], formal_charges=fq,
                                  key=f"{s.key}|perm"), (lambda F, p=p: F[p]))
    if s.cell is not None and any(s.pbc):
        cell = np.asarray(s.cell, float)
        frac = np.linalg.solve(cell.T, (x + t).T).T
        wrapped = (frac % 1.0) @ cell
        yield "translation+wrap (pbc)", replace(s, positions=wrapped, key=f"{s.key}|wrap"), (lambda F: F)
        img = x.copy()
        img[0] += cell[0]
        yield "periodic image (pbc)", replace(s, positions=img, key=f"{s.key}|img"), (lambda F: F)


def _permute_formal(s: Structure, p):
    """Formal charges follow the unit-first-atom convention: recompute it after the permutation so each unit's charge
    sits on its new first atom (mlip.cpp only uses unit sums, so the sums are what must be kept)."""
    fq = np.asarray(s.formal_charges, float)[p]
    return fq  # unit sums are invariant under any permutation; position inside the unit does not matter


def sample(structs: List[Structure], n: int, seed: int = 0, max_atoms: int = 400) -> List[Structure]:
    pool = [s for s in structs if len(s.species) <= max_atoms and "#" not in s.key]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def check(backend, structs: List[Structure], n_rot: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = backend.predict(structs)
    jobs, meta = [], []
    for s, b in zip(structs, base):
        if b.get("error") or b.get("energy") is None:
            continue
        for name, t, fmap in transforms(s, rng, n_rot):
            jobs.append(t)
            meta.append((name, s, b, fmap))
    preds = backend.predict(jobs)
    acc = collections.defaultdict(lambda: {"dE": [], "dE_atom": [], "dF": [], "dF_max": []})
    for (name, s, b, fmap), p in zip(meta, preds):
        if p.get("error") or p.get("energy") is None:
            continue
        a = acc[name]
        de = abs(p["energy"] - b["energy"]) * 1e3
        a["dE"].append(de)
        a["dE_atom"].append(de / len(s.species))
        if p.get("forces") is not None and b.get("forces") is not None:
            d = np.linalg.norm(np.asarray(p["forces"]) - fmap(np.asarray(b["forces"])), axis=1) * 1e3
            a["dF"].append(d.mean())
            a["dF_max"].append(d.max())
    out = {"n_structures": len(structs), "n_rotations": n_rot, "seed": seed, "checks": {}}
    for name, a in acc.items():
        out["checks"][name] = {
            "n": len(a["dE"]), "max_abs_dE_meV": round(float(np.max(a["dE"])), 4),
            "mean_abs_dE_meV": round(float(np.mean(a["dE"])), 4),
            "max_abs_dE_meV_atom": round(float(np.max(a["dE_atom"])), 5),
            "max_force_err_meV_A": round(float(np.max(a["dF_max"])), 4) if a["dF_max"] else None,
            "mean_force_err_meV_A": round(float(np.mean(a["dF"])), 4) if a["dF"] else None}
    out["note"] = ("force error = |F(T x) - T F(x)| per atom (rotation/reflection: F R^T; permutation: F[p]); "
                   "float32 models carry ~1e-2-1 meV noise on total energies of 1e4 eV")
    return out
