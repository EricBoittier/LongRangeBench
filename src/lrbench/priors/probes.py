"""Probe geometries for the priors suite, built from the canonical dimer records (no new labels needed).

  - ``templates``: one record per (pair, orientation) at the canonical distance closest to 8 A, with its fragment
    index arrays, charges and spins;
  - ``scan``: B moved rigidly along the A->B COM axis to closest-contact distances d (``place``, the bisection of
    tools/scans.py); each point carries r_com, so rigid-translation work integrals are exact in r_com;
  - ``monomers``: A and B alone at their own fragment charge / spin (and at alternative charge assignments);
  - ``boxed``: a structure centred in a cubic box of edge L (pbc on).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data import Structure, combine_spins

FINE_GRID = (list(np.round(np.arange(3.0, 6.0, 0.25), 3)) + list(np.round(np.arange(6.0, 12.0, 0.05), 3))
             + [12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0])


def _dmin(x, y):
    return float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1)))


def place(species, positions, ia, ib, d):
    """Move fragment ib rigidly along the (unweighted) A->B centroid axis so the closest A-B atom distance is d."""
    x = np.asarray(positions, float).copy()
    ia, ib = np.asarray(ia), np.asarray(ib)
    u = x[ib].mean(0) - x[ia].mean(0)
    u /= np.linalg.norm(u)
    lo, hi = -_dmin(x[ia], x[ib]) + 1e-3, d + 200.0
    for _ in range(100):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if _dmin(x[ia], x[ib] + s * u) < d else (lo, s)
    x[ib] += 0.5 * (lo + hi) * u
    return x


@dataclass
class Template:
    pair: str
    orient: int
    cls: str
    rec: dict
    ia: np.ndarray
    ib: np.ndarray
    qa: int
    qb: int
    sa: int
    sb: int

    @property
    def scan_id(self):
        return f"{self.pair}:o{self.orient}"

    @property
    def bare_ions(self):
        return len(self.ia) == 1 and len(self.ib) == 1

    def ab(self, d: float, tag: str = "scan") -> Structure:
        r = self.rec
        x = place(r["species"], r["positions"], self.ia, self.ib, d)
        return Structure(f"{self.scan_id}|{tag}|d{d:g}", list(r["species"]), x, self.qa + self.qb,
                         combine_spins([self.sa, self.sb]))

    def frag(self, k: int, charge: Optional[int] = None, spin: Optional[int] = None) -> Structure:
        idx = self.ia if k == 0 else self.ib
        r = self.rec
        q = (self.qa, self.qb)[k] if charge is None else charge
        s = (self.sa, self.sb)[k] if spin is None else spin
        return Structure(f"{self.scan_id}|frag{k}|q{q}|s{s}", [r["species"][i] for i in idx],
                         np.asarray(r["positions"], float)[idx], int(q), int(s))


def templates(ds, orients: Sequence[int] = (0,), near: float = 8.0) -> List[Template]:
    by: Dict[tuple, dict] = {}
    for r in ds.records:
        ex = r.get("extra") or {}
        if r.get("fragment_id") is None or r.get("distance") is None or len(r.get("fragment_charges") or []) != 2:
            continue
        k = (ex.get("pair", r.get("group")), int(ex.get("orient", 0)))
        if k[1] not in orients:
            continue
        if k not in by or abs(r["distance"] - near) < abs(by[k]["distance"] - near):
            by[k] = r
    out = []
    for (pair, o), r in sorted(by.items()):
        fid = np.asarray(r["fragment_id"])
        sp = r.get("fragment_spins") or [1, 1]
        out.append(Template(pair, o, r.get("cls") or "", r, np.where(fid == 0)[0], np.where(fid == 1)[0],
                            int(r["fragment_charges"][0]), int(r["fragment_charges"][1]), int(sp[0]), int(sp[1])))
    return out


def r_com(s: Structure, ia, ib) -> float:
    x = np.asarray(s.positions)
    return float(np.linalg.norm(x[ib].mean(0) - x[ia].mean(0)))


def boxed(s: Structure, L: float) -> Structure:
    x = np.asarray(s.positions, float)
    x = x - x.mean(0) + L / 2
    return replace(s, key=f"{s.key}|box{L:g}", positions=x, cell=np.eye(3) * L, pbc=(True, True, True))
