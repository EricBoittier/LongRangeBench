"""Multipole oracle: the long-range energy a model could reach from the fragments' DFT moments alone.

Model spec ``oracle:multipole?order=2&ind=1&moments=frags.json&rcut=8`` (lrbench eval --model ...):
- dimer records at distance >= rcut get E(AB) = point-multipole energy of the two rigid fragments (multipole.energies,
  truncated at rank ``order``, plus induction if ``ind``); the fragment terms (#frag=) get 0, so E_int = the oracle;
- everything else (other sets, points inside rcut) is returned as failed and not scored.
Moments: frags.json of the dimer build ({name: {R (Angstrom), charge, mu, theta, alpha (au, about the COM)}}); each
fragment is placed by a Kabsch fit of its reference geometry onto the record's atoms. The integer fragment charge
replaces the DFT trace (as the dataset's E_mp). Only differences of oracle E_int are meaningful (tail eps, slopes,
offset-free errors); its absolute energies and forces are not, and the JSON says so.
Pulled-apart set A (merge-aug): ``pulled_energies`` uses monomers.jsonl (lab-frame mu_au/theta_au/alpha_au; U is only
translated, so no rotation)."""
from __future__ import annotations

import ast
import json
import os
from typing import Dict, List

import numpy as np

from . import multipole as MP
from .backends import Backend


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_frags(path: str) -> Dict[str, dict]:
    F = json.load(open(path))
    out = {}
    for k, v in F.items():
        v = {a: _lit(b) for a, b in v.items()}
        if "mu" not in v:
            continue
        out[k] = dict(R=np.asarray(v["R"], float), q=float(v["charge"]), mu=np.asarray(v["mu"], float).ravel()[:3],
                      theta=np.asarray(v["theta"], float).reshape(3, 3),
                      alpha=None if v.get("alpha") is None else np.asarray(v["alpha"], float).reshape(3, 3))
    return out


def dimer_moments(rec: dict, frags: dict):
    """Lab-frame moments and COMs of both fragments of a dimer record, or None if a fragment has no moments."""
    ex = rec.get("extra") or {}
    names = [ex.get("A"), ex.get("B")]
    fid = np.asarray(rec["fragment_id"]); x = np.asarray(rec["positions"], float); sp = list(rec["species"])
    out = []
    for f, n in enumerate(names):
        if n not in frags:
            return None
        m = fid == f; xf = x[m]; F = frags[n]
        if len(F["R"]) != len(xf):
            return None
        M = MP.rotate(dict(q=float((rec.get("fragment_charges") or [F["q"], F["q"]])[f]), mu=F["mu"], theta=F["theta"],
                           alpha=F["alpha"]), MP.kabsch(F["R"], xf))
        out.append((M, MP.com([s for s, k in zip(sp, m) if k], xf)))
    return out


def dimer_energies(rec: dict, frags: dict):
    mm = dimer_moments(rec, frags)
    if mm is None:
        return None
    (A, ca), (B, cb) = mm
    return MP.energies(A, B, cb - ca)


class MultipoleOracle(Backend):
    name = "oracle"
    passes_charge_spin = True

    def __init__(self, order: int = 2, ind: bool = True, moments: str = "", rcut: float = 8.0):
        self.order, self.ind, self.rcut = int(order), bool(int(ind)), float(rcut)
        self.moments = moments or os.environ.get("LRBENCH_FRAG_MOMENTS", "")
        if not self.moments or not os.path.exists(self.moments):
            raise FileNotFoundError("oracle: give moments=frags.json (or $LRBENCH_FRAG_MOMENTS)")
        self.frags = load_frags(self.moments); self.table: Dict[str, float] = {}
        self.key = f"O{self.order}" + ("+ind" if self.ind else "")

    def bind(self, ds):
        for r in ds.records:
            d = r.get("distance")
            if ds.name != "dimers" or d is None or float(d) < self.rcut - 1e-6 or not r.get("fragment_id"):
                continue
            e = dimer_energies(r, self.frags)
            if e is not None:
                self.table[r["id"]] = e[self.key]

    def predict(self, structs) -> List[dict]:
        out = []
        for s in structs:
            rid, _, frag = s.key.partition("#")
            e = (0.0 if frag else self.table[rid]) if rid in self.table else None
            out.append({"energy": e, "forces": None, "charges": None,
                        "error": None if e is not None else "outside the oracle's domain (not a dimer point >= rcut)"})
        return out

    def info(self):
        return dict(super().info(), oracle=self.key, moments=os.path.abspath(self.moments), rcut=self.rcut,
                    note="E_int differences only; absolute energies/forces of the oracle are not meaningful")


def from_spec(spec: str) -> MultipoleOracle:
    kind, _, q = spec[len("oracle:"):].partition("?")
    if kind != "multipole":
        raise ValueError(f"unknown oracle {kind!r} (only oracle:multipole)")
    kw = dict(p.split("=", 1) for p in q.split("&") if "=" in p)
    return MultipoleOracle(**kw)


def pulled_energies(points: List[dict], frames_xyz: str, monomers: str, rcut_min: float = 0.0):
    """Oracle energies (eV, dict key -> array over points) for the pulled-apart val points (prep_tail.py points.jsonl,
    frames in val.xyz order, fragment 0 = R, 1 = U)."""
    from ase.io import read
    mono = {}
    for l in open(monomers):
        m = json.loads(l); mono.setdefault(m["id"], {})[m["which"]] = m
    fr = read(frames_xyz, ":")
    keys = [f"O{L}{s}" for L in (0, 1, 2) for s in ("", "+ind")]
    E = {k: np.full(len(fr), np.nan) for k in keys}
    for p in points:
        m = mono.get(p["group"])
        if not m or "U" not in m or "R" not in m:
            continue
        at = fr[p["k"]]; fid = np.asarray(p["frag"]); x = at.positions; sp = at.get_chemical_symbols()
        M = {w: dict(q=float(Q), mu=np.asarray(m[w]["mu_au"]), theta=np.asarray(m[w]["theta_au"]),
                     alpha=None if m[w].get("alpha_au") is None else np.asarray(m[w]["alpha_au"]))
             for w, Q in (("R", p["QR"]), ("U", p["QU"]))}
        c = [MP.com([s for s, f in zip(sp, fid) if f == k], x[fid == k]) for k in (0, 1)]
        e = MP.energies(M["R"], M["U"], c[1] - c[0])
        for k in keys:
            E[k][p["k"]] = e[k]
    return E


def _selftest():
    rng = np.random.default_rng(0)
    qa, qb = rng.normal(size=5), rng.normal(size=4); xa, xb = rng.normal(size=(5, 3)) * .4, rng.normal(size=(4, 3)) * .4
    R = np.array([30.0, 4.0, -2.0])
    exact = sum(qa[i] * qb[j] / np.linalg.norm(R + xb[j] - xa[i]) for i in range(5) for j in range(4))
    errs = [abs(MP.electrostatic(MP.point_moments(qa, xa), MP.point_moments(qb, xb), R, L) - exact) for L in (0, 1, 2)]
    assert errs[2] < errs[1] < errs[0], errs  # each order closes part of the gap
    Rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]; Rot *= np.sign(np.linalg.det(Rot))
    assert np.allclose(MP.kabsch(xa, xa @ Rot.T + 1.0), Rot)


_selftest()


TRIMER_FRAGS = {"Li": "Li+", "Bz": "C6H6", "PF6": "PF6-", "C60": None}


def trimer_three_body(ds, frags: Dict[str, dict], L: int = 2) -> List[dict]:
    """Three-body induction of each trimer from the fragments' DFT moments (frags.json of the dimer build):
    E3_ind = E_ind(ABC) - sum over pairs E_ind(pair), with E_ind = -1/2 sum_i F_i.alpha_i.F_i and F_i the field of the
    other fragments' permanent moments (rank <= L, no back-polarisation). Electrostatics is pairwise additive, so this
    is the whole non-additive part of the oracle: E3_ind = -sum_i sum_{j<k} F_ij.alpha_i.F_ik (meV). Fragments without
    moments (C60) make the system unscorable (listed with a reason)."""
    import re
    out = []
    for r in ds.records:
        if r.get("role") != "ABC":
            continue
        name = (r.get("extra") or {}).get("geom", r["id"])
        parts = re.match(r"([A-Z][a-z]*)(Bz|C60)([A-Za-z0-9]+)_", name)
        keys = [TRIMER_FRAGS.get(p) for p in (parts.groups() if parts else ())]
        if not parts or any(k is None or k not in frags for k in keys):
            out.append({"system": name, "E3_ind_meV": None, "reason": "no DFT moments for a fragment"})
            continue
        fid = np.asarray(r["fragment_id"]); x = np.asarray(r["positions"], float); sp = list(r["species"])
        M, C = [], []
        for f, k in enumerate(keys):
            m = fid == f; F = frags[k]
            M.append(MP.truncate(MP.rotate(dict(q=float(r["fragment_charges"][f]), mu=F["mu"], theta=F["theta"],
                                                alpha=F["alpha"]), MP.kabsch(F["R"], x[m])), L))
            C.append(MP.com([s for s, k2 in zip(sp, m) if k2], x[m]) / MP.BOHR)
        field = lambda j, i: -MP.potential_derivs(M[j], MP.tensors(C[i] - C[j]))[1]  # field of j at i
        e3 = 0.0
        for i in range(3):
            if M[i].get("alpha") is None:
                continue
            j, k = [n for n in range(3) if n != i]
            e3 -= field(j, i) @ np.asarray(M[i]["alpha"]) @ field(k, i)
        out.append({"system": name, "E3_ind_meV": e3 * MP.HA * 1e3, "order": L,
                    "fragments": keys, "d": (r.get("extra") or {}).get("d")})
    return out
