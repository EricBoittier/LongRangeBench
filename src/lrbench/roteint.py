"""Rotational spread of the dimer interaction energy at fixed separations (`lrbench eval --rot-eint 8,12,16,24`).

For every dimer scan (pair x orientation) the record at the scan's 8 A point is taken as a rigid template. Fragment B
is translated along the COM(A) -> COM(B) axis until the closest-atom distance d_min (the dataset's `distance`) equals
each requested d. The whole dimer is then rigidly rotated about its centre of mass by n random rotations (plus the
identity), and E_int = E(AB) - E(A) - E(B) is evaluated with the same rotation applied to all three structures
(monomers are the rotated rigid slices, at their own fragment charges, as in the canonical E_int groups).

An exactly invariant model gives zero spread. PET is not rotation-equivariant by construction, so the spread is its
rotational noise on E_int at that separation: the quantity to set beside long-range tail signals of a few meV (the
charges-only multipole floor has median 2.9 meV beyond 8 A). No DFT is needed: the numbers are model-only.

Reported per distance and per class: median / max over scans of the standard deviation (ddof 1) and of the range
(max - min) of E_int over the rotations, and the median |E_int| for scale."""
from __future__ import annotations

import collections
from typing import Dict, List, Sequence

import numpy as np

from .data import DataSet, structure
from .equivariance import random_rotation


def _com(species, x):
    from ase.data import atomic_masses, atomic_numbers
    m = np.array([atomic_masses[atomic_numbers[s]] for s in species])
    return (m[:, None] * x).sum(0) / m.sum()


def _dmin(xa, xb):
    return float(np.min(np.linalg.norm(xa[:, None] - xb[None], axis=-1)))


def place(rec: dict, d: float) -> np.ndarray:
    """Positions of `rec` with fragment 1 moved along the COM axis so that the closest-atom distance is d (A)."""
    x = np.asarray(rec["positions"], float).copy()
    fid = np.asarray(rec["fragment_id"])
    a, b = fid == 0, fid == 1
    sp = rec["species"]
    u = _com([sp[i] for i in np.where(b)[0]], x[b]) - _com([sp[i] for i in np.where(a)[0]], x[a])
    u /= np.linalg.norm(u)
    lo, hi = -_dmin(x[a], x[b]) + 1e-3, d + 50.0
    for _ in range(80):  # bisection on the shift s: d_min(s) is monotone along the COM axis beyond contact
        s = 0.5 * (lo + hi)
        if _dmin(x[a], x[b] + s * u) < d:
            lo = s
        else:
            hi = s
    x[b] += 0.5 * (lo + hi) * u
    return x


def templates(ds: DataSet, d_template: float = 8.0) -> List[dict]:
    """One dimer record per scan (record.group), the one whose distance is closest to d_template, unflagged."""
    by = collections.defaultdict(list)
    for r in ds.records:
        if r.get("role") != "AB" or r.get("fragment_id") is None or ds.flagged(r["id"]) or ds.partner_copy(r["id"]):
            continue
        by[r.get("group") or r["id"]].append(r)
    out = []
    for g, rs in sorted(by.items()):
        r = min(rs, key=lambda r: abs(float(r.get("distance") or 0) - d_template))
        if abs(float(r.get("distance") or 0) - d_template) < 0.5:
            out.append(r)
    return out


def run(backend, ds: DataSet, distances: Sequence[float], n_rot: int = 8, seed: int = 0,
        max_scans: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    T = templates(ds)
    if max_scans and len(T) > max_scans:
        T = [T[i] for i in sorted(np.random.default_rng(seed).choice(len(T), max_scans, replace=False))]
    jobs, meta = [], []
    for r in T:
        nf = len(r.get("fragment_charges") or [])
        if nf != 2:
            continue
        Rs = [np.eye(3)] + [random_rotation(rng) for _ in range(n_rot)]
        for d in distances:
            x0 = place(r, d)
            c = x0.mean(0)
            for k, R in enumerate(Rs):
                rr = dict(r, positions=((x0 - c) @ R.T + c).tolist())
                key = f"{r['id']}|d{d:g}|r{k}"
                for t, frags in (("AB", None), ("A", (0,)), ("B", (1,))):
                    s = structure(rr, frags, key=f"{key}|{t}")
                    jobs.append(s)
                    meta.append((r, d, k, t))
    preds = backend.predict(jobs) if jobs else []
    E = collections.defaultdict(dict)  # (scan id, d) -> k -> {AB, A, B}
    for (r, d, k, t), p in zip(meta, preds):
        if p.get("error") or p.get("energy") is None:
            continue
        E[(r["id"], d)].setdefault(k, {})[t] = p["energy"]
    rows = []
    cls = {r["id"]: r.get("cls") for r in T}
    dfar = max(distances)
    for (rid, d), per in E.items():
        far = E.get((rid, dfar), {})
        ks = [k for k, v in per.items() if len(v) == 3]
        ei = np.array([(per[k]["AB"] - per[k]["A"] - per[k]["B"]) * 1e3 for k in ks])
        if len(ei) < 3:
            continue
        row = dict(scan=rid, cls=cls[rid], d=d, n=len(ei), sd=float(ei.std(ddof=1)), range=float(ei.max() - ei.min()),
                   mean=float(ei.mean()), e0=float(ei[0]), E_int_rot=ei.round(4).tolist())
        kt = [k for k in ks if len(far.get(k, {})) == 3]
        if d != dfar and len(kt) >= 3:  # offset-free tail at fixed rotation, as the tail metrics difference it
            tl = np.array([(per[k]["AB"] - per[k]["A"] - per[k]["B"] - far[k]["AB"] + far[k]["A"] + far[k]["B"]) * 1e3
                           for k in kt])
            row.update(tail_sd=float(tl.std(ddof=1)), tail_range=float(tl.max() - tl.min()), tail_mean=float(tl.mean()))
        rows.append(row)
    summ: Dict[str, dict] = {}
    for d in distances:
        sel = [x for x in rows if x["d"] == d]
        if not sel:
            continue
        b = {"n_scans": len(sel), **_stats(sel), "by_class": {}}
        for c in sorted({x["cls"] for x in sel}):
            b["by_class"][c] = {"n_scans": sum(x["cls"] == c for x in sel), **_stats([x for x in sel if x["cls"] == c])}
        summ[f"{d:g}"] = b
    return {"distances_A": list(map(float, distances)), "n_rotations": n_rot, "seed": seed, "template_d_A": 8.0,
            "n_templates": len(T), "by_distance": summ, "per_scan": rows,
            "definition": "E_int(R) = E(R AB) - E(R A) - E(R B) for n random rigid rotations R (+ identity) about the "
                          "dimer centroid; fragment B translated along the COM axis to closest-atom distance d from "
                          "each scan's 8 A record; spread = std (ddof 1) and range of E_int over rotations, meV",
            "tail": "tail_* = spread over rotations of E_int(d) - E_int(d_far) at the same rotation (d_far = largest "
                    "requested d), the offset-free difference the tail metrics score",
            "reference": "multipole oracle, charges-only tier: median floor error 2.9 meV beyond 8 A (PR #15)"}


def _stats(sel):
    sd = np.array([x["sd"] for x in sel]); rg = np.array([x["range"] for x in sel])
    return {"sd_median_meV": round(float(np.median(sd)), 4), "sd_p90_meV": round(float(np.percentile(sd, 90)), 4),
            "sd_max_meV": round(float(sd.max()), 4), "range_median_meV": round(float(np.median(rg)), 4),
            "range_max_meV": round(float(rg.max()), 4),
            "abs_Eint_median_meV": round(float(np.median([abs(x["mean"]) for x in sel])), 4),
            **({} if "tail_sd" not in sel[0] else {
                "tail_sd_median_meV": round(float(np.median([x["tail_sd"] for x in sel])), 4),
                "tail_sd_p90_meV": round(float(np.percentile([x["tail_sd"] for x in sel], 90)), 4),
                "tail_range_max_meV": round(float(max(x["tail_range"] for x in sel)), 4),
                "abs_tail_median_meV": round(float(np.median([abs(x["tail_mean"]) for x in sel])), 4)})}
