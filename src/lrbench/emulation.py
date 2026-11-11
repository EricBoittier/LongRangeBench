"""DFT-emulation scores (`lrbench emulation`): models are judged against DFT's own outputs, pathologies included.

The rule (program memory lr-model-emulates-dft): a model that takes only positions, total charge and spin should
reproduce ωB97M-V/def2-TZVPD, including where DFT is physically "wrong" (delocalisation-error charge transfer,
fractional fragment charges, basis-bound anions, broken-symmetry states). Consequences made executable here:
  - a *charge leak* is a deviation from DFT's fragment charges (Löwdin sums), not from formal integers. A model that
    snaps fragments to formal integers is wrong wherever DFT is fractional;
  - *size consistency* is judged against DFT's own E_int(d) decay, not against zero;
  - two energy scores: the **emulation score** uses every point with a DFT label (tagged pathologies included); the
    **physics score** excludes points tagged ct_fractional / ct_ground_state / unbound_anion / spin_contaminated
    (and every canonical `ct_ambiguous` class). A model that emulates DFT well can lose on the physics score, because
    on the excluded points it follows DFT away from the "true" (exact-functional) behaviour, and a model with
    integer-charge physics built in can win the physics score while failing emulation.

Fragment-charge scoring: rows (model, set, id, class, d, fragment k, q_model, q_dft, q_formal). A point is
`integer-like` if |q_dft - q_formal| <= `frac_tol` (0.02 e, the dataset's label rule) else `fractional/CT`. Pass if
|q_model - q_dft| <= `tol` (0.05 e). The formal-charge deviation |q_model - q_formal| is reported next to it."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

PHYSICS_EXCLUDE_TAGS = ("ct_fractional", "ct_ground_state", "unbound_anion", "spin_contaminated")
TOL, FRAC_TOL = 0.05, 0.02


def dbin(d):
    if d is None:
        return "all"
    return "0-4" if d < 4 else "4-8" if d < 8 else "8-12" if d < 12 else ">=12"


def classify(q_dft, q_formal, frac_tol=FRAC_TOL):
    return "integer-like" if abs(q_dft - q_formal) <= frac_tol else "fractional/CT"


def score_rows(rows: Iterable[dict], tol=TOL, frac_tol=FRAC_TOL) -> dict:
    """rows: dicts with model, cls, d, q_model, q_dft, q_formal. Returns per model: pass rates and median deviations by
    point kind (integer-like / fractional/CT), and by class x distance bin."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("q_model") is None or r.get("q_dft") is None:
            continue
        kind = classify(r["q_dft"], r["q_formal"], frac_tol)
        dv, df = abs(r["q_model"] - r["q_dft"]), abs(r["q_model"] - r["q_formal"])
        for key in (("kind", kind), ("cls_d", f"{r['cls']}|{dbin(r.get('d'))}|{kind}"), ("all", "all")):
            by[r["model"]][key].append((dv, df))
    out = {}
    for m, d in by.items():
        out[m] = {"tol_e": tol, "frac_tol_e": frac_tol}
        for (grp, k), v in sorted(d.items()):
            a = np.array(v)
            out[m].setdefault(grp, {})[k] = {
                "n": len(a), "pass_rate_vs_dft": float(np.mean(a[:, 0] <= tol)),
                "median_abs_dev_vs_dft_e": float(np.median(a[:, 0])), "max_abs_dev_vs_dft_e": float(a[:, 0].max()),
                "pass_rate_vs_formal": float(np.mean(a[:, 1] <= tol)), "median_abs_dev_vs_formal_e": float(np.median(a[:, 1]))}
    return out


def rows_from_result(R: dict, data: Optional[str] = None, model: Optional[str] = None) -> List[dict]:
    """Fragment-charge rows from an `lrbench eval --save-predictions` result (sets[*].qfrag, member mean)."""
    from .data import load_set
    rows = []
    for s, blk in R.get("sets", {}).items():
        qf = blk.get("qfrag")
        if not qf:
            continue
        ds = load_set(s, data or (R.get("data", {}).get(s) or {}).get("data"))
        for rid, per in qf.items():
            r = ds.by_id.get(rid)
            per = [p for p in per if p is not None]
            if r is None or not per or (r.get("refs") or {}).get("charges_lowdin") is None:
                continue
            ql = np.asarray(r["refs"]["charges_lowdin"], float); fid = np.asarray(r["fragment_id"])
            qm = np.mean(np.asarray(per, float), axis=0)
            for k, qf_k in enumerate(r.get("fragment_charges") or []):
                rows.append(dict(model=model or R.get("label_short") or R.get("label"), set=s, id=rid, cls=r.get("cls"),
                                 d=r.get("distance"), frag=k, q_model=float(qm[k]), q_dft=float(ql[fid == k].sum()),
                                 q_formal=float(qf_k), tags=r.get("flags") or []))
    return rows


def rows_from_chemiscope(path: str, tag: str) -> List[dict]:
    """Fragment-charge rows from an `lrbench viz` chemiscope file (q_latent/TAG, q_lowdin, q_formal, fragment_id)."""
    import gzip
    d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    P = d["properties"]
    if f"q_latent/{tag}" not in P:
        return []
    sizes = [s["size"] for s in d["structures"]]
    off = np.concatenate([[0], np.cumsum(sizes)])
    qm, ql, qf = (np.asarray(P[k]["values"], float) for k in (f"q_latent/{tag}", "q_lowdin", "q_formal"))
    fid = np.asarray(P["fragment_id"]["values"], float)
    rows = []
    for i, s in enumerate(d["structures"]):
        sl = slice(off[i], off[i + 1]); f = fid[sl]
        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(ql[sl])) or P["role"]["values"][i] != "AB":
            continue
        for k in sorted(set(f.astype(int).tolist())):
            m = f == k
            rows.append(dict(model=tag, set=P["set"]["values"][i] if "set" in P else "dimers", id=P["id"]["values"][i],
                             cls=P["class"]["values"][i], d=float(P["distance"]["values"][i]), frag=int(k),
                             q_model=float(qm[sl][m].sum()), q_dft=float(ql[sl][m].sum()),
                             q_formal=float(np.round(qf[sl][m].sum())), tags=[]))
    return rows


def physics_ids(ds) -> List[str]:
    """Record ids kept by the physics score."""
    return [r["id"] for r in ds.records
            if not set(r.get("flags") or []) & set(PHYSICS_EXCLUDE_TAGS) and "ct_ambiguous" not in str(r.get("cls"))]


def size_consistency(model_rows: List[dict], dft_rows: List[dict], tol_meV=2.0, rel=0.25) -> Dict[str, dict]:
    """model_rows / dft_rows: dicts sys, d, E_int_meV. Pass at a distance if |model - DFT| <= max(tol_meV, rel*|DFT|),
    i.e. the model matches DFT's own decay (a DFT tail with residual CT counts as correct)."""
    D = {(r["sys"], float(r["d"])): r["E_int_meV"] for r in dft_rows if r.get("E_int_meV") is not None}
    out = {}
    for r in model_rows:
        k = (r["sys"], float(r["d"]))
        if k not in D or r.get("E_int_meV") is None:
            continue
        err = r["E_int_meV"] - D[k]
        out[f"{k[0]}|{k[1]:g}"] = dict(model=r["E_int_meV"], dft=D[k], err=err, ok=bool(abs(err) <= max(tol_meV, rel * abs(D[k]))))
    return out


def energy_scores(R: dict, data: Optional[str] = None) -> dict:
    """Emulation vs physics E_int MAE (meV) from a result's saved predictions, per set and class: emulation = every
    E_int group with a DFT label (tagged points included; `dft_pathology_do_not_score` = SCF failures excluded from both);
    physics = the same minus groups whose AB record carries a physics-exclusion tag or a ct_ambiguous class."""
    from .data import base_id, load_set
    out = {}
    for s, blk in R.get("sets", {}).items():
        P = blk.get("predictions")
        if not P:
            continue
        ds = load_set(s, data or (R.get("data", {}).get(s) or {}).get("data"))
        keep = set(physics_ids(ds))
        acc = defaultdict(lambda: {"emulation": [], "physics": []})
        for g in ds.groups:
            if g.get("kind") != "E_int" or g.get("ref") is None:
                continue
            rid = base_id(g["terms"][0][0]); r = ds.by_id.get(rid) or {}
            if "dft_pathology_do_not_score" in (r.get("flags") or []):
                continue
            try:
                v = sum(c * np.nanmean([x for x in P[t] if x is not None]) for t, c in g["terms"])
            except (KeyError, ZeroDivisionError):
                continue
            if not np.isfinite(v):
                continue
            e = abs(v - g["ref"]) * 1e3
            for c in (r.get("cls") or "all", "all"):
                acc[c]["emulation"].append(e)
                if rid in keep:
                    acc[c]["physics"].append(e)
        out[s] = {c: {k: {"n": len(v), "MAE_meV": float(np.mean(v)) if v else None, "median_meV": float(np.median(v)) if v else None}
                      for k, v in a.items()} for c, a in sorted(acc.items())}
    return out
