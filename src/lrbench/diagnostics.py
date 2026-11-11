"""Distil ``lrbench eval`` / ``lrbench priors`` result JSONs down to what a comparison view needs.

One implementation, two consumers: ``notebooks/long_range_diagnostics.py`` calls it when a filesystem is there, and
``tools/build_diagnostics_site.py`` calls it to freeze the same numbers into the exported (WebAssembly) site, which
has no filesystem. Keeping it here rather than in the notebook is what stops the two from drifting.

    from lrbench.diagnostics import collect
    data = collect("runs")          # {"results": {...}, "priors": {...}, "source": "/abs/path"}

Every number keeps its 95 % interval as ``(estimate, lo, hi)``; a quantity the harness reported without one comes
back as ``(estimate, None, None)``, and one it could not compute at all as ``(None, None, None)`` -- never 0.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional, Tuple

Triple = Tuple[Optional[float], Optional[float], Optional[float]]


def est(v) -> Triple:
    """``[estimate, lo, hi]`` -> a 3-tuple; a bare number -> ``(x, None, None)``; anything else -> all ``None``."""
    if isinstance(v, (list, tuple)) and v:
        return (v[0], v[1] if len(v) > 1 else None, v[2] if len(v) > 2 else None)
    return (v, None, None) if isinstance(v, (int, float)) else (None, None, None)


def distil(name: str, J: dict) -> dict:
    """One ``lrbench eval`` result -> model provenance plus per-set, per-class diagnostics."""
    m = (J.get("members") or [{}])[0]
    ext = m.get("external_model") or {}
    lr = str(ext.get("long_range", "") or "?")
    out = {
        "model": name,
        "label": J.get("label", name),
        "level": (J.get("level") or {}).get("model_level", "?"),
        "mismatch": bool((J.get("level") or {}).get("mismatch")),
        "long_range": lr,
        # "has an explicit long-range term" is the adapter's own declaration, not a guess from the name
        "has_lr": not lr.lower().startswith(("none", "?")),
        "charge_spin": str(ext.get("charge_spin", "?")),
        "licence": str(ext.get("licence", "?")),
        "package": f"{ext.get('package', '')} {ext.get('version', '')}".strip(),
        "wall_s": J.get("wall_s"),
        "sets": {},
    }
    for s, S in (J.get("sets") or {}).items():
        E = S.get("E_int") or {}
        dt = ((S.get("normalised") or {}).get("dimer_tail")) or {}
        by_class = dt.get("by_class") or {}
        byrd = dt.get("by_class_distance") or {}
        rows = list(byrd.values()) if isinstance(byrd, dict) else list(byrd)
        charges = S.get("charges")
        d = {
            "n_records": S.get("n_records"),
            "n_failed": (E.get("n_failed") or 0) + (S.get("n_prediction_errors") or 0),
            "ddE": est((S.get("ddE") or {}).get("slope")),
            "ddE_mae": est((S.get("ddE") or {}).get("mae")),
            "F_mae": est(((S.get("energy_force") or {}).get("all") or {}).get("F_mae_meV_A")),
            "E_mae": est(((S.get("energy_force") or {}).get("all") or {}).get("E_mae_meV_atom")),
            "charges_skipped": charges.get("skipped") if isinstance(charges, dict) else None,
            "classes": {},
            "eps_r": [],
        }
        for c, t in (E.get("tail") or {}).items():
            e = by_class.get(c) or {}
            off = (E.get("offset_free_8A") or {}).get(c) or {}
            far = (E.get("abs_ge12") or {}).get(c) or {}
            d["classes"][c] = {
                "tail": est(t.get("slope")),
                "n_scans": t.get("n_scans"), "n_points": t.get("n_points"),
                "dft_span_meV": t.get("dft_span_meV"),
                "eps": est(e.get("median")),
                "kBT": e.get("tail_err_kBT_median"),
                "frac_gt_1kBT": e.get("frac_tail_err_gt_1kBT"),
                "prior_check": e.get("prior_check_median_dft_over_phys"),
                "off8": est(off.get("mae")),
                "flat8": off.get("flat_model_mae"),
                "far12": est(far.get("mae")),
                "dft_far_mean_abs": far.get("dft_mean_abs"),
            }
        for r in rows:
            if r.get("r") is not None and r.get("cls"):
                d["eps_r"].append({"cls": r["cls"], "r": float(r["r"]), "eps": est(r.get("median"))[0]})
        out["sets"][s] = d
    return out


def distil_priors(name: str, J: dict) -> dict:
    """One ``lrbench priors`` result -> the check statuses and their one-line summaries."""
    return {
        "model": name,
        "label": J.get("label", name),
        "backend": J.get("backend", ""),
        "cutoff_A": J.get("cutoff_A"),
        "declared_local": J.get("declared_local"),
        "checks": {k: {"status": str(v.get("status", "?")), "summary": str(v.get("summary", ""))}
                   for k, v in (J.get("checks") or {}).items()},
    }


def scans(J: dict, setname: str) -> list:
    """Class-mean E_int against distance, as the harness already binned it: the material for a distance scan.

    ``[{cls, d, n, dft, model, mae}]``, meV. These are means over the scans of a class at that distance, so the
    curve is the class's behaviour rather than one pair's.
    """
    rows = (((J.get("sets") or {}).get(setname) or {}).get("E_int") or {}).get("by_distance") or []
    out = []
    for r in rows:
        if r.get("d") is None or r.get("dft") is None or r.get("model") is None:
            continue
        out.append({"cls": r.get("cls", "?"), "d": float(r["d"]), "n": r.get("n"),
                    "dft": float(r["dft"]), "model": float(r["model"]),
                    "mae": None if r.get("mae") is None else float(r["mae"])})
    return out


def _find(path: str, base: Optional[str]) -> Optional[str]:
    """A result records the data paths as the eval saw them, which is usually relative to where it ran."""
    for cand in (path, os.path.join(base or "", path), os.path.join(base or "", os.path.basename(path))):
        if cand and os.path.exists(cand):
            return cand
    return None


def parity(J: dict, setname: str, groups: Optional[list] = None, kind: str = "E_int",
           base: Optional[str] = None) -> list:
    """Per-group model value against its reference, rebuilt from the saved predictions.

    ``lrbench eval --save-predictions`` stores one energy per evaluated term; a group's value is the same linear
    combination of those terms that defines the reference, so model and reference are formed identically and a
    parity plot compares like with like. ``[{cls, d, ref, model}]`` in meV. Groups that touch a DFT pathology, or
    any term the model failed, are left out -- never plotted as zero.

    ``groups`` defaults to the groups file the result itself records (``data.<set>.groups``), so a result carries
    everything needed to rebuild this.
    """
    S = (J.get("sets") or {}).get(setname) or {}
    P = S.get("predictions") or {}
    if not P:
        return []
    if groups is None:
        gp = ((J.get("data") or {}).get(setname) or {}).get("groups")
        gp = _find(gp, base) if gp else None
        if not gp:
            return []
        with open(gp) as f:
            groups = json.load(f)
    out = []
    for g in groups or []:
        if g.get("kind") != kind or g.get("ref") is None:
            continue
        meta = g.get("meta") or {}
        if any("pathology" in str(x) for x in (meta.get("flags") or [])):
            continue
        v, ok = 0.0, True
        for term, coef in g.get("terms") or []:
            vals = P.get(term)
            if not vals or vals[0] is None:
                ok = False
                break
            good = [x for x in vals if isinstance(x, (int, float))]
            if not good:
                ok = False
                break
            v += float(coef) * (sum(good) / len(good))      # ensemble members are averaged, as in scoring
        if ok:
            out.append({"cls": meta.get("cls", "?"), "d": meta.get("dmin"),
                        "ref": round(1e3 * float(g["ref"]), 4), "model": round(1e3 * v, 4)})
    return out


def collect(runs: str = "runs", with_points: bool = True, max_points: int = 1200,
            base: Optional[str] = None) -> dict:
    """Every result in ``runs/*.json`` and every prior suite in ``runs/priors/*.json``, distilled."""
    results, priors = {}, {}
    base = base or os.path.dirname(os.path.abspath(runs))   # where the evals ran, for the recorded data paths
    for p in sorted(glob.glob(os.path.join(runs, "*.json"))):
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue  # a file still being written
        if "sets" not in J:
            continue
        name = os.path.basename(p)[:-5]
        d = distil(name, J)
        if with_points:
            for s in d["sets"]:
                d["sets"][s]["scans"] = scans(J, s)
                pts = parity(J, s, base=base)
                if len(pts) > max_points:        # thin evenly; a parity cloud does not need every point
                    step = len(pts) / max_points
                    pts = [pts[int(i * step)] for i in range(max_points)]
                d["sets"][s]["parity"] = pts
        results[name] = d
    for p in sorted(glob.glob(os.path.join(runs, "priors", "*.json"))):
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue
        if "checks" in J:
            priors[os.path.basename(p)[:-5]] = distil_priors(os.path.basename(p)[:-5], J)
    return {"results": results, "priors": priors, "source": os.path.abspath(runs)}
