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


def collect(runs: str = "runs") -> dict:
    """Every result in ``runs/*.json`` and every prior suite in ``runs/priors/*.json``, distilled."""
    results, priors = {}, {}
    for p in sorted(glob.glob(os.path.join(runs, "*.json"))):
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue  # a file still being written
        if "sets" in J:
            results[os.path.basename(p)[:-5]] = distil(os.path.basename(p)[:-5], J)
    for p in sorted(glob.glob(os.path.join(runs, "priors", "*.json"))):
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue
        if "checks" in J:
            priors[os.path.basename(p)[:-5]] = distil_priors(os.path.basename(p)[:-5], J)
    return {"results": results, "priors": priors, "source": os.path.abspath(runs)}
