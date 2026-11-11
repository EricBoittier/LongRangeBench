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
import math
import os
from typing import Optional, Tuple

Triple = Tuple[Optional[float], Optional[float], Optional[float]]


def est(v) -> Triple:
    """``[estimate, lo, hi]`` -> a 3-tuple; a bare number -> ``(x, None, None)``; anything else -> all ``None``."""
    if isinstance(v, (list, tuple)) and v:
        return (v[0], v[1] if len(v) > 1 else None, v[2] if len(v) > 2 else None)
    return (v, None, None) if isinstance(v, (int, float)) else (None, None, None)


#: The ΔΔE family that *is* task B1 — the fixed-charge quartet of PROTOCOL_B: two local arrangements, one charge
#: far away, differenced twice so composition and total charge cancel.
#:
#: Naming it is the point. ``families_fixedq`` holds several ΔΔE families whose reference magnitudes differ by more
#: than an order of magnitude, and a single slope fitted through the pooled points is dominated by the few largest
#: and describes none of them. Measured on the real runs, models that predict *nothing* on the 90-group fixed-charge
#: family (median |ΔΔE| of 0.000–0.001 meV, per-family slope 0.0000) scored a pooled slope of 0.43–0.85, while two
#: models that genuinely respond (per-family slope 0.87 and 1.03) were pulled *down* to 0.72 and 0.77. The pooled
#: value is therefore unusable as the response component in either direction.
DDE_RESPONSE_FAMILY = "wire_disc_near_vs_far"


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
            # every family separately, so a pooled number can never be the only one available
            "ddE_by_family": {k: est(v.get("slope"))
                              for k, v in (((S.get("ddE") or {}).get("by_family")) or {}).items()},
            "ddE_response": est((((S.get("ddE") or {}).get("by_family")) or {})
                                .get(DDE_RESPONSE_FAMILY, {}).get("slope")),
            "ddE_response_family": DDE_RESPONSE_FAMILY,
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
        # the name only: the full spec carries model paths, and these summaries get published
        "backend": (J.get("backend") or {}).get("name", "") if isinstance(J.get("backend"), dict)
        else str(J.get("backend", "")).split(":")[0],
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
    # `lrbench priors --out` writes wherever it is told, and the docs show a bare file name, so look in the
    # conventional subdirectory *and* alongside the eval results rather than silently finding nothing
    prior_paths = sorted(set(glob.glob(os.path.join(runs, "priors", "*.json")))
                         | set(glob.glob(os.path.join(runs, "*.priors.json")))
                         | set(glob.glob(os.path.join(runs, "priors-*.json"))))
    for p in prior_paths:
        try:
            with open(p) as f:
                J = json.load(f)
        except Exception:
            continue
        if "checks" in J:
            priors[os.path.basename(p)[:-5]] = distil_priors(os.path.basename(p)[:-5], J)
    return {"results": results, "priors": priors, "source": os.path.abspath(runs)}


# ----------------------------------------------------------------------------------------- the long-range score
#
# One number, built only from quantities that already have a known-correct value, and never shown without its
# parts. Each component is mapped to [0, 1] where 1 is correct and 0 is the no-long-range floor:
#
#     decay      s1 = clip(tail fraction, 0, 1)        the fraction of the r-dependence reproduced
#     scale      s2 = 1 - min(|eps|, 1)                how much of the physical magnitude is captured
#     response   s3 = clip(ddE slope, 0, 1)            the reply to a charge that is far away
#     accuracy   s4 = 1 - min(E_8A / E_8A(flat), 1)    the offset-free error against a flat model's
#
# LRS = 100 (s1 s2 s3 s4)^(1/4). The *geometric* mean is the point: a model that scores zero on any one of these
# has no long-range physics, and no amount of credit elsewhere should disguise that. An arithmetic mean would give
# a model with no tail at all a respectable 50 for doing well on the other three.
#
# It is a summary of these four tests on these sets, not a ranking of models in general, and it is always printed
# beside the bars it is made of.
COMPONENTS = [("decay", "tail fraction"), ("scale", "|ε|"), ("response", "ΔΔE slope"),
              ("accuracy", "8 Å error vs flat")]


def _clip01(x):
    return None if x is None else max(0.0, min(1.0, float(x)))


def score_components(r: dict) -> dict:
    """The four sub-scores and the long-range score for one distilled result, or Nones where a test did not run."""
    S = r["sets"].get("dimers") or {}
    c = (S.get("classes") or {}).get("ion_pair") or {}
    # The response component is the fixed-charge family alone, never the pooled slope across families. If that
    # family did not run there is no response measurement, and None (which marks the score partial) is the honest
    # answer -- falling back to the pooled value would reinstate exactly the error this avoids.
    fq = r["sets"].get("families_fixedq") or {}
    ddE = fq.get("ddE_response") or (fq.get("ddE_by_family") or {}).get(DDE_RESPONSE_FAMILY)
    tail = (c.get("tail") or [None])[0]
    eps = (c.get("eps") or [None])[0]
    off8 = (c.get("off8") or [None])[0]
    flat8 = c.get("flat8")
    s = {
        "decay": _clip01(tail),
        "scale": None if eps is None else _clip01(1.0 - min(abs(eps), 1.0)),
        "response": _clip01((ddE or [None])[0]),
        "accuracy": None if (off8 is None or not flat8) else _clip01(1.0 - min(off8 / flat8, 1.0)),
    }
    have = [v for v in s.values() if v is not None]
    s["lrs"] = None if len(have) < 2 else round(
        100.0 * (math.prod(have) ** (1.0 / len(have))), 1)
    s["n_components"] = len(have)
    # A geometric mean over three components is not comparable with one over four: a model that was never asked
    # the response question scores as though it had passed it. Consumers must show this, so it travels with the
    # score rather than being left for them to look up.
    s["partial"] = len(have) < len(COMPONENTS)
    return s
