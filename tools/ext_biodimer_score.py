#!/usr/bin/env python3
"""Offset-free scores on the biodimer scans (ext_biodimers) from `lrbench eval --save-predictions` output.

    python ext_biodimer_score.py DATA_DIR OUT.json RESULT.json [RESULT.json ...]

The source E_int of charged-charged curves carries periodic-cell terms (see ext_biodimers.py), so only differences
along a curve are scored: dE(d) = E_int(d) - E_int(d_last), d_last = the curve's longest point
(closest contact ~12-13 A). Per class (ion_pair, like_charge, ion_neutral, neutral) and closest-contact band:
- tail fraction: slope through 0 of model dE on reference dE over points with d >= 8 A (lrbench prep_fig rule);
- offset-free MAE in bands 4-8 A and >= 8 A, with the flat-model MAE (|dE_ref|) for scale;
95 % intervals by bootstrap over curves. Reference: HSE06 (75 % HF) + MBD-NL; models: wB97M-V emulators, so the
level mismatch is part of every number (shapes beyond ~8 A are dominated by electrostatics, which the mismatch
affects through the fragment multipoles only).

Box correction (default on; --no-box-correction to disable): the source energies are periodic in a 30 A cell, monomers
included. For like-charge curves (cell charge +-2) the source E_int - K q_a q_b / r follows the monopole cell term
-K xi / L + 2 pi K r^2 / (3 V) (fit: intercept -1.39 eV, slope 1.09e-3 eV/A^2 vs theory -1.362, 1.12e-3), so the
r^2 term (0.16 eV across 9-15 A) is removed from the reference with r = heavy-atom centroid distance. Ion pairs
(neutral cell): measured slope 1.4e-4 eV/A^2 (theory for tin-foil -1.12e-3) and shapes follow open boundary to
25 meV, so no correction; the constant Madelung part cancels in every offset-free difference.
"""
import collections
import json
import os
import sys

import numpy as np


def boot(f, groups, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = sorted(set(groups)); idx = {g: np.where(groups == g)[0] for g in u}
    v = [f(np.concatenate([idx[g] for g in rng.choice(u, len(u))])) for _ in range(nb)]
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


BOX_SLOPE = 2 * np.pi * 14.399645 / (3 * 30.0 ** 3)  # eV / A^2, monopole cell term of a 30 A cubic cell


def box_term(r):
    """r^2 part of the monopole cell term for a like-charge pair (cell charge +-2); 0 for other classes."""
    if r["cls"] != "like_charge":
        return 0.0
    P = np.array(r["positions"]); f = np.array(r["fragment_id"]); h = np.array([s != "H" for s in r["species"]])
    d = np.linalg.norm(P[(f == 1) & h].mean(0) - P[(f == 0) & h].mean(0))
    return r["fragment_charges"][0] * r["fragment_charges"][1] * BOX_SLOPE * d * d


def main(data, out, *results):
    corr = "--no-box-correction" not in results
    results = [r for r in results if not r.startswith("--")]
    recs = [json.loads(l) for l in open(os.path.join(data, "canonical", "ext_biodimers.jsonl"))]
    bterm = {r["id"]: box_term(r) if corr else 0.0 for r in recs}
    res = {}
    for p in results:
        d = json.load(open(p)); P = d["sets"]["ext_biodimers"]["predictions"]
        E = lambda t: float(np.mean(P[t]))
        by = collections.defaultdict(list)
        for r in recs:
            try:
                m = E(r["id"]) - E(r["id"] + "#frag=0") - E(r["id"] + "#frag=1")
            except (KeyError, TypeError):
                continue
            by[r["group"]].append((r["distance"], r["extra"]["E_int_source_eV"] - bterm[r["id"]], m, r["cls"]))
        rows = []
        for g, v in by.items():
            v.sort(); dl, rl, ml, c = v[-1]
            rows += [(g, c, dd, 1e3 * (rr - rl), 1e3 * (mm - ml)) for dd, rr, mm, _ in v[:-1]]
        G = np.array([r[0] for r in rows]); C = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows])
        X = np.array([r[3] for r in rows]); Y = np.array([r[4] for r in rows])
        tab = {}
        for c in sorted(set(C.tolist())):
            t = {}
            k = np.where((C == c) & (D >= 8))[0]
            if len(k) and np.isfinite(Y[k]).all():
                sl = lambda i: float((X[k][i] @ Y[k][i]) / (X[k][i] @ X[k][i]))
                t["tail_fraction"] = dict(value=round(sl(np.arange(len(k))), 4), ci=boot(sl, G[k]),
                                          n_points=int(len(k)), n_curves=int(len(set(G[k]))),
                                          ref_span_meV=round(float(np.abs(X[k]).max()), 1))
            for lab, lo, hi in (("4-8", 4, 8), (">=8", 8, 99)):
                k = np.where((C == c) & (D >= lo) & (D < hi) & np.isfinite(Y))[0]
                if len(k):
                    f = lambda i: float(np.mean(np.abs(Y[k][i] - X[k][i])))
                    t[f"offset_free_mae_{lab}"] = dict(meV=round(f(np.arange(len(k))), 2), ci=boot(f, G[k]),
                                                       flat_model_meV=round(float(np.mean(np.abs(X[k]))), 2),
                                                       n=int(len(k)))
            tab[c] = t
        res[d["label"]] = dict(box_correction=corr, member=d["members"][0]["path"], sha256=d["members"][0].get("sha256"), classes=tab,
                               n_curves=len(by))
    json.dump(res, open(out, "w"), indent=1)
    for m, v in res.items():
        for c, t in v["classes"].items():
            tf = t.get("tail_fraction", {}); a = t.get("offset_free_mae_4-8", {}); b = t.get("offset_free_mae_>=8", {})
            print(f"| {m} | {c} | {tf.get('value')} {tf.get('ci')} | {a.get('meV')} {a.get('ci')} (flat {a.get('flat_model_meV')}) | "
                  f"{b.get('meV')} {b.get('ci')} (flat {b.get('flat_model_meV')}) |")


if __name__ == "__main__":
    main(*sys.argv[1:])
