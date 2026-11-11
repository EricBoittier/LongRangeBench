#!/usr/bin/env python3
"""Illustrative: multipole / charge oracles on EVERY dimer point (contact to 25 A), alone and added to a local model.
The oracles use fragment information (integer fragment charges, the fragments' own DFT moments and polarisabilities)
that the models do not get; inside ~8 A pure electrostatics misses penetration, exchange and charge transfer, which is
what the short-range bins show.

    python oracle_allpoints.py DATA_DIR OUT.json LOCAL.json ORACLE.json [ORACLE.json ...]

LOCAL.json / ORACLE.json: `lrbench eval --save-predictions` outputs on the dimers set (oracles with rcut=0, so they
are defined at every point). "local + oracle" = E_int(local model) + E_int(oracle), point by point;
"local + sw.oracle" switches the oracle on with a smootherstep in d_min from 6 to 8 A (as the zero-shot combined
oracle's inter-cluster Coulomb), so the local model keeps the short range; "local-off + sw.oracle" also switches off the
local model's own disconnected constant (its E_int at the scan's farthest point, the charge-state offset) over the same
6->8 A window, i.e. what a cluster-conditioned local model plus oracle Coulomb would give (cf. petlr-combined zs E8c+C). Reported per class
x d_min bin (<4, 4-8, 8-12, >=12 A): MAE and median |E_int error| (meV); per-point rows for full curves."""
import json
import sys
from collections import defaultdict

import numpy as np

BINS = [(0, 4, "<4"), (4, 8, "4-8"), (8, 12, "8-12"), (12, 1e9, ">=12")]


def eint(P, rid):
    try:
        v = [np.mean([x for x in P[t] if x is not None]) for t in (rid, f"{rid}#frag=0", f"{rid}#frag=1")]
    except (KeyError, ZeroDivisionError):
        return None
    return None if any(np.isnan(x) for x in v) else (v[0] - v[1] - v[2]) * 1e3


def main(data, out, local, *oracles):
    recs = [json.loads(l) for l in open(f"{data}/dimers.jsonl")]
    groups = {g["terms"][0][0]: g for g in json.load(open(f"{data}/dimers.groups.json")) if g["kind"] == "E_int"}
    L = json.load(open(local)); PL = L["sets"]["dimers"]["predictions"]
    O = [(json.load(open(p))) for p in oracles]
    labels = {"local": L["label"]} | {f"oracle{i}": o["label"] for i, o in enumerate(O)}
    rows = []
    for r in recs:
        if r.get("role") != "AB" or r["id"] not in groups or "dft_pathology_do_not_score" in (r.get("flags") or []):
            continue
        ref = groups[r["id"]]["ref"] * 1e3
        row = dict(id=r["id"], cls=r["cls"], d=float(r["distance"]), E_int_DFT=ref, local=eint(PL, r["id"]))
        for i, o in enumerate(O):
            e = eint(o["sets"]["dimers"]["predictions"], r["id"])
            row[f"oracle{i}"] = e
            row[f"local+oracle{i}"] = None if e is None or row["local"] is None else row["local"] + e
            t = min(max((row["d"] - 6.0) / 2.0, 0.0), 1.0); sw = t**3 * (t * (6 * t - 15) + 10)  # smootherstep 6->8 A
            row[f"local+sw.oracle{i}"] = None if e is None or row["local"] is None else row["local"] + sw * e
        row["scan"] = r.get("group")
        rows.append(row)
    far = {}
    for x in rows:  # local model's disconnected constant per scan = its E_int at the scan's farthest point
        if x["local"] is not None and (x["scan"] not in far or x["d"] > far[x["scan"]][0]):
            far[x["scan"]] = (x["d"], x["local"])
    for x in rows:
        for i in range(len(O)):
            e = x[f"oracle{i}"]
            t = min(max((x["d"] - 6.0) / 2.0, 0.0), 1.0); sw = t**3 * (t * (6 * t - 15) + 10)
            x[f"local-off+sw.oracle{i}"] = (None if e is None or x["local"] is None else
                                            x["local"] - sw * far[x["scan"]][1] + sw * e)
    keys = ["local"] + [k for i in range(len(O)) for k in (f"oracle{i}", f"local+oracle{i}", f"local+sw.oracle{i}",
                                                             f"local-off+sw.oracle{i}")]
    summ = defaultdict(dict)
    for c in sorted({x["cls"] for x in rows}) + ["all"]:
        for lo, hi, b in BINS:
            sel = [x for x in rows if (c == "all" or x["cls"] == c) and lo <= x["d"] < hi]
            if not sel:
                continue
            s = {"n": len(sel), "DFT_median_abs_E_int": float(np.median([abs(x["E_int_DFT"]) for x in sel]))}
            for k in keys:
                e = [abs(x[k] - x["E_int_DFT"]) for x in sel if x[k] is not None]
                s[k] = {"n": len(e), "MAE": float(np.mean(e)) if e else None, "median": float(np.median(e)) if e else None}
            summ[c][b] = s
    json.dump({"note": __doc__, "labels": labels, "summary": summ, "rows": rows}, open(out, "w"), indent=1)
    print("| region | n | " + " | ".join(labels.get(k, k.replace("local-off+sw.", labels["local"] + " offset-free + switched ").replace("local+sw.", labels["local"] + " + switched ").replace("local+", labels["local"] + " + ")) for k in keys) + " |")
    for b in [b for *_, b in BINS]:
        s = summ["all"].get(b)
        if s:
            print(f"| {b} A | {s['n']} | " + " | ".join(f"{s[k]['median']:.1f}" if s[k]["median"] is not None else "–"
                                                  for k in keys) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
