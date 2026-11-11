#!/usr/bin/env python3
"""Trimer three-body term: models vs DFT vs the multipole oracle's three-body induction (lrbench.oracle.trimer_three_body).

    python trimer3body.py DATA_DIR FRAGS.json OUT.json RESULT.json [RESULT.json ...]   (eval --save-predictions outputs)

Per trimer system: E3 (DFT group ref), E3_ind (oracle: DFT fragment moments + polarisabilities, no back-polarisation),
model E3 and the geometry-relative dE3 vs the family's far geometry (dE3_vs_far groups; removes the charge-state offset that
dominates absolute E3 for total-charge-conditioned models). The oracle is illustrative (fragment information the models
do not get). Per model: dE3 slope through 0 against DFT and against the oracle, sign agreement, and |model| / |oracle|."""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")
from lrbench.data import load_set  # noqa: E402
from lrbench.oracle import load_frags, trimer_three_body  # noqa: E402


def gval(g, P):
    try:
        return sum(c * np.mean(P[t]) for t, c in g["terms"]) * 1e3
    except (KeyError, TypeError):
        return None


def main(data, frags, out, *results):
    ds = load_set("trimers", data)
    orc = {r["system"]: r["E3_ind_meV"] for r in trimer_three_body(ds, load_frags(frags))}
    E3 = {g["meta"]["system"]: g for g in ds.groups if g["kind"] == "E3"}
    dE = [g for g in ds.groups if g["kind"] == "dE"]
    # oracle and DFT dE3 vs far: same arithmetic on E3 values (the dE groups are E3(sys) - E3(far))
    far = {}
    for g in dE:
        sysn = g["meta"].get("system") or g["name"].split(":", 1)[1]
        far[sysn] = g
    res = {"definition": __doc__, "systems": {}, "models": {}}
    for s, g in E3.items():
        res["systems"][s] = {"E3_DFT_meV": g["ref"] * 1e3, "E3_ind_oracle_meV": orc.get(s)}
    for s, g in far.items():
        res["systems"].setdefault(s, {})["dE3_vs_far_DFT_meV"] = g["ref"] * 1e3 if g.get("ref") is not None else None
        fs = g["meta"].get("ref_system")
        if fs and orc.get(s) is not None and orc.get(fs) is not None:
            res["systems"][s]["dE3_vs_far_oracle_meV"] = orc[s] - orc[fs]
    for p in results:
        R = json.load(open(p)); P = R["sets"].get("trimers", {}).get("predictions")
        if not P:
            continue
        rows = {}
        for s, g in E3.items():
            rows[s] = {"E3": gval(g, P)}
        for s, g in far.items():
            rows.setdefault(s, {})["dE3_vs_far"] = gval(g, P)
        x = [(v.get("dE3_vs_far"), res["systems"][s].get("dE3_vs_far_DFT_meV"), res["systems"][s].get("dE3_vs_far_oracle_meV"))
             for s, v in rows.items()]
        x = [t for t in x if None not in t[:2]]
        m = np.array([t[0] for t in x]); d = np.array([t[1] for t in x])
        o = np.array([t[2] for t in x if t[2] is not None]); mo = np.array([t[0] for t in x if t[2] is not None])
        res["models"][R.get("label_short", R["label"])] = {
            "label": R["label"], "rows": rows, "n": len(x),
            "slope_vs_DFT": float(m @ d / (d @ d)) if len(d) else None,
            "slope_vs_oracle": float(mo @ o / (o @ o)) if len(o) else None,
            "sign_agree_DFT": float(np.mean(np.sign(m) == np.sign(d))) if len(d) else None,
            "median_abs_ratio_to_DFT": float(np.median(np.abs(m) / np.maximum(np.abs(d), 1e-3))) if len(d) else None}
    json.dump(res, open(out, "w"), indent=1)
    print({s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for s, r in res["systems"].items()})
    for k, v in res["models"].items():
        print(f"{k:14s} n={v['n']} slope_vs_DFT={v['slope_vs_DFT'] and round(v['slope_vs_DFT'], 2)} "
              f"slope_vs_oracle={v['slope_vs_oracle'] and round(v['slope_vs_oracle'], 2)} sign={v['sign_agree_DFT']} "
              f"|m|/|DFT|={v['median_abs_ratio_to_DFT'] and round(v['median_abs_ratio_to_DFT'], 1)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
