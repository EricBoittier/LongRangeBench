#!/usr/bin/env python3
"""Summary of `lrbench eval --sets omol25_val_lr --save-predictions` results: per model, absolute energy MAE (meV/atom,
lrbench's in-sample per-element offsets) by subset x charge sign and by separation-scale bin, force MAE by subset (from the
eval block), and the fixed-composition dE = E(compressed) - E(expanded) of the same cluster by compressed-scale bin and
charge (offsets cancel). All references ωB97M-V/def2-TZVPD (OMol25 validation; unseen by every model here).

    python omolval_summary.py DATA_DIR OUT.json RESULT.json [RESULT.json ...]"""
import json
import sys
from collections import defaultdict

import numpy as np


def sbin(s):
    return "1.0 (unscaled)" if s == 1.0 else f"{np.floor(s * 10) / 10:.1f}-{np.floor(s * 10) / 10 + 0.1:.1f}"


def main(data, out, *results):
    R0 = [json.loads(l) for l in open(f"{data}/omol25_val_lr.jsonl")]
    rec = {r["id"]: r for r in R0}
    G = json.load(open(f"{data}/omol25_val_lr.groups.json"))
    res = {"definition": __doc__, "models": {}}
    for p in results:
        R = json.load(open(p)); S = R["sets"]["omol25_val_lr"]; P = S.get("predictions") or {}
        E = {k: np.nanmean([x for x in v if x is not None]) if any(x is not None for x in v) else np.nan for k, v in P.items()}
        by = defaultdict(list); byq = defaultdict(list)
        for i, r in rec.items():
            if not np.isfinite(E.get(i, np.nan)):
                continue
            e = abs(E[i] - r["energy"]) / len(r["species"]) * 1e3
            by[(r["cls"], sbin(r["distance"]))].append(e)
            byq[(r["cls"], int(np.sign(r["charge"])))].append(e)
        dE = defaultdict(list)
        for g in G:
            v = [E.get(t, np.nan) for t, _ in g["terms"]]
            if not all(np.isfinite(v)):
                continue
            m = (v[0] - v[1]) * 1e3; ref = g["ref"] * 1e3
            dE[(g["meta"]["family"], int(np.sign(g["meta"]["charge"])))].append((m, ref))
        lab = R.get("label_short", R.get("label"))
        res["models"][lab] = {
            "label": R.get("label"), "level": R.get("level", {}).get("model_level"),
            "n_failed": sum(1 for i in rec if not np.isfinite(E.get(i, np.nan))),
            "E_mae_meV_atom_by_subset_scale": {f"{c}|{b}": [len(v), float(np.mean(v))] for (c, b), v in sorted(by.items())},
            "E_mae_meV_atom_by_subset_charge": {f"{c}|q{q:+d}": [len(v), float(np.mean(v))] for (c, q), v in sorted(byq.items())},
            "F_mae_meV_A_by_subset": {x["cls"]: x["F_mae_meV_A"] for x in S["energy_force"]["by_class_distance"]},
            "dE_compressed_minus_expanded": {f"{f}|q{q:+d}": dict(n=len(v), MAE_meV=float(np.mean([abs(a - b) for a, b in v])),
                                                                 MAE_rel=float(np.median([abs(a - b) / max(abs(b), 1) for a, b in v])),
                                                                 slope=float(np.dot([a for a, _ in v], [b for _, b in v]) / np.dot([b for _, b in v], [b for _, b in v])))
                                             for (f, q), v in sorted(dE.items())}}
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m, v in res["models"].items():
        e = v["E_mae_meV_atom_by_subset_charge"]; d = v["dE_compressed_minus_expanded"]
        print(f"{m:14s} fail {v['n_failed']:4d} | E/atom 5A {np.mean([x[1] for k, x in e.items() if k.startswith('5A')]):9.2f} "
              f"| dE 0.7-0.8 q0 {d.get('compressed 0.7-0.8|q+0', {}).get('MAE_meV', float('nan')):9.1f} "
              f"q+ {d.get('compressed 0.7-0.8|q+1', {}).get('MAE_meV', float('nan')):9.1f} "
              f"q- {d.get('compressed 0.7-0.8|q-1', {}).get('MAE_meV', float('nan')):9.1f} meV")


if __name__ == "__main__":
    main(*sys.argv[1:])
