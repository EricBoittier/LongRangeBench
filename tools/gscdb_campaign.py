#!/usr/bin/env python3
"""Calculation list and cost estimate for a GSCDB137 ωB97M-V/def2-TZVPD label campaign (OMol25 ORCA protocol).

    python -I gscdb_campaign.py MANIFEST.json TIMINGS.jsonl[,MORE.jsonl] OUTDIR [--chf-per-core-h 0.0056] [--packing 0.62]

TIMINGS: parse_pilot.py outputs (n_basis, core_h per point; lr-datagen pilot + GSCDB smoke). Fit: log core_h =
a + b log nbf (least squares), with a 90 % prediction band from the residual quantiles; allocated core-h = CPU core-h /
packing efficiency (the pilot packed 62 %). Writes OUTDIR/calcs_<campaign>.jsonl and OUTDIR/cost.json."""
import json
import os
import sys

import numpy as np

man = json.load(open(sys.argv[1])); tim = sys.argv[2].split(","); O = sys.argv[3]; os.makedirs(O, exist_ok=True)
arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
CHF, PACK = arg("--chf-per-core-h", 0.0056), arg("--packing", 0.62)

x, y, src = [], [], []
for p in tim:
    for line in open(p):
        r = json.loads(line)
        if r.get("n_basis") and r.get("core_h") and str(r.get("status", "")).startswith("done"):
            x.append(r["n_basis"]); y.append(r["core_h"]); src.append(os.path.basename(p))
x, y = np.log(np.array(x, float)), np.log(np.array(y, float))
b, a = np.polyfit(x, y, 1); res = y - (a + b * x); q05, q95 = np.quantile(res, [0.05, 0.95])
pred = lambda n: np.exp(a + b * np.log(n)) * np.exp(np.mean(res ** 2) / 2)   # mean, log-normal correction

CAMPAIGNS = {
    "P0": [(S, None) for S in ("IHB100x2", "AHB21", "CHB6", "IL16", "S66Rel7", "3BHET")],
    "NBC10_8-10A": [("NBC10", {"NBC10_71", "NBC10_89", "NBC10_183", "NBC10_184"})],
    "NBC10_all": [("NBC10", None)],
}
done_keys = set()
for p in tim:
    for line in open(p):
        r = json.loads(line)
        k = (r.get("meta") or {}).get("gscdb_key")
        if k and str(r.get("status", "")).startswith("done"):
            done_keys.add(k)
out = dict(fit=dict(form="core_h = exp(a) * nbf^b (CPU core-h per serial point)", a=float(a), b=float(b),
                    n_points=int(len(x)), sources=sorted(set(src)), resid_q05=float(q05), resid_q95=float(q95),
                    max_nbf_in_fit=int(np.exp(x).max().round())),
           packing=PACK, chf_per_core_h=CHF, campaigns={})
for name, parts in CAMPAIGNS.items():
    keys = []
    for S, ids in parts:
        for e in man["subsets"][S]["entries"]:
            if ids is None or e["id"] in ids:
                for _, k in e["terms"]:
                    if k not in keys:
                        keys.append(k)
    nb = np.array([man["molecules"][k]["nbf_tzvpd"] for k in keys], float)
    todo = [k for k in keys if k not in done_keys]
    nbt = np.array([man["molecules"][k]["nbf_tzvpd"] for k in todo], float)
    cpu = float(pred(nbt).sum()) if len(nbt) else 0.0
    lo, hi = float(np.exp(a + b * np.log(nbt) + q05).sum()), float(np.exp(a + b * np.log(nbt) + q95).sum())
    extrap = int((nbt > out["fit"]["max_nbf_in_fit"]).sum())
    with open(f"{O}/calcs_{name}.jsonl", "w") as f:
        for k in keys:
            m = man["molecules"][k]
            f.write(json.dumps(dict(key=k, sets=m["sets"], n_atoms=m["n_atoms"], charge=m["charge"], mult=m["mult"],
                                    nbf_tzvpd=m["nbf_tzvpd"], role="complex" if any(
                                        k == e["complex"] for S, _ in parts for e in man["subsets"][S]["entries"])
                                    else "fragment/term", done_in_smoke=k in done_keys,
                                    est_core_h=float(pred(m["nbf_tzvpd"])))) + "\n")
    out["campaigns"][name] = dict(n_calcs=len(keys), n_todo=len(todo), n_done_in_smoke=len(keys) - len(todo),
                                  nbf_median=float(np.median(nb)), nbf_max=int(nb.max()),
                                  n_beyond_fit_range=extrap, cpu_core_h=cpu, cpu_core_h_band_5_95=[lo, hi],
                                  allocated_core_h=cpu / PACK, chf=cpu / PACK * CHF,
                                  largest_point_core_h=float(pred(nbt.max())) if len(nbt) else 0.0)
json.dump(out, open(f"{O}/cost.json", "w"), indent=1)
for k, v in out["campaigns"].items():
    print(f"{k}: {v['n_calcs']} calcs ({v['n_todo']} to do), nbf median {v['nbf_median']:.0f} max {v['nbf_max']}, "
          f"CPU {v['cpu_core_h']:.0f} core-h [5-95 % sum {v['cpu_core_h_band_5_95'][0]:.0f}-{v['cpu_core_h_band_5_95'][1]:.0f}], "
          f"allocated {v['allocated_core_h']:.0f} core-h = {v['chf']:.1f} CHF; {v['n_beyond_fit_range']} beyond fit range")
print("fit", out["fit"])
