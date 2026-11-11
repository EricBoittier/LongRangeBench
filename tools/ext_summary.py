#!/usr/bin/env python3
"""Per-subset / category / closest-contact-bin tables for the ext_* sets from `lrbench eval --save-predictions`.

    python ext_summary.py DATA_DIR SET OUT.json RESULT.json [RESULT.json ...]

Every group value = sum coeff * E_model(term) (same stoichiometry as the reference). MAE with a 95 % bootstrap interval
over bootstrap groups (curves). Reference level is the set's own (usually CCSD(T)/CBS); the models are OMol25
(wB97M-V/def2-TZVPD) emulators, so every number includes the functional error: an end-to-end score, not an ML error.
"""
import collections
import json
import os
import sys

import numpy as np

KCAL = 23.060548  # kcal/mol per eV
BINS = [(0, 4, "<4"), (4, 8, "4-8"), (8, 12, "8-12"), (12, 1e9, ">=12")]


def dbin(d):
    return "n/a" if d is None else next(l for lo, hi, l in BINS if lo <= d < hi)


def boot(err, grp, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = sorted(set(grp)); idx = {g: np.where(np.array(grp) == g)[0] for g in u}
    if len(u) < 3:
        return None
    s = [np.mean(np.concatenate([err[idx[g]] for g in rng.choice(u, len(u))])) for _ in range(nb)]
    return [round(float(np.percentile(s, 2.5)), 2), round(float(np.percentile(s, 97.5)), 2)]


def table(rows, key):
    out = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    for k, rs in sorted(by.items()):
        e = np.array([r["err"] for r in rs]); g = [r["grp"] for r in rs]
        out[k] = dict(n=len(rs), n_curves=len(set(g)), mae_meV=round(float(np.abs(e).mean()), 2),
                      ci=boot(np.abs(e), g), mse_meV=round(float(e.mean()), 2),
                      ref_mean_abs_meV=round(float(np.mean([abs(r["ref"]) for r in rs])), 2))
    return out


def main(data, name, out, *results):
    groups = json.load(open(os.path.join(data, "canonical", f"{name}.groups.json")))
    res = {}
    for p in results:
        d = json.load(open(p)); s = d["sets"][name]; P = s["predictions"]
        rows = []
        for g in groups:
            try:
                v = sum(c * np.mean(P[t]) for t, c in g["terms"])
            except (KeyError, TypeError):
                continue
            if not np.isfinite(v):
                continue
            m = g["meta"]
            rows.append(dict(name=g["name"], subset=m.get("subset"), cat=m.get("category"), kind=g["kind"],
                             d=m.get("distance") if g["kind"] == "E_int" else m.get("d_min"),
                             atomic=bool(m.get("atomic_ion_fragment")), before=bool(m.get("scored_before")),
                             grp=m.get("bootstrap_group", g["name"]), ref=1e3 * g["ref"], model=1e3 * v,
                             err=1e3 * (v - g["ref"]), flags=m.get("flags") or []))
        res[d["label"]] = dict(
            n=len(rows), n_groups=len(groups), level_note=("reference wB97M-V (level-matched ORCA relabel; isolates model error)"
                                              if "wb97mv" in name else "reference CCSD(T)-quality vs wB97M-V-trained model "
                                              "(end-to-end, includes functional error)"), member=d["members"][0]["path"],
            sha256=d["members"][0].get("sha256"),
            by_subset=table(rows, lambda r: r["subset"]),
            by_cat_bin=table([r for r in rows if r["kind"] == "E_int"],
                             lambda r: f"{r['cat']}|{dbin(r['d'])}|{'atomic-ion' if r['atomic'] else 'molecular'}"),
            by_bin=table(rows, lambda r: dbin(r["d"])),
            far=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows
                 if r["d"] is not None and r["d"] >= 8])
    json.dump(res, open(out, "w"), indent=1)
    labs = list(res)
    print("| subset | n | " + " | ".join(f"{l} MAE meV [95 % CI]" for l in labs) + " |")
    print("|---|---|" + "---|" * len(labs))
    for S in sorted(set().union(*[res[l]["by_subset"] for l in labs])):
        cells = []
        for l in labs:
            t = res[l]["by_subset"].get(S)
            cells.append("–" if not t else f"{t['mae_meV']:.1f}" + (f" [{t['ci'][0]:.1f}, {t['ci'][1]:.1f}]" if t["ci"] else ""))
        n = next(res[l]["by_subset"][S]["n"] for l in labs if S in res[l]["by_subset"])
        print(f"| {S} | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
