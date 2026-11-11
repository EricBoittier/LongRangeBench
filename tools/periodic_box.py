#!/usr/bin/env python3
"""Total-charge-0 dimer scans in large cubic boxes, for models whose long range needs (or supports) a periodic cell.

    python periodic_box.py make DATA_DIR OUT.extxyz [--L 30,50,80,120] [--orient o0]
    python periodic_box.py analyse OUT.extxyz DATA_DIR OUT.json PRED.json [PRED.json ...]   (label = parent dir name)

make: every canonical dimer record with total charge 0 (ion_pair, neutral_polar_control, neutral_nonpolar_control;
orientation o0) as AB in the open boundary (L = 0) and centred in cubic boxes of each L, plus the two monomers in the
open boundary at their own charges (charged monomers are never put in a box, so no neutralising background is needed).
analyse, per model:
  - periodic-vs-open: dE_pbc(L) = E_box(AB, L) - E_open(AB); for a total-charge-0 system it must vanish as L grows
    (dipolar systems ~ 1/L^3 with tinfoil boundaries); fit dE = c + a/L^3 and c' + b/L: a residual c or a 1/L term
    signals a net monopole leak / non-neutral latent charges;
  - E_int(d, L) = E_box(AB) - E_open(A) - E_open(B) and its L -> inf extrapolation E_inf (fit E = E_inf + a/L^3 over the
    three largest L), compared with the open-boundary E_int and with DFT;
  - offset-free tail on the extrapolated values: dE(d) = E_int(d) - E_int(d_far), against DFT, and the tail fraction
    (slope through 0) per class for d >= 8 A; charge-blind models (no charged monomers) still get dE_pbc and the tail,
    which only needs AB frames.
Element coverage and level mismatches are recorded per model."""
import json
import os
import sys
from collections import defaultdict

import numpy as np


def make(data, out, *a):
    from ase import Atoms
    from ase.io import write
    Ls = [float(x) for x in (a[a.index("--L") + 1] if "--L" in a else "30,50,80,120").split(",")]
    orient = a[a.index("--orient") + 1] if "--orient" in a else "o0"
    F = []
    for l in open(f"{data}/dimers.jsonl"):
        r = json.loads(l)
        if r.get("role") != "AB" or int(r["charge"]) != 0 or f":{orient}:" not in r["id"]:
            continue
        fid = np.asarray(r["fragment_id"]); x = np.asarray(r["positions"]); sp = r["species"]
        base = dict(rid=r["id"], pair=r["extra"]["pair"], cls=r["cls"], d=float(r["distance"]),
                    E_int_DFT_meV=r["refs"]["E_int"] * 1e3)
        xc = x - x.mean(0)
        fq = np.zeros(len(sp)); fq[np.where(fid == 0)[0][0]] = r["fragment_charges"][0]; fq[np.where(fid == 1)[0][0]] = r["fragment_charges"][1]
        for L in [0.0] + Ls:
            at = Atoms(sp, positions=xc + (L / 2 if L else 0))
            if L:
                at.set_cell([L, L, L]); at.set_pbc(True)
            at.arrays["formal_charges"] = fq
            at.info.update(base, role="AB", L=L, charge=0, spin_multiplicity=1)
            F.append(at)
        for k, role in ((0, "A"), (1, "B")):
            m = fid == k
            at = Atoms([s for s, q in zip(sp, m) if q], positions=xc[m])
            f = np.zeros(m.sum()); f[0] = r["fragment_charges"][k]
            at.arrays["formal_charges"] = f
            at.info.update(base, role=role, L=0.0, charge=int(r["fragment_charges"][k]), spin_multiplicity=1)
            F.append(at)
    write(out, F, format="extxyz")
    print(out, len(F), "frames,", len({f.info["rid"] for f in F}), "dimer points")


def fit(L, y, p):
    A = np.c_[np.ones(len(L)), np.asarray(L, float) ** -p]
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1]), float(np.sqrt(np.mean((A @ c - y) ** 2)))


def analyse(xyz, data, out, *preds):
    from ase.io import read
    A = read(xyz, ":"); info = [a.info for a in A]
    res = {"definition": __doc__, "models": {}}
    for p in preds:
        fr = json.load(open(p))["frames"]; lab = os.path.basename(os.path.dirname(p))
        E = defaultdict(dict); Q = {}
        for i, f in zip(info, fr):
            if f.get("energy") is None:
                continue
            E[i["rid"]][(i["role"], float(i["L"]))] = f["energy"] * 1e3
            if f.get("charges") is not None and i["role"] == "AB" and float(i["L"]) == 0:
                Q[i["rid"]] = f["charges"]
        meta = {i["rid"]: i for i in info if i["role"] == "AB"}
        rows = []
        for rid, e in E.items():
            if ("AB", 0.0) not in e:
                continue
            m = meta[rid]; Ls = sorted(L for (r, L) in e if r == "AB" and L > 0)
            row = dict(rid=rid, pair=m["pair"], cls=m["cls"], d=m["d"], E_int_DFT=m["E_int_DFT_meV"],
                       dE_pbc={f"{L:g}": e[("AB", L)] - e[("AB", 0.0)] for L in Ls})
            if len(Ls) >= 3:
                y = np.array([e[("AB", L)] - e[("AB", 0.0)] for L in Ls])
                row["fit_L3"] = fit(Ls, y, 3); row["fit_L1"] = fit(Ls, y, 1)
            if ("A", 0.0) in e and ("B", 0.0) in e:
                mono = e[("A", 0.0)] + e[("B", 0.0)]
                row["E_int_open"] = e[("AB", 0.0)] - mono
                row["E_int_L"] = {f"{L:g}": e[("AB", L)] - mono for L in Ls}
                if len(Ls) >= 3:
                    big = Ls[-3:]
                    row["E_int_inf"] = fit(big, np.array([e[("AB", L)] - mono for L in big]), 3)[0]
            if len(Ls) >= 3:  # monomer-free AB energy at L -> inf (for the offset-free tail)
                big = Ls[-3:]
                row["E_AB_inf"] = fit(big, np.array([e[("AB", L)] for L in big]), 3)[0]
            if rid in Q:
                qa = np.asarray(Q[rid])
                row["q_sum_AB_open"] = float(qa.sum())
            rows.append(row)
        # offset-free tail per scan on the extrapolated AB energies (monomer terms cancel)
        tails = defaultdict(list)
        by = defaultdict(list)
        for r in rows:
            by[r["pair"]].append(r)
        for pair, rs in by.items():
            rs = [r for r in rs if "E_AB_inf" in r]
            if len(rs) < 3:
                continue
            far = max(rs, key=lambda r: r["d"])
            for r in rs:
                if r["d"] >= 8 and r is not far:
                    tails[r["cls"]].append((r["E_AB_inf"] - far["E_AB_inf"], r["E_int_DFT"] - far["E_int_DFT"]))
        tf = {c: dict(n=len(v), tail_fraction=float(np.dot([a for a, _ in v], [b for _, b in v]) / np.dot([b for _, b in v], [b for _, b in v])),
                      eps_median=float(np.median([(a - b) / b for a, b in v if abs(b) > 0.1])) if any(abs(b) > 0.1 for _, b in v) else None)
              for c, v in tails.items() if v}
        c3 = [r["fit_L3"][0] for r in rows if "fit_L3" in r]; b1 = [r["fit_L1"][1] for r in rows if "fit_L1" in r]
        dl = defaultdict(list)
        for r in rows:
            for L, v in r["dE_pbc"].items():
                dl[L].append(abs(v))
        res["models"][lab] = {
            "n_points": len(rows), "n_with_monomers": sum("E_int_open" in r for r in rows),
            "pairs": sorted({r["pair"] for r in rows}),
            "median_abs_dE_pbc_meV_by_L": {L: float(np.median(v)) for L, v in sorted(dl.items(), key=lambda t: float(t[0]))},
            "median_abs_c_inf_L3_meV": float(np.median(np.abs(c3))) if c3 else None,
            "median_abs_b_1overL_meV_A": float(np.median(np.abs(b1))) if b1 else None,
            "tail_by_class_extrapolated": tf, "rows": rows}
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m, v in res["models"].items():
        print(f"{m:14s} n={v['n_points']:3d} |dE_pbc| median by L {({k: round(x, 2) for k, x in v['median_abs_dE_pbc_meV_by_L'].items()})} "
              f"c_inf {v['median_abs_c_inf_L3_meV']} b(1/L) {v['median_abs_b_1overL_meV_A']} tails {({c: (t['n'], round(t['tail_fraction'], 3)) for c, t in v['tail_by_class_extrapolated'].items()})}")


if __name__ == "__main__":
    {"make": make, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
