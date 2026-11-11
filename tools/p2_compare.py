#!/usr/bin/env python3
"""Models vs the P2 ORCA references on the same geometries (scans/p2_dft.extxyz, frames AB, A, B per calc):
E_int(d) = E(AB) - E(A) - E(B) for Ca2+-H2O (|q| = 2, ionic), Mg2+-H2O (singlet and triplet total-charge-2 inputs;
DFT's lowest state is the Mg+ + H2O+ charge-transfer state, so this scan is a CT test, not a q^2 test) and CH3F...I-
(size consistency, 8-100 A). Latent / model charges summed on fragment A where the model returns charges.

    python p2_compare.py P2_DFT.json P2.extxyz OUT.json PRED_DIR [PRED_DIR ...]   (each PRED_DIR/<model>/p2_dft.json)"""
import glob
import json
import os
import sys

import numpy as np


def main(dft, xyz, out, *dirs):
    from ase.io import read
    D = {(x["sys"], x["state"], x["d"]): x for x in json.load(open(dft))["scans"]}
    A = read(xyz, ":")
    res = {"definition": __doc__, "dft": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in D.items()}, "models": {}}
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/*/p2_dft.json")):
            fr = json.load(open(p))["frames"]; m = os.path.basename(os.path.dirname(p))
            by = {}
            for a, f in zip(A, fr):
                i = a.info
                by.setdefault((i["sys"], i["state"], i["d"]), {})[i["role"]] = (f.get("energy"), f.get("charges"), len(a))
            rows = []
            for k, v in sorted(by.items()):
                if any(v.get(r, (None,))[0] is None for r in ("AB", "A", "B")):
                    continue
                e = (v["AB"][0] - v["A"][0] - v["B"][0]) * 1e3
                q = None if v["AB"][1] is None else float(np.sum(v["AB"][1][:v["A"][2]]))
                ref = D.get(k, {})
                rows.append(dict(sys=k[0], state=k[1], d=k[2], E_int_meV=e, qA_model=q, E_int_DFT=ref.get("E_int_meV"),
                                 qA_DFT=ref.get("qA_lowdin"), err=None if ref.get("E_int_meV") is None else e - ref["E_int_meV"]))
            res["models"][m] = rows
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m, rows in res["models"].items():
        s = {(r["sys"], r["state"], r["d"]): r for r in rows}
        f = lambda k: None if k not in s else round(s[k]["E_int_meV"], 1)
        print(f"{m:14s} Ca8 {f(('Ca_H2O','singlet',8))} Ca24 {f(('Ca_H2O','singlet',24))} Mg8s {f(('Mg_H2O','singlet',8))} "
              f"Mg24t {f(('Mg_H2O','triplet',24))} I32 {f(('CH3F_I','singlet',32))} I100 {f(('CH3F_I','singlet',100))}")


if __name__ == "__main__":
    main(*sys.argv[1:])
