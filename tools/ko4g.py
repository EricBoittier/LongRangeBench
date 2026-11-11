#!/usr/bin/env python3
"""Ko et al. 2021 4G-HDNNP subsample (the Space's ext_ko4g, 980 structures) through `lrbench predict`.

    python ko4g.py export ext_ko4g.json.gz ko4g.extxyz
    python ko4g.py analyse ext_ko4g.json.gz OUT.json PRED.json [PRED.json ...]   (label = parent dir name)

Reference level PBE (FHI-aims, light), periodic AuMgO slabs, every structure closed-shell (spin 1) at its total
charge. The Space stores only E_rel (energy relative to the lowest kept structure of the same system/charge/size) and
per-atom |F|, so the scores are: MAE of E_rel (model energies relative to the model's value on the same reference
structure, i.e. composition-constant differences: element offsets and the level's absolute shift cancel) and MAE of
per-atom |F| (a norm comparison, weaker than a vector force error). Level mismatch (PBE vs each model's level) stays in
every number."""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))


def export(src, out):
    from ase import Atoms
    from ase.io import write
    d = load(src); P = d["properties"]; A = []
    for i, s in enumerate(d["structures"]):
        a = Atoms(s["names"], positions=np.c_[s["x"], s["y"], s["z"]])
        if s.get("cell"):
            a.cell = np.reshape(s["cell"], (3, 3)); a.pbc = True
        a.info.update(charge=int(round(P["charge"]["values"][i])), spin_multiplicity=1, id=P["id"]["values"][i],
                      cls=P["class"]["values"][i])
        A.append(a)
    write(out, A, format="extxyz")
    print(out, len(A))


def analyse(src, out, *preds):
    d = load(src); P = d["properties"]; n = len(d["structures"])
    sizes = [s["size"] for s in d["structures"]]
    cls, erel = P["class"]["values"], np.array(P["E_rel/REF"]["values"], float)
    fn = np.array(P["F_norm/REF"]["values"], float); off = np.concatenate([[0], np.cumsum(sizes)])
    key = [(c, sizes[i]) for i, c in enumerate(cls)]
    res = {"definition": __doc__, "level_ref": "PBE (FHI-aims, light)", "models": {}}
    for p in preds:
        fr = json.load(open(p))["frames"]; lab = os.path.basename(os.path.dirname(p))
        E = np.array([np.nan if f.get("energy") is None else f["energy"] for f in fr])
        anchor = {}
        for i in range(n):  # model energy on the structure the reference uses as its zero
            if erel[i] == 0 and np.isfinite(E[i]):
                anchor[key[i]] = E[i]
        by = defaultdict(lambda: {"dE": [], "dF": []})
        for i in range(n):
            if key[i] not in anchor or not np.isfinite(E[i]):
                continue
            by[cls[i]]["dE"].append(abs((E[i] - anchor[key[i]]) * 1e3 - erel[i]))  # E_rel stored in meV
            if fr[i].get("forces"):
                by[cls[i]]["dF"] += list(np.abs(np.linalg.norm(fr[i]["forces"], axis=1) - fn[off[i]:off[i + 1]]))
        res["models"][lab] = {c: {"n": len(v["dE"]), "E_rel_MAE_meV": float(np.mean(v["dE"])) if v["dE"] else None,
                                  "Fnorm_MAE_eV_A": float(np.mean(v["dF"])) if v["dF"] else None}
                              for c, v in sorted(by.items())}
        res["models"][lab]["n_failed"] = int(np.sum(~np.isfinite(E)))
    json.dump(res, open(out, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    {"export": export, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
