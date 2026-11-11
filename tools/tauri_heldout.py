#!/usr/bin/env python3
"""SO3LR-Tauri held-out PBE0+MBD-NL test splits (test_ion = qcell_ions, test_des = qcell_des400k) as a physics check:
force MAE (eV/A) by total charge, and energy MAE after a per-element offset fit (meV/atom, in sample). The labels are
PBE0+MBD(-NL): level-matched only for SO3LR-T; every other model carries the level gap (forces are the fairer number).

    python tauri_heldout.py OUT.json TEST_DIR PRED_DIR [PRED_DIR ...]     (PRED_DIR/<model>/test_{ion,des}.json)"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(out, tdir, *dirs):
    from ase.io import read
    res = {"definition": __doc__, "models": {}}
    T = {s: read(f"{tdir}/test_{s}.extxyz", ":") for s in ("ion", "des")}
    for d in dirs:
        for p in sorted(glob.glob(f"{d}/*/test_ion.json")):
            m = os.path.basename(os.path.dirname(p)); res["models"][m] = {}
            for s, A in T.items():
                q = f"{d}/{m}/test_{s}.json"
                if not os.path.exists(q):
                    continue
                fr = json.load(open(q))["frames"]
                fe, rows = defaultdict(list), []
                for a, f in zip(A, fr):
                    if f.get("energy") is None:
                        continue
                    ch = int(a.info.get("charge", 0))
                    if f.get("forces") is not None:
                        fe[ch].append(float(np.mean(np.abs(np.asarray(f["forces"]) - a.get_forces()))))
                    rows.append((a.get_chemical_symbols(), f["energy"] - a.get_potential_energy(), ch))
                els = sorted({x for r in rows for x in r[0]})
                X = np.array([[r[0].count(e) for e in els] for r in rows], float); y = np.array([r[1] for r in rows])
                b = np.linalg.lstsq(X, y, rcond=None)[0] if len(rows) else []
                n = X.sum(1) if len(rows) else []
                eres = np.abs(y - X @ b) / n * 1e3 if len(rows) else []
                res["models"][m][s] = {"n": len(rows), "n_failed": len(A) - len(rows),
                                       "F_mae_eV_A_by_charge": {str(k): [len(v), float(np.mean(v))] for k, v in sorted(fe.items())},
                                       "F_mae_eV_A": float(np.mean([x for v in fe.values() for x in v])) if fe else None,
                                       "E_mae_meV_atom_offsetfit": float(np.mean(eres)) if len(rows) else None}
    json.dump(res, open(out, "w"), indent=1)
    for m, v in res["models"].items():
        print(m, {s: (x["n"], x["F_mae_eV_A"] and round(x["F_mae_eV_A"], 3), x["E_mae_meV_atom_offsetfit"] and round(x["E_mae_meV_atom_offsetfit"], 2)) for s, x in v.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
