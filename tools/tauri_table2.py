#!/usr/bin/env python3
"""SO3LR-Tauri Table II dimer benchmarks (SAPT10k, PLA15, QUID, JSCH-2005) in two E_int conventions.

    python tauri_table2.py make TABLE2_DIR OUT.extxyz [--sapt-neutral 1000]
    python tauri_table2.py analyse OUT.extxyz OUT.json PRED.json [PRED.json ...]   (label = parent dir name)

make: per dimer four frames, role = orig (bound, total charge Q), trans (one monomer moved beyond the cutoff, same Q:
the paper's frame), A, B (the monomers alone at charge_a / charge_b, closed shell, rigid slices of the bound frame).
Every frame carries formal_charges (fragment charge on its first atom). SAPT10k: every charged dimer (1,148) plus a
seed-0 stratified sample of neutral dimers (by system family); PLA15, QUID, JSCH-2005: all.
analyse:
  (a) E_int^trans = E(orig) - E(trans)          (Tauri / SI convention: same total charge on both frames)
  (b) E_int^sep   = E(orig) - E(A) - E(B)        (lrbench convention: separate monomers at their own charges)
  (a) - (b) = E(A) + E(B) - E(trans) = minus the model's separated-pair charge-state offset; zero for a size-consistent
  model. Errors vs ref_Eint (kcal/mol in the file, meV here) by set and by charged / neutral, and the offset statistics.
Reference levels differ from wB97M-V (SAPT2+(3)(CCD)/aug-cc-pVTZ, LNO-CCSD(T)/CBS, CCSD(T)/CBS or MP2/CBS,
MP2-F12+DLPNO-CCSD(T)): physics scores, not emulation scores."""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

KCAL = 43.364104  # meV
LEVEL = {"sapt10k": "SAPT2+(3)(CCD)/aug-cc-pVTZ", "quid": "LNO-CCSD(T)/CBS",
         "jsch2005": "CCSD(T)/CBS (102) or MP2/CBS (22)", "pla15": "MP2-F12 + dDLPNO-CCSD(T) (+ B3LYP-D3 MBC)"}


def sel(s):
    out = []
    for p in str(s).split(","):
        a, _, b = p.partition("-")
        out += list(range(int(a) - 1, int(b or a)))
    return out


def make(d, out, *a):
    from ase.io import read, write
    nneu = int(a[a.index("--sapt-neutral") + 1]) if "--sapt-neutral" in a else 1000
    rng = np.random.default_rng(0)
    F = []
    for name in ("sapt10k", "pla15", "quid", "jsch2005"):
        A = read(f"{d}/{name}.xyz", ":")
        pairs = [(A[i], A[i + 1]) for i in range(0, len(A), 2)]
        if name == "sapt10k":
            ch = [p for p in pairs if int(p[0].info["charge_a"]) or int(p[0].info["charge_b"])]
            neu = [p for p in pairs if not (int(p[0].info["charge_a"]) or int(p[0].info["charge_b"]))]
            fam = defaultdict(list)
            for p in neu:
                fam[p[0].info["system_id"].rsplit("_", 1)[0]].append(p)
            frac = nneu / len(neu); keep = []
            for f, ps in sorted(fam.items()):
                k = max(1, int(round(frac * len(ps))))
                keep += [ps[i] for i in sorted(rng.choice(len(ps), min(k, len(ps)), replace=False))]
            pairs = ch + keep
        for o, t in pairs:
            qa, qb = int(o.info["charge_a"]), int(o.info["charge_b"]); ia, ib = sel(o.info["selection_a"]), sel(o.info["selection_b"])
            base = dict(set_name=name, system_id=o.info["system_id"], ref_Eint_meV=float(o.info["ref_Eint"]) * KCAL,
                        charge_a=qa, charge_b=qb, charged=int(bool(qa or qb)), level=LEVEL[name])
            for role, at, idx, q in (("orig", o, None, qa + qb), ("trans", t, None, qa + qb), ("A", o, ia, qa), ("B", o, ib, qb)):
                x = at.copy() if idx is None else o[idx]
                fq = np.zeros(len(x))
                if idx is None:
                    fq[ia[0]] = qa; fq[ib[0]] = qb
                else:
                    fq[0] = q
                x.arrays = {k: v for k, v in x.arrays.items() if k in ("numbers", "positions")}
                x.arrays["formal_charges"] = fq
                x.info = dict(base, role=role, charge=q, spin_multiplicity=1)
                F.append(x)
    write(out, F, format="extxyz")
    print(out, len(F) // 4, "dimers", Counter(f.info["set_name"] for f in F[::4]))


def analyse(xyz, out, *preds):
    from ase.io import read
    A = read(xyz, ":"); info = [a.info for a in A]
    res = {"definition": __doc__, "levels": LEVEL, "models": {}}
    for p in preds:
        fr = json.load(open(p))["frames"]; lab = os.path.basename(os.path.dirname(p))
        E = [f.get("energy") for f in fr]
        rows = []
        for i in range(0, len(info), 4):
            e = E[i:i + 4]
            if any(x is None for x in e):
                continue
            o, t, a, b = (x * 1e3 for x in e)
            I = info[i]
            rows.append(dict(set=I["set_name"], id=I["system_id"], charged=I["charged"], ref=I["ref_Eint_meV"],
                             trans=o - t, sep=o - a - b, offset=t - a - b))
        S = {}
        for s in sorted({r["set"] for r in rows}):
            for ch in (0, 1, None):
                sel_ = [r for r in rows if r["set"] == s and (ch is None or r["charged"] == ch)]
                if not sel_:
                    continue
                f = lambda k: float(np.mean([abs(r[k] - r["ref"]) for r in sel_]))
                S[f"{s}|{'all' if ch is None else ('charged' if ch else 'neutral')}"] = dict(
                    n=len(sel_), MAE_trans_meV=f("trans"), MAE_sep_meV=f("sep"),
                    offset_median_meV=float(np.median([r["offset"] for r in sel_])),
                    offset_abs_median_meV=float(np.median([abs(r["offset"]) for r in sel_])),
                    MAE_trans_kcal=f("trans") / KCAL, MAE_sep_kcal=f("sep") / KCAL)
        res["models"][lab] = {"summary": S, "n_failed_dimers": len(info) // 4 - len(rows), "rows": rows}
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for m, v in res["models"].items():
        for k, s in v["summary"].items():
            if k.endswith("charged") or k.endswith("neutral") or k.startswith(("pla15", "quid", "jsch")):
                print(f"{m:14s} {k:20s} n={s['n']:5d} MAE kcal/mol trans {s['MAE_trans_kcal']:7.2f} sep {s['MAE_sep_kcal']:7.2f}"
                      f"  |offset| med {s['offset_abs_median_meV']:8.1f} meV")


if __name__ == "__main__":
    {"make": make, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
