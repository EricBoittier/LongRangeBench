#!/usr/bin/env python3
"""Geometry scans for `lrbench predict` and their analysis (size consistency, SN2 cutoff scan, box scan).

    python scans.py make DATA_DIR SN2_FRAME0.xyz OUTDIR      # writes sizecons.extxyz, sn2_scan.extxyz, box_scan.extxyz
    python scans.py analyse OUTDIR PRED_DIR OUT.json          # PRED_DIR/<model>/<scan>.json from `lrbench predict`

Every frame carries info charge / spin_multiplicity / sys / d / role (AB, A, B; box frames: L, role box|open) and a
per-atom formal_charges array (fragment charge on each fragment's first atom), so `lrbench predict --formal-charges
from-data` hands fragment-charge GGUFs the same unit charges as the canonical sets.

sizecons: E(AB, d) - E(A) - E(B), fragments rigid, B moved along the COM axis to closest-atom distance d = 8 ... 100 A,
fragment charges and spins set explicitly (A, B as isolated structures). Systems: CH3F...I- (the etc/analysis SN2
frame), and three ion-neutral dimers of the canonical set (orientation 0, 8 A record): Na+-H2O, Cl- -H2O, NH4+-C6H6.
sn2_scan: I- pulled along C->I, d(C-I) 1.5 ... 32 A on top of frame0 (62 points, as etc/analysis), plus the fragments.
box_scan: CH3F (neutral) and CH3F...I- (charged, frame0) in cubic boxes L = 10 ... 40 A (8 sizes) and open boundary.
"""
import json
import os
import sys

import numpy as np

D_SIZE = [8, 10, 12, 16, 20, 24, 32, 48, 64, 100]
L_BOX = [10, 12, 15, 20, 25, 30, 35, 40]
ION_NEUTRAL = ["Na_H2O:o0", "Cl_H2O:o0", "NH4_C6H6:o0"]


def _com(at, idx):
    m = at.get_masses()[idx]
    return (m[:, None] * at.positions[idx]).sum(0) / m.sum()


def _dmin(x, y):
    return float(np.min(np.linalg.norm(x[:, None] - y[None], axis=-1)))


def _place(at, ia, ib, d):
    u = _com(at, ib) - _com(at, ia); u /= np.linalg.norm(u)
    x = at.positions.copy(); lo, hi = -_dmin(x[ia], x[ib]) + 1e-3, d + 200.0
    for _ in range(100):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if _dmin(x[ia], x[ib] + s * u) < d else (lo, s)
    x[ib] += 0.5 * (lo + hi) * u
    return x


def _frame(at, idx, x, q, mult, fq, **info):
    from ase import Atoms
    a = Atoms([at.get_chemical_symbols()[i] for i in idx], positions=x[idx])
    f = np.zeros(len(idx)); f[0] = 0.0
    starts = {}
    for k, i in enumerate(idx):  # fragment charge on each fragment's first atom (unit-first-atom convention)
        starts.setdefault(fq[i][0], k)
    for frag, k in starts.items():
        f[k] = fq[idx[k]][1]
    a.arrays["formal_charges"] = f
    a.info.update(charge=int(q), spin_multiplicity=int(mult), **info)
    return a


def systems(data_dir, sn2):
    from ase import Atoms
    from ase.io import read
    out = []
    s = read(sn2)
    out.append(("CH3F_I", Atoms(s.get_chemical_symbols(), s.positions), [0, 1, 2, 3, 4], [5], 0, -1))
    recs = {}
    for l in open(os.path.join(data_dir, "dimers.jsonl")):
        r = json.loads(l)
        g = r.get("group")
        if g in ION_NEUTRAL and r.get("role") == "AB" and abs(float(r["distance"]) - 8) < 0.01:
            recs[g] = r
    for g in ION_NEUTRAL:
        r = recs[g]; fid = np.asarray(r["fragment_id"]); q = r["fragment_charges"]
        at = Atoms(r["species"], positions=np.asarray(r["positions"]))
        out.append((g.split(":")[0], at, list(np.where(fid == 0)[0]), list(np.where(fid == 1)[0]), q[0], q[1]))
    return out


def make(data_dir, sn2, outdir):
    from ase.io import write
    os.makedirs(outdir, exist_ok=True)
    F = []
    for name, at, ia, ib, qa, qb in systems(data_dir, sn2):
        fq = {i: (0, qa) for i in ia} | {i: (1, qb) for i in ib}
        ia, ib = np.array(ia), np.array(ib); ab = np.concatenate([ia, ib])
        for d in D_SIZE:
            x = _place(at, ia, ib, d)
            F.append(_frame(at, ab, x, qa + qb, 1, fq, sys=name, d=d, role="AB"))
            F.append(_frame(at, ia, x, qa, 1, fq, sys=name, d=d, role="A"))
            F.append(_frame(at, ib, x, qb, 1, fq, sys=name, d=d, role="B"))
    write(f"{outdir}/sizecons.extxyz", F, format="extxyz")
    # SN2 cutoff scan (etc/analysis displace_atom_test geometry: I moved along C->I)
    name, at, ia, ib, qa, qb = systems(data_dir, sn2)[0]
    fq = {i: (0, 0) for i in ia} | {5: (1, -1)}
    u = at.positions[5] - at.positions[0]; u /= np.linalg.norm(u)
    S = []
    for d in np.linspace(1.5, 32.0, 62):
        x = at.positions.copy(); x[5] = x[0] + d * u
        S.append(_frame(at, np.arange(6), x, -1, 1, fq, sys="CH3F_I", d=float(d), role="AB"))
        S.append(_frame(at, np.array(ia), x, 0, 1, fq, sys="CH3F_I", d=float(d), role="A"))
    S.append(_frame(at, np.array([5]), at.positions, -1, 1, fq, sys="CH3F_I", d=0.0, role="B"))
    write(f"{outdir}/sn2_scan.extxyz", S, format="extxyz")
    # box scan: open boundary then cubic boxes, molecule centred
    B = []
    for nm, idx, q in (("CH3F", np.array(ia), 0), ("CH3F_I", np.arange(6), -1)):
        x = at.positions - at.positions[idx].mean(0)
        B.append(_frame(at, idx, x, q, 1, fq, sys=nm, L=0.0, role="open"))
        for L in L_BOX:
            a = _frame(at, idx, x + L / 2, q, 1, fq, sys=nm, L=float(L), role="box")
            a.set_cell([L, L, L]); a.set_pbc(True)
            B.append(a)
    write(f"{outdir}/box_scan.extxyz", B, format="extxyz")
    print(outdir, len(F), len(S), len(B))


def analyse(outdir, pred_dir, out):
    from ase.io import read
    res = {"definition": __doc__, "models": {}}
    for m in sorted(os.listdir(pred_dir)):
        R = res["models"][m] = {}
        for scan in ("sizecons", "sn2_scan", "box_scan"):
            p = f"{pred_dir}/{m}/{scan}.json"
            if not os.path.exists(p):
                continue
            fr = json.load(open(p))["frames"]; at = read(f"{outdir}/{scan}.extxyz", ":")
            E = [f.get("energy") for f in fr]
            info = [a.info for a in at]
            if scan == "sizecons":
                by = {}
                for i, a in enumerate(info):
                    by.setdefault((a["sys"], a["d"]), {})[a["role"]] = (E[i], fr[i].get("charges"))
                rows = []
                for (sy, d), v in sorted(by.items()):
                    if any(v.get(k, (None,))[0] is None for k in ("AB", "A", "B")):
                        continue
                    qA = None
                    if v["AB"][1] is not None:
                        na = len(at[[i for i, a in enumerate(info) if a["sys"] == sy and a["role"] == "A"][0]])
                        qA = float(np.sum(v["AB"][1][:na]))
                    rows.append(dict(sys=sy, d=d, E_int_meV=(v["AB"][0] - v["A"][0] - v["B"][0]) * 1e3, q_A_latent=qA))
                R["sizecons"] = rows
            elif scan == "sn2_scan":
                eB = E[-1]; rows = []
                for i in range(0, len(info) - 1, 2):
                    if E[i] is None or E[i + 1] is None or eB is None:
                        continue
                    ch = fr[i].get("charges")
                    rows.append(dict(d=info[i]["d"], E_int_meV=(E[i] - E[i + 1] - eB) * 1e3,
                                     F_CH3F=np.asarray(fr[i]["forces"])[:5].tolist() if fr[i].get("forces") else None,
                                     q_CH3F_latent=None if ch is None else float(np.sum(ch[:5]))))
                if rows and rows[-1]["F_CH3F"] is not None:
                    Ff = np.asarray(rows[-1]["F_CH3F"])
                    for r in rows:
                        r["max_dF_CH3F"] = float(np.linalg.norm(np.asarray(r.pop("F_CH3F")) - Ff, axis=1).max())
                else:
                    for r in rows:
                        r.pop("F_CH3F")
                R["sn2_scan"] = rows
            else:
                rows = []
                for sy in ("CH3F", "CH3F_I"):
                    idx = [i for i, a in enumerate(info) if a["sys"] == sy]
                    e0 = next(E[i] for i in idx if info[i]["role"] == "open")
                    for i in idx:
                        if info[i]["role"] == "box" and E[i] is not None and e0 is not None:
                            rows.append(dict(sys=sy, L=info[i]["L"], dE_box_minus_open_meV=(E[i] - e0) * 1e3))
                    sel = [r for r in rows if r["sys"] == sy and abs(r["dE_box_minus_open_meV"]) > 1e-3]
                    if len(sel) >= 3:
                        k, c = np.polyfit(np.log([r["L"] for r in sel]), np.log([abs(r["dE_box_minus_open_meV"]) for r in sel]), 1)
                        R.setdefault("box_power_law", {})[sy] = {"exponent": float(k), "n": len(sel)}
                R["box_scan"] = rows
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out)


if __name__ == "__main__":
    {"make": make, "analyse": analyse}[sys.argv[1]](*sys.argv[2:])
