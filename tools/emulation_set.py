#!/usr/bin/env python3
"""Build the lrbench set `emulation`: the DFT points the canonical dimer set leaves out because DFT itself does something
"unphysical" (charge transfer, fractional charges, unbound anions, spin-contaminated broken-symmetry states), plus the
P2 ORCA scans, all with ωB97M-V/def2-TZVPD labels and per-atom Löwdin charges. A DFT emulator should reproduce these
(memory: lr-model-emulates-dft); a "physics" score excludes them.

    python emulation_set.py OUTDIR FULL_PAIRS_DIR FRAGS.json P2_DFT.json

Sources:
  - longrange-dimers full scans (gpu4pyscf RKS, the canonical set's own protocol): every point the label rule flagged
    (fractional fragment charge at >= 8 A, ionic state above the CT fragments), 54 points;
  - P2 (ORCA 6.1.1, UKS + stability, OMol25 settings): Mg2+-H2O singlet (broken-symmetry CT state) and triplet,
    Ca2+-H2O, SO4 2- -H2O (anion unbound in the basis), CH3F...I- 8-100 A.
Record tags (`flags`, used by the physics filter; none of them is `dft_pathology_do_not_score`, so `lrbench eval`
scores every record = the emulation score):
  ct_fractional      DFT fragment charge deviates from formal by > 0.02 e (charge transfer / delocalisation)
  ct_ground_state    DFT's lowest state is the charge-transferred one (ionic state above the CT fragments)
  unbound_anion      fragment HOMO > 0 (bound only in the finite basis)
  spin_contaminated  <S^2> deviates from the pure-spin value by > 0.1 (broken-symmetry singlet)
  scf_wrong_state    gpu4pyscf RKS landed on a state above the lowest stable one (Mg2+-H2O >= 15 A, +18.7 eV);
                     superseded by the ORCA UKS points and excluded from both scores (not DFT's answer); also any point
                     whose ion charge collapsed (< 10 % of formal) or reversed sign (Na+Cl- and Mg2+Cl- at 25 A)"""
import ast
import json
import os
import sys

import numpy as np

HA = 27.211386245988


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def main(out, pairs, frags, p2):
    F = {k: {a: lit(b) for a, b in v.items()} for k, v in json.load(open(frags)).items()}
    recs, groups = [], []

    def add(rid, sp, x, q, mult, nA, qa, qb, E, EA, EB, ql, tags, cls, pair, d, src, sA=1, sB=1):
        fid = [0] * nA + [1] * (len(sp) - nA)
        fq = [0.0] * len(sp); fq[0] = qa; fq[nA] = qb
        qlw = None if ql is None else [float(v) for v in ql]
        recs.append(dict(id=rid, set="emulation", species=list(sp), positions=np.asarray(x).tolist(), cell=None,
                         pbc=[False] * 3, charge=int(q), spin=int(mult), energy=float(E), forces=None,
                         level="wB97M-V/def2-TZVPD", formal_charges=fq, fragment_id=fid, fragment_charges=[qa, qb],
                         fragment_spins=[sA, sB], cls=cls, group=pair, role="AB", distance=float(d),
                         refs=dict(E_int=float(E - EA - EB), E_A=float(EA), E_B=float(EB), charges_lowdin=qlw,
                                   qA_lowdin=None if qlw is None else float(sum(qlw[:nA]))),
                         flags=list(tags), extra=dict(source=src, pair=pair)))
        groups.append(dict(name=f"E_int:{rid}", kind="E_int", terms=[[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]],
                           ref=float(E - EA - EB), meta=dict(cls=cls, pair=pair, dmin=float(d), tags=list(tags),
                                                             q_fixed=True, note="monomers = rigid slices at formal charge")))
    # 1. flagged full-scan points
    import glob
    for f in sorted(glob.glob(f"{pairs}/*.jsonl")):
        R = [json.loads(l) for l in open(f)]
        mono = {r["orient"]: r for r in R if r.get("kind") == "monomers"}
        for r in R:
            if r.get("kind") != "dimer" or not r.get("flag"):
                continue
            m = mono.get(r["orient"])
            if m is None or r.get("E") is None:
                continue
            nA = len(F[r["A"]]["sym"]) if isinstance(F[r["A"]]["sym"], list) else len(lit(F[r["A"]]["sym"]))
            tags = []
            if "fractional" in r["flag"]:
                tags.append("ct_fractional")
            if "ionic state above" in r["flag"]:
                tags.append("ct_ground_state")
            qa, qf = r.get("qA_lowdin"), r["qA_formal"]
            if (r["pair"] == "Mg_H2O" and float(r["dmin"]) >= 15) or (
                    qa is not None and (np.sign(qa) != np.sign(qf) or abs(qa) < 0.1 * abs(qf))):
                # SCF landed on a state that is not DFT's lowest (audit: Mg2+-H2O >= 15 A at +18.7 eV; Na+Cl-, Mg2+Cl- at
                # 25 A with the fragment charge collapsed or reversed, several eV from the default-guess state)
                tags = ["scf_wrong_state", "dft_pathology_do_not_score"]
            cls = "ct:" + r["cls"]
            add(f"{r['pair']}:o{r['orient']}:d{r['dmin']:g}:full", r["sym"], r["R"], r["charge"], 1, nA, r["qA_formal"],
                r["qB_formal"], r["E"], m["EA"], m["EB"], r.get("q"), tags, cls, f"{r['pair']}:o{r['orient']}", r["dmin"],
                "longrange-dimers full scan (gpu4pyscf RKS)")
    # 2. P2 ORCA
    P = json.load(open(p2))["calcs"]
    mono = {(r["sys"], r["role"]): r for r in P.values() if r.get("role") in ("A", "B") and "E_eV" in r}
    for k, r in sorted(P.items()):
        if r.get("role") != "AB" or "E_eV" not in r:
            continue
        A, B = mono[(r["sys"], "A")], mono[(r["sys"], "B")]
        tags = []
        if r.get("qA_lowdin") is not None and abs(r["qA_lowdin"] - A["charge"]) > 0.02:
            tags.append("ct_fractional")
        if r["sys"] == "Mg_H2O":
            tags.append("ct_ground_state")
        pure = (r["mult"] - 1) / 2 * ((r["mult"] - 1) / 2 + 1)
        if abs((r.get("S2") or 0) - pure) > 0.1:
            tags.append("spin_contaminated")
        if r["sys"] == "SO4_H2O":
            tags.append("unbound_anion")
        cls = {"CH3F_I": "ion_polar", "Ca_H2O": "ion_polar_q2", "Mg_H2O": "ion_polar_q2", "SO4_H2O": "ion_polar_q2"}[r["sys"]]
        add(f"{k}:orca", r["species"], r["positions"], r["charge"], r["mult"], r["nA"], A["charge"], B["charge"],
            r["E_eV"], A["E_eV"], B["E_eV"], r.get("q_lowdin"), sorted(set(tags)), cls, r["sys"] + ("_t" if r["mult"] == 3 else ""),
            r["d"], "P2 ORCA UKS + stability")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/emulation.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(f"{out}/emulation.groups.json", "w"))
    from collections import Counter
    print(len(recs), "records", Counter(t for r in recs for t in (r["flags"] or ["none"])), Counter(r["cls"] for r in recs))


if __name__ == "__main__":
    main(*sys.argv[1:])
