#!/usr/bin/env python3
"""NCI Atlas packaged sets (Řezáč et al.; github.com/Honza-R/NCIAtlas packaged_sets, CC BY 4.0) -> lrbench canonical
set `ext_ncia`.

    python -I ext_ncia.py UNPACKED_DIR OUTDIR

Each package xyz carries charge, charge_a/b, selection_a/b (1-based ranges), scaling (x R_e of the closest contact),
benchmark_Eint (kcal/mol, CCSD(T)/CBS composite) and group. Rigid monomers taken from the dimer, so E_int is the
vertical interaction energy: groups AB - AB#frag=0 - AB#frag=1. distance = closest A-B atom distance (computed);
group = system id (file stem without the scaling suffix) so curves stay together in the bootstrap. All closed shell
(even electron count at the published fragment charges is asserted). Level: CCSD(T)/CBS; scoring an OMol25 model is
an end-to-end score (functional error included), --allow-level-mismatch.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np

Z = {s: i for i, s in enumerate("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga "
                                 "Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split())}
KCAL_EV = 1 / 23.060548


def sel(s):
    out = []
    for p in s.split(","):
        a, _, b = p.partition("-")
        out += list(range(int(a) - 1, int(b or a)))
    return out


def cls_of(qa, qb):
    return ("ion_pair" if qa * qb < 0 else "like_charge" if qa * qb > 0 else "ion_neutral" if qa or qb
            else "neutral")


def main(src, outdir):
    recs, groups, n = [], [], collections.Counter()
    for f in sorted(glob.glob(os.path.join(src, "*", "*", "geometries", "*.xyz"))):
        S = re.search(r"NCIA_([^/]+)/geometries", f).group(1)
        L = open(f).read().splitlines()
        nat = int(L[0]); kv = dict(re.findall(r"(\w+)=(\S+)", L[1]))
        sp = [l.split()[0] for l in L[2:2 + nat]]
        R = np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2 + nat]])
        a, b = sel(kv["selection_a"]), sel(kv["selection_b"])
        assert sorted(a + b) == list(range(nat)), f
        fid = np.zeros(nat, int); fid[b] = 1
        qa, qb = int(kv["charge_a"]), int(kv["charge_b"])
        assert (sum(Z[s] for s in sp) - qa - qb) % 2 == 0, f"odd electrons {f}"
        stem = os.path.basename(f)[:-4]
        sysid = re.sub(r"_\d{3}$", "", stem)
        rid = f"{S}:{stem}"
        d = float(np.linalg.norm(R[a][:, None] - R[b][None], axis=-1).min())
        c = cls_of(qa, qb)
        e = float(kv["benchmark_Eint"]) * KCAL_EV
        recs.append(dict(id=rid, set="ext_ncia", species=sp, positions=R.round(9).tolist(), cell=None,
                         pbc=[False] * 3, charge=qa + qb, spin=1, energy=None, forces=None,
                         level="CCSD(T)/CBS (NCI Atlas)", formal_charges=None, unit_id=None,
                         fragment_id=fid.tolist(), fragment_charges=[qa, qb], fragment_spins=[1, 1], locality=None,
                         q_fixed_in_pair=True, cls=c, group=f"{S}:{sysid}", role="AB", distance=round(d, 4),
                         refs=dict(E_int=e), flags=[],
                         extra=dict(subset=S, scaling=float(kv.get("scaling", "nan")), ncia_group=kv.get("group"))))
        groups.append(dict(name=f"{rid}:E_int", kind="E_int",
                           terms=[[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]], ref=e,
                           meta=dict(cls=c, distance=round(d, 4), scan=f"{S}:{sysid}", pair=f"{S}:{sysid}",
                                     bootstrap_group=f"{S}:{sysid}", subset=S, category=c, q_fixed=True,
                                     scaling=float(kv.get("scaling", "nan")),
                                     atomic_ion_fragment=bool((qa and len(a) == 1) or (qb and len(b) == 1)),
                                     flags=["unbound_anion_candidate"] if min(qa, qb) <= -2 else [],
                                     flags_note="provisional, from charges only; not DFT-verified")))
        n[(S, c, "<4" if d < 4 else "4-8" if d < 8 else "8-12" if d < 12 else ">=12")] += 1
    os.makedirs(os.path.join(outdir, "canonical"), exist_ok=True)
    with open(os.path.join(outdir, "canonical", "ext_ncia.jsonl"), "w") as fo:
        for r in recs:
            fo.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(outdir, "canonical", "ext_ncia.groups.json"), "w"))
    json.dump({"|".join(k): v for k, v in sorted(n.items())},
              open(os.path.join(outdir, "canonical", "ext_ncia.meta.json"), "w"), indent=1)
    for k, v in sorted(n.items()):
        print(*k, v)


if __name__ == "__main__":
    main(*sys.argv[1:3])
