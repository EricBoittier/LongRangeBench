#!/usr/bin/env python3
"""Per-record contamination flags ``seen_by`` for the canonical sets -> one JSON {set: {record_id: [sources]}}.

    python seen_by.py DATA_DIR OUT.json --stores SEEN_EVAL.json DIM_ALL.xyz --split SPLIT.json --tail TAIL_PAIRS.txt

Sources:
  - store hits (Tauri training stores: qcell_des400k, OMol25 subsets, QCML, SPICE...) from the Tauri leak check:
    SEEN_EVAL.json maps frame indices of DIM_ALL.xyz (the 1,254 x 3 dimer / monomer frames) to the stores whose
    geometry keys match; frames are matched to lrbench terms by species + positions (1e-4 A). A dimer record is
    'seen' by a store if its AB frame matched; its monomer slices are flagged separately ('<store>:monomer');
  - 'train_aug': record ids on the train side of SPLIT.json (pairs used to train augmented models);
  - 'tail_data': dimer pairs (pair names, one per line in TAIL_PAIRS.txt) that occur in a lane's tail training data.
A record with no flag is 'unseen'. The id lists are per canonical set (dimers, trimers, families_*)."""
import json
import sys
from collections import defaultdict

import numpy as np


def frames(path):
    out, L = [], open(path).read().splitlines()
    i = 0
    while i < len(L):
        n = int(L[i]); sp = [l.split()[0] for l in L[i + 2:i + 2 + n]]
        x = np.array([[float(v) for v in l.split()[1:4]] for l in L[i + 2:i + 2 + n]])
        out.append((sp, x)); i += n + 2
    return out


def key(sp, x):
    return (tuple(sp), tuple(np.round(np.asarray(x, float), 4).ravel().tolist()))


def main():
    a = sys.argv[1:]
    data, out = a[0], a[1]
    opt = lambda k, n=1: a[a.index(k) + 1:a.index(k) + 1 + n] if k in a else None
    seen = defaultdict(lambda: defaultdict(set))
    st = opt("--stores", 2)
    if st:
        S = json.load(open(st[0]))["dimers"]; F = frames(st[1])
        hit = {key(*F[int(i)]): v for i, v in S.items()}
        for l in open(f"{data}/dimers.jsonl"):
            r = json.loads(l)
            x = np.asarray(r["positions"]); fid = np.asarray(r["fragment_id"]); sp = r["species"]
            k = key(sp, x)
            if k in hit:
                seen["dimers"][r["id"]] |= {h.split("/")[0] for h in hit[k]}
            for f in (0, 1):
                m = fid == f
                kf = key([s for s, q in zip(sp, m) if q], x[m])
                if kf in hit:
                    seen["dimers"][r["id"]] |= {h.split("/")[0] + ":monomer" for h in hit[kf]}
    sp = opt("--split")
    if sp:
        J = json.load(open(sp[0]))
        for s, ids in J["train_aug_point_ids"].items():
            for i in ids:
                seen[s][i].add("train_aug")
    tp = opt("--tail")
    if tp:
        pairs = {l.strip() for l in open(tp[0]) if l.strip()}
        for l in open(f"{data}/dimers.jsonl"):
            r = json.loads(l)
            if (r.get("extra") or {}).get("pair") in pairs:
                seen["dimers"][r["id"]].add("tail_data")
    res = {s: {i: sorted(v) for i, v in d.items()} for s, d in seen.items()}
    json.dump({"definition": __doc__, "seen_by": res}, open(out, "w"), indent=1)
    for s, d in res.items():
        print(s, len(d), sorted({x for v in d.values() for x in v})[:12])


if __name__ == "__main__":
    main()
