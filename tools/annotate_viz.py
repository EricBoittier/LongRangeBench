#!/usr/bin/env python3
"""Add per-structure `split` (train_aug / heldout / none, from SPLIT.json) and `seen_by`
(comma list of contamination sources, 'unseen' when empty; from tools/seen_by.py) to lrbench viz chemiscope files.

    python annotate_viz.py SPLIT.json SEEN_BY.json IN.json.gz [IN.json.gz ...]     (rewritten in place)
Entries are matched on the `id` property and the set (`set` property, else the file-name prefix)."""
import gzip
import json
import sys


def main(split, seen, *files):
    S = json.load(open(split)); B = json.load(open(seen))["seen_by"]
    side = {}
    for k in ("train_aug", "heldout"):
        for s, ids in S[f"{k}_point_ids"].items():
            for i in ids:
                side[(s, i)] = k
    for f in files:
        d = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
        P = d["properties"]; ids = P["id"]["values"]
        sets = P["set"]["values"] if "set" in P else [f.split("/")[-1].split("_")[0]] * len(ids)
        P["split"] = {"target": "structure", "values": [side.get((s, i), "none") for s, i in zip(sets, ids)],
                      "description": "pair-level split (lr-augment/split.json): train_aug = pairs used to train augmented models"}
        P["seen_by"] = {"target": "structure", "values": [",".join(B.get(s, {}).get(i, [])) or "unseen" for s, i in zip(sets, ids)],
                        "description": "training sets containing this structure (Tauri store geometry keys; ':monomer' = only a "
                                       "monomer slice; train_aug); 'unseen' otherwise"}
        (gzip.open if f.endswith(".gz") else open)(f, "wt").write(json.dumps(d))
        print(f, sum(v != "unseen" for v in P["seen_by"]["values"]), "seen;", sum(v == "heldout" for v in P["split"]["values"]), "heldout")


if __name__ == "__main__":
    main(*sys.argv[1:])
