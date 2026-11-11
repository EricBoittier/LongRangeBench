#!/usr/bin/env python3
"""Build the lrbench set `omol25_val_lr` (canonical jsonl + groups) from OMol25 validation records (ωB97M-V/def2-TZVPD,
held out of OMol25 train): the LR-relevant subsets electrolytes_scaled_sep (compressed clusters, scale < 1),
scaled_separations_exp (expanded, scale > 1), 5A_elytes, ml_elytes and droplet, extracted from the val aselmdb shards
(the OMol25 validation aselmdb directory).

    python omol25_val_lr.py OUTDIR shard.extxyz [shard.extxyz ...] [--n-other 300] [--max-parents 600]

Records: cls = subset, group = parent cluster (source path without the scale / charge / spin suffix), distance = the
separation scale factor (1.0 for the unscaled subsets), charge / spin from the record, energy (eV) and forces.
Groups (kind dE, fixed composition and total charge, so per-element offsets cancel): for every parent with >= 2 scale
points, E(scale s) - E(anchor), anchor = the parent's largest scale; meta.family = "<compressed|expanded> s-bin",
meta.pair = parent (bootstrap unit). Kept: every multi-scale parent up to --max-parents (seed 0), plus --n-other random
structures from each of 5A_elytes / ml_elytes / droplet (absolute energy/force metrics only). Structure counts per stratum
are printed. seen_by: these are OMol25 *validation* records. PET-OMol-S/M/L, MACE-OMOL-0 and MACE-POLAR-1 train on OMol25
train, and SO3LR-T's OMol25 subsets are OMol25 train (its val split is listed as held out in the Tauri manifest), so the
set is unseen by every model scored here; our fine-tunes train on train4m_sel (OMol25 train) and set A."""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

PAT = re.compile(r"^(?P<sub>[^/]+)/(?P<stem>.+)_(?P<scale>\d+\.\d+)_(?P<q>-?\d+)_(?P<m>\d+)/orca")


def main(out, *a):
    from ase.io import read
    files = [x for x in a if x.endswith(".extxyz")]
    n_other = int(a[a.index("--n-other") + 1]) if "--n-other" in a else 300
    max_par = int(a[a.index("--max-parents") + 1]) if "--max-parents" in a else 600
    rng = np.random.default_rng(0)
    F = [x for f in files for x in read(f, ":")]
    par = defaultdict(list); other = defaultdict(list)
    for x in F:
        s = x.info["source"]; m = PAT.match(s)
        if x.info["subset"] in ("electrolytes_scaled_sep", "scaled_separations_exp") and m:
            par[(m["stem"], int(x.info["charge"]), int(x.info["spin_multiplicity"]))].append((float(m["scale"]), x))
        else:
            other[x.info["subset"]].append(x)
    multi = [k for k, v in par.items() if len({s for s, _ in v}) >= 2]
    multi = [multi[i] for i in sorted(rng.choice(len(multi), min(max_par, len(multi)), replace=False))]
    os.makedirs(out, exist_ok=True)
    recs, groups = [], []
    rid = lambda x: x.info["source"].rsplit("/orca", 1)[0]

    def rec(x, scale, group):
        return dict(id=rid(x), set="omol25_val_lr", species=x.get_chemical_symbols(), positions=x.positions.tolist(),
                    cell=None, pbc=[False] * 3, charge=int(x.info["charge"]), spin=int(x.info["spin_multiplicity"]),
                    energy=float(x.get_potential_energy()), forces=x.get_forces().tolist(), level="wB97M-V/def2-TZVPD",
                    cls=x.info["subset"], group=group, distance=scale, flags=[], role="cluster",
                    extra=dict(source=x.info["source"], s2_dev=x.info.get("s2_dev"), split="OMol25 val"))
    seen = set()
    for k in multi:
        pts = sorted({s: x for s, x in par[k]}.items())
        stem = f"{k[0]}|q{k[1]}|m{k[2]}"
        for s, x in pts:
            if rid(x) not in seen:
                recs.append(rec(x, s, stem)); seen.add(rid(x))
        sa, xa = pts[-1]
        for s, x in pts[:-1]:
            fam = ("compressed" if s < 1 else "expanded") + f" {np.floor(s * 10) / 10:.1f}-{np.floor(s * 10) / 10 + 0.1:.1f}"
            groups.append(dict(name=f"dE:{rid(x)}->{sa}", kind="dE", terms=[[rid(x), 1.0], [rid(xa), -1.0]],
                               ref=float(x.get_potential_energy() - xa.get_potential_energy()),
                               meta=dict(q_fixed=True, family=fam, pair=stem, scale=s, scale_anchor=sa, charge=k[1],
                                         cls="scaled_separation", note="E(scale s) - E(largest scale) of one cluster")))
    for sub, xs in sorted(other.items()):
        for i in sorted(rng.choice(len(xs), min(n_other, len(xs)), replace=False)):
            x = xs[i]
            if rid(x) not in seen:
                recs.append(rec(x, 1.0, rid(x))); seen.add(rid(x))
    with open(f"{out}/omol25_val_lr.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(f"{out}/omol25_val_lr.groups.json", "w"))
    from collections import Counter
    print(len(recs), "records", Counter(r["cls"] for r in recs), len(groups), "dE groups",
          Counter(g["meta"]["family"] for g in groups))


if __name__ == "__main__":
    main(*sys.argv[1:])
