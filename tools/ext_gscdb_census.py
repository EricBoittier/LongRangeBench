"""Census of every GSCDB137 noncovalent (NC) and EF subset: charges, closest contact (our cluster split and the
source fragments), category. Reads the pinned clone only (parsed, never executed)."""
import sys, json, collections
sys.path.insert(0, sys.argv[1])
from lrbench import gscdb
repo, out = sys.argv[2], sys.argv[3]
mols, rx, sets, *_ = gscdb.read_repo(repo)
nc = [n for n, r in sets.items() if r.get("Datatype_Short") == "NC"]
man = gscdb.build_manifest(repo, nc, "", cross_check_qchem=False)
rows = {}
for S, d in man["subsets"].items():
    bins = collections.Counter(); cats = collections.Counter(); dmax = 0; charged = 0
    for e in d["entries"]:
        dm = e.get("d_min_source") if e.get("d_min_source") is not None else e.get("d_min_cluster")
        b = "n/a" if dm is None else "<4" if dm < 4 else "4-8" if dm < 8 else "8-12" if dm < 12 else ">=12"
        bins[b] += 1; cats[e.get("category", "other")] += 1
        if dm is not None: dmax = max(dmax, dm)
        q = [man["molecules"][k]["charge"] for _, k in e["terms"]]
        charged += any(x != 0 for x in q)
    rows[S] = dict(n=d["n_entries"], theory=d["theory"], desc=d["description"], bins=dict(bins), cats=dict(cats),
                   dmax=round(dmax, 2), n_charged=charged, rms_kcal=d["rms_kcal"])
json.dump(dict(rows=rows, problems=man["problems"][:50]), open(out, "w"), indent=1)
for S, r in sorted(rows.items()):
    print(f"{S:12s} n={r['n']:4d} chg={r['n_charged']:4d} dmax={r['dmax']:6.2f} {r['bins']} {r['cats']}")
