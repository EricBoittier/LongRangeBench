#!/usr/bin/env python3
"""GSCDB137 noncovalent (NC) subsets -> lrbench canonical set `ext_gscdb_nc` (records + groups).

    python -I ext_gscdb_nc.py LRBENCH_SRC GSCDB_REPO OUTDIR

Source: Liang & Head-Gordon, GSCDB137 (arXiv:2508.13468v2; JCTC 21, 12601 (2025)), repository JiashuLiang/GSCDB,
pinned commit 0624edaa (CC BY 4.0). The clone is untrusted data: it is parsed (lrbench.gscdb.read_repo/build_manifest),
never imported or executed.

Conventions:
- One record per GSCDB molecule (id = the repository key), energy null: these sets carry reaction references only.
  `level` = "CCSD(T)/CBS (GSCDB137 ref)" on every record so groups never mix levels; the subset's own reference
  method is in group meta `theory`. Scoring a wB97M-V-trained model needs --allow-level-mismatch, and every number
  carries that mismatch (reference = coupled cluster, model = OMol25 DFT emulator).
- Each reaction is one group with the repository stoichiometry (terms = molecule ids, coefficients as published).
  kind "E_int" when the reaction is complex - monomers with a two-fragment atom map (distance = closest atom-atom
  distance between the source fragments, cls = charge category); otherwise kind "dE" (cluster binding energies,
  relative energies, three-body terms), family = subset.
- Complex records of E_int groups carry fragment_id / fragment_charges / fragment_spins from the reaction's monomer
  terms. The monomers are separate records (published geometry; rigid displaced copies in several subsets), so
  E_int here is the source's convention, not the #frag slicing of our own dimer set.
- Provisional class labels (no DFT check): meta.flags `unbound_anion_candidate` when any term is a multiply charged
  anion (oxalate/carbonate dianions in IHB100(x2), SO4 2- in SW49); kept and scored, own stratum.
- Group meta: subset, theory, category, d_min (source fragments), d_min_cluster (our covalent split),
  source_coordinate, atomic_ion_fragment, open_shell, scored_before (the subsets already labelled and scored
  through lrbench.gscdb: P0 + NBC10), bootstrap_group (curve).
"""
import json
import os
import sys
from collections import Counter

SCORED_BEFORE = {"IHB100x2", "AHB21", "CHB6", "IL16", "S66Rel7", "3BHET", "NBC10"}
LEVEL = "CCSD(T)/CBS (GSCDB137 ref)"
HA_EV = 27.211386245988


def cat_key(c):
    return {"ion pair": "ion_pair", "anion-neutral": "anion_neutral", "cation-neutral": "cation_neutral",
            "neutral PEC": "neutral", "neutral": "neutral", "rare-gas PEC": "rare_gas", "three-body": "three_body",
            "other charged": "other_charged"}.get(c, c.replace(" ", "_").replace("-", "_"))


def main(src, repo, outdir):
    sys.path.insert(0, src)
    from lrbench import gscdb
    mols, rx, sets, *_ = gscdb.read_repo(repo)
    nc = [n for n, r in sets.items() if r.get("Datatype_Short") == "NC"]
    man = gscdb.build_manifest(repo, nc, "", cross_check_qchem=False)
    M = man["molecules"]
    recs, groups = {}, []
    for k, m in M.items():
        recs[k] = dict(id=k, set="ext_gscdb_nc", species=[gscdb.SYM[z] for z in m["Z"]], positions=m["R"], cell=None,
                       pbc=[False] * 3, charge=int(m["charge"]), spin=int(m["mult"]), energy=None, forces=None,
                       level=LEVEL, formal_charges=None, unit_id=None, fragment_id=None, fragment_charges=None,
                       fragment_spins=None, locality=None, q_fixed_in_pair=None, cls=None, group=None, role="molecule",
                       distance=m.get("cluster_min_distance"), refs={}, flags=[],
                       extra=dict(gscdb_sets=m["sets"], n_covalent_components=m["n_covalent_components"]))
    kinds = Counter()
    for S, d in man["subsets"].items():
        for e in d["entries"]:
            pos = [(c, k) for c, k in e["terms"] if c > 0]
            two = (e.get("d_min_source") is not None and len(e.get("fragments") or []) >= 2 and len(pos) == 1
                   and all(abs(c + 1) < 1e-9 for c, k in e["terms"] if c < 0) and abs(pos[0][0] - 1) < 1e-9)
            qs = [M[k]["charge"] for _, k in e["terms"]]
            cat = cat_key(e.get("category", "other"))
            if cat in ("other", "neutral") and any(q != 0 for q in qs):
                cat = "charged_other"
            meta = dict(subset=S, theory=d["theory"], category=cat, cls=cat, series=S,
                        d_min=e.get("d_min_source"), d_min_cluster=e.get("d_min_cluster"),
                        source_coordinate=e.get("source_coordinate"), atomic_ion_fragment=e.get("atomic_ion_fragment"),
                        open_shell=e.get("open_shell"), scored_before=S in SCORED_BEFORE,
                        bootstrap_group=e["group"], pair=e["group"], q_fixed=True, locality=None,
                        ref_kcal=e["ref_kcal"],
                        flags=(["unbound_anion_candidate"] if any(q <= -2 for q in qs) else []),
                        flags_note="provisional, from charges only (multiply charged anion in a term); not DFT-verified")
            if two:
                kind = "E_int"
                meta.update(distance=e["d_min_source"], scan=e["group"])
                c = recs[e["complex"]]
                c.update(fragment_id=e["fragment_map"], fragment_charges=e["fragment_charges"],
                         fragment_spins=e["fragment_mults"], cls=cat, group=e["group"], role="AB",
                         distance=e["d_min_source"])
            else:
                kind = "dE"
            kinds[(S, kind)] += 1
            groups.append(dict(name=f"{S}:{e['id']}", kind=kind, terms=[[k, float(c)] for c, k in e["terms"]],
                               ref=e["ref_ha"] * HA_EV, meta=meta))
    os.makedirs(os.path.join(outdir, "canonical"), exist_ok=True)
    with open(os.path.join(outdir, "canonical", "ext_gscdb_nc.jsonl"), "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(outdir, "canonical", "ext_gscdb_nc.groups.json"), "w"))
    json.dump(dict(source=man["source"], kinds={f"{a}|{b}": n for (a, b), n in sorted(kinds.items())},
                   problems=man["problems"]), open(os.path.join(outdir, "canonical", "ext_gscdb_nc.meta.json"), "w"),
              indent=1)
    print(len(recs), "records,", len(groups), "groups;", dict(Counter(k for (_, k) in kinds.elements())))


if __name__ == "__main__":
    main(*sys.argv[1:4])
