#!/usr/bin/env python3
"""Biodimer tier-A wB97M-V/def2-TZVPD relabel -> level-matched open-boundary lrbench set `ext_biodimers_wb97mv`.

    python3 -I ext_biodimer_relabel.py parse CALCDIR MANIFEST OUT.jsonl          (stdlib only, runs anywhere)
    python  -I ext_biodimer_relabel.py build PARSED.jsonl POINTS.jsonl BIODIMERS.jsonl OUTDIR

parse: per calc dir STATUS, FINAL SINGLE POINT ENERGY (Eh), Loewdin atomic charges (ALLPOP), SCF cycles.
build: records = relabelled dimer points (energy eV, level wB97M-V/def2-TZVPD, fragment ids/charges as in
ext_biodimers) + one record per rigid monomer (the curve's #frag slices, energies from their own calcs);
groups E_int = AB - monA - monB (the monomers are rigid along each curve, so one monomer calc per curve is exact).
Labels (kept, never dropped): `ct_ambiguous` when a fragment's Loewdin sum departs from its formal charge by more
than 0.1 e at closest contact >= 8 A (contact-range polarisation of the partition is not CT); the provisional
`ct_candidate` (ion pairs >= 9 A) is kept as given; failed SCFs are excluded and listed.
"""
import json
import os
import re
import sys

HA_EV = 27.211386245988


def parse(calcdir, manifest, out):
    with open(out, "w") as fo:
        for k in (l.strip() for l in open(manifest)):
            if not k:
                continue
            d = os.path.join(calcdir, k); r = dict(key=k)
            try:
                r["status"] = open(os.path.join(d, "STATUS")).read().strip()
            except OSError:
                r["status"] = "missing"
            o = os.path.join(d, "r0.out")
            if r["status"].startswith("done") and os.path.exists(o):
                txt = open(o, errors="replace").read()
                e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
                r["E_h"] = float(e[-1]) if e else None
                m = list(re.finditer(r"LOEWDIN ATOMIC CHARGES\s*\n-+\n(.*?)\n\s*\n", txt, re.S))
                if m:
                    r["q_lowdin"] = [float(l.split(":")[-1]) for l in m[-1].group(1).splitlines() if ":" in l]
                c = re.findall(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", txt)
                r["scf_cycles"] = int(c[0]) if c else None
            fo.write(json.dumps(r) + "\n")


def build(parsed, points, biodimers, outdir, fixpoints=None):
    P = {r["key"]: r for r in map(json.loads, open(parsed))}
    fixpts = {r["key"]: r for r in map(json.loads, open(fixpoints))} if fixpoints else {}
    pts = {r["key"]: r for r in map(json.loads, open(points))}
    src = {r["id"]: r for r in map(json.loads, open(biodimers))}
    recs, groups, bad = {}, [], []
    ok = lambda k: k in P and P[k]["status"].startswith("done") and P[k].get("E_h") is not None
    for k, pt in pts.items():
        if pt["role"] != "complex":
            continue
        s = src[pt["source_key"]]; cid = s["group"]
        # charge-corrected monomer calcs (source_charges_swapped curves) carry a trailing "q" and take precedence
        mons = [f"{cid}_00_f{i}q" if ok(f"{cid}_00_f{i}q") else f"{cid}_00_f{i}" for i in (0, 1)]
        if not (ok(k) and all(ok(m) for m in mons)):
            bad.append(dict(key=k, status=P.get(k, {}).get("status"), monomers=[P.get(m, {}).get("status") for m in mons]))
            continue
        for i, m in enumerate(mons):
            if m not in recs:
                mp = pts.get(m) or fixpts[m]; s0 = src[f"{cid}_00"]; fid = s0["fragment_id"]
                recs[m] = dict(id=m, set="ext_biodimers_wb97mv", species=[x for x, f in zip(s0["species"], fid) if f == i],
                               positions=mp["R"], cell=None, pbc=[False] * 3, charge=mp["charge"], spin=1,
                               energy=P[m]["E_h"] * HA_EV, forces=None, level="wB97M-V/def2-TZVPD",
                               formal_charges=None, unit_id=None, fragment_id=None, fragment_charges=None,
                               fragment_spins=None, locality=None, q_fixed_in_pair=True, cls=s["cls"], group=cid,
                               role=f"mon{'AB'[i]}", distance=None, refs={}, flags=[],
                               extra=dict(status=P[m]["status"], source="biodimer tier-A relabel, ORCA"))
        q = P[k].get("q_lowdin"); fq = None; flags = []
        if q and len(q) == len(s["species"]):
            fq = [round(sum(x for x, f in zip(q, s["fragment_id"]) if f == i), 4) for i in (0, 1)]
            if s["distance"] >= 8 and max(abs(a - b) for a, b in zip(fq, s["fragment_charges"])) > 0.1:
                flags.append("ct_ambiguous")
        e_ab = P[k]["E_h"] * HA_EV
        e_int = e_ab - recs[mons[0]]["energy"] - recs[mons[1]]["energy"]
        recs[k] = dict(s, id=k, set="ext_biodimers_wb97mv", energy=e_ab, level="wB97M-V/def2-TZVPD",
                       flags=flags, refs=dict(E_int=e_int, fragment_charges_lowdin=fq, charges_lowdin=q),
                       extra=dict(s["extra"], status=P[k]["status"], scf_cycles=P[k].get("scf_cycles"),
                                  source_id=s["id"]))
        meta_flags = flags + (["ct_candidate"] if s["cls"] == "ion_pair" and s["distance"] >= 9 else [])
        groups.append(dict(name=f"{k}:E_int", kind="E_int", terms=[[k, 1.0], [mons[0], -1.0], [mons[1], -1.0]],
                           ref=e_int, meta=dict(cls=s["cls"], distance=s["distance"], scan=cid, pair=cid,
                                                q_fixed=True, flags=meta_flags, subset=("ct-labelled " if meta_flags else "") + s["cls"],
                                                category=s["cls"], bootstrap_group=cid, source_label=s["extra"]["source_label"],
                                                E_int_source_eV=s["extra"]["E_int_source_eV"])))
    os.makedirs(os.path.join(outdir, "canonical"), exist_ok=True)
    with open(os.path.join(outdir, "canonical", "ext_biodimers_wb97mv.jsonl"), "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(outdir, "canonical", "ext_biodimers_wb97mv.groups.json"), "w"))
    json.dump(dict(n_records=len(recs), n_groups=len(groups), excluded=bad),
              open(os.path.join(outdir, "canonical", "ext_biodimers_wb97mv.meta.json"), "w"), indent=1)
    print(len(recs), "records,", len(groups), "groups,", len(bad), "excluded")


if __name__ == "__main__":
    {"parse": parse, "build": build}[sys.argv[1]](*sys.argv[2:])
