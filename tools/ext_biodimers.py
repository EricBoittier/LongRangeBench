#!/usr/bin/env python3
"""BFDB side-chain biodimer scans (Huguenin-Dumittan, Loche, Ni, Ceriotti, JPCL 14, 9612 (2023); Materials Cloud
2023.151 v1 record 405an-d8183, CC BY 4.0) -> lrbench canonical set `ext_biodimers` + an ORCA relabel plan.

    python -I ext_biodimers.py LRBENCH_SRC bio_dimers.xyz BAS OUTDIR [--seed 0]

Source facts (checked on the files): 2,291 rigid-monomer curves x 13 points (29,783 dimers), H C N O only, every
dimer even-electron (closed shell at the published fragment charges). Reference: FHI-aims HSE06 with
hybrid_xc_coeff 0.75 + MBD-NL, computed in a 30 A cubic cell (k_grid_density 4), monomers included although the
monomer file says pbc F. Checked here (tools/ext_biodimer_pbc_check.py; results/biodimer_offsets.json): source E_int
minus the open-boundary point-charge Coulomb of the fragment centroids, regressed on r^2 per curve, is ~0 for neutral
and ion-neutral pairs, -1.39 eV + 1.09e-3 r^2 for like-charge pairs (monopole cell term of a 30 A cell:
-1.362 + 1.12e-3 r^2) and +1.11 eV + 1.4e-4 r^2 for ion pairs (constant Madelung term of the two charged monomer
cells; ion-pair shapes follow open-boundary Coulomb to a median 25 meV). So absolute charged-pair E_int is a
periodic-cell reference (flag `charged_cell_offset`; ion-pair E_int crosses zero at 9-10 A); offset-free shapes are
usable (like-charge after removing the r^2 cell term, done in ext_biodimer_score.py). Records carry
`energy_do_not_score` so lrbench's own E_int scorer skips them; the source E_int sits in `extra.E_int_source_eV`.

Records: one per dimer (cell dropped, open boundary), fragment_id 0 = the first indexB atoms (A), 1 = B,
fragment_charges [chargeA, chargeB], spins [1, 1], cls from the charges (ion_pair, like_charge, ion_neutral, neutral;
source class label CC/CA/CP/AA/PA/PP in extra), distance = closest A-B atom distance, group = curve.
Source charge labels: in 248 of 477 ion-neutral curves the published chargeA/chargeB sit on the wrong side of
indexB (both monomers would be odd-electron); charges are reassigned by electron parity and the record flagged
`source_charges_swapped`. Group meta.flags `ct_candidate` (provisional, not DFT-verified) on ion-pair points >= 9 A.
Groups: E_int = AB - AB#frag=0 - AB#frag=1 (vertical; the monomers are rigid along each curve).
Relabel plan: stratified curve selection, one ORCA point per dimer + one per rigid monomer per curve.
"""
import collections
import json
import os
import sys

import numpy as np

LEVEL = "HSE06(0.75 HF)+MBD-NL/FHI-aims (dimers periodic 30 A, monomers open)"
TIERS = {"ion_pair": None, "like_charge": None, "ion_neutral": 100, "neutral_polar": 40, "neutral_apolar": 10}


def cls_of(qa, qb):
    return ("ion_pair" if qa * qb < 0 else "like_charge" if qa * qb > 0 else "ion_neutral" if qa or qb
            else "neutral")


def main(src, xyz, bas, outdir, seed=0):
    sys.path.insert(0, src)
    from ase.io import iread
    from lrbench.gscdb import tzvpd_counts
    nbf_el = tzvpd_counts(bas)
    recs, groups, curves = [], [], collections.defaultdict(list)
    for a in iread(xyz, index=":"):
        i = a.info; nb = int(i["indexB"]); R = a.positions; sp = a.get_chemical_symbols()
        qa, qb = int(i["chargeA"]), int(i["chargeB"])
        za, zb = int(a.numbers[:nb].sum()), int(a.numbers[nb:].sum())
        swapped = (za - qa) % 2 == 1 and (zb - qb) % 2 == 1 and (za - qb) % 2 == 0 and (zb - qa) % 2 == 0
        if swapped:  # source chargeA/chargeB do not belong to the atoms before/after indexB (both monomers odd-electron)
            qa, qb = qb, qa
        dmin = float(np.linalg.norm(R[:nb, None] - R[None, nb:], axis=-1).min())
        cid = f"bd{int(i['dimer_id']):04d}"
        rid = f"{cid}_{len(curves[cid]):02d}"
        c = cls_of(qa, qb)
        e_int = float(a.get_potential_energy() - i["energyA"] - i["energyB"])
        recs.append(dict(id=rid, set="ext_biodimers", species=sp, positions=R.round(8).tolist(), cell=None,
                         pbc=[False] * 3, charge=qa + qb, spin=1, energy=None, forces=None, level=LEVEL,
                         formal_charges=None, unit_id=None, fragment_id=[0] * nb + [1] * (len(a) - nb),
                         fragment_charges=[qa, qb], fragment_spins=[1, 1], locality=None, q_fixed_in_pair=True,
                         cls=c, group=cid, role="AB", distance=round(dmin, 4), refs={},
                         flags=["energy_do_not_score", "source_cell_30A"] + (["charged_cell_offset"] if qa * qb else [])
                         + (["source_charges_swapped"] if swapped else []),
                         extra=dict(source_label=i["label"], dimer_id=int(i["dimer_id"]),
                                    distance_source=float(i["distance"]),
                                    distance_initial_source=float(i["distance_initial"]),
                                    E_int_source_eV=e_int, source_record="materialscloud 405an-d8183 bio_dimers.xyz")))
        curves[cid].append(recs[-1])
        groups.append(dict(name=f"{rid}:E_int", kind="E_int",
                           terms=[[rid, 1.0], [f"{rid}#frag=0", -1.0], [f"{rid}#frag=1", -1.0]], ref=e_int,
                           meta=dict(cls=c, distance=round(dmin, 4), scan=cid, pair=cid, q_fixed=True,
                                     source_label=i["label"], flags=["source_cell_30A"] + (["charged_cell_offset"] if qa * qb else [])
                                     + (["ct_candidate"] if qa * qb < 0 and dmin >= 9 else []),
                                     flags_note="ct_candidate is provisional (ion pair >= 9 A, where gas-phase CT states "
                                                "can undercut the ionic state); not DFT-verified")))
    os.makedirs(os.path.join(outdir, "canonical"), exist_ok=True)
    with open(os.path.join(outdir, "canonical", "ext_biodimers.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(groups, open(os.path.join(outdir, "canonical", "ext_biodimers.groups.json"), "w"))

    # ---- relabel plan: stratified curves, all 13 points + 2 rigid monomers per curve
    rng = np.random.default_rng(seed)
    polar = lambda rs: rs[0]["extra"]["source_label"] in ("PP", "PA")
    strata = collections.defaultdict(list)
    for cid, rs in curves.items():
        c = rs[0]["cls"]
        strata[c if c != "neutral" else "neutral_polar" if polar(rs) else "neutral_apolar"].append(cid)
    nbf = lambda sp: int(sum(nbf_el[s] for s in sp))
    cost_path = os.environ.get("LRBENCH_COST_FIT", "")
    if not cost_path or not os.path.exists(cost_path):
        raise SystemExit("set LRBENCH_COST_FIT to a campaign cost fit JSON "
                         '({"fit": {"a":..,"b":..,"resid_q05":..,"resid_q95":..}}); core_h = exp(a) nbf^b)')
    cost = json.load(open(cost_path))["fit"]
    est = lambda n: float(np.exp(cost["a"]) * n ** cost["b"])
    calcs, summ = [], {}
    for s, ids in sorted(strata.items()):
        k = TIERS[s]
        pick = sorted(ids) if k is None or k >= len(ids) else sorted(rng.choice(sorted(ids), k, replace=False))
        n0 = len(calcs)
        for cid in pick:
            rs = curves[cid]
            for r in rs:
                calcs.append(dict(key=r["id"], stratum=s, n_atoms=len(r["species"]), charge=r["charge"], mult=1,
                                  nbf_tzvpd=nbf(r["species"]), role="complex"))
            r = rs[0]; fid = np.array(r["fragment_id"])
            for k_, q in enumerate(r["fragment_charges"]):
                sp = [x for x, f in zip(r["species"], fid) if f == k_]
                calcs.append(dict(key=f"{r['id']}#frag={k_}", stratum=s, n_atoms=len(sp), charge=q, mult=1,
                                  nbf_tzvpd=nbf(sp), role="monomer (rigid along the curve)"))
        for c in calcs[n0:]:
            c["est_core_h"] = est(c["nbf_tzvpd"])
        summ[s] = dict(n_curves_available=len(ids), n_curves=len(pick), n_calcs=len(calcs) - n0,
                       cpu_core_h=round(sum(c["est_core_h"] for c in calcs[n0:]), 1))
    tot = sum(c["est_core_h"] for c in calcs)
    plan = dict(level="wB97M-V/def2-TZVPD (ORCA, OMol25 protocol, open boundary)", strata=summ, n_calcs=len(calcs),
                nbf_max=max(c["nbf_tzvpd"] for c in calcs), cpu_core_h=round(tot, 1),
                cpu_core_h_band_5_95=[round(tot * np.exp(cost["resid_q05"]), 1), round(tot * np.exp(cost["resid_q95"]), 1)],
                allocated_core_h=round(tot / 0.62, 1), chf=round(tot / 0.62 * 0.0056, 2),
                cost_model="$LRBENCH_COST_FIT (core_h = exp(a) nbf^b, packing 0.62)",
                ion_pair_note="ion pairs at >= 9 A may sit on a CT state in DFT: run with UKS stability like the P2 "
                              "Mg/Ca scans and keep ct_ambiguous / unbound labels")
    os.makedirs(os.path.join(outdir, "calcs"), exist_ok=True)
    with open(os.path.join(outdir, "calcs", "biodimers_relabel.jsonl"), "w") as f:
        for c in calcs:
            f.write(json.dumps(c) + "\n")
    json.dump(plan, open(os.path.join(outdir, "calcs", "biodimers_relabel_cost.json"), "w"), indent=1)
    print(len(recs), "records,", len(groups), "groups;", json.dumps(plan, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
