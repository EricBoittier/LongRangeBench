#!/usr/bin/env python3
"""Parse the P2 ORCA runs (the ORCA driver scripts of the P2 campaign): final stable solution per calc dir ->
E (eV), <S^2>, Loewdin charges summed per fragment, HOMO/LUMO (eV); then E_int(d) = E(AB) - E(A) - E(B) per system,
the singlet-triplet gap for Mg2+/Ca2+-H2O and the multipole reference (charge-dipole + induction would need moments;
here the bare charge-dipole -q mu cos / r^2 from the DFT water dipole is NOT applied: E_int is reported as computed).

    python p2_parse.py CALCDIR OUT.json"""
import glob
import json
import os
import re
import sys

HA = 27.211386245988


def parse(d):
    st = open(f"{d}/STATUS_r").read().split() if os.path.exists(f"{d}/STATUS_r") else ["missing"]
    meta = json.load(open(f"{d}/meta.json"))
    out = dict(meta, status=" ".join(st))
    if st[0] != "stable" or not os.path.exists(f"{d}/{st[1]}.out"):
        return out
    t = open(f"{d}/{st[1]}.out").read()
    out["E_eV"] = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", t)[-1]) * HA
    s2 = re.findall(r"Expectation value of <S\*\*2>\s*:\s*(-?\d+\.\d+)", t)
    out["S2"] = float(s2[-1]) if s2 else 0.0
    blk = t.rsplit("LOEWDIN ATOMIC CHARGES", 1)
    if len(blk) == 2:
        q = []
        for l in blk[1].splitlines()[2:2 + len(meta["species"])]:
            q.append(float(l.split(":")[1].split()[0]))
        out["q_lowdin"] = q
        nA = meta.get("nA")
        if nA:
            out["qA_lowdin"] = sum(q[:nA])
    orb = t.rsplit("ORBITAL ENERGIES", 1)
    if len(orb) == 2:
        occ, virt = [], []
        for l in orb[1].splitlines()[4:]:
            p = l.split()
            if len(p) != 4:
                if occ and virt:
                    break
                continue
            (occ if float(p[1]) > 0 else virt).append(float(p[3]))
        out["HOMO_eV"] = max(occ) if occ else None
        out["LUMO_eV"] = min(virt) if virt else None
    out["stab_steps"] = int(st[1][1:])
    return out


def main(cdir, outp):
    R = {os.path.basename(d.rstrip("/")): parse(d.rstrip("/")) for d in sorted(glob.glob(f"{cdir}/*/")) if os.path.exists(f"{d}/meta.json")}
    mono = {}
    for k, r in R.items():
        if r.get("role") in ("A", "B") and "E_eV" in r:
            mono[(r["sys"], r["role"])] = r["E_eV"]
    # monomers named H2O_mono_<ion> are role B of <ion>_H2O
    rows = []
    for k, r in R.items():
        if r.get("role") != "AB" or "E_eV" not in r:
            continue
        eA, eB = mono.get((r["sys"], "A")), mono.get((r["sys"], "B"))
        rows.append(dict(name=k, sys=r["sys"], d=r["d"], state=r.get("state", "singlet"), mult=r["mult"],
                         E_int_meV=None if eA is None or eB is None else (r["E_eV"] - eA - eB) * 1e3,
                         qA_lowdin=r.get("qA_lowdin"), S2=r.get("S2"), stab_steps=r.get("stab_steps"),
                         HOMO_eV=r.get("HOMO_eV"), LUMO_eV=r.get("LUMO_eV")))
    json.dump({"level": "wB97M-V/def2-TZVPD (ORCA 6.1.1, OMol25 settings, UKS stability chain)", "calcs": R,
               "scans": sorted(rows, key=lambda x: (x["sys"], x["state"], x["d"]))}, open(outp, "w"), indent=1)
    for x in sorted(rows, key=lambda x: (x["sys"], x["state"], x["d"])):
        print(f"{x['sys']:8s} {x['state']:8s} d={x['d']:5} E_int={x['E_int_meV'] if x['E_int_meV'] is None else round(x['E_int_meV'], 2)} "
              f"qA={x['qA_lowdin'] if x['qA_lowdin'] is None else round(x['qA_lowdin'], 3)} S2={x['S2']} steps={x['stab_steps']} HOMO={x['HOMO_eV']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
