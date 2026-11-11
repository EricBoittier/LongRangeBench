#!/usr/bin/env python3
"""Vertical IP / EA table from the run_frame.sh chains (N, N-1, N+1 per fragment; OMol25 ORCA settings, lowest
stable UKS solution of each chain).

    python props_ipea.py IPEA_CALCDIR OUT.json

Per fragment (eV): IP = E(N-1) - E(N), EA = E(N) - E(N+1), chi = (IP + EA)/2, J = IP - EA; orbital energies of the N
state (HOMO, LUMO; Koopmans -HOMO / -LUMO next to the Delta-SCF values: their gap vs J is the delocalisation-error
signature), <S^2> of each state. Flags: N+1 HOMO > 0 (anion basis-bound, kept), chain failed / no stable solution,
spin contamination (> 10 % of the next spin state's <S^2> gap)."""
import glob
import json
import os
import re
import sys

HA = 27.211386245988


def final_out(d):
    st = open(f"{d}/STATUS_r").read().split() if os.path.exists(f"{d}/STATUS_r") else ["missing"]
    return (f"{d}/{st[1]}.out" if st[0] == "stable" else None), " ".join(st)


def parse(path, mult):
    t = open(path).read()
    E = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(\S+)", t)[-1]) * HA
    occ, vir = [], []
    if "ORBITAL ENERGIES" in t:
        blk = t[t.rindex("ORBITAL ENERGIES"):]
        blk = blk[:blk.find("MULLIKEN POPULATION ANALYSIS")] if "MULLIKEN POPULATION ANALYSIS" in blk else blk
        for o, e in re.findall(r"^\s*\d+\s+([0-2]\.\d+)\s+-?\d+\.\d+\s+(-?\d+\.\d+)\s*$", blk, re.M):
            (occ if float(o) > 0 else vir).append(float(e))
    s2 = re.findall(r"Expectation value of <S\*\*2>\s*:\s*(\S+)", t)
    S = (mult - 1) / 2
    s2v = float(s2[-1]) if s2 else S * (S + 1)
    return {"E_eV": E, "HOMO_eV": max(occ) if occ else None, "LUMO_eV": min(vir) if vir else None, "S2": s2v,
            "spin_contaminated": abs(s2v - S * (S + 1)) > 0.1 * (2 * S + 2)}


def main(calc, out):
    st = {}
    for d in sorted(glob.glob(f"{calc}/*_N*")):
        m = json.load(open(f"{d}/meta.json"))
        f, status = final_out(d)
        st.setdefault(m["frag"], {"meta": m})[m["state"]] = (parse(f, m["mult"]) if f else None, status, m)
    rows = []
    for k, v in sorted(st.items()):
        m0 = v["meta"]; flags = []
        g = {s: (v.get(s) or (None, "missing", None)) for s in ("N", "Nm1", "Np1")}
        for s, (r, status, _) in g.items():
            if r is None:
                flags.append(f"{s}:{status.replace(' ', '_')}")
            elif r["spin_contaminated"]:
                flags.append(f"{s}:spin_contaminated")
        N, A, B = (g[s][0] for s in ("N", "Nm1", "Np1"))
        IP = None if N is None or A is None else A["E_eV"] - N["E_eV"]
        EA = None if N is None or B is None else N["E_eV"] - B["E_eV"]
        if B is not None and B["HOMO_eV"] is not None and B["HOMO_eV"] > 0:
            flags.append("Np1:HOMO>0_basis_bound")
        sp = m0["species"]
        rows.append({"frag": k, "source": m0.get("source"), "charge": m0["charge_N"], "mult": m0["mult_N"],
                     "formula": "".join(f"{e}{sp.count(e) if sp.count(e) > 1 else ''}" for e in sorted(set(sp))),
                     "elements": sorted(set(sp)), "n_atoms": len(sp), "IP_eV": IP, "EA_eV": EA,
                     "chi_eV": None if IP is None or EA is None else (IP + EA) / 2,
                     "J_eV": None if IP is None or EA is None else IP - EA,
                     "HOMO_N_eV": N and N["HOMO_eV"], "LUMO_N_eV": N and N["LUMO_eV"],
                     "orbital_gap_eV": None if not N or N["HOMO_eV"] is None or N["LUMO_eV"] is None else N["LUMO_eV"] - N["HOMO_eV"],
                     "S2": {s: (g[s][0] or {}).get("S2") for s in g}, "mult_states": {s: (g[s][2] or {}).get("mult") for s in g},
                     "flags": flags, **{x: m0[x] for x in ("val_sel_clean_idx", "dimer_monomer") if x in m0}})
    ok = [r for r in rows if r["J_eV"] is not None]
    res = {"definition": __doc__, "level": "wB97M-V/def2-TZVPD (ORCA 6.1.1, OMol25 SCF settings, EnGrad, UKS stability chain)",
           "n_fragments": len(rows), "n_complete": len(ok), "fragments": rows}
    json.dump(res, open(out, "w"), indent=1)
    print(out, len(rows), "fragments,", len(ok), "with IP and EA")


if __name__ == "__main__":
    main(*sys.argv[1:])
