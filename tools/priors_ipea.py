#!/usr/bin/env python3
"""IP/EA probe extxyz for `lrbench priors --ipea` from the props lane's vertical IP/EA table (tools/props_ipea.py).

    python priors_ipea.py IPEA.json IPEA_CALCDIR OUT.extxyz

One frame per fragment with a DFT IP or EA: the N-state geometry (CALCDIR/<frag>_N/geom.xyz), info charge,
spin_multiplicity, frag, IP_eV, EA_eV, mult_Nm1, mult_Np1, flags (';'-joined). IP is withheld (None) when the N-1
state is spin contaminated, EA when the N+1 state is spin contaminated or basis-bound (HOMO > 0: an unbound anion,
kept as a labelled probe and reported by ct_unbound_labelled, never scored)."""
import json
import sys

from ase.io import read, write


def main(table, calc, out):
    T = json.load(open(table))
    frames = []
    for r in T["fragments"]:
        fl = r["flags"]
        ip = None if any(f.startswith(("Nm1:", "N:")) for f in fl) else r["IP_eV"]
        ea = None if any(f.startswith(("Np1:", "N:")) for f in fl) else r["EA_eV"]
        if r["IP_eV"] is None and r["EA_eV"] is None:
            continue
        a = read(f"{calc}/{r['frag']}_N/geom.xyz")
        ms = r["mult_states"]
        a.info = {"charge": int(r["charge"]), "spin_multiplicity": int(r["mult"]), "frag": r["frag"],
                  "IP_eV": "None" if ip is None else ip, "EA_eV": "None" if ea is None else ea,
                  "mult_Nm1": int(ms["Nm1"]), "mult_Np1": int(ms["Np1"]), "flags": ";".join(fl) or "none",
                  "source": r["source"]}
        frames.append(a)
    write(out, frames, format="extxyz")
    print(out, len(frames), "probes;", sum(f.info["IP_eV"] != "None" for f in frames), "IP,",
          sum(f.info["EA_eV"] != "None" for f in frames), "EA scored")


if __name__ == "__main__":
    main(*sys.argv[1:])
