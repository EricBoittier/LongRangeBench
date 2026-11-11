#!/usr/bin/env python3
"""Build the `props` set (lrbench.props) from the ORCA %elprop runs and the dimer scans.

    python props_build.py CALCDIR DIMERS_DATA_DIR OUTDIR [geometry]   # OUTDIR/props.jsonl (+ empty props.groups.json)
    ("geometry": every calc dir, references left empty where ORCA has not finished: for running the models first and
    re-scoring later with props_eval.py --rescore)

mol records: every CALCDIR/<name> with a normal ORCA termination. Parsed from p.out: total dipole (au, ORCA's origin =
centre of mass), the TOT Cartesian second moment (au, COM) turned into the traceless Buckingham form
Θ = 3/2 M - 1/2 tr(M) I (the convention of the canonical sets' quadrupole_au and frags.json theta), the static
dipole-dipole polarisability tensor (analytic CP-SCF; VV10 enters post-SCF, as in a no-gradient OMol25-settings run),
and Löwdin charges. scan records: ion+polar / ion+nonpolar dimer scans (orientation 0, unflagged points) copied from
the canonical dimers set with their Löwdin charges, for the induced-dipole curves."""
import json
import os
import re
import sys

import numpy as np

LEVEL = "wB97M-V/def2-TZVPD"


def parse(path):
    t = open(path).read()
    if "ORCA TERMINATED NORMALLY" not in t or "SCF NOT CONVERGED" in t:
        return None
    mu = re.findall(r"Total Dipole Moment\s*:\s*(\S+)\s+(\S+)\s+(\S+)", t)
    tot = re.findall(r"\n\s*TOT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\(a\.u\.\)", t)
    pol = re.search(r"The raw cartesian tensor \(atomic units\):\s*\n(.*)\n(.*)\n(.*)\n", t)
    low = re.findall(r"LOEWDIN ATOMIC CHARGES(?: AND SPIN POPULATIONS)?\n-+\n(.*?)\n\n", t, re.S)
    if not mu or not tot:
        return None
    xx, yy, zz, xy, xz, yz = map(float, tot[-1])
    M = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    out = {"dipole_au": [float(v) for v in mu[-1]],
           "quadrupole_au": (1.5 * M - 0.5 * np.trace(M) * np.eye(3)).tolist(), "second_moment_au": M.tolist()}
    if pol:
        out["alpha_au"] = [[float(v) for v in pol.group(k).split()] for k in (1, 2, 3)]
    if low:
        out["charges_lowdin"] = [float(l.split(":")[1].split()[0]) for l in low[0].strip().splitlines()]
    return out


def main(calc, data, outdir, mode="final"):
    recs = []
    for name in sorted(os.listdir(calc)):
        d = f"{calc}/{name}"
        if not os.path.exists(f"{d}/meta.json"):
            continue
        refs = parse(f"{d}/p.out") if os.path.exists(f"{d}/p.out") else None
        if refs is None and mode != "geometry":
            continue
        refs = refs or {}
        m = json.load(open(f"{d}/meta.json"))
        recs.append({"id": f"props:{name}", "set": "props", "species": m["species"], "positions": m["positions"],
                     "cell": None, "pbc": [False] * 3, "charge": m["charge"], "spin": m["mult"], "energy": None,
                     "level": LEVEL, "role": "mol", "cls": m.get("stratum"), "group": name, "refs": refs, "flags": [],
                     "extra": {k: m[k] for k in ("source", "stratum", "frag", "val_sel_clean_idx", "source_index") if k in m}})
    n_mol = len(recs)
    for l in open(f"{data}/dimers.jsonl"):
        r = json.loads(l)
        if (r.get("cls") in ("ion_polar", "ion_nonpolar") and str(r.get("group", "")).endswith(":o0")
                and r.get("role") == "AB" and not r.get("flags") and r["refs"].get("charges_lowdin")):
            recs.append({"id": f"props:scan:{r['id']}", "set": "props", "species": r["species"], "positions": r["positions"],
                         "cell": None, "pbc": [False] * 3, "charge": r["charge"], "spin": r["spin"], "energy": None,
                         "level": LEVEL, "role": "scan", "cls": r["cls"], "group": r["group"], "distance": r["distance"],
                         "fragment_id": r["fragment_id"], "fragment_charges": r["fragment_charges"],
                         "fragment_spins": r.get("fragment_spins"), "formal_charges": r.get("formal_charges"),
                         "refs": {"charges_lowdin": r["refs"]["charges_lowdin"]}, "flags": []})
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/props.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump([], open(f"{outdir}/props.groups.json", "w"))
    print(f"{outdir}/props.jsonl: {n_mol} molecules, {len(recs) - n_mol} scan points")


if __name__ == "__main__":
    main(*sys.argv[1:])
