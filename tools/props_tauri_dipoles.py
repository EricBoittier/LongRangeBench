#!/usr/bin/env python3
"""SO3LR-Tauri Zenodo dipole sets (test_benchmarks/table2_external: qm7b.xyz 7,211 and alphaml.xyz 52 neutral
molecules, Yang et al. 2019) -> a props-schema data dir (role mol, refs.dipole_au only). Reference level
LR-CCSD/d-aug-cc-pVDZ unrelaxed dipoles, NOT wB97M-V: every table must say so. Neutral, so μ is origin-independent.
    python props_tauri_dipoles.py qm7b.xyz alphaml.xyz OUTDIR"""
import json, os, sys
from ase.io import read
*srcs, out = sys.argv[1:]
os.makedirs(out, exist_ok=True); n = 0
with open(f"{out}/props.jsonl", "w") as f:
    for src in srcs:
        for i, a in enumerate(read(src, ":")):
            nm = a.info.get("set_name", os.path.basename(src).split(".")[0])
            mu = a.info["ref_dipole_au"]; mu = [float(v) for v in (mu.split() if isinstance(mu, str) else mu)]
            f.write(json.dumps({"id": f"{nm}:{a.info.get('mol_id', i)}", "set": "props", "species": a.get_chemical_symbols(),
                                "positions": a.positions.tolist(), "cell": None, "pbc": [False] * 3,
                                "charge": int(a.info.get("charge", 0)), "spin": int(a.info.get("multiplicity", 1)),
                                "energy": None, "level": "LR-CCSD/d-aug-cc-pVDZ", "role": "mol", "cls": nm, "group": nm,
                                "refs": {"dipole_au": mu}, "flags": [], "extra": {"source": nm, "stratum": nm}}) + "\n"); n += 1
json.dump([], open(f"{out}/props.groups.json", "w")); print(out, n)
