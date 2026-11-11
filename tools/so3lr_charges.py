#!/usr/bin/env python3
"""Analyse tools/so3lr_terms.py outputs: where does SO3LR(-T)'s far-field E_int offset live?

    python so3lr_charges.py OUT.json FIG.json TERMS.json [TERMS.json ...]

Per model, class and d_min:
  - fragment charge sums q_A, q_B in AB vs formal and vs DFT Loewdin (dimer records); leak = |q_neutral| on ion+neutral;
  - E_int (model) vs DFT, and its exact split dE_nn + dE_elec + dE_disp + dE_rep (+ any remainder: E - sum of terms);
  - dE_elec = inter-fragment Coulomb of the model's AB charges (bare 1/r; at d >= 8 A the damping is negligible)
              + an intra-fragment remainder (charge redistribution inside each fragment vs the isolated monomer);
  - counterfactual (a): E_int with the inter-fragment Coulomb recomputed from formal fragment charges (the isolated
    monomers' own model charges, which sum to the formal charge) or from DFT Loewdin charges, everything else kept;
  - attribution of the offset at d >= 12 A, offset = E_int - E_int_DFT:
        charge-partition part = (Coulomb of model AB charges - Coulomb of DFT charges) + intra-fragment remainder,
        NN part = dE_nn (charge-state energetics of the network), other = dE_disp + dE_rep + remainder - (E_int_DFT - Coulomb_DFT).
Medians over scans/orientations."""
import json
import sys
from collections import defaultdict

import numpy as np


def main(out, fig, *files):
    res = {"definition": __doc__, "models": {}}
    figd = {}
    for f in files:
        D = json.load(open(f)); m = D["model"]; rep = [k for k in D["observables"] if "repulsion" in k][0]
        rows = [r for r in D["rows"] if "error" not in r]
        for r in rows:
            terms = sum(r.get(f"dE_{k}", 0.0) for k in ("nn_energy", "electrostatic_energy", "dispersion_energy", rep))
            r["dE_remainder"] = r["E_int"] - terms
            r["elec_intra"] = r.get("dE_electrostatic_energy", np.nan) - r["coul_model_AB"]
            r["E_int_cf_formal"] = r["E_int"] - r["coul_model_AB"] + r["coul_model_iso"]
            if r.get("coul_lowdin") is not None:
                r["E_int_cf_lowdin"] = r["E_int"] - r["coul_model_AB"] + r["coul_lowdin"]
            if r.get("E_int_DFT") is not None:
                r["offset"] = r["E_int"] - r["E_int_DFT"]
                if r.get("coul_lowdin") is not None:
                    r["part_charge"] = (r["coul_model_AB"] - r["coul_lowdin"]) + r["elec_intra"]
                    r["part_nn"] = r.get("dE_nn_energy", np.nan)
                    r["part_other"] = r["offset"] - r["part_charge"] - r["part_nn"]
            qf = r["qf"]
            r["dqA_formal"] = r["qA"] - qf[0]; r["dqB_formal"] = r["qB"] - qf[1]
            if r.get("qA_lowdin") is not None:
                r["dqA_lowdin"] = r["qA"] - r["qA_lowdin"]
        by = defaultdict(list)
        for r in rows:
            by[(r["cls"], r["d"])].append(r)
        tab = {}
        for (c, d), rs in sorted(by.items()):
            med = lambda k: (float(np.nanmedian([x.get(k, np.nan) if x.get(k) is not None else np.nan for x in rs]))
                             if any(x.get(k) is not None for x in rs) else None)
            tab[f"{c}|{d:g}"] = {k: med(k) for k in (
                "E_int", "E_int_DFT", "offset", "dE_nn_energy", "dE_electrostatic_energy", "dE_dispersion_energy",
                f"dE_{rep}", "dE_remainder", "coul_model_AB", "coul_model_iso", "coul_lowdin", "elec_intra",
                "E_int_cf_formal", "E_int_cf_lowdin", "qA", "qB", "qA_iso", "qB_iso", "qA_lowdin", "dqA_formal",
                "dqB_formal", "dqA_lowdin", "part_charge", "part_nn", "part_other")}
            tab[f"{c}|{d:g}"]["n"] = len(rs)
        far = defaultdict(list)
        for r in rows:
            if r["d"] >= 12 and r.get("offset") is not None and r.get("part_charge") is not None:
                far[r["cls"]].append(r)
        attrib = {c: {k: float(np.median([x[k] for x in rs])) for k in ("offset", "part_charge", "part_nn", "part_other")} | {"n": len(rs)}
                  for c, rs in far.items()}
        res["models"][m] = {"by_class_d": tab, "far_attribution_d_ge_12": attrib,
                            "n_failed": sum("error" in r for r in D["rows"]),
                            "failed_examples": [r["error"] for r in D["rows"] if "error" in r][:3]}
        figd[m] = [{k: r.get(k) for k in ("id", "cls", "d", "E_int", "E_int_DFT", "qA", "qB", "qA_lowdin", "dE_nn_energy",
                                          "dE_electrostatic_energy", "dE_dispersion_energy", f"dE_{rep}", "E_int_cf_formal",
                                          "E_int_cf_lowdin")} for r in rows]
        print(m)
        for c, v in attrib.items():
            print(f"   {c:12s} n={v['n']:3d} offset {v['offset']:8.1f}  charge-partition {v['part_charge']:8.1f}  NN {v['part_nn']:8.1f}  other {v['part_other']:8.1f} meV")
        for k, v in tab.items():
            if k.startswith("sizecons") and (k.endswith("|32") or k.endswith("|100")):
                print(f"   {k:24s} E_int {v['E_int']:8.1f}  nn {v['dE_nn_energy']:8.1f} elec {v['dE_electrostatic_energy']:8.1f} "
                      f"disp {v['dE_dispersion_energy']:6.2f} qA {v['qA']:.3f} qB {v['qB']:.3f}")
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump({"note": "per-row data for figures: E_int split and fragment charges vs d (meV, e)", "models": figd},
              open(fig, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main(*sys.argv[1:])
