#!/usr/bin/env python3
"""Collect props results (props_eval.py outputs) into one figure-ready JSON + markdown table, and cross-check the ORCA
references against the dimer build's gpu4pyscf moments (frags.json: same functional/basis, alpha by finite field).

    python props_summary.py DATA_DIR FRAGS.json OUT_PREFIX RESULT.json [RESULT.json ...]
    -> OUT_PREFIX.json, OUT_PREFIX.md"""
import ast
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")
from lrbench.props import BOHR  # noqa: E402,F401


def crosscheck(data, frags):
    L = lambda z: ast.literal_eval(z) if isinstance(z, str) else z
    F = json.load(open(frags)); out = []
    for l in open(f"{data}/props.jsonl"):
        r = json.loads(l)
        k = (r.get("extra") or {}).get("frag")
        if not k or not r["refs"].get("dipole_au") or k not in F or "mu" not in F[k]:
            continue
        f = F[k]
        row = {"frag": k, "dmu": float(np.linalg.norm(np.ravel(L(f["mu"]))[:3] - np.asarray(r["refs"]["dipole_au"]))),
               "dtheta": float(np.abs(np.reshape(L(f["theta"]), (3, 3)) - np.asarray(r["refs"]["quadrupole_au"])).max())}
        if f.get("alpha") not in (None, "None") and r["refs"].get("alpha_au"):
            a1, a2 = np.reshape(L(f["alpha"]), (3, 3)), np.asarray(r["refs"]["alpha_au"])
            row["alpha_iso_gpu4pyscf_ff"], row["alpha_iso_orca_cpscf"] = float(np.trace(a1) / 3), float(np.trace(a2) / 3)
        out.append(row)
    return out


def main(data, frags, prefix, *results):
    rows, md = {}, []
    f = lambda x, k=3: "–" if x is None else f"{x:.{k}f}"
    md += ["| model | level | μ vector MAE all (e bohr) [95 % CI] | neutral | cation | anion | open shell | μ slope | "
           "Θ norm MAE (e bohr²) | α_iso MAE (bohr³) | α slope | Δμ(d) MAE <8 / ≥8 Å (e bohr) |", "|" + "---|" * 12]
    for p in results:
        R = json.load(open(p)); S = R["sets"]["props"]; D = S.get("dipole") or {}
        lab = R["label"]; lv = R["level"]["model_level"]
        g = lambda s: (D.get(s) or {}).get("vector_err", {}).get("MAE")
        A = S.get("polarisability") or {}; I = (S.get("induced_dipole") or {}).get("vector_err_by_region") or {}
        ci = (D.get("all") or {}).get("vector_err", {}).get("MAE_ci95") or [None, None]
        rows[lab] = {"file": p, "level": lv, "notes": R.get("notes"), "dipole": D, "quadrupole": S.get("quadrupole"),
                     "polarisability": {k: v for k, v in A.items()},
                     "induced_dipole": {"vector_err_by_region": I,
                                        "curves": (S.get("induced_dipole") or {}).get("curves")},
                     "status": S.get("status")}
        md.append(f"| {lab} | {lv} | {f(g('all'))} [{f(ci[0])}, {f(ci[1])}] | {f(g('neutral'))} | {f(g('cation'))} | "
                  f"{f(g('anion'))} | {f(g('open_shell'))} | {f((D.get('all') or {}).get('slope'), 2)} | "
                  f"{f(((S.get('quadrupole') or {}).get('all') or {}).get('norm_err', {}).get('MAE'))} | "
                  f"{f(A.get('iso_err', {}).get('MAE'), 2) if 'iso_err' in A else 'N/A'} | "
                  f"{f(A.get('slope_iso'), 2) if 'iso_err' in A else '–'} | "
                  f"{f(I.get('<8', {}).get('MAE'))} / {f(I.get('>=8', {}).get('MAE'))} |")
    cc = crosscheck(data, frags)
    out = {"definition": "lrbench props suite; references wB97M-V/def2-TZVPD (ORCA 6.1.1, OMol25 settings, %elprop); "
                         "μ, Θ about the COM (origin-dependent for ions), α static dipole-dipole; atomic units",
           "models": rows, "reference_crosscheck_gpu4pyscf": cc}
    json.dump(out, open(f"{prefix}.json", "w"), indent=1)
    md += ["", "μ, Θ about the centre of mass (origin-dependent for ions; same origin for model and reference). Latent "
           "charges (LES-type) are not trained to be physical charges: scored as given, slope = their scale. α: "
           "-d²E/dF² (field-capable models), Σ ratio·α_free (SO3LR, isotropic, no coupling), else N/A. Δμ(d): neutral "
           "fragment, charges only, vs DFT Löwdin charges."]
    if cc:
        md += ["", f"Reference cross-check vs the dimer build (gpu4pyscf, {len(cc)} monomers): max |Δμ| "
               f"{max(x['dmu'] for x in cc):.2e} e bohr, max |ΔΘ| {max(x['dtheta'] for x in cc):.2e} e bohr²"
               + (", α_iso ORCA CP-SCF vs finite field: max |Δ| "
                  f"{max(abs(x['alpha_iso_orca_cpscf'] - x['alpha_iso_gpu4pyscf_ff']) for x in cc if 'alpha_iso_orca_cpscf' in x):.3f} bohr³"
                  if any('alpha_iso_orca_cpscf' in x for x in cc) else "")]
    open(f"{prefix}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
