#!/usr/bin/env python3
"""Decompose dimer E_int errors at a chosen separation into dimer and monomer charge-state errors.

    python likecharge.py DATA_DIR OUT.json RESULT.json [RESULT.json ...]     # lrbench eval --save-predictions outputs

For every dimer scan point at d_min = 8 A (default; --d) the absolute errors err(X) = E_model(X) - E_DFT(X) of the
dimer AB (total charge Q) and of the isolated monomers A, B (their fragment / formal charges, rigid slices) are
reported, with E_int error = err(AB) - err(A) - err(B). Absolute energies need per-element offsets: the saved
predictions carry lrbench's in-sample fit; here it is refitted on closed-shell structures whose every fragment is
neutral (neutral monomers and dimers of two neutral fragments; ion pairs, Q = 0 but charged fragments, are excluded), so a charged structure's error keeps its charge-state part instead of having it
absorbed by the offsets. The model's own E_int splits exactly as
  E_int_model = dimer_part - monomer_part,  dimer_part = E_model(AB) - [E_DFT(A) + E_DFT(B)] = err(AB) + E_int_DFT,
  monomer_part = err(A) + err(B);
for a model without a far-field term, E_int_model beyond its cutoff is the spurious charge-state offset, so
-monomer_part / E_int_model is the share of the offset carried by the monomers' charge-state energies. Summaries by class: median err(AB), err(A) + err(B), E_int error, and the share
-(err(A) + err(B)) / E_int error (1 = the E_int error is entirely the monomers' charge-state energy)."""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")
from lrbench.data import load_set, term_structure  # noqa: E402


def main(data, out, results, d=float(__import__("os").environ.get("LRB_D", 8.0))):
    ds = load_set("dimers", data)
    res = {"definition": __doc__, "d_A": d, "models": {}}
    for path in results:
        R = json.load(open(path))
        P = R["sets"]["dimers"].get("predictions")
        if not P:
            continue
        E = {t: float(np.mean([x for x in v if x is not None])) for t, v in P.items() if any(x is not None for x in v)}
        ref, comp, neutral = {}, {}, []
        for r in ds.records:
            if r.get("role") != "AB" or ds.flagged(r["id"]):
                continue
            for t, e in ((r["id"], r["energy"]), (f"{r['id']}#frag=0", r["refs"].get("E_A")),
                         (f"{r['id']}#frag=1", r["refs"].get("E_B"))):
                if e is None or t not in E:
                    continue
                s = term_structure(ds, t)
                ref[t] = e
                comp[t] = s.species
                # fit set: closed-shell neutral structures whose every fragment is neutral (an ion pair has Q = 0 but
                # charged fragments, and would let the offsets absorb part of the dimer charge-state error)
                fq = r.get("fragment_charges") or []
                frag_neutral = all(int(q) == 0 for q in fq) if "#" not in t else True
                if s.charge == 0 and s.spin == 1 and frag_neutral:
                    neutral.append(t)
        els = sorted({x for t in ref for x in comp[t]})
        N = lambda t: np.array([comp[t].count(x) for x in els], float)
        A = np.array([N(t) for t in neutral]); y = np.array([ref[t] - E[t] for t in neutral])
        db = np.linalg.lstsq(A, y, rcond=None)[0] if len(neutral) else np.zeros(len(els))
        err = {t: (E[t] + N(t) @ db - ref[t]) * 1e3 for t in ref}
        rows = []
        for r in ds.records:
            if r.get("role") != "AB" or ds.flagged(r["id"]) or abs(float(r["distance"]) - d) > 0.01:
                continue
            ts = [r["id"], f"{r['id']}#frag=0", f"{r['id']}#frag=1"]
            if not all(t in err for t in ts):
                continue
            eAB, eA, eB = (err[t] for t in ts)
            eint_dft = (r["energy"] - r["refs"]["E_A"] - r["refs"]["E_B"]) * 1e3
            rows.append(dict(id=r["id"], cls=r["cls"], q=r["fragment_charges"], err_AB=eAB, err_A=eA, err_B=eB,
                             err_Eint=eAB - eA - eB, E_int_DFT=eint_dft, E_int_model=eAB - eA - eB + eint_dft,
                             dimer_part=eAB + eint_dft, monomer_part=eA + eB))
        summ = {}
        for c in sorted({x["cls"] for x in rows}):
            s = [x for x in rows if x["cls"] == c]
            m = lambda k: float(np.median([x[k] for x in s]))
            ei = np.array([x["err_Eint"] for x in s]); mono = np.array([x["err_A"] + x["err_B"] for x in s])
            em = np.array([x["E_int_model"] for x in s]); dp = np.array([x["dimer_part"] for x in s])
            summ[c] = dict(n=len(s), err_AB_median=m("err_AB"), err_monomers_sum_median=float(np.median(mono)),
                           err_Eint_median=float(np.median(ei)), E_int_DFT_median=m("E_int_DFT"),
                           E_int_model_median=float(np.median(em)), dimer_part_median=float(np.median(dp)),
                           monomer_part_median=float(np.median(mono)),
                           monomer_share_of_model_Eint_median=float(np.nanmedian(-mono / np.where(np.abs(em) > 1, em, np.nan)))
                           if np.any(np.abs(em) > 1) else None,
                           E_int_model_mean=float(em.mean()), E_int_model_absmax=float(np.abs(em).max()),
                           monomer_share_median=float(np.nanmedian(-mono / np.where(np.abs(ei) > 1, ei, np.nan)))
                           if np.any(np.abs(ei) > 1) else None)
        neutral_mae = float(np.mean([abs(err[t]) for t in neutral])) if neutral else None
        res["models"][R.get("label_short", R.get("label", path))] = {"offsets_refit_fit_residual_meV_per_structure_mae": neutral_mae,
                                               "n_neutral_fit": len(neutral), "by_class": summ, "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    for m, v in res["models"].items():
        for c, s in v["by_class"].items():
            if c in ("same_sign", "ion_pair", "ion_polar"):
                print(f"{m[:14]:14s} {c:9s} n={s['n']:3d} E_int DFT {s['E_int_DFT_median']:8.0f} model {s['E_int_model_median']:8.0f}"
                      f"  dimer part {s['dimer_part_median']:8.0f}  monomer part {s['monomer_part_median']:8.0f}"
                      f"  monomer share {s['monomer_share_of_model_Eint_median']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2:])
