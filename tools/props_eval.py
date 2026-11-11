#!/usr/bin/env python3
"""Score one model on the props set (lrbench.props) with optional polarisability, writing an lrbench results JSON.

    python props_eval.py --model SPEC --label L --data DIR --out OUT.json [--device cuda] [--mlipcpp-bin BIN]
                         [--ensemble SPEC2,SPEC3] [--alpha none|field|so3lr] [--field 0.005] [--model-level LEVEL]
    python props_eval.py --oracle lowdin --data DIR --out OUT.json     # charge-only floor: DFT Löwdin charges

Charges / dipoles come from the lrbench backend (GGUF latent charges via mlipcpp-eval; adapters' charges and dipole).
--alpha field: finite field ±F (V/Å) along x, y, z through atoms.info['external_field'] (MACE-POLAR-1), α_ab =
dμ_a/dF_b by central differences, converted to atomic units. --alpha so3lr: Σ_i ratio_i α_free(Z_i) (Tauri: predicted
alpha_0 ratios with the refitted free-atom α; v1: Hirshfeld ratios with the legacy table), isotropic, no coupling."""
import argparse
import datetime
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/src")
from lrbench.backends import make_backend  # noqa: E402
from lrbench.data import load_set, term_structure  # noqa: E402
from lrbench.metrics import Preds  # noqa: E402
from lrbench.props import BOHR, score  # noqa: E402

HA_EV = 27.211386245988
ALPHA_AU = HA_EV / BOHR ** 2  # e A^2 / V -> bohr^3 (atomic units of polarisability)


def field_alpha(calc, structs, F):
    """alpha = -d2E/dF dF by central differences of the energy in a uniform field (convention-free: the sign of the
    field and of the model's dipole output do not enter). Also the dipole route dmu/dF, kept as a diagnostic."""
    out, diag = [], []

    def E(a, f):
        a.info["external_field"] = list(map(float, f)); a.calc = calc
        for c in (calc, getattr(calc, "inner", None)):  # ASE caches on positions only, not on info
            if c is not None:
                c.reset()
        e = float(a.get_potential_energy())
        mu = (calc.results or {}).get("dipole")
        return e, None if mu is None else np.asarray(mu, float).reshape(-1)[:3]
    I = np.eye(3) * F
    for s in structs:
        a = s.atoms()
        try:
            e0, _ = E(a, np.zeros(3))
            ep = [E(a, I[k]) for k in range(3)]; em = [E(a, -I[k]) for k in range(3)]
            A = np.zeros((3, 3))
            for k in range(3):
                A[k, k] = -(ep[k][0] + em[k][0] - 2 * e0) / F ** 2
            for k, l in ((0, 1), (0, 2), (1, 2)):
                pp, pm = E(a, I[k] + I[l])[0], E(a, I[k] - I[l])[0]
                mp, mm = E(a, -I[k] + I[l])[0], E(a, -I[k] - I[l])[0]
                A[k, l] = A[l, k] = -(pp - pm - mp + mm) / (4 * F ** 2)
            out.append(A * ALPHA_AU)
            dm = None if ep[0][1] is None else np.array([(ep[k][1] - em[k][1]) / (2 * F) for k in range(3)]).T * ALPHA_AU
            diag.append(None if dm is None else dm.tolist())
        except Exception as err:
            print("field alpha failed", s.key, err, file=sys.stderr); out.append(None); diag.append(None)
    return out, diag


def so3lr_predict(model, structs):
    """One SO3LR pass: partial charges + alpha_0 (Tauri) / Hirshfeld (v1) ratios -> charges and isotropic alpha."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from so3lr import So3lrCalculator
    from so3lr.mlff.nn.observable.dispersion_ref_data import alphas, alphas_legacy
    legacy = model in ("so3lr", "so3lr-v1")
    key = "hirshfeld_ratios" if legacy else "a0_ratios"
    calc = So3lrCalculator(model=model, lr_cutoff=None, dispersion_energy_cutoff_lr_damping=None, dtype=np.float64,
                           observables=[key, "partial_charges"], has_aux=True)
    tab = np.asarray(alphas_legacy if legacy else alphas, float)
    out = [None] * len(structs)
    for i in sorted(range(len(structs)), key=lambda i: len(structs[i].species)):  # one JIT compile per size
        s = structs[i]; a = s.atoms(); a.info["multiplicity"] = s.spin; a.calc = calc
        if getattr(calc, "_n", None) != len(a):
            calc.neighbors, calc.spatial_partitioning, calc._n = None, None, len(a)
        try:
            e = float(a.get_potential_energy()); aux = calc.results.get("aux") or {}
            r = np.asarray(aux[key], float).reshape(-1)[:len(a)]
            q = np.asarray(aux["partial_charges"], float).reshape(-1)[:len(a)]
            out[i] = {"energy": e, "charges": q, "alpha": (np.eye(3) * float((r * tab[a.numbers - 1]).sum())).tolist()}
        except Exception as err:
            out[i] = {"energy": None, "charges": None, "error": str(err)}
    return out, f"sum of {key} x {'alphas_legacy' if legacy else 'alphas'} (so3lr dispersion_ref_data), isotropic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--ensemble", default=""); ap.add_argument("--label")
    ap.add_argument("--oracle", choices=["lowdin"]); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--mlipcpp-bin"); ap.add_argument("--alpha", default="none", choices=["none", "field", "so3lr"])
    ap.add_argument("--field", type=float, default=0.01); ap.add_argument("--model-level")
    ap.add_argument("--rescore", help="re-score an earlier props_eval output with the references in --data")
    a = ap.parse_args()
    ds = load_set("props", a.data)
    if a.rescore:
        R = json.load(open(a.rescore))
        R["sets"]["props"] = score(ds, Preds(R["props_predictions"])); R["data"] = {"props": ds.source}
        json.dump(R, open(a.out, "w"), indent=1)
        print(a.out); return
    ids = [r["id"] for r in ds.records]
    structs = [term_structure(ds, i) for i in ids]
    notes, members = [], []
    if a.oracle == "lowdin":
        members.append({r["id"]: {"energy": None, "charges": np.asarray(r["refs"]["charges_lowdin"], float)}
                        for r in ds.records if r["refs"].get("charges_lowdin")})
        label, level = "oracle: DFT Löwdin charges (charge-only floor; illustrative)", "wB97M-V/def2-TZVPD"
        notes.append("oracle uses the reference calculation's own Löwdin charges: the best a charge-only (no atomic "
                     "dipoles) model can do; not a model")
    else:
        specs = [a.model] + [s for s in a.ensemble.split(",") if s]
        label = a.label or os.path.basename(a.model)
        from lrbench.data import default_model_level
        level = a.model_level or default_model_level(a.model)
        if a.alpha == "so3lr":
            model = a.model.split("model='")[1].split("'")[0] if "model='" in a.model else "so3lr-tauri-m-2606"
            preds, how = so3lr_predict(model, structs)
            members.append(dict(zip(ids, preds))); notes.append("charges: SO3LR partial charges; alpha: " + how)
        else:
            for sp in specs:
                be = make_backend(sp, device=a.device, mlipcpp_bin=a.mlipcpp_bin)
                members.append(dict(zip(ids, be.predict(structs))))
            if a.alpha == "field":
                calc = getattr(be, "calc", None)
                al, dmu = field_alpha(calc, structs, a.field) if calc is not None else ([None] * len(structs),) * 2
                notes.append(f"alpha: -d2E/dF2 by central differences, uniform field +-{a.field} V/A through "
                             "atoms.info['external_field'] (19 energies per molecule), atomic units; the model's "
                             "dipole-output derivative dmu/dF is kept per molecule as 'alpha_from_dipole_au' (diagnostic)")
                for i, x, y in zip(ids, al, dmu):
                    if x is not None:
                        members[0][i]["alpha"] = np.asarray(x).tolist()
                        members[0][i]["alpha_from_dipole"] = y
    S = score(ds, Preds(members))
    keep = ("charges", "dipole", "alpha", "alpha_from_dipole", "error")
    pred_out = [{i: {k: (np.asarray(v).tolist() if v is not None and k != "error" else v) for k, v in p.items() if k in keep}
                 for i, p in m.items()} for m in members]
    ref_levels = sorted({r.get("level", "?") for r in ds.records})
    if any(level != x for x in ref_levels):
        notes.insert(0, f"LEVEL MISMATCH: model {level} vs references {', '.join(ref_levels)}")
    R = {"label": label, "created": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
         "argv": sys.argv[1:], "level": {"model_level": level, "reference_levels": ref_levels,
                   "mismatch": {"props": [x for x in ref_levels if x != level]} if any(level != x for x in ref_levels) else {}},
         "members": [{"spec": s} for s in ([a.model] + [x for x in a.ensemble.split(",") if x] if a.model else ["oracle"])],
         "sets": {"props": S}, "notes": notes, "data": {"props": ds.source}, "props_predictions": pred_out}
    json.dump(R, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    d = S.get("dipole", {}).get("all", {})
    print(label, "dipole vec MAE", d.get("vector_err", {}).get("MAE"), "slope", d.get("slope"),
          "alpha", (S.get("polarisability") or {}).get("iso_err", {}).get("MAE"))


if __name__ == "__main__":
    main()
