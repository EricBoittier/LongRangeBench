#!/usr/bin/env python3
"""SO3LR / SO3LR-Tauri internals on the dimer scans: equilibrated partial charges, dipole and the energy split
(NN, electrostatic, dispersion, repulsion) for AB and for the monomers A, B alone at their formal charges.

    python so3lr_terms.py MODEL DATA_DIR SCANS_DIR OUT.json     (MODEL: so3lr | so3lr-tauri-{s,m,l}-2606)

Structures: canonical `dimers` AB records of classes ion_pair, same_sign, ion_polar, ion_nonpolar at d_min >= 4 A (all
orientations; pathology-flagged records skipped), and the size-consistency frames (scans/sizecons.extxyz, 8-100 A).
For each: per-term E(AB) - E(A) - E(B) (meV), fragment charge sums q_A, q_B in AB (and in the isolated monomers),
the inter-fragment bare Coulomb of (i) the model's AB charges, (ii) the isolated monomers' model charges placed in the AB
geometry, (iii) the dimer's DFT Loewdin charges (where the record has them), and the DFT E_int. Gas phase, float64,
no long-range cutoff (as lrbench.adapters.so3lr)."""
import json
import sys

import numpy as np

KE = 14.399645  # eV A


def calc(model):
    import jax
    jax.config.update("jax_enable_x64", True)
    from so3lr import So3lrCalculator
    rep = "zbl_repulsion" if model in ("so3lr", "so3lr-v1") else "nlh_repulsion"
    obs = ["partial_charges", "nn_energy", "electrostatic_energy", "dispersion_energy", rep]
    c = So3lrCalculator(model=model, lr_cutoff=None, dispersion_energy_cutoff_lr_damping=None, dtype=np.float64,
                        observables=obs, has_aux=True)
    return c, obs, rep


def ev(c, obs, sp, x, q, m):
    from ase import Atoms
    a = Atoms(sp, positions=x)
    a.info.update(charge=int(q), multiplicity=int(m))
    c.neighbors, c.spatial_partitioning = None, None  # cached neighbour list assumes a fixed atom count
    a.calc = c
    e = float(a.get_potential_energy())
    aux = (c.results.get("aux") or {})
    out = {"E": e}
    for k in obs:
        v = aux.get(k)
        if v is not None:
            v = np.asarray(v, float).reshape(len(sp), -1)[:, 0]
            out[k] = v if k == "partial_charges" else float(v.sum())
    return out


def coul(qa, xa, qb, xb):
    r = np.linalg.norm(xa[:, None] - xb[None], axis=-1)
    return float(KE * np.sum(qa[:, None] * qb[None] / r)) * 1e3


def main(model, data, scans, out):
    c, obs, rep = calc(model)
    items = []
    for l in open(f"{data}/dimers.jsonl"):
        r = json.loads(l)
        if (r.get("role") == "AB" and r["cls"] in ("ion_pair", "same_sign", "ion_polar", "ion_nonpolar")
                and float(r["distance"]) >= 3.99 and "dft_pathology_do_not_score" not in (r.get("flags") or [])):
            fid = np.asarray(r["fragment_id"]); x = np.asarray(r["positions"])
            ql = (r.get("refs") or {}).get("charges_lowdin")
            items.append(dict(id=r["id"], cls=r["cls"], d=float(r["distance"]), sp=r["species"], x=x, fid=fid,
                              q=r["charge"], qf=r["fragment_charges"], ql=None if ql is None else np.asarray(ql),
                              eint_dft=(r["energy"] - r["refs"]["E_A"] - r["refs"]["E_B"]) * 1e3))
    from ase.io import read
    S = read(f"{scans}/sizecons.extxyz", ":")
    for i in range(0, len(S), 3):
        ab = S[i]; nA = len(S[i + 1])
        fid = np.r_[np.zeros(nA, int), np.ones(len(ab) - nA, int)]
        items.append(dict(id=f"sizecons:{ab.info['sys']}:d{ab.info['d']}", cls=f"sizecons:{ab.info['sys']}", d=float(ab.info["d"]),
                          sp=ab.get_chemical_symbols(), x=ab.positions, fid=fid, q=int(ab.info["charge"]),
                          qf=[int(S[i + 1].info["charge"]), int(S[i + 2].info["charge"])], ql=None, eint_dft=None))
    rows = []
    for it in items:
        sp, x, fid = it["sp"], it["x"], it["fid"]
        ia, ib = np.where(fid == 0)[0], np.where(fid == 1)[0]
        try:
            AB = ev(c, obs, sp, x, it["q"], 1)
            A = ev(c, obs, [sp[i] for i in ia], x[ia], it["qf"][0], 1)
            B = ev(c, obs, [sp[i] for i in ib], x[ib], it["qf"][1], 1)
        except Exception as e:  # unsupported element etc.
            rows.append(dict(id=it["id"], cls=it["cls"], d=it["d"], error=f"{type(e).__name__}: {e}"[:200]))
            continue
        row = dict(id=it["id"], cls=it["cls"], d=it["d"], qf=it["qf"], E_int_DFT=it["eint_dft"],
                   E_int=(AB["E"] - A["E"] - B["E"]) * 1e3)
        for k in obs[1:]:
            if k in AB:
                row[f"dE_{k}"] = (AB[k] - A[k] - B[k]) * 1e3
        q = AB["partial_charges"]
        row.update(qA=float(q[ia].sum()), qB=float(q[ib].sum()), qA_iso=float(A["partial_charges"].sum()),
                   qB_iso=float(B["partial_charges"].sum()),
                   coul_model_AB=coul(q[ia], x[ia], q[ib], x[ib]),
                   coul_model_iso=coul(A["partial_charges"], x[ia], B["partial_charges"], x[ib]),
                   dipole_AB=(q[:, None] * (x - x.mean(0))).sum(0).tolist())
        if it["ql"] is not None:
            ql = it["ql"]
            row.update(qA_lowdin=float(ql[ia].sum()), coul_lowdin=coul(ql[ia], x[ia], ql[ib], x[ib]))
        rows.append(row)
        print(row["id"], round(row["E_int"], 1), row.get("E_int_DFT") and round(row["E_int_DFT"], 1),
              round(row["qA"], 3), round(row["qB"], 3), flush=True)
    json.dump({"model": model, "definition": __doc__, "observables": obs, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
