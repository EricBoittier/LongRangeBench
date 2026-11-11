"""Electrostatic properties (``--sets props``): dipoles, quadrupoles and polarisabilities of isolated molecules, and the
induced dipole of a neutral fragment as an ion approaches (Δμ(d) on dimer scans).

References (``props.jsonl``, built by ``tools/props_build.py``), all wB97M-V/def2-TZVPD:
  - ``role: mol``: ORCA 6.1.1 at the OMol25 settings + ``%elprop Dipole Quadrupole Polar 1`` (analytic CP-SCF static
    polarisability) on a stratified val_sel_clean subset (neutral / cation / anion / open shell, <= 40 atoms) and the
    24 dimer-benchmark monomers; refs ``dipole_au``, ``quadrupole_au`` (traceless Buckingham, 3/2 M2 - 1/2 tr M2),
    ``alpha_au`` (3x3), plus ``charges_lowdin``;
  - ``role: scan``: dimer-set records of ion+polar / ion+nonpolar scans (orientation 0) with their DFT Löwdin charges.
Conventions: atomic units (e bohr, e bohr^2, bohr^3); origin = centre of mass of the molecule (fragment for Δμ). For
charged species μ and Θ depend on the origin: everything here is about the COM, for model and reference alike.

Model side: μ = Σ q_i (r_i - COM), Θ = Σ q_i (3/2 r r - 1/2 r^2 I) from the per-atom charges the backend returns
(latent charges for LES-type models, which are not trained to be physical charges: fair to test, stated in the
output; no atomic dipoles, so a charge-only model misses intra-atomic polarisation). A backend returning a model dipole
is used for μ of neutral species only (origin-independent). α needs a field-response model: ``alpha`` in the
prediction (finite field) or N/A. Δμ(d): μ_B(d) - μ_B(d_far) of the neutral fragment B from per-atom charges, same
scan, model and Löwdin reference alike (Löwdin charges also carry no atomic dipoles: a charge-only reference).

Metrics: MAE / RMSE with 95 % percentile bootstrap over molecules, by charge state: |μ| error, vector error |Δμ|,
Θ component MAE and traceless-norm error, α isotropic and anisotropy errors, and the zero-intercept slope of model on
reference components (s = 1 exact; for latent charges s measures the charge scale)."""
from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

BOHR = 0.529177210903
NOTE_LATENT = ("latent charges (LES-type models) are not trained to be physical charges; scored as given, with the "
               "slope s reporting their scale")


def com(species, x):
    from ase.data import atomic_masses, atomic_numbers
    w = np.array([atomic_masses[atomic_numbers[s]] for s in species])
    return (w[:, None] * np.asarray(x, float)).sum(0) / w.sum()


def moments(species, x_ang, q, origin=None):
    """(μ, Θ) in atomic units about `origin` (default COM) from point charges q at x (Angstrom)."""
    x = (np.asarray(x_ang, float) - (com(species, x_ang) if origin is None else origin)) / BOHR
    q = np.asarray(q, float)
    mu = q @ x
    th = 1.5 * np.einsum("i,ia,ib->ab", q, x, x) - 0.5 * np.einsum("i,i->", q, (x ** 2).sum(1)) * np.eye(3)
    return mu, th


def _boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    i = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    m = v[i].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _stat(err):
    err = np.asarray(err, float)
    return {"n": int(len(err)), "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE_ci95": _boot(np.abs(err))} if len(err) else {"n": 0}


def _slope(m, r):
    m, r = np.ravel(m), np.ravel(r)
    return float(m @ r / (r @ r)) if r @ r > 0 else None


def charge_state(rec):
    q, s = int(rec["charge"]), int(rec.get("spin") or 1)
    return "open_shell" if s > 1 else ("neutral" if q == 0 else ("cation" if q > 0 else "anion"))


def score(ds, P) -> dict:
    """ds: the props DataSet; P: metrics.Preds over its record ids."""
    mols = [r for r in ds.records if r.get("role") == "mol" and r["refs"].get("dipole_au") is not None]
    rows, missing = [], 0
    for r in mols:
        q = P.charges(r["id"])
        ref = r["refs"]
        row = {"id": r["id"], "state": charge_state(r), "source": (r.get("extra") or {}).get("source")}
        mu_r = np.asarray(ref["dipole_au"], float)
        th_r = None if ref.get("quadrupole_au") is None else np.asarray(ref["quadrupole_au"], float)
        if q is not None:
            mu_m, th_m = moments(r["species"], r["positions"], q.mean(0))
            dips = [(m.get(r["id"]) or {}).get("dipole") for m in P.members]
            if row["state"] == "neutral" and all(d is not None for d in dips):  # model dipole (e A) if origin-free
                mu_m = np.mean([np.asarray(d, float) for d in dips], 0) / BOHR
            row.update(mu_ref=mu_r.tolist(), mu_model=mu_m.tolist())
            if th_r is not None:
                row.update(th_ref=th_r.tolist(), th_model=th_m.tolist())
        else:
            missing += 1
        al = [(m.get(r["id"]) or {}).get("alpha") for m in P.members]
        if ref.get("alpha_au") is not None and all(a is not None for a in al):
            row.update(alpha_ref=np.asarray(ref["alpha_au"]).tolist(), alpha_model=np.mean(al, 0).tolist())
            ad = [(m.get(r["id"]) or {}).get("alpha_from_dipole") for m in P.members]
            if all(x is not None for x in ad):
                row["alpha_from_dipole"] = np.mean(ad, 0).tolist()
        rows.append(row)
    out = {"note": __doc__.split("\n\n")[1], "latent_note": NOTE_LATENT, "n_mol": len(mols),
           "n_without_charges": missing}
    have = [x for x in rows if "mu_model" in x]
    if not have:
        out["status"] = "N/A: the model returns no per-atom charges or dipoles"
    groups = defaultdict(list)
    for x in have:
        groups["all"].append(x); groups[x["state"]].append(x)
    for g, xs in groups.items():
        mr, mm = np.array([x["mu_ref"] for x in xs]), np.array([x["mu_model"] for x in xs])
        xq = [x for x in xs if "th_ref" in x]
        out.setdefault("dipole", {})[g] = {
            "abs_mu_err": _stat(np.linalg.norm(mm, axis=1) - np.linalg.norm(mr, axis=1)),
            "vector_err": _stat(np.linalg.norm(mm - mr, axis=1)), "slope": _slope(mm, mr),
            "ref_abs_mu_mean": float(np.linalg.norm(mr, axis=1).mean())}
        if xq:
            tr, tm = np.array([x["th_ref"] for x in xq]), np.array([x["th_model"] for x in xq])
            out.setdefault("quadrupole", {})[g] = {
                "component_err": _stat(np.abs(tm - tr).reshape(len(xq), -1).mean(1)),
                "norm_err": _stat(np.linalg.norm(tm.reshape(len(xq), -1), axis=1) - np.linalg.norm(tr.reshape(len(xq), -1), axis=1)),
                "slope": _slope(tm, tr)}
    al = [x for x in rows if "alpha_model" in x]
    if al:
        ar, am = np.array([x["alpha_ref"] for x in al]), np.array([x["alpha_model"] for x in al])
        iso = lambda a: np.trace(a, axis1=1, axis2=2) / 3
        ani = lambda a: np.array([np.sqrt(max(0.0, 0.5 * ((e[0] - e[1]) ** 2 + (e[1] - e[2]) ** 2 + (e[2] - e[0]) ** 2)))
                                  for e in np.linalg.eigvalsh(a)])
        out["polarisability"] = {"iso_err": _stat(iso(am) - iso(ar)), "aniso_err": _stat(ani(am) - ani(ar)),
                                 "slope_iso": _slope(iso(am), iso(ar)), "ref_iso_mean": float(iso(ar).mean()),
                                 "by_state": {}}
        for g in sorted({x["state"] for x in al}):
            k = [j for j, x in enumerate(al) if x["state"] == g]
            out["polarisability"]["by_state"][g] = {"iso_err": _stat(iso(am[k]) - iso(ar[k])),
                                                    "slope_iso": _slope(iso(am[k]), iso(ar[k]))}
        dd = [x for x in al if "alpha_from_dipole" in x]
        if dd:
            ad = np.array([x["alpha_from_dipole"] for x in dd]); r2 = np.array([x["alpha_ref"] for x in dd])
            out["polarisability"]["diagnostic_from_model_dipole"] = {
                "iso_err": _stat(iso(ad) - iso(r2)), "slope_iso": _slope(iso(ad), iso(r2)),
                "note": "alpha from the derivative of the model's dipole output; differs from -d2E/dF2 if the dipole "
                        "output omits part of the field response"}
    else:
        out["polarisability"] = {"status": "N/A: no field response from this model"}
    out["induced_dipole"] = induced(ds, P)
    out["per_molecule"] = rows
    return out


def induced(ds, P) -> dict:
    """Δμ_B(d) = μ_B(d) - μ_B(d_far) of the neutral fragment (COM origin), model charges vs DFT Löwdin charges."""
    scans = defaultdict(list)
    for r in ds.records:
        if r.get("role") == "scan":
            scans[r.get("group")].append(r)
    curves, err = [], defaultdict(list)
    for g, rs in sorted(scans.items()):
        rs = sorted(rs, key=lambda r: float(r["distance"]))
        fid = np.asarray(rs[0]["fragment_id"])
        nb = int(np.argmin([abs(c) for c in rs[0]["fragment_charges"]]))  # the neutral fragment
        m = fid == nb

        def mu(r, q):
            x = np.asarray(r["positions"], float)[m]
            return moments([s for s, k in zip(r["species"], m) if k], x, np.asarray(q, float)[m])[0]
        far = rs[-1]
        ref = [mu(r, r["refs"]["charges_lowdin"]) - mu(far, far["refs"]["charges_lowdin"]) for r in rs]
        qs = [P.charges(r["id"]) for r in rs]
        mod = None if any(q is None for q in qs) else [mu(r, q.mean(0)) - mu(far, qs[-1].mean(0)) for r, q in zip(rs, qs)]
        c = {"scan": g, "cls": rs[0].get("cls"), "d": [float(r["distance"]) for r in rs],
             "dmu_ref_au": [float(np.linalg.norm(v)) for v in ref],
             "dmu_model_au": None if mod is None else [float(np.linalg.norm(v)) for v in mod]}
        curves.append(c)
        if mod is not None:
            for d, a, b in zip(c["d"][:-1], ref[:-1], mod[:-1]):
                err["<8" if d < 8 else ">=8"].append(float(np.linalg.norm(b - a)))
    return {"definition": "Δμ_B(d) = μ_B(d) - μ_B(d_far), neutral fragment, COM origin, charges only (no atomic "
                          "dipoles) for both the Löwdin reference and the model; au",
            "vector_err_by_region": {k: _stat(v) for k, v in err.items()}, "curves": curves,
            "status": None if err else "N/A: no per-atom charges from this model"}


def headline_rows(block: dict, sn: str = "props") -> List[tuple]:
    """Rows for report.headline_rows (metric, value, n, note)."""
    rows = []
    f = lambda x: "–" if x is None else f"{x:.3f}"
    for g in ("all", "neutral", "cation", "anion", "open_shell"):
        d = (block.get("dipole") or {}).get(g)
        if d and d["vector_err"].get("n"):
            ci = d["vector_err"]["MAE_ci95"]
            rows.append((f"{sn}: dipole vector error, {g} (e bohr)", f"{f(d['vector_err']['MAE'])} [{f(ci[0])}, {f(ci[1])}]",
                         str(d["vector_err"]["n"]), f"slope {f(d['slope'])}; COM origin"))
    q = (block.get("quadrupole") or {}).get("all")
    if q and q["norm_err"].get("n"):
        rows.append((f"{sn}: quadrupole traceless-norm error (e bohr^2)", f(q["norm_err"]["MAE"]), str(q["norm_err"]["n"]),
                     f"slope {f(q['slope'])}"))
    a = block.get("polarisability") or {}
    rows.append((f"{sn}: isotropic polarisability error (bohr^3)", f(a["iso_err"]["MAE"]) if "iso_err" in a else "N/A",
                 str(a["iso_err"]["n"]) if "iso_err" in a else "", "" if "iso_err" in a else a.get("status", "")))
    ind = (block.get("induced_dipole") or {}).get("vector_err_by_region") or {}
    for k, v in ind.items():
        if v.get("n"):
            rows.append((f"{sn}: induced dipole Δμ(d) error, d {k} A (e bohr)", f(v["MAE"]), str(v["n"]), "vs Löwdin charges"))
    return rows
