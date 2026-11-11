"""Metrics on one data set, from per-term predictions of one model or an ensemble.

All energies in the output are meV (forces meV/A, per-atom energies meV/atom) unless a key says otherwise.
Definitions follow the program scripts (paths relative to ~/data):
  - E_int MAE by class x distance, offset-free error, tail fraction: petlr-lrd/lrd_an.py, oracle/oracle.py,
    figures/scripts/prep_fig.py. Offset-free dE(d) = E_int(d) - E_int(scan's longest point). Tail fraction = least-
    squares slope through 0 of the model dE on the DFT dE over all points of a class with d >= 8 A (scans with >= 2
    such points), 95 % hierarchical bootstrap over seeds x scans.
  - |E_int| error at >= 12 A: absolute MAE pooled over d >= 12 A (scorecard footnote f).
  - fixed-Q double differences: figures/scripts/analyse_fixedq.py (slope through 0 on DFT, MAE, bootstrap over
    seeds x pairs).
  - zwitterion series: analyse_fixedq.py (dE_z(m) and the local-zero score dE_z(m) - dE_z(m_max)).
  - trimer E3, geometry-relative dE3_vs_far: lr-suite-v2/scripts/analyse.py.
  - latent-charge parity and dipoles: analyse.py (point-charge dipole about the centre of mass, atomic units).
Records flagged "dft_pathology_do_not_score" (and every group touching one) are never scored; they are listed under
"pathology" with the model values.
"""
from __future__ import annotations

import collections
from typing import Dict, List, Optional

import numpy as np

from .data import DataSet, base_id
from .stats import conformal, cqr, hier_bootstrap, slope0

DBINS = [(0, 4, "0-4"), (4, 8, "4-8"), (8, 12, "8-12"), (12, np.inf, ">=12")]
R_CUT = 8.0
BOHR = 0.529177210903
MASS = {"H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
        "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305, "Al": 26.98, "Si": 28.085, "P": 30.974, "S": 32.06,
        "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078, "Zn": 65.38, "Br": 79.904, "I": 126.9}
CLASS_GROUP = {"ion_pair": "ion pair", "ion_pair_ct_ambiguous": "ion pair", "same_sign": "same sign",
               "like_charge": "same sign", "ion_polar": "ion-neutral", "ion_nonpolar": "ion-neutral",
               "neutral_polar_control": "neutral", "neutral_nonpolar_control": "neutral"}


def dbin(d) -> Optional[str]:
    if d is None or not np.isfinite(d):
        return None
    return next(lab for lo, hi, lab in DBINS if lo - 1e-6 <= d < hi - 1e-6)


def _r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def _ci(t, k=3):
    return [_r(t[0], k), _r(t[1], k), _r(t[2], k)]


class Preds:
    """Per-term member predictions: E[term] -> (M,) energies (nan where a member failed)."""

    def __init__(self, members: List[Dict[str, dict]]):
        self.members = members
        self.M = len(members)
        terms = set().union(*[m.keys() for m in members]) if members else set()
        self.E = {t: np.array([np.nan if (m.get(t) or {}).get("energy") is None else m[t]["energy"] for m in members])
                  for t in terms}
        self.errors = {t: (m.get(t) or {}).get("error") for m in members for t in terms if (m.get(t) or {}).get("error")}

    def energy(self, t):
        return self.E.get(t, np.full(self.M, np.nan))

    def forces(self, t):
        fs = [(m.get(t) or {}).get("forces") for m in self.members]
        return None if any(f is None for f in fs) else np.array([np.asarray(f, float) for f in fs])

    def charges(self, t):
        qs = [(m.get(t) or {}).get("charges") for m in self.members]
        return None if any(q is None for q in qs) else np.array([np.asarray(q, float) for q in qs])


def group_value(g, P: Preds):
    v = sum(c * P.energy(t) for t, c in g["terms"])
    return np.asarray(v, float)


def _main_record(ds: DataSet, g):
    for t, c in g["terms"]:
        if "#" not in t and c > 0 and t in ds.by_id:
            return ds.by_id[t]
    return ds.by_id.get(base_id(g["terms"][0][0]))


def _gattr(ds, g):
    rec = _main_record(ds, g) or {}
    m = g.get("meta") or {}
    first = lambda *ks: next((m[k] for k in ks if m.get(k) is not None), None)
    return {"cls": first("cls", "class") or rec.get("cls") or "all",
            "distance": first("distance", "dmin", "d") if first("distance", "dmin", "d") is not None else rec.get("distance"),
            "scan": first("scan") or rec.get("group") or g["name"],
            "family": first("series", "family", "fam", "type", "probe") or rec.get("cls"),
            "pair": first("pair", "bootstrap_group") or rec.get("group") or g["name"],
            "size": first("m", "n", "length", "size"),
            "rec": rec}


def _composition_changing(ds: DataSet, g) -> bool:
    """True if sum_terms coeff * (element counts) != 0: then the difference depends on absolute per-element energies
    (and on the element offsets lrbench fits for reference-subtracted models)."""
    from .data import term_structure
    c = collections.Counter()
    for t, k in g["terms"]:
        for z in term_structure(ds, t).species:
            c[z] += k
    return any(abs(v) > 1e-9 for v in c.values())


class SetScorer:
    def __init__(self, ds: DataSet, P: Preds, n_boot: int = 2000, conformal_on: bool = False,
                 normalise: Optional[dict] = None):
        self.ds, self.P, self.nb, self.conf, self.norm = ds, P, n_boot, conformal_on, normalise
        self.M = P.M
        self.scored_groups = [g for g in ds.groups if not ds.group_flagged(g) and ds.group_level(g) is not None]
        self.level_excluded = [g["name"] for g in ds.groups if not ds.group_flagged(g) and ds.group_level(g) is None]
        self.flag_groups = [g for g in ds.groups if ds.group_flagged(g)]

    # ------------------------------------------------------------------ absolute energies and forces
    def energy_force(self):
        rows = collections.defaultdict(lambda: {"dE": [], "dF": [], "nF": 0, "sF": 0.0, "sF2": 0.0, "w": []})
        strata = collections.defaultdict(list)
        n_e = n_f = 0
        for r in self.ds.records:
            if self.ds.flagged(r["id"]) or self.ds.partner_copy(r["id"]):
                continue
            key = (r.get("cls") or "all", dbin(r.get("distance")) or "all")
            n = len(r["species"])
            if r.get("energy") is not None and self.ds.energy_ok(r["id"]):
                e = np.nanmean(self.P.energy(r["id"]))
                if np.isfinite(e):
                    de = 1e3 * abs(e - r["energy"]) / n
                    rows[key]["dE"].append(de)
                    n_e += 1
                    for s in (r.get("extra") or {}).get("strata") or (r.get("refs") or {}).get("strata") or []:
                        strata[s].append(de)
            if r.get("forces") is not None:
                F = self.P.forces(r["id"])
                if F is not None:
                    d = np.abs(F.mean(0) - np.asarray(r["forces"], float)) * 1e3
                    rows[key]["nF"] += d.size
                    rows[key]["sF"] += d.sum()
                    rows[key]["sF2"] += (d ** 2).sum()
                    n_f += 1
        out = {"n_energy": n_e, "n_forces": n_f, "by_class_distance": []}
        for (c, b), v in sorted(rows.items()):
            out["by_class_distance"].append({
                "cls": c, "dbin": b, "n_energy": len(v["dE"]),
                "E_mae_meV_atom": _r(np.mean(v["dE"])) if v["dE"] else None,
                "F_mae_meV_A": _r(v["sF"] / v["nF"]) if v["nF"] else None,
                "F_rmse_meV_A": _r(np.sqrt(v["sF2"] / v["nF"])) if v["nF"] else None})
        allE = [x for v in rows.values() for x in v["dE"]]
        nF, sF = sum(v["nF"] for v in rows.values()), sum(v["sF"] for v in rows.values())
        out["all"] = {"E_mae_meV_atom": _r(np.mean(allE)) if allE else None, "F_mae_meV_A": _r(sF / nF) if nF else None}
        if strata:
            out["strata"] = {s: {"n": len(v), "E_mae_meV_atom": _r(np.mean(v))} for s, v in sorted(strata.items())}
        out["note"] = ("Absolute energies compare the model's total energy with the DFT total; meaningful only for "
                       "models trained on the same level (OMol25 wB97M-V/def2-TZVPD).")
        return out

    # ------------------------------------------------------------------ interaction energies
    def e_int(self):
        G = [g for g in self.scored_groups if g["kind"] == "E_int"]
        if not G:
            return None
        A = [_gattr(self.ds, g) for g in G]
        V = np.array([group_value(g, self.P) for g in G]).T  # (M, n)
        ref = np.array([g["ref"] for g in G])
        cls = np.array([a["cls"] for a in A])
        d = np.array([np.nan if a["distance"] is None else float(a["distance"]) for a in A])
        scan = np.array([str(a["scan"]) for a in A])
        ok = np.all(np.isfinite(V), axis=0)
        out = {"n": int(len(G)), "n_failed": int((~ok).sum()), "bins": [], "by_distance": [], "tail": {},
               "offset_free_8A": {}, "abs_ge12": {}, "inside_median": {}}
        V, ref, cls, d, scan = V[:, ok], ref[ok], cls[ok], d[ok], scan[ok]
        if not ok.any():  # e.g. a model whose species map lacks every structure's elements
            out["note"] = "every group failed (model returned no energy); nothing scored"
            return out
        pv = V.mean(0)
        M = V.shape[0]
        mae = lambda idx: (lambda sd, pts: 1e3 * np.mean(np.abs(V[sd][:, idx[pts]].mean(0) - ref[idx[pts]])))
        CL = sorted(set(cls.tolist()))
        for c in CL:
            for lo, hi, lab in DBINS:
                i = np.where((cls == c) & (d >= lo - 1e-6) & (d < hi - 1e-6))[0]
                if not len(i):
                    continue
                est = hier_bootstrap(mae(i), M, scan[i], self.nb, points=True)
                out["bins"].append({"cls": c, "dbin": lab, "n": int(len(i)), "mae": _ci(est, 2),
                                    "mean_signed": _r(1e3 * np.mean(pv[i] - ref[i]), 2),
                                    "dft_mean_abs": _r(1e3 * np.mean(np.abs(ref[i])), 2)})
            for dd in sorted(set(np.round(d[cls == c], 2).tolist())):
                i = (cls == c) & (np.abs(d - dd) < 5e-3)
                out["by_distance"].append({"cls": c, "d": dd, "n": int(i.sum()),
                                           "mae": _r(1e3 * np.mean(np.abs(pv[i] - ref[i])), 2),
                                           "dft": _r(1e3 * np.mean(ref[i]), 2), "model": _r(1e3 * np.mean(pv[i]), 2)})
        # offset-free dE relative to each scan's longest point
        last = {s: np.where(scan == s)[0][np.argmax(d[scan == s])] for s in set(scan.tolist())}
        il = np.array([last[s] for s in scan], int)
        dref, dV = ref - ref[il], V - V[:, il]
        for c in CL:
            # tail fraction over d >= 8 A, scans with >= 2 such points (prep_fig.py)
            j = np.where((cls == c) & (d >= R_CUT - 1e-6))[0]
            keep = [k for k in j if ((scan[j] == scan[k]).sum() >= 2)]
            if keep:
                k = np.array(keep)
                l8 = {s: k[scan[k] == s][np.argmax(d[k][scan[k] == s])] for s in set(scan[k].tolist())}
                lk = np.array([l8[s] for s in scan[k]])
                x, Y = ref[k] - ref[lk], V[:, k] - V[:, lk]
                if x @ x > 0:
                    est = hier_bootstrap(lambda sd, pts: slope0(x[pts], Y[sd][:, pts].mean(0)), M, scan[k], self.nb)
                    out["tail"][c] = {"slope": _ci(est, 4), "n_scans": int(len(set(scan[k].tolist()))),
                                      "n_points": int(len(k)), "dft_span_meV": _r(1e3 * np.abs(x).max(), 1),
                                      "per_member": [_r(slope0(x, Y[m]), 4) for m in range(M)]}
            i8 = np.where((cls == c) & (np.abs(d - R_CUT) < 1e-2))[0]
            i8 = np.array([k for k in i8 if il[k] != k], int)
            if len(i8):
                f = lambda sd, pts: 1e3 * np.mean(np.abs(dV[sd][:, i8[pts]].mean(0) - dref[i8[pts]]))
                out["offset_free_8A"][c] = {"n": int(len(i8)), "mae": _ci(hier_bootstrap(f, M, scan[i8], self.nb,
                                                                                        points=True), 2),
                                            "flat_model_mae": _r(1e3 * np.mean(np.abs(dref[i8])), 2)}
            i12 = np.where((cls == c) & (d >= 12 - 1e-6))[0]
            if len(i12):
                out["abs_ge12"][c] = {"n": int(len(i12)), "mae": _ci(hier_bootstrap(mae(i12), M, scan[i12], self.nb,
                                                                                    points=True), 2),
                                      "dft_mean_abs": _r(1e3 * np.mean(np.abs(ref[i12])), 2)}
            iin = np.where((cls == c) & (d < R_CUT - 1e-6))[0]
            if len(iin):
                med = lambda sd, pts: 1e3 * np.median(np.abs(V[sd][:, iin[pts]].mean(0) - ref[iin[pts]]))
                out["inside_median"][c] = {"n": int(len(iin)), "median_abs_err": _ci(
                    hier_bootstrap(med, M, np.arange(len(iin)), self.nb), 2)}
        self._eint = dict(V=V, ref=ref, cls=cls, d=d, scan=scan, recs=[a["rec"] for a, o in zip(A, ok) if o])
        return out

    # ------------------------------------------------------------------ double / single differences
    def differences(self, kind):
        G = [g for g in self.scored_groups if g["kind"] == kind]
        if not G:
            return None
        A = [_gattr(self.ds, g) for g in G]
        V = np.array([group_value(g, self.P) for g in G]).T
        ok = np.all(np.isfinite(V), axis=0)
        if not ok.any():
            return {"n": int(len(G)), "n_failed": int(len(G)), "note": "every group failed (model returned no energy); nothing scored"}
        y = np.array([g["ref"] for g in G])[ok] * 1e3
        V = V[:, ok] * 1e3
        pair = np.array([str(a["pair"]) for a, o in zip(A, ok) if o])
        fam = np.array([str(a["family"]) for a, o in zip(A, ok) if o])
        pv = V.mean(0)
        M = V.shape[0]
        sl = hier_bootstrap(lambda sd, pts: slope0(y[pts], V[sd][:, pts].mean(0)), M, pair, self.nb, seed=11)
        ma = hier_bootstrap(lambda sd, pts: np.mean(np.abs(V[sd][:, pts].mean(0) - y[pts])), M, pair, self.nb, seed=11)
        comp = np.array([_composition_changing(self.ds, g) for g, o in zip(G, ok) if o], bool)
        out = {"n": int(len(y)), "n_failed": int((~ok).sum()), "n_pairs": int(len(set(pair.tolist()))),
               "n_composition_changing": int(comp.sum()),
               "dft_median_abs": _r(np.median(np.abs(y)), 2), "median_abs_pred": _r(np.median(np.abs(pv)), 3),
               "slope": _ci(sl, 4), "mae": _ci(ma, 2), "zero_model_mae": _r(np.mean(np.abs(y)), 2), "by_family": {},
               "rows": [{"name": g["name"], "family": a["family"], "dft": _r(yy, 3), "model": _r(p, 3)}
                        for g, a, yy, p in zip([g for g, o in zip(G, ok) if o], [a for a, o in zip(A, ok) if o], y, pv)]}
        for f in sorted(set(fam.tolist())):
            i = np.where(fam == f)[0]
            yi, Vi, pi = y[i], V[:, i], pair[i]
            fs = hier_bootstrap(lambda sd, pts: slope0(yi[pts], Vi[sd][:, pts].mean(0)), M, pi, self.nb, seed=11)
            fm = hier_bootstrap(lambda sd, pts: np.mean(np.abs(Vi[sd][:, pts].mean(0) - yi[pts])), M, pi, self.nb,
                                seed=11)
            out["by_family"][f] = {"n": int(len(i)), "n_pairs": int(len(set(pi.tolist()))), "slope": _ci(fs, 4),
                                   "mae": _ci(fm, 2), "zero_model_mae": _r(np.mean(np.abs(yi)), 2),
                                   "dft_median_abs": _r(np.median(np.abs(yi)), 2),
                                   "median_abs_pred": _r(np.median(np.abs(pv[i])), 3),
                                   "per_member_slope": [_r(slope0(yi, Vi[m]), 4) for m in range(M)]}
        setattr(self, "_" + kind, dict(V=V, y=y, pair=pair, fam=fam, groups=[g for g, o in zip(G, ok) if o]))
        return out

    def zwitterion(self):
        G = [g for g in self.scored_groups if g["kind"] == "zwitterion_dE"]
        if not G:
            return None
        series = collections.defaultdict(list)
        for g in G:
            a = _gattr(self.ds, g)
            v = group_value(g, self.P)
            size = a["size"]
            if size is None:
                digits = "".join(ch for ch in g["name"] if ch.isdigit())
                size = int(digits) if digits else None
            series[str(a["family"])].append((size, g["ref"] * 1e3, v.mean() * 1e3, g["name"]))
        out = {}
        for f, rows in sorted(series.items()):
            rows.sort(key=lambda r: (r[0] is None, r[0]))
            ref = rows[-1]
            lz = [(r[1] - ref[1], r[2] - ref[2]) for r in rows[:-1] if r[0] is not None and r[0] >= 36]
            out[f] = {"rows": [{"size": r[0], "dft": _r(r[1], 2), "model": _r(r[2], 3), "name": r[3]} for r in rows],
                      "reference_size": ref[0],
                      "local_zero": [{"size": r[0], "dft_rel": _r(r[1] - ref[1], 2), "model_rel": _r(r[2] - ref[2], 3)}
                                     for r in rows[:-1] if r[0] is not None and r[0] >= 36],
                      "local_zero_mae": _r(np.mean([abs(m - t) for t, m in lz]), 2) if lz else None}
        return out

    def e3(self):
        G = [g for g in self.scored_groups if g["kind"] == "E3"]
        if not G:
            return None
        rows = []
        for g in G:
            a = _gattr(self.ds, g)
            v = group_value(g, self.P)
            if not np.all(np.isfinite(v)):
                continue
            rows.append({"name": g["name"], "family": str(a["family"]), "distance": a["distance"],
                         "far": bool((g.get("meta") or {}).get("far", False)), "dft": g["ref"] * 1e3,
                         "members": v * 1e3})
        if not rows:
            return {"n": 0}
        y = np.array([r["dft"] for r in rows])
        V = np.array([r["members"] for r in rows]).T
        fam = np.array([r["family"] for r in rows])
        M = V.shape[0]
        ma = hier_bootstrap(lambda sd, pts: np.mean(np.abs(V[sd][:, pts].mean(0) - y[pts])), M, fam, self.nb, points=True)
        out = {"n": len(rows), "abs_mae": _ci(ma, 2), "dft_mean_abs": _r(np.mean(np.abs(y)), 2),
               "zero_model_mae": _r(np.mean(np.abs(y)), 2), "rows": [], "geometry_relative": None}
        rel = []
        for f in sorted(set(fam.tolist())):
            ii = [k for k in range(len(rows)) if fam[k] == f]
            far = [k for k in ii if rows[k]["far"]] or [max(ii, key=lambda k: rows[k]["distance"] or -1)]
            kf = far[0]
            for k in ii:
                if k != kf:
                    rel.append((f, rows[k]["name"], y[k] - y[kf], V[:, k] - V[:, kf]))
        for k, r in enumerate(rows):
            out["rows"].append({"name": r["name"], "family": r["family"], "distance": r["distance"],
                                "dft": _r(r["dft"], 3), "model": _r(V[:, k].mean(), 3)})
        self._E3 = dict(y=y, V=V, fam=fam, rel=rel)
        if rel:
            yr = np.array([x[2] for x in rel])
            Vr = np.array([x[3] for x in rel]).T
            fr = np.array([x[0] for x in rel])
            mr = hier_bootstrap(lambda sd, pts: np.mean(np.abs(Vr[sd][:, pts].mean(0) - yr[pts])), M, fr, self.nb,
                                points=True)
            out["geometry_relative"] = {
                "n": len(rel), "mae": _ci(mr, 2), "zero_model_mae": _r(np.mean(np.abs(yr)), 2),
                "slope": _r(slope0(yr, Vr.mean(0)), 4),
                "rows": [{"family": f, "name": n, "dft": _r(a, 3), "model": _r(b.mean(), 3)} for f, n, a, b in rel],
                "note": "dE3 = E3(geometry) - E3(the family's far geometry: meta.far, else the largest distance); the "
                        "geometry-independent charge-state offset cancels"}
        return out

    # ------------------------------------------------------------------ charges and moments
    def charges(self):
        recs = [r for r in self.ds.records if not self.ds.flagged(r["id"]) and not self.ds.partner_copy(r["id"])]
        have = [r for r in recs if self.P.charges(r["id"]) is not None]
        if not have:
            return {"skipped": "model outputs no per-atom charges"}
        ratio = collections.defaultdict(list)
        scheme = collections.defaultdict(list)
        dip, diprel = [], []
        for r in have:
            q = self.P.charges(r["id"]).mean(0)
            fid, qf = r.get("fragment_id"), r.get("fragment_charges")
            if fid is not None and qf is not None:
                fid = np.asarray(fid)
                for k, Q in enumerate(qf):
                    if Q:
                        ratio[(r.get("cls") or "all", dbin(r.get("distance")) or "all")].append(q[fid == k].sum() / Q)
            refs = r.get("refs") or {}
            for k, v in refs.items():
                if k.startswith("charges_") and v is not None and len(v) == len(q):
                    scheme[k[8:]].append(np.mean(np.abs(q - np.asarray(v, float))))
            if refs.get("dipole_au") is not None:
                R = np.asarray(r["positions"], float)
                m = np.array([MASS.get(s, 12.0) for s in r["species"]])
                o = np.asarray(refs.get("dipole_origin_A") or (m[:, None] * R).sum(0) / m.sum())
                mu = q @ ((R - o) / BOHR)
                dref = np.asarray(refs["dipole_au"], float)
                dip.append(float(np.linalg.norm(mu - dref)))
                if np.linalg.norm(dref) > 1e-3:  # Born prior: dipole error relative to |mu_DFT| (neutral-ish only)
                    diprel.append(dip[-1] / float(np.linalg.norm(dref)))
        return {"n": len(have),
                "latent_over_formal": [{"cls": c, "dbin": b, "n": len(v), "mean": _r(np.mean(v), 3),
                                        "min": _r(np.min(v), 3), "max": _r(np.max(v), 3)}
                                       for (c, b), v in sorted(ratio.items())],
                "per_atom_mae_vs_scheme_e": {k: _r(np.mean(v), 4) for k, v in scheme.items()},
                "dipole_err_rel": {"n": len(diprel), "median": _r(np.median(diprel), 3) if diprel else None,
                                   "max": _r(np.max(diprel), 3) if diprel else None,
                                   "note": "|mu_model - mu_DFT| / |mu_DFT| (|mu_DFT| > 1e-3 au)"},
                "dipole_err_au": {"n": len(dip), "median": _r(np.median(dip), 3) if dip else None,
                                  "max": _r(np.max(dip), 3) if dip else None},
                "note": "fragment sums of the model's per-atom charges over the formal fragment charge (1 = formal); "
                        "dipole of the model charges about the centre of mass vs refs.dipole_au"}

    def pathology(self):
        rows = []
        for r in self.ds.records:
            if self.ds.flagged(r["id"]):
                e = self.P.energy(r["id"])
                rows.append({"id": r["id"], "kind": "record", "dft_energy": r.get("energy"),
                             "model_energy": _r(np.nanmean(e), 6) if np.isfinite(e).any() else None})
        for g in self.flag_groups:
            v = group_value(g, self.P)
            rows.append({"name": g["name"], "kind": g["kind"], "dft_meV": _r(1e3 * g["ref"], 3),
                         "model_meV": _r(1e3 * np.nanmean(v), 3) if np.isfinite(v).any() else None})
        return rows

    # ------------------------------------------------------------------ calibration
    def calibration(self):
        out = {}
        has_sig = self.M > 1
        if hasattr(self, "_eint"):
            e = self._eint
            pv, ref = e["V"].mean(0), e["ref"]
            sig = e["V"].std(0, ddof=1) if has_sig else None
            grp = np.array([CLASS_GROUP.get(c, c) for c in e["cls"]])
            reg = np.where(e["d"] >= R_CUT - 1e-6, "beyond 8 A", "inside 8 A")
            mb = np.array([f"{g} | {r}" for g, r in zip(grp, reg)])
            out["E_int"] = {"split_conformal": conformal(ref, pv, None, e["scan"], np.full(len(ref), "all")),
                            "mondrian": conformal(ref, pv, sig, e["scan"], mb), "unit": "eV",
                            "bins": "class group x inside/beyond 8 A", "groups": "scan"}
            if len(set(e["scan"].tolist())) >= 10:
                qq = np.array([np.prod(r.get("fragment_charges") or [0, 0]) for r in e["recs"]], float)
                X = np.column_stack([e["d"], (e["d"] >= R_CUT).astype(float), qq] +
                                    [(e["cls"] == c).astype(float) for c in sorted(set(e["cls"].tolist()))] +
                                    ([np.log(sig + 1e-6)] if has_sig else []))
                out["E_int"]["cqr"] = cqr(ref, pv, X, e["scan"], mb)
        for kind in ("ddE", "dE"):
            if hasattr(self, "_" + kind):
                e = getattr(self, "_" + kind)
                pv = e["V"].mean(0) / 1e3
                sig = e["V"].std(0, ddof=1) / 1e3 if has_sig else None
                out[kind] = {"mondrian": conformal(e["y"] / 1e3, pv, sig, e["pair"], e["fam"]), "unit": "eV",
                             "bins": "family", "groups": "pair"}
        out["note"] = ("Split-conformal by group (20 random 50/50 calibration/test splits); score |e|/sigma_ensemble "
                       "when an ensemble is given (normalised), else |e|. Coverage = median [2.5, 97.5 %] over splits; "
                       "width = median full interval width (eV). Exchangeability holds only within a bin: nothing is "
                       "guaranteed for new classes, distances or models.")
        return out

    def run(self):
        res = {"n_records": len(self.ds.records), "n_groups": len(self.ds.groups),
               "n_pathology_records": sum(self.ds.flagged(r["id"]) for r in self.ds.records),
               "n_pathology_groups": len(self.flag_groups), "levels": self.ds.levels(),
               "n_groups_excluded_level": len(self.level_excluded),
               "groups_excluded_level": self.level_excluded[:50], "group_kinds": dict(collections.Counter(
                   g["kind"] for g in self.ds.groups)), "source": self.ds.source}
        res["energy_force"] = self.energy_force()
        for k, f in (("E_int", self.e_int), ("ddE", lambda: self.differences("ddE")),
                     ("dE", lambda: self.differences("dE")), ("zwitterion", self.zwitterion), ("E3", self.e3),
                     ("charges", self.charges)):
            v = f()
            if v is not None:
                res[k] = v
        if self.norm is not None:
            from .normalise import normalised
            res["normalised"] = normalised(self, **self.norm)
        res["pathology"] = self.pathology()
        if self.conf or self.M > 1:
            res["calibration"] = self.calibration()
        if self.P.errors:
            res["prediction_errors"] = dict(list(self.P.errors.items())[:50])
            res["n_prediction_errors"] = len(self.P.errors)
        return res
