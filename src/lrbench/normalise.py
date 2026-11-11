"""Physics-normalised errors (error relative to a physical prior; see ``docs/metrics.md``, "Normalised errors").

- Dimer tail: eps(r) = (dE_model - dE_DFT) / dE_phys, dE(r) = E_int(r) - E_int(d_ref) (offset-free, d >= 8 A,
  d_ref = the scan's longest point), dE_phys the same difference of refs.E_es_exact + refs.E_mp_ind when both points
  have E_es_exact, else of refs.E_mp (the fragments' own DFT multipoles + induction); r = the canonical distance
  (d_min), with the mean centre-of-mass distance per row as r_com_mean. The scale is used *signed* (dE_phys carries the sign of
  the tail), so eps = 0 is exact and eps = -1 is a missing tail, as in the normalised-error table of docs/metrics.md; dividing by
  |dE_phys| would flip the sign of eps for attractive tails. Points with |dE_phys| below a floor are not scored.
- Errors in k_BT (default 300 K): offset-free tail error and E_int error per class x distance.
- Fixed-Q ddE: (ddE_model - ddE_DFT) / |ddE_DFT| per series (and per wire length/site where meta has them).
- Trimers: (E3_model - E3_DFT) / |E3_DFT| and the geometry-relative dE3 the same way.
- Dipoles: |mu_model - mu_DFT| / |mu_DFT| (in the charges block, when the model returns charges).
- Coulomb prior: a LES model with gate g and latent fragment charges lambda * Q_f has a far tail g lambda_A lambda_B
  q_A q_B / r, so eps_pred = g <lambda_A lambda_B> - 1 (flat in r); compared with the measured ion-pair eps.
  Fragment-charge GGUFs (pet.plato_lr.frag_charges) add the exact formal-charge Coulomb beyond the cutoff, so
  eps_pred = g <lambda_A lambda_B> + 1 - 1; with g = 0 the latent charges do not enter and lambda is not reported.
Units: --distance-unit A|bohr (r/a0), --energy-unit meV|eV|Eh|kBT for absolute errors in this block.
"""
from __future__ import annotations

import collections

import numpy as np

from .stats import hier_bootstrap

KB_EV = 8.617333262e-5
HARTREE_EV = 27.211386245988
BOHR = 0.529177210903
R_CUT, R_CONNECTED = 8.0, 24.0


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def _ci(t, k=4):
    return [_r(t[0], k), _r(t[1], k), _r(t[2], k)]


def energy_factor(unit: str, T: float) -> float:
    """meV -> unit."""
    return {"meV": 1.0, "eV": 1e-3, "Eh": 1e-3 / HARTREE_EV, "kBT": 1.0 / (1e3 * KB_EV * T)}[unit]


def dist(d, unit):
    return d / BOHR if unit == "bohr" else d


def _summary(eps, M, groups, nb, k=4):
    """median and mean of eps (member-mean values per point) with hierarchical bootstrap over members x groups."""
    if not eps.shape[-1]:
        return None
    med = hier_bootstrap(lambda sd, pts: np.median(eps[sd][:, pts].mean(0)), M, groups, nb, points=True)
    mean = hier_bootstrap(lambda sd, pts: np.mean(eps[sd][:, pts].mean(0)), M, groups, nb, points=True)
    return {"n": int(eps.shape[-1]), "median": _ci(med, k), "mean": _ci(mean, k)}


def dimer_tail(sc, floor_meV: float, T: float, dunit: str, eunit: str):
    e = getattr(sc, "_eint", None)
    if e is None:
        return None
    V, ref, cls, d, scan, recs = (e[k] for k in ("V", "ref", "cls", "d", "scan", "recs"))
    M = V.shape[0]
    # exact electrostatics + multipole induction where the audit has it (as figures/scripts/normalised.py)
    es = np.array([_f((r.get("refs") or {}).get("E_es_exact")) + np.nan_to_num(_f((r.get("refs") or {}).get("E_mp_ind")))
                   for r in recs])
    rcom = np.array([_f((r.get("extra") or {}).get("r_com")) for r in recs])
    mp = np.array([_f((r.get("refs") or {}).get("E_mp")) for r in recs])
    far = np.where(d >= R_CUT - 1e-6)[0]
    last = {}
    for k in far:
        if scan[k] not in last or d[k] > d[last[scan[k]]]:
            last[scan[k]] = k
    pts, phys, src = [], [], []
    for k in far:
        l = last[scan[k]]
        if l == k:
            continue
        if np.isfinite(es[k]) and np.isfinite(es[l]):
            p, s_ = es[k] - es[l], "E_es_exact+E_mp_ind"
        elif np.isfinite(mp[k]) and np.isfinite(mp[l]):
            p, s_ = mp[k] - mp[l], "E_mp"
        else:
            continue
        pts.append((k, l))
        phys.append(p)
        src.append(s_)
    if not pts:
        return {"skipped": "no dimer points with refs.E_es_exact / refs.E_mp beyond 8 A"}
    k_, l_ = np.array(pts).T
    phys = 1e3 * np.array(phys)
    dft = 1e3 * (ref[k_] - ref[l_])
    mod = 1e3 * (V[:, k_] - V[:, l_])  # (M, n)
    keep = np.abs(phys) >= floor_meV
    kt = 1e3 * KB_EV * T  # meV
    fac = energy_factor(eunit, T)
    eps = (mod - dft) / np.where(keep, phys, np.nan)
    out = {"definition": "eps = (dE_model - dE_DFT) / dE_phys, signed scale; dE relative to the scan's longest point",
           "phys_source": dict(collections.Counter(src)), "floor_meV": floor_meV, "n_points": int(len(k_)),
           "n_below_floor": int((~keep).sum()), "temperature_K": T, "kBT_meV": _r(kt, 4),
           "distance_unit": dunit, "energy_unit": eunit, "by_class": {}, "by_class_distance": []}
    C, D, S = cls[k_], d[k_], scan[k_]
    for c in sorted(set(C.tolist())):
        i = np.where((C == c) & keep)[0]
        if not len(i):
            continue
        row = _summary(eps[:, i], M, S[i], sc.nb)
        row["prior_check_median_dft_over_phys"] = _r(np.median(dft[i] / phys[i]), 3)
        err = np.abs(mod[:, i].mean(0) - dft[i])
        row["tail_err_kBT_median"] = _r(np.median(err) / kt, 4)
        row["frac_tail_err_gt_1kBT"] = _r(np.mean(err > kt), 3)
        out["by_class"][c] = row
        for dd in sorted(set(np.round(D[i], 2).tolist())):
            j = i[np.abs(D[i] - dd) < 5e-3]
            r = _summary(eps[:, j], M, S[j], sc.nb)
            ej = np.abs(mod[:, j].mean(0) - dft[j])
            r.update(cls=c, r=_r(dist(dd, dunit), 3), r_com_mean=_r(dist(np.nanmean(rcom[k_[j]]), dunit), 3),
                     tail_err_mae=_r(np.mean(ej) * fac, 6),
                     tail_err_kBT=_r(np.mean(ej) / kt, 4), dft_dE_mean_abs_kBT=_r(np.mean(np.abs(dft[j])) / kt, 4))
            out["by_class_distance"].append(r)
    # E_int (absolute, all distances) error in kBT and the chosen unit
    pv = V.mean(0)
    out["E_int_err_by_class_distance"] = [
        {"cls": c, "r": _r(dist(dd, dunit), 3), "n": int(m.sum()),
         "mae": _r(1e3 * np.mean(np.abs(pv[m] - ref[m])) * fac, 6),
         "mae_kBT": _r(1e3 * np.mean(np.abs(pv[m] - ref[m])) / kt, 3)}
        for c in sorted(set(cls.tolist())) for dd in sorted(set(np.round(d[cls == c], 2).tolist()))
        for m in [(cls == c) & (np.abs(d - dd) < 5e-3)]]
    out["marks"] = {"receptive_field": _r(dist(R_CUT, dunit), 3), "connected_reach": _r(dist(R_CONNECTED, dunit), 3)}
    sc._eps = dict(eps=eps, keep=keep, cls=C, d=D, scan=S, kl=(k_, l_))
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def ddE(sc, floor_meV: float):
    e = getattr(sc, "_ddE", None)
    if e is None:
        return None
    V, y, pair, fam, G = e["V"], e["y"], e["pair"], e["fam"], e["groups"]  # meV
    M = V.shape[0]
    keep = np.abs(y) >= floor_meV
    eps = (V - y) / np.where(keep, np.abs(y), np.nan)
    out = {"definition": "(ddE_model - ddE_DFT) / |ddE_DFT|; -sign(ddE_DFT) = flat model", "floor_meV": floor_meV,
           "n_below_floor": int((~keep).sum()), "by_series": {}, "by_series_n": []}
    for f in sorted(set(fam.tolist())):
        i = np.where((fam == f) & keep)[0]
        if not len(i):
            continue
        out["by_series"][f] = _summary(eps[:, i], M, pair[i], sc.nb)
        # signed-scale view: (model - DFT) / DFT, -1 = flat, 0 = exact (comparable with the dimer eps)
        out["by_series"][f]["signed_median"] = _r(np.median((V[:, i].mean(0) - y[i]) / y[i]), 4)
        ns = [(g.get("meta") or {}).get("n") for g in np.array(G, dtype=object)[i]]
        for n in sorted({x for x in ns if x is not None}):
            j = i[[x == n for x in ns]]
            r = {"series": f, "n_units": n, "n": int(len(j)),
                 "median": _r(np.median(eps[:, j].mean(0)), 4),
                 "signed_median": _r(np.median((V[:, j].mean(0) - y[j]) / y[j]), 4)}
            out["by_series_n"].append(r)
    return out


def e3(sc, floor_meV: float):
    e = getattr(sc, "_E3", None)
    if e is None:
        return None
    y, V, fam, rel = e["y"], e["V"], e["fam"], e["rel"]
    M = V.shape[0]
    out = {"definition": "(E3_model - E3_DFT) / |E3_DFT| (body-order prior); geometry-relative dE3 the same way",
           "floor_meV": floor_meV}
    k = np.abs(y) >= floor_meV
    if k.any():
        eps = (V[:, k] - y[k]) / np.abs(y[k])
        out["E3"] = _summary(eps, M, fam[k], sc.nb)
        out["E3"]["note"] = "includes the charge-state offset of the monomers (see the geometry-relative row)"
    if rel:
        yr = np.array([x[2] for x in rel])
        Vr = np.array([x[3] for x in rel]).T
        fr = np.array([x[0] for x in rel])
        k = np.abs(yr) >= floor_meV
        if k.any():
            out["dE3_vs_far"] = _summary((Vr[:, k] - yr[k]) / np.abs(yr[k]), M, fr[k], sc.nb)
    return out


def coulomb_prior(sc, gates, frag=None):
    """g * <lambda_A lambda_B> - 1 from the LES gate(s) and the model's latent fragment charges (ion pairs, d >= 8 A);
    + 1 per member whose GGUF carries the exact fragment-charge Coulomb (frag[k] true)."""
    if not gates or all(g is None for g in gates):
        return {"skipped": "no LES gate in the model (GGUF tensor plato_les_gate / pet.plato_lr.les)"}
    fr = np.array([bool(f) for f in (frag or [False] * len(gates))], float)
    g0 = np.array([np.nan if x is None else x for x in gates], float)
    if np.all(g0 == 0):  # LES switched off: the latent charges are untrained and never enter the energy
        out = {"gate_g": [_r(x, 4) for x in g0], "frag_coulomb": fr.astype(bool).tolist(), "eps_pred_per_member": [_r(x, 4) for x in fr - 1],
               "eps_pred": _r(float(np.mean(fr - 1)), 4), "note": "gate 0: eps_pred = (1 if exact fragment Coulomb else 0) - 1"}
        t = getattr(sc, "_eps", None)
        if t is not None:
            i = (t["cls"] == "ion_pair") & t["keep"]
            if i.any():
                out["eps_measured_median"] = _r(float(np.median(t["eps"][:, i].mean(0))), 4)
        return out
    lam = []
    for r in sc.ds.records:
        if r.get("cls") not in ("ion_pair",) or (r.get("distance") or 0) < R_CUT - 1e-6 or sc.ds.flagged(r["id"]):
            continue
        Q, fid, qf = sc.P.charges(r["id"]), r.get("fragment_id"), r.get("fragment_charges")
        if Q is None or fid is None or not qf or len(qf) != 2 or not all(qf):
            continue
        fid = np.asarray(fid)
        lam.append([(q[fid == 0].sum() / qf[0]) * (q[fid == 1].sum() / qf[1]) for q in Q])  # per member
    if not lam:
        return {"skipped": "the backend returned no latent charges (mlipcpp-eval writes latent_charges for LES models)"}
    lam = np.array(lam)  # (n, M)
    g = np.array([np.nan if x is None else x for x in gates], float)
    pred = g * lam.mean(0) + fr - 1
    out = {"gate_g": [_r(x, 4) for x in g], "frag_coulomb": fr.astype(bool).tolist(), "lambda_A_lambda_B_mean": [_r(x, 4) for x in lam.mean(0)],
           "n_records": int(len(lam)), "eps_pred_per_member": [_r(x, 4) for x in pred],
           "eps_pred": _r(float(np.nanmean(pred)), 4),
           "note": "eps_pred = g <lambda_A lambda_B> - 1, flat in r; compare eps_measured (ion pairs)"}
    t = getattr(sc, "_eps", None)
    if t is not None:
        i = (t["cls"] == "ion_pair") & t["keep"]
        if i.any():
            out["eps_measured_median"] = _r(float(np.median(t["eps"][:, i].mean(0))), 4)
    return out


def normalised(sc, floor_meV: float = 0.1, temperature: float = 300.0, distance_unit: str = "A",
               energy_unit: str = "meV", gates=None, frag=None) -> dict:
    out = {"spec": "docs/metrics.md#normalised-errors"}
    for k, v in (("dimer_tail", dimer_tail(sc, floor_meV, temperature, distance_unit, energy_unit)),
                 ("ddE", ddE(sc, floor_meV)), ("E3", e3(sc, floor_meV))):
        if v is not None:
            out[k] = v
    if getattr(sc, "_eint", None) is not None and "ion_pair" in set(sc._eint["cls"].tolist()):
        out["coulomb_prior"] = coulomb_prior(sc, gates, frag)
    return out
