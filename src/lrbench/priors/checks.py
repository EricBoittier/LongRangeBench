"""The priors and assumptions checks. Each is registered with its prior, its pass/fail rule (thresholds fixed
before the validation runs; see ``docs/priors.md``) and the known defect it is built to catch.

Conventions: energies from backends are eV, reported in meV unless named otherwise; d = closest inter-fragment
contact (A), r = centroid distance of the two fragments (A); E_int = E(AB) - E(A, q_A, s_A) - E(B, q_B, s_B) with the
monomers as rigid slices at their own fragment charge and spin.
"""
from __future__ import annotations

import collections
from typing import Dict, List

import numpy as np

from ..data import Structure, base_id, combine_spins, record_level, term_structure
from .core import FAIL, NA, NE, PASS, WARN, Ctx, check, outcome
from .probes import FINE_GRID, boxed, r_com, templates
from .synthetic import KE

ABS_FLOOR = 1e-4  # eV: 0.1 meV
DFT_FLOOR = 5e-4  # eV: SCF / grid noise of the labels
AMU_A2_FS2 = 103.642697  # eV per amu A^2 / fs^2
MASS = {"H": 1.008, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.99, "Mg": 24.305,
        "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Br": 79.904, "B": 10.81, "I": 126.9}


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def is_ct(cls: str) -> bool:
    return "ct_ambiguous" in (cls or "")


def is_ion_class(cls: str) -> bool:
    return not (cls or "").startswith("neutral")


def is_monopole_pair(cls: str) -> bool:
    """Both fragments carry a net charge, so the leading far-field term is monopole-monopole and E ~ 1/r exactly.

    This is the only family for which p = 1 is the right expectation. An ion against a neutral polar molecule is
    charge-dipole, p = 2; against a nonpolar one it is charge-induced-dipole, p = 4. Demanding 1 of those would
    fail a correct model for having the right physics.
    """
    c = (cls or "").lower()
    return c.startswith("ion_pair") or c.startswith("same_sign")


def _boot_median(x, groups, n=1000, seed=0):
    x, groups = np.asarray(x, float), np.asarray(groups)
    u = np.unique(groups)
    if len(u) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = {g: np.where(groups == g)[0] for g in u}
    m = [np.median(np.concatenate([x[idx[g]] for g in rng.choice(u, len(u))])) for _ in range(n)]
    return [_r(np.percentile(m, 2.5)), _r(np.percentile(m, 97.5))]


# ---------------------------------------------------------------------------------------------- shared probe data
def _templates(ctx: Ctx):
    if "_tpl" not in ctx.opts:
        ctx.opts["_tpl"] = templates(ctx.dimers) if ctx.dimers is not None else []
    return ctx.opts["_tpl"]


def scans(ctx: Ctx) -> List[dict]:
    """Fine closest-contact scans of every template (orientation 0): E_int(d), r, and the force on B along the axis."""
    if "_scans" in ctx.opts:
        return ctx.opts["_scans"]
    grid = ctx.opts.get("grid", FINE_GRID)
    out = []
    T = _templates(ctx)
    ab = [[t.ab(d) for d in grid] for t in T]
    mono = [[t.frag(0), t.frag(1)] for t in T]
    P = ctx.predict([s for row in ab for s in row] + [s for row in mono for s in row])
    k = 0
    for t, row in zip(T, ab):
        p = P[k:k + len(row)]
        k += len(row)
        out.append(dict(t=t, structs=row, preds=p))
    for o in out:
        pa, pb = P[k], P[k + 1]
        k += 2
        t, row, p = o["t"], o["structs"], o["preds"]
        E = np.array([np.nan if q.get("error") or q.get("energy") is None else q["energy"] for q in p])
        ea = np.nan if pa.get("error") or pa.get("energy") is None else pa["energy"]
        eb = np.nan if pb.get("error") or pb.get("energy") is None else pb["energy"]
        x0 = [np.asarray(s.positions) for s in row]
        u = x0[0][t.ib].mean(0) - x0[0][t.ia].mean(0)
        u /= np.linalg.norm(u)
        fB = np.array([np.nan if q.get("forces") is None else float(np.asarray(q["forces"])[t.ib].sum(0) @ u)
                       for q in p])
        o.update(d=np.asarray(grid, float), r=np.array([r_com(s, t.ia, t.ib) for s in row]), E=E, EA=ea, EB=eb,
                 Eint=E - ea - eb, fB=fB, noise=np.array([ctx.noise(e, ea, eb) for e in E]),
                 errors=[q.get("error") for q in p if q.get("error")] + [x.get("error") for x in (pa, pb) if x.get("error")])
    ctx.opts["_scans"] = out
    return out


def canon(ctx: Ctx) -> Dict[str, np.ndarray]:
    """Model and DFT E_int on every canonical E_int group of the dimer set (pathology-flagged groups excluded)."""
    if "_canon" in ctx.opts:
        return ctx.opts["_canon"]
    ds = ctx.dimers
    G = [g for g in ds.groups if g.get("kind") == "E_int" and not ds.group_flagged(g)]
    terms = sorted({t for g in G for t, _ in g["terms"]})
    S = [term_structure(ds, t) for t in terms]
    pred = dict(zip(terms, ctx.predict(S)))
    E = {t: (np.nan if p.get("error") or p.get("energy") is None else p["energy"]) for t, p in pred.items()}
    rows = collections.defaultdict(list)
    for g in G:
        rec = ds.by_id[base_id(g["terms"][0][0])]
        ex = rec.get("extra") or {}
        v = sum(w * E[t] for t, w in g["terms"])
        rows["id"].append(rec["id"])
        rows["scan"].append(f"{ex.get('pair', rec.get('group'))}:o{ex.get('orient', 0)}")
        rows["pair"].append(ex.get("pair", rec.get("group")))
        rows["cls"].append(rec.get("cls") or "")
        rows["d"].append(float(rec["distance"]))
        rows["r"].append(float(ex.get("r_com") or np.nan))
        rows["ref"].append(float(g["ref"]))
        rows["model"].append(v)
        rows["noise"].append(ctx.noise(*[E[t] for t, _ in g["terms"]]))
        rows["Q"].append(int(rec["charge"]))
        rows["fq"].append(tuple(int(x) for x in (rec.get("fragment_charges") or [0, 0])))
        refs = rec.get("refs") or {}
        es, ind, mp = (refs.get(k) for k in ("E_es_exact", "E_mp_ind", "E_mp"))
        rows["phys_es"].append(np.nan if es is None else es + (ind or 0.0))
        rows["phys_mp"].append(np.nan if mp is None else mp)
        rows["error"].append(any(pred[t].get("error") for t, _ in g["terms"]))
        rows["rec"].append(rec)
    out = {k: (np.array(v) if k not in ("rec", "fq") else v) for k, v in rows.items()}
    ctx.opts["_canon"] = out
    return out


def _need_dimers(ctx):
    return None if ctx.dimers is not None else outcome(NE, "no dimer set loaded")


# ============================================================================================== physical priors
def _exp_fit(r, E, floor):
    """Offset-free far-field exponent: fit log|dE/dr| between consecutive points against log r_mid; E ~ r^-n gives
    slope -(n + 1). Needs >= 2 differences above ``floor`` (3 points). Returns (n, n_used)."""
    o = np.argsort(r)
    r, E = r[o], E[o]
    dr, dE = np.diff(r), np.diff(E)
    fl = floor if np.ndim(floor) == 0 else np.maximum(floor[o][1:], floor[o][:-1])
    ok = np.isfinite(dE) & (np.abs(dE) > fl) & (dr > 0)
    if ok.sum() < 2:
        return None, int(ok.sum())
    rm = np.sqrt(r[1:] * r[:-1])[ok]
    s = np.polyfit(np.log(rm), np.log(np.abs(dE[ok]) / dr[ok]), 1)[0]
    return float(-s - 1), int(ok.sum())


@check("far_field_exponent", "physical", "far-field asymptotics: E_int ~ r^-n with n = 1 ion-ion, 2 ion-dipole, "
       "3 dipole-dipole (ion + nonpolar mixes 3 and 4)",
       "per scan, fit n from log|dE/dr| on the canonical 10-25 A points (model and DFT on the same geometries); a scan "
       "passes if |n_model - n_DFT| <= 0.5 with the same sign of dE; a model whose differences are all below its "
       "numerical floor where DFT's are not has a missing tail (fail). The check fails if any non-CT class has a pass "
       "fraction < 0.8 among DFT-evaluable scans",
       "flat local models beyond the cutoff (PET-OMol-S, FTB): tail missing")
def far_field_exponent(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    C = canon(ctx)
    far = C["d"] >= 10 - 1e-6
    by = collections.defaultdict(lambda: collections.Counter())
    nref, nmod, scans_out = collections.defaultdict(list), collections.defaultdict(list), []
    for sc in sorted(set(C["scan"][far].tolist())):
        i = np.where(far & (C["scan"] == sc))[0]
        cls = C["cls"][i[0]]
        nr, kr = _exp_fit(C["r"][i], C["ref"][i], DFT_FLOOR)
        if nr is None:
            by[cls]["dft_below_floor"] += 1
            continue
        nm, km = _exp_fit(C["r"][i], C["model"][i], np.maximum(ABS_FLOOR, 4 * C["noise"][i]))
        o = np.argsort(C["r"][i])
        sgn = np.sign(np.diff(C["model"][i][o])[-1]) == np.sign(np.diff(C["ref"][i][o])[-1])
        if nm is None:
            v = "tail_missing"
        else:
            v = "ok" if abs(nm - nr) <= 0.5 and sgn else "wrong_exponent"
        by[cls][v] += 1
        nref[cls].append(nr)
        if nm is not None:
            nmod[cls].append(nm)
        scans_out.append({"scan": sc, "cls": cls, "n_dft": _r(nr, 3), "n_model": _r(nm, 3), "verdict": v})
    # asymptotic exponent of the model alone on the fine scans (25-100 A), against the class expectation
    asym = {}
    for o in scans(ctx):
        j = o["d"] >= 25 - 1e-6
        n, _ = _exp_fit(o["r"][j], o["Eint"][j], np.maximum(ABS_FLOOR, 4 * o["noise"][j]))
        asym.setdefault(o["t"].cls, []).append(n)
    rows, fails = {}, []
    for cls, c in sorted(by.items()):
        ev = c["ok"] + c["wrong_exponent"] + c["tail_missing"]
        frac = c["ok"] / ev if ev else None
        rows[cls] = {"n_scans_evaluable": ev, **dict(c), "pass_fraction": _r(frac, 3),
                     "median_n_dft": _r(np.median(nref[cls]), 3) if nref[cls] else None,
                     "median_n_model": _r(np.median(nmod[cls]), 3) if nmod[cls] else None,
                     "model_n_25_100A_median": _r(np.median([x for x in asym.get(cls, []) if x is not None]), 3)
                     if any(x is not None for x in asym.get(cls, [])) else None}
        if ev and not is_ct(cls) and frac < 0.8:
            fails.append(f"{cls} {c['ok']}/{ev} (tail missing {c['tail_missing']})")
    if not any(r["n_scans_evaluable"] for r in rows.values()):
        return outcome(NE, "no scan with DFT far-field differences above the floor", by_class=rows)
    st = FAIL if fails else PASS
    return outcome(st, "; ".join(fails) if fails else "exponents match DFT in every class",
                   by_class=rows, scans=scans_out)


#: Forces below this (eV/A) are indistinguishable from a model that returns nothing in the tail.
FORCE_FLOOR = 1e-4


def _tail_mask(o, dmin):
    """Fine-scan points beyond ``dmin`` where the model's own E_int and axial force are both above their floors.

    Both floors matter and for different reasons. Below the energy floor the ratio below is 0/0. Below the force
    floor there is no force to compare anything with -- which is the finding for a flat model, and it must be
    reported as *not evaluable* rather than as a pass.
    """
    import numpy as _np
    f, E, noise = o["fB"], o["Eint"], o["noise"]
    ok = _np.isfinite(f) & _np.isfinite(E) & (o["d"] >= dmin - 1e-6)
    return ok & (_np.abs(E) > _np.maximum(ABS_FLOOR, 4 * noise)) & (_np.abs(f) > FORCE_FLOOR)


@check("conservativeness", "model", "a conservative potential: the axial force equals minus the energy gradient, "
       "F_B.u = -dE_int/dr, at every separation. This is a property of the model alone -- no reference enters it",
       "fine closest-contact scans, rigid translation of B along the axis so dr = dd. Central differences of "
       "E_int(r) against the reported force, scored as the median |F + dE/dr| / max(|F|, force floor), near "
       "(d < 8 A) and far (d >= 8 A) separately: median relative deviation <= 0.1 in each region with >= 4 points",
       "direct-force architectures, whose forces are a separate head rather than a gradient, and any model whose "
       "tail force and tail energy disagree")
def conservativeness(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    regions = {"near": (0.0, 8.0), "far": (8.0, np.inf)}
    acc = {k: [] for k in regions}
    rows = []
    for o in scans(ctx):
        k = np.argsort(o["r"])                      # one ordering, applied to every array together
        r, E, f, d = o["r"][k], o["Eint"][k], o["fB"][k], o["d"][k]
        good = np.isfinite(E) & np.isfinite(f)
        if good.sum() < 4:
            continue
        r, E, f, d = r[good], E[good], f[good], d[good]
        # central differences on a possibly uneven grid; the endpoints are dropped rather than taken one-sided, so
        # a scan's two extreme points cannot dominate the median through a worse derivative estimate
        dEdr = (E[2:] - E[:-2]) / (r[2:] - r[:-2])
        fm, d_m = f[1:-1], d[1:-1]
        scale = np.maximum(np.abs(fm), FORCE_FLOOR)
        rel = np.abs(fm + dEdr) / scale
        for name, (lo, hi) in regions.items():
            m = (d_m >= lo) & (d_m < hi)
            if m.sum() >= 1:
                acc[name].extend(rel[m].tolist())
        rows.append({"template": getattr(o["t"], "name", None) or str(getattr(o["t"], "cls", "")),
                     "cls": getattr(o["t"], "cls", None), "n": int(len(rel)),
                     "median_rel": _r(np.median(rel), 4)})
    out, fails = {}, []
    for name in regions:
        v = np.array(acc[name], float)
        v = v[np.isfinite(v)]
        out[name] = {"n_points": int(v.size), "median_rel_deviation": _r(np.median(v), 4) if v.size else None}
        if v.size >= 4 and np.median(v) > 0.1:
            fails.append(f"{name} median |F + dE/dr|/|F| = {np.median(v):.3f}")
    if not any(out[n]["n_points"] for n in regions):
        return outcome(NE, "no scan with four finite energy and force points", by_region=out)
    st = FAIL if fails else PASS
    return outcome(st, "; ".join(fails) if fails else "forces agree with the energy gradient everywhere measured",
                   by_region=out, scans=rows[:40])


@check("virial_tail", "physical", "the virial relation in the tail: for E_int ~ A/r^p the model's own force and "
       "energy satisfy r F_r / E_int = p, so p = 1 for a Coulomb tail. The expected value is known without any "
       "reference, which is what makes this measurable at separations where DFT would be prohibitive",
       "fine-scan points at d >= 12 A with |E_int| above the model floor and |F| > 1e-4 eV/A: per class with >= 4 "
       "such points, |median p - 1| <= 0.5 for the classes where *both* fragments are charged, the only ones whose "
       "leading term is monopole-monopole; p is reported but not scored for ion-neutral classes, whose leading "
       "terms are charge-dipole (p = 2) and charge-induced-dipole (p = 4). A class whose points all fall below "
       "those floors is not evaluable, not a pass: there is no tail to measure",
       "a flat tail (no force to measure), a tail of the wrong power, and a tail whose force and energy imply "
       "different exponents (which a reference-based exponent fit cannot see)")
def virial_tail(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    dmin = float(ctx.opts.get("virial_dmin", 12.0))
    by = collections.defaultdict(list)
    floored = collections.Counter()
    for o in scans(ctx):
        cls = getattr(o["t"], "cls", None) or "all"
        m = _tail_mask(o, dmin)
        n_far = int((o["d"] >= dmin - 1e-6).sum())
        if not m.any():
            floored[cls] += n_far
            continue
        p = o["r"][m] * o["fB"][m] / o["Eint"][m]
        by[cls].extend(p[np.isfinite(p)].tolist())
        floored[cls] += n_far - int(m.sum())
    rows, fails = {}, []
    for cls in sorted(set(by) | set(floored)):
        v = np.array(by.get(cls, []), float)
        rows[cls] = {"n_points": int(v.size), "n_below_floor": int(floored[cls]),
                     "scored": bool(is_monopole_pair(cls) and not is_ct(cls)),
                     "expected_p": 1 if is_monopole_pair(cls) else None,
                     "median_p": _r(np.median(v), 3) if v.size else None,
                     "iqr_p": [_r(np.percentile(v, 25), 3), _r(np.percentile(v, 75), 3)] if v.size >= 4 else None}
        if v.size >= 4 and is_monopole_pair(cls) and not is_ct(cls) and abs(np.median(v) - 1.0) > 0.5:
            fails.append(f"{cls} p = {np.median(v):.2f}")
    if not any(r["n_points"] for r in rows.values()):
        return outcome(NE, f"no point beyond {dmin:g} A with both |E_int| and |F| above their floors "
                           "(a flat tail has no virial to measure)", by_class=rows)
    st = FAIL if fails else PASS
    return outcome(st, "; ".join(fails) if fails else "r F / E_int = 1 in every charged class measured",
                   by_class=rows, d_min=dmin)


@check("eps_far", "physical", "physics-normalised far-field error eps = (dE_model - dE_DFT)/dE_phys "
       "(ERM-with-priors; eps = g lambda^2 - 1 for a gated Coulomb head, -1 = tail missing)",
       "dE relative to the scan's longest canonical point, d >= 12 A, dE_phys = E_es_exact + E_mp_ind else E_mp, "
       "|dE_phys| >= 0.1 meV and above the model floor; per class with >= 3 points: |median eps| <= 0.25",
       "flat models (eps = -1), over/under-scaled Coulomb heads")
def eps_far(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    C = canon(ctx)
    rows, fails, d8 = {}, [], {}
    eps_all, cls_all, scan_all, dist_all = [], [], [], []
    for sc in set(C["scan"].tolist()):
        i = np.where(C["scan"] == sc)[0]
        l = i[np.argmax(C["d"][i])]
        for k in i[(C["d"][i] >= 8 - 1e-6) & (i != l)]:
            phys = C["phys_es"][k] - C["phys_es"][l]
            if not np.isfinite(phys):
                phys = C["phys_mp"][k] - C["phys_mp"][l]
            if not np.isfinite(phys) or abs(phys) < ABS_FLOOR or abs(phys) < 4 * (C["noise"][k] + C["noise"][l]):
                continue
            dm, dd = C["model"][k] - C["model"][l], C["ref"][k] - C["ref"][l]
            if not np.isfinite(dm):
                continue
            eps_all.append((dm - dd) / phys)
            cls_all.append(C["cls"][k])
            scan_all.append(sc)
            dist_all.append(C["d"][k])
    eps_all, cls_all, scan_all, dist_all = map(np.array, (eps_all, cls_all, scan_all, dist_all))
    if not len(eps_all):
        return outcome(NE, "no far-field points with a physical scale above the floor")
    for cls in sorted(set(cls_all.tolist())):
        j = (cls_all == cls) & (dist_all >= 12 - 1e-6)
        j8 = (cls_all == cls) & (np.abs(dist_all - 8) < 1e-2)
        if j8.any():
            d8[cls] = _r(np.median(eps_all[j8]), 3)
        if j.sum() < 3:
            continue
        med = float(np.median(eps_all[j]))
        rows[cls] = {"n": int(j.sum()), "median_eps": _r(med, 3), "ci95": _boot_median(eps_all[j], scan_all[j]),
                     "mean_eps": _r(np.mean(eps_all[j]), 3)}
        if abs(med) > 0.25 and not is_ct(cls):
            fails.append(f"{cls} eps {med:+.2f}")
    if not rows:
        return outcome(NE, "fewer than 3 points per class at d >= 12 A", median_eps_at_8A=d8)
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "|median eps| <= 0.25 in every class",
                   by_class=rows, median_eps_at_8A_info=d8)


def _runs(mask, x):
    """Contiguous True runs of mask -> list of (i0, i1) inclusive."""
    out, i0 = [], None
    for i, m in enumerate(mask):
        if m and i0 is None:
            i0 = i
        if not m and i0 is not None:
            out.append((i0, i - 1))
            i0 = None
    if i0 is not None:
        out.append((i0, len(mask) - 1))
    return out


@check("seam", "physical", "smoothness: no plateau where the physical prior has a slope, no ramp/cliff steeper than "
       "the prior (switch / cutoff / conditioning-window seams)",
       "fine scans (0.05 A in d = 6-12 A). Plateau: |dE/dr| < 0.1 |dE_Coul/dr| over >= 0.3 A of r for a pair with "
       "|q_A q_B| >= 1, followed by a recovery (|dE/dr| >= 0.5 |dE_Coul/dr|) before d = 20 A (a plateau that never "
       "recovers is a missing tail, scored by far_field_exponent). Cliff: |dE/dr| > 5 max(|dE_Coul/dr|, 0.05 eV/A) and "
       "> 0.25 eV/A at d >= 6 A, any class. Fail on any bounded plateau or cliff outside the CT-ambiguous class",
       "the FTq 7.5-9.3 A plateau and 14-20 eV/A ramp; the FTB / bare-ion edge-loss cliff at 7.25-7.8 A")
def seam(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    hits, rows, n_eval = [], [], 0
    for o in scans(ctx):
        t = o["t"]
        ok = np.isfinite(o["Eint"])
        if ok.sum() < 10:
            continue
        d, r, E = o["d"][ok], o["r"][ok], o["Eint"][ok]
        n_eval += 1
        dm, rm = 0.5 * (d[1:] + d[:-1]), 0.5 * (r[1:] + r[:-1])
        s = np.diff(E) / np.diff(r)
        sp = -KE * t.qa * t.qb / rm ** 2
        nz = np.maximum(4 * np.maximum(o["noise"][ok][1:], o["noise"][ok][:-1]) / np.diff(r), 0)
        win = (dm >= 6 - 1e-6) & (dm <= 12 + 1e-6)
        res = {"scan": t.scan_id, "cls": t.cls, "plateaus": [], "cliffs": []}
        if abs(t.qa * t.qb) >= 1:
            flat = win & (np.abs(s) < np.maximum(0.1 * np.abs(sp), nz))
            for a, b in _runs(flat, dm):
                if rm[b] - rm[a] < 0.3 - 1e-9:
                    continue
                after = (np.arange(len(s)) > b) & (dm <= 20 + 1e-6) & (np.abs(s) >= 0.5 * np.abs(sp))
                if after.any():
                    res["plateaus"].append({"d": [_r(dm[a], 2), _r(dm[b], 2)], "level_eV": _r(np.mean(E[a:b + 2]), 3),
                                            "recovers_by_d": _r(dm[np.where(after)[0][0]], 2)})
        ratio = np.abs(s) / np.maximum(np.abs(sp), 0.05)
        cl = (dm >= 6 - 1e-6) & (ratio > 5) & (np.abs(s) > 0.25)
        for a, b in _runs(cl, dm):
            res["cliffs"].append({"d": [_r(dm[a], 2), _r(dm[b], 2)], "max_slope_eV_A": _r(np.abs(s[a:b + 1]).max(), 3),
                                  "prior_slope_eV_A": _r(np.abs(sp[a:b + 1]).max(), 4)})
        res["max_slope_d_ge_6_eV_A"] = _r(np.abs(s[dm >= 6 - 1e-6]).max(), 3) if (dm >= 6 - 1e-6).any() else None
        rows.append(res)
        if (res["plateaus"] or res["cliffs"]) and not is_ct(t.cls):
            hits.append(res)
    if not n_eval:
        return outcome(NE, "no fine scan evaluated")
    win = collections.defaultdict(lambda: [np.inf, -np.inf])
    for h in rows:
        for x in h["plateaus"] + h["cliffs"]:
            w = win[h["cls"]]
            w[0], w[1] = min(w[0], x["d"][0]), max(w[1], x["d"][1])
    windows = {k: v for k, v in win.items()}
    msg = (f"{len(hits)} scans with seams; windows (d, A) " +
           ", ".join(f"{k} {v[0]:.2f}-{v[1]:.2f}" for k, v in sorted(windows.items()))) if hits else "no seam"
    return outcome(FAIL if hits else PASS, msg, windows_by_class=windows, scans=rows)


@check("size_consistency", "physical", "separability: E(A...B at 100 A) = E(A, q_A) + E(B, q_B) + k q_A q_B / r",
       "residual R = E(AB, d = 100 A) - E(A) - E(B) - 14.40 q_A q_B / r; pass if |R| <= 10 meV (+ 4x float floor) for "
       ">= 95 % of non-CT pairs. Also reports R at 25 / 50 A and which charge assignment (own, total-charge rows "
       "(Q, Q), (Q, 0), (0, Q)) the 100 A energy is closest to",
       "conditioning on total-charge rows after the fragments disconnect; separated-fragment offsets")
def size_consistency(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    S = scans(ctx)
    rows, bad, n = [], [], 0
    alt_jobs = []
    for o in S:
        t = o["t"]
        Q = t.qa + t.qb
        if Q != 0 or t.qa != 0:
            for qa_, qb_ in ((Q, Q), (Q, 0), (0, Q)):
                if (qa_, qb_) != (t.qa, t.qb):
                    if (qa_, qb_) == (Q, Q):  # what a conditioned backbone sees once the graph splits: (Q, S_AB) rows
                        sa = sb = combine_spins([t.sa, t.sb])
                    else:
                        sa = t.sa if (qa_ - t.qa) % 2 == 0 else (2 if t.sa == 1 else t.sa - 1)
                        sb = t.sb if (qb_ - t.qb) % 2 == 0 else (2 if t.sb == 1 else t.sb - 1)
                    alt_jobs.append((t.scan_id, (qa_, qb_), t.frag(0, qa_, sa), t.frag(1, qb_, sb)))
    altE = ctx.energies([s for j in alt_jobs for s in j[2:]])
    alt = collections.defaultdict(dict)
    for k, j in enumerate(alt_jobs):
        alt[j[0]][j[1]] = altE[2 * k] + altE[2 * k + 1]
    for o in S:
        t = o["t"]
        res = {"scan": t.scan_id, "cls": t.cls, "qA": t.qa, "qB": t.qb}
        for D in (25.0, 50.0, 100.0):
            k = np.where(np.abs(o["d"] - D) < 1e-6)[0]
            if len(k):
                k = k[0]
                res[f"R_{D:g}A_meV"] = _r(1e3 * (o["Eint"][k] - KE * t.qa * t.qb / o["r"][k]), 2)
        k = np.where(np.abs(o["d"] - 100.0) < 1e-6)[0]
        if not len(k) or not np.isfinite(o["Eint"][k[0]]):
            rows.append(res)
            continue
        k = k[0]
        R = o["Eint"][k] - KE * t.qa * t.qb / o["r"][k]
        tol = 0.010 + 4 * o["noise"][k]
        cand = {"own": abs(R)}
        for (qa_, qb_), es in alt[t.scan_id].items():
            if np.isfinite(es):
                # (Q, 0) / (0, Q) carry no Coulomb; (Q, Q) rows are the bare fragment sum (the conditioning artefact)
                cand[f"({qa_},{qb_})"] = abs(o["E"][k] - es)
        res.update(R_meV=_r(1e3 * R, 2), tol_meV=_r(1e3 * tol, 2), closest_assignment=min(cand, key=cand.get),
                   ok=bool(abs(R) <= tol))
        rows.append(res)
        if not is_ct(t.cls):
            n += 1
            if abs(R) > tol:
                bad.append(res)
    if not n:
        return outcome(NE, "no pair evaluated at 100 A")
    frac = 1 - len(bad) / n
    worst = max(bad, key=lambda r: abs(r["R_meV"]), default=None)
    msg = (f"{n - len(bad)}/{n} pairs separable; worst {worst['scan']} R = {worst['R_meV']:+.1f} meV "
           f"(closest to {worst['closest_assignment']})") if bad else f"{n}/{n} pairs separable to 10 meV"
    return outcome(PASS if frac >= 0.95 else FAIL, msg, pass_fraction=_r(frac, 3),
                   assignments=dict(collections.Counter(r.get("closest_assignment") for r in rows if "R_meV" in r)),
                   pairs=rows)


@check("energy_force", "physical", "energy-force consistency: F = -dE/dR (work-energy theorem)",
       "(a) along every fine scan (rigid B translation), per step |dE_AB + 0.5 (f_k + f_k+1) dr| <= 2 meV + 2 % |dE| "
       "+ 4x float floor; a step that misses is re-integrated on 20 sub-steps, then its worst sub-step is bisected to "
       "~1e-5 A: if the work closes it is a conservative cliff (reported), if an energy jump remains that no force "
       "carries it is a discontinuity; (b) central differences, h = 0.01 A, on "
       "<= 12 structures (bare ions included): |F_fd - F| <= 0.02 eV/A + 2 floor/h, components whose h and h/2 estimates disagree are counted as unresolved, not failed. Fail if > 1 % of resolved steps "
       "or any finite-difference component fails, or any discontinuity exceeds 50 meV",
       "finite-difference conditioning forces on edge-free (bare-ion) structures (mlip.cpp D2, 3 eV/A)")
def energy_force(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    n_res = n_unres = n_refined_ok = 0
    bad_steps, worst, cand = [], 0.0, []
    have_f = False
    for o in scans(ctx):
        E, f, r = o["E"], o["fB"], o["r"]
        ok = np.isfinite(E) & np.isfinite(f)
        if ok.sum() < 2:
            continue
        have_f = True
        E, f, r, nz, d = E[ok], f[ok], r[ok], o["noise"][ok], o["d"][ok]
        dE, dr = np.diff(E), np.diff(r)
        W = -0.5 * (f[1:] + f[:-1]) * dr
        work_err = np.abs(dE - W)
        tol = 0.002 + 0.02 * np.abs(dE) + 4 * np.maximum(nz[1:], nz[:-1])
        for k in np.where(work_err > tol)[0]:
            cand.append((o, d[k], d[k + 1], E[k], E[k + 1], f[k], f[k + 1], r[k], r[k + 1], tol[k],
                         work_err[k]))
        n_res += len(dE)
    # every mismatched step is re-integrated on 20 sub-steps; it fails only if the refined work still misses dE,
    # and is 'unresolved' if the force changes by > 0.05 eV per sub-step (a cliff narrower than the sub-grid)
    cand.sort(key=lambda c: -c[-1])
    for o, d0, d1, e0, e1, f0, f1, r0, r1, tl, m0 in cand[:ctx.opts.get("refine_max", 400)]:
        t = o["t"]
        dd = np.linspace(d0, d1, 21)[1:-1]
        sub = [t.ab(x, "ef") for x in dd]
        P = ctx.predict(sub)
        u = (np.asarray(sub[0].positions)[t.ib].mean(0) - np.asarray(sub[0].positions)[t.ia].mean(0))
        u /= np.linalg.norm(u)
        try:
            ff = np.array([f0] + [float(np.asarray(p["forces"])[t.ib].sum(0) @ u) for p in P] + [f1])
        except (TypeError, KeyError):
            continue
        rr = np.array([r0] + [r_com(x, t.ia, t.ib) for x in sub] + [r1])
        ee = np.array([e0] + [np.nan if p.get("energy") is None else p["energy"] for p in P] + [e1])
        Wj = -0.5 * (ff[1:] + ff[:-1]) * np.diff(rr)
        mj = np.abs(np.diff(ee) - Wj)
        mr = abs((e1 - e0) - Wj.sum())
        if mr <= tl:
            n_refined_ok += 1
            continue
        # bisect the worst sub-step down to ~1e-5 A: a narrow conservative cliff shows its force spike on the way and
        # the work closes; an energy jump that no force carries (a discontinuity) keeps its mismatch
        j = int(np.nanargmax(mj))
        da, db = np.r_[d0, dd, d1][j], np.r_[d0, dd, d1][j + 1]
        A = (ee[j], ff[j], rr[j])
        B = (ee[j + 1], ff[j + 1], rr[j + 1])
        closed = False
        # hoisted: defining this inside the loop left `mis` bound to the work-error *array* on the break path
        # below, so a backend that failed one bisection probe raised TypeError and the check reported
        # not_evaluable instead of the discontinuity it had already found
        def mis(X, Y):
            return abs((Y[0] - X[0]) + 0.5 * (X[1] + Y[1]) * (Y[2] - X[2]))

        for _ in range(11):
            dm = 0.5 * (da + db)
            sm = t.ab(dm, "bis")
            pm = ctx.predict([sm])[0]
            if pm.get("energy") is None or pm.get("forces") is None:
                break
            M = (pm["energy"], float(np.asarray(pm["forces"])[t.ib].sum(0) @ u), r_com(sm, t.ia, t.ib))
            if mis(A, M) <= tl and mis(M, B) <= tl:
                closed = True
                break
            if mis(A, M) >= mis(M, B):
                db, B = dm, M
            else:
                da, A = dm, M
        jump = mis(A, B) if not closed else 0.0
        if closed or jump <= tl:
            n_unres += 1  # conservative, but steeper than the 0.05 A grid resolves
        else:
            bad_steps.append({"scan": t.scan_id, "d": [_r(d0, 3), _r(d1, 3)], "dE_meV": _r(1e3 * (e1 - e0), 2),
                              "work_meV": _r(1e3 * Wj.sum(), 2), "mismatch_meV": _r(1e3 * mr, 2),
                              "jump_at_d_A": _r(0.5 * (da + db), 5), "jump_width_A": float(db - da),
                              "jump_meV": _r(1e3 * jump, 2)})
            worst = max(worst, jump)
    n_res -= n_unres
    if not have_f:
        return outcome(NA, "backend returns no forces")
    # (b) finite differences
    T = _templates(ctx)
    pick = [t for t in T if t.bare_ions][:4] + [t for t in T if not t.bare_ions][:4]
    structs = [t.ab(d, "fd") for t in pick for d in ((8.5, 9.0, 12.0) if t.bare_ions else (6.0, 9.0))][:12]
    rng = np.random.default_rng(0)
    h, fd_rows, fd_bad, fd_unres = 0.01, [], [], 0
    jobs, meta = [], []
    for s in structs:
        for a in rng.choice(len(s.species), min(2, len(s.species)), replace=False):
            for c in range(3):
                for hh in (h, h / 2):
                    for sg in (1, -1):
                        x = np.asarray(s.positions, float).copy()
                        x[a, c] += sg * hh
                        jobs.append(Structure(f"{s.key}|fd{a}{c}{sg}{hh}", s.species, x, s.charge, s.spin))
                meta.append((s, a, c))
    base = ctx.predict(structs)
    Efd = ctx.energies(jobs).reshape(-1, 4)
    bmap = {s.key: b for s, b in zip(structs, base)}
    for (s, a, c), e4 in zip(meta, Efd):
        b = bmap[s.key]
        if b.get("forces") is None or not np.isfinite(e4).all():
            continue
        f1, f2 = -(e4[0] - e4[1]) / (2 * h), -(e4[2] - e4[3]) / h
        fa = float(np.asarray(b["forces"])[a, c])
        nz = 2 * ctx.noise(b["energy"]) / (h / 2)
        tol = 0.02 + nz
        # h and h/2 disagree beyond the float noise: curvature unresolved at this h (a ramp), not a force error
        resolved = abs(f1 - f2) <= 0.25 * tol + nz
        row = {"structure": s.key, "atom": int(a), "comp": c, "F": _r(fa, 4), "F_fd": _r(f2, 4), "tol": _r(tol, 4),
               "resolved": bool(resolved)}
        fd_rows.append(row)
        if not resolved:
            fd_unres += 1
        elif abs(f2 - fa) > tol:
            fd_bad.append(row)
    frac_bad = len(bad_steps) / max(n_res, 1)
    st = FAIL if (frac_bad > 0.01 or fd_bad or worst > 0.05) else PASS
    if not n_res and not fd_rows:
        st = NE
    msg = (f"work-energy: {len(bad_steps)}/{n_res} steps off after refinement (worst {1e3 * worst:.1f} meV), "
           f"{n_unres} unresolved cliffs, {n_refined_ok} fixed by refinement; FD: {len(fd_bad)}/{len(fd_rows) - fd_unres} components off, "
           f"{fd_unres} unresolved at h = 0.01 / 0.005 A")
    return outcome(st, msg, n_resolved=n_res, n_unresolved=n_unres, n_fixed_by_refinement=n_refined_ok, n_candidates=len(cand), bad_steps=bad_steps[:50], fd=fd_rows,
                   fd_bad=fd_bad)


def _verlet(ctx: Ctx, s: Structure, v0: np.ndarray, dt: float, nmax: int, stop=None):
    m = np.array([MASS.get(e, 12.0) for e in s.species])[:, None]
    x, v = np.asarray(s.positions, float).copy(), v0.copy()

    def ef(x):
        p = ctx.backend.predict([Structure(s.key, s.species, x, s.charge, s.spin)])[0]
        ctx.n_calls += 1
        if p.get("error") or p.get("forces") is None:
            raise RuntimeError(p.get("error") or "no forces")
        return p["energy"], np.asarray(p["forces"], float)

    E, F = ef(x)
    traj = []
    for k in range(nmax):
        ke = 0.5 * float((m * v * v).sum()) * AMU_A2_FS2
        traj.append((k * dt, E + ke, E, x.copy()))
        if stop and stop(x):
            break
        v += 0.5 * dt * F / m / AMU_A2_FS2
        x += dt * v
        E, F = ef(x)
        v += 0.5 * dt * F / m / AMU_A2_FS2
    return traj


@check("nve_short", "physical", "energy conservation in short NVE (velocity Verlet, dt = 0.5 fs)",
       "bare-ion pairs fired at each other from d = 12 A (repulsive pairs with the Coulomb energy to reach 4 A, "
       "attractive ones with 0.5 eV) until d < 4 A or > 13 A (<= 1200 steps), plus one ion + molecule pair at 4.5 A "
       "and 300 K for 400 steps: max |E_tot(t) - E_tot(0)| <= 20 meV per trajectory",
       "the bare-ion cliff: Li+...Na+ ~10 eV NVE error from the edge-loss cliff at 7.6-7.8 A (PR #23 notes)")
def nve_short(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    T = _templates(ctx)
    bare = [t for t in T if t.bare_ions][:6]
    mol = [t for t in T if not t.bare_ions and t.qa * t.qb == 0 and (t.qa or t.qb)][:1]
    rows, bad = [], []
    for t in bare + mol:
        try:
            if t.bare_ions:
                s = t.ab(12.0, "nve")
                x = np.asarray(s.positions)
                u = x[1] - x[0]
                u /= np.linalg.norm(u)
                m = np.array([MASS.get(e, 12.0) for e in s.species])
                mu = m[0] * m[1] / (m[0] + m[1])
                ke = KE * t.qa * t.qb * (1 / 4.0 - 1 / 12.0) * 1.1 if t.qa * t.qb > 0 else 0.5
                vrel = np.sqrt(2 * ke / mu / AMU_A2_FS2)
                v0 = np.array([u * vrel * m[1] / (m[0] + m[1]), -u * vrel * m[0] / (m[0] + m[1])])
                stop = lambda x: not (4.0 < np.linalg.norm(x[1] - x[0]) < 13.0)
                tr = _verlet(ctx, s, v0, 0.5, 1200, stop)
            else:
                s = t.ab(4.5, "nve")
                m = np.array([MASS.get(e, 12.0) for e in s.species])[:, None]
                rng = np.random.default_rng(1)
                v0 = rng.normal(size=(len(m), 3)) * np.sqrt(8.617333e-5 * 300 / m / AMU_A2_FS2)
                v0 -= (m * v0).sum(0) / m.sum()
                tr = _verlet(ctx, s, v0, 0.5, 400)
        except RuntimeError as err:
            rows.append({"scan": t.scan_id, "error": str(err)[:200]})
            continue
        Et = np.array([q[1] for q in tr])
        dev = np.abs(Et - Et[0])
        k = int(np.argmax(np.abs(np.diff(Et)))) if len(Et) > 1 else 0
        xk = tr[k][3]
        dk = float(np.min(np.linalg.norm(xk[t.ia][:, None] - xk[t.ib][None], axis=-1)))
        row = {"scan": t.scan_id, "steps": len(tr), "max_dev_meV": _r(1e3 * dev.max(), 2),
               "largest_jump_meV": _r(1e3 * np.abs(np.diff(Et)).max(), 2) if len(Et) > 1 else 0,
               "largest_jump_at_d_A": _r(dk, 2), "min_d_A": _r(min(
                   float(np.min(np.linalg.norm(q[3][t.ia][:, None] - q[3][t.ib][None], axis=-1))) for q in tr), 2)}
        rows.append(row)
        if dev.max() > 0.020:
            bad.append(row)
    done = [r for r in rows if "error" not in r]
    if not done:
        return outcome(NA if rows else NE, "no trajectory ran" + (f": {rows[0]['error']}" if rows else ""), runs=rows)
    msg = ("; ".join(f"{r['scan']} {r['max_dev_meV']:.0f} meV (jump at d {r['largest_jump_at_d_A']} A)" for r in bad)
           if bad else f"{len(done)} trajectories conserve energy to 20 meV")
    return outcome(FAIL if bad else PASS, msg, runs=rows)


@check("invariance", "physical", "translational / rotational / permutational invariance, sum of forces = 0",
       "<= 12 dimer records: translation and permutation |dE| <= 0.1 meV + 4x float floor and force error <= 1 meV/A "
       "+ floor; rotation / reflection |dE| <= 5 meV (PET learns rotations: an accuracy bound, not exactness); "
       "|sum_i F_i| <= 5 meV/A",
       "")
def invariance(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    from ..equivariance import check as eq_check
    from ..data import structure
    recs = {}
    for r in ctx.dimers.records:
        if len(r["species"]) <= 60 and r.get("distance") in (4.0, 8.0) and r.get("cls") not in recs:
            recs[r.get("cls")] = r
    structs = [structure(r) for r in list(recs.values())[:12]]
    if not structs:
        return outcome(NE, "no structures")
    res = eq_check(ctx.backend, structs, n_rot=3, seed=0)
    base = ctx.predict(structs)
    sumF = [float(np.linalg.norm(np.asarray(b["forces"]).sum(0))) for b in base if b.get("forces") is not None]
    fl = max(ctx.noise(b["energy"]) for b in base if b.get("energy") is not None)
    fails = []
    for name, c in res["checks"].items():
        exact = name.startswith(("translation", "permutation", "periodic"))
        tolE = 1e3 * (1e-4 + 4 * fl) if exact else 5.0
        c["tol_dE_meV"] = _r(tolE, 3)
        if c["max_abs_dE_meV"] > tolE:
            fails.append(f"{name} |dE| {c['max_abs_dE_meV']:.3g} meV")
        if exact and c.get("max_force_err_meV_A") is not None and c["max_force_err_meV_A"] > 1.0 + 1e3 * 4 * fl:
            fails.append(f"{name} force {c['max_force_err_meV_A']:.3g} meV/A")
    if sumF and max(sumF) > 5e-3:
        fails.append(f"|sum F| {1e3 * max(sumF):.3g} meV/A")
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "invariant within tolerance",
                   transforms=res["checks"], max_sum_F_meV_A=_r(1e3 * max(sumF), 4) if sumF else None)


def _flip_spin(s, dq):
    return s if dq % 2 == 0 else (2 if s == 1 else s - 1)


@check("charge_state", "physical", "charge-state energetics: vertical IP / EA against DFT, convexity "
       "E(N+1) + E(N-1) - 2 E(N) = IP - EA > 0",
       "on the IP/EA probe set (DFT Delta-SCF, same geometry): |mean signed IP error| <= 0.3 eV and IP MAE <= 0.5 eV "
       "(same for EA), IP - EA > 0 for every fragment; with --baseline, the paired IP MAE increase over the baseline "
       "must not exceed 0.1 eV with its 95 % bootstrap interval above 0 (fine-tune damage). Without probes: convexity "
       "only, on the dimer monomers",
       "IP damage from fine-tuning (IP30 -4.9 -> -41.5 kcal/mol, FTB / FTq vs PET-OMol-S)")
def charge_state(ctx: Ctx):
    if not ctx.conditioned:
        return outcome(NA, "backend passes no charge / spin")
    probes = ctx.ipea
    if not probes:
        if ctx.dimers is None:
            return outcome(NE, "no IP/EA probes and no dimer set")
        seen, probes = set(), []
        for t in _templates(ctx):
            for k in (0, 1):
                f = t.frag(k)
                if " ".join(f.species) + f"|{f.charge}" not in seen:
                    seen.add(" ".join(f.species) + f"|{f.charge}")
                    probes.append({"frag": f.key, "structure": f, "mult_states": {}, "IP_eV": None, "EA_eV": None})
    jobs = []
    for p in probes:
        s = p["structure"]
        ms = p.get("mult_states") or {}
        sm = ms.get("Nm1") or _flip_spin(s.spin, 1)
        sp = ms.get("Np1") or _flip_spin(s.spin, 1)
        jobs += [s, Structure(s.key + "|N-1", s.species, s.positions, s.charge + 1, int(sm)),
                 Structure(s.key + "|N+1", s.species, s.positions, s.charge - 1, int(sp))]
    E = ctx.energies(jobs).reshape(-1, 3)
    rows, eIP, eEA, nonconvex = [], [], [], []
    for p, (e0, em, ep) in zip(probes, E):
        ip, ea = em - e0, e0 - ep
        row = {"frag": p["frag"], "charge": int(p["structure"].charge), "IP_model": _r(ip, 3), "EA_model": _r(ea, 3),
               "IP_dft": _r(p.get("IP_eV"), 3), "EA_dft": _r(p.get("EA_eV"), 3)}
        if np.isfinite(ip) and p.get("IP_eV") is not None:
            row["IP_err"] = _r(ip - p["IP_eV"], 3)
            eIP.append(ip - p["IP_eV"])
        if np.isfinite(ea) and p.get("EA_eV") is not None:
            row["EA_err"] = _r(ea - p["EA_eV"], 3)
            eEA.append(ea - p["EA_eV"])
        if np.isfinite(ip) and np.isfinite(ea) and ip - ea <= 0:
            nonconvex.append(p["frag"])
        rows.append(row)
    n_conv = int(np.isfinite(E).all(1).sum())
    if not n_conv:
        return outcome(NE, "no probe with all three charge states in the model's range", probes=rows)
    fails, summ = [], {}
    for nm, e in (("IP", eIP), ("EA", eEA)):
        if e:
            e = np.asarray(e)
            summ[nm] = {"n": len(e), "mae_eV": _r(np.mean(np.abs(e)), 3), "mean_signed_eV": _r(np.mean(e), 3),
                        "median_abs_eV": _r(np.median(np.abs(e)), 3), "median_signed_eV": _r(np.median(e), 3)}
            if abs(np.mean(e)) > 0.3 or np.mean(np.abs(e)) > 0.5:
                fails.append(f"{nm} MAE {np.mean(np.abs(e)):.2f} / MSE {np.mean(e):+.2f} eV")
    by_q = collections.defaultdict(list)
    for r in rows:
        if r.get("IP_err") is not None:
            by_q[r["charge"]].append(r["IP_err"])
    summ["IP_err_by_charge"] = {q: {"n": len(v), "mean_signed_eV": _r(np.mean(v), 3)} for q, v in sorted(by_q.items())}
    if nonconvex:
        fails.append(f"IP - EA <= 0 for {len(nonconvex)}/{n_conv}")
    base = ((ctx.baseline or {}).get("checks", {}).get("charge_state") or {}).get("probes")
    if base and eIP:
        bm = {r["frag"]: r.get("IP_err") for r in base}
        d = np.array([abs(r["IP_err"]) - abs(bm[r["frag"]]) for r in rows
                      if r.get("IP_err") is not None and bm.get(r["frag"]) is not None])
        if len(d) >= 3:
            rng = np.random.default_rng(0)
            bs = [np.mean(rng.choice(d, len(d))) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            summ["paired_vs_baseline_IP_abs_err_eV"] = {"n": len(d), "mean": _r(np.mean(d), 3), "ci95": [_r(lo, 3), _r(hi, 3)],
                                                        "baseline": (ctx.baseline or {}).get("label")}
            if np.mean(d) > 0.1 and lo > 0:
                fails.append(f"IP damage vs {(ctx.baseline or {}).get('label')}: +{np.mean(d):.2f} eV [{lo:.2f}, {hi:.2f}]")
    if not eIP and not eEA:
        st = FAIL if nonconvex else PASS
        return outcome(st, ("; ".join(fails) if fails else f"convex for {n_conv} fragments") + " (no DFT IP/EA)",
                       stats=summ, probes=rows)
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "IP / EA within tolerance, convex",
                   stats=summ, probes=rows, n_nonconvex=len(nonconvex))


@check("charge_offset_by_Q", "physical", "no energy offset that grows with total charge Q on separated fragments "
       "beyond the single-component offset at the same Q (X9)",
       "separated-frame probes (multi-component, gap >= 12 A, DFT energies; dE = E_model + element refs - E_DFT): for "
       "every Q with >= 3 frames in both strata, |mean dE_sep(Q) - mean dE_single(Q)| <= 0.3 eV, and the OLS slope of "
       "dE_sep on Q is within +-0.2 eV/Q. Without probes: not_evaluable (the dimer 25 A table is reported)",
       "the separated-cation offset -1.48 eV (FTq, X9) and the charge-state offset by Q")
def charge_offset_by_Q(ctx: Ctx):
    info = {}
    if ctx.dimers is not None:
        C = canon(ctx)
        j = (C["d"] >= 25 - 1e-6) & ~np.array([is_ct(c) for c in C["cls"]])
        tab = collections.defaultdict(list)
        for k in np.where(j)[0]:
            if np.isfinite(C["model"][k]):
                tab[(int(C["Q"][k]), C["cls"][k])].append(1e3 * (C["model"][k] - C["ref"][k]))
        info["dimers_25A_signed_err_meV"] = {f"Q={q} {c}": {"n": len(v), "mean": _r(np.mean(v), 1)}
                                             for (q, c), v in sorted(tab.items())}
    if not ctx.sepframes:
        return outcome(NE, "no separated-frame probe set given (--sep-frames)", **info)
    S = [f["structure"] for f in ctx.sepframes]
    E = ctx.energies(S)
    rows = collections.defaultdict(lambda: {"sep": [], "single": []})
    for f, e in zip(ctx.sepframes, E):
        if np.isfinite(e):
            rows[int(f["structure"].charge)]["sep" if f["stratum"] == "separated" else "single"].append(
                e + f["ref_offset"] - f["E_dft"])
    tab, fails = {}, []
    qs, ys = [], []
    for q, v in sorted(rows.items()):
        a, b = np.array(v["sep"]), np.array(v["single"])
        tab[q] = {"n_sep": len(a), "sep_mean_eV": _r(a.mean(), 3) if len(a) else None,
                  "n_single": len(b), "single_mean_eV": _r(b.mean(), 3) if len(b) else None}
        qs += [q] * len(a)
        ys += list(a)
        if len(a) >= 3 and len(b) >= 3:
            dlt = a.mean() - b.mean()
            tab[q]["sep_minus_single_eV"] = _r(dlt, 3)
            if abs(dlt) > 0.3:
                fails.append(f"Q={q:+d} sep - single {dlt:+.2f} eV")
    slope = float(np.polyfit(qs, ys, 1)[0]) if len(set(qs)) >= 2 else None
    if slope is not None and abs(slope) > 0.2:
        fails.append(f"slope {slope:+.2f} eV/Q")
    if not any("sep_minus_single_eV" in v for v in tab.values()):
        return outcome(NE, "no Q with >= 3 frames in both strata", by_Q=tab, **info)
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "no Q-dependent separated offset",
                   by_Q=tab, slope_eV_per_Q=_r(slope, 3), **info)


@check("fragment_charges", "physical", "charge conservation, DFT fragment charges far apart, no charge on a "
       "neutral partner",
       "physical model charges (ASE 'charges'; mlip.cpp sqe_charges, the projected charges of the Coulomb term; "
       "mlipcpp-eval latent_charges are reported, not ruled) on canonical dimers at d >= 8 A and the 100 A scan points: max |sum q - Q| "
       "<= 1e-3 e; mean |q_frag - q_DFT(Lowdin)| <= 0.05 e at 8-25 A; neutral partner mean |q| <= 0.02 e at 8-25 A "
       "and <= 0.01 e at 100 A. CT-ambiguous pairs reported, not ruled",
       "charge leak onto neutral partners (Fukui summary: E 0.033 e raw)")
def fragment_charges(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    ds = ctx.dimers
    recs = [r for r in ds.records if (r.get("distance") or 0) >= 8 - 1e-6 and r.get("fragment_id") is not None
            and len(r.get("fragment_charges") or []) == 2]
    from ..data import structure
    P = ctx.predict([structure(r) for r in recs])
    key, lkey = ctx.opts.get("charge_key", "charges"), ctx.opts.get("latent_key")

    def stats(key):
        cons, frag_err, neut, neut100, ct = [], [], [], [], []
        for r, p in zip(recs, P):
            q = p.get(key)
            if q is None:
                continue
            q = np.asarray(q, float)
            fid = np.asarray(r["fragment_id"])
            cons.append(abs(q.sum() - r["charge"]))
            qa = q[fid == 0].sum()
            qd = (r.get("refs") or {}).get("qA_lowdin")
            if is_ct(r.get("cls")):
                ct.append({"id": r["id"], "qA_model": _r(qa, 3), "qA_dft": _r(qd, 3)})
                continue
            if qd is not None:
                frag_err.append(abs(qa - qd))
            fqs = r["fragment_charges"]
            if (fqs[0] == 0) != (fqs[1] == 0):
                neut.append(abs(q[fid == (0 if fqs[0] == 0 else 1)].sum()))
        for o in scans(ctx):
            t = o["t"]
            if (t.qa == 0) == (t.qb == 0):
                continue
            k = np.where(np.abs(o["d"] - 100) < 1e-6)[0]
            if len(k) and o["preds"][k[0]].get(key) is not None:
                q = np.asarray(o["preds"][k[0]][key], float)
                neut100.append(abs(q[t.ia if t.qa == 0 else t.ib].sum()))
        return cons, frag_err, neut, neut100, ct

    latent = None
    if lkey and any(p.get(lkey) is not None for p in P):
        c_, f_, n_, n100_, _ = stats(lkey)
        latent = {"max_conservation_err_e": _r(max(c_), 6), "fragment_err_mean_e": _r(np.mean(f_), 4) if f_ else None,
                  "neutral_partner_mean_e": _r(np.mean(n_), 4) if n_ else None,
                  "neutral_partner_100A_mean_e": _r(np.mean(n100_), 4) if n100_ else None,
                  "note": "mlipcpp-eval latent_charges (first channel; not normalised partial charges for these models), reported, not ruled"}
    if not any(p.get(key) is not None for p in P):
        why = "no physical (projected) charges" + (" (latent charges only, reported)" if latent else "")
        return outcome(NA, f"backend {getattr(ctx.backend, 'name', '?')} returns {why}", latent=latent)
    cons, frag_err, neut, neut100, ct = stats(key)
    fails = []
    mc = max(cons) if cons else np.nan
    if mc > 1e-3:
        fails.append(f"max |sum q - Q| {mc:.2g} e")
    if frag_err and np.mean(frag_err) > 0.05:
        fails.append(f"fragment error {np.mean(frag_err):.3f} e")
    if neut and np.mean(neut) > 0.02:
        fails.append(f"neutral partner {np.mean(neut):.3f} e at 8-25 A")
    if neut100 and np.mean(neut100) > 0.01:
        fails.append(f"neutral partner {np.mean(neut100):.3f} e at 100 A")
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "charges conserved, no neutral leak",
                   max_conservation_err_e=_r(mc, 6), fragment_err_mean_e=_r(np.mean(frag_err), 4) if frag_err else None,
                   neutral_partner_mean_e=_r(np.mean(neut), 4) if neut else None,
                   neutral_partner_frac_gt_0p05=_r(np.mean(np.array(neut) > 0.05), 3) if neut else None,
                   neutral_partner_100A_mean_e=_r(np.mean(neut100), 4) if neut100 else None,
                   ct_ambiguous=ct[:40], n_ct_ambiguous=len(ct), charges=key, latent=latent)


@check("periodic_open", "physical", "periodic <-> open consistency: a neutral cluster in a growing box converges to "
       "the open-boundary energy",
       "total-charge-0 dimers at d = 8 A (<= 6) in cubic boxes L = 30, 50, 80 A: |E(L = 80) - E_open| <= 5 meV + "
       "4x float floor and |dE(L)| non-increasing (within the floor). A model / backend that rejects cells is "
       "not_applicable",
       "")
def periodic_open(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    T = [t for t in _templates(ctx) if t.qa + t.qb == 0][:6]
    if not T:
        return outcome(NE, "no neutral dimer")
    Ls = ctx.opts.get("box_L", (30.0, 50.0, 80.0))
    rows, fails, errs = [], [], []
    for t in T:
        s = t.ab(8.0, "box")
        P = ctx.predict([s] + [boxed(s, L) for L in Ls])
        if P[0].get("error"):
            continue
        if any(p.get("error") for p in P[1:]):
            errs.append(next(p["error"] for p in P[1:] if p.get("error")))
            continue
        e0 = P[0]["energy"]
        dE = [p["energy"] - e0 for p in P[1:]]
        tol = 0.005 + 4 * ctx.noise(e0, P[-1]["energy"])
        mono = all(abs(dE[i + 1]) <= abs(dE[i]) + tol for i in range(len(dE) - 1))
        rows.append({"scan": t.scan_id, "dE_meV": {f"{L:g}": _r(1e3 * x, 3) for L, x in zip(Ls, dE)},
                     "tol_meV": _r(1e3 * tol, 3), "monotone": mono})
        if abs(dE[-1]) > tol or not mono:
            fails.append(f"{t.scan_id} dE(L={Ls[-1]:g}) {1e3 * dE[-1]:+.2f} meV")
    if not rows:
        return outcome(NA if errs else NE, f"cells rejected: {errs[0][:160]}" if errs else "nothing evaluated")
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "box energies converge to open", boxes=rows)


# ============================================================================================== model assumptions
@check("locality", "model", "locality: no spurious interaction beyond the model's cutoff between neutral fragments; "
       "exact locality for a model that declares itself local",
       "dE = E_int(d) - E_int(100 A) (the dependence on the partner's position; a constant separated offset is "
       "size_consistency's business). Neutral-neutral fine-scan points at d >= r_c + 1 A: |dE| <= max(1 meV + 4x "
       "floor, 2 |E_int,DFT| at the nearest canonical d below); if the model declares no long-range term, every pair at "
       "d >= 2 r_c must have |dE| <= 1 meV + 4x floor",
       "spurious long range from leaked charges or a non-neutral latent charge on neutral molecules")
def locality(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    rc = ctx.cutoff or 8.0
    C = canon(ctx)
    fails, rows = [], []
    n_eval = 0  # every point actually compared: without this, a far point that fails to evaluate makes every
    #             difference NaN, rows stays empty, and "no violations" is indistinguishable from "no evidence"
    for o in scans(ctx):
        t = o["t"]
        neutral = t.qa == 0 and t.qb == 0
        kf = int(np.argmax(o["d"]))
        for k in np.where((o["d"] >= rc + 1 - 1e-6) & (np.arange(len(o["d"])) != kf))[0]:
            Ei = o["Eint"][k] - o["Eint"][kf]  # dependence on the partner's position, not the separated offset
            if not np.isfinite(Ei):
                continue
            n_eval += 1
            fl = 1e-3 + 4 * o["noise"][k]
            if neutral:
                j = np.where((C["scan"] == t.scan_id) & (C["d"] <= o["d"][k] + 1e-6))[0]
                ref = abs(C["ref"][j[np.argmax(C["d"][j])]]) if len(j) else 0.0
                bound = max(fl, 2 * ref)
                if abs(Ei) > bound:
                    rows.append({"scan": t.scan_id, "d": _r(o["d"][k], 2), "E_int_meV": _r(1e3 * Ei, 3),
                                 "bound_meV": _r(1e3 * bound, 3)})
            if ctx.declared_local and o["d"][k] >= 2 * rc and abs(Ei) > fl:
                rows.append({"scan": t.scan_id, "d": _r(o["d"][k], 2), "E_int_meV": _r(1e3 * Ei, 3),
                             "declared_local": True})
    nscan = len({r["scan"] for r in rows})
    if rows:
        fails.append(f"{len(rows)} beyond-cutoff points in {nscan} scans exceed the bound")
    if not n_eval:
        return outcome(NE, f"no beyond-cutoff point could be compared (r_c = {rc:g} A): the far reference of "
                            "every scan failed to evaluate", cutoff_A=rc, n_evaluated=0)
    return outcome(FAIL if fails else PASS,
                   "; ".join(fails) if fails else
                   f"no spurious interaction beyond r_c = {rc:g} A in {n_eval} points",
                   cutoff_A=rc, cutoff_source=ctx.opts.get("cutoff_source"), declared_local=ctx.declared_local,
                   n_evaluated=n_eval, violations=rows[:60])


@check("charge_conditioning", "model", "the model reacts to total charge and spin, and its answer does not depend on "
       "call order (no caching across charge states)",
       "single-fragment probes at (Q, S), (Q+1, S'), (Q-1, S'), (Q, S+2): |E(Q+-1) - E(Q)| > 1 meV for >= 90 % of "
       "probes; the same states evaluated in forward and reversed order through the backend's own state must agree to "
       "1e-5 eV + float floor (else: cached / order dependent)",
       "the upet 0.3.1 ASE wrapper returning the previous charge state's energy (PARITY.md)")
def charge_conditioning(ctx: Ctx):
    if not ctx.conditioned:
        return outcome(NA, "backend passes no charge / spin")
    probes, seen = [], set()
    for t in _templates(ctx) if ctx.dimers is not None else []:
        for k in (0, 1):
            f = t.frag(k)
            key = " ".join(f.species) + f"|{f.charge}"
            if key not in seen:
                seen.add(key)
                probes.append(f)
    probes += [p["structure"] for p in (ctx.ipea or [])[:10]]
    if not probes:
        return outcome(NE, "no probes")
    states = []
    for s in probes[:24]:
        states += [s, Structure(s.key + "|+1", s.species, s.positions, s.charge + 1, _flip_spin(s.spin, 1)),
                   Structure(s.key + "|-1", s.species, s.positions, s.charge - 1, _flip_spin(s.spin, 1)),
                   Structure(s.key + "|S+2", s.species, s.positions, s.charge, s.spin + 2)]
    fw = ctx.backend.predict(states)
    rv = ctx.backend.predict(states[::-1])[::-1]
    ctx.n_calls += 2 * len(states)
    e = lambda p: np.nan if p.get("error") or p.get("energy") is None else p["energy"]
    Ef, Er = np.array([e(p) for p in fw]), np.array([e(p) for p in rv])
    ok = np.isfinite(Ef) & np.isfinite(Er)
    tol = 1e-5 + np.array([ctx.noise(x) if np.isfinite(x) else 0 for x in Ef])
    order = np.abs(Ef - Er)
    n_order = int((ok & (order > tol)).sum())
    Ef4 = Ef.reshape(-1, 4)
    resp_q, resp_s = [], []
    for row in Ef4:
        if np.isfinite(row[0]) and (np.isfinite(row[1]) or np.isfinite(row[2])):
            resp_q.append(np.nanmax(np.abs(row[1:3] - row[0])) > 1e-3)
        if np.isfinite(row[0]) and np.isfinite(row[3]):
            resp_s.append(abs(row[3] - row[0]) > 1e-3)
    ident = int(sum(np.sum(np.abs(np.diff(row[np.isfinite(row)])) < 1e-9) for row in Ef4))
    fails = []
    if n_order:
        fails.append(f"{n_order}/{int(ok.sum())} energies depend on call order (max {order[ok].max():.3g} eV): cached")
    if resp_q and np.mean(resp_q) < 0.9:
        fails.append(f"no response to Q for {len(resp_q) - sum(resp_q)}/{len(resp_q)} probes")
    if not resp_q:
        return outcome(NE, "no probe with a neighbouring charge state in range")
    return outcome(FAIL if fails else PASS, "; ".join(fails) if fails else "reacts to Q and spin, order independent",
                   n_probes=len(Ef4), q_response_fraction=_r(np.mean(resp_q), 3),
                   spin_response_fraction=_r(np.mean(resp_s), 3) if resp_s else None,
                   n_identical_consecutive_states=ident, max_order_dE_eV=_r(order[ok].max() if ok.any() else None, 9))


@check("determinism", "model", "repeated calls give the same energies and forces",
       "12 structures predicted 3x (fresh calls, alternating order): max |dE| <= max(1e-6 eV, float floor), "
       "max |dF| <= 1e-4 eV/A",
       "")
def determinism(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    T = _templates(ctx)[:6]
    S = [t.ab(d, "det") for t in T for d in (4.0, 10.0)]
    runs = [ctx.backend.predict(S), ctx.backend.predict(S[::-1])[::-1], ctx.backend.predict(S)]
    ctx.n_calls += 3 * len(S)
    dE, dF, tolE = 0.0, 0.0, 1e-6
    for k in range(len(S)):
        es = [r[k].get("energy") for r in runs]
        if any(x is None for x in es):
            continue
        dE = max(dE, max(es) - min(es))
        tolE = max(tolE, ctx.noise(es[0]))
        fs = [r[k].get("forces") for r in runs]
        if all(f is not None for f in fs):
            dF = max(dF, max(float(np.abs(np.asarray(f) - np.asarray(fs[0])).max()) for f in fs))
    st = FAIL if (dE > tolE or dF > 1e-4) else PASS
    return outcome(st, f"max |dE| {dE:.3g} eV, max |dF| {dF:.3g} eV/A", max_dE_eV=dE, max_dF_eV_A=dF, tol_E_eV=tolE)


@check("precision", "model", "the result does not depend on the arithmetic (fp32 vs fp64, CPU vs GPU, TF32)",
       "against --ref-model on 24 scan points: max |dE_int| <= 1 meV and max |dE| <= max(1 meV, 10x float floor)",
       "TF32 / NVIDIA_TF32_OVERRIDE geometry-precision loss")
def precision(ctx: Ctx):
    if ctx.ref_backend is None:
        return outcome(NA, "no reference backend (--ref-model)")
    if (e := _need_dimers(ctx)):
        return e
    T = _templates(ctx)[:6]
    S = [t.ab(d, "prec") for t in T for d in (4.0, 8.0, 12.0, 25.0)] + [t.frag(k) for t in T for k in (0, 1)]
    A = ctx.energies(S)
    B = np.array([np.nan if p.get("error") else p["energy"] for p in ctx.ref_backend.predict(S)])
    n = 4 * len(T)
    eint = lambda E: np.array([E[i] - E[n + 2 * (i // 4)] - E[n + 2 * (i // 4) + 1] for i in range(n)])
    dI = np.abs(eint(A) - eint(B))
    dA = np.abs(A - B)
    ok = np.isfinite(dA)
    if not ok.any():
        return outcome(NE, "no common finite energies")
    tolA = max(1e-3, 10 * max(ctx.noise(x) for x in A[ok]))
    st = FAIL if (np.nanmax(dI) > 1e-3 or np.nanmax(dA) > tolA) else PASS
    return outcome(st, f"max |dE_int| {1e3 * np.nanmax(dI):.3g} meV, max |dE| {1e3 * np.nanmax(dA):.3g} meV",
                   max_dEint_meV=_r(1e3 * np.nanmax(dI), 4), max_dE_meV=_r(1e3 * np.nanmax(dA), 4),
                   reference=ctx.opts.get("ref_model"))


# ============================================================================================== data assumptions
def _mae_by_class(C, mask):
    out = {}
    for cls in sorted(set(C["cls"][mask].tolist())):
        j = mask & (C["cls"] == cls) & np.isfinite(C["model"])
        if j.any():
            out[cls] = {"n": int(j.sum()), "mae_meV": _r(1e3 * np.mean(np.abs(C["model"][j] - C["ref"][j])), 2)}
    return out


@check("contamination", "data", "headline numbers on records no training set has seen",
       "with seen_by flags: E_int MAE (d >= 8 A) by class on all records and on the unseen ones; pass if no scored "
       "record is seen, warn otherwise (the unseen table is the headline)",
       "train_aug / store overlaps inflating the dimer numbers")
def contamination(ctx: Ctx):
    if ctx.seen is None:
        return outcome(NA, "no seen_by flags (--seen)")
    if (e := _need_dimers(ctx)):
        return e
    C = canon(ctx)
    far = C["d"] >= 8 - 1e-6
    seen = np.array([bool(ctx.seen.get(i)) for i in C["id"]])
    allm, uns = _mae_by_class(C, far), _mae_by_class(C, far & ~seen)
    n_seen = int((far & seen).sum())
    src = collections.Counter(s.split(":")[0] for i in C["id"][far] for s in (ctx.seen.get(i) or []))
    return outcome(WARN if n_seen else PASS, f"{n_seen}/{int(far.sum())} far points seen ({dict(src)}); "
                   "headline = unseen" if n_seen else "no scored record seen",
                   all=allm, unseen_headline=uns, n_seen=n_seen, sources=dict(src))


@check("level_of_theory", "data", "labels and model share one level of theory",
       "model training level equals every scored record's level; unknown model level -> not_evaluable",
       "SO3LR PBE0+MBD vs wB97M-V labels")
def level_of_theory(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    lv = collections.Counter(record_level(r) for r in ctx.dimers.records)
    if not ctx.model_level or ctx.model_level == "unknown":
        return outcome(NE, f"model level unknown; labels {dict(lv)}")
    mism = {k: v for k, v in lv.items() if k != ctx.model_level}
    if not mism:
        return outcome(PASS, f"model and labels at {ctx.model_level}")
    return outcome(FAIL, f"labels at {dict(mism)} vs model {ctx.model_level}"
                   + (" (allowed, recorded)" if ctx.allow_level_mismatch else ""), labels=dict(lv))


@check("ct_unbound_labelled", "data", "CT-ambiguous ion pairs and unbound anions are kept and reported in their own "
       "class, never pooled and never dropped",
       "every CT-ambiguous dimer record and every unbound-anion IP/EA probe is predicted and reported in a separate "
       "row; fail if any is dropped (prediction error) or pooled",
       "CT-ambiguous points dropped or pooled with the ion pairs")
def ct_unbound_labelled(ctx: Ctx):
    rows = {}
    if ctx.dimers is not None:
        C = canon(ctx)
        ct = np.array([is_ct(c) for c in C["cls"]])
        if ct.any():
            ok = ct & np.isfinite(C["model"])
            rows["ct_ambiguous"] = {"n": int(ct.sum()), "n_predicted": int(ok.sum()),
                                    "by_distance_mae_meV": {f"{d:g}": _r(1e3 * np.mean(np.abs(C["model"][ok & (np.abs(
                                        C["d"] - d) < 1e-2)] - C["ref"][ok & (np.abs(C["d"] - d) < 1e-2)])), 1)
                                        for d in sorted(set(np.round(C["d"][ok], 2).tolist()))}}
    unb = [p for p in (ctx.ipea or []) if any("basis_bound" in f or "unbound" in f for f in p.get("flags", []))]
    if unb:
        E = ctx.energies([p["structure"] for p in unb])
        rows["unbound_anion"] = {"n": len(unb), "n_predicted": int(np.isfinite(E).sum())}
    if not rows:
        return outcome(NE, "no CT-ambiguous or unbound-anion records in the loaded data")
    dropped = sum(v["n"] - v["n_predicted"] for v in rows.values())
    return outcome(FAIL if dropped else PASS, f"{dropped} dropped" if dropped else
                   ", ".join(f"{k} {v['n']} kept, own row" for k, v in rows.items()), classes=rows)


@check("offset_free_degeneracy", "data", "a metric must be able to tell the model from a flat line",
       "offset-free 8 A error |[E(8) - E(d_max)]_model - [E(8) - E(d_max)]_DFT|: per ion class with DFT span >= 1 meV, "
       "if the median model span is < 5 % of the DFT span the metric equals the DFT span for any flat model; fail "
       "(degenerate) and report the absolute |E_int| error at 8 A instead",
       "the degenerate offset-free 8 A metric for flat models (FTB seed-identical, metric_check/NOTE.md)")
def offset_free_degeneracy(ctx: Ctx):
    if (e := _need_dimers(ctx)):
        return e
    C = canon(ctx)
    rows, deg = {}, []
    for cls in sorted(set(C["cls"].tolist())):
        if not is_ion_class(cls):
            continue
        ms, ds_, ab = [], [], []
        for sc in set(C["scan"][C["cls"] == cls].tolist()):
            i = np.where(C["scan"] == sc)[0]
            l = i[np.argmax(C["d"][i])]
            k8 = i[np.abs(C["d"][i] - 8) < 1e-2]
            for k in k8:
                if k != l and np.isfinite(C["model"][k]) and np.isfinite(C["model"][l]):
                    ms.append(abs(C["model"][k] - C["model"][l]))
                    ds_.append(abs(C["ref"][k] - C["ref"][l]))
                    ab.append(abs(C["model"][k] - C["ref"][k]))
        if not ms or np.median(ds_) < 1e-3:
            continue
        ratio = float(np.median(ms) / np.median(ds_))
        rows[cls] = {"n": len(ms), "median_model_span_meV": _r(1e3 * np.median(ms), 2),
                     "median_dft_span_meV": _r(1e3 * np.median(ds_), 2), "ratio": _r(ratio, 3),
                     "abs_err_8A_meV": _r(1e3 * np.mean(ab), 2)}
        if ratio < 0.05:
            deg.append(cls)
    if not rows:
        return outcome(NE, "no ion class with an 8 A point and a DFT span >= 1 meV")
    return outcome(FAIL if deg else PASS, (f"degenerate for {', '.join(deg)}: report abs |E_int| error at 8 A")
                   if deg else "offset-free 8 A metric informative", by_class=rows)


def _cov_class(qa, qb):
    a, b = abs(qa) >= 0.5, abs(qb) >= 0.5
    if not a and not b:
        return "neutral"
    if a != b:
        return "ion_neutral"
    return "same_sign" if qa * qb > 0 else "ion_pair"


COV_BINS = [(6, 7), (7, 7.5), (7.5, 8), (8, 9), (9, 10), (10, 12), (12, 1e9)]
CLS_MAP = {"ion_pair": "ion_pair", "ion_pair_ct_ambiguous": "ion_pair", "same_sign": "same_sign",
           "ion_polar": "ion_neutral", "ion_nonpolar": "ion_neutral", "neutral_polar_control": "neutral",
           "neutral_nonpolar_control": "neutral"}


@check("distribution_shift", "data", "test points lie inside the training distribution (gap distance x charge class)",
       "with training metadata (rows with a fragment gap g or d and fragment charges qa/qb or a class): every "
       "(class, gap bin >= 6 A) that holds canonical dimer points must have >= 10 training rows; warn and list the "
       "uncovered bins otherwise (training coverage below 6 A is not in the metadata)",
       "ion pairs at 7-12 A: 5 training rows (seam_band coverage)")
def distribution_shift(ctx: Ctx):
    if not ctx.train_coverage:
        return outcome(NA, "no training metadata (--train-coverage)")
    if (e := _need_dimers(ctx)):
        return e
    tr = collections.Counter()
    for r in ctx.train_coverage:
        g = r.get("g", r.get("d"))
        if g is None:
            continue
        c = r.get("cls")
        c = {"ion_neutral": "ion_neutral", "ion_pair": "ion_pair", "same_sign": "same_sign", "like_charge": "same_sign",
             "neutral": "neutral"}.get(c) if c else _cov_class(r.get("qa", 0), r.get("qb", 0))
        b = next((k for k, (lo, hi) in enumerate(COV_BINS) if lo <= g < hi), None)
        if b is not None and c:
            tr[(c, b)] += 1
    C = canon(ctx)
    te = collections.Counter()
    for cls, d in zip(C["cls"], C["d"]):
        b = next((k for k, (lo, hi) in enumerate(COV_BINS) if lo <= d < hi), None)
        if b is not None and cls in CLS_MAP:
            te[(CLS_MAP[cls], b)] += 1
    table, gaps = [], []
    for (c, b), n in sorted(te.items()):
        lo, hi = COV_BINS[b]
        row = {"cls": c, "bin_A": f"{lo:g}-{hi:g}" if hi < 1e8 else f">={lo:g}", "n_test": n, "n_train": tr[(c, b)]}
        table.append(row)
        if tr[(c, b)] < 10:
            gaps.append(f"{c} {row['bin_A']} ({tr[(c, b)]} train / {n} test)")
    return outcome(WARN if gaps else PASS, ("uncovered: " + "; ".join(gaps)) if gaps else "every test bin covered",
                   table=table, n_train_rows=len(ctx.train_coverage))


ORDER = ["far_field_exponent", "virial_tail", "eps_far", "seam", "size_consistency", "energy_force",
         "conservativeness", "nve_short", "invariance",
         "charge_state", "charge_offset_by_Q", "fragment_charges", "periodic_open", "locality",
         "charge_conditioning", "determinism", "precision", "contamination", "level_of_theory",
         "ct_unbound_labelled", "offset_free_degeneracy", "distribution_shift"]
