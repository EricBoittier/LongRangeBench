"""Orientation-sampled far-field probes, scored against the multipole oracle, and an adversarial search over them.

A fine scan moves fragment B along one axis in one orientation, and the checks that read it score medians. A defect
that lives in a minority of orientations -- a link that switches on only when two particular atoms face each other,
a charge assignment that depends on how a molecular ion is turned -- is invisible to both. These probes sample the
orientation instead, and need no new DFT to do it: ``oracle:multipole`` (``lrbench.oracle``) gives the point-multipole
energy of two rigid fragments from their own DFT moments for *any* orientation, so it can referee every sample.

Geometry: both fragments rigid (the canonical monomers of the dimer set), each turned by its own uniformly random
rotation about its centre of mass, A's COM at the origin and B's at r z. r is the centre-of-mass separation (the
oracle's natural coordinate and that of a constrained-distance MD window); the closest contact d_min is recorded per
geometry. The quantity scored is the axial force on B, F = sum_{i in B} F_i . z (positive = repulsive), against the
oracle's -dE/dr at fixed orientation, and the orientation-dependent energy error (offset removed per pair and r).

Oracle trust. The sign of the oracle force is only taken as a reference where it cannot be flipped by what the
oracle leaves out or approximates: |F_O| >= max(2 meV/A, 3 sigma_O), sigma_O = |F_O2+ind - F_O1+ind| + |F_O2+ind - F_O2|
(the two terms it added last: the quadrupole and the classical induction). A model force below max(1e-4 eV/A,
0.05 |F_O|) at such a point is a missing tail, not a wrong sign. Bands:

- ``far``: r >= 10 A. The oracle is the reference; the pass/fail rule applies here.
- ``mid``: 7.5 <= r < 10 A with d_min >= 4 A. Reported with the caveat that the oracle has not been validated
  against DFT there (charge penetration, exchange, charge transfer); a violation is a ``warn``, and the geometries a
  DFT validation should start from are listed (``dft_stage``).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .. import multipole as MP
from ..data import Structure, combine_spins
from .core import FAIL, NE, PASS, WARN, Ctx, check, outcome

KBT_300 = 0.025852  # eV
R_FAR = (10.0, 12.0, 15.0)
R_MID = (7.5, 8.0, 8.5, 9.0, 9.5)
BANDS = {"far": (10.0, np.inf), "mid": (7.5, 10.0)}
MID_DMIN = 4.0
F_ABS = 2e-3  # eV/A: below this the oracle force is not used as a sign reference
F_MODEL = 1e-4  # eV/A: model force floor


def _rot(n: int, seed: int):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=seed)


@dataclass
class Pair:
    """One dimer pair as two rigid fragments about their COMs, with their DFT moments in the same frame."""
    name: str
    cls: str
    species: List[str]
    xa: np.ndarray  # (nA, 3) about A's COM
    xb: np.ndarray
    qa: int
    qb: int
    spin: int
    ma: dict  # moments of A in the frame of xa
    mb: dict

    def positions(self, ra, rb, r: float) -> np.ndarray:
        return np.vstack([ra.apply(self.xa), rb.apply(self.xb) + np.array([0.0, 0.0, r])])

    def structure(self, ra, rb, r: float, tag: str) -> Structure:
        return Structure(f"{self.name}|{tag}|r{r:.4f}", list(self.species), self.positions(ra, rb, r),
                         self.qa + self.qb, self.spin)

    def dmin(self, ra, rb, r: float) -> float:
        a, b = ra.apply(self.xa), rb.apply(self.xb) + np.array([0.0, 0.0, r])
        return float(np.min(np.linalg.norm(a[:, None] - b[None], axis=-1)))

    def oracle(self, ra, rb, r: float, h: float = 1e-3) -> dict:
        """Oracle energies and axial forces on B (eV, eV/A) for O1+ind, O2 and O2+ind at this orientation."""
        A = MP.rotate(self.ma, ra.as_matrix())
        B = MP.rotate(self.mb, rb.as_matrix())
        e0, ep, em = (MP.energies(A, B, np.array([0.0, 0.0, x])) for x in (r, r + h, r - h))
        out = {}
        for k in ("O1+ind", "O2", "O2+ind"):
            out["E_" + k] = e0[k]
            out["F_" + k] = -(ep[k] - em[k]) / (2 * h)
        return out


def pairs(ctx: Ctx) -> List[Pair]:
    """Every dimer template whose two fragments have moments, as rigid fragments centred on their COMs."""
    if "_orient_pairs" in ctx.opts:
        return ctx.opts["_orient_pairs"]
    from .checks import _templates
    out = []
    for t in _templates(ctx):
        ex = t.rec.get("extra") or {}
        names = ex.get("A"), ex.get("B")
        if not ctx.moments or any(n not in ctx.moments for n in names):
            continue
        sp = list(t.rec["species"])
        x = np.asarray(t.rec["positions"], float)
        frag = []
        for idx, n, q in ((t.ia, names[0], t.qa), (t.ib, names[1], t.qb)):
            F = ctx.moments[n]
            if len(F["R"]) != len(idx):
                break
            xf = x[idx]
            xf = xf - MP.com([sp[i] for i in idx], xf)
            M = MP.rotate(dict(q=float(q), mu=F["mu"], theta=F["theta"], alpha=F.get("alpha")), MP.kabsch(F["R"], xf))
            frag.append((xf, M))
        if len(frag) != 2:
            continue
        out.append(Pair(t.pair, t.cls, [sp[i] for i in t.ia] + [sp[i] for i in t.ib], frag[0][0], frag[1][0],
                        t.qa, t.qb, combine_spins([t.sa, t.sb]), frag[0][1], frag[1][1]))
    ctx.opts["_orient_pairs"] = out
    return out


def _axial(p: dict, nA: int) -> float:
    if p.get("error") or p.get("forces") is None:
        return np.nan
    return float(np.asarray(p["forces"], float)[nA:, 2].sum())


def _energy(p: dict) -> float:
    return np.nan if p.get("error") or p.get("energy") is None else float(p["energy"])


def _band(r: float, dmin: float) -> Optional[str]:
    if r >= BANDS["far"][0] - 1e-9:
        return "far"
    if r >= BANDS["mid"][0] - 1e-9 and dmin >= MID_DMIN:
        return "mid"
    return None


def classify(F_m: float, o: dict) -> dict:
    """Sign trust, missing tail and wrong sign of one model force against the oracle."""
    F_o = o["F_O2+ind"]
    sig = abs(o["F_O2+ind"] - o["F_O1+ind"]) + abs(o["F_O2+ind"] - o["F_O2"])
    robust = bool(abs(F_o) >= max(F_ABS, 3 * sig))
    missing = robust and abs(F_m) < max(F_MODEL, 0.05 * abs(F_o))
    wrong = robust and not missing and np.sign(F_m) != np.sign(F_o)
    return {"robust": robust, "missing": bool(missing), "wrong_sign": bool(wrong), "sigma_O": sig}


def sample(ctx: Ctx) -> List[dict]:
    """Model and oracle on n uniformly random orientations per pair at every r of the far and mid grids."""
    if "_orient_rows" in ctx.opts:
        return ctx.opts["_orient_rows"]
    n = int(ctx.opts.get("orient_n", 128))
    seed = int(ctx.opts.get("orient_seed", 2026))
    grid = list(ctx.opts.get("orient_r", R_MID + R_FAR))
    rows, structs = [], []
    for ip, P in enumerate(pairs(ctx)):
        RA, RB = _rot(n, seed + 2 * ip), _rot(n, seed + 2 * ip + 1)
        for k in range(n):
            for r in grid:
                dm = P.dmin(RA[k], RB[k], r)
                band = _band(r, dm)
                if band is None:
                    continue
                rows.append(dict(pair=P.name, cls=P.cls, k=k, r=float(r), d_min=dm, band=band, nA=len(P.xa),
                                 rot_a=RA[k].as_rotvec(), rot_b=RB[k].as_rotvec(), **P.oracle(RA[k], RB[k], r)))
                structs.append(P.structure(RA[k], RB[k], r, f"o{k}"))
    for row, p in zip(rows, ctx.predict(structs)):
        row["F_m"] = _axial(p, row["nA"])
        row["E_m"] = _energy(p)
        row["error"] = p.get("error")
        row.update(classify(row["F_m"], row) if np.isfinite(row["F_m"]) else
                   {"robust": False, "missing": False, "wrong_sign": False, "sigma_O": np.nan})
    # per (pair, r): force scale (rms of the oracle over orientations) and the offset-free energy error
    by = collections.defaultdict(list)
    for row in rows:
        by[(row["pair"], row["r"])].append(row)
    for grp in by.values():
        s = float(np.sqrt(np.mean([g["F_O2+ind"] ** 2 for g in grp])))
        de = np.array([g["E_m"] - g["E_O2+ind"] for g in grp])
        off = np.nanmedian(de) if np.isfinite(de).any() else np.nan
        for g, x in zip(grp, de):
            g["F_scale"] = s
            g["eps_F"] = (g["F_m"] - g["F_O2+ind"]) / s if s > 0 else np.nan
            g["dE_orient"] = x - off
    ctx.opts["_orient_rows"] = rows
    if ctx.opts.get("orient_rows"):
        write_rows(ctx.opts["orient_rows"], rows)
    return rows


ROW_KEYS = ("pair", "cls", "k", "r", "d_min", "band", "F_m", "F_O2+ind", "F_O2", "F_O1+ind", "E_m", "E_O2+ind",
            "sigma_O", "robust", "missing", "wrong_sign", "F_scale", "eps_F", "dE_orient", "rot_a", "rot_b")


def write_rows(path: str, rows: List[dict]):
    """Every sampled point as JSON lines (rotation vectors included, so each geometry can be rebuilt)."""
    import json
    conv = lambda v: (v.tolist() if hasattr(v, "tolist") else
                      (None if isinstance(v, float) and not np.isfinite(v) else v))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: conv(r.get(k)) for k in ROW_KEYS}) + "\n")


def _q(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)), 4)


def summarise(rows: List[dict]) -> Dict[str, dict]:
    """Per class: n, wrong-sign and missing fractions over oracle-trusted points; median / q95 / max |eps_F| and
    |dE_orient| (meV), and the worst geometry."""
    out = {}
    by = collections.defaultdict(list)
    for r in rows:
        if np.isfinite(r["F_m"]):
            by[r["cls"]].append(r)
    for cls, R in sorted(by.items()):
        rob = [r for r in R if r["robust"]]
        eps = np.abs([r["eps_F"] for r in R if np.isfinite(r["eps_F"])])
        de = 1e3 * np.abs([r["dE_orient"] for r in R if np.isfinite(r["dE_orient"])])
        worst = max(R, key=lambda r: abs(r["eps_F"]) if np.isfinite(r["eps_F"]) else -1)
        out[cls] = {"n": len(R), "n_sign_trusted": len(rob),
                    "wrong_sign_frac": round(sum(r["wrong_sign"] for r in rob) / len(rob), 4) if rob else None,
                    "missing_frac": round(sum(r["missing"] for r in rob) / len(rob), 4) if rob else None,
                    "eps_F_median": _q(eps, 50), "eps_F_q95": _q(eps, 95), "eps_F_max": _q(eps, 100),
                    "dE_orient_meV_median": _q(de, 50), "dE_orient_meV_q95": _q(de, 95),
                    "dE_orient_meV_max": _q(de, 100),
                    "worst": _row_out(worst)}
    return out


def _row_out(r: dict) -> dict:
    return {"pair": r["pair"], "r": round(r["r"], 4), "d_min": round(r["d_min"], 3), "k": r.get("k"),
            "F_model": round(r["F_m"], 5), "F_oracle": round(r["F_O2+ind"], 5), "sigma_O": round(r["sigma_O"], 5),
            "eps_F": None if not np.isfinite(r["eps_F"]) else round(r["eps_F"], 3),
            "rot_a": [round(float(v), 6) for v in r["rot_a"]], "rot_b": [round(float(v), 6) for v in r["rot_b"]]}


def is_monopole(cls: str) -> bool:
    from .checks import is_monopole_pair
    return is_monopole_pair(cls)


def violations(S: Dict[str, dict]) -> List[str]:
    """The rule, per class (CT-ambiguous classes are reported, never ruled): wrong-sign fraction <= 0.02 over >= 20
    oracle-trusted points; for monopole pairs also q95 |eps_F| <= 0.5 and q95 |dE_orient| <= k_BT (300 K)."""
    from .checks import is_ct
    bad = []
    for cls, v in S.items():
        if is_ct(cls):
            continue
        if v["n_sign_trusted"] >= 20 and v["wrong_sign_frac"] > 0.02:
            bad.append(f"{cls} wrong sign {100 * v['wrong_sign_frac']:.0f} %")
        if is_monopole(cls) and v["n"] >= 20:
            if v["eps_F_q95"] is not None and v["eps_F_q95"] > 0.5:
                bad.append(f"{cls} q95 |eps_F| {v['eps_F_q95']:.2f}")
            if v["dE_orient_meV_q95"] is not None and v["dE_orient_meV_q95"] > 1e3 * KBT_300:
                bad.append(f"{cls} q95 |dE_orient| {v['dE_orient_meV_q95']:.0f} meV")
    return bad


def dft_stage(rows: List[dict], per_pair: int = 4) -> List[dict]:
    """Mid-band geometries a DFT validation of the oracle should start from: per pair, the largest model-oracle
    disagreements and the largest oracle truncation spreads (rigid monomers, so 1 dimer + finite-difference partners
    per geometry; monomers once per pair)."""
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        if r["band"] == "mid" and np.isfinite(r["F_m"]):
            by[r["pair"]].append(r)
    for pair, R in sorted(by.items()):
        a = sorted(R, key=lambda r: -abs(r["eps_F"]) if np.isfinite(r["eps_F"]) else 0)[:per_pair // 2]
        b = sorted(R, key=lambda r: -r["sigma_O"] if np.isfinite(r["sigma_O"]) else 0)[:per_pair - len(a)]
        seen = set()
        for r in a + b:
            if (r["k"], r["r"]) not in seen:
                seen.add((r["k"], r["r"]))
                out.append(dict(_row_out(r), why="model-oracle disagreement" if r in a else "oracle spread"))
    return out


@check("orientation_far", "physical", "the far field holds in every orientation, not only the one a scan happens to "
       "use: the force between two fragments has the sign and size of the point-multipole oracle (the fragments' own "
       "DFT moments, O2 + induction) at any rigid orientation",
       "per dimer pair with moments (--moments frags.json), 128 uniformly random rigid orientations of both fragments "
       "at COM separations r = 7.5-9.5 A (mid, d_min >= 4 A) and 10, 12, 15 A (far). Axial force on B against the "
       "oracle's -dE/dr; sign trusted only where |F_O| >= max(2 meV/A, 3 sigma_O), sigma_O the oracle's last two terms "
       "(quadrupole, induction); a model force below 5 % of the oracle's is a missing tail, not a wrong sign. "
       "eps_F = (F_model - F_O)/rms_orient(F_O) per pair and r; dE_orient = energy error minus its orientation median. "
       "Per class, far band: fail if the wrong-sign fraction > 0.02 (>= 20 trusted points), or, for monopole pairs, "
       "q95 |eps_F| > 0.5 or q95 |dE_orient| > k_BT (300 K). Mid band: the same rule gives warn (oracle not validated "
       "against DFT there). Median, q95 and max are all reported",
       "orientation-dependent mid/far-range defects that a single-orientation scan and a median miss (a link-state "
       "mixture whose mask switches on in some orientations: wrong-sign NH4+...Gdm+ force at 8.5 A)")
def orientation_far(ctx: Ctx):
    from .checks import _need_dimers
    if (e := _need_dimers(ctx)):
        return e
    if not ctx.moments:
        return outcome(NE, "no fragment moments (--moments frags.json)")
    rows = sample(ctx)
    if not rows:
        return outcome(NE, "no dimer pair has moments for both fragments")
    if not any(np.isfinite(r["F_m"]) for r in rows):
        return outcome(NE, "the backend returned no forces")
    bands = {b: summarise([r for r in rows if r["band"] == b]) for b in BANDS}
    far, mid = violations(bands["far"]), violations(bands["mid"])
    if not any(v["n_sign_trusted"] or v["n"] for v in bands["far"].values()):
        return outcome(NE, "no far-band point scored", bands=bands)
    if far:
        st, msg = FAIL, "far: " + "; ".join(far)
    elif mid:
        st, msg = WARN, "far ok; mid (oracle not validated by DFT): " + "; ".join(mid)
    else:
        st, msg = PASS, "sign and size agree with the oracle in every orientation sampled"
    if far and mid:
        msg += "; mid: " + "; ".join(mid)
    return outcome(st, msg, bands=bands, n_points=len(rows), dft_stage=dft_stage(rows),
                   grid_r=list(ctx.opts.get("orient_r", R_MID + R_FAR)), n_orient=int(ctx.opts.get("orient_n", 128)))


# --------------------------------------------------------------------------------------------- adversarial search
def search(ctx: Ctx, P: Pair, band: str, seeds: List[dict], n_random: int, maxfev: int, rng) -> dict:
    """Maximise |eps_F| over (rotvec A, rotvec B, r) within the band, Nelder-Mead from each seed. eps_F is normalised
    by the oracle's orientation rms at the nearest sampled r, so the search does not simply run to the inner edge."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    lo, hi = BANDS[band]
    hi = min(hi, 15.0)
    scale = ctx.opts["_orient_scale"][P.name]
    best = {"obj": -1.0}

    def evaluate(z):
        r = float(np.clip(z[6], lo, hi))
        ra, rb = Rotation.from_rotvec(z[:3]), Rotation.from_rotvec(z[3:6])
        dm = P.dmin(ra, rb, r)
        if band == "mid" and dm < MID_DMIN:
            return 0.0, None
        o = P.oracle(ra, rb, r)
        p = ctx.predict([P.structure(ra, rb, r, "adv")])[0]
        F_m = _axial(p, len(P.xa))
        if not np.isfinite(F_m):
            return 0.0, None
        rs = sorted(scale)
        s = scale[min(rs, key=lambda x: abs(x - r))]
        eps = (F_m - o["F_O2+ind"]) / s if s > 0 else 0.0
        row = dict(pair=P.name, cls=P.cls, k=None, r=r, d_min=dm, band=band, F_m=F_m, eps_F=eps, rot_a=z[:3],
                   rot_b=z[3:6], **o)
        row.update(classify(F_m, row))
        return abs(eps), row

    def f(z):
        v, row = evaluate(z)
        # wrong sign at a trusted oracle point is what the search is for: rank it above any same-sign error
        score = v + (10.0 if row is not None and row["wrong_sign"] else 0.0)
        if row is not None and score > best["obj"]:
            best.update(obj=score, row=row)
        return -score

    starts = [np.r_[s["rot_a"], s["rot_b"], s["r"]] for s in seeds]
    starts += [np.r_[Rotation.random(random_state=rng.integers(1 << 31)).as_rotvec(),
                     Rotation.random(random_state=rng.integers(1 << 31)).as_rotvec(), rng.uniform(lo, hi)]
               for _ in range(n_random)]
    n_eval = 0
    for z0 in starts:
        res = minimize(f, z0, method="Nelder-Mead",
                       options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-4,
                                "initial_simplex": z0 + np.vstack([np.zeros(7), 0.4 * np.eye(7)])})
        n_eval += int(res.nfev)
    return dict(best.get("row") or {}, n_eval=n_eval, n_starts=len(starts))


@check("orientation_adversarial", "physical", "no rigid orientation and separation in the far field where the model's "
       "force disagrees with the multipole oracle in sign (a witness geometry, found by search, not by sampling)",
       "per pair with moments, in each band (far r = 10-15 A; mid r = 7.5-10 A, d_min >= 4 A): Nelder-Mead over "
       "(rotation of A, rotation of B, r) maximising |eps_F|, from the 3 worst sampled orientations of orientation_far "
       "plus 3 random starts, <= 40 model calls each. Far band: fail if any search ends on a wrong-sign point where "
       "the oracle's sign is trusted, or, for monopole pairs, at |eps_F| > 1. Mid band: warn on the same. The worst "
       "geometry per pair and band is reported (and written as extxyz with --orient-out)",
       "an orientation-dependent wrong-sign force in a small solid angle that a random sample can miss")
def orientation_adversarial(ctx: Ctx):
    from .checks import _need_dimers, is_ct
    if (e := _need_dimers(ctx)):
        return e
    if not ctx.moments:
        return outcome(NE, "no fragment moments (--moments frags.json)")
    rows = sample(ctx)
    if not rows or not any(np.isfinite(r["F_m"]) for r in rows):
        return outcome(NE, "no sampled orientation to start from")
    ctx.opts["_orient_scale"] = collections.defaultdict(dict)
    for r in rows:
        ctx.opts["_orient_scale"][r["pair"]][r["r"]] = r["F_scale"]
    rng = np.random.default_rng(int(ctx.opts.get("orient_seed", 2026)))
    n_seed, n_rand, maxfev = (int(ctx.opts.get(k, d)) for k, d in
                              (("adv_seeds", 3), ("adv_random", 3), ("adv_maxfev", 40)))
    found, bad, warn, n_eval = [], [], [], 0
    for P in pairs(ctx):
        for band in BANDS:
            R = [r for r in rows if r["pair"] == P.name and r["band"] == band and np.isfinite(r["eps_F"])]
            if not R:
                continue
            seeds = sorted(R, key=lambda r: -abs(r["eps_F"]) - (10.0 if r["wrong_sign"] else 0.0))[:n_seed]
            w = search(ctx, P, band, seeds, n_rand, maxfev, rng)
            n_eval += w.get("n_eval", 0)
            if "F_m" not in w:
                continue
            out = dict(_row_out(w), cls=P.cls, band=band, robust=w["robust"], wrong_sign=w["wrong_sign"],
                       n_eval=w["n_eval"])
            found.append((out, w))
            if is_ct(P.cls):
                continue
            hit = w["wrong_sign"] or (is_monopole(P.cls) and abs(w["eps_F"]) > 1.0)
            if hit:
                (bad if band == "far" else warn).append(out)
    if not found:
        return outcome(NE, "the search found no scorable geometry")
    if ctx.opts.get("orient_out"):
        write_witnesses(ctx.opts["orient_out"], [w for _, w in found], {P.name: P for P in pairs(ctx)})
    fmt = lambda o: (f"{o['pair']} r {o['r']:.2f} A: F {o['F_model']:+.3f} vs oracle {o['F_oracle']:+.3f} eV/A"
                     + (" (wrong sign)" if o["wrong_sign"] else ""))
    if bad:
        st, msg = FAIL, "far: " + "; ".join(fmt(o) for o in bad[:4])
    elif warn:
        st, msg = WARN, "far ok; mid (oracle not validated by DFT): " + "; ".join(fmt(o) for o in warn[:4])
    else:
        st, msg = PASS, "no wrong-sign or |eps_F| > 1 witness found"
    worst = sorted((o for o, _ in found), key=lambda o: -abs(o["eps_F"] or 0))
    return outcome(st, msg, n_model_calls=n_eval, far_witnesses=bad, mid_witnesses=warn, worst=worst[:60])


def write_witnesses(path: str, found: List[dict], P: Dict[str, Pair]):
    """Worst geometry per pair and band as extxyz, with model and oracle forces in info."""
    from ase import Atoms
    from ase.io import write
    from scipy.spatial.transform import Rotation
    frames = []
    for w in sorted(found, key=lambda w: -abs(w["eps_F"])):
        p = P[w["pair"]]
        a = Atoms(p.species, positions=p.positions(Rotation.from_rotvec(w["rot_a"]), Rotation.from_rotvec(w["rot_b"]),
                                                   w["r"]))
        a.info.update(pair=w["pair"], cls=w["cls"], band=w["band"], r_com=w["r"], d_min=w["d_min"],
                      charge=p.qa + p.qb, spin_multiplicity=p.spin, nA=len(p.xa), F_model=w["F_m"],
                      F_oracle=w["F_O2+ind"], sigma_oracle=w["sigma_O"], eps_F=w["eps_F"],
                      wrong_sign=bool(w["wrong_sign"]), oracle_sign_trusted=bool(w["robust"]))
        frames.append(a)
    if frames:
        write(path, frames, format="extxyz")
