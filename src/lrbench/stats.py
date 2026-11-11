"""Uncertainty: hierarchical bootstrap percentile intervals and split-conformal / Mondrian / CQR calibration.

Ports of the program's recipes:
  - ``hier_bootstrap``: figures/scripts/prep_fig.py and analyse_fixedq.py (resample seeds, then groups (scans/pairs),
    optionally points inside each group); 2.5/97.5 percentiles.
  - ``conformal``: figures/scripts/calib2.py (split by group, 20 random 50/50 calibration/test splits, Mondrian bins,
    score |e|/sigma when a spread exists else |e|; coverage median and 2.5-97.5 % over splits, median width,
    conditional coverage min/max with the worst bin, naive +-z sigma coverage).
  - ``cqr``: figures/scripts/calib.py (quantile regression of the signed error on features, conformalised globally and
    per Mondrian bin). Gradient boosting when scikit-learn is present, else linear quantile regression (scipy HiGHS).
"""
from __future__ import annotations

import collections
from typing import Callable, Sequence

import numpy as np

LEVELS = (0.68, 0.90, 0.95)
ZNORM = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96}


def hier_bootstrap(stat: Callable[[np.ndarray, np.ndarray], float], n_members: int, groups: Sequence,
                   n_boot: int = 2000, seed: int = 3, points: bool = False, level: float = 0.95):
    """stat(member_idx, point_idx) -> float. Resample members (seeds) with replacement, then groups, then (optionally)
    points inside each resampled group. Returns (estimate, lo, hi) with percentile bounds."""
    groups = np.asarray(groups)
    G = sorted(set(groups.tolist()))
    members = np.arange(n_members)
    allp = np.arange(len(groups))
    est = float(stat(members, allp))
    if len(G) < 2:
        return est, float("nan"), float("nan")
    gidx = {g: np.where(groups == g)[0] for g in G}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        sd = rng.integers(0, n_members, n_members)
        gs = rng.integers(0, len(G), len(G))
        parts = []
        for k in gs:
            ii = gidx[G[k]]
            parts.append(ii[rng.integers(0, len(ii), len(ii))] if points else ii)
        v = stat(sd, np.concatenate(parts))
        if np.isfinite(v):
            bs.append(v)
    a = (1 - level) / 2
    return est, float(np.percentile(bs, 100 * a)), float(np.percentile(bs, 100 * (1 - a)))


def _cq(s, lv):
    s = np.sort(s[np.isfinite(s)])
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * lv))
    return s[min(k, n) - 1] if k <= n else np.inf


def conformal(y, pred, sig, groups, bins, n_splits: int = 20, seed: int = 7, levels=LEVELS) -> dict:
    """Mondrian split-conformal intervals pred +- q_b * sig (sig None -> 1, i.e. unnormalised |e|)."""
    y, pred, groups, bins = map(np.asarray, (y, pred, groups, bins))
    has_sig = sig is not None
    sig = np.ones_like(y, float) if sig is None else np.maximum(np.asarray(sig, float), 1e-3 * (np.nanstd(y) + 1e-12))
    s = np.abs(y - pred) / sig
    G = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    res = {lv: {"cov": [], "w": [], "cond": collections.defaultdict(list), "condw": collections.defaultdict(list)}
           for lv in levels}
    for _ in range(n_splits):
        cal = set(rng.permutation(G)[: len(G) // 2].tolist())
        ic = np.array([g in cal for g in groups.tolist()])
        it = ~ic
        if not it.any() or not ic.any():
            continue
        for lv in levels:
            q = np.full(len(y), np.inf)
            for b in set(bins.tolist()):
                j = bins == b
                q[j] = _cq(s[ic & j], lv)
            inn = s <= q
            w = 2 * q * sig
            res[lv]["cov"].append(float(inn[it].mean()))
            fw = it & np.isfinite(w)
            res[lv]["w"].append(float(np.median(w[fw])) if fw.any() else float("nan"))
            for b in set(bins.tolist()):
                j = it & (bins == b)
                if j.sum():
                    res[lv]["cond"][b].append(float(inn[j].mean()))
                    if np.isfinite(w[j]).any():
                        res[lv]["condw"][b].append(float(np.median(w[j][np.isfinite(w[j])])))
    out = {"n": int(len(y)), "n_groups": int(len(G)), "bins": sorted(set(bins.tolist())), "normalised": has_sig,
           "n_splits": n_splits}
    for lv in levels:
        c = np.array(res[lv]["cov"])
        if not len(c):
            continue
        cm = {b: float(np.median(v)) for b, v in res[lv]["cond"].items()}
        out[str(int(round(lv * 100)))] = {
            "coverage": float(np.median(c)), "lo": float(np.percentile(c, 2.5)), "hi": float(np.percentile(c, 97.5)),
            "width": float(np.nanmedian(res[lv]["w"])), "cond": cm,
            "cond_width": {b: float(np.median(v)) for b, v in res[lv]["condw"].items()},
            "cond_min": min(cm.values()) if cm else None, "cond_max": max(cm.values()) if cm else None,
            "worst_bin": min(cm, key=cm.get) if cm else None,
            "naive_coverage": float(np.mean(np.abs(y - pred) <= ZNORM[lv] * sig)) if has_sig else None}
    return out


def _quantile_models(X, e, alpha, seed):
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=150, max_depth=3, learning_rate=0.05,
                                      random_state=seed).fit(X, e)
        return m.predict, "gradient boosting (sklearn)"
    except ImportError:
        pass
    from scipy.optimize import linprog  # linear quantile regression: min sum alpha u+ + (1-alpha) u-
    A = np.column_stack([np.ones(len(X)), X])
    n, p = A.shape
    c = np.concatenate([np.zeros(p), alpha * np.ones(n), (1 - alpha) * np.ones(n)])
    Aeq = np.hstack([A, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    r = linprog(c, A_eq=Aeq, b_eq=e, bounds=bounds, method="highs")
    beta = r.x[:p] if r.success else np.r_[np.quantile(e, alpha), np.zeros(p - 1)]
    return (lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ beta), "linear quantile regression (scipy)"


def cqr(y, pred, X, groups, bins, n_splits: int = 20, seed: int = 1, level: float = 0.90) -> dict:
    """Conformalised quantile regression of the error e = pred - y; 40/30/30 train/calibration/test split by group.
    Intervals on y: [pred - (q_hi + Q), pred - (q_lo - Q)], Q global or per Mondrian bin."""
    y, pred, groups, bins = map(np.asarray, (y, pred, groups, bins))
    X = np.asarray(X, float)
    e = pred - y
    G = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    a = 1 - level
    cov = {"CQR": [], "Mondrian CQR": []}
    wid = {k: [] for k in cov}
    cond = {k: collections.defaultdict(list) for k in cov}
    method = None
    for sp in range(n_splits):
        perm = rng.permutation(G)
        n = len(perm)
        tr, ca = set(perm[: int(.4 * n)].tolist()), set(perm[int(.4 * n): int(.7 * n)].tolist())
        itr = np.array([g in tr for g in groups.tolist()])
        ica = np.array([g in ca for g in groups.tolist()])
        ite = ~(itr | ica)
        if itr.sum() < 5 or not ica.any() or not ite.any():
            continue
        flo, method = _quantile_models(X[itr], e[itr], a / 2, sp)
        fhi, _ = _quantile_models(X[itr], e[itr], 1 - a / 2, sp)
        ql, qh = flo(X), fhi(X)
        s = np.maximum(ql - e, e - qh)
        ivs = {}
        Q = _cq(s[ica], level)
        ivs["CQR"] = (pred - (qh + Q), pred - (ql - Q))
        lo, hi = np.empty_like(pred, dtype=float), np.empty_like(pred, dtype=float)
        for b in set(bins.tolist()):
            j = bins == b
            Qb = _cq(s[ica & j], level)
            lo[j], hi[j] = pred[j] - (qh[j] + Qb), pred[j] - (ql[j] - Qb)
        ivs["Mondrian CQR"] = (lo, hi)
        for m, (l, h) in ivs.items():
            inn = (y >= l) & (y <= h)
            cov[m].append(float(inn[ite].mean()))
            w = (h - l)[ite]
            wid[m].append(float(np.median(w[np.isfinite(w)])) if np.isfinite(w).any() else float("nan"))
            for b in set(bins.tolist()):
                j = ite & (bins == b)
                if j.sum():
                    cond[m][b].append(float(inn[j].mean()))
    out = {"level": level, "method": method, "n": int(len(y)), "n_groups": int(len(G))}
    for m in cov:
        c = np.array(cov[m])
        if not len(c):
            continue
        cm = {b: float(np.median(v)) for b, v in cond[m].items()}
        out[m] = {"coverage": float(np.median(c)), "lo": float(np.percentile(c, 2.5)), "hi": float(np.percentile(c, 97.5)),
                  "width": float(np.nanmedian(wid[m])), "cond": cm, "cond_min": min(cm.values()),
                  "cond_max": max(cm.values()), "worst_bin": min(cm, key=cm.get)}
    return out


def slope0(x, y) -> float:
    """Least-squares slope through the origin of y on x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x @ x
    return float(x @ y / d) if d > 0 else float("nan")
