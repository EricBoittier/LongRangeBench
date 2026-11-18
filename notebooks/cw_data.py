"""Data layer and the species-charge toy for the ChargeWorld notebook (chargeworld.py).

Everything reads one folder, ``data/chargeworld/`` next to this file, written by ``tools/build_chargeworld_data.py``.
A missing file is not an error: ``load`` returns None for it and the notebook shows that block as still running.

The toy: 12 per-species charges c fitted to the exact point-charge Coulomb energies and forces of the 200 ChargeWorld
T1 validation structures. Energy E_n = 1/2 c^T A_n c; the force MSE of structure n is v^T M_n v - 2 b_n . v + y2_n
with v the 78 products c_i c_j (i <= j). Loss per structure = ((E_n - E*_n) / N_n)^2 + force MSE, in (eV/atom)^2 and
(eV/A)^2, as the trainer's MSE with unit weights. numpy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KE = 14.399645
IU = np.triu_indices(12)
IU78 = np.triu_indices(78)
# Z -> short species name (configs/species12.json of ChargeWorld; Z is only a label)
CHARGE = {"neg": "#2A6BC4", "zero": "#F5F5F5", "pos": "#BF382B"}
INK, GREY = "#1A1A1A", ["#4D4D4D", "#8C8C8C", "#BDBDBD", "#E3E3E3"]
SCHEME_COL = {"S0": "#4D4D4D", "S1": "#0072B2", "S2": "#009E73", "S4": "#E69F00", "-": "#4D4D4D"}
FILES = ["tiers.json", "structures.json", "distributions.json", "runs.json", "traces.json", "toy_ref.json",
         "conflict.json"]


def data_dir(notebook_dir) -> Path:
    return Path(notebook_dir) / "data" / "chargeworld"


def load(d: Path) -> dict:
    out = {}
    for f in FILES:
        p = Path(d) / f
        out[f.split(".")[0]] = json.loads(p.read_text()) if p.exists() else None
    p = Path(d) / "toy.npz"
    out["toy"] = dict(np.load(p)) if p.exists() else None
    return out


def stamp(d: Path) -> str:
    fs = [p for p in Path(d).iterdir()] if Path(d).exists() else []
    if not fs:
        return "no data folder"
    import datetime as _dt
    t = max(p.stat().st_mtime for p in fs)
    return f"{len(fs)} files, {sum(p.stat().st_size for p in fs) / 1e6:.1f} MB, newest {_dt.datetime.fromtimestamp(t):%d %b %H:%M}"


def cp_interval(k: int, n: int, a: float = 0.05):
    """Clopper-Pearson interval for k of n (scipy-free: bisection on the binomial tail)."""
    from math import comb

    def tail_ge(p):  # P(X >= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if (f(m) < target) == increasing:
                lo = m
            else:
                hi = m
        return 0.5 * (lo + hi)
    lo = 0.0 if k == 0 else solve(tail_ge, a / 2, True)
    hi = 1.0 if k == n else solve(tail_le, a / 2, False)
    return lo, hi


# ------------------------------------------------------------------------------------------------ toy
class Toy:
    """species-charge toy on the T1 validation structures (optionally + the 50 combo structures)."""

    def __init__(self, T: dict, combo: bool = False):
        parts = ["val", "combo"] if combo else ["val"]
        cat = lambda k: np.concatenate([np.atleast_1d(T[f"{p}_{k}"]) for p in parts])  # noqa: E731
        self.q0 = T["q0"].astype(float)
        self.A = cat("A").astype(float); self.E = cat("E").astype(float); self.N = cat("N").astype(float)
        self.b = cat("b").astype(float); self.y2 = cat("y2").astype(float); self.counts = cat("counts")
        self.n = len(self.E)
        self.Msum = sum(T[f"{p}_Msum"] for p in parts); self.bsum = sum(T[f"{p}_bsum"] for p in parts)
        self.y2sum = float(sum(T[f"{p}_y2sum"] for p in parts))
        Mu = np.concatenate([T[f"{p}_Mu"] for p in parts]).astype(float)
        self.Mn = np.zeros((self.n, 78, 78)); self.Mn[:, IU78[0], IU78[1]] = Mu
        self.Mn = self.Mn + self.Mn.transpose(0, 2, 1) - np.einsum("nii->ni", self.Mn)[:, :, None] * np.eye(78)[None]
        self.species_atoms = self.counts.sum(0).astype(float)

    @staticmethod
    def v(c):
        return c[:, IU[0]] * c[:, IU[1]]                                      # (B, 78)

    @staticmethod
    def dv(c):
        """J[b, k, j] = d v_k / d c_j."""
        B = c.shape[0]; J = np.zeros((B, 78, 12)); k = np.arange(78)
        np.add.at(J, (slice(None), k, IU[0]), c[:, IU[1]]); np.add.at(J, (slice(None), k, IU[1]), c[:, IU[0]])
        return J

    def full(self, c):
        """full-batch mean loss (B,) and gradient (B, 12): fp64 sums, no per-structure work for the force part."""
        v = self.v(c); J = self.dv(c)
        Mv = v @ self.Msum
        LF = (np.einsum("bk,bk->b", v, Mv) - 2 * v @ self.bsum + self.y2sum) / self.n
        gF = np.einsum("bk,bkj->bj", 2 * (Mv - self.bsum[None]), J) / self.n
        Ac = np.einsum("nij,bj->bni", self.A, c); E = 0.5 * np.einsum("bi,bni->bn", c, Ac)
        rE = (E - self.E[None]) / self.N[None]
        LE = (rE ** 2).mean(1); gE = np.einsum("bn,bni->bi", 2 * rE / self.N[None], Ac) / self.n
        return LE + LF, gE + gF

    def per_struct(self, c, idx=None):
        """per-structure losses (B, m) and gradients (B, m, 12) on structures idx (B, m) or all."""
        B = c.shape[0]
        idx = np.broadcast_to(np.arange(self.n), (B, self.n)) if idx is None else idx
        v = self.v(c); J = self.dv(c)
        M = self.Mn[idx]                                                       # (B, m, 78, 78)
        Mv = np.einsum("bmkl,bl->bmk", M, v)
        LF = np.einsum("bk,bmk->bm", v, Mv) - 2 * np.einsum("bmk,bk->bm", self.b[idx], v) + self.y2[idx]
        gF = np.einsum("bmk,bkj->bmj", 2 * (Mv - self.b[idx]), J)
        Ac = np.einsum("bmij,bj->bmi", self.A[idx], c); E = 0.5 * np.einsum("bi,bmi->bm", c, Ac)
        rE = (E - self.E[idx]) / self.N[idx]
        return rE ** 2 + LF, gF + 2 * (rE / self.N[idx])[:, :, None] * Ac

    def hessian(self, c, h=1e-5):
        c = np.atleast_2d(c); H = np.zeros((c.shape[0], 12, 12))
        for j in range(12):
            e = np.zeros(12); e[j] = h
            H[:, :, j] = (self.full(c + e)[1] - self.full(c - e)[1]) / (2 * h)
        return 0.5 * (H + H.transpose(0, 2, 1))


def classify(c, q0):
    """global sign by |q0|-weighted agreement; flipped charged species relative to it; charge RMSE after the sign."""
    s = np.sign((c * q0).sum(1, keepdims=True)); s[s == 0] = 1
    cc = c * s; charged = np.abs(q0) > 0.1
    flipped = (np.sign(cc) != np.sign(q0)) & charged[None]
    return flipped, np.sqrt(((cc - q0) ** 2).mean(1)), s[:, 0]


def train(D: Toy, B=64, steps=600, lr=0.05, b1=0.9, b2=0.999, warm=0.05, clip=None, batch=None, init_scale=0.05,
          seed=0, ema=None, anchor=None, w_anchor=0.0, log_every=10):
    """Adam with warm-up and cosine decay (as gradconf/scripts/toy.py). anchor: None, 'mse' or 'corr' on the exact
    per-atom charges. Returns final charges (B, 12) and a trace [(step, c, loss)]."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, init_scale, (B, 12))
    m = np.zeros_like(c); v = np.zeros_like(c); ce = c.copy(); trace = [(0, c.copy(), D.full(c)[0])]
    na = D.species_atoms; Mtot = na.sum()
    for t in range(1, steps + 1):
        if batch is None:
            L, g = D.full(c)
        else:
            idx = np.argsort(rng.random((B, D.n)), 1)[:, :batch]
            Ls, Gs = D.per_struct(c, idx); L, g = Ls.mean(1), Gs.mean(1)
        if anchor == "mse" and w_anchor > 0:
            g = g + w_anchor * 2 * na[None] * (c - D.q0[None]) / Mtot
        elif anchor == "corr" and w_anchor > 0:                              # 1 - Pearson(q, q_ref) over atoms
            mu_q = (na * c).sum(1, keepdims=True) / Mtot; mu_t = (na * D.q0).sum() / Mtot
            x = c - mu_q; y = D.q0 - mu_t
            sx = np.sqrt((na * x ** 2).sum(1, keepdims=True)); sy = np.sqrt((na * y ** 2).sum())
            rr = (na * x * y).sum(1, keepdims=True) / (sx * sy)
            g = g - w_anchor * na * (y[None] / (sx * sy) - rr * x / sx ** 2)
        if clip is not None:
            nrm = np.linalg.norm(g, axis=1, keepdims=True); g = g * np.minimum(1, clip / (nrm + 1e-30))
        lr_t = lr * min(1.0, t / max(1, warm * steps)) * (0.5 * (1 + np.cos(np.pi * t / steps)) * 0.99 + 0.01)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        c = c - lr_t * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-12)
        if ema is not None:
            ce = ema * ce + (1 - ema) * c
        if log_every and t % log_every == 0:
            trace.append((t, (ce if ema is not None else c).copy(), L.copy()))
    return (ce if ema is not None else c), trace


def flip_blocks(trap_c, q0):
    """index masks of the two sign blocks of a trapped minimum: species flipped against the rest (charged only)."""
    s = np.sign(trap_c @ q0) or 1.0
    charged = np.abs(q0) > 0.1
    flipped = (np.sign(s * trap_c) != np.sign(q0)) & charged
    return flipped, charged & ~flipped


def slice_grid(D: Toy, trap_c, n=41, lim=1.6):
    """loss on the plane c(a, b) = a * (trap restricted to block A) + b * (trap restricted to block B).
    (1, 1) is the trap, (-1, 1) its sign-repaired copy, (-1, -1) and (1, -1) their global mirrors."""
    fA, fB = flip_blocks(trap_c, D.q0)
    uA = np.where(fA, trap_c, 0.0); uB = np.where(~fA, trap_c, 0.0)
    a = np.linspace(-lim, lim, n)
    AA, BB = np.meshgrid(a, a, indexing="ij")
    C = AA.reshape(-1, 1) * uA[None] + BB.reshape(-1, 1) * uB[None]
    L = D.full(C)[0].reshape(n, n)
    return a, L, uA, uB


def project(c, uA, uB):
    return np.stack([(c @ uA) / (uA @ uA), (c @ uB) / (uB @ uB)], -1)


# ------------------------------------------------------------------------------------------------ sign flip demo
def coulomb_point(quad, c):
    """point-charge Coulomb energy (eV) and forces (eV/A) of one T1 structure for per-species charges c (12,)."""
    A = np.asarray(quad["A"]); sp = np.asarray(quad["sp"]); H = np.asarray(quad["H"]).reshape(len(sp), 3, 12)
    return 0.5 * c @ A @ c, c[sp][:, None] * np.einsum("iaj,j->ia", H, c)


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """one-sided Fisher exact p that group 1 (k1 of n1) fails more often than group 2 (k2 of n2)."""
    from math import comb
    K, N = k1 + k2, n1 + n2
    return sum(comb(K, i) * comb(N - K, n1 - i) for i in range(k1, min(K, n1) + 1)) / comb(N, n1)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    from math import comb
    K, N = k1 + k2, n1 + n2
    pr = [comb(K, i) * comb(N - K, n1 - i) / comb(N, n1) for i in range(max(0, K - n2), min(K, n1) + 1)]
    p0 = comb(K, k1) * comb(N - K, n1 - k1) / comb(N, n1)
    return min(1.0, sum(p for p in pr if p <= p0 * (1 + 1e-9)))
