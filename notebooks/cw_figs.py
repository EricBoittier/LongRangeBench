"""Figures for the ChargeWorld notebook (chargeworld.py). matplotlib only; one function per figure.

Style: Nature-like widths (89 / 183 mm), 7-8 pt sans-serif, no top/right spines, greys for context. Colour follows the
notebook's grammar: the blue/red pair is used for charge sign only; trapped versus learned runs use the purple/green
pair; schemes use fixed Okabe-Ito hues (cw_data.SCHEME_COL). Intervals are Clopper-Pearson 95 % for rates.
"""
from __future__ import annotations

import numpy as np

import cw_data as C

W1, W2 = 89 / 25.4, 183 / 25.4
TRAP, OK = "#6C4F9E", "#2E8B57"
A_COL = {"A0": "#8C8C8C", "A1": "#1B3F73"}


def style(plt):
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#4D4D4D", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 130,
                         "savefig.bbox": "tight", "font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "legend.frameon": False})


def _nominor(ax):
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())


def _letter(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def qcol(q, qmax=3.0):
    from matplotlib.colors import to_rgb
    t = float(np.sign(q) * np.sqrt(min(1.0, abs(q) / qmax))); z = np.array(to_rgb(C.CHARGE["zero"]))
    end = np.array(to_rgb(C.CHARGE["pos"] if t >= 0 else C.CHARGE["neg"]))
    return tuple(z + abs(t) * (end - z))


# ------------------------------------------------------------------------------------------------ section 1
def fixed_oracle(t):
    """per-species-mean-charge oracle with the tier's own kernel (T2 on); the point kernel is exact for T0 and T1."""
    b = t["baselines"]
    if "fixed_species_q_tier_kernel" in b:
        return b["fixed_species_q_tier_kernel"]["F_mae_meV_A_clean"]
    if t["tier"] in ("T0", "T1") and "fixed_species_q" in b:
        return b["fixed_species_q"]["F_mae_meV_A_clean"]
    return np.nan


def tier_baselines(plt, tiers):
    rows = [t for t in tiers if t["baselines"]]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    x = np.arange(len(rows))
    series = [("local two-body ridge, 6 Å cutoff", "#8C8C8C", "s",
               [t["baselines"].get("local2b_rc6", {}).get("F_mae_meV_A_clean", np.nan) for t in rows]),
              ("oracle: per-species mean charges, rest exact", "#1B3F73", "o", [fixed_oracle(t) for t in rows]),
              ("oracle: true charges as point charges, rest exact", "#E69F00", "D",
               [t["baselines"].get("point_true_q", {}).get("F_mae_meV_A_clean", np.nan) for t in rows])]
    for lab, c, m, y in series:
        y = np.asarray(y, float); y = np.where(y <= 1e-3, 1e-3, y)
        ax.plot(x, y, m, color=c, ms=4.5, label=lab, ls="none")
    fl = [t["noise_floor_F"] or np.nan for t in rows]
    ax.plot(x, fl, "_", color="#BF382B", ms=12, mew=1.6, label="label-noise floor (analytic)")
    ax.set_yscale("log"); ax.set_xticks(x, [t["tier"] for t in rows]); ax.set_ylabel("test force MAE (meV/Å)")
    ax.set_ylim(5e-4, 3e3)
    ax.set_title("Reference errors per tier")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return fig


# ------------------------------------------------------------------------------------------------ section 2
def structure_q(plt, true_q, model_q, names, title):
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.9))
    t = np.asarray(true_q); m = np.asarray(model_q)
    lim = max(3.3, np.abs(t).max() + 0.3, np.abs(m).max() + 0.3)
    ax.plot([-lim, lim], [-lim, lim], color="#BDBDBD", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [lim, -lim], color="#E3E3E3", lw=0.8, ls=":", zorder=0)
    wrong = (np.sign(m) != np.sign(t)) & (np.abs(t) > 0.25)
    ax.scatter(t[~wrong], m[~wrong], s=10, c=[qcol(v) for v in t[~wrong]], edgecolor="#4D4D4D", lw=0.3, label="right sign")
    ax.scatter(t[wrong], m[wrong], s=22, c=[qcol(v) for v in t[wrong]], edgecolor="#E69F00", lw=1.0, label="wrong sign")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("exact charge (e)"); ax.set_ylabel("model latent charge (e)")
    ax.set_title(title); ax.legend(loc="upper left")
    ax.text(0.98, 0.02, f"n = {len(t)} atoms, {int(wrong.sum())} wrong sign", transform=ax.transAxes, ha="right", fontsize=6.5)
    return fig


# ------------------------------------------------------------------------------------------------ section 3
def distributions(plt, dist, tiers=("T1", "T2", "T3b", "T5")):
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9))
    for k, t in enumerate(tiers):
        d = dist[t]; e = np.asarray(d["q"]["edges"]); c = np.asarray(d["q"]["counts"])
        axs[0].plot(0.5 * (e[1:] + e[:-1]), c / c.sum(), lw=1.0, color=["#1A1A1A", "#1B3F73", "#0072B2", "#E69F00"][k], label=t)
        e = np.asarray(d["Fnorm"]["edges"]); c = np.asarray(d["Fnorm"]["counts"])
        axs[1].plot(0.5 * (e[1:] + e[:-1]), c / c.sum(), lw=1.0, color=["#1A1A1A", "#1B3F73", "#0072B2", "#E69F00"][k])
        e = np.asarray(d["E_atom"]["edges"]); c = np.asarray(d["E_atom"]["counts"])
        axs[2].plot(0.5 * (e[1:] + e[:-1]), c / c.sum(), lw=1.0, color=["#1A1A1A", "#1B3F73", "#0072B2", "#E69F00"][k])
    axs[0].set_yscale("log"); axs[0].set_xlabel("atomic charge (e)"); axs[0].set_ylabel("fraction of atoms")
    axs[0].set_title("a  charges"); axs[0].legend(loc="upper left", fontsize=6)
    axs[1].set_xlabel("|F| (eV/Å)"); axs[1].set_title("b  force magnitude"); axs[1].set_yticks([])
    axs[2].set_xlabel("E / atom (eV)"); axs[2].set_title("c  energy per atom"); axs[2].set_yticks([])
    Q = dist["T1"]["Q"]; ks = sorted(Q, key=int)
    axs[3].bar([int(k) for k in ks], [Q[k] for k in ks], color="#8C8C8C", width=0.7)
    Qo = dist.get("T1_Qood", {}).get("Q", {})
    if Qo:
        ko = sorted(Qo, key=int); axs[3].bar([int(k) for k in ko], [Qo[k] for k in ko], color="#E69F00", width=0.7, label="Qood split")
        axs[3].legend(fontsize=6)
    axs[3].set_xlabel("total charge Q (e)"); axs[3].set_title("d  T1 test: total charge")
    fig.tight_layout()
    return fig


def cooc(plt, dist, names, add_combo=False):
    co = dist["_cooc"]["counts"]; Ct = np.asarray(co["train"], float)
    Cc = np.asarray(co["combo"], float)
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.95))
    M = Ct + (Cc if add_combo else 0)
    show = np.where(M > 0, np.log10(M + 1), np.nan)
    ax.imshow(show, cmap="Greys", vmin=0, vmax=np.log10(Ct.max() + 1))
    for i in range(12):
        for j in range(12):
            if Ct[i, j] == 0 and i != j:
                ax.text(j, i, "0" if not (add_combo and Cc[i, j] > 0) else f"+{int(Cc[i, j])}", ha="center", va="center",
                        fontsize=6, color="#E69F00" if add_combo and Cc[i, j] > 0 else "#BF382B", fontweight="bold")
    ax.set_xticks(range(12), names, rotation=90, fontsize=6); ax.set_yticks(range(12), names, fontsize=6)
    ax.set_title("T1 train: structures shared by two species" + (" (+ combo)" if add_combo else ""))
    return fig


# ------------------------------------------------------------------------------------------------ section 4
def overview(plt, runs):
    P = [x for x in runs if x["set"] == "phase1"]
    groups = [("T0", "lo"), ("T0", "hi"), ("T1", "mid"), ("T2", "mid")]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    for gi, (t, cap) in enumerate(groups):
        for ai, a in enumerate(("A0", "A1")):
            v = [x["F_mae"] for x in P if x["tier"] == t and x["arm"] == f"{a} {cap}"]
            fails = [x.get("fail") for x in P if x["tier"] == t and x["arm"] == f"{a} {cap}"]
            xs = gi + (ai - 0.5) * 0.32 + np.linspace(-0.05, 0.05, len(v))
            for xx, vv, f in zip(xs, v, fails):
                ax.plot(xx, vv, "o", ms=4.2, color=A_COL[a], mfc="white" if f else A_COL[a], mew=1.1)
    ax.set_yscale("log"); _nominor(ax); ax.set_xticks(range(4), [f"{t} {c}" for t, c in groups]); ax.set_ylabel("test force MAE (meV/Å)")
    ax.set_title("a  charge head (A1) vs none (A0), 4 seeds each")
    ax.plot([], [], "o", color=A_COL["A0"], label="A0 plain PET"); ax.plot([], [], "o", color=A_COL["A1"], label="A1 PET + LES charges")
    ax.plot([], [], "o", color=A_COL["A1"], mfc="white", label="A1 seed failed (sign rule)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=6)
    _letter(ax, "")
    ax = axs[1]
    T1 = [x for x in P if x["tier"] == "T1" and x["arm"] == "A1 mid"]
    for i, x in enumerate(sorted(T1, key=lambda r: r["seed"])):
        c = TRAP if x["fail"] else OK
        if x["F_mae_mixed"] is not None:
            ax.plot([i - 0.12], [x["F_mae_mixed"]], "s", color=c, ms=5)
            ax.annotate(f"{x['F_mae_mixed']:.0f}", (i - 0.12, x["F_mae_mixed"]), xytext=(-9, 0), textcoords="offset points", ha="right", va="center", fontsize=6)
        ax.plot([i + 0.12], [x["F_mae_coherent"]], "o", color=c, ms=5)
        ax.annotate(f"{x['F_mae_coherent']:.0f}", (i + 0.12, x["F_mae_coherent"]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=6)
    ax.set_xticks(range(len(T1)), [f"s{x['seed']}\n{x['n_mixed']} mixed" for x in sorted(T1, key=lambda r: r["seed"])], fontsize=6.3)
    ax.set_yscale("log"); _nominor(ax); ax.set_ylim(8, 600); ax.set_xlim(-0.6, len(T1) - 0.4); ax.set_ylabel("test force MAE (meV/Å)")
    ax.plot([], [], "s", color="#4D4D4D", label="sign-mixed structures"); ax.plot([], [], "o", color="#4D4D4D", label="sign-coherent structures")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6); ax.set_title("b  T1, A1: error sits in sign-mixed structures")
    fig.tight_layout()
    return fig


def species_signs(plt, runs, set_, tier, names_by_Z, schemes=None):
    R = [x for x in runs if x["set"] == set_ and x["tier"] == tier and x.get("species") and (schemes is None or x["scheme"] in schemes)]
    R = sorted(R, key=lambda r: (r["scheme"], r["seed"]))
    if not R:
        return None
    Zs = sorted({int(z) for x in R for z in x["species"]}, key=lambda z: R[0]["species"][str(z)]["truth"])
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    w = 0.8 / len(R)
    for k, x in enumerate(R):
        for i, z in enumerate(Zs):
            s = x["species"][str(z)]; xx = i - 0.4 + (k + 0.5) * w
            wrong = s["wrong"] > 0.5
            ax.bar(xx, s["mean"], width=w * 0.85, color=qcol(s["mean"]), edgecolor="#E69F00" if wrong else "none", lw=1.0)
            ax.plot([xx], [np.sign(s["truth"]) * s["absmean"]], "_", color="#1A1A1A", ms=4, mew=1.0)
    for i, z in enumerate(Zs):
        ax.plot([i - 0.45, i + 0.45], [R[0]["species"][str(z)]["truth"]] * 2, color="#1A1A1A", lw=1.0)
    ax.axhline(0, color="#8C8C8C", lw=0.5)
    ax.set_xticks(range(len(Zs)), [names_by_Z.get(z, str(z)) for z in Zs], fontsize=6.5)
    ax.set_ylabel("mean charge, global gauge (e)")
    lab = ", ".join(f"{x['scheme'] + ' ' if x['scheme'] != '-' else ''}s{x['seed']}{' ✗' if x['fail'] else ''}" for x in R)
    ax.set_title(f"{tier}: per-species mean charge per run (bars, left to right: {lab}); black line = exact")
    return fig


def curves(plt, runs, keys):
    fig, ax = plt.subplots(figsize=(W1 * 1.15, 2.2))
    for x in runs:
        k = run_key(x)
        if k not in keys or not x.get("curve"):
            continue
        c = x["curve"]; e = np.asarray(c["epoch"]); f = np.asarray(c["F_rmse"], float)
        col = TRAP if x.get("fail") else (OK if x.get("fail") is False else "#8C8C8C")
        ax.plot(e[e >= 0], f[e >= 0], lw=1.0, color=col)
        ax.annotate(k.split(" · ")[-1].replace("seed ", "s"), (e[-1], f[-1]), xytext=(3, 0), textcoords="offset points", fontsize=5.8, va="center", color=col)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation force RMSE (meV/Å)")
    ax.set_title("validation curves (purple = failed by the sign rule)")
    return fig


def run_key(x):
    s = "" if x["scheme"] == "-" else f" {x['scheme']}"
    st = {"phase1": "first set", "anchors": "replicate set", "ladder": "noise ladder"}.get(x["set"], x["set"])
    return f"{st} · {x['tier']} {x['arm']}{s} · seed {x['seed']}"


def rates(plt, runs, set_, key, order, names, tier="T1"):
    fig, ax = plt.subplots(figsize=(W1 * 1.2, 0.5 + 0.32 * len(order)))
    for i, s in enumerate(order):
        R = [x for x in runs if x["set"] == set_ and x["tier"] == tier and x[key] == s and x.get("fail") is not None]
        if not R:
            ax.text(0.02, i, f"{names.get(s, s)}: no finished runs", va="center", fontsize=6.5, color="#8C8C8C"); continue
        k, n = sum(x["fail"] for x in R), len(R); lo, hi = C.cp_interval(k, n)
        ax.plot([lo, hi], [i, i], color=C.SCHEME_COL.get(s, "#4D4D4D"), lw=2.2, solid_capstyle="round")
        ax.plot([k / n], [i], "o", color=C.SCHEME_COL.get(s, "#4D4D4D"), ms=5)
        ax.text(1.03, i, f"{k}/{n}  (seeds {', '.join(str(x['seed']) for x in R if x['fail']) or 'none'} failed)", va="center", fontsize=6.3)
    ax.set_yticks(range(len(order)), [names.get(s, s) for s in order]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("failure rate (sign rule), 95 % Clopper-Pearson")
    return fig


def traces(plt, tr, keys, label=lambda k: k):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.1))
    for k in keys:
        t = tr.get(k)
        if not t:
            continue
        st = np.maximum(np.asarray(t["step"]), 1)
        axs[0].plot(st, t["n_mixed"], lw=0.9, label=label(k))
        axs[1].plot(st, t["n_wrong"], lw=0.9, label=label(k))
        axs[1].plot([t["choice_step"]], [t["n_wrong"][-1]], "|", ms=9, color="#1A1A1A")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("training step"); ax.axvline(80, color="#BDBDBD", lw=0.6, ls=":")
    axs[0].set_ylabel("probe structures with mixed signs (of 32)"); axs[0].set_title("a  sign-mixed structures on the fixed probe")
    axs[1].set_ylabel("charged species at the wrong relative sign"); axs[1].set_title("b  flipped species (tick: sign-choice step)")
    axs[0].legend(fontsize=5.8, ncol=2)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ section 5
TOY_GROUPS = [
    ("optimiser smoothing", [("base", "reference: Adam, lr 0.05, warm-up 5 %"), ("lr0.2", "lr ×4"), ("lr0.01", "lr ×0.2 (not converged)"),
                             ("warm0", "no warm-up"), ("warm0.3", "warm-up 30 %"), ("beta1_0", "β1 = 0"), ("beta1_0.99", "β1 = 0.99"),
                             ("clip0.1", "gradient clip 0.1"), ("ema0.999", "EMA 0.999")]),
    ("gradient noise and start", [("mb16", "minibatch 16 structures"), ("mb4", "minibatch 4 structures"), ("init0.5", "10× larger initial charges")]),
    ("data composition", [("combo", "+ 50 structures where held-out pairs meet")]),
    ("E/F label noise", [("W1", "white 1× (2 meV/Å)"), ("W25", "white 25× (50 meV/Å)"), ("P1", "pathology tail 1×"),
                         ("P5", "pathology tail 5×"), ("S1", "ionicity bias 3 % (strong)"), ("S2", "ionicity bias 6 % (strong)")]),
    ("charge anchors (weight 0.1)", [("mse_exact", "MSE, exact labels"), ("mse_R1", "MSE, random noise 1×"), ("mse_B1", "MSE, bias 1×"),
                                     ("mse_BRT1", "MSE, bias+random+tail 1×"), ("mse_BRT2", "MSE, all 2×"), ("mse_X", "MSE, sign-crossing label"),
                                     ("corr_exact", "correlation, exact"), ("corr_R1", "correlation, random 1×"), ("corr_B1", "correlation, bias 1×"),
                                     ("corr_BRT1", "correlation, all 1×"), ("corr_BRT2", "correlation, all 2×"), ("corr_X", "correlation, sign-crossing"),
                                     ("WP1_mse_BRT1", "E/F noise 1× + MSE anchor all 1×")]),
]


def toy_arms(plt, arms, live=None):
    rows = []
    for g, items in TOY_GROUPS:
        for k, lab in items:
            if k in arms:
                rows.append((g, k, lab))
    fig, ax = plt.subplots(figsize=(W2 * 0.75, 0.6 + 0.165 * (len(rows) + (1 if live else 0))))
    y = 0; yt, yl = [], []; last = None; base = np.asarray(arms["base"]["trapped"])
    for g, k, lab in rows:
        if g != last:
            y += 0.4 if last else 0
            ax.text(-0.01, y, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.6, fontweight="bold")
            y += 1.0; last = g
        t = np.asarray(arms[k]["trapped"]); n = t.size; kk = int(t.sum()); lo, hi = C.cp_interval(kk, n)
        c = TRAP if kk else OK
        ax.plot([lo * n, hi * n], [y, y], color="#BDBDBD", lw=2.0, solid_capstyle="round")
        ax.plot([kk], [y], "o", color=c, ms=4)
        moved = int((base & ~t).sum()); new = int((~base & t).sum())
        ax.text(n + 1.5, y, f"{kk}/{n}" + (f"   {moved} rescued, {new} newly trapped" if k != "base" else ""), va="center", fontsize=6.0)
        yt.append(y); yl.append(lab); y += 1
    if live is not None:
        y += 0.5; kk, n = live; lo, hi = C.cp_interval(kk, n)
        ax.plot([lo * n, hi * n], [y, y], color="#F2C66D", lw=2.0); ax.plot([kk], [y], "D", color="#E69F00", ms=4.5)
        ax.text(64 + 1.5, y, f"{kk}/{n}  your run", va="center", fontsize=6.0); yt.append(y); yl.append("live trainer (settings above)")
    ax.set_yticks(yt, yl, fontsize=6.3); ax.invert_yaxis(); ax.set_xlim(0, 64)
    ax.set_xlabel("trapped runs of 64 paired starts (bar: 95 % Clopper-Pearson)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    return fig


def toy_live(plt, c, trace, q0, names, base_trapped=None):
    fl, rm, s = C.classify(c, q0); tr = fl.any(1)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0))
    ax = axs[0]
    for b in range(c.shape[0]):
        L = [t[2][b] for t in trace]; st = [t[0] for t in trace]
        ax.plot(st, L, lw=0.5, color=TRAP if tr[b] else OK, alpha=0.6)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(f"a  loss, {int(tr.sum())}/{len(tr)} trapped")
    ax = axs[1]
    order = np.argsort(q0)
    ax.bar(range(12), fl.sum(0)[order], color=[qcol(q0[i]) for i in order], edgecolor="#4D4D4D", lw=0.4)
    ax.set_xticks(range(12), [names[i] for i in order], rotation=90, fontsize=6); ax.set_ylabel("runs with this species flipped")
    ax.set_title("b  which species flip")
    ax = axs[2]
    ax.hist(np.log10(rm + 1e-9), bins=30, color="#8C8C8C")
    ax.set_xlabel("log10 charge RMSE after global sign (e)"); ax.set_title("c  final charge error")
    fig.tight_layout()
    return fig


def barrier(plt, D, trap_c):
    fA, _ = C.flip_blocks(trap_c, D.q0)
    t = np.linspace(-1.3, 1.3, 241)
    Cs = np.where(fA[None], t[:, None] * trap_c[None], trap_c[None])
    L = D.full(Cs)[0]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.plot(t, L, color="#1A1A1A", lw=1.1)
    i1, i2 = np.argmin(np.abs(t - 1)), np.argmin(np.abs(t + 1))
    ax.plot([1], [L[i1]], "o", color=TRAP); ax.plot([-1], [L[i2]], "o", color=OK)
    ax.annotate("trap", (1, L[i1]), xytext=(4, 6), textcoords="offset points", fontsize=6.5, color=TRAP)
    ax.annotate("block A flipped back", (-1, L[i2]), xytext=(4, 6), textcoords="offset points", fontsize=6.5, color=OK)
    k = np.argmax(L[(t > -1) & (t < 1)]) + np.argmax(t > -1)
    ax.plot([t[k]], [L[k]], "_", color="#1A1A1A", ms=8)
    ax.text(0.5, 0.04, f"barrier: {L[k] - L[i1]:.0f} above the trap, {L[k] - L[i2]:.0f} above the repaired point",
            transform=ax.transAxes, ha="center", fontsize=6.2)
    ax.set_ylim(bottom=min(0, L.min()) - 0.12 * (L.max() - L.min()))
    ax.set_xlabel("scale of the flipped species block (1 = trap)"); ax.set_ylabel("loss")
    ax.set_title("loss along the sign-repair line")
    return fig


def hessian(plt, eigs):
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, (lab, w) in enumerate(eigs):
        w = np.asarray(w)
        ax.scatter(np.full(w.size, i) + np.linspace(-0.15, 0.15, w.size), w, s=12, c=[TRAP if v < 0 else OK for v in w], lw=0)
        ax.text(i, w.max() * 1.25 + 0.5, f"{int((w < 0).sum())} negative", ha="center", fontsize=6)
    ax.set_yscale("symlog", linthresh=0.1); ax.axhline(0, color="#8C8C8C", lw=0.5)
    ax.set_xticks(range(len(eigs)), [e[0] for e in eigs]); ax.set_ylabel("Hessian eigenvalue")
    ax.set_title("curvature: unstable at the start, stable at the trap")
    return fig


# ------------------------------------------------------------------------------------------------ section 6
def conflict(plt, proj, both, G):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2))
    ax = axs[0]; proj = np.asarray(proj); both = np.asarray(both, bool)
    lim = np.quantile(np.abs(proj), 0.98)
    bins = np.linspace(-lim, lim, 41)
    ax.hist(np.clip(proj[both], -lim, lim), bins=bins, color="#1B3F73", alpha=0.85, label=f"contain both sign blocks (n = {both.sum()})")
    ax.hist(np.clip(proj[~both], -lim, lim), bins=bins, color="#E69F00", alpha=0.85, label=f"one block only (n = {(~both).sum()})")
    ax.axvline(0, color="#1A1A1A", lw=0.6)
    ax.set_xlabel("pull towards repair: −g_n · u (per structure)"); ax.set_ylabel("structures")
    ax.set_title("a  per-structure gradients at the trap"); ax.legend(fontsize=6, loc="upper left")
    ax = axs[1]
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-30); cos = Gn @ Gn.T
    o = np.argsort(proj); im = ax.imshow(cos[np.ix_(o, o)], cmap="PRGn", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("structures, sorted by pull →"); ax.set_title("b  gradient cosine between structures")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return fig


def bf_cos(plt, rows):
    rows = [dict(r, cos_late=np.nan if r["cos_late"] is None else r["cos_late"],
                 cos_early=np.nan if r["cos_early"] is None else r["cos_early"]) for r in rows]
    keep = {(r["system"], r["arm"]) for r in rows if np.isfinite(r["cos_late"]) or np.isfinite(r["cos_early"])}
    rows = [r for r in rows if (r["system"], r["arm"]) in keep]
    arms = sorted({(r["system"], r["arm"]) for r in rows},
                  key=lambda a: (a[0], np.nan_to_num(np.nanmedian([r["cos_late"] for r in rows if (r["system"], r["arm"]) == a] + [np.nan]), nan=9)))
    fig, ax = plt.subplots(figsize=(W2 * 0.7, 0.5 + 0.2 * len(arms)))
    for i, a in enumerate(arms):
        v = [r["cos_late"] for r in rows if (r["system"], r["arm"]) == a]
        e = [r["cos_early"] for r in rows if (r["system"], r["arm"]) == a]
        ax.plot(e, [i + 0.15] * len(e), "o", mfc="white", color="#8C8C8C", ms=3.5)
        ax.plot(v, [i] * len(v), "o", color="#1B3F73", ms=3.8)
        ax.text(1.02, i, f"n = {int(np.isfinite(v).sum())}", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#1A1A1A", lw=0.6)
    ax.set_yticks(range(len(arms)), [f"{s}: {a.replace('gpu2_', '').replace('gpu_', '').replace(s + '_', '')}" for s, a in arms], fontsize=6.3)
    ax.set_xlabel("cosine between force-loss and Born-charge-loss gradients on the charge parameters")
    ax.plot([], [], "o", color="#1B3F73", label="late (after step 300)"); ax.plot([], [], "o", mfc="white", color="#8C8C8C", label="early (steps 0-300)")
    ax.legend(loc="lower left", fontsize=6); ax.set_xlim(-1, 1)
    return fig


# ------------------------------------------------------------------------------------------------ section 7
def noise_scale(plt, rows):
    fig, ax = plt.subplots(figsize=(W2 * 0.68, 2.6))
    y = 0; yt = []
    for i, (lab, lo, mid, hi, g, k) in enumerate(rows):
        if i == 7:
            y += 0.7
        c = "#1A1A1A" if g == "dft" else "#1B3F73"
        if k == "rng":
            ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="round")
        elif k == "upper":
            ax.plot([mid], [y], marker=(3, 0, 90), color=c, ms=5, ls="none"); ax.plot([0.3, mid], [y, y], color=c, lw=0.6, ls=(0, (2, 2)))
        else:
            ax.plot([mid], [y], "o", color=c, ms=4)
        yt.append((y, lab)); y += 1
    ax.set_xscale("log"); ax.set_xlim(0.3, 3000); ax.invert_yaxis()
    ax.set_yticks([v for v, _ in yt], [lab for _, lab in yt], fontsize=6.3)
    ax.set_xlabel("force error or noise (meV/Å)")
    ax.set_title("DFT label noise (black) against ChargeWorld scales (blue)")
    return fig


def transfer(plt, ood):
    """test vs combo vs Qood force MAE per seed, A0 and A1, T1 and T2 (first set)."""
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for ax, tier in zip(axs, ("T1", "T2")):
        for si, split in enumerate(("", "_combo", "_Qood")):
            blk = ood.get(f"{tier}{split}", {})
            for ai, a in enumerate(("A0_mid", "A1_mid")):
                v = blk.get(a, {}).get("F_mae_meV_A_given")
                if not v:
                    continue
                xs = si + (ai - 0.5) * 0.3 + np.linspace(-0.06, 0.06, len(v))
                ax.plot(xs, v, "o", ms=4, color=A_COL[a[:2]])
        ax.set_xticks(range(3), ["test", "combo\n(held-out pairs meet)", "Qood\n(|Q| = 3, 4)"])
        ax.set_yscale("log"); ax.set_title(f"{'a' if tier == 'T1' else 'b'}  {tier}: force MAE by split")
    axs[0].set_ylabel("force MAE (meV/Å)")
    axs[0].plot([], [], "o", color=A_COL["A0"], label="A0 plain PET"); axs[0].plot([], [], "o", color=A_COL["A1"], label="A1 PET + LES")
    axs[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    return fig
