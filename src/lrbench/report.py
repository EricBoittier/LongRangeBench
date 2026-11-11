"""Tables (markdown / HTML) and figures from lrbench result JSONs.

Every interval shown is the 95 % hierarchical-bootstrap percentile interval (seeds x scans/pairs, see stats.py);
asymmetric, so always given as [lo, hi], never as +-."""
from __future__ import annotations

import html as _html
import json
import os
from typing import List

HEAD_CLASSES = ["ion_pair", "same_sign", "ion_polar", "ion_nonpolar", "neutral_polar_control"]
WIRE = "wire_disc_near_vs_far"


def _f(x, k=3):
    return "–" if x is None else f"{x:.{k}f}"


def _ci(v, k=3):
    if not v:
        return "–"
    e, lo, hi = (list(v) + [None, None])[:3]
    return f"{_f(e, k)} [{_f(lo, k)}, {_f(hi, k)}]" if lo is not None else _f(e, k)


def _label(r):
    return r.get("label") or os.path.basename(r["members"][0]["spec"])


def _wire(ddE):
    return (ddE.get("by_family") or {}).get(WIRE) if ddE else None


def headline_rows(r) -> List[tuple]:
    """(metric, value, n, note) for one result, set by set."""
    rows = []
    for sn, S in r["sets"].items():
        E = S.get("E_int") or {}
        for c in [c for c in HEAD_CLASSES if c in (E.get("tail") or {})] if sn != "trimers" else []:
            t = E["tail"][c]
            rows.append((f"{sn}: tail fraction, {c} (d >= 8 A)", _ci(t["slope"], 3),
                         f"{t['n_points']} pts / {t['n_scans']} scans",
                         "slope through 0 of model dE on DFT dE; 1 = follows DFT, 0 = flat"))
        for c in HEAD_CLASSES[:2]:
            o = (E.get("offset_free_8A") or {}).get(c)
            if o:
                rows.append((f"{sn}: offset-free dE error at 8 A, {c} (meV)", _ci(o["mae"], 1), str(o["n"]),
                             f"flat model {_f(o['flat_model_mae'], 1)} meV"))
            a = (E.get("abs_ge12") or {}).get(c)
            if a:
                rows.append((f"{sn}: |E_int| error >= 12 A, {c} (meV)", _ci(a["mae"], 1), str(a["n"]),
                             f"DFT mean |E_int| {_f(a['dft_mean_abs'], 1)} meV"))
        for kind in ("ddE", "dE"):
            X = S.get(kind)
            if X and X.get("n") and "slope" in X and not (kind == "ddE" and _wire(X)):
                rows.append((f"{sn}: all {kind} slope / MAE (meV)", f"{_ci(X['slope'], 3)} / {_ci(X['mae'], 1)}",
                             f"{X['n']} / {X['n_pairs']} groups", f"zero model MAE {_f(X['zero_model_mae'], 1)} meV"))
        w = _wire(S.get("ddE"))
        if w:
            rows.append((f"{sn}: fixed-Q wire ddE slope", _ci(w["slope"], 4), f"{w['n']} ddE / {w['n_pairs']} pairs",
                         "1 = follows DFT, 0 = flat"))
            rows.append((f"{sn}: fixed-Q wire ddE MAE (meV)", _ci(w["mae"], 1), str(w["n"]),
                         f"zero model {_f(w['zero_model_mae'], 1)} meV"))
        for f, v in (S.get("zwitterion") or {}).items():
            if v.get("local_zero_mae") is not None:
                rows.append((f"{sn}: zwitterion local-zero MAE, {f} (meV)", _f(v["local_zero_mae"], 1),
                             str(len(v["local_zero"])), f"dE_z(m) - dE_z(m={v['reference_size']})"))
        T = S.get("E3")
        if T and T.get("n"):
            rows.append((f"{sn}: E3 MAE (meV)", _ci(T["abs_mae"], 2), str(T["n"]),
                         f"zero model {_f(T['zero_model_mae'], 2)}"))
            g = T.get("geometry_relative")
            if g:
                rows.append((f"{sn}: dE3 vs far geometry MAE (meV)", _ci(g["mae"], 2), str(g["n"]),
                             f"zero model {_f(g['zero_model_mae'], 2)}; slope {_f(g['slope'], 3)}"))
        ef = (S.get("energy_force") or {}).get("all") or {}
        if ef.get("F_mae_meV_A") is not None:
            rows.append((f"{sn}: force MAE (meV/A)", _f(ef["F_mae_meV_A"], 1), str(S["energy_force"]["n_forces"]),
                         "absolute; same-level models only"))
        if S.get("n_pathology_records") or S.get("n_pathology_groups"):
            rows.append((f"{sn}: pathology-flagged (not scored)", str(len(S.get("pathology") or [])),
                         f"{S['n_pathology_records']} rec / {S['n_pathology_groups']} groups", "listed in JSON"))
    for sn, S in r["sets"].items():
        N = S.get("normalised") or {}
        t = N.get("dimer_tail") or {}
        for c in [c for c in HEAD_CLASSES[:2] if c in (t.get("by_class") or {})]:
            v = t["by_class"][c]
            rows.append((f"{sn}: normalised tail eps, {c} (d >= 8 A)", _ci(v["median"], 3), str(v["n"]),
                         f"median; 0 exact, -1 tail missing; tail error {_f(v['tail_err_kBT_median'], 2)} k_BT "
                         f"({t['temperature_K']:g} K)"))
        cp = N.get("coulomb_prior") or {}
        if cp.get("eps_pred") is not None:
            rows.append((f"{sn}: g<lambda_A lambda_B> - 1 (pred.) vs measured eps, ion pairs",
                         f"{_f(cp['eps_pred'], 3)} vs {_f(cp.get('eps_measured_median'), 3)}", str(cp.get("n_records", "")),
                         f"g = {cp['gate_g']}"))
        dd = (N.get("ddE") or {}).get("by_series") or {}
        if WIRE in dd:
            rows.append((f"{sn}: normalised wire ddE, (model - DFT)/DFT", _f(dd[WIRE]["signed_median"], 3),
                         str(dd[WIRE]["n"]), "median; -1 = flat model"))
        e3 = (N.get("E3") or {}).get("dE3_vs_far")
        if e3:
            rows.append((f"{sn}: normalised dE3 vs far, (model - DFT)/|DFT|", _ci(e3["median"], 3), str(e3["n"]), "median"))
    eq = r.get("equivariance")
    if eq:
        for k, v in eq["checks"].items():
            rows.append((f"equivariance {k}: max |dE| (meV), max |dF| (meV/A)",
                         f"{_f(v['max_abs_dE_meV'], 3)}, {_f(v['max_force_err_meV_A'], 3)}", str(v["n"]), ""))
    es = r.get("emulation_scores")
    if es:
        for k, lab in (("emulation", "emulation score: E_int MAE vs wB97M-V, DFT pathology points included (meV)"),
                       ("physics", "physics score: same, CT / fractional / unbound / spin-contaminated points excluded (meV)")):
            v = es.get(k) or {}
            if v.get("MAE_meV") is not None:
                rows.append((lab, _f(v["MAE_meV"], 1), str(v["n"]), "lrbench emulation set; see lrbench.emulation"))
    ro = r.get("rotational_eint")
    if ro:
        for d, v in ro["by_distance"].items():
            rows.append((f"rotational spread of dimer E_int at d_min {d} A: median sd, max range (meV)",
                         f"{_f(v['sd_median_meV'], 3)}, {_f(v['range_max_meV'], 3)}", str(v["n_scans"]),
                         f"{ro['n_rotations']} rotations; charges-only oracle floor 2.9 meV"))
            if v.get("tail_sd_median_meV") is not None:
                rows.append((f"rotational spread of offset-free tail E_int({d} A) - E_int(far): median sd, p90 sd (meV)",
                             f"{_f(v['tail_sd_median_meV'], 3)}, {_f(v['tail_sd_p90_meV'], 3)}", str(v["n_scans"]),
                             "same rotation at both distances"))
    if "props" in r["sets"]:
        from .props import headline_rows as props_rows
        rows += props_rows(r["sets"]["props"])
    return rows


def headline(r) -> str:
    out = [f"## {_label(r)}", "", "| metric | value [95 % CI] | n | note |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in headline_rows(r)]
    return "\n".join(out)


def _table(R):
    labels = [_label(r) for r in R]
    per = [dict((a, (b, c)) for a, b, c, _ in headline_rows(r)) for r in R]
    keys = []
    for r in R:
        for a, *_ in headline_rows(r):
            if a not in keys:
                keys.append(a)
    rows = [[k] + [(p.get(k) or ("–", ""))[0] for p in per] + [next((p[k][1] for p in per if k in p), "")]
            for k in keys]
    return ["metric"] + labels + ["n"], rows


def markdown(R) -> str:
    h, rows = _table(R)
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    esc = lambda c: c.replace("|", "\\|")
    out += ["| " + " | ".join(esc(c) for c in x) + " |" for x in rows]
    out += ["", "Values: estimate [95 % hierarchical-bootstrap percentile interval, members x scans/pairs]. "
            "Pathology-flagged records are never scored."]
    for r in R:
        mm = (r.get("level") or {}).get("mismatch")
        if mm:
            out.append(f"- **{_label(r)}: level mismatch** (model {r['level']['model_level']} vs labels {mm}); "
                       "its energy metrics compare different levels of theory")
    for r in R:
        notes = r.get("notes") or []
        if notes:
            out.append(f"- {_label(r)}: " + "; ".join(notes))
    return "\n".join(out)


def figures(R, figdir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from . import style
    style.setup()
    os.makedirs(figdir, exist_ok=True)
    labels = [_label(r) + (" (level mismatch)" if (r.get("level") or {}).get("mismatch") else "") for r in R]
    col = {l: style.PALETTE[k % len(style.PALETTE)] for k, l in enumerate(labels)}
    made = []
    # 1. tail fraction per class
    first = lambda r, key, sets: next((r["sets"][x][key] for x in sets if (r["sets"].get(x) or {}).get(key)), {})
    dimset = lambda r: [x for x in ["dimers"] + list(r["sets"]) if x != "trimers"]
    E = [(first(r, "E_int", dimset(r)) or {}).get("tail") or {} for r in R]
    cls = [c for c in HEAD_CLASSES if any(c in e for e in E)]
    if cls:
        fig, ax = plt.subplots(figsize=(style.W1, 0.5 + 0.28 * len(cls) * max(1, len(R)) ** 0.5))
        for k, (l, e) in enumerate(zip(labels, E)):
            for j, c in enumerate(cls):
                if c not in e:
                    continue
                s, lo, hi = e[c]["slope"]
                y = j + (k - (len(R) - 1) / 2) * 0.18
                ax.errorbar(s, y, xerr=[[s - lo], [hi - s]] if lo is not None else None, fmt="o", color=col[l],
                            ms=3, lw=0.8, capsize=0, label=l if j == 0 else None)
        ax.axvline(0, color=style.G3, lw=0.6)
        ax.axvline(1, color=style.G3, lw=0.6, ls="--")
        ax.text(1, -0.55, "follows DFT", fontsize=5.3, color=style.G1, ha="center", va="bottom")
        ax.text(0, -0.55, "flat (8 Å local)", fontsize=5.3, color=style.G1, ha="center", va="bottom")
        ax.set_yticks(range(len(cls)))
        ax.set_yticklabels([c.replace("_", " ") for c in cls])
        ax.set_ylim(len(cls) - 0.4, -0.75)
        ax.set_xlabel("tail fraction (slope of model ΔE on DFT ΔE, d ≥ 8 Å)")
        style.title(ax, "How much of the DFT dimer tail beyond 8 Å each model captures")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=5.5)
        style.note(ax, "Points: estimate; bars: 95 % hierarchical bootstrap (members x scans). dE relative to each "
                   "scan's longest point.", y=-0.25)
        style.save(fig, "tail_fraction", figdir)
        plt.close(fig)
        made.append("tail_fraction")
    # 2. fixed-Q wire ddE parity
    W = [first(r, "ddE", ["families_fixedq"] + list(r["sets"])) for r in R]
    if any(_wire(w) for w in W):
        fig, ax = plt.subplots(figsize=(style.W1, style.W1 * 0.8))
        lim = 0
        for l, w in zip(labels, W):
            rows = [x for x in w.get("rows", []) if x["family"] == WIRE]
            if not rows:
                continue
            y = np.array([x["dft"] for x in rows])
            m = np.array([x["model"] for x in rows])
            lim = max(lim, np.abs(y).max(), np.abs(m).max())
            s = _wire(w)["slope"]
            ax.plot(y, m, "o", ms=2.2, color=col[l], alpha=0.8, mec="none",
                    label=f"{l}: slope {_ci(s, 3)}, n = {len(y)}")
        lim *= 1.05
        ax.plot([-lim, lim], [-lim, lim], color=style.G2, lw=0.6, ls="--")
        ax.text(lim * 0.6, lim * 0.68, "y = x (DFT)", fontsize=5.3, color=style.G1, rotation=35)
        ax.axhline(0, color=style.G3, lw=0.5)
        ax.text(-lim * 0.95, lim * 0.03, "flat model", fontsize=5.3, color=style.G1)
        ax.set_xlabel("DFT ddE (meV)")
        ax.set_ylabel("model ddE (meV)")
        style.title(ax, "Fixed-Q wire double differences: near (9 Å) minus far (60 Å) probe")
        ax.legend(loc="upper left", fontsize=5.3)
        style.note(ax, "ddE = [E(s', 9 Å) - E(s, 9 Å)] - [E(s', 60 Å) - E(s, 60 Å)], total charge fixed. "
                   "Slope through 0 with 95 % hierarchical bootstrap [lo, hi] (members x pairs).", y=-0.2)
        style.save(fig, "fixedq_wire_parity", figdir)
        plt.close(fig)
        made.append("fixedq_wire_parity")
    # 3. E_int vs distance, ion pairs and same sign
    D = [(first(r, "E_int", dimset(r)) or {}).get("by_distance") or [] for r in R]
    cl2 = [c for c in ("ion_pair", "same_sign") if any(x["cls"] == c for d in D for x in d)]
    if cl2:
        fig, axs = plt.subplots(1, len(cl2), figsize=(style.W2, style.W1 * 0.6), squeeze=False)
        fig.subplots_adjust(wspace=0.35, bottom=0.25)
        for ax, c in zip(axs[0], cl2):
            ref = None
            for l, d in zip(labels, D):
                xs = [x for x in d if x["cls"] == c]
                if not xs:
                    continue
                dd = np.array([x["d"] for x in xs])
                ref = (dd, np.array([x["dft"] for x in xs]))
                ax.plot(dd, [x["model"] for x in xs], "-o", ms=2, color=col[l], label=l)
            if ref is not None:
                ax.plot(*ref, "-", color=style.INK, lw=1.2, label="DFT")
            ax.axvline(8, color=style.G3, lw=0.6, ls=":")
            ax.text(8.3, 0.02, "8 Å cutoff", fontsize=5.3, color=style.G1, va="bottom",
                    transform=ax.get_xaxis_transform())
            ax.set_xlabel("d_min (Å)")
            ax.set_ylabel("mean E_int (meV)")
            style.title(ax, c.replace("_", " ") + ": mean E_int vs separation")
        axs[0][0].legend(fontsize=5.3)
        style.note(axs[0][0], "Mean over scans at each d_min (rigid monomers at fragment charges); n per point in the "
                   "result JSON (E_int.by_distance). The flat model curves beyond 8 Å are the charge-state offset.",
                   y=-0.3)
        style.save(fig, "eint_vs_distance", figdir)
        plt.close(fig)
        made.append("eint_vs_distance")
    # 4. normalised tail eps(r) (ERM_priors.md plot 1)
    T = [next((S["normalised"]["dimer_tail"] for S in r["sets"].values()
               if ((S.get("normalised") or {}).get("dimer_tail") or {}).get("by_class_distance")), None) for r in R]
    if any(T):
        cls = [c for c in HEAD_CLASSES if any(t and c in t["by_class"] for t in T)]
        fig, axs = plt.subplots(1, len(cls), figsize=(style.W2, style.W1 * 0.7), sharey=True, squeeze=False)
        fig.subplots_adjust(wspace=0.12, bottom=0.3)
        for ax, c in zip(axs[0], cls):
            for l, t in zip(labels, T):
                if not t:
                    continue
                a0 = 1.0 if t["distance_unit"] == "bohr" else 1 / 0.529177210903
                xs = [x for x in t["by_class_distance"] if x["cls"] == c]
                if not xs:
                    continue
                rr = np.array([x["r"] * a0 for x in xs])
                m = np.array([x["median"][0] for x in xs], float)
                lo = np.array([np.nan if x["median"][1] is None else x["median"][1] for x in xs], float)
                hi = np.array([np.nan if x["median"][2] is None else x["median"][2] for x in xs], float)
                ax.fill_between(rr, lo, hi, color=col[l], alpha=0.2, lw=0)
                ax.plot(rr, m, "-o", ms=2, color=col[l], label=l)
            for yv, txt in ((0, "exact"), (-1, "tail missing")):
                ax.axhline(yv, color=style.G3, lw=0.6, ls="--" if yv else "-")
            ax.axvline(8 / 0.529177210903, color=style.G2, lw=0.5, ls=":")
            ax.axvline(24 / 0.529177210903, color=style.G2, lw=0.5, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([15, 20, 30, 40, 50])
            ax.set_xticks([], minor=True)
            ax.set_xticklabels(["15", "20", "30", "40", "50"])
            ax.set_ylim(-3, 1)
            ax.set_xlabel("r / a₀")
            style.title(ax, c.replace("_", " "))
        ax0 = axs[0][0]
        ax0.set_ylabel("eps = (dE_model - dE_DFT) / dE_phys")
        ax0.text(0.02, -0.93, "tail missing", transform=ax0.get_yaxis_transform(), fontsize=5.3, color=style.G1)
        ax0.text(0.02, 0.07, "exact", transform=ax0.get_yaxis_transform(), fontsize=5.3, color=style.G1)
        ax0.legend(fontsize=5.3)
        style.note(ax0, "Normalised tail error beyond 8 Å vs r/a0 (log). Dotted: receptive field 8 Å and connected "
                   "reach 24 Å. dE relative to each scan's longest point; dE_phys from E_es_exact, else the multipole "
                   "E_mp. Line: median over points; band: 95 % hierarchical bootstrap (members x scans); y clipped to "
                   "[-3, 1].", y=-0.3)
        style.save(fig, "normalised_tail", figdir)
        plt.close(fig)
        made.append("normalised_tail")
    return made


def compare(paths, html=None, figdir=None, md_out=None, emulation=None) -> str:
    R = [json.load(open(p)) for p in paths]
    if emulation:  # lrbench emulation output: attach the emulation / physics E_int scores by model label
        E = json.load(open(emulation)).get("energy", {})
        for r in R:
            for key in (r.get("label_short"), r.get("label")):
                v = ((E.get(key) or {}).get("emulation") or {}).get("all")
                if v:
                    r["emulation_scores"] = v
                    break
    md = markdown(R)
    if md_out:
        open(md_out, "w").write(md + "\n")
    figs = figures(R, figdir) if figdir else []
    if html:
        h, rows = _table(R)
        t = "<table><tr>" + "".join(f"<th>{_html.escape(x)}</th>" for x in h) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{_html.escape(x)}</td>" for x in r) + "</tr>" for r in rows) + "</table>"
        imgs = ""
        if figdir:
            rel = os.path.relpath(figdir, os.path.dirname(os.path.abspath(html)) or ".")
            imgs = "".join(f'<figure><img src="{rel}/{f}.svg" alt="{f}"></figure>' for f in figs)
        open(html, "w").write(
            "<!doctype html><html><head><meta charset='utf-8'><title>lrbench comparison</title><style>"
            "body{font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#1a1a1a;background:#fff;margin:16px;"
            "max-width:1100px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:3px 8px;"
            "text-align:left}th{border-bottom:1px solid #4d4d4d}figure{margin:16px 0}img{max-width:100%}"
            "</style></head><body><h1>lrbench comparison</h1>" + t +
            "<p>Values: estimate [95 % hierarchical-bootstrap percentile interval, members x scans/pairs]. "
            "Pathology-flagged records are never scored.</p>" + imgs + "</body></html>")
    return md


def perf_table(rows) -> str:
    h = ["model", "box", "atoms", "code path", "precision", "GPU", "force ms", "us/atom", "MD ms/step",
         "ns/day (0.5 fs)", "peak GPU MiB"]
    out = ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['model']} | {r['box']} | {r['n_atoms']} | {r['code_path']} | {r['precision']} | "
                       f"{r.get('gpu')} | error: {str(r['error'])[:80]} |||||")
            continue
        out.append("| " + " | ".join(str(x) for x in [
            r["model"], r["box"], r["n_atoms"], r["code_path"], r["precision"], r.get("gpu"), r.get("force_ms", "–"),
            r.get("force_us_atom", "–"), r.get("md_ms_step", "–"),
            f"{r.get('ns_day', '–')} ({r.get('ns_day_basis', '')})", r.get("peak_gpu_mib", "–")]) + " |")
    return "\n".join(out)
