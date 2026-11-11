"""Compare every scored model's long-range diagnostics.

    marimo edit notebooks/long_range_diagnostics.py       # interactive
    marimo run  notebooks/long_range_diagnostics.py       # read-only app

Reads every ``*.json`` written by ``lrbench eval`` from ``$LRBENCH_RUNS`` (default ``./runs``). Models that are
still running simply do not appear yet; re-run the first cell to pick them up.

Colours encode one thing only: whether the model has an explicit long-range term. Two hues, checked for
colour-vision separation (OKLab dE 21.9 protan, 31.2 normal), with every bar directly labelled so identity never
rests on colour alone.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import json
    import os

    import marimo as mo
    import numpy as np
    import pandas as pd

    RUNS = os.environ.get("LRBENCH_RUNS", "runs")

    def load(path):
        with open(path) as f:
            return json.load(f)

    results = {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        try:
            J = load(p)
            if "sets" in J:
                results[os.path.basename(p)[:-5]] = J
        except Exception as e:  # a file still being written
            print(f"skipped {p}: {type(e).__name__}")

    mo.md(f"# Long-range diagnostics\n\n**{len(results)} result files** from `{os.path.abspath(RUNS)}`: "
          + ", ".join(f"`{k}`" for k in results))
    return RUNS, mo, np, pd, results


@app.cell
def _(mo, pd, results):
    # ---------------------------------------------------------------- what each model is, from its own result JSON
    def provenance(name, J):
        m = (J.get("members") or [{}])[0]
        ext = m.get("external_model") or {}
        lr = str(ext.get("long_range", "")) or "?"
        return {
            "model": name,
            "level": J.get("level", {}).get("model_level", "?"),
            "long-range term": lr,
            "has LR": not lr.lower().startswith("none"),
            "charge / spin": str(ext.get("charge_spin", "?")),
            "licence": str(ext.get("licence", "?")),
            "version": str(ext.get("version", "")),
            "wall (s)": J.get("wall_s"),
        }

    prov = pd.DataFrame([provenance(k, v) for k, v in results.items()]).sort_values(
        ["has LR", "model"], ascending=[False, True])
    mo.vstack([
        mo.md("## What is being compared\n\n`has LR` is read from each adapter's own declaration, not guessed. "
              "A level that differs from the sets' `wB97M-V/def2-TZVPD` means energies are **not** comparable; the "
              "tail fraction and ΔΔE slope are dimensionless shape diagnostics and stay readable."),
        mo.ui.table(prov, selection=None),
    ])
    return (prov,)


@app.cell
def _(mo, results):
    sets_present = sorted({s for J in results.values() for s in J.get("sets", {})})
    set_pick = mo.ui.dropdown(options=sets_present, value="dimers" if "dimers" in sets_present
                              else (sets_present[0] if sets_present else None), label="set")
    set_pick
    return set_pick, sets_present


@app.cell
def _(mo, np, pd, results, set_pick):
    # ------------------------------------------------------------------------------- the headline diagnostics table
    def est(v):
        """[estimate, lo, hi] -> estimate; a bare number stays as it is."""
        if isinstance(v, (list, tuple)) and v:
            return v[0]
        return v if isinstance(v, (int, float)) else np.nan

    def row(name, J, setname, cls):
        S = (J.get("sets") or {}).get(setname) or {}
        E = S.get("E_int") or {}
        nrm = ((S.get("normalised") or {}).get("dimer_tail") or {}).get("by_class") or {}
        tail = (E.get("tail") or {}).get(cls) or {}
        off = (E.get("offset_free_8A") or {}).get(cls) or {}
        far = (E.get("abs_ge12") or {}).get(cls) or {}
        eps = nrm.get(cls) or {}
        ddE = S.get("ddE") or {}
        ef = (S.get("energy_force") or {}).get("all") or {}
        return {
            "model": name,
            "tail fraction": est(tail.get("slope")),
            "ε (median)": est(eps.get("median")),
            "ΔΔE slope": est(ddE.get("slope")),
            "offset-free 8 Å MAE (meV)": est(off.get("mae")),
            "|E_int| ≥12 Å MAE (meV)": est(far.get("mae")),
            "force MAE (meV/Å)": est(ef.get("F_mae_meV_A") or ef.get("F_mae")),
            "tail err > 1 k_BT": eps.get("frac_tail_err_gt_1kBT"),
            "n scans": tail.get("n_scans"),
        }

    classes = sorted({c for J in results.values()
                      for c in ((((J.get("sets") or {}).get(set_pick.value) or {}).get("E_int") or {})
                                .get("tail") or {})})
    cls_pick = mo.ui.dropdown(options=classes, value="ion_pair" if "ion_pair" in classes
                              else (classes[0] if classes else None), label="class")

    def table_for(setname, cls):
        df = pd.DataFrame([row(k, v, setname, cls) for k, v in results.items()])
        return df.sort_values("tail fraction", ascending=False, na_position="last")

    mo.vstack([
        mo.md("## Headline diagnostics\n\n"
              "A correct model reads **tail fraction 1**, **ε 0** and **ΔΔE slope 1**. A model with no long-range "
              "term reads **0, −1, 0**. That is why these columns need no baseline to interpret."),
        cls_pick,
    ])
    return classes, cls_pick, est, row, table_for


@app.cell
def _(cls_pick, mo, set_pick, table_for):
    df = table_for(set_pick.value, cls_pick.value)
    mo.vstack([mo.md(f"**{set_pick.value} / {cls_pick.value}**"),
               mo.ui.table(df.round(4), selection=None)])
    return (df,)


@app.cell
def _():
    # --------------------------------------------------------------------------------------- figure style + palette
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lrbench import style

    style.setup()
    LR_BLUE = "#0072B2"      # has an explicit long-range term
    LOCAL_RED = "#D55E00"    # local only
    INK, GREY = "#1a1a1a", "#8c8c8c"
    return GREY, INK, LOCAL_RED, LR_BLUE, plt, style


@app.cell
def _(GREY, INK, LOCAL_RED, LR_BLUE, df, mo, np, plt, prov, set_pick):
    # --------------------------------------------- tail fraction per model: magnitude by identity -> horizontal bars
    def tail_figure(d, prov_df, metric="tail fraction", good=1.0, floor=0.0, xlabel=None):
        d = d.dropna(subset=[metric])
        if d.empty:
            return None
        has_lr = dict(zip(prov_df["model"], prov_df["has LR"]))
        d = d.sort_values(metric)
        y = np.arange(len(d))
        colours = [LR_BLUE if has_lr.get(m) else LOCAL_RED for m in d["model"]]
        fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(d) + 1.25))
        ax.barh(y, d[metric], color=colours, height=0.62, zorder=3)
        ax.axvline(good, color=INK, lw=1.2, zorder=4)
        ax.axvline(floor, color=GREY, lw=1.0, ls=":", zorder=4)
        ax.text(good, len(d) - 0.35, "  correct", color=INK, va="center", fontsize=6)
        ax.text(floor, len(d) - 0.35, "  no long-range term", color=GREY, va="center", fontsize=6)
        for yi, (v, m) in enumerate(zip(d[metric], d["model"])):
            off = 0.012 * (max(d[metric].max(), good) - min(d[metric].min(), floor) or 1)
            ax.text(v + (off if v >= 0 else -off), yi, f"{v:.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6, color=INK)
        ax.set_yticks(y, d["model"], fontsize=6.5)
        ax.set_xlabel(xlabel or metric)
        ax.grid(axis="x", color="#e8e8e8", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=LR_BLUE), plt.Rectangle((0, 0), 1, 1, color=LOCAL_RED)]
        ax.legend(handles, ["explicit long-range term", "local only"], loc="lower right", frameon=False, fontsize=6)
        ax.set_title(f"{metric}, {set_pick.value}", loc="left", fontsize=7.5, fontweight="bold")
        fig.tight_layout()
        return fig

    fig_tail = tail_figure(df, prov)
    mo.vstack([mo.md("### Tail fraction\n\nSlope through zero of the model's ΔE on the reference ΔE beyond 8 Å. "
                     "The solid line is the correct answer; the dotted line is a model that is flat out there."),
               fig_tail if fig_tail is not None else mo.md("_no data yet_")])
    return fig_tail, tail_figure


@app.cell
def _(df, mo, prov, tail_figure):
    fig_eps = tail_figure(df, prov, metric="ε (median)", good=0.0, floor=-1.0,
                          xlabel="ε  (0 = exact, −1 = whole tail missing)")
    fig_ddE = tail_figure(df, prov, metric="ΔΔE slope", good=1.0, floor=0.0)
    mo.vstack([
        mo.md("### Normalised tail error ε, and the fixed-charge response"),
        fig_eps if fig_eps is not None else mo.md("_no ε data (run with `--normalise`)_"),
        fig_ddE if fig_ddE is not None else mo.md("_no ΔΔE data_"),
    ])
    return fig_ddE, fig_eps


@app.cell
def _(GREY, INK, LOCAL_RED, LR_BLUE, mo, np, plt, prov, results, set_pick):
    # ------------------------------------- eps(r): one small panel per model rather than 12 lines in one axes
    def eps_vs_r(setname, cls):
        rows = {}
        for name, J in results.items():
            dt = (((J.get("sets") or {}).get(setname) or {}).get("normalised") or {}).get("dimer_tail") or {}
            byrd = dt.get("by_class_distance") or {}
            items = byrd.values() if isinstance(byrd, dict) else byrd
            pts = sorted(((float(x["r"]), x["median"][0] if isinstance(x["median"], list) else x["median"])
                          for x in items if x.get("cls") == cls and x.get("r") is not None), key=lambda t: t[0])
            if pts:
                rows[name] = np.array(pts)
        if not rows:
            return None
        has_lr = dict(zip(prov["model"], prov["has LR"]))
        order = sorted(rows, key=lambda m: (not has_lr.get(m), m))
        ncol = min(4, len(order))
        nrow = int(np.ceil(len(order) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.95 * ncol, 1.7 * nrow), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, name in zip(axes.ravel(), order):
            a = rows[name]
            c = LR_BLUE if has_lr.get(name) else LOCAL_RED
            ax.axhline(0.0, color=INK, lw=0.9)
            ax.axhline(-1.0, color=GREY, lw=0.8, ls=":")
            ax.plot(a[:, 0], a[:, 1], "-o", color=c, lw=1.6, ms=3.5, zorder=3)
            ax.set_title(name, loc="left", fontsize=6.5, fontweight="bold")
            ax.set_ylim(-1.45, 0.65)
            ax.tick_params(labelsize=5.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for ax in axes.ravel()[len(order):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("r (Å)", fontsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("ε", fontsize=6)
        fig.suptitle(f"ε against distance — {setname} / {cls}   (0 = exact, −1 = tail missing)",
                     x=0.01, ha="left", fontsize=7.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    fig_r = eps_vs_r(set_pick.value, "ion_pair")
    mo.vstack([mo.md("### Where the error lives\n\nOne panel per model, same axes everywhere: the shape of ε(r) "
                     "separates a model that decays too fast from one that has no tail at all."),
               fig_r if fig_r is not None else mo.md("_no ε(r) data yet_")])
    return eps_vs_r, fig_r


@app.cell
def _(mo, results):
    # ------------------------------------------------------------------------------- what failed, and what was skipped
    import json as _json

    def issues(J):
        out = []
        for s, S in (J.get("sets") or {}).items():
            n = S.get("n_prediction_errors")
            if n:
                first = list((S.get("prediction_errors") or {}).values())[:1]
                out.append(f"- `{s}`: {n} failed predictions, e.g. {first[0] if first else ''}")
            ch = S.get("charges")
            if isinstance(ch, dict) and ch.get("skipped"):
                out.append(f"- `{s}`: charge block skipped — {ch['skipped']}")
        for note in (J.get("notes") or [])[:2]:
            out.append(f"- note: {note}")
        return out

    blocks = []
    for k, v in results.items():
        lines = issues(v)
        if lines:
            blocks.append(mo.md(f"**{k}**\n" + "\n".join(lines)))
    mo.vstack([mo.md("## Failures and skipped blocks\n\nA metric that could not be computed says so, which is the "
                     "only way to keep an absent number from being read as a good one."),
               *(blocks or [mo.md("_nothing to report_")])])
    return blocks, issues


@app.cell
def _(mo, results):
    pick = mo.ui.dropdown(options=sorted(results), label="raw result JSON")
    pick
    return (pick,)


@app.cell
def _(mo, pick, results):
    mo.json(results[pick.value]) if pick.value else mo.md("_pick a model above_")
    return


if __name__ == "__main__":
    app.run()
