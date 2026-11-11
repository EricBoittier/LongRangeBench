#!/usr/bin/env python3
"""Tables and figures for a GSCDB137 score (lrbench.gscdb score output).

    python -I gscdb_report.py SCORE.json COVERAGE.json OUTDIR [--name fig_gscdb_smoke] [--title "P0 smoke"]
                              [--style STYLE.py]

Writes OUTDIR/tables.md (per category / distance / subset, both targets, 95 % group-bootstrap CIs, paired vs
PET-OMol-S, PEC curve tables by recomputed closest-contact distance, coverage), OUTDIR/<name>.{png,pdf} and
OUTDIR/macros.tex. Surrogate tables use only entries with a ωB97M-V/def2-TZVPD label; end-to-end tables use every
entry all core models predict. Hollow markers = mismatched comparison (scratch one-seed checkpoints vs pretrained;
oracle with DFT inputs)."""
import importlib.util
import json
import os
import sys

import numpy as np

S, C, O = sys.argv[1:4]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
STYLE = opt("--style", os.environ.get("LRBENCH_PAGE_STYLE", ""))
NAME, TITLE = opt("--name", "fig_gscdb_smoke"), opt("--title", "P0 smoke")
spec = importlib.util.spec_from_file_location("house_style", STYLE)
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
import matplotlib.pyplot as plt

R = json.load(open(S)); cov = json.load(open(C)); os.makedirs(O, exist_ok=True)
SUM = R["summary"]["strata"]; models = R["summary"]["models"]
COL = {"PET-OMol-S": st.MODEL["local"], "R0": st.MODEL["control"], "R1": st.MODEL["ours3"], "R1q": st.MODEL["ours3"],
       "Fa (LES-in-PET)": st.MODEL["ours"], "FQ (LES-in-PET)": st.MODEL["ours2"], "MACE-POLAR-1-M": st.MODEL["polar"],
       "oracle": st.MODEL["oracle"]}
MISM = {"R0": "scratch, 97k structures, one seed", "R1": "scratch, one seed", "R1q": "scratch, one seed, other GPU",
        "oracle": "DFT monomer charges as input"}
col = lambda m: COL.get(m, st.INK)
mism = lambda m: m in MISM
MK = {"R1q": "s", "FQ (LES-in-PET)": "D", "oracle": "^"}
mk = lambda m: MK.get(m, "o")
n_of = lambda c, t: max((SUM[c][m][t]["n"] for m in models if m in SUM[c]), default=0)
CAT = [c for c in ("all", "ion pair", "cation-neutral", "anion-neutral", "neutral PEC", "three-body", "open-shell",
                   "atomic-ion fragment", "molecular fragments only") if c in SUM]
DIS = [c for c in SUM if c.startswith("closest contact") or c.startswith("IHB100x2 ") or c.startswith("set:")
       or c.startswith("unseen")]
f1 = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
ci = lambda c: "" if c is None or any(x is None or np.isnan(x) for x in c) else f" <small>[{f1(c[0])}, {f1(c[1])}]</small>"
TT = {"surrogate": "**Surrogate fidelity: |ΔE_model − ΔE_ωB97M-V/def2-TZVPD| on the published geometry** (labelled entries only).",
      "end_to_end": "**End-to-end: |ΔE_model − ΔE_GSCDB|** (model error plus ωB97M-V functional/basis error; not an ML error)."}


def table(cols, tgt):
    cols = [c for c in cols if n_of(c, tgt)]
    if not cols:
        return ["(no entries)", ""]
    L = ["| model | " + " | ".join(f"{c} (n)" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in models:
        cells = []
        for c in cols:
            s = SUM[c][m][tgt]
            cells.append(f"{f1(s.get('mae'))}{ci(s.get('mae_ci'))} ({s['n']})" if s["n"] else "–")
        L.append(f"| {m}{' †' if mism(m) else ''} | " + " | ".join(cells) + " |")
    if tgt == "end_to_end":
        L.append("| *ωB97M-V/def2-TZVPD − GSCDB (labelled entries)* | " + " | ".join(
            f"{f1(SUM[c]['functional'].get('mae'))}{ci(SUM[c]['functional'].get('mae_ci'))} ({SUM[c]['functional']['n']})"
            if SUM[c]["functional"]["n"] else "–" for c in cols) + " |")
        L.append("| *ωB97M-V/def2-QZVPPD (repository) − GSCDB, all entries* | " + " | ".join(
            f"{f1(SUM[c]['functional_qzvppd'].get('mae'))}{ci(SUM[c]['functional_qzvppd'].get('mae_ci'))} ({SUM[c]['functional_qzvppd']['n']})"
            if SUM[c].get("functional_qzvppd", {}).get("n") else "–" for c in cols) + " |")
    return L + [""]


def paired(cols, tgt):
    k = f"paired_dabs_{tgt}_vs_PET-OMol-S"
    cols = [c for c in cols if n_of(c, tgt)]
    L = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in models:
        if m == "PET-OMol-S":
            continue
        cells = []
        for c in cols:
            d = SUM[c][m].get(k)
            if d is None:
                cells.append("–"); continue
            lo, hi = d["ci"]
            sig = "" if lo is None or np.isnan(lo) else (" ▼" if hi < 0 else " ▲" if lo > 0 else "")
            cells.append(f"{f1(d['mean'])}{ci(d['ci'])}{sig}")
        L.append(f"| {m}{' †' if mism(m) else ''} | " + " | ".join(cells) + " |")
    return L + [""]


s0 = R["summary"]
L = [f"# GSCDB137 {TITLE}: tables", "",
     f"Entries: {s0['n_entries']}; end-to-end population (every core model predicts): {s0['n_common_end_to_end'] if 'n_common_end_to_end' in s0 else s0['n_common']}; "
     f"surrogate population (plus a ωB97M-V/def2-TZVPD label): {s0['n_common']}. meV throughout. Intervals: 95 % percentile "
     "bootstrap over chemical systems / curves (4,000 resamples; whole curves and both IHB100x2 distances of a system "
     "resampled together). n = 1 group: no interval. Entries sharing an ion across systems (CHB6, AHB21 halides) are not "
     "independent, so those intervals are too narrow.", ""]
for tgt in ("surrogate", "end_to_end"):
    L += [TT[tgt], "", "By category:", ""] + table(CAT, tgt) + ["By closest-contact distance (recomputed), subset, distance factor, overlap:", ""] + table(DIS, tgt)
    L += [f"Paired |e| difference vs PET-OMol-S, {tgt.replace('_', '-')} (same entries; negative = better; ▼/▲ interval excludes 0):", ""]
    L += paired(CAT + DIS, tgt)
L += ["† mismatched comparison: " + "; ".join(f"{k}: {v}" for k, v in MISM.items() if k in models) + ".", ""]
b = SUM["all"]["basis"]
if b.get("n"):
    L += [f"Basis check (our ORCA ωB97M-V/def2-TZVPD − the repository's ωB97M-V/def2-QZVPPD, same reactions): "
          f"MAE {f1(b.get('mae'))}{ci(b.get('mae_ci'))} meV, MSE {f1(b.get('mse'))}, max {f1(b.get('maxabs'))} meV (n = {b.get('n')}).", ""]
# curves: across-curve summary + per-point tables for NBC10 and short lists
CS = R.get("curve_summary", {})
for setname, o in CS.items():
    L += [f"**{setname} curves: mean curve MAE and mean |error of the longest increment| (two largest closest-contact "
          "distances), one value per curve, bootstrap over curves.**", "",
          "| model | target | curves | curve MAE | |longest increment| |", "|---|---|---|---|---|"]
    for m, v in o.items():
        for t in ("surrogate", "end_to_end"):
            if v[t]["n_curves"]:
                L.append(f"| {m} | {t} | {v[t]['n_curves']} | {f1(v[t]['curve_mae'])}{ci(v[t]['curve_mae_ci'])} | "
                         f"{f1(v[t]['abs_longest_increment'])}{ci(v[t]['abs_longest_increment_ci'])} |")
    L.append("")
for g, d in R["curves"].items():
    if d["set"] == "S66Rel7" and len(R["curves"]) > 3:
        continue
    L += [f"**{g}**: closest contact {', '.join(f'{v:.2f}' for v in d['d_min'])} Å (source coordinate "
          f"{', '.join(str(x) for x in d['source_x'])}). Signed error per point (meV).", "",
          "| model | target | " + " | ".join(f"{x:.2f} Å" for x in d["d_min"]) + " | curve MAE | longest increment |",
          "|---|---|" + "---|" * (len(d["d_min"]) + 2)]
    for m, v in d["models"].items():
        for t in ("surrogate", "end_to_end"):
            if any(x is not None for x in v[t]):
                L.append(f"| {m} | {t} | " + " | ".join(f1(x) for x in v[t]) +
                         f" | {f1(v['curve_mae_' + t])} | {f1(v['longest_increment_' + t])} |")
    if any(x is not None for x in d["dft"]):
        L.append("| ωB97M-V/TZVPD − GSCDB | | " + " | ".join(f1(None if a is None else a - r) for a, r in zip(d["dft"], d["ref"])) + " | | |")
    L.append("")
L += ["## Coverage (entries whose every molecule is inside the model's elements, charge and multiplicity range)", "",
      "| model | " + " | ".join(next(iter(cov.values()))["subsets"]) + " | source |",
      "|---|" + "---|" * (len(next(iter(cov.values()))["subsets"]) + 1)]
for m, d in cov.items():
    L.append(f"| {m} | " + " | ".join(f"{v['covered']}/{v['n']}" + (" (" + ", ".join(sorted(v['reasons'])) + ")" if v["reasons"] else "")
                                       for v in d["subsets"].values()) + f" | {d['meta'].get('source', '')} |")
open(f"{O}/tables.md", "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------------------------------ figure
st.setup()
FC = [c for c in ("ion pair", "cation-neutral", "anion-neutral", "neutral PEC", "three-body", "atomic-ion fragment",
                  "molecular fragments only") if c in SUM]
fig, axs = plt.subplots(1, 3, figsize=(st.W2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1], wspace=0.55))
for ax, tgt, ttl in ((axs[0], "surrogate", "a  surrogate (labelled entries)"), (axs[1], "end_to_end", "b  end-to-end: vs GSCDB")):
    ylab = []
    for j, c in enumerate(FC):
        for i, m in enumerate(models):
            s = SUM[c][m][tgt]; y = j * (len(models) + 1) + i
            if not s["n"]:
                continue
            lo, hi = s["mae_ci"]
            st.point(ax, s["mae"], y, None if np.isnan(lo) else lo, None if np.isnan(hi) else hi, color=col(m),
                     mismatched=mism(m), marker=mk(m))
        fk = "functional_qzvppd" if tgt == "end_to_end" else None
        if fk and SUM[c].get(fk, {}).get("n"):
            fm = SUM[c][fk]["mae"]
            ax.plot([fm, fm], [j * (len(models) + 1) - 0.6, j * (len(models) + 1) + len(models) - 0.4], color=st.G2,
                    lw=0.8, ls=(0, (2, 1)))
        ylab.append((j * (len(models) + 1) + (len(models) - 1) / 2, f"{c}\n(n={n_of(c, tgt)})"))
    ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([s for _, s in ylab]); ax.invert_yaxis()
    ax.minorticks_off(); ax.set_xscale("log"); ax.set_xlabel("MAE (meV)"); st.title(ax, ttl)
axs[1].set_yticklabels([f"n={n_of(c, 'end_to_end')}" for c in FC], fontsize=5.8)
st.note(axs[1], "dashed grey: ωB97M-V/def2-QZVPPD (repository) − GSCDB", y=-0.15)
handles = [plt.Line2D([], [], marker=mk(m), color=col(m), ms=3.6,
                      **(st.mism(col(m)) if mism(m) else dict(ls="none", mfc=col(m), mec="white"))) for m in models]
fig.legend(handles, [m + (" †" if mism(m) else "") for m in models], loc="lower center", ncol=4, bbox_to_anchor=(0.4, -0.27))
# panel c: |error| vs recomputed closest contact on the PEC curves (NBC10 if present, else the S66 curve):
# median per 0.5 Å bin, and the functional floor (repository ωB97M-V/QZVPPD − GSCDB) in grey
ax = axs[2]
pecs = [g for g, d in R["curves"].items() if d["set"] == "NBC10"] or list(R["curves"])
tgt = "surrogate" if all(any(x is not None for x in R["curves"][g]["dft"]) for g in pecs) else "end_to_end"
rowsby = {r["complex"]: r for r in R["rows"]}
pts = [r for r in R["rows"] if r["group"] in pecs]
dist = lambda r: r["d_min_source"] if r["d_min_source"] is not None else r["d_min_cluster"]
edges = np.arange(2.0, 10.6, 0.5)
def binned(vals):
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = [a for d, a in vals if lo <= d < hi]
        if v:
            x.append((lo + hi) / 2); y.append(np.median(v))
    return x, y
fl = binned([(dist(r), abs(r["functional_qzvppd"])) for r in pts if r.get("functional_qzvppd") is not None])
ax.plot(*fl, color=st.G2, lw=0.9, ls=(0, (2, 1)), zorder=1)
for m in models:
    vals = [(dist(r), abs(r["model"][m][tgt])) for r in pts if r["model"][m][tgt] is not None]
    if not vals:
        continue
    x, y = binned(vals)
    ax.plot(x, y, marker=mk(m), color=col(m), ms=2.6, lw=0.8, ls=(0, (2, 1)) if mism(m) else "-",
            **(dict(mfc="white") if mism(m) else dict(mfc=col(m), mec="white", mew=0.3)))
ax.set_yscale("log"); st.vline(ax, 8.0, "PET r_c 8 Å", ha="right")
ax.set_xlabel("closest atom–atom distance (Å)")
ax.set_ylabel(f"median |{'ΔE_model − ΔE_ωB97M-V' if tgt == 'surrogate' else 'ΔE_model − ΔE_GSCDB'}| (meV)")
st.title(ax, f"c  {R['curves'][pecs[0]]['set'] if pecs else ''} curves vs distance ({tgt.replace('_', '-')})")
st.save(fig, NAME, out=O)
print(f"{O}/tables.md", f"{O}/{NAME}.png")

# ------------------------------------------------------------------------------------------------------ macros
mac = []
for m in models:
    tag = "".join(ch for ch in m if ch.isalnum())
    for c in CAT + ["all"]:
        ctag = "".join(ch for ch in c.title() if ch.isalpha())
        for t, tt in (("surrogate", "Sur"), ("end_to_end", "EtE")):
            mac.append(f"\\newcommand{{\\gscdb{tag}{ctag}{tt}}}{{{f1(SUM[c][m][t].get('mae'))}}}")
mac.append(f"\\newcommand{{\\gscdbNcommon}}{{{s0['n_common']}}}")
open(f"{O}/macros.tex", "w").write("\n".join(mac) + "\n")
