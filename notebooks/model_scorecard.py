# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
# ]
# ///
"""Model scorecard: pick one model and see every number the benchmark and the PET-OMol-LR program hold for it.

    marimo edit notebooks/model_scorecard.py
    LRB_SCORECARD_MODEL=P python notebooks/model_scorecard.py          # headless, one model
    LRB_BREAKDOWN=exclude LRB_BREAKDOWN_RULE=physics python notebooks/model_scorecard.py   # breakdown toggle
    marimo export html notebooks/model_scorecard.py -o out.html -- --model C0

Data: ``data/lr_program/`` next to this file (shared with long_range_diagnostics.py), written by
``tools/build_lr_program_data.py`` (``bash tools/rebuild_lr_program.sh`` re-reads the score files); loaders and the
chemiscope assembly are in ``lrb_data.py`` (also next to this file). Every table is a marimo table: sortable, filterable, and
downloadable as CSV from its footer.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Model scorecard")


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(mo.notebook_dir()))
    import lrb_data as L

    DDIR = L.data_dir(mo.notebook_dir())
    D = L.load(DDIR)
    if not D["index"]:
        raise FileNotFoundError(f"no data bundle at {DDIR}; run bash tools/rebuild_lr_program.sh")
    MODELS = {m["id"]: m for m in D["index"]["models"]}
    return D, DDIR, L, MODELS, mo, np, os, pd


@app.cell
def _(L, MODELS, mo, os):
    _args = mo.cli_args() or {}
    DEFAULT = str(_args.get("model") or os.environ.get("LRB_SCORECARD_MODEL") or "P")
    if DEFAULT not in MODELS:
        raise ValueError(f"unknown model {DEFAULT!r}; known: {sorted(MODELS)}")
    _opts = {}
    for _m in MODELS.values():
        _tag = {"benchmark": "benchmark", "program": "program", "program+benchmark": "program + benchmark"}.get(_m["kind"], "program, MD only")
        _opts[f"{_m['name']}  [{_tag}]"] = _m["id"]
    _inv = {v: k for k, v in _opts.items()}
    model_pick = mo.ui.dropdown(options=_opts, value=_inv[DEFAULT], label="model", searchable=True, full_width=True)
    mv_switch = mo.ui.switch(value=os.environ.get("LRB_SCORECARD_MV") == "1", label="multivalent systems (|q| >= 2) count in scores")
    _bd = L.bd_config(_args)
    bd_mode = mo.ui.dropdown(options=list(L.BD_MODES), value=_bd["mode"], label="dielectric-breakdown frames")
    bd_rule = mo.ui.dropdown(options=list(L.BD_RULES), value=_bd["rule"], label="breakdown rule")
    bd_thr = mo.ui.number(start=0.01, stop=1.0, step=0.01, value=_bd["thr"], label="physics threshold |fragment Löwdin - formal| (e)")
    mo.vstack([mo.md("# Model scorecard"), mo.hstack([model_pick, mv_switch], widths=[3, 2]),
               mo.hstack([bd_mode, bd_rule, bd_thr], justify="start", gap=1.5, wrap=True)])
    return bd_mode, bd_rule, bd_thr, model_pick, mv_switch


@app.cell
def _(D, L, MODELS, bd_mode, bd_rule, bd_thr, mo, model_pick, mv_switch, pd):
    M = MODELS[model_pick.value]
    MID = M["id"]
    MV = mv_switch.value
    BD = dict(mode=bd_mode.value, rule=bd_rule.value, thr=float(bd_thr.value))
    BD_STATE = L.bd_state(BD["mode"], BD["rule"], BD["thr"])
    PROG = D["prog"]["labels"]
    LABS = [l for l in M.get("labels", []) if l in PROG]
    MDNAME = M.get("mdstab")

    def not_run(what, why=""):
        return mo.callout(mo.md(f"**{what}: not run for this model.** {why}"), kind="neutral")

    def table(df, label=None, page=15):
        if df is None or len(df) == 0:
            return None
        return mo.ui.table(df, page_size=page, selection=None, label=label or "", show_download=True)

    _files = pd.DataFrame([dict(file=l, seed=PROG[l]["seed"], weights=PROG[l]["weights"], hardware=PROG[l]["host"],
                                steps=PROG[l]["step"]) for l in LABS])
    _have = []
    _intro = [mo.md(f"## {M['name']}\n\n{M['recipe']}")]
    if LABS:
        _n_seed = len({PROG[l]['seed'].replace('13r', '13') for l in LABS if 'mid-run' not in PROG[l]['step']})
        _intro.append(mo.md(f"Scored files: {len(LABS)}, over {_n_seed} seed(s). Seed 13 ran on H100 for the "
                            "seed-13 screens; seed 13r is its same-seed replicate on GH200. Replicates of one "
                            "seed on two machines differed by up to 2.1x in the NH4+Gdm+ overshoot, so 13 and 13r are "
                            "not counted as two seeds. Files marked 'mid-run' are step-3,000 checkpoints and are not "
                            "in any seed summary. Four seeds are needed for a claim; fewer is a screen."))
        _intro.append(table(_files, "scored files"))
    _intro.append(mo.callout(mo.md(L.MV_NOTE + f" Current state: **{L.mv_state(MV)}**."), kind="warn" if MV else "info"))
    _intro.append(mo.callout(mo.md(L.BD_NOTE + f" Current state: **{BD_STATE}**."),
                             kind="info" if BD["mode"] == "include" else "warn"))
    mo.vstack([x for x in _intro if x is not None])
    return BD, BD_STATE, LABS, M, MDNAME, MID, MV, PROG, not_run, table


@app.cell
def _(DDIR, L, M, MID, MV, mo, pd, table):
    # running work and files that have no number yet; the 1-D scan block (AL vs PET-OMol-S, P, C0)
    def _status():
        X = L.program_extra(DDIR)
        st, S = X["status"], X["scans"]
        out = []
        if st:
            mine = [r for r in st.get("running", []) if MID in ("AL", "PAL") and r["name"].split()[0] == MID]
            for r in mine:
                if r.get("result"):
                    out.append(mo.callout(mo.md(f"**{r['name']}: scored.** {r['result']}"), kind="info"))
                    continue
                out.append(mo.callout(mo.md(f"**{r['name']}: {r['state']}.** {r['what']} {r['eta']}. Files not yet "
                                            f"scored: {', '.join(r['pending']) or 'none'}. Rows for these files show "
                                            "'not yet scored' until the bundle is rebuilt."), kind="warn"))
            pend = {k: v for k, v in X["pending"].items() if k.split()[0] == MID}
            if pend and not mine:
                out.append(mo.callout(mo.md("**Not yet scored:** " + "; ".join(f"{k}: {', '.join(v)}" for k, v in pend.items())),
                                      kind="warn"))
            md = st.get("md_stability") or {}
            run = pd.DataFrame([{"work": r["name"], "state": r["state"], "expected": r["eta"] or "–",
                                 "not yet scored": ", ".join(r["pending"]) or "–", "what": r["what"]}
                                for r in st.get("running", [])])
            _md = ""
            if md:
                a, b = md["without Mg2+"], md["Mg2+ apart"]
                _md = (f"MD stability, every program arm and the base model ({a['n'] + b['n']} runs of ion-water clusters, "
                       f"250-400 K): without Mg²⁺ {a['n']} runs, {a['stable']} stable / {a['flagged']} flagged / "
                       f"{a['unstable']} unstable; Mg²⁺ (multivalent, shown apart) {b['n']} runs, {b['stable']} / "
                       f"{b['flagged']} / {b['unstable']}.")
            out.append(mo.accordion({f"Running work and files not yet scored (bundle built {st.get('built', '?')})":
                                     mo.vstack([mo.md(st.get("not_yet_scored", "")), table(run, "running"), mo.md(_md)])}))
        scan_ids = {"pet-omol-s": "PET-OMol-S", "C0": "C0", "P": "P", "AL": "AL"}
        if S:
            head = mo.md("### Scan quality: 1-D scans against DFT\n\n" + S["note"] + " Evidence: " + S["evidence"] + ".")
            if MID in scan_ids:
                rows = L.scan_rows(S, include_mv=MV)
                df = pd.DataFrame(rows)
                out += [head, table(df, f"scan quality; this model's column is {scan_ids[MID]}", page=20)]
            else:
                out += [mo.md("### Scan quality: 1-D scans against DFT\n\n**Not yet scanned** for this model: the "
                              "1-D scans cover PET-OMol-S, C0, P and AL (seed 13) only.")]
            import base64 as _b64
            import io as _io
            figs = {f"Figure: {f}": mo.vstack([mo.image(_io.BytesIO(_b64.b64decode(v["png"])), width="100%"),
                                               mo.md(f"<small>{v['caption']}</small>")])
                    for f, v in S.get("figures", {}).items()}
            if figs:
                out.append(mo.accordion(figs))
        _P = L._json(str(DDIR), "program.json") or {}
        if _P.get("evidence_limits"):
            out.append(mo.accordion({"Evidence limits": mo.md("\n".join(f"- {x}" for x in _P["evidence_limits"]))}))
        if MID in ("C0", "P") and _P.get("typeheld_pdiff_text"):
            out.append(mo.md("**Type-held-out gate, P − C0:** " + _P["typeheld_pdiff_text"]))
        LB = X.get("ladder_large_n")
        lb_arm = {"AL": "AL", "PAL": "PAL", "P": "P", "C0": "C0"}.get(MID)
        if LB and lb_arm and lb_arm in LB.get("arms", {}):
            out += [mo.md("### Ion–water ladders at large n (1,000–4,000 waters)\n\n" + "\n\n".join(LB.get("text") or [])
                          + "\n\n<small>" + LB["note"] + "</small>"),
                    table(pd.DataFrame(L.ladder_big_rows(LB)), "large-n ladders: absolute error (draft limit) and slope (diagnostic), every arm run (this model: " + lb_arm + ")")]
        elif LB and MID not in ("pet-omol-s",) and M.get("kind", "").startswith("program"):
            out.append(mo.md("### Ion–water ladders at large n\n\n**Not run** for this model (large-n ladders exist for "
                             "AL, PAL, P and C0)."))
        return mo.vstack(out) if out else mo.md("")
    _status()
    return


@app.cell
def _(DDIR, L, mo):
    # the 1-D scan viewers: one at a time from a list (a cell re-run builds the iframe; works the same in a live
    # kernel, on molab and in the WebAssembly export)
    SCAN_VIEWS = {v["key"]: v for v in L.program_views(DDIR, "lrp-scan-")}
    scan_pick = mo.ui.dropdown(options={"(none)": "", **{v["title"]: k for k, v in SCAN_VIEWS.items()}}, value="(none)",
                               label="1-D scan structures", full_width=True)
    return SCAN_VIEWS, scan_pick


@app.cell
def _(L, SCAN_VIEWS, mo, scan_pick):
    _v = SCAN_VIEWS.get(scan_pick.value)
    (mo.vstack([scan_pick] + ([mo.md(_v["prose"]), mo.iframe(L.page_blob(_v["blob"], _v.get("settings")), height="680px")]
                              if _v else [])) if SCAN_VIEWS else mo.md(""))
    return


@app.cell
def _(BD, BD_STATE, L, LABS, MV, PROG, mo, not_run, pd, table):
    def _sec():
        _rows = []
        for l in LABS:
            r = PROG[l]
            if not r.get("val500"):
                continue
            for split, v in r["val500"].items():
                off = (r.get("val500_les_off") or {}).get(split, {})
                mvs = "multivalent only" in split
                excl = "no multivalent" in split
                if MV and excl:
                    continue           # switch on: the full splits are the score
                if not MV and not excl and not mvs:
                    continue           # switch off: the multivalent-free splits are the score
                _rows.append(dict(file=l, seed=r["seed"], weights=r["weights"], step=r["step"], host=r["host"],
                                  split=split.replace(" | no multivalent", ""),
                                  **{"max |q|": ">= 2" if mvs else ("<= 1" if excl else "any"),
                                     "counts in score": ((not mvs) or MV) and BD["mode"] != "only"},
                                  **{"F MAE meV/A": round(v["F"][0], 2), "F 95% lo": round(v["F"][1], 2), "F 95% hi": round(v["F"][2], 2),
                                     "E MAE meV/atom": round(v["E"][0], 2), "E 95% lo": round(v["E"][1], 2), "E 95% hi": round(v["E"][2], 2),
                                     "n frames": v["F"][3], "F MAE LES off": round(off["F"][0], 2) if off else None}))
        if not _rows:
            return not_run("val500 accuracy (500 OMol25 validation frames, wB97M-V/def2-TZVPD forces)",
                           "No val500 output exists for this model in the program score files.")
        df = pd.DataFrame(_rows)
        final = df[(df.split == "all") & ~df.step.str.contains("mid-run")]
        ema = final[final.weights.str.startswith("EMA") | final.weights.str.startswith("raw (no")]
        nmv = df[df["max |q|"] == ">= 2"]["n frames"].max() if (df["max |q|"] == ">= 2").any() else 0
        txt = (f"**val500 force MAE** ({L.mv_state(MV)}; {nmv} of 500 frames contain a species with |q| >= 2), "
               f"EMA files (raw where the arm has no EMA), final step: {L.seed_line(ema, 'F MAE meV/A')}; "
               "the value per file is atom-weighted over the frames "
               "with a 95 % percentile bootstrap over frames (asymmetric). The gate is 42.0 meV/A (amendment A28a). "
               "Energy MAE is per atom against the element-referenced residual energy. The charged / neutral split is by "
               "the frame's total charge. LES off = the same weights with the long-range gate at 0. Multivalent = a "
               "species with |q| >= 2 by covalent components and formal charges, with |Q| used where the formal-charge "
               "column does not sum to the frame charge (a heuristic).")
        _bdm = ("no electrode frames in this set" if BD["rule"] == "charge" else
                "no charge labels in val500 (single-fragment frames): every frame counted, none flagged")
        txt += (f" **Breakdown: {BD_STATE}.** {_bdm}; " + ("in 'only' mode no val500 frame counts (n = 0), so these "
                "rows are shown for reference and are not a score." if BD["mode"] == "only" else
                "these precomputed rows include every frame in every breakdown state."))
        print(f"[breakdown] val500 | {BD['rule']} | {BD['mode']} | flagged 0 | {_bdm}")
        mvdf = df[df["max |q|"] == ">= 2"]
        out = [mo.md("### Accuracy: val500"), mo.md(txt), table(df[df["max |q|"] != ">= 2"], "val500 per file and split (scored)")]
        if len(mvdf):
            out += [mo.md("**Multivalent frames only**" + (" (counted in the 'all' rows above, switch on)." if MV else
                                                             " (excluded: multivalent; not in the rows above).")),
                    table(mvdf, "val500, multivalent frames")]
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(LABS, PROG, mo, not_run, pd, table):
    def _sec():
        rows = []
        for l in LABS:
            c = PROG[l].get("cells")
            if not c:
                continue
            for cell, r in c.items():
                rows.append(dict(file=l, seed=PROG[l]["seed"], weights=PROG[l]["weights"], step=PROG[l]["step"], cell=cell,
                                 counted="held out, reported apart" if cell.startswith("K+") else "rule 4 (4 scorable cells)",
                                 **{"max |q|": 1},
                                 **{"attractive %": round(r["att"], 1), "median / Coulomb": round(r["med_over_c"], 2),
                                    "> 1.5x Coulomb %": round(r["over15"], 1), "worst eV/A": round(r["worst"], 3),
                                    "pass": r["pass"]}))
        if not rows:
            return not_run("Seam sign set (rule 4)", "No orientation-search output exists for this model.")
        df = pd.DataFrame(rows)
        cnt = (df[df.counted.str.startswith("rule")].groupby("file")["pass"].sum().astype(int).rename("cells passing of 4")
               .reset_index())
        kg = df[df.cell.str.startswith("K+")][["file", "pass"]].rename(columns={"pass": "K+Gdm+ 8 A pass"})
        cnt = cnt.merge(kg, on="file", how="left")
        txt = ("Rule 4 sign set: 2,000 random orientations per cell (Gdm+Gdm+ 8 and 9 A, NH4+NH4+ 8 A, NH4+Gdm+ 7.5 A), "
               "net force on ion B along the COM axis compared with the Coulomb oracle k/d^2 (an oracle reference, not DFT). "
               "A cell passes with <= 10 % attractive orientations, median <= 1.5x Coulomb and <= 10 % above 1.5x. The "
               "denominator is 4 scorable cells; NH4+NH4+ 9 A is not in the search set and cannot be scored; K+Gdm+ 8 A "
               "is held out of P's training rows and is reported apart, never counted. All cells are monovalent "
               "(max |q| = 1), so the multivalent switch does not change this count.")
        return mo.vstack([mo.md("### Seam sign set"), mo.md(txt), table(cnt, "cells passing per file"),
                          table(df, "cells per file")])
    _sec()
    return


@app.cell
def _(D, L, LABS, PROG, MV, mo, not_run, pd, table):
    def _sec():
        rows = []
        for l in LABS:
            lad = PROG[l].get("ladder") or {}
            for where, ions in lad.items():
                for ion, v in ions.items():
                    rows.append(dict(file=l, seed=PROG[l]["seed"], weights=PROG[l]["weights"], step=PROG[l]["step"],
                                     centres_LES=where, ion=ion, **{"max |q|": L.ION_Q.get(ion, 1)},
                                     scored="scored" if (L.ION_Q.get(ion, 1) < 2 or MV) else "excluded: multivalent",
                                     **{"slope eV/water": v["slope"], "within +-0.02": abs(v["slope"]) <= 0.02,
                                        "error at n = 32 / 64 / 96 (eV)": v["err_32_64_96"], "tail amplitude A": v["A"]}))
        if not rows:
            return not_run("Cluster-ladder slopes (rule 1)", "No ladder output exists for this model.")
        df = pd.DataFrame(rows)
        fin = df.step.str.contains("mid-run") | ~(df.weights.str.startswith("EMA") | df.weights.str.startswith("raw (no") | df.weights.str.startswith("pretrained"))
        sc = df[(df.scored == "scored") & df.centres_LES.str.startswith("held-out, LES on") & ~fin]
        npass = int(sc["within +-0.02"].sum())
        shown = df[df.scored == "scored"]
        apart = df[df.scored != "scored"]
        ions = "Na+, Cl- and Mg2+" if MV else "Na+ and Cl- (Mg2+ shown apart)"
        txt = (f"Slope of the vertical ion-binding-energy error dE_model(n) - dE_DFT(n) against cluster size, n >= 24 waters, "
               f"held-out centres, gate +-0.02 eV per water; {L.mv_state(MV)}. Scored ions: {ions}. On held-out "
               f"centres with LES on, {npass} of {len(sc)} scored slopes over this model's {sc.file.nunique()} final EMA files "
               "(raw where the arm has no EMA; H100 seed 13 and GH200 13r are both listed, they are one seed) are "
               f"inside the gate. DFT tail amplitudes for reference: {D['prog']['ladder_dft_tail_A']}. "
               "A slope cannot see a constant offset in dE: a constant-offset control passes the large-n slope rules for every "
               "offset up to 30 eV, so read the slope with the error columns beside it (ladder v2 draft, E1). "
               "The reference is wB97M-V/def2-TZVPD energies only (no DFT forces exist for the ladder).")
        out = [mo.md("### Ladder slopes"), mo.md(txt), table(shown, "ladder slopes per file (scored)")]
        if len(apart):
            out += [mo.md("**Excluded: multivalent (Mg2+).** Shown for reference; not in any count above."),
                    table(apart, "ladder slopes, excluded: multivalent")]
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(LABS, PROG, mo, not_run, pd, table):
    def _sec():
        keys = ["nacl1m interior Na shell: max abs per-ion error", "nacl1m interior Cl shell: max abs per-ion error",
                "nacl1m interior Na shell: model mean", "nacl1m interior Na shell: DFT mean",
                "nacl1m interior Cl shell: model mean", "nacl1m interior Cl shell: DFT mean",
                "nacl1m per-atom q RMSE vs Lowdin", "Na + whole waters (O within 4A): model", "Na + whole waters: Lowdin",
                "Cl + whole waters (O within 4A): model", "Cl + whole waters: Lowdin", "frag RMSE"]
        rows = []
        for l in LABS:
            s = PROG[l]["scorecard"]
            if not any(k in s for k in keys):
                continue
            r = dict(file=l, seed=PROG[l]["seed"], weights=PROG[l]["weights"], step=PROG[l]["step"], **{"max |q|": 1})
            r.update({k: (round(s[k], 3) if isinstance(s.get(k), float) else s.get(k)) for k in keys})
            na, cl = s.get(keys[0]), s.get(keys[1])
            r["shell gate (<= 0.2 e, Na and Cl)"] = (na is not None and cl is not None and na <= 0.2 and cl <= 0.2)
            rows.append(r)
        if not rows:
            return not_run("Ion shells (rule 5)", "No labelled cut-out output exists for this model.")
        txt = ("Ion shells: the 8 DFT-labelled 1 m NaCl cut-outs (242-283 atoms, wB97M-V/def2-TZVPD Löwdin charges), "
               "interior ions; ion plus every water whose O is within 4 A. The gate is a largest per-ion error <= 0.2 e "
               "for Na+ and Cl- (max |q| = 1; the cut-outs hold no multivalent species, so the switch does not change it). Model charges are the LES latent charges, "
               "compared with Löwdin, which is one DFT partition among several.")
        return mo.vstack([mo.md("### Ion shells and charges vs DFT (static cut-outs)"), mo.md(txt),
                          table(pd.DataFrame(rows), "ion shells per file")])
    _sec()
    return


@app.cell
def _(LABS, PROG, mo, not_run, pd, table):
    def _sec():
        rows = []
        for l in LABS:
            for tag, key in (("LES on", "setB"), ("LES off", "setB_les_off")):
                for pair, v in (PROG[l].get(key) or {}).items():
                    rows.append(dict(file=l, seed=PROG[l]["seed"], weights=PROG[l]["weights"], step=PROG[l]["step"], LES=tag,
                                     pair=pair, **{"max |q|": 1 if pair != "H2O_H2O" else 0, "F MAE meV/A": round(v[0], 2), "95% lo": round(v[1], 2),
                                                   "95% hi": round(v[2], 2), "n frames": v[3]}))
        if not rows:
            return not_run("Pairs gate (set B, r >= 8 A)", "No set-B output exists for this model.")
        txt = ("Pairs gate: held-out set-B rigid pair scans (K+MeCN, NH4+Cl-, NH4+NH4+, H2O.H2O; 6 orientations), DFT "
               "forces at COM distance >= 8 A, atom-weighted MAE with a frame bootstrap. Caveat stated with every P-family "
               "number: K+MeCN, NH4+Cl- and NH4+NH4+ are pair types present in P's training rows (different geometries, "
               "oracle labels), so this gate is in-distribution for P, A0 and A1. No pair is multivalent (max |q| <= 1).")
        return mo.vstack([mo.md("### Pairs gate"), mo.md(txt), table(pd.DataFrame(rows), "set-B pairs per file")])
    _sec()
    return


@app.cell
def _(D, L, LABS, MID, MV, PROG, mo, not_run, pd, table):
    def _sec():
        agg = D["typeheld_agg"] or {}
        files = [l for l in LABS if l in agg] + (["pet-omol-s"] if MID == "pet-omol-s" and "pet-omol-s" in agg else [])
        AGG_IN, AGG_EX = "all types incl. Mg2+", "K+Gdm+ and Gdm+I- (Mg2+ excluded)"
        rows = []
        for l in files:
            for tag in ("on", "off"):
                for k, v in (agg[l].get(tag) or {}).items():
                    if k == AGG_IN:
                        q, role = 2, ("score aggregate" if MV else "aggregate incl. multivalent (not the score)")
                    elif k == AGG_EX:
                        q, role = 1, ("score aggregate" if not MV else "aggregate without multivalent (not the score)")
                    else:
                        q, role = L.pair_maxq(k), "pair type"
                    rows.append(dict(file=l, seed=PROG.get(l, {}).get("seed", "-"),
                                     weights=PROG.get(l, {}).get("weights", "pretrained (GGUF export)"), LES=tag,
                                     pair_types=k, role=role, **{"max |q|": q,
                                     "counts in score": (q < 2) or MV,
                                     "F MAE meV/A": round(v["F"][0], 2), "95% lo": round(v["F"][1], 2), "95% hi": round(v["F"][2], 2),
                                     "n frames": v["F"][3], "zero-force MAE": round(v["zero_force"], 2)}))
        if not rows:
            return not_run("Type-held-out charged-pair DFT gate", "This model was not scored on the 17 Nov gate.")
        df = pd.DataFrame(rows)
        head = df[(df.role == "score aggregate") & (df.LES == "on")]
        agg_txt = "; ".join(f"{r['file']} {r['F MAE meV/A']} [{r['95% lo']}, {r['95% hi']}]" for _, r in head.iterrows())
        txt = ("Type-held-out gate: charged pair types absent from every training row (K+Gdm+, Gdm+I-, Gdm+Mg2+), 6 "
               "orientations x 9 COM distances 8-14 A, wB97M-V/def2-TZVPD forces. "
               f"Score aggregate ({L.mv_state(MV)}; over {'all three types' if MV else 'K+Gdm+ and Gdm+I- only'}), "
               f"force MAE at r >= 8 A, LES on, meV/A: {agg_txt}. Intervals are frame bootstraps and are narrower than a "
               "bootstrap over the 6 orientation scans would be; the zero-force MAE is the score of a model with no force.")
        main = df[df["counts in score"]]
        apart = df[~df["counts in score"]]
        out = [mo.md("### Type-held-out charged-pair gate"), mo.md(txt), table(main, "type-held-out gate per file (scored)")]
        if len(apart):
            out += [mo.md("**Excluded: multivalent (Gdm+Mg2+, max |q| = 2).** No arm learns this pair: its force MAE is at "
                          "or above the zero-force floor."), table(apart, "type-held-out gate, excluded: multivalent")]
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(D, M, MDNAME, MID, mo, not_run, pd, table):
    def _sec():
        rows = [r for r in D["speed"]["rows"] if r["model"] in (MID, MDNAME)]
        md = [r for r in D["md"]["runs"] if MDNAME and r["model"] == MDNAME and r.get("ms_step")]
        out = [mo.md("### Speed")]
        if rows:
            out.append(table(pd.DataFrame(rows), "speed numbers (cited source per row)"))
        if md:
            df = pd.DataFrame(md).groupby(["natoms", "gpu"])["ms_step"].agg(["median", "min", "max", "count"]).reset_index()
            df.columns = ["atoms", "GPU", "ms/step median", "min", "max", "n runs"]
            out.append(mo.md("MD step time from the stability runs (fp32, ragged, no TF32; one model per GPU):"))
            out.append(table(df.round(2), "MD ms/step"))
        cons = (D["prog"].get("cons") or {}).get(MID)
        if cons:
            out.append(mo.md("NVE lane ms/step (49-atom frame, RTX 4070 Ti SUPER shared with another scorer, so the "
                             "absolute values are contention-affected): " +
                             "; ".join(f"seed {s} {v['ms_step'][1]:.1f}" for s, v in cons.items())))
        if len(out) == 1:
            return not_run("Speed", "No measured speed for this model in the program files.")
        return mo.vstack([x for x in out if x is not None])
    _sec()
    return


@app.cell
def _(D, L, MDNAME, MV, mo, not_run, pd, table):
    def _sec():
        runs = [r for r in D["md"]["runs"] if MDNAME and r["model"] == MDNAME]
        if not runs:
            return not_run("MD stability (open ion + water clusters, 250-400 K)",
                           "The stability screen covered the route-B arms up to 15 Nov and PET-OMol-S; it was stopped "
                           "on 16 Nov before C0, P, C0H, A0, A1 and CEIL existed.")
        df = pd.DataFrame(runs)
        df["max |q|"] = [L.ION_Q.get(s, 1) for s in df.sys]
        df["counts in score"] = (df["max |q|"] < 2) | MV

        def grid(use):
            g = []
            for (s, nw), sub in use.groupby(["sys", "nw"]):
                row = {"system": f"{s} + {nw} H2O" if s != "W" else f"water {nw}", "max |q|": int(sub["max |q|"].iloc[0])}
                for T in (250, 300, 350, 400):
                    t = sub[sub.T0 == T]
                    row[f"{T} K"] = ("not run" if t.empty else
                                     f"{(t.verdict == 'stable').sum()} stable / {(t.verdict == 'flagged').sum()} flagged / "
                                     f"{(t.verdict == 'unstable').sum()} unstable of {len(t)}")
                g.append(row)
            return pd.DataFrame(g)
        sc = df[df["counts in score"]]
        ap = df[~df["counts in score"]]
        txt = (f"{L.mv_state(MV)}: {len(sc)} runs count, {len(ap)} multivalent (Mg2+) runs are shown apart. Stable "
               f"{int((sc.verdict == 'stable').sum())}, flagged {int((sc.verdict == 'flagged').sum())}, unstable "
               f"{int((sc.verdict == 'unstable').sum())} of the counted runs. Protocol: SVR NVT, tau 100 fs, dt 0.5 fs, 5 ps "
               "warm-up and 20 ps production, fp32, 1 or 2 start configurations per cell, so each cell is a screen. Drift "
               "is the conserved-quantity slope with its 95 % interval.")
        cols = ["sys", "nw", "T0", "seed", "natoms", "max |q|", "counts in score", "verdict", "flags", "fail",
                "drift_meV_atom_ps", "drift_ci", "T_mean", "t_prod_ps", "fmax_max", "nshell_first", "nshell_last",
                "broken_frames", "ms_step", "gpu"]
        use = df[cols].copy(); use["flags"] = use["flags"].astype(str); use["drift_ci"] = use["drift_ci"].astype(str)
        inc = [r for r in D["md"]["incomplete"] if r["model"] == MDNAME]
        out = [mo.md("### MD stability"), mo.md(txt), table(grid(sc), "stability grid (counted)")]
        if len(ap):
            out += [mo.md("**Excluded: multivalent (Mg2+ clusters).**"), table(grid(ap), "stability grid, excluded: multivalent")]
        out.append(table(use.round(4), "MD runs (all, tagged)"))
        if inc:
            out.append(mo.md(f"{len(inc)} further runs have no summary (killed by wall time or not started): " +
                             ", ".join(r["name"] for r in inc)))
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(D, MDNAME, MV, mo, not_run, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, REF, BAND = "#1f3a5f", "#8a8a8a", "#9db7d5"

    def _rdf():
        R = np.array(D["md"]["rdf_r"])
        rdf = D["md"]["rdf"]
        if not MDNAME or not any(k.startswith(MDNAME + "|") for k in rdf):
            return not_run("RDFs", "No MD trajectories for this model.")
        ions = ["Na", "Cl", "Mg"]
        cols = [(48, 250), (48, 300), (48, 350), (48, 400), (200, 300)]
        fig, axs = plt.subplots(len(ions), len(cols), figsize=(9.5, 1.9 * len(ions)), sharex=True, sharey="row")
        axs = np.atleast_2d(axs)
        for i, ion in enumerate(ions):
            for j, (nw, T) in enumerate(cols):
                ax = axs[i, j]
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ref = rdf.get(f"PET-OMol-S|{ion}|{nw}|{T}", {}).get("ionO")
                cur = rdf.get(f"{MDNAME}|{ion}|{nw}|{T}", {}).get("ionO")
                if ref and MDNAME != "PET-OMol-S":
                    ax.plot(R, ref["mean"], color=REF, lw=1.0, ls="--", label="PET-OMol-S (reference model)")
                if cur:
                    ax.fill_between(R, cur["lo"], cur["hi"], color=BAND, lw=0)
                    ax.plot(R, cur["mean"], color=INK, lw=1.3, label=f"{MDNAME} (n = {cur['n_blocks']} blocks)")
                else:
                    ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=REF, fontsize=8)
                if i == 0:
                    ax.set_title(f"n = {nw}, {T} K", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{ion}-O  g(r)" + ("\n(multivalent" + (", counted)" if MV else ", excluded)") if ion == "Mg" else ""), fontsize=8)
                if i == len(ions) - 1:
                    ax.set_xlabel("r (A)", fontsize=8)
                ax.tick_params(labelsize=7, direction="in")
                ax.set_xlim(1.5, 8)
        axs[0, 0].legend(fontsize=6.5, frameon=False, loc="upper right")
        fig.tight_layout()
        cap = ("No DFT MD of these clusters was found in the program data (mdstab, lr-md-compare, lrmd-analysis, "
               "condensed-lr and cluster-laws were searched); the reference curve is the base model. Ion-O radial density around the ion in open clusters (n waters), 2 ps blocks with seeds pooled, band = "
               "95 % bootstrap of the block mean. " + D["md"]["rdf_note"] + " The n = 200 column uses every second "
               "frame (40 fs).")
        return mo.vstack([mo.md("### RDFs (vs PET-OMol-S; no DFT MD of these clusters was found)"), mo.as_html(fig), mo.md(cap)])
    _rdf()
    return BAND, INK, REF, plt


@app.cell
def _(D, L, MDNAME, mo, not_run, pd, table):
    def _sec():
        s = D["md"]["charges_md"]["summary"]
        rows = [dict(ion=k.split("|")[1], **{kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
                for k, v in s.items() if MDNAME and k.split("|")[0] == MDNAME]
        st = D["md"]["charges_static"]["table"]
        srows = []
        for k, v in st.items():
            ion, n, scheme = k.split("_", 2)
            if scheme in (MDNAME, "Löwdin", "Hirshfeld"):
                srows.append(dict(ion=ion, n_waters=int(n), source=("model " + scheme) if scheme == MDNAME else f"DFT {scheme}",
                                  **{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
        if not rows and not any(r["source"].startswith("model") for r in srows):
            return not_run("Charges vs DFT along MD and on static clusters", "No charge output for this model.")
        for r in rows + srows:
            r["max |q|"] = L.ION_Q.get(r["ion"], 1)
        txt = ("Static: model LES charges against DFT Löwdin and Hirshfeld on the same cluster-laws clusters (mean over "
               "centres, 95 % bootstrap over centres; whole-water convention). MD: model charges averaged along the "
               "model's own trajectory (40 frames); there is no DFT along these trajectories, so the MD table has no DFT "
               "column and is compared with the static DFT rows only as context (different geometries). "
               + str(D["md"]["charges_md"].get("note") or ""))
        out = [mo.md("### Charges vs DFT"), mo.md(txt)]
        if srows:
            out.append(table(pd.DataFrame(srows), "static clusters: model vs DFT schemes"))
        if rows:
            out.append(table(pd.DataFrame(rows), "charges along MD"))
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(D, INK, M, MDNAME, MID, REF, mo, not_run, pd, plt, table):
    def _sec():
        pm = [e for e in D["md"]["permit"] if e["model_id"] == MID]
        em = [e for e in D["md"]["eps_md"] if MDNAME and e["model"] == MDNAME]
        if not pm and not em:
            return not_run("Permittivity", "No periodic or droplet water runs exist for this model.")
        out = [mo.md("### Permittivity (water, 300 K)")]
        if pm:
            rows = [dict(box=e["box"], N_water=e["N_water"], seeds=e["n_seeds"], eps=round(e["eps"], 1),
                         chi2_95=str([round(x, 1) for x in e["eps_chi2_95"]]),
                         block_boot_95=str([round(x, 1) for x in e["eps_blockboot_95"]]),
                         per_seed=str([round(x, 1) for x in e["eps_per_seed"]]), tau_M_ps=round(e["tau_M_int_ps"], 2),
                         t_used_ps=e["t_used_ps"]) for e in pm]
            ref = [e for e in D["md"]["permit"] if e["model_id"] == "pet-omol-s"]
            fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * (len(pm) + len(ref))))
            y = 0
            for lab, es, col in ((M["name"], pm, INK), ("PET-OMol-S (reference)", ref if MID != "pet-omol-s" else [], REF)):
                for e in es:
                    ax.plot(e["eps_chi2_95"], [y, y], color=col, lw=1.2)
                    ax.plot(e["eps_blockboot_95"], [y + 0.12, y + 0.12], color=col, lw=0.8, ls=":")
                    ax.plot(e["eps"], y, "o", color=col, ms=5)
                    ax.text(e["eps_chi2_95"][1] + 1, y, f"{lab}, {e['box']}", va="center", fontsize=7)
                    y += 1
            ax.axvline(78.4, color=REF, lw=0.8)
            ax.text(78.4, y - 0.4, " experiment 78.4", fontsize=7, color=REF)
            ax.set_yticks([]); ax.set_xlabel("static dielectric constant (D1 dipole proxy)", fontsize=8)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=7, direction="in"); fig.tight_layout()
            out += [mo.md("Protocol A (permit lane): periodic NVT at 1.032 g/cm3, 4 seeds of 100 ps, 10 ps discarded, "
                          "tin-foil boundary, fixed 2.9 D bisector dipole per water (an orientational proxy that leaves out "
                          "the model's own polarisation). Solid line: chi-square 95 % interval (N_eff = T / 2 tau_M); "
                          "dotted: moving-block bootstrap 95 %. Both are asymmetric."), mo.as_html(fig),
                    table(pd.DataFrame(rows), "permittivity, protocol A")]
        if em:
            def _r(x, n):
                return None if x is None else round(x, n)
            rows = [dict(run=e["name"], route=e.get("route"), N_water=e["N_water"], T=e["T"], eps=_r(e.get("eps"), 1),
                         rel_err_1sigma=_r(e.get("eps_minus1_rel_err", e.get("rel_err_var_1sigma")), 3), g_K=_r(e.get("g_K"), 2),
                         tau_M_ps=_r(e.get("tau_M_ps"), 2), t_prod_ps=e.get("t_prod_ps"), N_eff=_r(e.get("N_eff"), 1),
                         **{"max |q|": 0}) for e in em]
            out += [mo.md("Protocol B (MD-stability lane): one seed per row, 20-100 ps, periodic tin-foil boxes (eps) "
                          "or open clusters (Kirkwood g_K only; eps is not defined for a cluster, so that column is empty). "
                          "These rows are not comparable with protocol A (fewer samples, different routes); the 1-sigma "
                          "relative error is the chi-square estimate of the run itself (on eps - 1 for boxes, on the "
                          "dipole variance for clusters)."),
                    table(pd.DataFrame(rows), "permittivity, protocol B")]
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(D, MID, mo, not_run, pd, table):
    def _sec():
        cons = (D["prog"].get("cons") or {}).get(MID)
        if not cons:
            return not_run("Conservation suite and NVE drift",
                           "Run only for the C0 and P EMA files (seeds 13r-16, MD form) in the program.")
        rows, suite = [], []
        for s, v in cons.items():
            m1 = v["nve"].get("M1", {})
            rows.append(dict(seed=s, frame=v["frame"], atoms=v["n_atoms"], **{
                "M1 drift meV/atom/ns": round(m1.get("value", float("nan")), 2),
                "95% CI": str([round(x, 2) for x in (m1.get("ci") or [])]), "M1 tol": m1.get("tol"), "M1": m1.get("status")},
                **{k: x.get("status") for k, x in v["nve"].items() if k != "M1"}))
            for path in ("ragged", "padded"):
                for r in v.get(f"suite_{path}", []):
                    suite.append(dict(seed=s, path=path, **r))
        txt = ("NVE: one 49-atom Cl-(H2O)16-type open frame, 1 ps NVT then 3 ps NVE, 0.5 fs, one trajectory per file, so "
               "there is no spread within a file; tolerance 3 meV/atom/ns. Conservation suite 1.8 on the MD-form file "
               "(ragged and padded paths). Weights: EMA, GH200 seeds.")
        return mo.vstack([mo.md("### Conservation and NVE"), mo.md(txt), table(pd.DataFrame(rows), "NVE checks"),
                          table(pd.DataFrame(suite), "conservation suite", page=12)])
    _sec()
    return


@app.cell
def _(BD, BD_STATE, D, DDIR, L, M, MV, mo, not_run, np, pd, table):
    def _sec():
        b = M.get("bench") and D["bench"]["models"].get(M["bench"])
        if not b:
            return not_run("LongRangeBench sets (dimers, trimers, families, charge parity, external sets)",
                           "This model has no committed public row in results/.")
        ident, rows = [], []
        for s, v in b["sets"].items():
            ident.append(dict(set=s, model_level=v["model_level"], level_mismatch=v["level_mismatch"], evidence=v["evidence"],
                              date=v["date"], commit=v["code_commit"], n_scored=v["n_scored"], n_failed=v["n_failed"],
                              members="; ".join(f"{m.get('package')} {m.get('version')} {m.get('checkpoint')}" for m in v["members"]),
                              reference_level=", ".join(D["bench"]["sets"].get(s, {}).get("reference_level") or [])))
            for m in v["metrics"]:
                rows.append(dict(set=s, **{k: (str(x) if isinstance(x, (list, dict)) else x) for k, x in m.items()}))
        extra = []
        bd = L.block(str(DDIR), "dimers")
        if bd is not None and f"Eint|{M['bench']}" in bd:
            v = bd[f"Eint|{M['bench']}"]; e = np.abs(v - bd["ref"]); ok = np.isfinite(e)
            F = L.bd_flags(bd, BD["rule"], BD["thr"]); keep = L.bd_select(F["mask"], BD["mode"])
            subsets = [("all E_int groups", ok), ("max |q| <= 1", ok & (bd["maxq"] < 2)), ("multivalent only (max |q| >= 2)", ok & (bd["maxq"] >= 2))]
            for name, m_ in subsets:
                for part, sel, counts in ((BD["mode"], m_ & keep, {"all E_int groups": MV, "max |q| <= 1": not MV}.get(name, False)),
                                          ("excluded: breakdown (shown apart)", m_ & F["mask"], False)):
                    if part.startswith("excluded") and BD["mode"] != "exclude":
                        continue
                    row = dict(subset=name, breakdown=part, **{"E_int MAE meV": None, "95% lo": None, "95% hi": None,
                                                               "n groups": int(sel.sum()), "of which CT-ambiguous": int((sel & F["ct"]).sum()),
                                                               "counts in score": counts and not part.startswith("excluded")})
                    if sel.any():
                        rng = np.random.default_rng(0); x = e[sel]; bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
                        row.update({"E_int MAE meV": round(float(x.mean()), 2), "95% lo": round(float(np.percentile(bs, 2.5)), 2),
                                    "95% hi": round(float(np.percentile(bs, 97.5)), 2)})
                    elif m_.any():
                        row["breakdown"] += " (no frames in this state, n = 0)"
                    if m_.any():
                        extra.append(row)
            print(f"[breakdown] dimers E_int {M['bench']} | {BD['rule']} | {BD['mode']} | "
                  + "; ".join(f"{r['subset']} [{r['breakdown']}] n={r['n groups']}" for r in extra))
        txt = ("Multivalent: the committed rows aggregate every record of a set, and the dimers set holds 19 of 1,254 "
               "records with a species of |q| = 2, so the committed numbers include multivalent records whatever the switch "
               "says; they cannot be split from the committed row. The recomputed dimer E_int MAE below (from the public "
               f"saved predictions, structure bootstrap) does split them; {L.mv_state(MV)}; {BD_STATE}. The committed rows "
               "include breakdown-flagged records whatever the breakdown state. "
               "Benchmark rows as committed in results/ (one seed each, so every row is a screen). Values are estimate "
               "[95 % hierarchical-bootstrap interval] as rendered by lrbench. A row whose model level differs from the "
               "reference level is a level mismatch, flagged in the identity table and in the 'level_mismatch' column.")
        out = [mo.md("### LongRangeBench sets"), mo.md(txt), table(pd.DataFrame(ident), "identity per set"),
               table(pd.DataFrame(rows), "metrics per set (committed rows)", page=20)]
        if extra:
            out.append(table(pd.DataFrame(extra), "dimers E_int MAE recomputed, with and without multivalent records"))
        return mo.vstack(out)
    _sec()
    return


@app.cell
def _(BD, BD_STATE, DDIR, L, MV, mo, np, pd, table):
    # Dielectric-breakdown flags on the reference frames of every bundle block, model-independent: what the current
    # rule flags, how it overlaps the CT-ambiguous class, and how many frames count in a score in the current state.
    def _bdsec():
        rows, flagged = [], []
        for name in ("dimers", "trimers", "val500", "setB", "typeheld"):
            b = L.block(str(DDIR), name)
            if b is None:
                continue
            F = L.bd_flags(b, BD["rule"], BD["thr"]); m, ct = F["mask"], F["ct"]
            mq = (np.asarray(b["maxq"]) if "maxq" in b else
                  np.array([L.pair_maxq(str(x)) for x in b["pair"]]) if "pair" in b else np.zeros(len(m), int))
            mvok = np.ones(len(m), bool) if MV else (mq < 2)
            keep = L.bd_select(m, BD["mode"]) & mvok
            rows.append({"block": name, "frames": len(m), "flagged": int(m.sum()), "flagged, max |q| <= 1": int((m & (mq < 2)).sum()),
                         "no charge labels": len(m) - F["n_labelled"] if BD["rule"] == "physics" else None,
                         "CT-ambiguous": int(ct.sum()), "both": int((m & ct).sum()), "CT-ambiguous, not flagged": int((ct & ~m).sum()),
                         "CT-ambiguous in score": int((ct & keep).sum()), "count in score": int(keep.sum()),
                         "shown apart": int((~keep).sum()), "what the rule found": F["msg"]})
            print(f"[breakdown] {name} | {BD['rule']} | {BD['mode']} | thr {BD['thr']:g} | flagged {int(m.sum())}/{len(m)} | "
                  f"CT-ambiguous {int(ct.sum())} | both {int((m & ct).sum())} | CT-amb in score {int((ct & keep).sum())} | "
                  f"in score {int(keep.sum())} | {F['msg']}")
            for i in np.flatnonzero(m | ct):
                flagged.append({"block": name, "frame": str(b["name"][i]) if "name" in b else int(i),
                                "class": str(b["cls"][i]) if "cls" in b else "",
                                "|Löwdin - formal| max over fragments (e)": (round(float(b["dq_frag"][i]), 3)
                                                                             if "dq_frag" in b and np.isfinite(b["dq_frag"][i]) else None),
                                "breakdown": F["source"][i] or "not flagged", "CT-ambiguous": bool(ct[i]),
                                "max |q|": int(mq[i]), "counts in score": bool(keep[i])})
        txt = (f"**{BD_STATE}; {L.mv_state(MV)}.** Reference frames per bundle block, independent of the model. "
               "'flagged' is the breakdown label, 'CT-ambiguous' the dataset's own CT class; they are separate labels "
               "and are cross-tabulated, not merged ('both'). 'count in score' applies the breakdown mode and the "
               "multivalent switch together; 'CT-ambiguous in score' shows that CT-ambiguous frames the rule does not "
               "flag stay in under 'exclude'. The physics rule flags a frame when any fragment's DFT Löwdin (else "
               "Hirshfeld) sum differs from its formal charge by more than the threshold (default 0.1 e). Frames without "
               "fragment charge labels are counted under 'no charge labels' and never flagged. A check on the response (phi outside the band-edge window, or (dphi/dq)^-1 outside "
               f"C0 = {L.BD_C0} +- {L.BD_C0_TOL} uF/cm^2) runs only where those labels exist; no current set has them.")
        out = [mo.md("### Dielectric breakdown flags"), mo.md(txt), table(pd.DataFrame(rows), "breakdown flags per block")]
        if flagged:
            out.append(mo.accordion({f"Every flagged or CT-ambiguous frame ({len(flagged)})":
                                     table(pd.DataFrame(flagged), "breakdown and CT-ambiguous frames", page=20)}))
        return mo.vstack([x for x in out if x is not None])
    _bdsec()
    return


@app.cell
def _(LABS, PROG, mo, pd):
    _rows = []
    for _l in LABS:
        for _k, _v in PROG[_l]["scorecard"].items():
            _rows.append(dict(file=_l, seed=PROG[_l]["seed"], weights=PROG[_l]["weights"], step=PROG[_l]["step"], key=_k,
                              value=_v, **{"max |q|": 2 if "Mg" in _k else (1 if any(t in _k for t in ("Na", "Cl", "Li", "NH4", "Gdm", "K+", "PF6", "BF4", "OAc", "TMA", "seam", "wrong", "q95", "setB", "frag", "per-atom")) else "mixed / n.a."),
                                           "multivalent": "Mg2+ key: not a score while the switch is off" if "Mg" in _k else ""}))
    mo.vstack([mo.md("### Every scorecard key (raw, as written by the program's scorer)"),
               mo.md("Long form: one row per file and key, unedited. Units are in the key names; 'LES off' keys are the "
                     "same weights with the long-range gate at 0."),
               mo.ui.table(pd.DataFrame(_rows), page_size=12, selection=None, show_download=True)]) if _rows else mo.md("")
    return


@app.cell
def _(LABS, M, mo):
    variant = mo.ui.dropdown(options=LABS, value=(M.get("rep") if M.get("rep") in LABS else LABS[0]),
                             label="file for the structure viewers") if LABS else None
    mo.vstack([mo.md("## Structures (chemiscope)"),
               mo.md("One widget per sub-test with geometries. Arrows: model force, reference force and their difference "
                     "(pick one in chemiscope's structure settings). Map axes and colours are chemiscope's own controls."),
               variant if variant is not None else mo.md("")])
    return (variant,)


@app.cell
def _(DDIR, L, MID, mo, not_run, np, variant):
    def _views():
        lab = variant.value if variant is not None else ("pet-omol-s" if MID == "pet-omol-s" else None)
        out = []
        b = L.block(str(DDIR), "val500")
        if lab and b is not None and f"F|{lab}|on" in b:
            ds = L.force_view(b, lab, f"val500, {lab}", "120 OMol25 validation frames (60 charged, 60 neutral), "
                              "wB97M-V/def2-TZVPD forces; forces stored as float16.",
                              sprops_extra={"total charge": (b["charge"], "e", ""),
                                            "E error": (b[f"E|{lab}|on"], "meV/atom", "model - DFT residual energy per atom"),
                                            "max |q|": (b["maxq"], "e", "largest species charge in the frame")})
            out += [mo.md(f"**val500 subset** ({lab}): 120 of 500 frames, DFT forces."),
                    mo.iframe(L.page(ds, L.settings("total charge", "F MAE vs DFT", "E error", "force error")), height="700px")]
        else:
            out.append(not_run("val500 structures"))
        b = L.block(str(DDIR), "setB")
        if lab and b is not None and f"F|{lab}|on" in b:
            ds = L.force_view(b, lab, f"set-B pairs, {lab}", "held-out rigid pair scans, DFT forces",
                              sprops_extra={"COM distance": (b["r"], "A", ""), "pair": (b["pair"], "", ""),
                                            "orientation": (b["orient"], "", "")})
            out += [mo.md(f"**Pairs gate scans** ({lab}): DFT forces, all 360 pair frames (gate uses r >= 8 A)."),
                    mo.iframe(L.page(ds, L.settings("COM distance", "F MAE vs DFT", "F MAE vs DFT", "force error")), height="700px")]
        else:
            out.append(not_run("Pairs-gate structures"))
        b = L.block(str(DDIR), "typeheld")
        tl = lab if (lab and b is not None and f"F|{lab}|on" in b) else ("pet-omol-s" if MID == "pet-omol-s" else None)
        if tl and b is not None and f"F|{tl}|on" in b:
            sel = None
            ds = L.force_view(b, tl, f"type-held-out pairs, {tl}", "unseen charged pair types, DFT forces; "
                              "Gdm+Mg2+ (max |q| = 2) is multivalent and excluded from the score by default", sprops_extra={"COM distance": (b["r"], "A", ""), "pair": (b["pair"], "", ""),
                                                                         "dmin": (b["dmin"], "A", "closest cross-fragment distance"),
                                                                         "max |q|": (np.array([L.pair_maxq(p) for p in b["pair"]]), "e", "largest species charge")},
                              sel=sel)
            out += [mo.md(f"**Type-held-out pairs** ({tl}): K+Gdm+, Gdm+I- and Gdm+Mg2+ (colour or filter by max |q|; the multivalent pair does not count in the score unless the switch is on)."),
                    mo.iframe(L.page(ds, L.settings("COM distance", "F MAE vs DFT", "F MAE vs DFT", "force error")), height="700px")]
        else:
            out.append(not_run("Type-held-out structures"))
        b = L.block(str(DDIR), "ladder")
        if lab and b is not None and f"dE|{lab}|on" in b:
            keep = None
            err = b[f"dE|{lab}|on"] - b["dE_ref"]
            sp = {"waters": (b["nw"], "", ""), "dE DFT": (b["dE_ref"], "eV", "vertical ion binding energy, DFT"),
                  "dE error": (err, "eV", "model - DFT"), "ion": (b["ion"], "", ""), "centre": (b["centre"], "", ""),
                  "max |q|": (np.array([L.ION_Q.get(str(x), 1) for x in b["ion"]]), "e", "largest species charge")}
            ap, ar = None, None
            if f"F|{lab}|on" in b:
                F = b[f"F|{lab}|on"].astype(float)
                ap = {"|F model|": (np.linalg.norm(F, axis=1), "eV/A", "model force (no DFT forces exist for the ladder)")}
                ar = {"model force": F}
            ds = L.dataset(b, f"ladder, {lab}", "ion + n water clusters, held-out centres; DFT energies only", sp, ap, ar, sel=keep)
            out += [mo.md(f"**Cluster ladder** ({lab}): held-out centres, error of the vertical binding energy; "
                          "DFT energies only (no DFT forces); Mg2+ and Ca2+ clusters carry max |q| = 2 (multivalent)."),
                    mo.iframe(L.page(ds, L.settings("waters", "dE error", "dE error")), height="700px")]
        else:
            out.append(not_run("Ladder structures"))
        b = L.block(str(DDIR), "search")
        if lab and b is not None and f"F|{lab}|on" in b:
            off = b["offsets"]; n = len(off) - 1
            Fm = b[f"F|{lab}|on"].astype(float)
            fb = np.array([Fm[off[i] + b["nA"][i]:off[i + 1]].sum(0) @ b["axis"][i] for i in range(n)])
            sp = {"net force on B along axis": (fb, "eV/A", "LES on"), "ratio to Coulomb": (fb / b["coulomb"], "", "oracle k/d^2"),
                  "cell": (np.array([f"{p} {d:g}" for p, d in zip(b["pair"], b["d"])]), "", ""), "COM distance": (b["d"], "A", "")}
            ar = {"model force": Fm}
            if f"F|{lab}|off" in b:
                Fo = b[f"F|{lab}|off"].astype(float)
                fo = np.array([Fo[off[i] + b["nA"][i]:off[i + 1]].sum(0) @ b["axis"][i] for i in range(n)])
                sp["ratio to Coulomb, LES off"] = (fo / b["coulomb"], "", "backbone only")
                ar["model force, LES off"] = Fo
            ds = L.dataset(b, f"seam cells, {lab}", "40 orientations per cell; reference = Coulomb oracle, not DFT", sp,
                           {"|F model|": (np.linalg.norm(Fm, axis=1), "eV/A", "")}, ar)
            out += [mo.md(f"**Seam cell orientation scans** ({lab}): 40 of 2,000 orientations per cell; the reference is "
                          "the Coulomb oracle (no DFT for these frames). Ratio < 0 = attractive."),
                    mo.iframe(L.page(ds, L.settings("COM distance", "ratio to Coulomb", "ratio to Coulomb", "model force")), height="700px")]
        else:
            out.append(not_run("Seam orientation structures", "Stored for one EMA file per arm (the default file)."))
        return mo.vstack(out)
    _views()
    return


@app.cell
def _(DDIR, L, MDNAME, mo, not_run, np):
    def _md():
        b = L.block(str(DDIR), "md")
        if not MDNAME or b is None or not (b["model"] == MDNAME).any():
            return not_run("MD final frames")
        sel = np.where(b["model"] == MDNAME)[0]
        sp = {"T": (b["T"], "K", "thermostat"), "waters": (b["nw"], "", ""), "ion": (b["sys"], "", ""),
              "verdict": (b["verdict"], "", ""), "drift": (b["drift"], "meV/atom/ps", "conserved-quantity slope"),
              "first-shell waters, last frame": (b["nshell"], "", ""), "max force": (b["fmax"], "eV/A", "over the run"),
              "seed": (b["seed"], "", ""), "max |q|": (np.array([L.ION_Q.get(x, 1) for x in b["sys"]]), "e", "")}
        ds = L.dataset(b, f"MD final frames, {MDNAME}", "last frame of every completed run (n = 48 and 200); "
                       "positions only, no forces stored", sp, sel=sel)
        return mo.vstack([mo.md(f"**MD final frames** ({MDNAME}): the structures behind the stability grid and the RDFs."),
                          mo.iframe(L.page(ds, L.settings("T", "drift", "first-shell waters, last frame")), height="700px")])
    _md()
    return


@app.cell
def _(DDIR, L, M, mo, not_run, np):
    def _bench():
        out = []
        m = M.get("bench")
        for s in ("dimers", "trimers"):
            b = L.block(str(DDIR), s)
            if not m or b is None or f"Eint|{m}" not in b:
                out.append(not_run(f"Benchmark {s} structures", "No public saved-prediction run for this model."))
                continue
            v = b[f"Eint|{m}"]
            sp = {"E_int DFT": (b["ref"], "meV", ""), f"E_int {m}": (v, "meV", "energies only; the run saved no forces"),
                  "E_int error": (v - b["ref"], "meV", ""), "class": (b["cls"], "", ""), "dmin": (b["dmin"], "A", ""),
                  "total charge": (b["charge"], "e", ""),
                  "max |q|": (b["maxq"], "e", "largest species charge (multivalent >= 2)")}
            Fr = b["F_ref"].astype(float)
            ds = L.dataset(b, f"{s}, {m}", "E_int groups; DFT forces as arrows; model forces were not saved", sp,
                           {"|F DFT|": (np.linalg.norm(np.nan_to_num(Fr), axis=1), "eV/A", "")}, {"DFT force": np.nan_to_num(Fr)})
            out += [mo.md(f"**Benchmark {s}** ({m}): E_int error per structure; DFT forces only (the benchmark run saved "
                          "energies, not forces)."),
                    mo.iframe(L.page(ds, L.settings("dmin", "E_int error", "E_int error", "DFT force")), height="700px")]
        return mo.vstack(out)
    _bench()
    return


if __name__ == "__main__":
    app.run()
