# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "anywidget",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""ChargeWorld: a synthetic benchmark with exact charges, what charge heads learn on it, and why some runs fail.

    uvx marimo edit --sandbox notebooks/chargeworld.py        # edit; the sandbox installs the header's packages
    uvx marimo run --sandbox notebooks/chargeworld.py         # app view
    marimo export html notebooks/chargeworld.py -o chargeworld.html   # static page; the 3D scenes still rotate

Data: ``data/chargeworld/`` next to this file (about 3.6 MB), written by ``tools/build_chargeworld_data.py`` from the
ChargeWorld sources. Helpers next to this file: cw_data.py (loaders, the species-charge toy), cw_figs.py (figures),
cw_widgets.py + cw_scenes.js (three.js scenes; three.js r170 is vendored in vendor/, so nothing loads from the network).
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="ChargeWorld")


@app.cell
def _():
    import os
    import subprocess
    import sys

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(mo.notebook_dir()))
    import cw_data as C
    import cw_figs as Fg
    import cw_widgets as W

    Fg.style(plt)
    DDIR = C.data_dir(mo.notebook_dir())
    BUILDER = mo.notebook_dir().parent / "tools" / "build_chargeworld_data.py"
    return BUILDER, C, DDIR, Fg, W, mo, np, os, pd, plt, subprocess, sys


@app.cell
def _(mo):
    reread = mo.ui.button(label="Re-read data folder", value=0, on_click=lambda v: v + 1)
    rebuild = mo.ui.run_button(label="Rebuild run results from the ChargeWorld sources")
    return rebuild, reread


@app.cell
def _(BUILDER, os, rebuild, subprocess, sys):
    _src = os.path.expanduser(os.environ.get("CW_RUN", "~/data/lr-charge-world-run"))
    build_log = ""
    if rebuild.value:
        if os.path.isdir(_src):
            _p = subprocess.run([sys.executable, str(BUILDER), "--results-only"], capture_output=True, text=True)
            build_log = (_p.stdout + _p.stderr)[-1500:]
        else:
            build_log = f"The run folder {_src} is not on this machine; the data folder is unchanged."
    return (build_log,)


@app.cell
def _(C, DDIR, build_log, reread):
    _ = (reread.value, build_log)
    D = C.load(DDIR)
    SPEC = D["tiers"]["species"] if D["tiers"] else []
    SPEC_Z = sorted(SPEC, key=lambda s: s["Z"])
    NAME_Z = {s["Z"]: f"{s['name']} ({s['q0']:+g})" for s in SPEC}
    NAME_Z.update({1: "T0 cation (+1)", 2: "T0 anion (−1)"})
    TOY_NAMES = [f"{s['name']} ({s['q0']:+g})" for s in SPEC_Z]
    return D, NAME_Z, SPEC_Z, TOY_NAMES


@app.cell
def _(C, DDIR, build_log, mo, rebuild, reread):
    mo.vstack([
        mo.md("# ChargeWorld: exact charges, charge heads, and the sign trap"),
        mo.md(
            "This notebook explains the ChargeWorld benchmark, shows its data in 3D, and analyses the first models "
            "trained on it. It then follows one failure, charge heads that learn the right charge sizes with the wrong "
            "signs, from the trained networks down to a 12-parameter toy, and to the conflicting gradients that hold the "
            "toy in its trap. Every number is read from the data folder below; cells whose files are missing say so."
        ),
        mo.hstack([reread, rebuild, mo.md(f"<small>data folder: `{DDIR.name}/` · {C.stamp(DDIR)}</small>")],
                  justify="start", gap=1),
        mo.md(f"```\n{build_log}\n```") if build_log else mo.md(""),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. What ChargeWorld is and why it exists

    A long-range potential has several ingredients: charges, their smearing, charge equilibration, induction,
    screening, boundary conditions, and robustness to noisy labels. Density-functional labels mix all of them, and their
    "true" charges are not defined. ChargeWorld separates the ingredients. It is a ladder of tiers. Each tier adds one
    physical aspect to the one before, and reduces to it at a knob limit (validated numerically), so a failure can be
    traced to the aspect that caused it.

    All labels are exact: closed-form energies or one direct linear solve on CPU. Every structure ships its per-atom
    charges, induced dipoles, the electrostatic potential at 32 probe points, and the split of energy and forces into
    long- and short-range terms. Forces match central finite differences to 7 × 10⁻⁷ eV/Å on every tier.

    Two kinds of reference make results readable. **Noise floors** are the error of the exact generator against the
    labels as given (zero for clean tiers, about 40 meV/Å for the T4a white noise). **Oracles** replace one part of the
    generator with what a simple model could know: per-species mean charges (what a fixed-charge head can reach),
    the true charges as point charges (what a charge head with perfect charges leaves to its short-range network), and
    a local two-body ridge fit (what a short-range model reaches).
    """)
    return


@app.cell
def _(D, W, mo):
    def _ladder():
        if not D["structures"]:
            return mo.callout(mo.md("structures.json is missing: run tools/build_chargeworld_data.py."), kind="warn")
        _tiers = []
        for _t in ["T0", "T1", "T2", "T3a", "T3b", "T3c", "T3d", "T4a", "TS", "T5"]:
            _s = D["structures"][_t][0]
            _X = __import__("numpy").asarray(_s["R"]); _X = _X - _X.mean(0)
            _tiers.append(dict(tier=_t, X=_X.tolist(), q=_s["q"]))
        return mo.ui.anywidget(W.Scene("ladder", data=dict(tiers=_tiers, distance=46,
            note="First test structure of each tier. Atoms coloured by their exact charge (blue negative, red positive, "
                 "near-white zero; spacers with |q| < 0.05 e translucent). Drag to rotate, scroll to zoom.")))
    _ladder()
    return


@app.cell
def _(D, Fg, mo, pd, plt):
    def _tiers():
        if not D["tiers"]:
            return mo.md("")
        _rows = []
        for _t in D["tiers"]["tiers"]:
            _b = _t["baselines"]; _s = _t["splits"]
            _rows.append({"tier": _t["tier"], "adds": _t["adds"], "reduces to": _t["nests"],
                          "train / val / test": f"{_s.get('train', '–')} / {_s.get('val', '–')} / {_s.get('test', '–')}",
                          "test-only splits": ", ".join(k for k in _s if k not in ("train", "val", "test")),
                          "fixed-charge oracle F MAE (meV/Å)": round(Fg.fixed_oracle(_t), 1),
                          "two-body 6 Å F MAE (meV/Å)": round(_b.get("local2b_rc6", {}).get("F_mae_meV_A_clean", float("nan")), 0),
                          "noise floor F (meV/Å)": _t["noise_floor_F"]})
        _fig = Fg.tier_baselines(plt, D["tiers"]["tiers"])
        return mo.vstack([_fig, mo.md(
            "**Figure 1. A per-species fixed-charge model is exact on T0 and T1 and leaves 181 meV/Å on T2, where the "
            "charges respond to the environment.** Test-split force MAE against clean labels (log scale; exact zeros are "
            "drawn at 10⁻³) for three reference models per tier. Blue: per-species mean charges with the tier's own "
            "kernel and everything else exact (what a fixed-charge long-range head can reach; not computed for T1s). "
            "Orange: the true charges used as point charges with everything else exact, the part a charge head with "
            "perfect charges leaves to its short-range network (smearing, on-site and induction terms; 770-970 "
            "meV/Å on the periodic tiers from T1s on, 235 meV/Å on the open T3a). Grey: a local two-body ridge model with a 6 Å cutoff. Red bars: the analytic "
            "label-noise floor of the noisy tiers. n = 100 (T0) or 200 test structures per tier."),
            mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=20, label="Tiers (sortable, filterable)")])
    _tiers()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The structures in 3D

    Pick a tier and a test structure. Colour is the atom's charge, from blue (−3 e) through near-white (0) to red
    (+3 e). For T0, T1 and T2 you can replace the exact charges with the latent charges of a trained model. Atoms whose
    model charge has the wrong sign are ringed in orange. A model's charges carry an arbitrary global sign (Section 3),
    so they are shown in the run's global gauge, the sign that best matches the exact charges over the whole test set;
    tick "raw sign" to see what the network outputs.
    """)
    return


@app.cell
def _(D, mo):
    _tiers = [t for t in (D["structures"] or {}) if not t.startswith("_")]
    ex_tier = mo.ui.dropdown(options=_tiers, value="T1" if "T1" in _tiers else (_tiers[0] if _tiers else None), label="tier")
    return (ex_tier,)


@app.cell
def _(D, ex_tier, mo):
    _S = (D["structures"] or {}).get(ex_tier.value, [])
    _runs = [x for x in ((D["runs"] or {}).get("runs", [])) if x["tier"] == ex_tier.value and x.get("q_struct")]
    def _lab(x):
        _sch = {"-": "", "S0": "plain LES", "S1": "corr. anchor", "S2": "MSE anchor", "S4": "EMA"}.get(x["scheme"], x["scheme"])
        _set = "first set" if x["set"] == "phase1" else "replicate set"
        return f"{x['arm']} {_sch} seed {x['seed']} ({_set}){' – failed' if x['fail'] else ''}"
    ex_struct = mo.ui.dropdown(options={f"test #{s['index']} ({len(s['Z'])} atoms, Q = {s['Q']:+d})": i for i, s in enumerate(_S)},
                               value=next(iter([f"test #{s['index']} ({len(s['Z'])} atoms, Q = {s['Q']:+d})" for s in _S]), None),
                               label="structure")
    _opts = {"exact charges": -1}
    _opts.update({_lab(x): i for i, x in enumerate(_runs)})
    _dflt = next((k for k in _opts if "seed 15 (first" in k), "exact charges")
    ex_source = mo.ui.dropdown(options=_opts, value=_dflt, label="charges")
    ex_arrows = mo.ui.dropdown(options=["none", "exact force", "model force error", "induced dipole"], value="none", label="arrows")
    ex_raw = mo.ui.checkbox(value=False, label="raw sign")
    ex_regions = mo.ui.checkbox(value=True, label="region charges")
    EX_RUNS = _runs
    mo.hstack([ex_tier, ex_struct, ex_source, ex_arrows, ex_raw, ex_regions], justify="start", wrap=True, gap=1)
    return EX_RUNS, ex_arrows, ex_raw, ex_regions, ex_source, ex_struct


@app.cell
def _(D, EX_RUNS, Fg, NAME_Z, W, ex_arrows, ex_raw, ex_regions, ex_source, ex_struct, ex_tier, mo, np, plt):
    def _explorer():
        _S = (D["structures"] or {}).get(ex_tier.value, [])
        if not _S or ex_struct.value is None:
            return mo.md("No structures for this tier.")
        s = _S[ex_struct.value]
        R = np.asarray(s["R"]); t = np.asarray(s["q"]); Z = np.asarray(s["Z"])
        cell = np.asarray(s["cell"]).reshape(3, 3)
        centre = (cell.sum(0) / 2) if s["bc"] == 1 else R.mean(0)
        run = EX_RUNS[ex_source.value] if (ex_source.value is not None and ex_source.value >= 0) else None
        q = t.copy(); halo = None; Fm = None
        if run is not None and str(s["index"]) in run["q_struct"]:
            raw = np.asarray(run["q_struct"][str(s["index"])])
            q = raw if ex_raw.value else run["gauge"] * raw
            halo = ((np.sign(run["gauge"] * raw) != np.sign(t)) & (np.abs(t) > 0.25)).tolist()
            if run.get("F_struct"):
                Fm = np.asarray(run["F_struct"][str(s["index"])])
        Fx = np.asarray(s["F"])
        arrows = None; acol = None
        if ex_arrows.value == "exact force":
            arrows = [[i, (0.8 * f).tolist()] for i, f in enumerate(Fx)]
        elif ex_arrows.value == "model force error" and Fm is not None:
            arrows = [[i, (2.0 * (Fm[i] - Fx[i])).tolist()] for i in range(len(Fx)) if np.linalg.norm(Fm[i] - Fx[i]) > 0.05]
            acol = "#E69F00"
        elif ex_arrows.value == "induced dipole" and s.get("mu"):
            arrows = [[i, (4.0 * np.asarray(m)).tolist()] for i, m in enumerate(s["mu"]) if np.linalg.norm(m) > 0.005]
        frags = None
        fr = np.asarray(s["frag"])
        if ex_regions.value and len(np.unique(fr)) > 1:
            frags = [dict(c=R[fr == k].mean(0).tolist(), Q=float(t[fr == k].sum())) for k in np.unique(fr)]
        ext = float(np.ptp(R, 0).max()) if s["bc"] != 1 else float(np.linalg.norm(cell, axis=1).max())
        n_wrong = int(np.sum(halo)) if halo else 0
        note = (f"{ex_tier.value} test structure {s['index']}: {len(Z)} atoms, total charge Q = {s['Q']:+d} e, "
                f"{'periodic box' if s['bc'] == 1 else ('slab' if s['bc'] == 2 else 'open boundary')}. "
                f"Exact energy {s['E']:.2f} eV, of which point-charge Coulomb {s['terms'].get('E_coul_point', 0):.2f} eV. ")
        if run is not None:
            note += f"<b>{n_wrong}</b> atoms with |q| > 0.25 e have the wrong sign (orange rings). "
        if arrows is not None:
            note += {"exact force": "Arrows: exact forces (0.8 Å per eV/Å).",
                     "model force error": "Arrows: model minus exact force, shown where above 50 meV/Å (2 Å per eV/Å).",
                     "induced dipole": "Arrows: induced dipoles (4 Å per e Å)."}[ex_arrows.value]
        view = mo.ui.anywidget(W.Scene("explorer", data=dict(R=R.tolist(), q=q.tolist(), halo=halo, centre=centre.tolist(),
            cell=s["cell"] if s["bc"] != 0 else None, arrows=arrows, arrow_colour=acol, frags=frags, frag_lift=0.0,
            qmax=3.0, distance=2.6 * ext + 8, note=note)))
        side = None
        if run is not None:
            side = Fg.structure_q(plt, t, q, None, f"seed {run['seed']}: model vs exact")
            plt.close(side)
        return mo.hstack([view, side], widths=[1.7, 1]) if side is not None else view
    _explorer()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The data, the splits and the sign freedom

    T1, T1s, T2, T3b, T3c, T4a-g, TS and T5 share one set of geometries: periodic cubic boxes of 128-224 atoms at
    0.016-0.026 Å⁻³ with box sides of at least 20 Å, sampled by Monte Carlo. T3d cuts open droplets and slabs from
    the same boxes, and T3a uses open pairs of clusters. Each tier has 1,600 training, 200 validation and 200 test
    structures. Test-only splits probe transfer: `combo` holds the species pairs that never
    share a box in training (+3 with −3, and +2 with −2), `Qood` has total charges |Q| = 3 and 4, outside the
    training range of −2 to 2, and `big`, `dense` and `dilute` change size and density.
    """)
    return


@app.cell
def _(D, Fg, mo, plt):
    def _dist():
        if not D["distributions"]:
            return mo.md("distributions.json is missing.")
        _f = Fg.distributions(plt, D["distributions"])
        return mo.vstack([_f, mo.md(
            "**Figure 2. T2 equilibration smears the T1 charge lines; induction (T3b) and noise (T5) add little to the "
            "charge distribution.** Test splits, 200 structures each. (a) Atomic charges, fraction of atoms per 0.1 e "
            "bin (log scale). (b) Clean force magnitudes. (c) Energy per atom. (d) T1 total charges in the test split "
            "(grey) and in the Qood split (orange, 50 structures).")])
    _dist()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why the sign of the charges is not fixed by energies and forces

    The point-charge Coulomb energy of one structure is a quadratic form, $E = \tfrac12 q^\top J q$. Flipping every
    charge in the structure, $q \to -q$, leaves $E$ unchanged, and the forces $F = -\partial E/\partial R$ as well. So
    energies and forces fix each structure's charges only up to one sign per structure. The total-charge constraint
    does not break this in practice: it shifts each atom by only about $2Q/N \approx 0.02$ e. Only species that appear
    together in the same structure tie their relative signs. The demonstration below uses the exact per-species
    quadratic forms of one T1 test structure (point-charge Ewald energy and forces). Flip any set of species and
    compare.
    """)
    return


@app.cell
def _(D, NAME_Z, mo):
    _S = [s for s in ((D["structures"] or {}).get("T1", [])) if s.get("quad")]
    _present = sorted({z for s in _S for z in s["Z"]})
    flip_struct = mo.ui.dropdown(options={f"T1 test #{s['index']}": i for i, s in enumerate(_S)},
                                 value=f"T1 test #{_S[0]['index']}" if _S else None, label="structure")
    flip_species = mo.ui.multiselect(options={NAME_Z.get(z, str(z)): z for z in _present}, value=[], label="species to flip")
    FLIP_S = _S
    mo.hstack([flip_struct, flip_species], justify="start", gap=1)
    return FLIP_S, flip_species, flip_struct


@app.cell
def _(C, D, FLIP_S, NAME_Z, SPEC_Z, W, flip_species, flip_struct, mo, np):
    def _flip():
        if not FLIP_S or flip_struct.value is None:
            return mo.md("No T1 quadratic forms in the data folder.")
        s = FLIP_S[flip_struct.value]
        Zs = [sp["Z"] for sp in SPEC_Z]; q0 = np.array([sp["q0"] for sp in SPEC_Z])
        sign = np.array([-1.0 if z in (flip_species.value or []) else 1.0 for z in Zs])
        E0, F0 = C.coulomb_point(s["quad"], q0); E1, F1 = C.coulomb_point(s["quad"], q0 * sign)
        present = sorted(set(s["Z"]))
        all_flipped = all(z in (flip_species.value or []) for z in present)
        dF = F1 - F0; R = np.asarray(s["R"]); cell = np.asarray(s["cell"]).reshape(3, 3)
        zi = {z: i for i, z in enumerate(Zs)}
        q = [q0[zi[z]] * sign[zi[z]] for z in s["Z"]]
        big = [[i, (0.4 * d).tolist()] for i, d in enumerate(dF) if np.linalg.norm(d) > 0.01]
        note = (f"Point-charge Coulomb energy: exact charges {E0:.4f} eV, after the flip {E1:.4f} eV "
                f"(ΔE = {E1 - E0:+.4f} eV). Largest force change {np.abs(dF).max():.4f} eV/Å; RMS force change "
                f"{np.sqrt((dF ** 2).mean()):.4f} eV/Å. Orange arrows: force change (0.4 Å per eV/Å). ")
        if not flip_species.value:
            note += "Select species to flip."
        elif all_flipped:
            note += "<b>All species in the structure are flipped: energy and forces are unchanged to rounding.</b>"
        else:
            note += "Only part of the structure is flipped: energy and forces change."
        rows = []
        charged = [z for z in present if abs(q0[zi[z]]) > 0.1]
        for lab, zs in (("no flip", []), (f"flip one species ({NAME_Z.get(charged[0], '')})" if charged else "flip one species", charged[:1]),
                        ("flip every species in the structure", present)):
            sg = np.array([-1.0 if z in zs else 1.0 for z in Zs]); Ex, Fx_ = C.coulomb_point(s["quad"], q0 * sg)
            rows.append(f"| {lab} | {Ex - E0:+.6f} | {np.abs(Fx_ - F0).max():.6f} |")
        table = mo.md("| change of the exact charges | ΔE (eV) | largest force change (eV/Å) |\n|---|---|---|\n" + "\n".join(rows))
        view = mo.ui.anywidget(W.Scene("explorer", data=dict(R=R.tolist(), q=q, centre=(cell.sum(0) / 2).tolist(),
            cell=s["cell"], arrows=big, arrow_colour="#E69F00", qmax=3.0, distance=2.6 * np.linalg.norm(cell, axis=1).max() + 8,
            note=note)))
        return mo.vstack([view, table])
    _flip()
    return


@app.cell
def _(D, NAME_Z, SPEC_Z, mo):
    add_combo = mo.ui.switch(value=False, label="add the 50 combo structures (held-out pairs meet)")
    graph_trap = mo.ui.switch(value=False, label="colour species by the toy's trapped charges")
    mo.hstack([add_combo, graph_trap], justify="start", gap=2)
    return add_combo, graph_trap


@app.cell
def _(D, Fg, SPEC_Z, TOY_NAMES, W, add_combo, graph_trap, mo, np, plt):
    def _graph():
        if not D["distributions"]:
            return mo.md("")
        co = D["distributions"]["_cooc"]["counts"]
        Ct = np.asarray(co["train"], float); Cc = np.asarray(co["combo"], float)
        q0 = [s["q0"] for s in SPEC_Z]
        mx = Ct[~np.eye(12, dtype=bool)].max()
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                if Ct[i, j] > 0:
                    edges.append([i, j, Ct[i, j] / mx, 0])
                elif add_combo.value and Cc[i, j] > 0:
                    edges.append([i, j, max(0.25, Cc[i, j] / mx), 1])
        cq = q0; tl = None
        if graph_trap.value and D["toy_ref"]:
            cq = D["toy_ref"]["hess"]["trap_c"]
            tl = [bool(np.sign(a) != np.sign(b) and abs(b) > 0.1) for a, b in zip(cq, q0)]
        names = [n.split(" ")[0] + " " + n.split(" ")[1] for n in TOY_NAMES]
        note = ("Species of T1 placed on a ring, at a height set by their reference charge. Rod thickness: the number of "
                "training structures that contain both species (the coupling that ties their relative sign). "
                "+3/−3 and +2/−2 never share a training structure. ")
        if add_combo.value:
            note += "Orange rods: the ties added by the combo structures. "
        if graph_trap.value:
            note += "Colours: the toy's trapped minimum; translucent species carry the wrong sign there."
        view = mo.ui.anywidget(W.Scene("graph", data=dict(q0=q0, names=names, colour_q=cq, translucent=tl, edges=edges,
                                                          qmax=3.0, distance=24, note=note), elevation=28))
        f = Fg.cooc(plt, D["distributions"], names, add_combo.value); plt.close(f)
        return mo.hstack([view, f], widths=[1.6, 1])
    _graph()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Results on T0, T1 and T2

    Two architectures were trained on energies and forces only, four seeds each (13-16), at matched settings: **A0** is
    a plain short-range PET network; **A1** is the same network plus a latent-charge (LES) head whose charges enter an
    Ewald sum. T0 uses two capacities (lo: 3 Å cutoff, 1 layer; hi: 6 Å, 3 layers); T1 and T2 use the mid capacity
    (4.5 Å, 2 layers). A run **fails** by a pre-registered rule: choose one global sign for the run's test-set charges
    (the one that best matches the exact charges); the run fails if any species with |q| > 0.25 e has more than half of
    its atoms at the wrong sign, or if the sign-corrected charge RMSE is above 0.5 e.

    A second set of runs (the **replicate set**) repeats T1 A1 with a per-step probe of the charges on 32 fixed
    training structures, and tests three changes against plain LES (S0): a charge-correlation anchor (S1) and a
    charge-MSE anchor (S2), both training-only losses on the exact charges, and an EMA of the weights (S4, decay 0.999)
    as a negative control. Seeds 13-16 have the same initial weights as the first set; seeds 17-20 are new. A few
    T2 runs (plain LES and the correlation anchor, four seeds each) check the tier where no first-set seed failed.
    """)
    return


@app.cell
def _(D, Fg, mo, plt):
    def _ov():
        if not D["runs"]:
            return mo.callout(mo.md("runs.json is missing."), kind="warn")
        f = Fg.overview(plt, D["runs"]["runs"]); plt.close(f)
        bad = [x for x in D["runs"]["runs"] if x["set"] == "phase1" and x["tier"] == "T1" and x["arm"] == "A1 mid" and x["fail"]]
        share = [x["n_mixed"] * x["F_mae_mixed"] / (x["n_mixed"] * x["F_mae_mixed"] + (200 - x["n_mixed"]) * x["F_mae_coherent"]) for x in bad]
        ratio = [x["F_mae_mixed"] / x["F_mae_coherent"] for x in bad]
        return mo.vstack([f, mo.md(
            "**Figure 3. A charge head cuts the T1 force error ten-fold when its signs are consistent. In the two T1 "
            f"seeds that fail, a sign-mixed structure has {min(ratio):.0f}-{max(ratio):.0f}× the force error of a "
            f"sign-coherent one, and the mixed structures ({min(x['n_mixed'] for x in bad)}-{max(x['n_mixed'] for x in bad)} of 200) "
            f"carry {100 * min(share):.0f}-{100 * max(share):.0f} % of the total error.** (a) Test force "
            "MAE (meV/Å, inference with 16 neighbours) per seed, first set; open circles: A1 seeds that fail the sign "
            "rule (2 of 4 on T1, 0 of 4 on T2, 0 of 16 on T0). (b) T1, A1: the same force MAE split into test structures "
            "whose charges mix signs under the structure's own best sign (more than 1 % of the charged atoms at the "
            "wrong sign; squares) and the rest (circles); counts of mixed structures out of 200 below each seed. T2 "
            "charges respond to the environment, so the T2 A1 error floor (100 meV/Å) is not comparable with T1.")])
    _ov()
    return


@app.cell
def _(D, Fg, mo, plt):
    def _transfer():
        if not D["runs"] or not D["runs"].get("ood"):
            return mo.md("")
        f = Fg.transfer(plt, D["runs"]["ood"]); plt.close(f)
        return mo.vstack([f, mo.md(
            "**Figure 4. No seed transfers to the test-only splits, including the seeds that learn the right signs.** "
            "Force MAE (meV/Å, log scale) on the T1 and T2 test split (200 structures), the combo split, where the "
            "held-out pairs +3/−3 and +2/−2 share a box (50 structures), and the Qood split with total charges outside "
            "training (50 structures); first set, four seeds per architecture. On T1, the two A1 seeds that pass the "
            "sign rule go from 20-24 meV/Å on test to 238-282 meV/Å on combo and 59-108 meV/Å on Qood.")])
    _transfer()
    return


@app.cell
def _(D, Fg, NAME_Z, mo, plt):
    sign_set = mo.ui.dropdown(options={"first set, T1": ("phase1", "T1"), "first set, T2": ("phase1", "T2"),
                                       "first set, T0 hi": ("phase1", "T0"),
                                       "replicate set, T1": ("anchors", "T1")}, value="first set, T1", label="runs")
    sign_scheme = mo.ui.multiselect(options=["S0", "S1", "S2", "S4"], value=["S0"], label="schemes (replicate set)")
    mo.hstack([sign_set, sign_scheme], justify="start", gap=1)
    return sign_scheme, sign_set


@app.cell
def _(D, Fg, NAME_Z, mo, plt, sign_scheme, sign_set):
    def _signs():
        if not D["runs"]:
            return mo.md("")
        st, tier = sign_set.value
        R = [x for x in D["runs"]["runs"] if not (tier == "T0" and "hi" not in x["arm"])]
        f = Fg.species_signs(plt, R, st, tier, NAME_Z, sign_scheme.value if st == "anchors" else None)
        if f is None:
            return mo.md("No runs with charges for this choice.")
        plt.close(f)
        sel = [x for x in R if x["set"] == st and x["tier"] == tier and x.get("fail") and (st != "anchors" or x["scheme"] in sign_scheme.value)]
        fl = "; ".join(f"seed {x['seed']}{' ' + x['scheme'] if x['scheme'] != '-' else ''}: "
                       + ", ".join(NAME_Z.get(int(z), str(z)) for z in x["flipped"]) for x in sel) or "none"
        return mo.vstack([f, mo.md(
            "**Figure 5. Failed runs keep the right charge size per species and flip the sign of whole species.** "
            "Bars: mean latent charge per species over the test split (e), after the run's global sign; one bar per "
            "run in the order of the title (✗ = failed). Short black ticks: mean |q| per species, drawn on the side of "
            "the exact sign; they sit at the exact size even where the bar points the wrong way. Long black lines: exact "
            "charges. Orange outline: more than half of the species' atoms at the wrong sign under the run's global "
            f"sign. Species flipped under the global sign: {fl}. The global sign follows the majority of the charge, so "
            "a run that flips a small block of species can show the complementary block outlined.")])
    _signs()
    return


@app.cell
def _(D, Fg, mo):
    _R = (D["runs"] or {}).get("runs", [])
    _keys = [Fg.run_key(x) for x in _R if x.get("curve")]
    curve_pick = mo.ui.multiselect(options=_keys, value=[k for k in _keys if k.startswith("first set · T1 A1")], label="validation curves")
    curve_pick
    return (curve_pick,)


@app.cell
def _(D, Fg, curve_pick, mo, plt):
    def _cur():
        if not D["runs"]:
            return mo.md("")
        f = Fg.curves(plt, D["runs"]["runs"], set(curve_pick.value)); plt.close(f)
        return mo.vstack([f, mo.md(
            "**Figure 6. The failed seeds separate from the good ones by epoch 2-5 and then descend smoothly to a "
            "4-7× higher floor.** Validation force RMSE per epoch (meV/Å; 200 validation structures). Purple: failed by "
            "the sign rule; green: passed; grey: no charge head.")])
    _cur()
    return


@app.cell
def _(C, D, Fg, mo, plt):
    def _anch():
        if not D["runs"]:
            return mo.md("")
        R = D["runs"]["runs"]
        A = [x for x in R if x["set"] == "anchors" and x["tier"] == "T1"]
        sch = ("S0", "S1", "S2", "S4")
        seeds = {s_: sorted(x["seed"] for x in A if x["scheme"] == s_) for s_ in sch}
        k_by = {s_: sum(x["fail"] for x in A if x["scheme"] == s_) for s_ in sch}
        f = Fg.rates(plt, R, "anchors", "scheme", list(sch), D["runs"]["scheme_names"]); plt.close(f)
        missing = {s_: [sd for sd in range(13, 21) if sd not in seeds[s_]] for s_ in sch}
        msg = mo.callout(mo.md(
            "Partial: " + "; ".join(f"{s_} {len(seeds[s_])}/8 (missing {', '.join(map(str, missing[s_]))})" for s_ in sch if missing[s_])
            + ". Missing seeds appear after their runs finish and the rebuild button at the top is pressed."),
            kind="info") if any(missing.values()) else mo.md("")
        anch_ok = all(k_by[s_] == 0 for s_ in ("S1", "S2")) and k_by["S0"] > 0
        ema_f = [x["F_mae"] for x in A if x["scheme"] == "S4"] or [0]
        T2 = [x for x in R if x["set"] == "anchors" and x["tier"] == "T2"]
        t2 = {s_: (sum(x["fail"] for x in T2 if x["scheme"] == s_), sum(1 for x in T2 if x["scheme"] == s_)) for s_ in sch}
        t2_line = ("On T2 (charges respond to the environment): " + "; ".join(
            f"{D['runs']['scheme_names'][s_]} {k}/{n} failed" for s_, (k, n) in t2.items() if n)
            + "; the T2 MSE-anchor and EMA arms were not run.") if T2 else ""
        head = ("Both charge anchors remove the sign failures in the finished seeds; EMA does not." if anch_ok and k_by["S4"] > 0
                else "Failure rates per scheme in the finished seeds.")
        return mo.vstack([msg, f, mo.md(
            f"**Figure 7. {head}** "
            "Failure rate per scheme on T1 (replicate set, seeds 13-20) with 95 % Clopper-Pearson intervals; n is the "
            "number of finished seeds. Plain LES repeats the first set's outcome seed for seed (15 and 16 fail) and "
            "passes with the new seeds 17-20. EMA fails the same two seeds; its force MAE of "
            f"{min(ema_f):.0f}-{max(ema_f):.0f} meV/Å comes from averaging that starts at the random initial weights "
            "(about 2 % of them remain after 4,000 steps), so a fair EMA arm needs an EMA warm-up. "
            f"One-sided Fisher p = {C.fisher_greater(k_by['S0'], len(seeds['S0']), k_by['S1'], len(seeds['S1'])):.2f} for each "
            "anchor against plain LES: with eight seeds per arm this is a screen, not a test. The result does not reject "
            "the sign-trap reading; it does not yet confirm it either. " + t2_line)])
    _anch()
    return


@app.cell
def _(D, mo):
    _keys = sorted((D["traces"] or {}).keys())
    _lab = lambda k: {"anchors": "replicate", "ladder": "noise ladder"}.get(k.split("/")[0], "") + " · " + k.split("/")[-1].replace("T1_", "T1 ").replace("_s", " seed ")  # noqa: E731
    trace_pick = mo.ui.multiselect(options={_lab(k): k for k in _keys},
                                   value=[_lab(k) for k in _keys if k.startswith("anchors/T1_S0")][:4],
                                   label="probe traces")
    trace_pick
    return (trace_pick,)


@app.cell
def _(D, Fg, mo, plt, trace_pick):
    def _tr():
        if not D["traces"]:
            return mo.md("traces.json is missing.")
        f = Fg.traces(plt, D["traces"], trace_pick.value, label=lambda k: k.split("/")[-1].replace("T1_", "").replace("_s", " seed "))
        plt.close(f)
        cs = {k.split("/")[-1].replace("_s", " seed ").replace("_", " "): v["choice_step"] for k, v in D["traces"].items() if v}
        anch = [v for k, v in cs.items() if " S1 " in k or " S2 " in k]
        noan = [v for k, v in cs.items() if not (" S1 " in k or " S2 " in k)]
        late = {k: v for k, v in cs.items() if v > 100}
        lt = (" Runs that settle after step 100: " + ", ".join(f"{k} ({v})" for k, v in sorted(late.items())) + ".") if late else ""
        return mo.vstack([f, mo.md(
            f"**Figure 8. With a charge anchor, every run fixes the relative signs of its species by step "
            f"{max(anch) if anch else 0}; without one, the last relative-sign change falls between steps "
            f"{min(noan) if noan else 0} and {max(noan) if noan else 0}.** The 80-step warm-up is the dotted line.{lt} Probe of 32 fixed "
            f"training structures, every 10 steps (one epoch = 200 steps; log step axis). (a) Probe structures whose "
            f"charges mix signs. (b) Charged species (|q| > 0.25 e) at the wrong sign relative to the others, under the "
            f"probe's global sign; the tick marks the sign-choice step, the step after which this pattern no longer "
            f"changes.")])
    _tr()
    return


@app.cell
def _(D, NAME_Z, mo, pd):
    def _score():
        if not D["runs"]:
            return mo.md("")
        rows = []
        for x in D["runs"]["runs"]:
            rows.append({"set": {"phase1": "first", "anchors": "replicate", "ladder": "noise ladder"}[x["set"]],
                         "tier": x["tier"], "model": x["arm"], "scheme": D["runs"]["scheme_names"].get(x["scheme"], x["scheme"]),
                         "seed": x["seed"], "failed": x["fail"], "charge RMSE (e)": x["q_rmse"],
                         "force MAE (meV/Å)": x["F_mae"], "F MAE mixed": x["F_mae_mixed"], "F MAE coherent": x["F_mae_coherent"],
                         "mixed structures": x["n_mixed"],
                         "flipped species": ", ".join(NAME_Z.get(int(z), str(z)) for z in x["flipped"])})
        return mo.vstack([mo.md("### Run scorecard\nOne row per trained model; sort, filter or download from the table footer. "
                                "Charge RMSE is after the run's global sign; force MAE on the clean test split."),
                          mo.ui.table(pd.DataFrame(rows), page_size=50, selection=None)])
    _score()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The species-charge toy

    The networks are too large to probe directly. The toy keeps only what matters for the sign: 12 per-species
    charges $c$, fitted to the exact point-charge Coulomb energies and forces of the 200 T1 validation structures, with
    the trainer's loss (energy MSE per atom plus force MSE). At $c = \pm q_0$ the toy is exact. It is trained with Adam
    (warm-up, cosine decay, 600 steps) from 64 small random starts, $c \sim N(0, 0.05\,\mathrm{e})$, the same 64
    starts in every arm, so arms are paired.

    In the reference arm, 20 of the 64 starts end in one mixed-sign minimum. It flips one block of species
    (−2, −1 hard, −1 soft, −0.5 and +3) against the rest. The boundary between the blocks cuts both held-out pairs, and
    the flipped species of the failed networks are also held-out-pair members. The toy has no short-range network and no environment-dependent charges,
    so it shows that the data allow such a trap, not that the networks fall into the same one.
    """)
    return


@app.cell
def _(mo):
    toy_lr = mo.ui.slider(steps=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4], value=0.05, label="learning rate", show_value=True)
    toy_warm = mo.ui.slider(steps=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05, label="warm-up fraction", show_value=True)
    toy_b1 = mo.ui.slider(steps=[0.0, 0.5, 0.9, 0.95, 0.99], value=0.9, label="β1", show_value=True)
    toy_batch = mo.ui.slider(steps=[2, 4, 8, 16, 32, 64, 200], value=200, label="minibatch (200 = full)", show_value=True)
    toy_ema = mo.ui.slider(steps=[0.0, 0.9, 0.99, 0.999], value=0.0, label="EMA (0 = off)", show_value=True)
    toy_init = mo.ui.slider(steps=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.05, label="initial scale (e)", show_value=True)
    toy_anchor = mo.ui.dropdown(options={"none": "none", "MSE, exact": "mse", "correlation, exact": "corr"}, value="none", label="charge anchor")
    toy_wa = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.1, label="anchor weight", show_value=True)
    toy_combo = mo.ui.switch(value=False, label="+ combo structures")
    toy_go = mo.ui.run_button(label="Train 64 starts")
    mo.vstack([mo.md("### Live toy trainer\nReference settings: lr 0.05, warm-up 0.05, β1 0.9, full batch, no EMA, initial scale 0.05 e. "
                     "Full-batch runs take about 1 s; minibatches of 16-64 take 5-20 s."),
               mo.hstack([toy_lr, toy_warm, toy_b1, toy_batch, toy_ema], justify="start", wrap=True, gap=1),
               mo.hstack([toy_init, toy_anchor, toy_wa, toy_combo, toy_go], justify="start", wrap=True, gap=1)])
    return toy_anchor, toy_b1, toy_batch, toy_combo, toy_ema, toy_go, toy_init, toy_lr, toy_wa, toy_warm


@app.cell
def _(C, D, toy_anchor, toy_b1, toy_batch, toy_combo, toy_ema, toy_go, toy_init, toy_lr, toy_wa, toy_warm):
    import time as _time
    TOY = C.Toy(D["toy"]) if D["toy"] else None
    toy_result = None
    if TOY is not None:
        _D = C.Toy(D["toy"], combo=True) if toy_combo.value else TOY
        _t = _time.time()
        _c, _trace = C.train(_D, lr=toy_lr.value, warm=toy_warm.value, b1=toy_b1.value, batch=None if toy_batch.value >= 200 else int(toy_batch.value),
                             ema=toy_ema.value or None, init_scale=toy_init.value,
                             anchor=None if toy_anchor.value == "none" else toy_anchor.value, w_anchor=toy_wa.value)
        toy_result = dict(c=_c, trace=_trace, seconds=_time.time() - _t, clicked=toy_go.value)
    return TOY, toy_result


@app.cell
def _(C, D, Fg, TOY, TOY_NAMES, mo, plt, toy_result):
    def _live():
        if toy_result is None:
            return mo.callout(mo.md("toy.npz is missing."), kind="warn")
        c = toy_result["c"]; fl, rm, s = C.classify(c, TOY.q0); tr = fl.any(1)
        base = __import__("numpy").asarray(D["toy_ref"]["arms"]["base"]["trapped"]) if D["toy_ref"] else None
        f1 = Fg.toy_live(plt, c, toy_result["trace"], TOY.q0, TOY_NAMES); plt.close(f1)
        f2 = Fg.toy_arms(plt, D["toy_ref"]["arms"], live=(int(tr.sum()), len(tr))); plt.close(f2)
        cmp = ""
        if base is not None:
            cmp = (f" Against the reference arm (same 64 starts): {int((base & ~tr).sum())} starts rescued, "
                   f"{int((~base & tr).sum())} newly trapped.")
        return mo.vstack([
            mo.md(f"**Your run: {int(tr.sum())} of {len(tr)} starts trapped** ({toy_result['seconds']:.1f} s).{cmp}"), f1,
            mo.md("**Figure 9. Live run.** (a) Loss per start (purple: ends trapped; green: reaches the exact charges). "
                  "(b) How often each species ends with the wrong sign relative to the global sign. (c) Charge RMSE after "
                  "the global sign (e), log scale."), f2,
            mo.md("**Figure 10. Deterministic smoothing leaves the trap rate at 20/64; gradient noise and data that tie "
                  "the two blocks remove it.** Trapped starts of 64 per arm (dot) with the 95 % Clopper-Pearson interval "
                  "(bar); right: starts rescued from and newly sent into the trap relative to the reference arm (paired). "
                  "The 'ionicity bias' arms used 3 % and 6 % per-species charge errors, far above the DFT-calibrated level "
                  "(Section 7); read them as strong-bias arms. In the toy, charge anchors at weight 0.1 lower the trap rate "
                  "only to 14-18 of 64 and move different starts into the trap; with the networks, both anchors removed "
                  "the failures in the seeds finished so far (Figure 7). The live trainer's anchors use exact labels only.")])
    _live()
    return


@app.cell
def _(C, D, TOY, W, mo, np, toy_result):
    land_progress = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=1.0, label="training progress", show_value=True)
    def _land():
        if TOY is None or not D["toy_ref"]:
            return None, None
        trap = np.asarray(D["toy_ref"]["hess"]["trap_c"])
        a, L, uA, uB = C.slice_grid(TOY, trap, n=49, lim=1.5)
        runs = []
        for b in range(min(32, toy_result["c"].shape[0])):
            P = np.stack([C.project(t[1][b], uA, uB) for t in toy_result["trace"]])
            fl, _, s = C.classify(toy_result["c"][b:b + 1], TOY.q0)
            runs.append(dict(traj=np.round(P, 3).tolist(), trapped=bool(fl.any())))
        return dict(a=a.tolist(), L=np.round(L, 3).tolist(), runs=runs, lcap=60,
                    marks=[[1, 1, "trap"], [-1, -1, "trap (global mirror)"], [-1, 1, "repaired"], [1, -1, "repaired (mirror)"]]), trap
    LAND, TRAP_C = _land()
    land_progress
    return LAND, TRAP_C, land_progress


@app.cell
def _(Fg, LAND, TOY, TRAP_C, W, land_progress, mo, plt):
    def _lview():
        if LAND is None:
            return mo.md("")
        d = dict(LAND); d["progress"] = land_progress.value; d["distance"] = 26
        d["note"] = ("Loss on the plane through the trapped minimum: block A is the five flipped species, block B the rest. "
                     "(1, 1) is the trap; (−1, 1) flips block A back. Height: log(1 + loss). Tubes: the first 32 starts of "
                     "the live run projected onto the plane (purple: ends trapped; green: ends exact).")
        view = mo.ui.anywidget(W.Scene("landscape", data=d, elevation=35, azimuth=-30))
        f = Fg.barrier(plt, TOY, TRAP_C); plt.close(f)
        return mo.vstack([mo.hstack([view, f], widths=[1.6, 1]), mo.md(
            "**Figure 11. A loss barrier at zero charge separates the trap from its repaired copy.** Right: loss along "
            "the straight line that scales the flipped block from the trap (1) through zero to the repaired sign (−1), "
            "with block B held at the trap. The repaired point is not the exact minimum, because the trap's charge "
            "magnitudes differ from q0; the global minimum (loss ≈ 10⁻¹⁴) lies off this line.")])
    _lview()
    return


@app.cell
def _(D, Fg, TOY, TRAP_C, mo, np, plt):
    def _hess():
        if TOY is None:
            return mo.md("")
        e0 = np.linalg.eigvalsh(TOY.hessian(np.zeros(12))[0]); eq = np.linalg.eigvalsh(TOY.hessian(TOY.q0)[0])
        et = np.linalg.eigvalsh(TOY.hessian(TRAP_C)[0])
        h = D["toy_ref"]["hess"]
        f = Fg.hessian(plt, [("c = 0 (start)", e0), ("exact charges", eq), ("trap", et)]); plt.close(f)
        return mo.hstack([f, mo.md(
            f"**Figure 12. The start is a saddle; the trap is a strict, well-conditioned local minimum.** Eigenvalues "
            f"of the toy loss Hessian (symmetric-log axis; purple negative). At c = 0, {int((e0 < 0).sum())} of 12 "
            f"directions are unstable. Their span overlaps the exact sign pattern ({h['overlap_true_with_unstable']:.2f}) "
            f"and the trap pattern ({h['overlap_trap_with_unstable']:.2f}) almost equally, so the first descent "
            f"direction can lead to either. The most unstable direction leans to the exact pattern "
            f"({h['most_unstable_vs_true']:.2f} against {h['most_unstable_vs_trap']:.2f}). At the trap the smallest "
            f"eigenvalue is {et.min():.3f} > 0; condition number {et.max() / et.min():.0f} at the trap against "
            f"{eq.max() / eq.min():.0f} at the exact charges. Computed live by central differences.")], widths=[1, 1.2])
    _hess()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Conflicting gradients

    Why does the trap hold? At the trap the summed gradient is zero, but the per-structure gradients are not. Let $u$
    be the direction that scales the flipped species back through zero. A structure pulls towards repair if its
    descent direction has a positive component along $u$. Figure 13 splits the structures by whether they contain
    species from both sign blocks and shows how their pulls cancel. Section 5 showed what changes the outcome: noise in
    the gradient lowers the trap rate (minibatches of 4 structures: 7 of 64 trapped, of 16: 11 of 64, against 20), and
    data that tie the blocks together remove it (with the combo structures, none of 64 starts is trapped). These are toy
    results. In the network runs of Section 7 the same changes do not carry over at the tested strengths: batch 4 in the
    first epoch fails 1 of 8 seeds (the control fails 2 of 8), batch 4 throughout fails 0 of 4 in a screen that also
    takes 4× more optimiser updates, and adding the held-out pairs (3 % of the training structures, against 25 % in
    the toy) still fails 2 of 8, with different seeds.

    The same picture appears with real loss terms. In a separate bond-flow model trained on Born effective charges,
    the gradients of the force loss and of the Born-charge loss on the charge parameters point in opposite directions
    late in training (Figure 14). The three layers of the explanation are:

    1. **Ill-conditioning.** Energies and forces fix the charges only up to a sign per structure and a weak scale, so
       the loss has several basins separated by a barrier at q = 0 (Figures 11-12).
    2. **Gradient smoothing.** In the toy, deterministic smoothing (learning rate, warm-up, momentum, clipping, EMA) does
       not move a run between basins; the basin is decided by the start, and only stochastic gradient noise does
       (Figure 10). In the network runs, any change to the early optimisation (batch size, warm-up, data order)
       changes which seeds fall into a basin and does not remove the basins (Section 7).
    3. **Conflict.** Structures and loss terms pull the charges in different directions. Physically, "the charge"
       means different things to different labels: formal, partial, ESP-fitted, Löwdin, screened.

    DFT pathologies act on top as label noise; Section 7 sizes them.
    """)
    return


@app.cell
def _(C, D, Fg, TOY, TRAP_C, mo, np, plt):
    def _conf():
        if TOY is None:
            return mo.md("")
        _, G = TOY.per_struct(TRAP_C[None]); G = G[0]
        fA, fB = C.flip_blocks(TRAP_C, TOY.q0)
        u = np.where(fA, -TRAP_C, 0.0); u = u / np.linalg.norm(u)
        proj = -(G @ u)
        comp = TOY.counts[:TOY.n] > 0
        both = comp[:, fA].any(1) & comp[:, fB].any(1)
        ratio = np.linalg.norm(G, axis=1).sum() / (np.linalg.norm(G.sum(0)) + 1e-30)
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-30); cos = Gn @ Gn.T; iu = np.triu_indices(len(G), 1)
        f = Fg.conflict(plt, proj, both, G); plt.close(f)
        nb, no = int(both.sum()), int((~both).sum())
        return mo.vstack([f, mo.md(
            f"**Figure 13. At the trap, {len(G)} non-zero per-structure gradients add up to almost nothing: the "
            f"{nb} structures that contain both sign blocks pull away from repair in sum ({proj[both].sum():+.1f}), and the "
            f"{no} structures with one block pull towards it ({proj[~both].sum():+.1f}).** The sum of the per-structure "
            f"gradient norms is {ratio:,.0f} times the norm of their sum. Individually, {100 * (proj[both] > 0).mean():.0f} % of "
            f"the both-block structures and {100 * (proj[~both] > 0).mean():.0f} % of the one-block structures pull towards "
            f"repair; {100 * (cos[iu] < 0).mean():.0f} % of structure pairs have a negative gradient cosine. The plan for this "
            f"analysis expected the reverse grouping (both-block structures pulling towards repair). (a) Pull towards repair per structure, split by whether the structure contains species from both "
            f"sign blocks (blue) or from one block only (orange); clipped at the 98th percentile of |pull|. "
            f"(b) Cosine between the gradients of every pair of structures, ordered by pull (purple negative, green "
            f"positive). Toy, T1 validation structures, computed live.")])
    _conf()
    return


@app.cell
def _(D, Fg, mo, plt):
    def _bf():
        if not D["conflict"]:
            return mo.md("conflict.json is missing.")
        f = Fg.bf_cos(plt, D["conflict"]["bf"]); plt.close(f)
        return mo.vstack([f, mo.md(
            "**Figure 14. On SiO2 with the Coulomb term on, the median late cosine between the force-loss and "
            "Born-charge-loss gradients on the charge parameters is negative in every arm (two arms near zero); the few "
            "water and BaTiO3 seeds scatter on both sides.** Bond-flow charge model (fp64) on SiO2, BaTiO3 and water, "
            "trained on forces and Born effective charges; cosine of the two loss gradients restricted to the charge "
            "parameters, median over steps after 300 (filled) and up to 300 (open), one dot per seed (n = 1-8 per arm; "
            "arms with 2 seeds are screens). Arm names: iris = reference recipe, splr30 = charge learning rate ×30, "
            "coul0 / coul_detach = Coulomb term off or detached, wZ10 = Born-charge weight 10, sig05 = charge width "
            "0.5 Å, clip10 = gradient clip 10, spinit_elem / gate1 = charges started at the element mean or gate 1. Arms "
            "without a logged cosine (Coulomb term off or detached, and class E) are not drawn; the Coulomb-off and "
            "detached arms reach the lowest Born-charge errors (0.13-0.16 e against 0.22-0.31 e for the reference, "
            "2 seeds each).")])
    _bf()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Label noise at the DFT level: the noise ladder

    How large is the noise in DFT labels such as OMol25, and does it matter for the sign trap? Three parts are kept
    apart. **White** numerical noise is small: OMol25 rows recomputed in a second code differ by a median of
    1.4 meV/Å (n = 4,866), so the 1× level is set to σ_F = 2 meV/Å (uncertain by a factor of about 3). A
    **pathology tail** of wrong SCF states affects about 1 % of frames with 20-500 meV/Å errors. A **systematic**
    part (the functional's error) is set by the force error it produces, about 30 meV/Å; it is a literature-based
    estimate, not a measurement, and is uncertain by a factor of 2-5. Charge labels for the anchors get their own
    noise: per-species bias, per-atom random noise, a tail of charge-transfer-ambiguous structures, and one
    sign-crossing arm.
    """)
    return


@app.cell
def _(D, Fg, mo, pd, plt):
    def _noise():
        if not D["conflict"]:
            return mo.md("")
        f = Fg.noise_scale(plt, D["conflict"]["noise_rows"]); plt.close(f)
        V = D["conflict"]["variants"]
        rows = [{"variant": k, "E/F level": v["ef_m"], "E/F parts": v["ef_comps"] if v["ef_m"] else "–",
                 "charge-label level": v["q_m"], "charge-label parts": v["q_comps"] if v["q_m"] else ("sign-crossing" if v["cross"] else "–"),
                 "train ΔF RMS (meV/Å)": v["dF_rms"], "train ΔE RMS (meV/atom)": v["dE_rms"], "train Δq RMS (e)": v["dq_rms"]}
                for k, v in V.items()]
        return mo.vstack([f, mo.md(
            "**Figure 15. The 1× white level sits 15× below the force error of T1 runs that learn; the systematic part "
            "and the pathology tail are what reach the model's scale.** DFT-side numbers (black): differences between "
            "codes, settings or functionals (triangles: upper bounds; bars: ranges). ChargeWorld-side numbers (blue): "
            "noise levels and validation force RMSE of the first T1 A1 set. Log axis, meV/Å."),
            mo.md("**Noise variants of the T1 training data, as measured on the train split.** Test splits and exact "
                  "charges are byte copies of the clean data. The systematic part dominates E/F 1× and 2×, so E/F 1× is "
                  "not the near-null that white noise alone would be."),
            mo.ui.table(pd.DataFrame(rows), selection=None, page_size=15)])
    _noise()
    return


@app.cell
def _(D, Fg, mo, plt):
    def _ladder_runs():
        R = [x for x in ((D["runs"] or {}).get("runs", [])) if x["set"] == "ladder"]
        arms = ["EF0.5", "EF1", "EF2", "W25", "P5", "B4E1", "B4", "B64", "WU10", "COMBO"]
        if not R:
            return mo.callout(mo.md(
                "**Running.** Network runs of the noise ladder on T1 (plain LES, seeds 13-20, paired with the replicate "
                "set's S0): E/F noise at 0.5×, 1× and 2×, white noise at 25× and the pathology tail at 5× (positive "
                "controls); and the smoothing and data arms suggested by the toy: batch 4 in the first epoch, batch 64, "
                "warm-up 10 % and the held-out pairs added to training. This cell fills in when the run folders appear "
                "under the ChargeWorld run folder and the rebuild button at the top is pressed. Charge-label noise arms "
                "follow once the anchor results are complete."), kind="neutral")
        names = dict(D["runs"]["ladder_names"]); names["S0"] = "plain LES (reference, replicate set)"
        Rall = [dict(x, set="ladder", scheme="S0") for x in D["runs"]["runs"] if x["set"] == "anchors" and x["tier"] == "T1" and x["scheme"] == "S0"] + \
               [dict(x, scheme=x["arm"]) for x in R]
        f = Fg.rates(plt, Rall, "ladder", "scheme", ["S0"] + [a for a in arms if any(x["arm"] == a for x in R)], names); plt.close(f)
        return mo.vstack([f, mo.md(
            "**Figure 16. Noise-ladder failure rates.** Failure rate (sign rule) per arm with 95 % Clopper-Pearson "
            "intervals, T1, plain LES; n = finished seeds. Batch-4 arms take 4× more optimiser updates per epoch than "
            "the reference, batch 64 4× fewer, and the combo arm trains on 3 % more structures.")])
    _ladder_runs()
    return


@app.cell
def _(C, D, mo):
    def _ladder_results():
        R = (D["runs"] or {}).get("runs", [])
        L = [x for x in R if x["set"] == "ladder"]
        if not L:
            return mo.md("")
        ctrl = {x["seed"]: x for x in R if x["set"] == "anchors" and x["tier"] == "T1" and x["scheme"] == "S0"}
        names = D["runs"]["ladder_names"]
        order = [a for a in ("EF0.5", "EF1", "EF2", "W25", "P5", "B4E1", "B4", "B64", "WU10", "COMBO") if any(x["arm"] == a for x in L)]
        med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else 0.5 * (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2])
        k0 = sum(bool(x["fail"]) for x in ctrl.values())
        bullets, same_noise = [], True
        for a in order:
            v = sorted([x for x in L if x["arm"] == a], key=lambda x: x["seed"])
            n, k = len(v), sum(bool(x["fail"]) for x in v)
            lo, hi = C.cp_interval(k, n)
            fs = [x["seed"] for x in v if x["fail"]]
            agree = sum(bool(x["fail"]) == bool(ctrl[x["seed"]]["fail"]) for x in v if x["seed"] in ctrl)
            good = [x["F_mae"] for x in v if not x["fail"] and x.get("F_mae") is not None]
            if a in ("EF0.5", "EF1", "EF2", "W25", "P5") and agree != n:
                same_noise = False
            bullets.append(f"- **{names.get(a, a)}:** {k} of {n} fail ({'seeds ' + ', '.join(map(str, fs)) if len(fs) > 1 else 'seed ' + str(fs[0]) if fs else 'no seed'}; 95 % "
                           f"Clopper-Pearson {100 * lo:.0f}-{100 * hi:.0f} %); same outcome as the paired control in {agree} of "
                           f"{n} seeds; median clean-test force MAE of the runs with the right signs "
                           f"{med(good):.1f} meV/Å." if good else f"- **{names.get(a, a)}:** {k} of {n} fail.")
        b4 = [x for x in L if x["arm"] == "B4"]
        b4txt = ""
        if b4:
            kb = sum(bool(x["fail"]) for x in b4)
            kc = sum(bool(ctrl[x["seed"]]["fail"]) for x in b4 if x["seed"] in ctrl)
            b4txt = (f" Batch 4 throughout fails {kb} of {len(b4)} where its paired control fails {kc} of {len(b4)}: both "
                     f"control-failing seeds are rescued. This is a screen (one-sided Fisher p = "
                     f"{C.fisher_greater(kc, len(b4), kb, len(b4)):.2f}), and it takes 4 times more optimiser updates than the "
                     f"reference at the same epochs and learning rate, so the rescue and the lower force error are not "
                     f"separated from training length.")
        return mo.vstack([mo.md(
            f"**Results (replicate-set control: plain LES fails {k0} of {len(ctrl)}, seeds "
            f"{', '.join(str(s_) for s_, x in sorted(ctrl.items()) if x['fail'])}).**\n\n" + "\n".join(bullets)
            + "\n\n" + ("Noise at every tested level (E/F 0.5× to 2×, white 25×, pathology tail 5×) fails the same seeds as the "
                        "control, so label noise at these levels does not change which basin a seed falls into; it costs force "
                        "accuracy, through the systematic part only (the white and tail positive controls leave the force error "
                        "unchanged). " if same_noise else "")
            + "Warm-up changes which seeds fail, not how many. Batch 4 in the first epoch rescues seed 16 and adds no new "
              "failure; adding the held-out pairs rescues seeds 15 and 16 and lets seeds 19 and 20 fail. Batch 64 is not readable: at the same epochs and learning rate its runs are far from converged." + b4txt
            + " No contrast reaches significance: with 2 of 8 control failures no rate comparison can reach p < 0.05.")])
    _ladder_results()
    return


@app.cell
def _(C, D, mo, pd):
    def _caveats():
        R = (D["runs"] or {}).get("runs", [])
        def kn(sel):
            v = [x["fail"] for x in R if sel(x) and x.get("fail") is not None]
            return sum(v), len(v)
        k1, n1 = kn(lambda x: x["set"] == "phase1" and x["tier"] == "T1" and x["arm"] == "A1 mid")
        k2, n2 = kn(lambda x: x["set"] == "phase1" and x["tier"] == "T2" and x["arm"] == "A1 mid")
        ks, ns = kn(lambda x: x["set"] == "anchors" and x["tier"] == "T1" and x["scheme"] == "S0")
        k_1, n_1 = kn(lambda x: x["set"] == "anchors" and x["tier"] == "T1" and x["scheme"] == "S1")
        k_2, n_2 = kn(lambda x: x["set"] == "anchors" and x["tier"] == "T1" and x["scheme"] == "S2")
        lo, hi = C.cp_interval(k1, n1) if n1 else (0, 0)
        seeds = mo.md(
            f"**Seeds.** Every network claim rests on four seeds per arm (first set) or on the finished part of eight "
            f"(replicate set). T1 A1 fails in {k1} of {n1} seeds: the 95 % Clopper-Pearson interval is "
            f"{100 * lo:.0f}-{100 * hi:.0f} %. T1 against T2 ({k2} of {n2}) gives a two-sided Fisher exact p = "
            f"{C.fisher_two_sided(k1, n1, k2, n2):.2f}. In the replicate set, plain LES fails {ks} of {ns}, the correlation "
            f"anchor {k_1} of {n_1} and the MSE anchor {k_2} of {n_2}; one-sided Fisher p = {C.fisher_greater(ks, ns, k_1, n_1):.2f} "
            f"and {C.fisher_greater(ks, ns, k_2, n_2):.2f} for plain LES failing more often. With eight "
            f"seeds per arm these comparisons are screens.")
        rows = []
        for x in sorted([x for x in R if x["set"] == "phase1" and x["tier"] == "T1" and x.get("splits")], key=lambda r: (r["arm"], r["seed"])):
            sp = x["splits"]
            if not {"test", "Qood"} <= set(sp):
                continue
            ratio = sp["Qood"]["F_mae"] / sp["test"]["F_mae"]
            row = {"model": x["arm"], "seed": x["seed"], "Qood / test force MAE": round(ratio, 2)}
            if x["arm"].startswith("A1"):
                qt = max(sp["test"]["q_err_by_Z"].values()) if sp["test"]["q_err_by_Z"] else None
                qc = max(v for z, v in sp.get("combo", {}).get("q_err_by_Z", {}).items() if int(z) in (7, 8, 12, 13)) if sp.get("combo") else None
                row.update({"largest species charge error, test (e) [≤ 0.05]": qt,
                            "largest held-out-pair charge error, combo (e) [≤ 0.1]": qc})
                ok = (qt is not None and qt <= 0.05) and (qc is not None and qc <= 0.1) and ratio <= 1.5
            else:
                ok = ratio >= 2
            row["limit"] = "A1: Qood ratio ≤ 1.5" if x["arm"].startswith("A1") else "A0: Qood ratio ≥ 2"
            row["all limits met"] = ok
            rows.append(row)
        nmiss = sum(not r["all limits met"] for r in rows)
        lim = mo.vstack([mo.md(
            f"**Pre-registered T1 limits.** Before training, the test plan set limits for T1: A1 charge error at most "
            f"0.05 e on every species and at most 0.1 e on the held-out pairs, and a Qood-to-test force MAE ratio of at most "
            f"1.5 for A1 and at least 2 for A0 (A0 has no charges and should fail to transfer). {nmiss} of {len(rows)} "
            f"first-set T1 runs miss at least one limit. The two A1 seeds that pass the sign rule miss on the held-out pairs "
            f"(largest per-species charge error on combo above 0.1 e) or on the species limit, and their Qood ratio is 3.0-4.5; "
            f"A0 transfers better than predicted (ratio 1.3)."),
            mo.ui.table(pd.DataFrame(rows), selection=None, page_size=10)]) if rows else mo.md("")
        rest = mo.md(r"""
    **Transfer.** All T1 A1 seeds lose accuracy on the test-only splits (Figure 4). The two A1 seeds that pass the sign
    rule rise from 20-24 meV/Å on the test split to 238-282 meV/Å on `combo` and 59-108 meV/Å on `Qood`; their charges
    stay close on combo (sign-corrected charge RMSE 0.07-0.10 e), so the combo error is not a sign problem. On T2, the
    A1 Qood error (114-126 meV/Å) is close to its test error (100-104 meV/Å).

    **Noise calibration.** The 1× white level is uncertain by a factor of about 3, the tail fraction by about 3, and
    the systematic part by 2-5. The toy's 3 % and 6 % ionicity arms are strong-bias arms, not 1× and 2×.

    **The toy is not the network.** It has fixed per-species charges and no short-range part. It shows that the data
    permit a trap whose block boundary cuts the held-out pairs; it does not show that the networks' trap is the same
    minimum.

    **Not shown.** Tiers T3-T5 have no trained models yet. The explorer shows 6-8 test structures per tier, not the
    full splits. The DFT-noise numbers come from separate recomputations, summarised here without their raw data.

    **Route-B models have no sign trap.** The sign audit of the six route-B arms (A1, P, C0 with one frozen charge
    chain; AL, PAL, B with another) finds no wrong-sign species against DFT Löwdin charges (per-atom RMSE 0.018-0.035 e),
    and none of the sign trap seen here: their charges come from two frozen chains, so there are 2 charge functions, not
    24, and no charge head is trained inside those runs. Against the fixed-ESP oracle, C and N of Gdm⁺ carry the opposite sign
    in every model (the ESP charges are +1.0 and −0.9 e, Löwdin −0.4 and +0.2 e); that is a disagreement between two
    oracles, and the models follow the DFT Löwdin one. It does not explain force or interaction-energy errors in that
    data. The audit covers Gdm⁺ and NH₄⁺ pair sets, not organic molecules in general.

    **Next.** Charge-label noise on the anchors (paired on seeds 15 and 16: do noisy anchors still rescue
    them?), a combo arm with 25 % co-occurring structures (the toy's fraction), a batch-64 arm at 4 times the epochs
    (so batch size is separated from training length), and batch 4 at equal numbers of updates. Confirm the anchor screen
    with more seeds. Measure the loss barrier between a failed network and its sign-repaired copy directly (the network analogue of
    Figure 11). Train on T2-T5, where charges respond to the environment and the fixed-charge oracle fails.
    """)
        return mo.vstack([mo.md("## 8. Caveats, what is not shown, and next steps"), seeds, lim, rest])
    _caveats()
    return


if __name__ == "__main__":
    app.run()
