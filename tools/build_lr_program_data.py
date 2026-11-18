#!/usr/bin/env python3
"""Distil the PET-OMol-LR program results into the notebooks' one data folder (notebooks/data/lr_program).

    python tools/build_lr_program_data.py [--out notebooks/data/lr_program] [--only program|scorecard] [--quick]
    bash tools/rebuild_lr_program.sh            # the same, plus a summary of what is scored and what is pending

One builder, one folder, two notebooks: `program.json` and `chemiscope.json` feed long_range_diagnostics.py; the
scorecard stage (tools/lrp_scorecard.py) writes index.json, prog.json, bench.json, md.json, speed.json,
typeheld_*.json and chem_*.npz for model_scorecard.py (through notebooks/lrb_data.py). After every build the files
are rewritten to public labels (<arm>_s<seed>_<ema|raw>, hardware by GPU type) and gated (tools/lrp_public.py; the
label map and the list of forbidden strings are private and live outside the repository, LRP_PUBLIC_MAP).

The program results are not lrbench result rows: they are scorecards of fine-tuned PET-OMol-S models (route B, the
C0 / P / C0H / A0 / A1 arms and the CEIL screens) on the program's own gates. They do not fit the set/group schema, so
they travel as one compact JSON (`program.json`) and one file of chemiscope datasets (`chemiscope.json`), which the
notebook reads from `mo.notebook_dir()/data/lr_program/` (live, molab) or from its EMBEDDED literal (frozen copy).

Every number is read from a result file; nothing is retyped except where a result exists only as text, and those
entries carry their source path in `source`. Sources (default roots under ~/data):

    lr-fix-likecharge/seam/final/sc/scorecard.json     59 labels, scorecard.py keys
    lr-fix-likecharge/seam/final/cells_<label>.json     rule-4 sign-set cells (COM axis, 4 scorable + K+Gdm+ apart)
    lr-fix-likecharge/seam/final/sc/<label>/*.json      per-structure model outputs (forces, energies, charges)
    lr-fix-likecharge/seam/final/res/*.json             search-set and 8-frame adversarial set model outputs
    lr-fix-likecharge/cl/cl_heldout.{xyz,meta.json}     cluster ladders (DFT dE(n)); scorer cl/cl_gate.py
    lr-fix-likecharge/sched/rule6/rule6_summary_sched.json   ladders at n = 1,000-4,000 waters
    lr-fix-likecharge/sched/nve/*.json, sched/cons/*.txt      NVE drift, conservation suite
    lr-fix-likecharge/typeheld/                          type-held-out charged-pair DFT gate
    mdstab/results/*.json, mdstab/figs/rdf_*.json        MD stability, charges in MD, RDF peaks
    permit/analysis_pet2/permit_eps.json, permit/analysis_final/permit_eps.json   permittivity
    lr-fix-likecharge/scans_al/results_*.json, chemiscope_*.json, fig*.png   1-D scans of AL vs PET-OMol-S, P, C0

Nothing under results/private is read. Third-party sets (ext_*) are not embedded as structures.
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime
import glob
import gzip
import json
import os
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("LRNB_DATA_ROOT", "~/data")).expanduser()
LF = DATA / "lr-fix-likecharge"
SF = LF / "seam" / "final"
KE = 14.399645
SEEDS = ["13r", "14", "15", "16"]


MULTI = ("Mg", "Ca", "Zn", "SO4")


def maxq_of(name) -> int:
    """largest |formal charge| of any species in a system or test item (the multivalent switch keys on >= 2)"""
    return 2 if any(x in str(name) for x in MULTI) else 1


def rel(p) -> str:
    return str(p).replace(str(pathlib.Path.home()), "~")


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ rows (arm x file)
def row_defs():
    """Arm x weights x setup. A row collects the labels (one per seed) that form one comparison unit."""
    rows = {}
    setup4 = "GH200 training, seeds 13r/14/15/16; scored on RTX 4070 Ti SUPER (fp32, ragged, no TF32)"
    arms = {
        "C0": "route-B control (12,000 steps, cosine anneal, IPR4 x4)",
        "P": "C0 + seam pair-prior rows (data5p)",
        "C0H": "C0 with lr held at 30 % of peak, EMA 0.9995 without warm-up",
        "A0": "P init + 6,000 steps, N = 16 neighbours",
        "A1": "P init + 6,000 steps, N = 32 neighbours",
    }
    for a, desc in arms.items():
        for w, tag in (("EMA", "cl"), ("raw", "clraw")):
            rows[f"{a} {w}"] = dict(arm=a, weights=w, setup=setup4, n_seeds=4, evidence="claim (4 seeds)",
                                    desc=desc, labels=[f"{a}{tag}_s{s}" for s in SEEDS], seeds=list(SEEDS),
                                    group="4-seed arms")
    h100 = "H100 training, seed 13 (same-seed replicate of GH200 13r, not an extra seed)"
    for a, e, r in (("C0", "C0fema", "C0fraw"), ("P", "Pfema", "Pfraw")):
        rows[f"{a} EMA, H100 s13"] = dict(arm=a, weights="EMA", setup=h100, n_seeds=1, evidence="screen (1 seed)",
                                          desc=arms[a], labels=[e], seeds=["13"], group="replicates")
        rows[f"{a} raw, H100 s13"] = dict(arm=a, weights="raw", setup=h100, n_seeds=1, evidence="screen (1 seed)",
                                          desc=arms[a], labels=[r], seeds=["13"], group="replicates")
    for a in ("A0", "A1"):
        rows[f"{a} EMA, H100 s13"] = dict(arm=a, weights="EMA", setup=h100, n_seeds=1, evidence="screen (1 seed)",
                                          desc=arms[a], labels=[f"{a}fema"], seeds=["13"], group="replicates")
    rows["ORIG"] = dict(arm="ORIG", weights="none (final iterate, no EMA)", setup="H100, seed 13 (Bq4m3_c01, cut "
                        "at 12,000 of a 14,170-step cosine, no IPR4)", n_seeds=1, evidence="screen (1 seed)",
                        desc="route-B starting point before the seam arms", labels=["ORIG"], seeds=["13"],
                        group="screens")
    ceil = {"CEILA": "CEIL arm A: energy tower without total-charge embedding",
            "CEILB": "CEIL arm B", "CEILC": "CEIL arm C",
            "CEILAL": "CEIL arm AL: lr 3e-5, 24,000 steps (A36)"}
    for a, desc in ceil.items():
        for w, suf in (("EMA", "fema"), ("raw", "fraw")):
            rows[f"{a} {w}"] = dict(arm=a, weights=w, setup="GH200, seed 13, CEIL build; scored on "
                                    "GH200 (padded cached path)", n_seeds=1, evidence="screen (1 seed)",
                                    desc=desc, labels=[f"{a}{suf}"], seeds=["13"], group="screens")
    # four-seed repeats, registered before their results exist (A38: AL seeds 14-16; A39: PAL seeds 13-16). A row
    # keeps only the labels that are scored on disk (prune_rows); the rest are listed as pending, never as numbers.
    for w, suf in (("EMA", "fema"), ("raw", "fraw")):
        r = rows[f"CEILAL {w}"]
        r["labels"] = [f"CEILAL{suf}"] + [f"CEILAL{suf}_s{s}" for s in ("14", "15", "16")]
        r["seeds"] = ["13", "14", "15", "16"]
        r["group"] = "4-seed arms"
        rows[f"PAL {w}"] = dict(arm="PAL", weights=w, setup="GH200 training, seeds 13-16, AL recipe (CEIL build)",
                                n_seeds=4, evidence="claim (4 seeds)", group="4-seed arms",
                                desc="P's band-prior rows (data5p) with AL's Q-free ceiling tower, lr 3e-5, 24,000 steps (A39)",
                                labels=[f"PAL{suf}_s{s}" for s in ("13", "14", "15", "16")],
                                seeds=["13", "14", "15", "16"])
    return rows


def scored(label, SC) -> bool:
    return label in SC or (SF / "sc" / label / "val500.json").exists()


def prune_rows(rows, SC):
    """keep the scored labels of each row; record the rest as pending; drop rows with nothing scored"""
    out, pending = {}, {}
    for rk, r in rows.items():
        keep = [(L, s) for L, s in zip(r["labels"], r["seeds"]) if scored(L, SC)]
        miss = [L for L in r["labels"] if not scored(L, SC)]
        if miss:
            pending[rk] = miss
        if not keep:
            continue
        r = dict(r)
        r["labels"], r["seeds"] = [k[0] for k in keep], [k[1] for k in keep]
        r["n_seeds"] = len(keep)
        r["evidence"] = "claim (4 seeds)" if len(keep) >= 4 else f"screen ({len(keep)} seed{'s' if len(keep) > 1 else ''})"
        if r["group"] == "4-seed arms" and len(keep) < 4:
            r["group"] = "screens"
        out[rk] = r
    return out, pending


def merged_scorecard():
    sys.path.insert(0, str(HERE / "tools"))
    from lrp_scorecard import merged_scorecard as _m
    return _m()


# --------------------------------------------------------------------------------------------------- metrics
#: (key in the scorecard, short label, unit, family, direction)
SC_METRICS = [
    ("val500 F MAE", "val500 F MAE", "meV/Å", "accuracy", "low"),
    ("val500 E MAE", "val500 E MAE", "meV/atom", "accuracy", "low"),
    ("val500 F MAE, LES off", "val500 F MAE, LES off", "meV/Å", None, "low"),
    ("setB held-out r>=8A F MAE meV/A", "pairs gate F MAE (r ≥ 8 Å)", "meV/Å", "pairs", "low"),
    ("setB held-out r>=8A |dE_int| median eV", "pairs gate |ΔE_int| median", "eV", "pairs", "low"),
    ("setB held-out r>=8A F MAE meV/A, pair types not in P rows", "pairs gate F MAE, types not in P rows",
     "meV/Å", None, "low"),
    ("setB held-out r>=8A F MAE meV/A, LES off", "pairs gate F MAE, LES off", "meV/Å", None, "low"),
    ("q95|dF| NH4_Gdm 9-12A", "NH4⁺Gdm⁺ 9–12 Å q95 |ΔF|", "eV/Å", "seam", "low"),
    ("q95|dF| Gdm_OAc 9-12A", "Gdm⁺OAc⁻ 9–12 Å q95 |ΔF|", "eV/Å", "seam", "low"),
    ("nacl1m interior F MAE vs DFT meV/A", "NaCl(aq) cut-outs, interior F MAE", "meV/Å", "condensed", "low"),
    ("nacl1m interior Na shell: max abs per-ion error", "Na⁺ shell charge, max |error|", "e", "condensed", "low"),
    ("nacl1m interior Cl shell: max abs per-ion error", "Cl⁻ shell charge, max |error|", "e", "condensed", "low"),
    ("droplet: F MAE + full LES", "droplet F MAE", "meV/Å", "condensed", "low"),
    ("cut-outs interior: F MAE + full LES", "cut-outs interior F MAE (LES)", "meV/Å", None, "low"),
    ("frag RMSE", "fragment charge RMSE", "e", "charges", "low"),
    ("per-atom RMSE", "per-atom charge RMSE", "e", "charges", "low"),
    ("nacl1m per-atom q RMSE vs Lowdin", "NaCl(aq) per-atom q RMSE vs Löwdin", "e", "charges", "low"),
]
FAMILIES = {
    "accuracy": "near-equilibrium accuracy (val500)",
    "seam": "seam band (sign-set cells, overshoot, far-field force spread)",
    "pairs": "charged pairs beyond 8 Å (set-B gate)",
    "ladders": "ion–water cluster ladders (slope of the error)",
    "condensed": "condensed phase (NaCl(aq) cut-outs, droplet, ion shells)",
    "charges": "charges against DFT (Löwdin)",
}


def ladder_curves(label):
    """dE(n) error per ion, centre and n, exactly as cl/cl_gate.py, from sc/<label>/cl_heldout.json."""
    js = SF / "sc" / label / "cl_heldout.json"
    P = jload(js)
    if P is None:
        return None
    M = _CL["meta"]

    def curves(E):
        X, W, XW = {}, {}, {}
        for i, m in enumerate(M):
            if m["role"] == "X" and m["sys"] not in X:
                X[m["sys"]] = E[i]
            elif m["role"] == "W":
                W[(m["sys"], m["centre"], m["nw"])] = E[i]
            elif m["role"] == "XW":
                XW[(m["sys"], m["centre"], m["nw"])] = E[i]
        return {k: XW[k] - W[k] - X[k[0]] for k in XW if k in W and k[0] in X}

    D = curves([m["E"] for m in M])
    C = curves([p.get("energy", float("nan")) for p in P["frames"]])
    out = {}
    for ion in sorted({str(m["sys"]) for m in M if m["role"] == "XW"}):   # Na, Cl, Mg gated; others reported
        err = collections.defaultdict(list)
        for (s, c, n), v in C.items():
            if s == ion and (s, c, n) in D and v == v:
                err[n].append(v - D[(s, c, n)])
        ns = sorted(err)
        mean = {int(n): float(np.mean(err[n])) for n in ns}
        fit = [n for n in ns if n >= 24]
        slope = float(np.polyfit(fit, [mean[n] for n in fit], 1)[0]) if len(fit) >= 2 else None
        per_frame = {f"{c}:{n}": float(v - D[(s, c, n)]) for (s, c, n), v in C.items() if s == ion and (s, c, n) in D}
        out[ion] = dict(err=mean, slope=slope, per_frame=per_frame)
    return out


def cells(label):
    d = jload(SF / f"cells_{label}.json")
    if d is None:
        return None
    srch = [k for k in d if not k.startswith("K+")]
    return dict(cells=d, n_pass=sum(bool(d[k]["pass"]) for k in srch), n_cells=len(srch))


def _res(kind, label, tag="on"):
    g = sorted(glob.glob(str(SF / "res" / f"{kind}F_{label}_s*_{tag}.json")))
    g = [x for x in g if not os.path.basename(x).startswith(f"{kind}F_{label}_cl_")]
    return g[0] if g else None


def adv(label):
    """8-frame adversarial sign set (adv_an.py): projected force on B against the Coulomb-oracle value."""
    p = _res("adv_", label)
    if p is None:
        return None
    P = jload(p)["frames"]
    out = []
    for m, a, fr in zip(_ADV["meta"], _ADV["frames"], P):
        R = a.positions
        nA = m["nA"]
        u = R[nA:].mean(0) - R[:nA].mean(0)
        u /= np.linalg.norm(u)
        F = float(np.array(fr["forces"]).reshape(-1, 3)[nA:].sum(0) @ u)
        out.append(dict(pair=m["pair"], r=m["r"], F=F, orc=m.get("orc")))
    return out


def build_program():
    SC = merged_scorecard() or {}
    rows, pending = prune_rows(row_defs(), SC)
    metrics = {}
    for key, lab, unit, fam, better in SC_METRICS:
        vals = {}
        for rk, r in rows.items():
            v = [SC.get(L, {}).get(key) for L in r["labels"]]
            vals[rk] = [None if x is None else float(x) for x in v]
        metrics[key] = dict(label=lab, unit=unit, family=fam, better=better, values=vals, source="scorecard.json")
    # sign-set cells (COM axis; 4 scorable search cells, K+Gdm+ apart; program decision 2026-11-17 06:56)
    cellrows = {}
    for rk, r in rows.items():
        cellrows[rk] = [cells(L) for L in r["labels"]]
    metrics["cells passing"] = dict(
        label="sign-set cells passing (of 4)", unit="cells", family="seam", better="high", source="cells_<label>.json",
        values={rk: [None if c is None else c["n_pass"] for c in v] for rk, v in cellrows.items()})
    for cell in ("Gdm+_Gdm+ 8", "Gdm+_Gdm+ 9"):
        metrics[f"att {cell}"] = dict(
            label=f"{cell.replace('_', '·')} Å attractive share", unit="%", family="seam", better="low",
            source="cells_<label>.json",
            values={rk: [None if c is None else c["cells"][cell]["att"] for c in v] for rk, v in cellrows.items()})
    for cell in ("NH4+_NH4+ 8", "NH4+_Gdm+ 7.5"):
        metrics[f"over {cell}"] = dict(
            label=f"{cell.replace('_', '·')} Å above 1.5× Coulomb", unit="%", family="seam", better="low",
            source="cells_<label>.json",
            values={rk: [None if c is None else c["cells"][cell]["over15"] for c in v] for rk, v in cellrows.items()})
    # 8-frame adversarial set: NH4+Gdm+ 7.5 A overshoot (x oracle)
    advrows = {rk: [adv(L) for L in r["labels"]] for rk, r in rows.items()}

    def _ov(a, pair, rr):
        if a is None:
            return None
        x = [z for z in a if z["pair"] == pair and abs(z["r"] - rr) < 0.05]
        return None if not x or not x[0]["orc"] else x[0]["F"] / x[0]["orc"]

    metrics["overshoot NH4 Gdm 7.5"] = dict(
        label="NH4⁺Gdm⁺ 7.5 Å overshoot (× oracle)", unit="×", family="seam", better="target1",
        source="res/adv_*_on.json + adv.meta.json (adv_an.py)",
        values={rk: [_ov(a, "NH4_Gdm", 7.5) for a in v] for rk, v in advrows.items()})
    # ladders
    ladd = {rk: [ladder_curves(L) for L in r["labels"]] for rk, r in rows.items()}
    for ion in ("Na", "Cl", "Mg"):
        metrics[f"ladder slope {ion}"] = dict(
            label=f"{ion} ladder slope |error| (n ≥ 24)", unit="eV/water", better="abs0",
            # program decision 2026-11-17: Mg2+ is excluded from scoring (shown apart, never in a rank or count)
            family=None if ion == "Mg" else "ladders", excluded="Mg2+" if ion == "Mg" else None,
            source="sc/<label>/cl_heldout.json + cl/cl_heldout.meta.json (cl_gate.py)",
            values={rk: [None if c is None else c[ion]["slope"] for c in v] for rk, v in ladd.items()})
    ladder_err = {rk: [None if c is None else {ion: c[ion]["err"] for ion in c} for c in v] for rk, v in ladd.items()}

    # cells detail for figures
    cell_detail = {rk: [None if c is None else c["cells"] for c in v] for rk, v in cellrows.items()}
    adv_detail = {rk: v for rk, v in advrows.items()}

    # ladders at n = 1,000-4,000 (rule 3 of A29a): P and C0 EMA, seeds 13r and 14
    big = jload(LF / "sched" / "rule6" / "rule6_summary_sched.json") or {}
    big_out = {k: {ion: dict(slope=v["ions"][ion]["slope_1000_4000"],
                             err={int(n): e for n, e in v["ions"][ion]["err_by_n"].items()})
                   for ion in v.get("ions", {})} for k, v in big.items()}

    # NVE + conservation (rule 6 of A29a), P and C0 EMA, MD form
    nve = {}
    for p in sorted(glob.glob(str(LF / "sched" / "nve" / "*.json"))):
        d = jload(p)
        if not d or "checks" not in d:
            continue
        m1 = d["checks"].get("M1", {})
        nve[os.path.basename(p)[:-5]] = dict(drift=m1.get("value"), ci=m1.get("ci"), tol=m1.get("tol"),
                                             status=m1.get("status"), ms_step=d.get("ms_step_med"),
                                             gpu=(d.get("gpu") or {}).get("gpu"))
    z2, gate = {}, {}
    for p in sorted(glob.glob(str(LF / "sched" / "cons" / "*_md.ragged.ragged.txt"))):
        lab = os.path.basename(p).split("_md")[0]
        with open(p) as f:
            for line in f:
                m = re.match(r"Z2\s.*?(\S+)@100: .*?= ([+-][0-9.]+) eV", line)
                if m:
                    z2.setdefault(lab, {})[m.group(1)] = float(m.group(2))
                if line.startswith("scorecard gate"):
                    gate[lab] = "PASS" in line
    z2 = dict(values=z2, gate=gate, note="E(AB) - E(A) - E(B) - kQaQb/d at 100 A (eV), tol 0.01; LiNa2+ is Li+ with "
              "Na+ (two monovalent ions); MD-form EMA files, ragged path")
    # type-held-out DFT gate
    typeheld = parse_typeheld()
    typeheld["pooled"] = typeheld_aggregates()
    # MD stability
    mdstab = build_mdstab()
    # permittivity
    permit = []
    for src in (DATA / "permit" / "analysis_pet2" / "permit_eps.json", DATA / "permit" / "analysis_final" / "permit_eps.json"):
        for x in jload(src) or []:
            if x["model"] == "PET" and src.parent.name == "analysis_final":
                continue      # the 2-seed W64 PET value is superseded by the 4-seed one in analysis_pet2
            permit.append(dict(model={"PET": "PET-OMol-S"}.get(x["model"], x["model"]), box=x["box"],
                               n_seeds=x.get("n_seeds"), eps=x["eps"], chi2=x.get("eps_chi2_95"),
                               boot=x.get("eps_blockboot_95"), seed_t=x.get("eps_acrossseed_t95"),
                               per_seed=x.get("eps_per_seed"), tau=x.get("tau_M_int_ps"), source=rel(src)))
    # paired statistics, as written by the pre-registered scripts
    stats = parse_stats()
    speed = dict(
        source=rel(LF / "A30_REPEATS_RESULTS.md") + " (section 4; transcribed: no machine-readable file)",
        inference=[dict(atoms=600, N16=23.27, N32=34.58), dict(atoms=3000, N16=226.07, N32=292.97)],
        inference_note="MD-form route-B graph, RTX 4070 Ti SUPER, min of 3 x 25 evaluations, ms per "
                       "evaluation; measured on same-graph reference files ref_N16 / ref_N32, not on the trained models",
        training=dict(A0=[30.4, 31.0], A1=[20.1, 20.4], unit="structures/s, GH200, train.log means"),
    )
    for k, m in metrics.items():
        m["maxq"] = 2 if (k.endswith(" Mg") or m.get("excluded")) else 1
    other = other_arms()
    return dict(other=other, rows=rows, pending=pending, metrics=metrics, families=FAMILIES, cells=cell_detail, adv=adv_detail,
                ladder_err=ladder_err, ladder_dft=_CL["dft_amp"], ladder_big=big_out, nve=nve, z2=z2,
                typeheld=typeheld, mdstab=mdstab, permit=permit, stats=stats, speed=speed)


def other_arms():
    """Arms and probes with no scorecard row here (unreleased kernels, charge-head probes, taper and warm-up screens)
    are not published with this bundle; the list is kept empty on purpose."""
    return []


def parse_typeheld():
    p = LF / "typeheld" / "scores_all.txt"
    out, cur = {}, None
    if not p.exists():
        return {}
    num = r"([0-9.]+) \[([0-9.]+),([0-9.]+)\] meV/A \(zero-force ([0-9.]+)\), \|dE_int\| median ([0-9.]+) eV, n=(\d+)"
    for line in p.read_text().splitlines():
        if line.startswith("== "):
            cur = line.split()[1]
            out[cur] = {}
            continue
        m = re.match(r"\s+(\S+(?: types)?): (.*)", line)
        if not (m and cur):
            continue
        t = m.group(1)
        for band, key in (("r>=8 A (gate band)", "gate"), ("6-8 A", "6-8"), ("all r>=6 A", "all")):
            mm = re.search(re.escape(band) + r": F MAE " + num, m.group(2))
            if mm:
                out[cur].setdefault(t, {})[key] = dict(F=float(mm.group(1)), lo=float(mm.group(2)),
                                                       hi=float(mm.group(3)), zero=float(mm.group(4)),
                                                       dE=float(mm.group(5)), n=int(mm.group(6)))
    return dict(scores=out, maxq={"K+_Gdm+": 1, "Gdm+_I-": 1, "Gdm+_Mg2+": 2}, source=rel(p), note=(
        "234 pair frames (K+Gdm+, Gdm+I-, Gdm+Mg2+; 6 orientation scans per type x 13 r_COM 6-14 A) + 4 monomers, "
        "wB97M-V/def2-TZVPD forces (gpu4pyscf). Gdm+ is the common partner of all three types. 4 seeds per arm "
        "(GH200 13r, 14, 15, 16) plus an H100 seed-13 file, a same-seed replicate of 13r that is not counted. PET-OMol-S = base model, no "
        "long-range term. 95 % intervals: cluster bootstrap over the 6 scans of a type (wide)."))


def typeheld_aggregates():
    """Gate-band (r_COM >= 8 A) force MAE pooled over chosen pair types, as score_typeheld.py pools 'ALL types'
    (all atoms and components; cluster bootstrap over (type, orientation) scans, 500 draws, seed 0). the program decision
    of 2026-11-17 excludes Mg2+ from scoring, so the headline pool is K+Gdm+ and Gdm+I-; Gdm+Mg2+ is kept apart."""
    TL = jload(LF / "typeheld" / "typeheld_labels.json")
    TM = jload(LF / "typeheld" / "typeheld_meta.json")
    ctb = {(a, o) for a, o, _ in TL.get("ct_flagged_scans", [])}
    out = {}
    for p in sorted(glob.glob(str(LF / "typeheld" / "out" / "*.json"))):
        lab = os.path.basename(p)[:-5]
        P = jload(p)["frames"]
        mono = {m["frag"]: (q["energy"], l["E"]) for m, l, q in zip(TM, TL["frames"], P) if m["mono"]}
        scans = collections.defaultdict(list)
        for m, l, q in zip(TM, TL["frames"], P):
            if m["mono"] or m["r_com"] < 8 or (m["pair"], m["orient"]) in ctb:
                continue
            a, b = m["pair"].split("_")
            de = (q["energy"] - mono[a][0] - mono[b][0]) - (l["E"] - mono[a][1] - mono[b][1])
            scans[(m["pair"], m["orient"])].append((np.abs(np.array(q["forces"]).reshape(-1, 3) - np.array(l["F"])).ravel(),
                                                    de, np.abs(np.array(l["F"])).ravel()))
        res = {}
        for tag, types in (("K+Gdm+ and Gdm+I- (Mg2+ excluded)", ("K+_Gdm+", "Gdm+_I-")),
                           ("all three (incl. Gdm+Mg2+)", ("K+_Gdm+", "Gdm+_I-", "Gdm+_Mg2+"))):
            ks = [k for k in scans if k[0] in types]

            def val(keys):
                return 1e3 * np.concatenate([x[0] for k in keys for x in scans[k]]).mean()
            rng = np.random.default_rng(0)
            b = [val([ks[i] for i in rng.integers(0, len(ks), len(ks))]) for _ in range(500)]
            res[tag] = dict(F=float(val(ks)), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
                            zero=float(1e3 * np.concatenate([x[2] for k in ks for x in scans[k]]).mean()),
                            dE=float(np.median(np.abs([x[1] for k in ks for x in scans[k]]))),
                            n=sum(len(scans[k]) for k in ks), scans=len(ks))
        out[lab] = res
    return out


def parse_stats():
    out = {}
    for name, p in (("A34/A35", SF / "out" / "stats_sched_final.md"), ("A30", SF / "out_a30rep" / "stats_a30.md")):
        if not p.exists():
            continue
        sec = None
        for line in p.read_text().splitlines():
            if line.startswith("### "):
                sec = line[4:].strip()
                out.setdefault(name, {})[sec] = []
                continue
            m = re.match(r"- (.+?): mean ([+-]?[0-9.e+-]+), sd [0-9.e+-]+, 95% t \[([+-]?[0-9.e+-]+), ([+-]?[0-9.e+-]+)\]"
                         r", range \[[^\]]*\], per seed ((?:[+-]?[0-9.e+-]+ ?)+)", line)
            if m and sec:
                out[name][sec].append(dict(metric=m.group(1), mean=float(m.group(2)), lo=float(m.group(3)),
                                           hi=float(m.group(4)), per_seed=[float(x) for x in m.group(5).split()]))
        out.setdefault("_source", {})[name] = rel(p)
    return out


def build_mdstab():
    st = jload(DATA / "mdstab" / "results" / "stability.json") or {}
    runs = [dict(model=r["model"], sys=r["sys"], maxq=maxq_of(r["sys"]), nw=r["nw"], T=r["T0"], seed=r["seed"], verdict=r["verdict"],
                 drift=r.get("drift_meV_atom_ps"), ms=r.get("ms_step"), natoms=r.get("natoms"),
                 flags=r.get("flags") or [], gpu=(r.get("gpu") or {}).get("gpu"))
            for r in st.get("runs", [])]
    cnt = collections.Counter(r["verdict"] for r in runs)
    if len(runs) != 309 or cnt != {"stable": 283, "flagged": 18, "unstable": 8}:   # the 17 Nov reading; more runs land
        print(f"[lrp] note: MD stability now {len(runs)} runs {dict(cnt)} (17 Nov reading: 309 runs, 283/18/8)")
    ch = jload(DATA / "mdstab" / "results" / "charges_md.json") or {}
    chs = jload(DATA / "mdstab" / "results" / "charges_static.json") or {}
    rdf = {}
    for p in sorted(glob.glob(str(DATA / "mdstab" / "figs" / "rdf_*.json"))):
        d = jload(p) or {}
        rdf[os.path.basename(p)[4:-5]] = d.get("summary", {})
    return dict(runs=runs, counts=dict(cnt), charges_md=ch.get("summary", {}), charges_static=chs.get("table", {}),
                charges_static_note=chs.get("note"), rdf=rdf, source=rel(DATA / "mdstab" / "results"))


# ------------------------------------------------------------------------------------------------- chemiscope
def encode(payload) -> str:
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def _clean(v):
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def arrows(vectors, scale=1.0, colour=None):
    """chemiscope arrow shape from per-atom vectors (list over structures of (n_atoms, 3))."""
    atom = []
    for V in vectors:
        for v in np.asarray(V, float).reshape(-1, 3):
            atom.append({"vector": [float(x) * scale for x in v]})
    g = {"baseRadius": 0.07, "headRadius": 0.15, "headLength": 0.25}
    if colour:
        g["color"] = colour
    return {"kind": "arrow", "parameters": {"global": g, "atom": atom}}


def cs_input(frames, props, name, desc, shapes=None, atom_props=None, settings=None):
    import chemiscope

    P = {}
    for k, (vals, unit, d) in props.items():
        vals = [_clean(v) for v in vals]
        if all(isinstance(v, str) or v is None for v in vals):
            vals = ["–" if v is None else v for v in vals]
            P[k] = {"target": "structure", "values": vals, "description": d}
        else:
            P[k] = {"target": "structure", "values": [None if v is None else float(v) for v in vals],
                    "units": unit, "description": d}
    env = None
    if atom_props or shapes:
        for k, (vals, unit, d) in (atom_props or {}).items():
            P[k] = {"target": "atom", "values": [float(v) if v is not None and np.isfinite(v) else float("nan") for v in vals],
                    "units": unit, "description": d}
        env = chemiscope.all_atomic_environments(frames, cutoff=3.5)
    # chemiscope.create_input refuses None in numeric arrays: NaN is turned into null by the notebook loader
    for k, p in P.items():
        if p["target"] == "structure" and not all(isinstance(v, str) for v in p["values"]):
            p["values"] = [float("nan") if v is None else v for v in p["values"]]
    ds = chemiscope.create_input(structures=frames, properties=P, environments=env, shapes=shapes or None,
                                 metadata={"name": name, "description": desc}, settings=settings or None)
    return ds


def _nan_to_null(o):
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _nan_to_null(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_null(v) for v in o]
    return o


def model_frames(label, setname):
    d = jload(SF / "sc" / label / f"{setname}.json")
    return None if d is None else d["frames"]


REP = {"P": "Pcl_s{}", "C0": "C0cl_s{}", "C0H": "C0Hcl_s{}", "A0": "A0cl_s{}", "A1": "A1cl_s{}"}


def per_frame_mean(setname, fn, arms=("P", "C0", "C0H", "A0", "A1")):
    """per-frame metric fn(model_frame, i) averaged over the 4 GH200 EMA seeds of each arm."""
    out = {}
    for a in arms:
        acc = []
        for s in SEEDS:
            P = model_frames(REP[a].format(s), setname)
            if P is None:
                continue
            acc.append([fn(p, i) for i, p in enumerate(P)])
        if acc:
            n = min(len(x) for x in acc)
            out[a] = np.nanmean(np.array([x[:n] for x in acc], float), axis=0)
    return out


def build_chemiscope(verbose=True):
    from ase.io import read

    built, views = {}, []

    def add(key, ds, title, prose, settings=None):
        settings = dict(settings or {})
        if ds.get("shapes") and "structure" not in settings:
            settings["structure"] = [{"shape": next(iter(ds["shapes"]))}]   # show the first arrow set on load
        blob = encode(_nan_to_null(ds))
        built[key] = blob
        views.append(dict(key=key, title=title, prose=prose, settings=settings or {}))
        if verbose:
            print(f"[lrp-chemiscope] {key}: {len(ds['structures'])} structures, {len(blob) / 1e6:.2f} MB")

    st = LF / "ceil" / "stage"

    # 1. seam cells: search-set orientations of the 5 rule-4 cells, Coulomb-oracle reference
    meta = jload(SF / "search_meta.json")
    FR = read(str(DATA / "esp-q" / "gdm" / "hard" / "search.xyz"), ":")
    assert len(FR) == len(meta)
    want = [("Gdm+_Gdm+", 8), ("Gdm+_Gdm+", 9), ("NH4+_NH4+", 8), ("NH4+_Gdm+", 7.5), ("K+_Gdm+", 8)]
    axis = []
    for m, a in zip(meta, FR):
        R = a.positions
        u = R[m["nA"]:].mean(0) - R[:m["nA"]].mean(0)
        axis.append(u / np.linalg.norm(u))
    proj = {}
    forces14 = {}
    for arm in ("P", "C0", "C0H", "A0", "A1"):
        acc = []
        for s in SEEDS:
            p = _res("srch_", REP[arm].format(s))
            if p is None:
                continue
            fr = jload(p)["frames"]
            assert len(fr) == len(meta)
            acc.append([float(np.array(f["forces"]).reshape(-1, 3)[m["nA"]:].sum(0) @ u)
                        for f, m, u in zip(fr, meta, axis)])
            if s == "14" and arm in ("P", "C0"):
                forces14[arm] = fr
        proj[arm] = np.array(acc)
    rng = np.random.default_rng(0)
    sel = []
    for pair, dd in want:
        idx = [i for i, m in enumerate(meta) if m["pair"] == pair and m["d"] == dd]
        worst = set(np.argsort(proj["P"].mean(0)[idx])[:30]) | set(np.argsort(proj["C0"].mean(0)[idx])[:30])
        keep = [idx[j] for j in sorted(worst)]
        rest = [i for i in idx if i not in keep]
        keep += list(rng.choice(rest, size=min(len(rest), 140 - len(keep)), replace=False))
        sel += sorted(keep)
    frames = [FR[i] for i in sel]
    for f, i in zip(frames, sel):
        f.info.clear()
    props = {"cell": ([f"{meta[i]['pair'].replace('_', '·')} {meta[i]['d']:g} Å" for i in sel], "", "rule-4 cell"),
             "d COM (Å)": ([meta[i]["d"] for i in sel], "Å", "centre-of-mass separation"),
             "dmin (Å)": ([float(np.min(np.linalg.norm(FR[i].positions[:meta[i]["nA"], None] - FR[i].positions[None, meta[i]["nA"]:], axis=-1))) for i in sel], "Å", "closest cross-fragment distance"),
             "Coulomb k/d² (eV/Å)": ([KE / meta[i]["d"] ** 2 for i in sel], "eV/Å", "Coulomb-oracle force on B along A→B (q = +1, +1)")}
    for arm, A in proj.items():
        c = np.array([KE / meta[i]["d"] ** 2 for i in sel])
        props[f"F_B/Coulomb {arm} EMA"] = (list(A[:, sel].mean(0) / c), "", f"projected force on B / Coulomb, mean of "
                                           f"{len(A)} GH200 EMA seeds ({arm}); < 0 = attractive (wrong sign)")
        props[f"attractive seeds {arm}"] = (list((A[:, sel] < 0).sum(0).astype(float)), "", f"seeds (of {len(A)}) "
                                            f"with the wrong (attractive) sign, {arm} EMA")
    shapes = {f"F {arm} s14 EMA": arrows([np.array(forces14[arm][i]["forces"]).reshape(-1, 3) for i in sel], 2.0,
                                         "#4E79A7" if arm == "P" else "#E15759") for arm in forces14}
    ds = cs_input(frames, props, "Seam sign-set cells", "search-set orientations of the 5 rule-4 cells; reference "
                  "is the Coulomb oracle, not DFT", shapes=shapes)
    add("lrp-seam-cells", ds, "Seam sign-set cells (rule 4)",
        "Orientations of the four scorable cells and K⁺Gdm⁺ 8 Å (held out, scored apart), 140 per cell: the 30 most "
        "attractive for P and for C0 plus a random draw. The reference is the **Coulomb oracle** k/d², not DFT. "
        "Colour by the arm's F_B/Coulomb; arrows are the seed-14 EMA forces of P (blue) and C0 (red), ×2 Å per eV/Å.",
        {"map": {"x": {"property": "dmin (Å)"}, "y": {"property": "F_B/Coulomb P EMA"},
                 "color": {"property": "F_B/Coulomb C0 EMA"}}})

    # 2. 8-frame adversarial set
    am = jload(SF / "adv.meta.json")
    AX = read(str(SF / "adv.xyz"), ":")
    for a in AX:
        a.info.clear()
    props = {"pair": ([m["pair"] for m in am], "", "pair"), "r (Å)": ([m["r"] for m in am], "Å", "COM separation"),
             "oracle F_B (eV/Å)": ([m.get("orc") for m in am], "eV/Å", "Coulomb-oracle projected force (signed)")}
    for arm in ("P", "C0", "C0H", "A0", "A1"):
        vals = []
        for s in SEEDS:
            p = _res("adv_", REP[arm].format(s))
            if p is None:
                continue
            fr = jload(p)["frames"]
            v = []
            for m, a, f in zip(am, AX, fr):
                R = a.positions
                u = R[m["nA"]:].mean(0) - R[:m["nA"]].mean(0)
                v.append(float(np.array(f["forces"]).reshape(-1, 3)[m["nA"]:].sum(0) @ (u / np.linalg.norm(u))))
            vals.append(v)
        props[f"F_B {arm} EMA (eV/Å)"] = (list(np.mean(vals, 0)), "eV/Å", f"projected force, mean of {len(vals)} seeds")
    ds = cs_input(AX, props, "8-frame sign set", "adversarial frames; reference = Coulomb oracle")
    add("lrp-adv", ds, "Eight-frame adversarial sign set",
        "The frames the overshoot rule (≤ 1.5× oracle at 7.5–9 Å) is read on. Reference is the Coulomb oracle.",
        {"map": {"x": {"property": "oracle F_B (eV/Å)"}, "y": {"property": "F_B P EMA (eV/Å)"},
                 "color": {"property": "r (Å)"}}})

    # 3. pairs gate (set B held-out, DFT forces)
    SB = jload(LF / "score" / "setB_eval.meta.json")
    SX = read(str(st / "setB_eval.xyz"), ":")
    assert len(SB) == len(SX)
    for a in SX:
        a.info.clear()

    def fmae(i, p):
        m = SB[i]
        return 1e3 * float(np.abs(np.array(p["forces"]).reshape(-1, 3) - np.array(m["F"])).mean())

    pf = per_frame_mean("setB_eval", lambda p, i: fmae(i, p))
    props = {"pair": ([m["pair"] for m in SB], "", "pair type"),
             "r COM (Å)": ([m["r"] if not m["mono"] else None for m in SB], "Å", "COM separation"),
             "|F_DFT| mean (meV/Å)": ([1e3 * float(np.abs(np.array(m["F"])).mean()) for m in SB], "meV/Å",
                                     "zero-force floor of this frame")}
    for a, v in pf.items():
        props[f"F MAE {a} EMA (meV/Å)"] = (list(v), "meV/Å", f"frame force MAE vs DFT, mean of 4 seeds ({a})")
    P14 = model_frames("Pcl_s14", "setB_eval")
    shapes = {"F DFT": arrows([np.array(m["F"]) for m in SB], 2.0, "#15171a"),
              "F P s14 − DFT": arrows([np.array(p["forces"]).reshape(-1, 3) - np.array(m["F"]) for p, m in zip(P14, SB)],
                                      10.0, "#E15759")}
    ds = cs_input(SX, props, "Pairs gate (set B)", "held-out set-B pairs, wB97M-V/def2-TZVPD forces", shapes=shapes)
    add("lrp-pairs-gate", ds, "Pairs gate: set-B held-out charged pairs (DFT)",
        "DFT-labelled pair scans 4–14 Å (the gate scores r ≥ 8 Å). Arrows: the DFT force (×2) and P seed 14's error "
        "(×10). K⁺MeCN, NH₄⁺Cl⁻ and NH₄⁺NH₄⁺ are pair types present in P's training rows (different geometries).",
        {"map": {"x": {"property": "r COM (Å)"}, "y": {"property": "F MAE P EMA (meV/Å)"},
                 "color": {"property": "F MAE C0 EMA (meV/Å)"}}})

    # 4. type-held-out gate (DFT)
    TL = jload(LF / "typeheld" / "typeheld_labels.json")
    TM = jload(LF / "typeheld" / "typeheld_meta.json")
    TX = read(str(LF / "typeheld" / "typeheld_eval.xyz"), ":")
    assert len(TX) == len(TM) == len(TL["frames"])
    for a in TX:
        a.info.clear()
    tf = {}
    for lab in ["C0fema"] + [f"C0cl_s{s}" for s in SEEDS] + ["Pfema"] + [f"Pcl_s{s}" for s in SEEDS] + ["PETOMOLS"]:
        d = jload(LF / "typeheld" / "out" / f"{lab}.json")
        if d:
            assert len(d["frames"]) == len(TM)
            tf[lab] = d["frames"]

    def th_eint(P):
        mono = {m["frag"]: (p["energy"], l["E"]) for m, l, p in zip(TM, TL["frames"], P) if m["mono"]}
        out = []
        for m, l, p in zip(TM, TL["frames"], P):
            if m["mono"]:
                out.append(None)
                continue
            a, b = m["pair"].split("_")
            out.append((p["energy"] - mono[a][0] - mono[b][0]) - (l["E"] - mono[a][1] - mono[b][1]))
        return out

    def th_f(P):
        return [1e3 * float(np.abs(np.array(p["forces"]).reshape(-1, 3) - np.array(l["F"])).mean())
                for p, l in zip(P, TL["frames"])]

    props = {"pair": ([m.get("pair", m.get("frag", "monomer")) if not m["mono"] else f"monomer {m['frag']}" for m in TM], "", "pair type"),
             "r COM (Å)": ([m.get("r_com") if not m["mono"] else None for m in TM], "Å", "COM separation"),
             "orientation": ([float(m.get("orient", -1)) if not m["mono"] else None for m in TM], "", "scan index (6 per type)"),
             "max |q|": ([float(maxq_of(m.get("pair", m.get("frag")))) for m in TM], "e", "largest |formal charge| (≥ 2 is excluded from scoring by default)"),
             "|F_DFT| mean (meV/Å)": ([1e3 * float(np.abs(np.array(l["F"])).mean()) for l in TL["frames"]], "meV/Å", "zero-force floor")}
    for arm, labs in (("C0", ["C0fema"] + [f"C0cl_s{s}" for s in SEEDS]), ("P", ["Pfema"] + [f"Pcl_s{s}" for s in SEEDS]),
                      ("PET-OMol-S", ["PETOMOLS"])):
        labs = [x for x in labs if x in tf]
        F = np.mean([th_f(tf[x]) for x in labs], 0)
        E = np.array([[np.nan if v is None else v for v in th_eint(tf[x])] for x in labs], float).mean(0)
        nn = f"mean of {len(labs)} EMA files" if len(labs) > 1 else "one file"
        props[f"F MAE {arm} (meV/Å)"] = (list(F), "meV/Å", f"frame force MAE vs DFT ({arm}, {nn})")
        props[f"ΔE_int error {arm} (eV)"] = (list(E), "eV", f"E_int(model) − E_int(DFT), own monomers ({arm}, {nn})")
    shapes = {"F DFT": arrows([np.array(l["F"]) for l in TL["frames"]], 2.0, "#15171a")}
    if "Pcl_s14" in tf:
        shapes["F P s14 − DFT"] = arrows([np.array(p["forces"]).reshape(-1, 3) - np.array(l["F"])
                                         for p, l in zip(tf["Pcl_s14"], TL["frames"])], 10.0, "#4E79A7")
    if "PETOMOLS" in tf:
        shapes["F PET-OMol-S − DFT"] = arrows([np.array(p["forces"]).reshape(-1, 3) - np.array(l["F"])
                                              for p, l in zip(tf["PETOMOLS"], TL["frames"])], 10.0, "#E15759")
    ds = cs_input(TX, props, "Type-held-out charged pairs", "234 pair frames + 4 monomers, wB97M-V/def2-TZVPD forces",
                  shapes=shapes)
    add("lrp-typeheld", ds, "Type-held-out charged-pair gate (DFT)",
        "K⁺Gdm⁺, Gdm⁺I⁻ and Gdm⁺Mg²⁺: pair types absent from every training row, 6 orientation scans per type × 13 "
        "distances (6–14 Å), plus the 4 monomers. Gdm⁺ is the common partner, and the other partner is an atom, so "
        "orientation variety is the Gdm⁺ rotation only. C0 and P are means of 5 EMA files (H100 s13 + GH200 "
        "13r–16; 13 and 13r are same-seed replicates). Arrows: DFT force (×2), P seed-14 and PET-OMol-S errors (×10).",
        {"map": {"x": {"property": "r COM (Å)"}, "y": {"property": "F MAE P (meV/Å)"},
                 "color": {"property": "F MAE PET-OMol-S (meV/Å)"}}})

    # 5. cluster ladders (DFT energies; no forces in the set)
    M = _CL["meta"]
    CX = _CL["frames"]
    idx = [i for i, m in enumerate(M) if m["role"] == "XW"]
    frames = [CX[i] for i in idx]
    for a in frames:
        a.info.clear()
    props = {"max |q|": ([float(maxq_of(M[i]["sys"])) for i in idx], "e", "largest |formal charge|"),
             "ion": ([M[i]["sys"] for i in idx], "", "ion"), "n waters": ([M[i]["nw"] for i in idx], "", "waters"),
             "centre": ([float(M[i]["centre"]) for i in idx], "", "held-out centre")}
    for arm in ("P", "C0", "C0H", "A0", "A1"):
        acc = []
        for s in SEEDS:
            lc = ladder_curves(REP[arm].format(s))
            if lc:
                acc.append([lc[M[i]["sys"]]["per_frame"].get(f"{M[i]['centre']}:{M[i]['nw']}", np.nan) for i in idx])
        props[f"dE error {arm} EMA (eV)"] = (list(np.nanmean(acc, 0)), "eV", f"vertical binding energy error, mean "
                                              f"of {len(acc)} seeds ({arm})")
    ds = cs_input(frames, props, "Cluster ladders", "ion + n waters, held-out centres; DFT energies")
    add("lrp-ladders", ds, "Ion–water cluster ladders",
        "Vertical binding energy dE(n) = E[X·W_n] − E[W_n] − E[X], error against DFT, per held-out centre. The gate is "
        "the slope of this error for n ≥ 24 (±0.02 eV per water).",
        {"map": {"x": {"property": "n waters"}, "y": {"property": "dE error P EMA (eV)"},
                 "color": {"property": "ion"}}})

    # 6. NaCl(aq) cut-outs: per-atom charges and force errors
    NM = jload(LF / "score" / "nacl1m.meta.json")
    NX = read(str(st / "nacl1m.xyz"), ":")
    DD = np.load(DATA / "cluster-laws" / "extra" / "nacl1m" / "charges.npz", allow_pickle=True)
    Pn = model_frames("Pcl_s14", "nacl1m")
    Cn = model_frames("C0cl_s14", "nacl1m")
    for a in NX:
        a.info.clear()
        a.calc = None
    for k, a in enumerate(NX):
        assert (DD[f"Z_{k}"] == a.numbers).all() and np.allclose(DD[f"R_{k}"], a.positions, atol=1e-4), k
    FD = [DD[f"F_{k}"] for k in range(len(NX))]
    atomP = {"q Löwdin (DFT)": (np.concatenate([DD[f"q_lowdin_{k}"] for k in range(len(NX))]).tolist(), "e",
                                "DFT Löwdin charge"),
             "q Hirshfeld (DFT)": (np.concatenate([DD[f"q_hirshfeld_{k}"] for k in range(len(NX))]).tolist(), "e",
                                   "DFT Hirshfeld charge")}
    if Pn and len(Pn) == len(NX):
        atomP["|F error| P s14 (meV/Å)"] = (np.concatenate([1e3 * np.linalg.norm(np.array(p["forces"]).reshape(-1, 3)
                                                                               - f, axis=1) for p, f in zip(Pn, FD)]).tolist(),
                                           "meV/Å", "per-atom force error norm, P seed 14 EMA")
    for lab, P in (("P s14", Pn), ("C0 s14", Cn)):
        if P and all(len(p.get("latent_charges", [])) == len(a) for p, a in zip(P, NX)):
            atomP[f"q {lab}"] = (sum([list(p["latent_charges"]) for p in P], []), "e", f"latent charge ({lab} EMA)")
    props = {"cut-out": ([float(m["cut"]) for m in NM], "", "cut-out index"),
             "Q": ([float(m["Q"]) for m in NM], "e", "total charge"), "atoms": ([float(m["n"]) for m in NM], "", "")}
    ds = cs_input(NX, props, "NaCl(aq) cut-outs", "8 DFT-labelled cut-outs of the 1 m NaCl(aq) box",
                  atom_props=atomP, shapes={"F DFT": arrows(FD, 1.0, "#15171a")})
    add("lrp-nacl1m", ds, "NaCl(aq) cut-outs: charges per atom",
        "Eight spherical cut-outs of a 1 m NaCl(aq) box. Per-atom DFT Löwdin and model latent charges (P and C0, seed "
        "14 EMA), with the DFT forces as arrows (×1). The ion-shell gate (≤ 0.2 e) reads the whole-water shells of "
        "the interior ions.",
        {"target": "atom", "map": {"x": {"property": "q Löwdin (DFT)"}, "y": {"property": "q P s14"},
                                   "color": {"property": "|F error| P s14 (meV/Å)"}}})

    # 6b. nacl1 droplet: one 302-atom DFT frame, per-atom charges and force errors
    D1 = np.load(DATA / "cluster-laws" / "extra" / "nacl1" / "charges.npz", allow_pickle=True)
    ND = read(str(st / "nacl1.xyz"))
    ND.info.clear()
    ND.calc = None
    k1 = sorted(D1.keys())
    F1 = D1["F"] if "F" in k1 else D1[[k for k in k1 if k.startswith("F")][0]]
    q1 = D1["q_lowdin"] if "q_lowdin" in k1 else D1[[k for k in k1 if k.startswith("q_lowdin")][0]]
    assert len(F1) == len(ND) == len(q1) and (D1["Z"] == ND.numbers).all() and np.allclose(D1["R"], ND.positions, atol=1e-4)
    atomP = {"q Löwdin (DFT)": (list(map(float, q1)), "e", "DFT Löwdin charge")}
    for lab, L in (("P s14", "Pcl_s14"), ("C0 s14", "C0cl_s14")):
        P1 = model_frames(L, "nacl1")
        if P1 and len(P1[0].get("latent_charges", [])) == len(ND):
            atomP[f"q {lab}"] = (list(map(float, P1[0]["latent_charges"])), "e", f"latent charge ({lab} EMA)")
            atomP[f"|F error| {lab} (meV/Å)"] = (list(1e3 * np.linalg.norm(np.array(P1[0]["forces"]).reshape(-1, 3)
                                                                            - F1, axis=1)), "meV/Å", "")
    ds = cs_input([ND], {"atoms": ([float(len(ND))], "", "")}, "NaCl droplet", "one DFT-labelled NaCl(aq) droplet",
                  atom_props=atomP, shapes={"F DFT": arrows([F1], 1.0, "#15171a")})
    add("lrp-nacl1", ds, "NaCl droplet (nacl1): charges and forces per atom",
        "One 302-atom droplet with DFT forces (arrows ×1) and Löwdin charges, against the P and C0 seed-14 EMA charges "
        "and per-atom force errors. It feeds the droplet F MAE of the condensed-phase family.",
        {"target": "atom", "map": {"x": {"property": "q Löwdin (DFT)"}, "y": {"property": "q P s14"},
                                   "color": {"property": "|F error| P s14 (meV/Å)"}}})

    # 6c. orientation rows (orient4): four orientations x four distances per pair, model forces only
    O4 = read(str(st / "orient4.xyz"), ":")
    M4 = jload(LF / "score" / "orient4.meta.json")
    assert len(O4) == len(M4)
    for a in O4:
        a.info.clear()
        a.calc = None
    props = {"pair": ([str(m[0]).split(":")[-1] for m in M4], "", "pair"),
             "orientation": ([float(m[1]) for m in M4], "", "orientation index"),
             "r (Å)": ([float(m[2]) for m in M4], "Å", "separation")}
    for arm in ("P", "C0", "C0H", "A1"):
        acc = []
        for sd in SEEDS:
            Po = model_frames(REP[arm].format(sd), "orient4")
            if Po and len(Po) == len(O4):
                v = []
                for m, a, f in zip(M4, O4, Po):
                    nA = int(m[3]); R = a.positions
                    u = R[nA:].mean(0) - R[:nA].mean(0)
                    v.append(float(np.array(f["forces"]).reshape(-1, 3)[nA:].sum(0) @ (u / np.linalg.norm(u))))
                acc.append(v)
        if acc:
            props[f"F_B {arm} EMA (eV/Å)"] = (list(np.mean(acc, 0)), "eV/Å", f"projected force on B, mean of {len(acc)} seeds")
    Po = model_frames("Pcl_s14", "orient4")
    ds = cs_input(O4, props, "Orientation rows", "pairs at 8-12 Å in several orientations; model forces only",
                  shapes={"F P s14 EMA": arrows([np.array(f["forces"]).reshape(-1, 3) for f in Po], 2.0, "#4E79A7")}
                  if Po and len(Po) == len(O4) else None)
    add("lrp-orient4", ds, "Orientation rows (orient4)",
        "Charged pairs at 8–12 Å in several orientations, scored for the far-field force spread (the q95 |ΔF| "
        "metrics of the seam family). No DFT reference; arrows are P seed-14 EMA forces (×2).",
        {"map": {"x": {"property": "r (Å)"}, "y": {"property": "F_B P EMA (eV/Å)"},
                 "color": {"property": "F_B C0 EMA (eV/Å)"}}})

    # 7. val500 subsample (DFT forces in the xyz)
    VX = read(str(st / "val500.xyz"), ":")
    pick = list(range(0, len(VX), 5))
    Fd = [np.asarray(VX[i].get_forces() if VX[i].calc is not None else VX[i].arrays["forces"]) for i in pick]
    props = {"atoms": ([float(len(VX[i])) for i in pick], "", "atoms"),
             "charge": ([float(VX[i].info.get("charge", 0)) for i in pick], "e", "total charge")}
    for i in pick:
        VX[i].calc = None
    for arm in ("P", "C0", "C0H", "A1"):
        acc = []
        for s in SEEDS:
            P = model_frames(REP[arm].format(s), "val500")
            if P and len(P) == len(VX):
                acc.append([1e3 * float(np.abs(np.array(P[i]["forces"]).reshape(-1, 3) - Fd[k]).mean())
                            for k, i in enumerate(pick)])
        if acc:
            props[f"F MAE {arm} EMA (meV/Å)"] = (list(np.mean(acc, 0)), "meV/Å", f"mean of {len(acc)} seeds")
    frames = []
    for i in pick:
        a = VX[i].copy()
        a.info.clear()
        for k in list(a.arrays):
            if k not in ("numbers", "positions"):
                del a.arrays[k]
        frames.append(a)
    ds = cs_input(frames, props, "val500 subsample", "every 5th of the 500 validation structures, DFT forces",
                  shapes={"F DFT": arrows(Fd, 1.0, "#15171a")})
    add("lrp-val500", ds, "val500 (every 5th structure)",
        "The near-equilibrium accuracy gate (V1: val500 F MAE ≤ 42 meV/Å). Arrows: DFT forces (×1).",
        {"map": {"x": {"property": "F MAE C0 EMA (meV/Å)"}, "y": {"property": "F MAE A1 EMA (meV/Å)"},
                 "color": {"property": "charge"}}})

    # 8. pair scans: NH4+-Gdm+ (subsampled) and Li+-Na+ (model only, Coulomb reference)
    for nm, step in (("NH4+-Gdm+", 11), ("Li+Na+", 9)):
        X = read(str(st / "pairs" / f"{nm}.xyz"), ":")
        pick = list(range(0, len(X), step))
        frames = []
        for i in pick:
            a = X[i].copy()
            a.info.clear()
            frames.append(a)
        Pm = {arm: model_frames(REP[arm].format("14"), nm) for arm in ("P", "C0")}
        Pm["ORIG"] = model_frames("ORIG", nm)
        props = {}
        # COM distance from the largest gap in a 2-fragment split: fragments are the first atom group (by charge)
        d = []
        for a in frames:
            if nm.startswith("Li"):
                d.append(float(a.get_distance(0, 1)))
            else:
                nA = 5  # NH4+ first (5 atoms), Gdm+ after
                d.append(float(np.linalg.norm(a.positions[:nA].mean(0) - a.positions[nA:].mean(0))))
        props["r (Å)"] = (d, "Å", "separation (COM)")
        props["k/r (eV)"] = ([KE / x for x in d], "eV", "bare Coulomb for +1, +1")
        for arm, P in Pm.items():
            if P and len(P) == len(X):
                E = np.array([P[i]["energy"] for i in pick])
                props[f"E − k/r {arm} (eV)"] = (list(E - KE / np.array(d) - (E[-1] - KE / d[-1])), "eV",
                                                f"model energy minus Coulomb, shifted to 0 at the last point ({arm})")
        ds = cs_input(frames, props, f"{nm} scan", "rigid pair scan through the seam band; no DFT reference")
        add(f"lrp-scan-{nm.replace('+', 'p').replace('-', '')}", ds, f"{nm} scan through the seam",
            "A rigid scan; the model energy minus the bare Coulomb term should be flat beyond the cutoff. No DFT "
            "reference: this tests the seam handover, not the energy.",
            {"map": {"x": {"property": "r (Å)"}, "y": {"property": "E − k/r P (eV)"},
                     "color": {"property": "E − k/r C0 (eV)"}}})

    # 9. MD-stability starting structures, with each model's verdicts
    runs = build_mdstab()["runs"]
    starts = sorted(glob.glob(str(DATA / "mdstab" / "starts" / "*_n48_s*.xyz"))
                    + glob.glob(str(DATA / "mdstab" / "starts" / "*_n200_s*.xyz")))
    frames, nm = [], []
    for p in starts:
        a = read(p)
        a.info.clear()
        for k in list(a.arrays):
            if k not in ("numbers", "positions"):
                del a.arrays[k]
        frames.append(a)
        nm.append(os.path.basename(p)[:-4])
    props = {"system": (nm, "", "start"), "ion": ([n.split("_")[0] for n in nm], "", "ion"),
             "max |q|": ([float(maxq_of(n.split("_")[0])) if not n.startswith("W") else 0.0 for n in nm], "e",
                         "largest |formal charge|")}
    models = sorted({r["model"] for r in runs})
    for mdl in models:
        v = []
        for n in nm:
            ion, nw, sd = n.split("_")
            rr = [r for r in runs if r["model"] == mdl and r["sys"] == ion and r["nw"] == int(nw[1:])
                  and r["seed"] == int(sd[1:])]
            v.append(None if not rr else float(sum(r["verdict"] == "stable" for r in rr)) / len(rr))
        props[f"stable share {mdl}"] = (v, "", f"fraction of runs from this start (all T) judged stable, {mdl}")
    ds = cs_input(frames, props, "MD stability starts", "open ion–water clusters; MD verdicts per model")
    add("lrp-mdstab", ds, "MD-stability starting clusters",
        "Na⁺, Cl⁻ and Mg²⁺ with 48 or 200 waters (open boundary): the share of each model's runs from this start "
        "(250–400 K) judged stable. No forces are stored for these starts.",
        {"map": {"x": {"property": "stable share PET-OMol-S"}, "y": {"property": "stable share Bq4m3_c01"},
                 "color": {"property": "ion"}}})

    # 10. benchmark sets outside the score: reference structures with DFT forces and Löwdin charges
    lr_sets = pathlib.Path(os.environ.get("LRBENCH_DATA", "")) if os.environ.get("LRBENCH_DATA") else None
    for setname, cap, why in (
            ("charge_parity", 60, "charge parity: the same structure at several total charges"),
            ("dimers_q2", 60, "dimers with a doubly charged fragment"),
            ("emulation", 120, "the emulation set (DFT pathologies a model must reproduce)"),
            ("families_full", 160, "families (full ΔE, every member)"),
            ("families_corrected", 160, "families, corrected labels"),
            ("pathology_h2o2p", 10, "the H₂O₂⁺ pathology (DFT state ambiguity; never scored)"),
            ("pbe0mbd_subset", 120, "PBE0+MBD subset (another level of theory)"),
            ("maps2d", 150, "2-D maps")):
        if lr_sets is None:
            break
        p = lr_sets / f"{setname}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            recs = [json.loads(x) for x in f if x.strip()]
        if len(recs) > cap:
            step = len(recs) / cap
            recs = [recs[int(i * step)] for i in range(cap)]
        from ase import Atoms
        frames, Fs, qs = [], [], []
        for r in recs:
            frames.append(Atoms(symbols=r["species"], positions=r["positions"]))
            Fs.append(np.array(r["forces"]) if r.get("forces") is not None else np.zeros((len(r["species"]), 3)))
            ql = (r.get("refs") or {}).get("charges_lowdin")
            qs += list(ql) if ql is not None and len(ql) == len(r["species"]) else [float("nan")] * len(r["species"])
        props = {"max |q|": ([float(max([abs(x) for x in (r.get("fragment_charges") or [r.get("charge") or 0])] or [0]))
                              for r in recs], "e", "largest |fragment formal charge|"),
                 "class": ([str(r.get("cls")) for r in recs], "", "class"),
                 "group": ([str(r.get("group")) for r in recs], "", "group"),
                 "total charge": ([float(r.get("charge") or 0) for r in recs], "e", ""),
                 "distance (Å)": ([r.get("distance") for r in recs], "Å", "separation where defined"),
                 "E_DFT (eV)": ([r.get("energy") for r in recs], "eV", f"reference energy ({recs[0].get('level')})")}
        atomP = {"|F_DFT| (eV/Å)": (np.concatenate([np.linalg.norm(F, axis=1) for F in Fs]).tolist(), "eV/Å",
                                    "DFT force norm per atom")}
        if np.isfinite(np.array(qs, float)).any():
            atomP["q Löwdin"] = (qs, "e", "DFT Löwdin charge")
        ds = cs_input(frames, props, setname, why, shapes={"F DFT": arrows(Fs, 1.0, "#15171a")}, atom_props=atomP)
        add(f"lrb-{setname}", ds, f"{setname} (reference only)",
            f"{why[0].upper() + why[1:]}. {len(recs)} records, DFT forces as arrows (×1) and Löwdin charges per atom. "
            "Not part of the long-range score; shown so every structure the benchmark carries can be inspected.",
            {"target": "structure", "map": {"x": {"property": "distance (Å)"}, "y": {"property": "E_DFT (eV)"},
                                            "color": {"property": "total charge"}}})
    return built, views


# ------------------------------------------------------------------------------------------------------- main
_CL, _ADV = {}, {}


def _load_static():
    from ase.io import read

    _CL["meta"] = jload(LF / "cl" / "cl_heldout.meta.json")
    for m in _CL["meta"]:       # the fluoride ladder is stored as sys = false (a YAML "F" upstream)
        if m["sys"] is False:
            m["sys"] = "F"
    _CL["frames"] = read(str(LF / "cl" / "cl_heldout.xyz"), ":")
    assert len(_CL["frames"]) == len(_CL["meta"])
    _CL["dft_amp"] = None
    _ADV["meta"] = jload(SF / "adv.meta.json")
    _ADV["frames"] = read(str(SF / "adv.xyz"), ":")


# --------------------------------------------------------------------------------------------- 1-D scans (AL)
SCANS = LF / "scans_al"
SCAN_MODELS = {"PETOMOLS": "PET-OMol-S", "C0": "C0", "P": "P", "AL": "AL", "DFT": "DFT"}
SCAN_NOTE = (
    "Rigid centre-of-mass scans of charged pairs: fragment B shifted in 0.05 Å steps over 6–11 Å (101 points per scan; "
    "11–14 Å from the 0.5 Å DFT grids). Sets: type-held-out pairs (K⁺Gdm⁺, Gdm⁺I⁻, Gdm⁺Mg²⁺; 6 orientations each), "
    "the set-B gate pairs (K⁺MeCN, NH₄⁺Cl⁻, NH₄⁺NH₄⁺; 6 each) and 3 orientations of four sign-set cells. "
    "AL is ONE seed (13, EMA): every AL number here is a screen. C0 and P are their seed-13 EMA files (H100), "
    "PET-OMol-S the base model (GGUF, fp32). Reference: wB97M-V/def2-TZVPD for the type-held-out and set-B scans; the "
    "sign-set cells and the two dense pair scans have no DFT, their reference is the point-charge Coulomb force k/d². "
    "Gdm⁺Mg²⁺ (multivalent) and the neutral H₂O·H₂O pair are shown apart and are in no aggregate (AGG). Intervals: 95 % "
    "percentile block bootstrap over whole scans (asymmetric); n = number of scans. Kink = largest second difference "
    "of the net force on B minus the DFT-spline second difference, in meV/Å per (0.05 Å)²; a smooth DFT curve gives "
    "about 6.")


def _png_b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode() if pathlib.Path(p).exists() else None


def build_scans():
    """scan-quality block (numbers from results_{coarse,fine,pairs}.json, not from the note) + the five figures"""
    co, fi, pa = (jload(SCANS / f"results_{k}.json") for k in ("coarse", "fine", "pairs"))
    if not co:
        return None

    def ren(key):
        return "|".join(SCAN_MODELS.get(x, x) for x in key.split("|"))

    def ci(v):
        return None if v is None else [float(x) for x in v]
    mae = {ren(k): dict(F=ci(v.get("F_mae")), netB=ci(v.get("netB_err")), dEint=ci(v.get("dEint_med")),
                        zero=v.get("zero_floor"), n_scans=v.get("n_scans"), n_frames=v.get("n_frames"))
           for k, v in co["mae"].items()}
    tail = {ren(k): dict(err=ci(v.get("err_eV_A")), r2err=ci(v.get("r2err")), rel=ci(v.get("rel_to_dft")))
            for k, v in co["tail_pooled"].items()}
    sign = {ren(k): dict(frac=ci(v.get("right_sign_frac")), n_frames=v.get("n_frames"), n_scans=v.get("n_scans"))
            for k, v in co["sign"].items()}
    search = {ren(k): dict(n=v.get("n"), sign=ci(v.get("right_sign_frac")), med=ci(v.get("med_over_coulomb")),
                           over15=v.get("over15")) for k, v in co["search"].items()}
    smooth = {}
    for k, v in (fi or {}).get("stats", {}).items():
        smooth[ren(k)] = {m: v[m] for m in ("excess_max_cut", "excess_max_floor", "cons_rms_cut", "cons_rms_out", "cons_max_cut")
                          if m in v}
    dense = {ren(k): dict(row=v["row"], r=v["r"][::2], f_over_c=v["f_over_c"][::2]) for k, v in (pa or {}).items()}
    figs = {}
    for f, cap in (("fig1_typeheld", "Type-held-out pairs: net force on B along the COM axis against DFT, per pair type "
                    "(band = orientation range); Gdm⁺Mg²⁺ shown apart."),
                   ("fig2_setB", "Set-B gate pairs, same layout; H₂O·H₂O (neutral) shown apart."),
                   ("fig3_search_cells", "Sign-set cells (500 orientations per cell): right-sign share and median force "
                    "over the Coulomb reference k/d²."),
                   ("fig4_fine_scans", "0.05 Å scans through 6–11 Å: net force, model minus DFT, and the second "
                    "difference (kinks)."),
                   ("fig5_dense_pairs_kinks", "Dense +1/+1 scans (0.002 Å, Coulomb reference) and kink size in 6–9 Å "
                    "(median, 95 % CI).")):
        b = _png_b64(SCANS / f"{f}.png")
        if b:
            figs[f] = dict(png=b, caption=cap)
    return dict(note=SCAN_NOTE, models=["PET-OMol-S", "C0", "P", "AL"], mae=mae, tail=tail, sign=sign,
                search=search, smooth=smooth, dense=dense, figures=figs, seeds={"AL": 1, "C0": 1, "P": 1},
                evidence="screen (AL one seed; C0 and P single seed-13 files; their 5-file spread on the type-held-out "
                         "gate is 47.8–52.7 (C0) and 44.3–51.6 (P) meV/Å, Mg²⁺ included)")


SCAN_VIEWS = [
    ("typeheld", "lrp-scan-typeheld", "1-D scans: type-held-out charged pairs (DFT)",
     "234 frames: 18 rigid scans (K⁺Gdm⁺, Gdm⁺I⁻, Gdm⁺Mg²⁺ × 6 orientations), wB97M-V forces. Arrow sets F_DFT, "
     "F_PETOMOLS, F_C0, F_P, F_AL (2.5 Å per eV/Å; type several names, comma separated, in the structure panel's "
     "shape field). Gdm⁺Mg²⁺ is flagged `apart_from_aggregates`. AL = one seed (screen)."),
    ("setB", "lrp-scan-setB", "1-D scans: set-B gate pairs (DFT)",
     "312 frames: K⁺MeCN, NH₄⁺Cl⁻, NH₄⁺NH₄⁺ (6 orientations each) and the neutral H₂O·H₂O pair (apart). Same arrow "
     "sets and properties as above."),
    ("search_fine", "lrp-scan-search", "1-D scans: sign-set cells, 0.05 Å (no DFT)",
     "252 frames: 3 orientations of Gdm⁺Gdm⁺, NH₄⁺NH₄⁺, NH₄⁺Gdm⁺ and K⁺Gdm⁺ scanned through 6–11 Å. No DFT: compare the "
     "net force on B with the Coulomb reference k/d². Model arrows only."),
]


def build_scan_views():
    built, views = {}, []
    for f, key, title, prose in SCAN_VIEWS:
        d = jload(SCANS / f"chemiscope_{f}.json")
        if not d:
            continue
        d["meta"] = {"name": title, "description": prose.replace("`", "")}
        built[key] = encode(_nan_to_null(d))
        views.append(dict(key=key, title=title, prose=prose, settings=d.get("settings") or {}))
        print(f"[lrp-chemiscope] {key}: {len(d['structures'])} structures, {len(built[key]) / 1e6:.2f} MB")
    return built, views


# ----------------------------------------------------------------------------------------- large-n ladders (rule 3)
LADDER_FILES = re.compile(r"^(AL|PAL|P|C0)cl_s(13r|1[3-6])\.jsonl$")
LADDER_Q = {"Na": 1, "Cl": -1}
#: DRAFT absolute-error limits of the large-n ladder (ladder v2 spec A1 / E1, 18 Nov 2026; not frozen). Limit = 2 x the
#: DFT law's fit uncertainty, in eV, at n = 1,000 and 4,000 waters. Resolution about 1 eV: the law's fit windows disagree by
#: 0.4-0.7 eV. The slope rules (<= 0.01 and <= 0.02 eV/water) are no longer pass/fail: a constant-offset model passes both
#: for every offset up to 30 eV.
A1_LIMIT_EV = {"Na": {1000: 0.86, 4000: 1.14}, "Cl": {1000: 1.09, 4000: 1.36}}


def _ladder_errors(path, ion, ref):
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if r.get("kind") == "rec" and r.get("sys") == ion and r.get("E") is not None]
    X = [r["E"] for r in R if r["role"] == "X"]
    if not X:
        return None
    xw = {(r["n"], r["t"]): r for r in R if r["role"] == "XW"}
    w = {(r["n"], r["t"]): r for r in R if r["role"] == "W"}
    d = ref["dft_tail"][ion]
    err, tail = {}, []
    for k in sorted(xw):
        if k in w:
            n = k[0]
            dE = xw[k]["E"] - w[k]["E"] - X[0]
            x = LADDER_Q[ion] ** 2 * KE / (2 * xw[k]["Rn"])
            err.setdefault(n, []).append(dE - (d["dEinf"] + d["A"] * x))
            tail.append((n, x, dE))
    return err, tail


def _ladder_slope(e, nmin=1000):
    ns = [n for n in e if n >= nmin]
    x = np.concatenate([[n] * len(e[n]) for n in ns])
    y = np.concatenate([e[n] for n in ns])
    return float(np.polyfit(x, y, 1)[0])


def _ladder_boot(e, B=10000, seed=0, nmin=1000):
    """frame bootstrap: resample the frames within each n (the program's slopes_boot recipe), 95 % percentile"""
    g = np.random.default_rng(seed)
    ns = [n for n in e if n >= nmin]
    s = []
    for _ in range(B):
        x, y = [], []
        for n in ns:
            a = np.asarray(e[n])[g.integers(0, len(e[n]), len(e[n]))]
            x += [n] * len(a)
            y += list(a)
        s.append(np.polyfit(x, y, 1)[0] if len(set(x)) > 1 else np.nan)
    lo, hi = np.nanpercentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def build_ladder_big():
    """large-n ion-water ladders (n = 200-4,000 waters, 4 frames per n), error of dE(n) against the DFT reference law;
    absolute error at n = 1,000 and 4,000 against the draft A1 limits (pass/fail); slope over n = 1,000-4,000 and the own
    tail fit A over n >= 500 (DFT range [1.0, 2.0]) are kept as diagnostics"""
    ref = jload(DATA / "cluster-laws" / "ml_results" / "box_cutouts" / "reference.json")
    if not ref:
        return None
    files = {}
    for d in (LF / "ladders_big_AL", LF / "sched" / "rule6"):
        for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
            m = LADDER_FILES.match(f.name)
            if m:
                files.setdefault((m.group(1), m.group(2)), f)
    per = {}
    for (arm, seed), f in sorted(files.items()):
        for ion in LADDER_Q:
            r = _ladder_errors(f, ion, ref)
            if not r or not any(n >= 1000 for n in r[0]):
                continue
            e, tail = r
            lo, hi = _ladder_boot(e)
            sel = [(x, dE) for n, x, dE in tail if n >= 500]
            A = float(np.polyfit([a for a, _ in sel], [b for _, b in sel], 1)[0]) if len(sel) > 2 else None
            nrec = sum(1 for x in open(f) if '"kind": "rec"' in x)
            per.setdefault(arm, {}).setdefault(ion, []).append(dict(
                seed=seed, slope=_ladder_slope(e), lo=lo, hi=hi, slope_ge200=_ladder_slope(e, 200), tail_A=A,
                records=nrec, n_used=sorted(int(n) for n in e if n >= 1000),
                frames={int(n): len(v) for n, v in e.items()}, err_mean={int(n): float(np.mean(v)) for n, v in e.items()},
                err_sd={int(n): float(np.std(v)) for n, v in e.items()}))
    arms = {}
    for arm, ions in per.items():
        arms[arm] = {}
        for ion, rows in ions.items():
            s = np.array([r["slope"] for r in rows])
            k = len(s)
            summ = dict(n_seeds=k, seeds=[r["seed"] for r in rows], mean=float(s.mean()))
            if k >= 2:
                from scipy import stats as _st
                h = float(_st.t.ppf(0.975, k - 1) * s.std(ddof=1) / np.sqrt(k))
                summ.update(sd=float(s.std(ddof=1)), lo=summ["mean"] - h, hi=summ["mean"] + h)
            summ["within_0.01"] = bool(k >= 1 and all(abs(x) <= 0.01 for x in s) and
                                       (k < 2 or max(abs(summ["lo"]), abs(summ["hi"])) <= 0.01))
            summ["A"] = [r["tail_A"] for r in rows]
            summ.update(a1_summary(rows, ion))
            arms[arm][ion] = dict(summary=summ, files=rows)
    return dict(arms=arms, text=large_n_text(arms, {i: ref["dft_tail"][i]["A"] for i in LADDER_Q}),
                dft_A={i: ref["dft_tail"][i]["A"] for i in LADDER_Q}, abs_limit_ev=A1_LIMIT_EV_JSON(), limit_status="draft",
                note=LARGE_N_NOTE)


def A1_LIMIT_EV_JSON():
    return {ion: {str(n): t for n, t in v.items()} for ion, v in A1_LIMIT_EV.items()}


LARGE_N_NOTE = (
    "Error of dE(n) = E[ion + n waters] - E[n waters] - E[ion] against the DFT reference law dE_inf + A q^2 k / 2R_n, "
    "n = 200-4,000 waters (to 12,001 atoms), 4 frames per n. Pass rule (DRAFT, ladder v2 A1; not frozen): the absolute "
    "error at n = 1,000 and at n = 4,000 is inside 2 x the law's fit uncertainty (Na⁺ 0.86 and 1.14 eV, Cl⁻ 1.09 and "
    "1.36 eV); resolution about 1 eV. Slope = OLS over n = 1,000-4,000 (3 sizes x 4 frames) and the tail fit A over "
    "n >= 500 (DFT range [1.0, 2.0]) are diagnostics: a constant-offset model passes the slope rules for every offset up "
    "to 30 eV, and the DFT oracle itself passes the tail-A rule in only 57-65 % of draws. Per-file slope interval = 95 % "
    "percentile bootstrap over the 4 frames within each n (10,000 resamples); per-arm interval = 95 % t over seeds "
    "(4 seeds for AL and PAL; P and C0 have 2 seeds each, listed, no seed interval; those are screens). Na⁺ and Cl⁻ "
    "only; Mg²⁺ was not run at large n. The files hold the as-trained model only, so there is no LES-off ladder. EMA "
    "final files; same reference and analysis for every arm. The reference law is a fit to DFT clusters up to 192 waters, "
    "extrapolated to 12,001 atoms. A slope is fitted to the matched (XW, W) frame pairs present: where a run ended "
    "before its last frames (PAL seeds 13-15, Cl⁻ at n = 4,000, no failed evaluations) the slope uses fewer frames at "
    "that n, and the table gives the frames per n used.")


def a1_summary(rows, ion):
    """per-seed absolute errors at n = 1,000 and 4,000 against the draft A1 limits (see A1_LIMIT_EV)"""
    lim = A1_LIMIT_EV[ion]
    g = lambda r, n: r["err_mean"].get(n, r["err_mean"].get(str(n)))   # int keys when built, str keys after a json round trip
    e = {n: [g(r, n) for r in rows] for n in lim}
    ok = [all(g(r, n) is not None and abs(g(r, n)) <= t for n, t in lim.items()) for r in rows]
    return dict(abs_err={str(n): v for n, v in e.items()}, abs_pass_seeds=int(sum(ok)), abs_pass_all=bool(all(ok)))


def large_n_text(arms, dft_A):
    """one paragraph per arm for the notebooks; the absolute rule first, the slope as a diagnostic"""
    text = []
    for arm in ("AL", "PAL", "C0", "P"):
        if arm not in arms or any(i not in arms[arm] for i in LADDER_Q):
            continue
        S = {i: arms[arm][i]["summary"] for i in LADDER_Q}
        k = min(S[i]["n_seeds"] for i in LADDER_Q)
        iv = lambda s: (f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]" if s["n_seeds"] >= 3 else f"{s['mean']:+.4f}")
        rng = lambda i, n: (lambda v: f"{min(v):+.1f} to {max(v):+.1f}")([x for x in S[i]["abs_err"][str(n)] if x is not None])
        A = {i: [a_ for a_ in S[i]["A"] if a_ is not None] for i in LADDER_Q}
        text.append(
            f"{arm}, {k} seeds{' (a screen)' if k < 4 else ''}: error against the DFT law at n = 4,000 Na⁺ {rng('Na', 4000)} eV, "
            f"Cl⁻ {rng('Cl', 4000)} eV; at n = 1,000 Na⁺ {rng('Na', 1000)} eV, Cl⁻ {rng('Cl', 1000)} eV (range over seeds). "
            f"Draft absolute limits (ladder v2, not frozen): Na⁺ {A1_LIMIT_EV['Na'][1000]} eV at n = 1,000 and "
            f"{A1_LIMIT_EV['Na'][4000]} eV at 4,000; Cl⁻ {A1_LIMIT_EV['Cl'][1000]} and {A1_LIMIT_EV['Cl'][4000]} eV. Seeds inside "
            f"the limit at both n: Na⁺ {S['Na']['abs_pass_seeds']} of {S['Na']['n_seeds']}, Cl⁻ {S['Cl']['abs_pass_seeds']} of "
            f"{S['Cl']['n_seeds']}. Diagnostics, not pass/fail: slope over n = 1,000-4,000 Na⁺ {iv(S['Na'])}, Cl⁻ {iv(S['Cl'])} "
            f"eV/water (mean over seeds; 95 % t interval from 3 seeds up; per-seed intervals are a frame bootstrap); a "
            f"constant-offset model passes the slope rules for every offset up to 30 eV, so a small slope does not show "
            f"extrapolation (the error curve is concave and n beyond 4,000 is not tested). Tail-fit A {min(A['Na']):.0f} "
            f"to {max(A['Na']):.0f} for Na⁺ and {min(A['Cl']):.0f} to {max(A['Cl']):.0f} for Cl⁻ against DFT "
            f"{dft_A['Na']:.2f} and {dft_A['Cl']:.2f} (range 1.0-2.0); Na⁺ and Cl⁻ only, Mg²⁺ not run; as-trained model "
            "only, no LES on/off comparison.")
    return text


# ------------------------------------------------------------------------------------- claims-audit numbers + limits
def parse_typeheld_pdiff():
    """P - C0 on the type-held-out gate per pair set, 4 independent seeds (13r, 14, 15, 16), 95 % t (claims audit)"""
    p = DATA / "lr-program" / "claims_audit" / "typeheld_mg.txt"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = re.match(r"P - C0 (.+?): per seed \[([^\]]+)\], mean ([+-]?[0-9.]+) \[([+-]?[0-9.]+), ([+-]?[0-9.]+)\]", line)
        if m:
            key = ("without Mg2+" if m.group(1).startswith("without") else "with Mg2+" if m.group(1).startswith("with Mg")
                   else m.group(1).split()[0].replace("only", "").strip())
            out[key] = dict(per_seed=[float(x) for x in m.group(2).split(",")], mean=float(m.group(3)),
                            lo=float(m.group(4)), hi=float(m.group(5)), n=4)
    return out


def typeheld_sentence(D):
    if not D or "without Mg2+" not in D:
        return ""
    f = lambda d: f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
    s = (f"P − C0, gate-band force MAE over 4 independent seeds (13r, 14, 15, 16; 95 % t): without Mg²⁺ "
         f"{f(D['without Mg2+'])} meV/Å, with Mg²⁺ {f(D['with Mg2+'])} meV/Å.")
    if "Gdm+I-" in D and "K+Gdm+" in D:
        s += (f" Without Mg²⁺ the result rests on Gdm⁺I⁻ ({f(D['Gdm+I-'])}), and halide analogues of that pair are in P's "
              f"training rows; K⁺Gdm⁺ alone is {f(D['K+Gdm+'])}, an interval that includes zero.")
    return s


def evidence_limits(td):
    return [
        "Seeds and structures are different intervals. A claim needs 4 seeds and uses a 95 % t interval over seeds; "
        "bootstrap intervals over frames or scans describe one file and are narrower. The H100 seed-13 files are "
        "replicates of GH200 seed 13r and are never counted as a fifth seed.",
        "Thresholds are conventions of the program, not physical bounds: the ±0.02 eV/water small-n ladder slope and the "
        "0.2 e ion-shell limit. The large-n ladder no longer uses a slope rule: its draft limit (ladder v2, not frozen) is "
        "an absolute error of 0.86 / 1.14 eV (Na⁺) and 1.09 / 1.36 eV (Cl⁻) at n = 1,000 / 4,000 waters, 2 x the DFT "
        "law's fit uncertainty, with a resolution of about 1 eV.",
        "Multivalent systems (|q| ≥ 2, Mg²⁺) are excluded from every score by default and shown apart; the switch "
        "includes them. " + typeheld_sentence(td),
        "Every arm is a fine-tune of the same pretrained PET-OMol-S; a 'seed' is a different data order and "
        "initialisation of the new parameters, not an independent pre-training.",
        "P's seam and pairs-gate gains are in-distribution: P's band-prior rows are labelled with the fixed ESP-charge "
        "Coulomb oracle that the seam metrics also use (not DFT), and the pairs gate and the lrbench dimer set use the "
        "same 24 monomers as P's rows.",
        "The lrbench dimer set is not an external test for our arms: its monomers are our set-B monomers (identical to "
        "1e-6 Å), and 1,042 of its 1,254 frames are pair types in P's training rows (456 are set-B train types of every "
        "route-B arm). On it P also loses the partial ion + polar tail (ε −0.241 [−0.275, −0.206]), and AL/PAL show a "
        "like-charge overshoot (ε +0.234 to +0.258, 4 of 4 seeds) and a spurious ion + nonpolar tail.",
        "AL and PAL fail the draft absolute-error limit of the large-n ladder in all 4 of 4 seeds, for both ions: the "
        "error is +12 to +17 eV (Na⁺) and -10 to -15 eV (Cl⁻) at n = 1,000 and 17–31 eV in size at 4,000 waters, "
        "against limits of 0.9–1.4 eV. Their small slopes (inside ±0.01 eV/water) are not a pass: a constant-offset "
        "model passes the slope rules for every offset up to 30 eV (an arm with constant offsets, SB_K1, did).",
        "EMA: in the C0H run EMA lowers the val500 force error by 0.81 [0.16, 1.46] meV/Å but raises the energy error "
        "by +0.20 [0.03, 0.36] meV/atom; with the lr annealed, EMA and raw weights agree within ±0.2 meV/Å. C0H against "
        "C0 changes the schedule as well, so it is not an EMA test.",
        "Builds and hardware differ between some compared files (training on H100 or GH200, scoring on different "
        "builds); every comparison here pairs files scored by the same chain unless a row says otherwise.",
        "The adversarial sets (seam search set, sign-set cells) were designed after earlier failures were seen, so they "
        "are not blind tests; the type-held-out gate is the only pair set no arm had seen by type.",
    ]

# ------------------------------------------------------------------------------------------------ 4-seed verdicts
_STAT = re.compile(r"mean ([+-]?[0-9.]+) sd ([0-9.]+) 95% \[([+-]?[0-9.]+), ([+-]?[0-9.]+)\] n=(\d+)")


def _stat(line):
    m = _STAT.search(line)
    return None if not m else dict(mean=float(m.group(1)), sd=float(m.group(2)), lo=float(m.group(3)),
                                   hi=float(m.group(4)), n=int(m.group(5)))


def build_verdicts():
    """AL (A38) and PAL (A39) four-seed verdicts, read from the analysis summaries (numbers parsed, wording ours).
    The PAL slope part (L) is recomputed from its individual Na/Cl test lines: the printed L verdict of the summary is
    wrong (a type check in the summary script; errata of 17 Nov), the individual tests are not."""
    out = {}
    al = LF / "ceil" / "out_al4" / "summ.txt"
    if al.exists():
        txt = al.read_text().split("== AL final raw")[0]
        st = {}
        for line in txt.splitlines():
            line = line.strip()
            for key, pat in (("val500", "val500 F MAE (limit 42.0)"), ("Na", "slope heldout LES on Na:"),
                             ("Cl", "slope heldout LES on Cl:"), ("Mg", "slope heldout LES on Mg:"),
                             ("Na_off", "slope heldout LES off Na:"), ("Cl_off", "slope heldout LES off Cl:")):
                if line.startswith(pat):
                    st[key] = _stat(line)
        seeds = re.findall(r"CEILALfema(?:_s(\d+))?: val500 F ([0-9.]+) E [0-9.]+; shells Na/Cl ([0-9.]+)/([0-9.]+); sign set (\d) of 4", txt)
        per = [dict(seed=s or "13", val500=float(v), shell_na=float(a), shell_cl=float(b), cells=int(c)) for s, v, a, b, c in seeds]
        lead = re.search(r"VERDICT without Mg2\+ \(headline\): val500 (\w+), slopes (\w+), shells (\w+) -> lead candidate: (\w+)", txt)
        if st.get("val500") and lead:
            out["AL"] = dict(stats=st, seeds=per, val500_pass=lead.group(1) == "True", slopes_pass=lead.group(2) == "True",
                             shells_pass=lead.group(3) == "True", lead=lead.group(4) == "True")
    pal = LF / "ceil" / "out_pal" / "summ.txt"
    if pal.exists():
        txt = pal.read_text()
        sec = txt.split("4-seed statistics")[1] if "4-seed statistics" in txt else ""
        arms = {}
        cur = None
        for line in sec.splitlines():
            m = re.match(r"\s+(PAL|P|AL|C0): (S1)", line)
            if m:
                cur = m.group(1)
            for key, tag in (("S1", "S1 |x-1|"), ("S2", "S2 dmin"), ("S3", "S3 cells"), ("V", "V val500")):
                if tag in line and cur:
                    arms.setdefault(cur, {})[key] = _stat(line)
        shells = [dict(seed=s, na=float(a), cl=float(b)) for s, a, b in
                  re.findall(r"PALfema_s(\d+): val500 F [0-9.]+; shells Na/Cl ([0-9.]+)/([0-9.]+)", txt)]
        sl = {}
        for ion in ("Na", "Cl"):
            m = re.search(rf"L heldout LES on {ion}: PAL (mean .*?n=\d+)", txt)
            if m:
                sl[ion] = _stat(m.group(1))
        v = re.search(r"VERDICT \(A39, headline WITHOUT Mg2\+\): S \(seam kept\) (\w+), L \(slopes kept\) (\w+), V (\w+), Q (\w+) -> ADDS UP: (\w+)", txt)
        ltests = re.findall(r"^L \w+ LES \w+ (Na|Cl): .*-> \|PAL\| <= \|AL\| \+ sd\(AL\): (\w+)", txt, re.M)
        if arms and v:
            S, Vv, Q = v.group(1) == "True", v.group(3) == "True", v.group(4) == "True"
            L_ = len(ltests) == 8 and all(x[1] == "True" for x in ltests)
            out["PAL"] = dict(arms=arms, shells=shells, slopes=sl, seam_kept=S, slopes_kept=L_, n_slope_tests=len(ltests),
                              val500_pass=Vv, shells_pass=Q, adds_up=S and L_ and Vv and Q)
    return out


def verdict_text(V):
    """plain sentences for the notebooks (one claim each, numbers with n and the 95 % t interval)"""
    t = {}
    a = V.get("AL")
    if a:
        s = a["stats"]
        f = lambda d, nd=4: f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"
        sh = ", ".join(f"{x['shell_na']:.3f}" for x in a["seeds"])
        t["AL"] = (
            f"AL, 4 seeds (13-16, EMA; claim level), scored without Mg²⁺: val500 force MAE {s['val500']['mean']:.2f} "
            f"[{s['val500']['lo']:.2f}, {s['val500']['hi']:.2f}] meV/Å, inside the 42.0 limit. Held-out ladder slopes "
            f"(LES on, eV/water, limit ±0.02): Na⁺ {f(s['Na'])} fails; Cl⁻ {f(s['Cl'])}, whose interval crosses the limit, "
            f"so it fails as written. Na⁺ shell error per seed {sh} e against 0.2 e: above the limit in "
            f"{sum(x['shell_na'] > 0.2 for x in a['seeds'])} of {len(a['seeds'])} seeds. Sign-set cells passing per seed: "
            f"{'/'.join(str(x['cells']) for x in a['seeds'])} of 4. Verdict: AL is not the lead candidate"
            + (" (Mg²⁺ apart: Mg²⁺ slope " + f(s['Mg']) + ", not scored)." if s.get("Mg") else "."))
    p = V.get("PAL")
    if p:
        A = p["arms"]
        g = lambda d, nd=3: f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"
        t["PAL"] = (
            f"PAL, 4 seeds (13-16, EMA), scored without Mg²⁺: the seam gain of P is kept (in-distribution: P's band-prior rows are "
            "labelled with the fixed ESP-charge Coulomb oracle that the seam metrics also use, not DFT). NH₄⁺Gdm⁺ 7.5 Å |overshoot − 1| "
            f"{g(A['PAL']['S1'])} against P {A['P']['S1']['mean']:.2f} ± {A['P']['S1']['sd']:.2f} (sd); Gdm⁺Gdm⁺ LES-off dmin "
            f"rms {g(A['PAL']['S2'])} eV/Å against P {A['P']['S2']['mean']:.3f} ± {A['P']['S2']['sd']:.3f}. val500 force MAE "
            f"{A['PAL']['V']['mean']:.2f} [{A['PAL']['V']['lo']:.2f}, {A['PAL']['V']['hi']:.2f}] meV/Å, inside 42.0. Held-out "
            f"ladder slopes (LES on, eV/water): Na⁺ {p['slopes']['Na']['mean']:+.4f} against AL +0.0241, Cl⁻ {p['slopes']['Cl']['mean']:+.4f} against AL −0.0192. "
            "Na⁺ shell error per seed " + ", ".join(f"{x['na']:.3f}" for x in p["shells"]) + " e: above 0.2 e in every seed, so "
            f"the shell test fails. Pre-registered reading: seam kept {p['seam_kept']}, slopes kept {p['slopes_kept']} "
            f"(within AL's seed spread on all {p['n_slope_tests']} Na⁺/Cl⁻ tests; PAL's slopes are not better than AL's, "
            f"and both miss ±0.02), val500 {p['val500_pass']}, shells {p['shells_pass']}. Verdict: PAL does "
            + ("" if p["adds_up"] else "not ") + "add up" + ("" if p["adds_up"] else ", on the shell test only")
            + ", with and without Mg²⁺.")
    return t

# ------------------------------------------------------------------------------------------------ status
#: work that is running when the bundle is built. Each entry names the labels that will hold its numbers; the state
#: is read from disk at build time (none scored = running, some = partly scored, all = scored), so a rebuild after the
#: files land flips the entry without editing this list.
RUNNING = [
    dict(name="AL four-seed repeat (seeds 14, 15, 16)", amendment="A38",
         what="Arm AL (Q-free ceiling tower, lr 3e-5, 24,000 steps) repeated at seeds 14-16 to make the seed-13 screen a "
              "4-seed claim; scored with the same chain, reported with and without Mg²⁺.",
         labels=[f"CEILAL{w}_s{s}" for w in ("fema", "fraw") for s in ("14", "15", "16")],
         eta="results expected about 16:30 CET, 17 Nov 2026"),
    dict(name="PAL (P's band-prior rows + AL recipe, seeds 13-16)", amendment="A39",
         what="P's data5p charged-pair rows trained with AL's recipe. Pre-registered 'adds up' reading: PAL within P's "
              "4-seed spread on the NH₄⁺Gdm⁺ 7.5 Å overshoot and the Gdm⁺Gdm⁺ LES-off dmin rms, within AL's spread on "
              "the Na⁺/Cl⁻ ladder slopes, val500 upper 95 % ≤ 42.0 meV/Å, ion shells ≤ 0.2 e.",
         labels=[f"PAL{w}_s{s}" for w in ("fema", "fraw") for s in ("13", "14", "15", "16")],
         eta="results expected about 16:30 CET, 17 Nov 2026"),
    dict(name="Batch size and learning rate for the AL recipe", amendment="A40",
         what="Gradient-noise-scale probe at frozen weights, then a one-seed sweep of batch size and peak lr at a fixed "
              "budget of 32,000 samples (EMA off, compressed cosine). A screen: any recipe change needs a 4-seed "
              "confirmation arm. Sweep numbers are not comparable with the final-file numbers above.",
         labels=[], eta="sweep results not yet in this bundle"),
]


def build_status(prog):
    SC = merged_scorecard() or {}
    run = []
    for r in RUNNING:
        have = [L for L in r["labels"] if scored(L, SC)]
        state = ("running" if not have else "scored" if len(have) == len(r["labels"]) else
                 f"partly scored ({len(have)} of {len(r['labels'])} files)") if r["labels"] else "running"
        run.append(dict(r, state=state, scored=have, pending=[L for L in r["labels"] if L not in have]))
    note = DATA / "lr-program" / "IRIS_PARITY_NOTE.md"
    if note.exists():
        txt = note.read_text()
        done = "IN PROGRESS" not in txt
        m = re.search(r"- mlip\.cpp: E ([0-9.]+) meV/atom, F ([0-9.]+) meV/A", txt)
        res = (f" Result for the released PET-MAD-1.6-XS on all 23,671 test structures (CUDA, fp32): E {m.group(1)} "
               f"meV/atom, F {m.group(2)} meV/Å against the published 15.01 / 71.62.") if (m and done) else ""
        run.append(dict(name="mlip.cpp parity on the MAD-1.6 test set (PET-MAD-1.6-XS)", amendment="",
                        what="Energy and force MAE of the mlip.cpp port against the JAX reference implementation on the "
                             "23,671 MAD-1.6 test structures (fp32, no TF32). Published reference for the released "
                             "model: 15.01 meV/atom and 71.62 meV/Å (arXiv 2603.02089v3, Table 4)." + res,
                        labels=[], eta="", state="running" if not done else ("finished" if res else "finished (numbers not yet in this bundle)"),
                        scored=[], pending=[]))
    runs = prog["mdstab"]["runs"]
    md = {}
    for tag, sel in (("without Mg2+", lambda r: r["maxq"] < 2), ("Mg2+ apart", lambda r: r["maxq"] >= 2)):
        c = collections.Counter(r["verdict"] for r in runs if sel(r))
        md[tag] = dict(n=sum(c.values()), stable=c.get("stable", 0), flagged=c.get("flagged", 0),
                       unstable=c.get("unstable", 0))
    for tag, ref in (("without Mg2+", (199, 15, 0)), ("Mg2+ apart", (84, 3, 8))):   # 17 Nov reading
        now = (md[tag]["stable"], md[tag]["flagged"], md[tag]["unstable"])
        print(f"[lrp] MD stability {tag}: {now[0]}/{now[1]}/{now[2]}" + ("" if now == ref else f" (17 Nov reading {ref[0]}/{ref[1]}/{ref[2]})"))
    V = build_verdicts()
    VT = verdict_text(V)
    for r in run:
        key = "AL" if r["name"].startswith("AL ") else "PAL" if r["name"].startswith("PAL") else None
        if key and key in VT and r["state"] == "scored":
            r["result"] = VT[key]
            r["eta"] = ""
    return dict(running=run, verdicts=V, verdict_text=VT, pending_rows=prog.get("pending", {}), md_stability=md,
                built=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                not_yet_scored="not yet scored: no number exists for this file yet; the row or panel fills in when the "
                               "bundle is rebuilt (bash tools/rebuild_lr_program.sh)")


def publish_dir(out: pathlib.Path) -> int:
    """rename every internal label to <arm>_s<seed>_<ema|raw>, then run the gate; returns the number of gate hits"""
    sys.path.insert(0, str(HERE / "tools"))
    import lrp_public
    M = lrp_public.load_map()
    for f in sorted(out.iterdir()):
        if f.suffix in (".json", ".npz"):
            lrp_public.apply_file(f, M)
    n = lrp_public.check_paths([out], M)
    print(f"[lrp] public-label rewrite done; gate: {n} hit(s)")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "notebooks" / "data" / "lr_program"))
    ap.add_argument("--only", choices=("program", "scorecard"), default=None,
                    help="program = program.json + chemiscope.json; scorecard = the per-model files")
    ap.add_argument("--no-chemiscope", action="store_true")
    ap.add_argument("--quick", action="store_true", help="scorecard: skip the search-set and MD-frame blocks")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "program"):
        _load_static()
        prog = build_program()
        prog["built"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        prog["scans"] = build_scans()
        prog["ladder_large_n"] = build_ladder_big()
        prog["typeheld_pdiff"] = parse_typeheld_pdiff()
        prog["typeheld_pdiff_text"] = typeheld_sentence(prog["typeheld_pdiff"])
        prog["evidence_limits"] = evidence_limits(prog["typeheld_pdiff"])
        prog["status"] = build_status(prog)
        prog["sources"] = {"scorecard": "scorecard.json (program scorer)", "cells": "cells_<label>.json",
                           "results notes": ["A34_A35_RESULTS.md", "A30_REPEATS_RESULTS.md", "SEAM_FINAL_NOTE.md",
                                             "TYPEHELD_NOTE.md", "PERMIT_PET_RESULTS.md", "E1E2_NOTE.md",
                                             "SCANS_AL_NOTE.md"]}
        txt = json.dumps(_nan_to_null(prog), separators=(",", ":"), allow_nan=False)
        (out / "program.json").write_text(txt)
        print(f"[lrp] program.json {len(txt) / 1e6:.2f} MB, {len(prog['rows'])} rows, {len(prog['metrics'])} metrics")
        if not a.no_chemiscope:
            built, views = build_chemiscope()
            b2, v2 = build_scan_views()
            built.update(b2)
            views += v2
            if not any(k.startswith("lrb-") for k in built):
                print("[lrp] ERROR: no reference-only viewers: set LRBENCH_DATA to the canonical lrbench sets")
                return 2
            cj = json.dumps({"chemiscope": built, "views": views}, separators=(",", ":"))
            (out / "chemiscope.json").write_text(cj)
            print(f"[lrp] chemiscope.json {len(cj) / 1e6:.2f} MB, {len(built)} datasets")
    if a.only in (None, "scorecard"):
        sys.path.insert(0, str(HERE / "tools"))
        import lrp_scorecard
        lrp_scorecard.main(["--out", str(out)] + (["--quick"] if a.quick else []))
    return 1 if publish_dir(out) else 0


if __name__ == "__main__":
    raise SystemExit(main())
