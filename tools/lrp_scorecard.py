#!/usr/bin/env python3
"""Scorecard stage of tools/build_lr_program_data.py: the per-model bundle for notebooks/model_scorecard.py.

    python tools/build_lr_program_data.py                 # every stage, then the public-label rewrite and the gate
    python tools/build_lr_program_data.py --only scorecard [--quick]

Reads (never writes) the program and benchmark sources and writes small JSON / NPZ files that the notebook loads:

* index.json   the model registry (arm level, with every scored file as a row: seed, EMA or raw, host, step)
* prog.json    per scored file: every scorecard key, the rule-4 cells, ladder slopes, val500 and pairs-gate
               accuracy recomputed per frame with percentile-bootstrap intervals, type-held-out gate rows
* bench.json   per benchmark model: every metric row of every set in results/ (public rows only)
* md.json      MD stability runs and grid, RDF curves (block bootstrap), charges (MD and static vs DFT),
               permittivity (two protocols, kept apart), conservation suite and NVE drift
* speed.json   speed numbers (measured files where they exist, otherwise the cited note)
* chem_*.npz   geometry once per sub-test, DFT / oracle references, per-file model forces (float16)

Numbers come from the score JSONs, not from the notes. The notes are used only for recipe text.
Private result rows (results/private) are never read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("LRNB_DATA_ROOT", "~/data")).expanduser()
LF = DATA / "lr-fix-likecharge"
SF = LF / "seam/final"
MD = DATA / "mdstab"
PERMIT = DATA / "permit"
REPO = Path(__file__).resolve().parent.parent
EXT_RUNS = DATA / "lrbench-ext/explorer_public/runs"     # public rows only; explorer_private is never read
EXT_DATA = DATA / "lrbench-ext/ext-data"
KE = 14.399645
RNG = np.random.default_rng(7)

# ------------------------------------------------------------------------------------------------ registry
# label -> (arm, seed, weights, hardware, step). Seed 13r = same-seed GH200 replicate of the H100 seed-13 run.
# Labels are the program's file names; the public bundle renames them to <arm>_s<seed>_<ema|raw> (tools/lrp_public.py).
def _lab_rows():
    rows = {}
    rows["ORIG"] = ("ORIG", "13", "raw (no EMA)", "H100", "12,000 of a 14,170-step cosine")
    for arm in ("C0", "P"):
        rows[f"{arm}fema"] = (arm, "13", "EMA 0.999", "H100", "12,000")
        rows[f"{arm}fraw"] = (arm, "13", "raw", "H100", "12,000")
        rows[f"{arm}3kema"] = (arm, "13", "EMA 0.999", "H100", "3,000 (mid-run, lr near peak)")
        rows[f"{arm}3kraw"] = (arm, "13", "raw", "H100", "3,000 (mid-run, lr near peak)")
    for arm, ema, steps in (("C0", "EMA 0.999", "12,000"), ("P", "EMA 0.999", "12,000"),
                            ("C0H", "EMA 0.9995 (no warm-up)", "12,000"),
                            ("A0", "EMA 0.999", "6,000 (from P seed-13 EMA)"), ("A1", "EMA 0.999", "6,000 (from P seed-13 EMA)")):
        for s in ("13r", "14", "15", "16"):
            rows[f"{arm}cl_s{s}"] = (arm, s, ema, "GH200", steps)
            rows[f"{arm}clraw_s{s}"] = (arm, s, "raw", "GH200", steps)
    rows["A0fema"] = ("A0", "13", "EMA 0.999", "H100", "6,000 (from P seed-13 EMA)")
    rows["A1fema"] = ("A1", "13", "EMA 0.999", "H100", "6,000 (from P seed-13 EMA)")
    for c in ("A", "B", "C", "AL"):
        st = "24,000 (peak lr 3e-5)" if c == "AL" else "12,000"
        rows[f"CEIL{c}fema"] = (f"CEIL-{c}", "13", "EMA 0.999", "GH200", st)
        rows[f"CEIL{c}fraw"] = (f"CEIL-{c}", "13", "raw", "GH200", st)
    # four-seed repeats registered before their results exist: a label appears once its score files are on disk
    for s in ("14", "15", "16"):
        rows[f"CEILALfema_s{s}"] = ("CEIL-AL", s, "EMA 0.999", "GH200", "24,000 (peak lr 3e-5)")
        rows[f"CEILALfraw_s{s}"] = ("CEIL-AL", s, "raw", "GH200", "24,000 (peak lr 3e-5)")
    for s in ("13", "14", "15", "16"):
        rows[f"PALfema_s{s}"] = ("PAL", s, "EMA 0.999", "GH200", "24,000 (peak lr 3e-5)")
        rows[f"PALfraw_s{s}"] = ("PAL", s, "raw", "GH200", "24,000 (peak lr 3e-5)")
    return rows

LABELS = _lab_rows()
# earlier arms scored by the same scorer before the seam lane (score/sc/F_*_final): raw final files, H100
EXTRA = {"F_Bq3_K1_s13_final": ("Bq3_K1", "13"), "F_Bq3_K1_s14_final": ("Bq3_K1", "14"),
         "F_Bq4m3L_c01_s13_final": ("Bq4m3L_c01", "13"), "F_CTLc_s13_final": ("CTLc", "13"),
         "F_CTLc_s14_final": ("CTLc", "14"), "F_SB_K1_s13_final": ("SB_K1", "13"),
         "F_Bq4RS_c01_s13_final": ("RS_c01", "13"), "F_Bq4HY_c01_s13_final": ("HY", "13"),
         "F_HYo_c01_s13_final": ("HYo", "13")}
for _k, (_a, _s) in EXTRA.items():
    LABELS[_k] = (_a, _s, "raw (no EMA)", "H100", "24,000" if "m3L" in _k else "12,000")
LABELS["PETOMOLS"] = ("pet-omol-s", "-", "pretrained (GGUF export)", "-", "-")


def merged_scorecard(force=()):
    """every scorecard row for the registered labels.

    Sources, first one wins: the shared scorecard; the extra scorecards named in the private map (earlier and later
    scorer runs; one of them is rewritten by every new analysis, so its rows are copied into a persistent cache the
    first time they are seen); the cache; and, for a registered label whose per-structure outputs are on disk but
    which no scorecard holds yet, the program's scorer run here (results cached). `force` re-scores labels (tests)."""
    import subprocess
    import tempfile
    sys.path.insert(0, str(REPO / "tools"))
    try:
        from lrp_public import load_map
        M = load_map()
    except SystemExit:
        M = {}
    sc = dict(jload(SF / "sc/scorecard.json"))
    cache_p = DATA / M.get("scorecard_cache", "lr-program/notebook_publish/scorecard_cache.json")
    cache = jload(cache_p) if cache_p.exists() else {}
    for rel_ in M.get("extra_scorecards", []):
        p = DATA / rel_
        if p.exists():
            for k, v in jload(p).items():
                if k in LABELS:
                    cache.setdefault(k, v)
    for k, v in cache.items():
        sc.setdefault(k, v)
    todo = [L for L in LABELS if (L not in sc or L in force) and (SF / "sc" / L / "val500.json").exists()]
    scorer = DATA / M.get("scorer", "lr-fix-likecharge/scripts/scorecard.py")
    if todo and scorer.exists():
        with tempfile.TemporaryDirectory() as tmp:
            for L in todo:
                os.symlink(SF / "sc" / L, Path(tmp) / L)
            cmd = ["uv", "run", "-q", "--with", "numpy", "--with", "ase", "--with", "scipy", "python", str(scorer), tmp] + todo
            print(f"[prog] scoring {len(todo)} label(s) with no scorecard row yet")
            r = subprocess.run(cmd, cwd=str(SF), capture_output=True, text=True)
            got = jload(Path(tmp) / "scorecard.json") if (Path(tmp) / "scorecard.json").exists() else {}
            if r.returncode or not got:
                print("[prog] scorer failed; these labels stay 'not yet scored':", todo, (r.stderr or "")[-400:])
            for k, v in got.items():
                cache[k] = v
                sc[k] = v
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(cache, indent=0))
    return sc


def scdir(L):
    if L in EXTRA:
        return LF / "score/sc" / L
    return SF / "sc" / L


def ladder_from_outputs(CM, P):
    """cl_gate.py logic: error of dE(n) = E[XW_n] - E[W_n] - E[X] against DFT, mean over held-out centres, slope for
    n >= 24 (eV per water) and the error at n = 32 / 64 / 96"""
    def curves(E):
        X, W, XW = {}, {}, {}
        for i, m in enumerate(CM):
            k = (m["sys"], m["centre"], m["nw"])
            if m["role"] == "X" and m["sys"] not in X:
                X[m["sys"]] = E[i]
            elif m["role"] == "W":
                W[k] = E[i]
            elif m["role"] == "XW":
                XW[k] = E[i]
        return {k: XW[k] - W[k] - X[k[0]] for k in XW if k in W and k[0] in X}
    D = curves([m["E"] for m in CM])
    C = curves([p.get("energy", float("nan")) for p in P])
    out = {}
    for ion, sysn in (("Na+", "Na"), ("Cl-", "Cl"), ("Mg2+", "Mg")):
        err = defaultdict(list)
        for k, v in C.items():
            if k[0] == sysn and k in D and v == v:
                err[k[2]].append(v - D[k])
        ns = sorted(n for n in err if n >= 24)
        if len(ns) < 2:
            continue
        sl = float(np.polyfit(ns, [np.mean(err[n]) for n in ns], 1)[0])
        e = lambda n: f"{np.mean(err[n]):+.2f}" if err.get(n) else "n/a"
        out[ion] = dict(err_32_64_96=f"{e(32)} / {e(64)} / {e(96)}", slope=sl, A=float("nan"))
    return out
# the file used for structure viewers when a model has several (first seed scored on the full chain, EMA weights)
REP = {"pet-omol-s": "PETOMOLS", "Bq3_K1": "F_Bq3_K1_s13_final", "Bq4m3L_c01": "F_Bq4m3L_c01_s13_final", "CTLc": "F_CTLc_s13_final",
       "SB_K1": "F_SB_K1_s13_final", "RS_c01": "F_Bq4RS_c01_s13_final", "HY": "F_Bq4HY_c01_s13_final",
       "HYo": "F_HYo_c01_s13_final", "ORIG": "ORIG", "C0": "C0fema", "P": "Pfema", "C0H": "C0Hcl_s13r", "A0": "A0fema", "A1": "A1fema",
       "CEIL-A": "CEILAfema", "CEIL-B": "CEILBfema", "CEIL-C": "CEILCfema", "CEIL-AL": "CEILALfema",
       "PAL": "PALfema_s13"}
# labels that only exist as search outputs (one-off rescoring); not in the scorecard and not shown
EXCLUDED = {"C0 rescoring duplicate": "rescoring duplicate of C0 seed 13 (not shown)",
            "C0 stray rescoring file": "stray C0 rescoring file (errata; not shown)"}
_EXCLUDED_LABELS = ("C0r13", "C0fema_cl")

PROGRAM = [
    dict(id="pet-omol-s", name="PET-OMol-S (base)", kind="program+benchmark", mdstab="PET-OMol-S", permit="PET",
         recipe="Pretrained PET-OMol-S, no fine-tune. Program numbers (type-held-out gate, MD, permittivity) use the "
                "mlip.cpp GGUF export (fp32, ragged, no TF32); benchmark rows use the metatomic-torch checkpoint "
                "through ASE. These are two builds of one model, and the claims name the build."),
    dict(id="ORIG", name="Route B, Bq4m3_c01 (ORIG)", kind="program", mdstab="Bq4m3_c01",
         recipe="Amendment 23: Bq4m2_c01 plus the cluster-ladder training records; global total-charge conditioning, "
                "LES latent-charge Coulomb term (erf, sigma 2.5 A). Cut at step 12,000 of a 14,170-step cosine, no EMA, "
                "no IPR4 row weighting. One seed (13), H100."),
    dict(id="C0", name="C0 (seam control)", kind="program",
         recipe="Amendment A29a control: from Bq4_K1, data4m3 (226,734 rows), IPR4 pair rows x4, batch 16, 12,000 steps "
                "cosine, EMA 0.999 with warm-up. Seed 13 on H100; seeds 13r (same-seed replicate), 14, 15, 16 on "
                "GH200 (A34). C0 differs from ORIG by EMA, IPR4 and the full anneal together."),
    dict(id="P", name="P (seam band prior)", kind="program",
         recipe="C0 recipe plus 32,760 oracle-labelled charged-pair rows (data5p; fixed ESP charges, not DFT; no Mg2+, "
                "Li+F-, Na+Cl- or K+Gdm+). Seeds as C0. W* of the A30 rule under the pairs-gate reading."),
    dict(id="C0H", name="C0H (held lr, EMA 0.9995)", kind="program",
         recipe="Amendment A35: C0 recipe with the lr held at 30 % of peak and EMA 0.9995 without warm-up. "
                "Seeds 13r, 14, 15, 16, GH200. The raw file is a non-annealed iterate."),
    dict(id="A0", name="A0 (N = 16 neighbours, from P)", kind="program",
         recipe="Amendment A30 GO 1: init P seed-13 EMA, 16 attention neighbours, data5p, 6,000 steps, EMA 0.999. "
                "Seed 13 H100; 13r, 14, 15, 16 GH200."),
    dict(id="A1", name="A1 (N = 32 neighbours, from P)", kind="program",
         recipe="A0 with 32 attention neighbours (training step x1.5). Seeds as A0. Each file is scored at its own N."),
    dict(id="CEIL-A", name="CEIL-A (no global Q, charges + potentials)", kind="program",
         recipe="Amendment A36 ceiling arm: global total-charge embedding replaced by the Q = 0 row and frozen; energy "
                "tower fed local charges and smoothed potentials from a frozen 3-layer charge pass; otherwise C0 recipe. "
                "Seed 13 only, GH200 (screen)."),
    dict(id="CEIL-B", name="CEIL-B (no global Q, charges only)", kind="program",
         recipe="A36 arm B: as CEIL-A with local charges only. Seed 13 only (screen)."),
    dict(id="CEIL-C", name="CEIL-C (control, global Q kept)", kind="program",
         recipe="A36 control: CEIL-A features and init with the global Q embedding kept. Seed 13 only (screen)."),
    dict(id="CEIL-AL", name="CEIL-AL (CEIL-A, 2x steps, 3x lr)", kind="program",
         recipe="A36 arm AL: CEIL-A with peak lr 3e-5 and 24,000 steps. Seeds 13-16 (seeds 14-16 are the "
                "four-seed repeat of amendment A38). The 1-D scans use seed 13 only."),
    dict(id="PAL", name="PAL (P's band-prior rows + AL recipe)", kind="program",
         recipe="Amendment A39: P's data5p charged-pair rows with AL's Q-free ceiling tower, peak lr 3e-5 and 24,000 "
                "steps; seeds 13-16, GH200. Pre-registered 'adds up' reading: PAL within P's 4-seed spread on the "
                "NH4+Gdm+ 7.5 A overshoot and Gdm+Gdm+ LES-off dmin rms, within AL's spread on the Na+/Cl- ladder "
                "slopes, val500 upper 95 % <= 42.0 meV/A, ion shells <= 0.2 e."),
]
MD_ONLY = [
    ("Bq3_K1", "Bq3 K1", "Earlier route-B arm (section 5): global conditioning, conditioning force in training. Seed 13."),
    ("Bq4m3L_c01", "Bq4m3L", "Amendment 26: Bq4m3_c01 with peak lr 3e-5 and 24,000 steps. Seed 13."),
    ("CTLc", "CTLc (control)", "Matched control fine-tune of the route-B program, no long-range head. Seed 13."),
    ("SB_K1", "SB K1 (local conditioning)", "Bq3 setup with local conditioning and the conditioning force in training. Seed 13."),
    ("FC3b13", "FC3b (frozen charges)", "Frozen-charge arm (lrc = 1, padded path). Seed 13."),
    ("FTq13", "FTq (frozen charges)", "Frozen-charge arm (lrc = 1, padded path). Seed 13."),
    ("RS_c01", "RS (range-separated Coulomb)", "Amendment 24: erf kernel times a C2 switch 6-9 A. Seed 13; open structures only."),
    ("HY", "HY (hybrid conditioning)", "Amendment 25: global + gated local conditioning, g learned from 0. Seed 13."),
    ("HYo", "HYo (hybrid, g0 = 0.5)", "Amendment 28: HY with g initialised at 0.5 (g did not move). Seed 13."),
]
PERMIT_NAME = {"PET": "pet-omol-s", "RS": "RS_c01", "HY": "HY", "HYo": "HYo"}


def jload(p):
    with open(p) as f:
        return json.load(f)


def frames(p):
    try:
        return jload(p)["frames"]
    except Exception:
        return None


def boot(per_unit_num, per_unit_den=None, nb=2000, q=(2.5, 97.5)):
    """ratio-of-sums estimate with a percentile bootstrap over units (frames); asymmetric by construction"""
    a = np.asarray(per_unit_num, float)
    b = np.ones_like(a) if per_unit_den is None else np.asarray(per_unit_den, float)
    if len(a) == 0:
        return None
    est = a.sum() / b.sum()
    idx = RNG.integers(0, len(a), (nb, len(a)))
    bs = a[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.percentile(bs, q)
    return [float(est), float(lo), float(hi), int(len(a))]


# ------------------------------------------------------------------------------------------------ program per file
def read_xyz_plain(path):
    from ase.io import read
    return read(str(path), ":")


def val500_stats(stage):
    """per-frame force/energy errors for val500, split by total charge; atom-weighted MAE with frame bootstrap"""
    import re as _re
    V = open(stage / "val500.xyz").read().splitlines()
    out, i = [], 0
    while i < len(V):
        n = int(V[i]); hdr = V[i + 1]
        er = float(_re.search(r"e_res=(\S+)", hdr).group(1)); q = int(_re.search(r"charge=(-?\d+)", hdr).group(1))
        pr = _re.search(r"Properties=(\S+)", hdr).group(1).split(":"); off = 0
        for k in range(0, len(pr), 3):
            if pr[k] == "forces":
                break
            off += int(pr[k + 2])
        body = [l.split() for l in V[i + 2:i + 2 + n]]
        F = np.array([[float(v) for v in l[off:off + 3]] for l in body])
        fq = np.array([float(l[4]) for l in body]) if pr[6] == "formal_charges" else np.zeros(n)
        Z = [l[0] for l in body]; R = np.array([[float(v) for v in l[1:4]] for l in body])
        out.append(dict(n=n, er=er, q=q, F=F, Z=Z, R=R, maxq=species_maxq(Z, R, fq, q))); i += n + 2
    return out


ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "As", "Te", "Ge"}


def species_maxq(Z, R, fq, Q=0):
    """largest |charge| of a chemical species in the frame (multivalent = >= 2).

    Covalent components are built among non-metal atoms only (every metal / alkali atom is its own component, so
    Na+ or K+ sharing a ligand never merges two ions). A component's charge is the sum of its formal charges, and
    the largest |atomic| formal charge also counts (Fe2+, V3+ centres bonded to ligands). When the frame is one
    covalent component with no metal, its total charge Q is the species charge. The formal-charge column of val500
    does not sum to Q in some frames, so this is a heuristic; the builder prints the count."""
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, NeighborList
    from scipy.sparse.csgraph import connected_components
    a = Atoms(symbols=Z, positions=R)
    nl = NeighborList(natural_cutoffs(a, mult=1.1), self_interaction=False, bothways=True)
    nl.update(a)
    C = nl.get_connectivity_matrix(sparse=False).copy()
    metal = np.array([z not in ORGANIC for z in Z])
    C[metal, :] = 0; C[:, metal] = 0
    nc, lab = connected_components(C)
    comp = max(abs(fq[lab == c].sum()) for c in range(nc))
    whole = abs(Q) if ((nc == 1 and not metal.any()) or abs(fq.sum() - Q) > 1e-6) else 0   # inconsistent column -> |Q|
    return int(round(max(comp, np.abs(fq).max(), whole)))


def _fragment_sums(r, scheme):
    """per-fragment DFT population sums [q_f] for one scheme ('lowdin', 'hirshfeld'), or None"""
    refs, fc = r.get("refs") or {}, r.get("fragment_charges") or []
    fq = refs.get(f"fragment_charges_{scheme}")
    if fq is not None and len(fq) == len(fc):
        return [float(x) for x in fq]
    q, fid = refs.get(f"charges_{scheme}"), r.get("fragment_id")
    if q is not None and fid is not None and len(q) == len(fid) and len(fc) > 1:
        s = [0.0] * len(fc)
        for qi, k in zip(q, fid):
            if 0 <= int(k) < len(fc):
                s[int(k)] += float(qi)
        return s
    if scheme == "lowdin" and refs.get("qA_lowdin") is not None and len(fc) == 2:
        a = float(refs["qA_lowdin"])
        return [a, float(r.get("charge", 0)) - a]
    return None


def breakdown_fields(r):
    """Per-record inputs of the dielectric-breakdown toggle (notebooks/lrb_data.py: bd_flags); the notebook applies
    the rule, threshold and mode, the bundle only stores the raw labels (as it does 'maxq' for the multivalent switch).

    dq_frag           largest |DFT population sum - formal charge| over the record's fragments (Löwdin, else
                      Hirshfeld), NaN when the record has one fragment or no population labels
    q_electrode       electrode charge (e), sigma_electrode surface charge (uC/cm^2): NaN unless the record carries
                      them (top level, 'extra' or 'refs'); no current set has electrodes, the fields are the hook for
                      electrode-water sets"""
    refs, ex = r.get("refs") or {}, r.get("extra") or {}
    fc = r.get("fragment_charges") or []
    pop = None
    if len(fc) > 1:
        pop = _fragment_sums(r, "lowdin") or _fragment_sums(r, "hirshfeld")

    def pick(*keys):
        for src in (r, ex, refs):
            for k in keys:
                if isinstance(src.get(k), (int, float)):
                    return float(src[k])
        return np.nan
    return dict(dq_frag=max(abs(p - f) for p, f in zip(pop, fc)) if pop else np.nan,
                q_electrode=pick("electrode_charge", "q_electrode"),
                sigma_electrode=pick("surface_charge", "sigma_electrode", "sigma"))


def per_file_val500(V, P):
    fe, fn, ee, q, mq = [], [], [], [], []
    for v, p in zip(V, P):
        f = np.array(p["forces"]).reshape(-1, 3)
        fe.append(np.abs(f - v["F"]).sum()); fn.append(f.size)
        ee.append(abs(p["energy"] - v["er"]) / v["n"]); q.append(v["q"]); mq.append(v["maxq"])
    fe, fn, ee, q, mq = map(np.array, (fe, fn, ee, q, mq))
    res = {}
    mono = mq < 2
    # splits with and without multivalent frames (any atom with |formal charge| >= 2); the notebook picks per toggle
    for name, m in (("all", np.ones(len(q), bool)), ("charged (Q != 0)", q != 0), ("neutral (Q = 0)", q == 0),
                    ("all | no multivalent", mono), ("charged (Q != 0) | no multivalent", (q != 0) & mono),
                    ("neutral (Q = 0) | no multivalent", (q == 0) & mono), ("multivalent only (max |q| >= 2)", ~mono)):
        if m.any():
            F = boot(1e3 * fe[m], fn[m]); E = boot(1e3 * ee[m])
            res[name] = dict(F=F, E=E)
    return res


def setB_stats(meta, P):
    """set-B held-out pairs (DFT forces), force MAE at r >= 8 A by pair type, frame bootstrap"""
    acc = defaultdict(lambda: ([], []))
    for m, p in zip(meta, P):
        if m["mono"] or m["r"] < 8:
            continue
        d = np.abs(np.array(p["forces"]).reshape(-1, 3) - np.array(m["F"]))
        for k in (m["pair"], "all types"):
            acc[k][0].append(1e3 * d.sum()); acc[k][1].append(d.size)
    return {k: boot(a, b) for k, (a, b) in acc.items()}


def parse_ladders():
    """ladder slopes per label from every scorer table (held-out and train centres, LES on and off)"""
    out = defaultdict(dict)
    files = [(p, "held-out" if "heldout" in p else "train") for p in
             glob.glob(str(SF / "out/*ladder_*.md")) + glob.glob(str(SF / "out_a30rep/*ladder_*.md"))
             + glob.glob(str(LF / "ceil/out_*/ladder_*.md"))]
    for p, cen in files:
        for line in open(p):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) < 10 or not re.match(r"^\S+-(on|off)$", c[0]):
                continue
            lab, les = c[0].rsplit("-", 1)
            if lab.endswith("_s13"):
                lab = lab[:-4]
            row = {}
            for ion, j in (("Na+", 1), ("Cl-", 4), ("Mg2+", 7)):
                try:
                    row[ion] = dict(err_32_64_96=c[j], slope=float(c[j + 1]), A=float(c[j + 2]))
                except ValueError:
                    pass
            out[lab][f"{cen}, LES {les}"] = row
    dft = {"Na+": 1.37, "Cl-": 1.84, "Mg2+": 1.54}
    return dict(out), dft


def parse_typeheld():
    out = {}
    cur = None
    for line in open(LF / "typeheld/scores_all.txt"):
        m = re.match(r"== (\S+)", line)
        if m:
            cur = m.group(1); out[cur] = {}; continue
        m = re.match(r"\s+(\S.*?): r>=8 A \(gate band\): F MAE ([\d.]+) \[([\d.]+),([\d.]+)\] meV/A \(zero-force ([\d.]+)\), "
                     r"\|dE_int\| median ([\d.]+) eV, n=(\d+) frames", line)
        if m and cur:
            out[cur][m.group(1)] = dict(F=[float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(7))],
                                        zero_force=float(m.group(5)), dEint_median=float(m.group(6)))
    return out


def parse_cons(path):
    rows = []
    for line in open(path):
        m = re.match(r"^(\w+)\s+(\w+)\s+(pass|fail|report|n/a|error)\s*(\S*)\s*(.*)$", line.rstrip())
        if m and len(m.group(1)) <= 4:
            rows.append(dict(id=m.group(1), check=m.group(2), status=m.group(3), value=m.group(4), detail=m.group(5)[:160]))
        if line.startswith("scorecard gate"):
            rows.append(dict(id="gate", check="scorecard gate", status="PASS" if ": PASS" in line else "FAIL", value="", detail=""))
    return rows


# ------------------------------------------------------------------------------------------------ chemiscope blocks
def pack_frames(frames_list):
    """list of (symbols, positions) -> species str array, positions float32, offsets"""
    sym = np.concatenate([np.asarray(s, dtype="U3") for s, _ in frames_list])
    pos = np.concatenate([np.asarray(r, np.float32) for _, r in frames_list])
    off = np.cumsum([0] + [len(s) for s, _ in frames_list]).astype(np.int64)
    return sym, pos, off


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "notebooks/data/lr_program"))
    ap.add_argument("--quick", action="store_true", help="skip the search-set and MD-frame chemiscope blocks")
    a = ap.parse_args(argv)
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    sc = merged_scorecard()
    labels = [l for l in LABELS if l in sc or scdir(l).is_dir()]
    CM_ = jload(LF / "cl/cl_heldout.meta.json")
    print(f"[prog] {len(labels)} scored files of {len(LABELS)} registered; scorecard has {len(sc)} labels")
    unk = sorted(set(sc) - set(LABELS) - set(_EXCLUDED_LABELS))
    if unk:
        print("[prog] scorecard labels not registered (not shown):", unk)

    stage = LF / "ceil/stage"
    V = val500_stats(stage)
    SB = jload(LF / "score/setB_eval.meta.json")
    ladders, ladder_dft = parse_ladders()
    typeheld = parse_typeheld()
    prog = {}
    for L in labels:
        d = scdir(L)
        rec = dict(zip(("arm", "seed", "weights", "host", "step"), LABELS[L]))
        rec["scorecard"] = sc.get(L, {})
        cj = SF / f"cells_{L}.json"
        rec["cells"] = jload(cj) if cj.exists() else None
        rec["ladder"] = dict(ladders.get(L) or {})
        CLEV = {"F_Bq3_K1_s13_final": "Bq3K1", "F_CTLc_s13_final": "CTLc13", "F_SB_K1_s13_final": "SB_K1"}
        for suf, tag in (("", "on"), ("_g0", "off")):
            P = frames(d / f"cl_heldout{suf}.json") or (frames(LF / f"cl/ev/{CLEV[L]}{suf}.json") if L in CLEV else None)
            if P and f"held-out, LES {tag}" not in rec["ladder"]:
                rec["ladder"][f"held-out, LES {tag} (recomputed)"] = ladder_from_outputs(CM_, P)
        rec["ladder"] = rec["ladder"] or None
        for suf, key in (("", "val500"), ("_g0", "val500_les_off")):
            P = frames(d / f"val500{suf}.json")
            rec[key] = per_file_val500(V, P) if P else None
        for suf, key in (("", "setB"), ("_g0", "setB_les_off")):
            P = frames(d / f"setB_eval{suf}.json")
            rec[key] = setB_stats(SB, P) if P else None
        rec["typeheld"] = typeheld.get(L)
        prog[L] = rec
        print(f"[prog] {L}: val500 {rec['val500']['all']['F'][0] if rec['val500'] else 'n/a'}; ladder {list((rec['ladder'] or {}).keys())}")
    P = frames(LF / "cl/ev/PETOMolS.json")
    if P:
        prog["PETOMOLS"] = dict(zip(("arm", "seed", "weights", "host", "step"), LABELS["PETOMOLS"]), scorecard={}, cells=None,
                                val500=None, val500_les_off=None, setB=None, setB_les_off=None, typeheld=typeheld.get("PETOMOLS"),
                                ladder={"held-out, LES on (recomputed)": ladder_from_outputs(CM_, P)})
        labels.append("PETOMOLS")
    # cross-check: the recomputed ladder slope equals the scorer table for a label that has both
    _P = frames(SF / "sc/Pfema/cl_heldout.json")
    if _P:
        _r = ladder_from_outputs(CM_, _P)["Na+"]["slope"]; _t = prog["Pfema"]["ladder"]["held-out, LES on"]["Na+"]["slope"]
        assert abs(_r - _t) < 6e-5, (_r, _t)
        print(f"[prog] ladder recompute check Pfema Na+ {_r:+.4f} vs table {_t:+.4f}")
    # sanity: recomputed val500 F MAE equals the scorecard value
    for L, r in prog.items():
        s = r["scorecard"].get("val500 F MAE")
        if s is not None and r["val500"]:
            assert abs(s - r["val500"]["all"]["F"][0]) < 0.02, (L, s, r["val500"]["all"]["F"][0])
    # conservation suite + NVE (C0 and P, seeds 13r-16, MD form of the EMA file)
    cons = {}
    for f in sorted(glob.glob(str(LF / "sched/nve/*.json"))):
        J = jload(f); k = Path(f).stem  # C0_s13r
        arm, s = k.split("_s")
        c = J.get("checks", {})
        cons.setdefault(arm, {})[s] = dict(
            nve={kk: dict(name=v.get("name"), status=v.get("status"), value=v.get("value"), ci=v.get("ci"),
                          unit=v.get("unit"), tol=v.get("tol")) for kk, v in c.items()},
            ms_step=[J.get("ms_step_p10"), J.get("ms_step_med"), J.get("ms_step_p90")], gpu=J.get("gpu", {}).get("gpu"),
            frame=J.get("frame"), n_atoms=J.get("n_atoms"))
        for path in ("ragged", "padded"):
            t = LF / f"sched/cons/{k}_md.{path}.{path}.txt"
            if t.exists():
                cons[arm][s][f"suite_{path}"] = parse_cons(t)
    json.dump(dict(labels=prog, ladder_dft_tail_A=ladder_dft, excluded=EXCLUDED, cons=cons), open(OUT / "prog.json", "w"))

    # type-held-out PET-OMol-S base row
    pet_th = typeheld.get("PETOMOLS")

    # ---------------------------------------------------------------------------------------- benchmark rows
    bench = {}
    # third-party sets (ext_*) and results/private are never read
    for setdir in sorted(p for p in (REPO / "results").iterdir() if p.is_dir() and p.name != "private"
                         and not p.name.startswith("ext")):
        for f in sorted(setdir.glob("*.json")):
            J = jload(f); m = J.get("model") or f.stem
            b = bench.setdefault(m, dict(sets={}))
            b["sets"][setdir.name] = dict(
                metrics=J.get("metrics", []), model_level=J.get("model_level"), level_mismatch=J.get("level_mismatch"),
                evidence=J.get("evidence"), date=J.get("date"), code_commit=J.get("code_commit"), n_scored=J.get("n_scored"),
                n_failed=J.get("n_failed"), members=[{k: v for k, v in mm.items() if k in ("backend", "package", "version", "checkpoint")}
                                                     for mm in J.get("members", [])])
    sets_json = jload(REPO / "results/sets.json")
    json.dump(dict(models=bench, sets={k: dict(description=v.get("description"), reference_level=v.get("reference_level"),
                                               licence=v.get("licence")) for k, v in sets_json.items()
                                       if not k.startswith("ext")}),
              open(OUT / "bench.json", "w"))
    print(f"[bench] {len(bench)} models")

    # ---------------------------------------------------------------------------------------- MD
    st = jload(MD / "results/stability.json")
    keep = ("model", "sys", "nw", "T0", "seed", "natoms", "status", "verdict", "flags", "drift_meV_atom_ps", "drift_ci",
            "T_mean", "T_sd", "t_prod_ps", "fmax_max", "oh_max", "broken_frames", "evap_max", "nshell_first", "nshell_last",
            "dion_min", "dion_max", "ms_step", "fail")
    runs = []
    for r in st["runs"]:
        x = {k: r.get(k) for k in keep}
        x["gpu"] = (r.get("gpu") or {}).get("gpu")
        if isinstance(x["fail"], dict) and x["fail"]:
            x["fail"] = f"{x['fail'].get('why')} at {x['fail'].get('prod_ps', 0):.2f} ps (fmax {x['fail'].get('fmax', 0):.0f})"
        else:
            x["fail"] = ""
        runs.append(x)
    incomplete = [dict(model=r["model"], name=r["name"], state=r["state"][:80]) for r in st.get("incomplete", [])]
    eps_md = jload(MD / "results/eps.json")
    eps_md = [{k: v for k, v in e.items() if k != "g_K_blocks"} for e in eps_md]
    permit = []
    for f, tag in ((PERMIT / "analysis_final/permit_eps.json", "final"), (PERMIT / "analysis_pet2/permit_eps.json", "pet2")):
        for e in jload(f):
            if tag == "final" and e["model"] == "PET":
                continue   # superseded by the 4-seed pet2 analysis (the final file has seeds 1-2 only)
            e = dict(e); e["model_id"] = PERMIT_NAME.get(e["model"], e["model"]); permit.append(e)
    ch_md = jload(MD / "results/charges_md.json")
    ch_st = jload(MD / "results/charges_static.json")
    ch_md = dict(summary=ch_md["summary"], note=ch_md.get("note"))
    # RDF curves from frames.npy (ion-O and O-O), 2 ps blocks, bootstrap band (asymmetric)
    rdf = {}
    md_frames = []  # chemiscope: final frame of each ok run with n <= 200
    RHO0, DR = 0.0334, 0.1
    RB = np.arange(1.5, 8.0 + DR, DR); RC = RB[:-1] + DR / 2; VOL = 4 * np.pi * RC ** 2 * DR
    ION = {11: "Na", 12: "Mg", 17: "Cl"}
    for r in st["runs"]:
        if r.get("status") != "ok" or r.get("seed") == 9 or r["nw"] not in (48, 200):
            continue
        dname = MD / "runs" / r["model"] / f"{r['sys']}_n{r['nw']}_T{r['T0']}_s{r['seed']}"
        fn = dname / "frames.npy"
        if not fn.exists():
            continue
        S = jload(dname / "summary.json"); Z = np.array(S["Z"])
        F = np.load(fn, mmap_mode="r")
        if not (r["nw"] == 48 or r["T0"] == 300):
            pass
        O = np.where(Z == 8)[0]; ion = [i for i, z in enumerate(Z) if z in ION]
        sel = F[:: 2] if r["nw"] == 200 else F   # every 40 fs for 601 atoms keeps it cheap
        gi, go = [], []
        for x in sel:
            x = np.asarray(x, np.float64)
            if ion:
                d = np.linalg.norm(x[O] - x[ion[0]], axis=1); gi.append(np.histogram(d, RB)[0] / (VOL * RHO0))
            XO = x[O]; D = np.linalg.norm(XO[:, None] - XO[None], axis=2); D = D[D > 0.5]
            go.append(np.histogram(D, RB)[0] / (len(O) * RHO0 * VOL))
        nb = 100 if r["nw"] == 48 else 50   # 2 ps blocks
        key = f"{r['model']}|{r['sys']}|{r['nw']}|{r['T0']}"
        for kind, G in (("ionO", gi), ("OO", go)):
            if not G:
                continue
            G = np.array(G); n = (len(G) // nb) * nb
            B = G[:n].reshape(-1, nb, G.shape[1]).mean(1)
            rdf.setdefault(key, {}).setdefault(kind, []).append(B)
        last = np.asarray(F[-1], np.float32)
        md_frames.append(dict(model=r["model"], sys=r["sys"], nw=r["nw"], T=r["T0"], seed=r["seed"], verdict=r.get("verdict"),
                              drift=r.get("drift_meV_atom_ps"), nshell=r.get("nshell_last"), fmax=r.get("fmax_max"),
                              Z=Z, R=last))
    rdf_out = {}
    for key, d in rdf.items():
        rdf_out[key] = {}
        for kind, Bs in d.items():
            B = np.concatenate(Bs)
            if not len(B):
                continue
            idx = RNG.integers(0, len(B), (1000, len(B)))
            bs = B[idx].mean(1)
            rdf_out[key][kind] = dict(mean=np.round(B.mean(0), 4).tolist(), lo=np.round(np.percentile(bs, 2.5, 0), 4).tolist(),
                                      hi=np.round(np.percentile(bs, 97.5, 0), 4).tolist(), n_blocks=int(len(B)),
                                      n_runs=len(Bs))
    json.dump(dict(runs=runs, incomplete=incomplete, eps_md=eps_md, permit=permit, charges_md=ch_md,
                   charges_static=ch_st, rdf=rdf_out, rdf_r=np.round(RC, 3).tolist(),
                   rdf_note="g_ion-O(r) = n(r) / (4 pi r^2 dr rho0), rho0 = 0.0334 A^-3: a radial number density in units "
                            "of bulk water, not a bulk g(r). O-O over all pairs in the cluster. 2 ps blocks, seeds pooled; "
                            "band = 2.5-97.5 % bootstrap of the block mean (asymmetric). No DFT MD exists for these "
                            "clusters; the reference curve is PET-OMol-S (base) run with the same protocol."),
              open(OUT / "md.json", "w"))
    print(f"[md] {len(runs)} runs, {len(rdf_out)} rdf keys, {len(md_frames)} final frames")

    # ---------------------------------------------------------------------------------------- speed
    speed = dict(
        md_ms_step="per MD run (stability runs, GH200 unless the GPU column says otherwise); see md.json",
        rows=[
            dict(model="A0", what="training throughput", value="30.4-31.0 struct/s (4 seeds)", hw="GH200", src="A30_REPEATS_RESULTS.md"),
            dict(model="A1", what="training throughput", value="20.1-20.4 struct/s (4 seeds), step cost x1.52", hw="GH200", src="A30_REPEATS_RESULTS.md"),
            dict(model="A0", what="training throughput", value="27.4 struct/s (seed 13)", hw="H100", src="program log, A30 one-seed read"),
            dict(model="A1", what="training throughput", value="18.4 struct/s (seed 13)", hw="H100", src="program log, A30 one-seed read"),
            dict(model="C0", what="training throughput", value="24.9 struct/s (seed 13, 121 logged steps)", hw="H100", src="program errata note"),
            dict(model="P", what="training throughput", value="27.4 struct/s (seed 13, 121 logged steps)", hw="H100", src="program errata note"),
            dict(model="A0", what="inference, MD-form graph, N = 16", value="600 atoms 23.27 ms; 3,000 atoms 226.07 ms (min of 3 x 25)", hw="RTX 4070 Ti SUPER", src="A30_REPEATS_RESULTS.md (same-graph reference file, not the trained file)"),
            dict(model="A1", what="inference, MD-form graph, N = 32", value="600 atoms 34.58 ms; 3,000 atoms 292.97 ms (x1.49 / x1.30)", hw="RTX 4070 Ti SUPER", src="A30_REPEATS_RESULTS.md (same-graph reference file, not the trained file)"),
            dict(model="ORIG", what="single structure, 12,001 atoms", value="1.55 s, 15.5 GB", hw="H100 fp32", src="program status note rule 6"),
            dict(model="pet-omol-s", what="single structure, 12,001 atoms", value="0.17 s, 13.6 GB", hw="H100 fp32", src="program status note rule 6"),
            dict(model="RS_c01", what="single structure, 12,001 atoms", value="1.51 s, 17.9 GB", hw="H100 fp32", src="program status note rule 6"),
            dict(model="CTLc", what="single structure, 12,001 atoms", value="2.2 s, 41 GB", hw="H100 fp32", src="program status note rule 6"),
            dict(model="SB_K1", what="single structure, 12,001 atoms", value="6.3 s, 32 GB", hw="H100 fp32", src="program status note rule 6"),
            dict(model="HYo", what="single structure, 12,001 atoms", value="6.3 s", hw="H100 fp32", src="program status note"),
        ])
    json.dump(speed, open(OUT / "speed.json", "w"))

    # ---------------------------------------------------------------------------------------- chemiscope blocks
    # val500: 120 frames, half charged half neutral, DFT forces; model forces for every scored file
    q = np.array([v["q"] for v in V]); ch = np.where(q != 0)[0]; ne = np.where(q == 0)[0]
    pick = np.sort(np.concatenate([RNG.choice(ch, min(60, len(ch)), replace=False), RNG.choice(ne, min(60, len(ne)), replace=False)]))
    sym, pos, off = pack_frames([(V[i]["Z"], V[i]["R"]) for i in pick])
    blk = dict(species=sym, positions=pos, offsets=off, F_ref=np.concatenate([V[i]["F"] for i in pick]).astype(np.float32),
               charge=q[pick], frame_index=pick, maxq=np.array([V[i]["maxq"] for i in pick]))
    for L in labels:
        for suf, tag in (("", "on"), ("_g0", "off")):
            P = frames(scdir(L) / f"val500{suf}.json")
            if P:
                blk[f"F|{L}|{tag}"] = np.concatenate([np.array(P[i]["forces"]).reshape(-1, 3) for i in pick]).astype(np.float16)
                blk[f"E|{L}|{tag}"] = np.array([1e3 * (P[i]["energy"] - V[i]["er"]) / V[i]["n"] for i in pick], np.float32)
    np.savez_compressed(OUT / "chem_val500.npz", **blk)

    # set-B pairs (all frames): DFT forces from the meta
    from ase.io import read
    FR = read(str(stage / "setB_eval.xyz"), ":")
    idx = [i for i, m in enumerate(SB) if not m["mono"]]
    sym, pos, off = pack_frames([(FR[i].get_chemical_symbols(), FR[i].positions) for i in idx])
    blk = dict(species=sym, positions=pos, offsets=off, F_ref=np.concatenate([np.array(SB[i]["F"]) for i in idx]).astype(np.float32),
               r=np.array([SB[i]["r"] for i in idx], np.float32), pair=np.array([SB[i]["pair"] for i in idx]),
               orient=np.array([SB[i]["o"] for i in idx]), nA=np.array([SB[i]["nA"] for i in idx]))
    for L in labels:
        for suf, tag in (("", "on"), ("_g0", "off")):
            P = frames(scdir(L) / f"setB_eval{suf}.json")
            if P:
                blk[f"F|{L}|{tag}"] = np.concatenate([np.array(P[i]["forces"]).reshape(-1, 3) for i in idx]).astype(np.float16)
    np.savez_compressed(OUT / "chem_setB.npz", **blk)

    # type-held-out charged pairs: DFT forces, model outputs per file in typeheld/out
    TL = jload(LF / "typeheld/typeheld_labels.json"); TM = jload(LF / "typeheld/typeheld_meta.json")
    TF = read(str(LF / "typeheld/typeheld_eval.xyz"), ":")
    lab_by_key = {}
    for v in TL.values():
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict) and "key" in x:
                    lab_by_key[x["key"]] = x
    idx = [i for i, m in enumerate(TM) if not m["mono"] and m["key"] in lab_by_key]
    sym, pos, off = pack_frames([(TF[i].get_chemical_symbols(), TF[i].positions) for i in idx])
    blk = dict(species=sym, positions=pos, offsets=off,
               F_ref=np.concatenate([np.array(lab_by_key[TM[i]["key"]]["F"]) for i in idx]).astype(np.float32),
               r=np.array([TM[i]["r_com"] for i in idx], np.float32), pair=np.array([TM[i]["pair"] for i in idx]),
               orient=np.array([TM[i]["orient"] for i in idx]), dmin=np.array([TM[i]["dmin"] for i in idx], np.float32))
    th_agg = {}
    for f in sorted((LF / "typeheld/out").glob("*.json")):
        L = f.stem
        if L.endswith("_g0"):
            L, tag = L[:-3], "off"
        else:
            tag = "on"
        L = "pet-omol-s" if L == "PETOMOLS" else L
        P = frames(f)
        blk[f"F|{L}|{tag}"] = np.concatenate([np.array(P[i]["forces"]).reshape(-1, 3) for i in idx]).astype(np.float16)
        # gate-band force MAE (r_COM >= 8 A), float32, frame bootstrap; aggregate WITHOUT Mg2+ (program decision 17 Nov)
        acc = defaultdict(lambda: ([], []))
        for i in idx:
            m = TM[i]
            if m["r_com"] < 8:
                continue
            dd = np.abs(np.array(P[i]["forces"]).reshape(-1, 3) - np.array(lab_by_key[m["key"]]["F"]))
            keys_ = [m["pair"], "all types incl. Mg2+"] + ([] if "Mg" in m["pair"] else ["K+Gdm+ and Gdm+I- (Mg2+ excluded)"])
            for k in keys_:
                acc[k][0].append(1e3 * dd.sum()); acc[k][1].append(dd.size)
        zf = defaultdict(lambda: ([], []))
        for i in idx:
            m = TM[i]
            if m["r_com"] < 8:
                continue
            dd = np.abs(np.array(lab_by_key[m["key"]]["F"]))
            for k in [m["pair"], "all types incl. Mg2+"] + ([] if "Mg" in m["pair"] else ["K+Gdm+ and Gdm+I- (Mg2+ excluded)"]):
                zf[k][0].append(1e3 * dd.sum()); zf[k][1].append(dd.size)
        th_agg.setdefault(L, {})[tag] = {k: dict(F=boot(a_, b_), zero_force=float(sum(zf[k][0]) / sum(zf[k][1])))
                                         for k, (a_, b_) in acc.items()}
    np.savez_compressed(OUT / "chem_typeheld.npz", **blk)
    json.dump(th_agg, open(OUT / "typeheld_agg.json", "w"))
    json.dump(dict(PETOMOLS=pet_th), open(OUT / "typeheld_base.json", "w"))

    # ladder (held-out centres): vertical binding energy error per cluster; DFT energies only, no DFT forces
    CM = jload(LF / "cl/cl_heldout.meta.json"); CF = read(str(stage / "cl_heldout.xyz"), ":")

    def curves(E):
        X, W, XW = {}, {}, {}
        for i, m in enumerate(CM):
            k = (m["sys"], m["centre"], m["nw"])
            if m["role"] == "X" and m["sys"] not in X:
                X[m["sys"]] = E[i]
            elif m["role"] == "W":
                W[k] = E[i]
            elif m["role"] == "XW":
                XW.setdefault(k, (E[i], i))
        return {k: (v[0] - W[k] - X[k[0]], v[1]) for k, v in XW.items() if k in W and k[0] in X}
    Dref = curves([m["E"] for m in CM])
    keys = sorted(Dref, key=lambda k: (str(k[0]), k[1], k[2]))
    fidx = [Dref[k][1] for k in keys]
    sym, pos, off = pack_frames([(CF[i].get_chemical_symbols(), CF[i].positions) for i in fidx])
    blk = dict(species=sym, positions=pos, offsets=off, ion=np.array([str(k[0]) for k in keys]),
               centre=np.array([k[1] for k in keys]), nw=np.array([k[2] for k in keys]),
               dE_ref=np.array([Dref[k][0] for k in keys], np.float64))
    for L in labels:
        P = frames(scdir(L) / "cl_heldout.json") if L != "PETOMOLS" else frames(LF / "cl/ev/PETOMolS.json")
        if not P:
            continue
        C = curves([p.get("energy", np.nan) for p in P])
        blk[f"dE|{L}|on"] = np.array([C[k][0] if k in C else np.nan for k in keys], np.float64)
        if L in REP.values():
            blk[f"F|{L}|on"] = np.concatenate([np.array(P[i]["forces"]).reshape(-1, 3) for i in fidx]).astype(np.float16)
    np.savez_compressed(OUT / "chem_ladder.npz", **blk)

    if not a.quick:
        # seam search set: 40 orientations per scored cell, oracle (Coulomb) reference, forces LES on and off
        M = jload(SF / "search_meta.json")
        cells = [("Gdm+_Gdm+", 8), ("Gdm+_Gdm+", 9), ("NH4+_NH4+", 8), ("NH4+_Gdm+", 7.5), ("K+_Gdm+", 8)]
        pickS = []
        for p, x in cells:
            ii = [i for i, m in enumerate(M) if m["pair"] == p and m["d"] == x]
            pickS += sorted(RNG.choice(ii, 40, replace=False).tolist())
        SR = read(str(stage / "search.xyz"), ":")
        sym, pos, off = pack_frames([(SR[i].get_chemical_symbols(), SR[i].positions) for i in pickS])
        U = []
        for i in pickS:
            R = SR[i].positions; nA = M[i]["nA"]; u = R[nA:].mean(0) - R[:nA].mean(0); U.append(u / np.linalg.norm(u))
        blk = dict(species=sym, positions=pos, offsets=off, pair=np.array([M[i]["pair"] for i in pickS]),
                   d=np.array([M[i]["d"] for i in pickS], np.float32), nA=np.array([M[i]["nA"] for i in pickS]),
                   axis=np.array(U, np.float32), coulomb=np.array([KE / M[i]["d"] ** 2 for i in pickS], np.float32))
        for arm, L in REP.items():
            if L in EXTRA or L == "PETOMOLS":
                continue
            for tag in ("on", "off"):
                cand = [SF / f"res/srch_F_{L}_s13_{tag}.json"] + [SF / f"res/srch_F_{L}_s{L.split('_s')[-1]}_{tag}.json"]
                f = next((c for c in cand if c.exists()), None)
                if f is None:
                    print("[search] missing", L, tag); continue
                P = jload(f)["frames"]
                blk[f"F|{L}|{tag}"] = np.concatenate([np.array(P[i]["forces"]).reshape(-1, 3) for i in pickS]).astype(np.float16)
                print("[search]", L, tag)
        np.savez_compressed(OUT / "chem_search.npz", **blk)

        # MD final frames (n = 48 and 200), positions only
        sym, pos, off = pack_frames([([{1: "H", 8: "O", 11: "Na", 12: "Mg", 17: "Cl"}.get(int(z), "X") for z in m["Z"]], m["R"])
                                     for m in md_frames])
        np.savez_compressed(OUT / "chem_md.npz", species=sym, positions=pos, offsets=off,
                            model=np.array([m["model"] for m in md_frames]), sys=np.array([m["sys"] for m in md_frames]),
                            nw=np.array([m["nw"] for m in md_frames]), T=np.array([m["T"] for m in md_frames]),
                            seed=np.array([m["seed"] for m in md_frames]), verdict=np.array([str(m["verdict"]) for m in md_frames]),
                            drift=np.array([m["drift"] if m["drift"] is not None else np.nan for m in md_frames], np.float32),
                            nshell=np.array([m["nshell"] if m["nshell"] is not None else -1 for m in md_frames]),
                            fmax=np.array([m["fmax"] if m["fmax"] is not None else np.nan for m in md_frames], np.float32))

    # benchmark dimers / trimers: geometry of each E_int group, DFT forces, per-model E_int (energies only)
    sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "src"))
    from build_chemiscope_views import load_set, group_value  # noqa: E402
    runs = {}
    for f in sorted(EXT_RUNS.glob("*.json")):
        J = jload(f)
        if f.stem in bench:   # only models with a committed public row
            runs[f.stem] = J
    for setname in ("dimers", "trimers"):
        recs, groups = load_set(str(EXT_DATA), setname)
        by_id = {r["id"]: r for r in recs}
        gs = [g for g in groups if g.get("kind") == "E_int" and g.get("ref") is not None and g["terms"][0][0] in by_id]
        sel = [by_id[g["terms"][0][0]] for g in gs]
        sym, pos, off = pack_frames([(r["species"], r["positions"]) for r in sel])
        blk = dict(species=sym, positions=pos, offsets=off,
                   F_ref=np.concatenate([np.array(r["forces"] if r["forces"] is not None else np.full((len(r["species"]), 3), np.nan))
                                         for r in sel]).astype(np.float32),
                   ref=np.array([1e3 * g["ref"] for g in gs], np.float32), cls=np.array([g["meta"].get("cls", "?") for g in gs]),
                   dmin=np.array([g["meta"].get("dmin") or np.nan for g in gs], np.float32),
                   name=np.array([g["name"] for g in gs]), charge=np.array([r.get("charge", 0) for r in sel]),
                   maxq=np.array([int(round(max([abs(x) for x in (r.get("fragment_charges") or [0])] + [abs(x) for x in (r.get("formal_charges") or [0])]))) for r in sel]))
        # dielectric-breakdown labels (raw; the notebook applies rule, threshold and mode). Electrode fields only
        # when some record carries them, so the bundle stays small for molecular sets.
        bdf = [breakdown_fields(r) for r in sel]
        blk["dq_frag"] = np.array([b["dq_frag"] for b in bdf], np.float32)
        for k in ("q_electrode", "sigma_electrode"):
            if any(np.isfinite(b[k]) for b in bdf):
                blk[k] = np.array([b[k] for b in bdf], np.float32)
        for m, J in runs.items():
            preds = (J["sets"].get(setname) or {}).get("predictions")
            if not preds:
                continue
            v = [group_value(g, preds) for g in gs]
            blk[f"Eint|{m}"] = np.array([1e3 * x if x is not None else np.nan for x in v], np.float32)
        np.savez_compressed(OUT / f"chem_{setname}.npz", **blk)
        print(f"[bench chem] {setname}: {len(gs)} groups, models {sum(k.startswith('Eint') for k in blk)}")

    # ---------------------------------------------------------------------------------------- index
    models = []
    for p in PROGRAM:
        p = dict(p)
        p["labels"] = [L for L in labels if LABELS[L][0] == p["id"]]
        p["rep"] = REP.get(p["id"])
        p["bench"] = p["id"] if p["id"] in bench else None
        models.append(p)
    for mid, name, rec in MD_ONLY:
        models.append(dict(id=mid, name=name, kind="program, earlier arm", recipe=rec,
                           labels=[L for L in labels if LABELS[L][0] == mid], rep=REP.get(mid),
                           mdstab=mid if any(r["model"] == mid for r in runs_list(st)) else None,
                           permit=next((k for k, v in PERMIT_NAME.items() if v == mid), None), bench=None))
    for m in sorted(bench):
        if m == "pet-omol-s":
            continue
        models.append(dict(id=m, name=m, kind="benchmark", recipe="External or group model evaluated by lrbench on the "
                           "public sets; see the identity table for checkpoint, package and level of theory.",
                           labels=[], rep=None, mdstab=None, permit=None, bench=m))
    json.dump(dict(models=models, built="from the program's score files, MD-stability and permittivity runs, and "
                   "results/ (public rows, ext_ sets left out)", rep=REP), open(OUT / "index.json", "w"), indent=0)
    tot = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"[done] {OUT} {tot / 1e6:.1f} MB, {len(models)} models")


def runs_list(st):
    return st["runs"]


if __name__ == "__main__":
    main()
