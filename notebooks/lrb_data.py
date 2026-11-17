"""Data layer for notebooks/model_scorecard.py: loads the distilled bundle and assembles chemiscope datasets.

The bundle (``data/lr_program/`` next to this file, shared with long_range_diagnostics.py) is written by
``tools/build_lr_program_data.py`` (one builder for both notebooks; ``bash tools/rebuild_lr_program.sh`` re-reads the
score files and fills in results that have landed since). Nothing here reads
``~/data`` directly, so the notebook runs unchanged on molab and in WebAssembly as long as ``data/`` travels with it.
Only numpy and the standard library are imported, so Pyodide can run it.

The chemiscope iframe page (``CHEMISCOPE_PAGE``) is a copy of the helper in ``long_range_diagnostics.py`` (chemiscope
1.1.0 from cdn.jsdelivr), copied rather than imported so that the two notebooks can change independently.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

KE = 14.399645  # e^2 / (4 pi eps0) in eV A
MV_NOTE = ("Multivalent systems (any species with |charge| >= 2: Mg2+, Ca2+, Zn2+, SO4 2-, ...) are excluded from "
           "scoring by default (program decision, 17 Nov 2026). Every row carries its largest species |charge| ('max |q|'). "
           "With the switch off they are left out of every aggregate and pass count and shown apart; with it on they "
           "count. Pre-registered rule texts are unchanged.")
ION_Q = {"Na": 1, "Na+": 1, "Cl": 1, "Cl-": 1, "Br": 1, "K": 1, "Li": 1, "Mg": 2, "Mg2+": 2, "Ca": 2, "Zn": 2,
         "NaCl_cip": 1, "NaCl_ssip": 1, "W": 0, "False": 0}


def mv_state(include: bool) -> str:
    return ("multivalent systems INCLUDED in this score (switch on)" if include
            else "multivalent systems EXCLUDED from this score (switch off, default)")


# ------------------------------------------------------------------------------------------- dielectric breakdown
# A frame is "breakdown" when charge has moved across a boundary that a polarisable medium should hold: the electrode
# charge has pushed the potential outside the medium's band-edge window (de Kam, Rinke, Stein & Loche 2026, LOREM,
# leave out their q = +1 e electrode data for this reason; Bergmann, Reuter & Hörmann, J. Chem. Phys. 164, 174110
# (2026), water polarisable only for phi ~ 2-7 V with PBE, accept (dphi/dq)^-1 ~ C0 = 2.124 +- 1.5 uF/cm^2). The
# benchmark emulates DFT, pathologies included, so the default is to KEEP these frames in every score and flag them.
# This is a separate label from the CT-ambiguous class (ion_pair_ct_ambiguous); the two are cross-tabulated, never merged.
BD_MODES = ("include", "exclude", "only")
BD_RULES = ("charge", "physics")
BD_Q_THR = 0.1              # e: |DFT Löwdin (else Hirshfeld) sum - formal charge| on any fragment above this = breakdown
BD_ELECTRODE_Q = 1.0        # e: electrode charge at or above this = breakdown (charge rule; LOREM's q = +1 e)
BD_SIGMA_THR = None         # uC/cm^2: |surface charge| at or above this = breakdown (charge rule); None = not used
BD_PHI_WINDOW = (2.0, 7.0)  # V: band-edge window of PBE water (Bergmann et al.); used only where 'phi' labels exist
BD_C0, BD_C0_TOL = 2.124, 1.5   # uF/cm^2: accepted inverse response (dphi/dq)^-1; used only where 'capacitance' exists
BD_NOTE = ("Dielectric breakdown: charge moved across a boundary a polarisable medium should hold. Rule 'charge' keys "
           "on an electrode charge (q >= +1 e, after de Kam, Rinke, Stein & Loche 2026, LOREM) or surface charge field; "
           "rule 'physics' keys on fractional charge transfer between fragments: |DFT Löwdin (else Hirshfeld) fragment "
           "sum - formal charge| above a threshold on any fragment (after Bergmann, Reuter & Hörmann, J. Chem. Phys. "
           "164, 174110 (2026)). Frames without fragment charge labels are counted as 'no charge labels' and not "
           "flagged. Default: "
           "include and flag, because the benchmark emulates DFT, pathologies included. 'exclude' drops flagged "
           "frames from the per-frame recomputed scores (dimer E_int) and the frame counts and shows them apart; 'only' "
           "keeps the flagged frames alone. Committed result rows and precomputed program scores are not split by this "
           "toggle; their panels say so. The dataset's CT-ambiguous class is a separate label, shown in its own column and "
           "never folded into the breakdown flag, so 'exclude' does not remove CT-ambiguous frames that the rule does "
           "not flag.")


def bd_config(args=None) -> dict:
    """breakdown mode, rule and threshold from marimo CLI args (--breakdown, --breakdown-rule, --breakdown-thr) or
    the environment (LRB_BREAKDOWN, LRB_BREAKDOWN_RULE, LRB_BREAKDOWN_THR); defaults include / physics / 0.1 e"""
    args = args or {}
    mode = str(args.get("breakdown") or os.environ.get("LRB_BREAKDOWN") or "include")
    rule = str(args.get("breakdown-rule") or os.environ.get("LRB_BREAKDOWN_RULE") or "physics")
    thr = float(args.get("breakdown-thr") or os.environ.get("LRB_BREAKDOWN_THR") or BD_Q_THR)
    if mode not in BD_MODES or rule not in BD_RULES:
        raise ValueError(f"breakdown={mode!r} rule={rule!r}; known {BD_MODES} x {BD_RULES}")
    return dict(mode=mode, rule=rule, thr=thr)


def bd_state(mode: str, rule: str, thr: float = BD_Q_THR) -> str:
    """the label every breakdown-aware score carries"""
    r = (f"rule physics, |fragment Löwdin - formal| > {thr:g} e" if rule == "physics"
         else f"rule charge, electrode q >= {BD_ELECTRODE_Q:g} e")
    s = {"include": "breakdown frames INCLUDED and flagged (default)", "exclude": "breakdown frames EXCLUDED",
         "only": "breakdown frames ONLY"}[mode]
    return f"{s}; {r}"


def bd_flags(blk: dict, rule: str, thr: float = BD_Q_THR) -> dict:
    """Per-frame breakdown flags of one bundle block (a dict of arrays as returned by :func:`block`).

    Returns mask (bool per frame), source (why each frame is flagged, '' if not), ct (CT-ambiguous class per frame),
    n_labelled (frames the rule could judge from labels) and msg (one line: what the rule found in this block).

    charge: reads 'q_electrode' (e) and 'sigma_electrode' (uC/cm^2) if the block has them; our sets have no
    electrodes, so this is an honest no-op. physics: 'dq_frag' (largest |DFT Löwdin, else Hirshfeld, fragment sum -
    formal charge| over the fragments, NaN without labels); frames without it are 'no charge labels', unflagged. The
    CT-ambiguous class is returned as its own label ('ct') and never enters the mask. 'phi' (V) and 'capacitance'
    (uF/cm^2) are judged when present (hooks, untested: no current set carries them)."""
    n = len(blk["offsets"]) - 1 if "offsets" in blk else len(next(iter(blk.values())))
    cls = np.asarray(blk["cls"]).astype(str) if "cls" in blk else np.full(n, "")
    ct = np.char.find(cls, "ct_ambiguous") >= 0
    mask = np.zeros(n, bool)
    src = np.full(n, "", dtype=object)
    if rule == "charge":
        lab = np.zeros(n, bool)
        if "q_electrode" in blk:
            q = np.asarray(blk["q_electrode"], float); ok = np.isfinite(q); lab |= ok
            hit = ok & (q >= BD_ELECTRODE_Q - 1e-9); mask |= hit; src[hit] = f"electrode q >= {BD_ELECTRODE_Q:g} e"
        if "sigma_electrode" in blk and BD_SIGMA_THR is not None:
            s = np.asarray(blk["sigma_electrode"], float); ok = np.isfinite(s); lab |= ok
            hit = ok & (np.abs(s) >= BD_SIGMA_THR) & ~mask; mask |= hit; src[hit] = f"|sigma| >= {BD_SIGMA_THR:g} uC/cm2"
        msg = ("no electrode frames in this set" if not lab.any()
               else f"{int(mask.sum())} of {int(lab.sum())} electrode frames flagged")
        if "sigma_electrode" not in blk:
            msg += "; surface charge: no labels in this set"
        return dict(mask=mask, source=src, ct=ct, n_labelled=int(lab.sum()), msg=msg)
    dq = np.asarray(blk["dq_frag"], float) if "dq_frag" in blk else np.full(n, np.nan)
    lab = np.isfinite(dq)
    hit = lab & (dq > thr); mask |= hit; src[hit] = f"|fragment Löwdin - formal| > {thr:g} e"
    hooks = []
    for key, test, why in (
            ("phi", lambda p: (p < BD_PHI_WINDOW[0]) | (p > BD_PHI_WINDOW[1]), f"phi outside {BD_PHI_WINDOW[0]:g}-{BD_PHI_WINDOW[1]:g} V"),
            ("capacitance", lambda c: np.abs(c - BD_C0) > BD_C0_TOL, f"capacitance outside {BD_C0:g} +- {BD_C0_TOL:g} uF/cm2")):
        if key not in blk:            # hook: no current set carries response labels (untested)
            hooks.append(f"{key}: no labels in this set")
            continue
        v = np.asarray(blk[key], float)
        h = np.isfinite(v) & test(np.nan_to_num(v)) & ~mask
        mask |= h; src[h] = why
    nolab = int((~lab).sum())
    msg = (f"{int(mask.sum())} of {n} frames flagged ({int(lab.sum())} with fragment charge labels, {nolab} with no "
           f"charge labels, counted and not flagged); CT-ambiguous {int(ct.sum())} (own label), both "
           f"{int((mask & ct).sum())}, CT-ambiguous not flagged {int((ct & ~mask).sum())}; " + "; ".join(hooks))
    if not lab.any():
        msg = (f"no charge labels in this set ({n} frames counted, none flagged); CT-ambiguous {int(ct.sum())} "
               "(own label); " + "; ".join(hooks))
    return dict(mask=mask, source=src, ct=ct, n_labelled=int(lab.sum()), msg=msg)


def bd_select(mask, mode: str):
    """frames that count in the score for a breakdown mode (include: all; exclude: unflagged; only: flagged)"""
    mask = np.asarray(mask, bool)
    return {"include": np.ones_like(mask), "exclude": ~mask, "only": mask}[mode]


_ELEMENT = r"[A-Z][a-z]?"


def species_charge(sp: str) -> int:
    """|charge| of one species name, read from an explicit charge suffix, never from formula digits.

    Accepted forms: 'Na+', 'Cl-', 'Gdm+', 'NH4+', 'NO3-', 'OAc-' (polyatomic, digits are stoichiometry: |q| = 1);
    'Mg2+', 'Ca2+' (a single element symbol followed by a count: the count is the charge); 'SO4 2-' or 'SO4_2-'
    style count separated by a space; 'SO4(2-)' (count and sign in brackets); '[Fe(H2O)6]3+' or '(SO4)2-' (count after
    a closing bracket); 'Mg++' / 'SO4--' (repeated signs). No trailing sign = neutral (0): 'H2O', 'MeCN'."""
    import re as _re
    s = str(sp).strip()
    m = _re.search(r"\((\d*)[+-]\)$", s)                     # 'SO4(2-)'
    if m:
        return int(m.group(1) or 1)
    if not s or s[-1] not in "+-":
        return 0
    m = _re.search(r"\s(\d+)\s*[+-]$", s)                    # 'SO4 2-'
    if m:
        return int(m.group(1))
    m = _re.search(r"[)\]](\d+)[+-]$", s)                     # '(SO4)2-', '[Fe(H2O)6]3+'
    if m:
        return int(m.group(1))
    m = _re.search(r"([+-])\1+$", s)                          # 'Mg++', 'SO4--'
    if m:
        return len(m.group(0))
    m = _re.fullmatch(rf"({_ELEMENT})(\d+)[+-]", s)          # 'Mg2+': monatomic, the count is the charge
    if m:
        return int(m.group(2))
    return 1                                                  # 'NH4+', 'Gdm+', 'NO3-', 'Cl-'


def pair_maxq(pair: str) -> int:
    """largest |charge| among the species named in a pair key such as 'Gdm+_Mg2+' (see :func:`species_charge`)"""
    return max([species_charge(sp) for sp in str(pair).split("_")] + [0])


def data_dir(notebook_dir=None) -> Path:
    env = os.environ.get("LRB_SCORECARD_DATA")
    if env:
        return Path(env)
    base = Path(notebook_dir) if notebook_dir else Path(__file__).resolve().parent
    return base / "data" / "lr_program"


#: in-memory bundle {file name: bytes}; set by a frozen (WebAssembly) copy of the notebook, which has no data folder
_MEM: dict = {}


def set_bundle(files: dict) -> None:
    _MEM.clear()
    _MEM.update(files)
    _json.cache_clear()
    block.cache_clear()


@lru_cache(maxsize=None)
def _json(d: str, name: str):
    if _MEM:
        return json.loads(_MEM[name]) if name in _MEM else None
    p = Path(d) / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def program_extra(d) -> dict:
    """status (running work, files not yet scored, MD-stability counts) and the 1-D scan block from program.json"""
    p = _json(str(d), "program.json") or {}
    return {"status": p.get("status") or {}, "scans": p.get("scans"), "pending": p.get("pending") or {},
            "ladder_large_n": p.get("ladder_large_n")}


def scan_rows(S, include_mv=False):
    """scan-quality table: one row per metric, one column per model; '[lo, hi]' are 95 % block-bootstrap bounds"""
    if not S:
        return []
    M = S["models"]
    AG = "AGG (excl. Mg2+)"

    def ci(t, nd=1):
        return "not yet scanned" if not t else f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"

    def med(d, nd=0):
        return "not yet scanned" if not d else f"{d['median']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}], max {d['max']:.{nd}f}"
    spec = [
        ("type-held-out pairs, F MAE 8–14 Å (meV/Å)", lambda m: ci((S["mae"].get(f"typeheld|{AG}|gate 8-14 A|{m}") or {}).get("F")), "12 scans"),
        ("type-held-out pairs, F MAE 6–8 Å (meV/Å)", lambda m: ci((S["mae"].get(f"typeheld|{AG}|6-8 A|{m}") or {}).get("F")), "12 scans"),
        ("type-held-out, net-force error on B, 10–14 Å (meV/Å)", lambda m: ci((S["mae"].get(f"typeheld|{AG}|tail 10-14 A|{m}") or {}).get("netB")), "12 scans"),
        ("type-held-out, median |ΔE_int| error, 8–14 Å (eV)", lambda m: ci((S["mae"].get(f"typeheld|{AG}|gate 8-14 A|{m}") or {}).get("dEint"), 3), "12 scans"),
        ("set-B gate pairs, F MAE 8–14 Å (meV/Å)", lambda m: ci((S["mae"].get(f"setB|{AG}|gate 8-14 A|{m}") or {}).get("F")), "18 scans"),
        ("set-B, net-force error on B, 10–14 Å (meV/Å)", lambda m: ci((S["mae"].get(f"setB|{AG}|tail 10-14 A|{m}") or {}).get("netB")), "18 scans"),
        ("kink 6–9 Å, type-held-out (meV/Å per step²)", lambda m: med(S["smooth"].get(f"typeheld|AGG|{m}", {}).get("excess_max_cut")), "12 scans"),
        ("kink 6–9 Å, set B (meV/Å per step²)", lambda m: med(S["smooth"].get(f"setB|AGG|{m}", {}).get("excess_max_cut")), "18 scans"),
        ("kink 6–9 Å, sign-set cells (meV/Å per step²)", lambda m: med(S["smooth"].get(f"search|AGG|{m}", {}).get("excess_max_cut")), "12 scans"),
        ("−dE/dr vs force, RMS 6–9 Å, type-held-out (meV/Å)", lambda m: med(S["smooth"].get(f"typeheld|AGG|{m}", {}).get("cons_rms_cut"), 1), "12 scans"),
        ("−dE/dr vs force, RMS 6–9 Å, set B (meV/Å)", lambda m: med(S["smooth"].get(f"setB|AGG|{m}", {}).get("cons_rms_cut"), 1), "18 scans"),
        ("right sign, Gdm⁺Gdm⁺ 8 Å (share)", lambda m: ci((S["search"].get(f"Gdm+_Gdm+|8|{m}") or {}).get("sign"), 3), "500 orientations"),
        ("right sign, K⁺Gdm⁺ 8 Å (share; held out)", lambda m: ci((S["search"].get(f"K+_Gdm+|8|{m}") or {}).get("sign"), 3), "500 orientations"),
        ("median force / Coulomb, Gdm⁺Gdm⁺ 10 Å", lambda m: ci((S["search"].get(f"Gdm+_Gdm+|10|{m}") or {}).get("med"), 2), "500 orientations"),
        ("median force / Coulomb, NH₄⁺NH₄⁺ 10 Å", lambda m: ci((S["search"].get(f"NH4+_NH4+|10|{m}") or {}).get("med"), 2), "500 orientations"),
    ]
    if include_mv:
        spec.append(("Gdm⁺Mg²⁺ (multivalent), F MAE 8–14 Å (meV/Å)",
                     lambda m: ci((S["mae"].get(f"typeheld|Gdm+_Mg2+|gate 8-14 A|{m}") or {}).get("F")), "6 scans"))
    rows = []
    for name, fn, n in spec:
        r = {"metric": name, "n": n}
        for m in M:
            r[m + (" (1 seed)" if m == "AL" else "")] = fn(m)
        rows.append(r)
    return rows


def ladder_big_rows(LB, arm=None):
    """large-n ladder table: one row per arm and ion (seed summary) plus the per-seed slopes with frame intervals"""
    rows = []
    if not LB:
        return rows
    for a in ("AL", "PAL", "P", "C0"):
        if a not in LB["arms"] or (arm and a != arm):
            continue
        for ion in ("Na", "Cl"):
            x = LB["arms"][a].get(ion)
            if not x:
                continue
            s = x["summary"]
            seed_txt = "; ".join(f"s{r['seed']} {r['slope']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for r in x["files"])
            fr = lambda r: "/".join(str(r["frames"].get(str(n), r["frames"].get(n, 0))) for n in r.get("n_used", []))
            frames_txt = "; ".join(f"s{r['seed']} n {r.get('n_used', ['?'])[0]}-{r.get('n_used', ['?'])[-1]}, frames {fr(r)}, "
                                   f"{r.get('records', '?')} records" for r in x["files"])
            e4 = [r["err_mean"].get("4000", r["err_mean"].get(4000)) for r in x["files"]]
            e4 = [v for v in e4 if v is not None]
            arm_txt = (f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] (95 % t, {s['n_seeds']} seeds)"
                       if s["n_seeds"] >= 3 else f"{s['mean']:+.4f} (mean of {s['n_seeds']} seed{'s' if s['n_seeds'] > 1 else ''}; no seed interval)")
            rows.append({"arm": a, "ion": {"Na": "Na⁺", "Cl": "Cl⁻"}[ion], "seeds": s["n_seeds"],
                         "slope n = 1,000-4,000 (eV/water)": arm_txt,
                         "per seed [frame bootstrap]": seed_txt,
                         "inside ±0.01": "yes" if s["within_0.01"] else "no",
                         "tail A per seed (DFT " + f"{LB['dft_A'][ion]:.2f}; pass 1.0-2.0)": ", ".join(f"{v:.1f}" for v in s["A"] if v is not None),
                         "error at n = 4,000 (eV, per-seed range)": (f"{min(e4):+.1f} to {max(e4):+.1f}" if e4 else "n/a"),
                         "n used, frames per n, records": frames_txt})
    return rows


def ladder_big_figure(LB, plt):
    """mean error of dE(n) against n per arm (seed mean; thin lines = seeds), Na⁺ and Cl⁻"""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    col = {"AL": "#C0392B", "PAL": "#8E44AD", "P": "#2C6FAC", "C0": "#7F7F7F"}
    for ax, ion in zip(axes, ("Na", "Cl")):
        for a in ("C0", "P", "AL", "PAL"):
            x = (LB["arms"].get(a) or {}).get(ion)
            if not x:
                continue
            ns = sorted(int(n) for n in x["files"][0]["err_mean"])
            for r in x["files"]:
                ax.plot(ns, [r["err_mean"][str(n)] if str(n) in r["err_mean"] else r["err_mean"][n] for n in ns],
                        color=col[a], lw=0.6, alpha=0.5)
            m = [sum((r["err_mean"].get(str(n)) if str(n) in r["err_mean"] else r["err_mean"][n]) for r in x["files"]) / len(x["files"]) for n in ns]
            ax.plot(ns, m, "o-", color=col[a], lw=1.6, ms=3, label=f"{a} ({len(x['files'])} seed{'s' if len(x['files']) > 1 else ''})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("waters n")
        ax.set_title({"Na": "Na⁺", "Cl": "Cl⁻"}[ion], loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("error of ΔE(n) vs DFT law (eV)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def load(d) -> dict:
    d = str(d)
    return {k: _json(d, f"{k}.json") for k in ("index", "prog", "bench", "md", "speed", "typeheld_agg", "typeheld_base")}


@lru_cache(maxsize=None)
def block(d: str, name: str):
    if _MEM:
        if f"chem_{name}.npz" not in _MEM:
            return None
        import io as _io
        z = np.load(_io.BytesIO(_MEM[f"chem_{name}.npz"]), allow_pickle=False)
        return {k: z[k] for k in z.files}
    p = Path(d) / f"chem_{name}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_ci(t, nd=2):
    """[est, lo, hi, n] -> 'est [lo, hi], n = n'"""
    if not t:
        return "n/a"
    s = f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"
    return s + (f" (n = {t[3]})" if len(t) > 3 else "")


def seed_summary(vals):
    """per-seed values -> 'median (min-max), n = k' (no symmetric spread is implied)"""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "n/a"
    return f"{np.median(v):.2f} (range {min(v):.2f} to {max(v):.2f}, n = {len(v)} seeds)"


def seed_line(df, col):
    """GH200 seeds 13r-16 summarised (median and range); H100 seed 13 listed apart (same seed as 13r)"""
    if df is None or len(df) == 0:
        return "n/a"
    cl = df[df["host"].str.contains("GH200")] if "host" in df else df.iloc[0:0]
    ku = df[~df.index.isin(cl.index)]
    parts = []
    if len(cl):
        parts.append("GH200 " + seed_summary(cl[col].tolist()).replace("seeds", f"seeds ({', '.join(cl['seed'].astype(str))})"))
    for _, r in ku.iterrows():
        parts.append(f"{r.get('host', '')} seed {r['seed']} {r[col]:.2f} (one file)")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------ chemiscope
def _frames(b):
    off = b["offsets"]
    return [(b["species"][off[i]:off[i + 1]], b["positions"][off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def dataset(b, name, description, sprops, aprops=None, arrows=None, sel=None):
    """chemiscope dataset dict.

    b: a block (species / positions / offsets); sprops: {name: (values per structure, units, description)};
    aprops: {name: (values per atom, units, description)}; arrows: {name: (N, 3) vectors per atom}; sel: structure
    indices to keep (the atom arrays are sliced to match)."""
    fr = _frames(b)
    off = b["offsets"]
    idx = list(range(len(fr))) if sel is None else list(sel)
    amask = np.concatenate([np.arange(off[i], off[i + 1]) for i in idx]) if idx else np.zeros(0, int)
    structures = []
    for i in idx:
        s, r = fr[i]
        structures.append({"size": int(len(s)), "names": [str(x) for x in s],
                           "x": np.round(r[:, 0].astype(float), 4).tolist(), "y": np.round(r[:, 1].astype(float), 4).tolist(),
                           "z": np.round(r[:, 2].astype(float), 4).tolist()})

    def _v(v):
        v = np.asarray(v)
        if v.dtype.kind in "US":
            return [str(x) for x in v]
        return [None if not np.isfinite(x) else round(float(x), 5) for x in v.astype(float)]
    props = {}
    for k, (v, u, d) in sprops.items():
        v = np.asarray(v)[idx]
        props[k] = {"target": "structure", "values": _v(v), "units": u, "description": d}
    for k, (v, u, d) in (aprops or {}).items():
        props[k] = {"target": "atom", "values": _v(np.asarray(v)[amask]), "units": u, "description": d}
    ds = {"meta": {"name": name, "description": description}, "structures": structures, "properties": props}
    if aprops or arrows:
        ds["environments"] = [{"structure": j, "center": int(c), "cutoff": 3.5}
                              for j, i in enumerate(idx) for c in range(off[i + 1] - off[i])]
    if arrows:
        ds["shapes"] = {}
        for k, vec in arrows.items():
            vec = np.asarray(vec, float)[amask]
            ds["shapes"][k] = {"kind": "arrow",
                               "parameters": {"global": {"baseRadius": 0.06, "headRadius": 0.14, "headLength": 0.25},
                                              "atom": [{"vector": np.round(x, 4).tolist()} for x in vec]}}
    return ds


def encode(ds) -> str:
    return base64.b64encode(gzip.compress(json.dumps(ds, separators=(",", ":")).encode(), 6)).decode()


CHEMISCOPE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chemiscope@1.1.0/dist/chemiscope.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js"></script>
<script>
// chemiscope without eval: SVG map instead of WebGL, a checked mount, and a 3Dmol stand-in (pattern shared with the program's pages)
(function () {   // chemiscope map: SVG scatter instead of WebGL scattergl (regl needs eval, blocked by the page CSP)
  try {
    const PM = window.Chemiscope && Chemiscope.PropertiesMap && Chemiscope.PropertiesMap.prototype;
    if (!PM || PM._svgPatched) return;
    const get = PM._getTraces;
    PM._getTraces = function () { const t = get.apply(this, arguments); for (const x of t) if (x.type === "scattergl") x.type = "scatter"; return t; };
    PM._svgPatched = true;
  } catch (e) { console.error("chemiscope SVG map patch failed", e); }
})();
const csEvalOK = (() => { try { return Function("return 1")() === 1; } catch (e) { return false; } })();
function csMapProblem(id) {          // "" when the chemiscope map in #id has drawn, else a reason
  const root = document.getElementById(id); if (!root) return "the map container is missing";
  let noGL = false, pts = 0, canv = 0, type = "";
  const walk = n => { if (!n) return; if (n.shadowRoot) walk(n.shadowRoot);
    if (n._fullData && n._fullData[0] && !type) type = n._fullData[0].type;
    if (n.classList) { if (n.classList.contains("no-webgl")) noGL = true; if (n.classList.contains("point")) pts++; }
    if (n.tagName === "CANVAS") canv++; for (const c of n.children || []) walk(c); };
  walk(root);
  if (noGL || (type === "scattergl" && !csEvalOK))
    return "the map's WebGL renderer cannot run here (it compiles code at run time, which this page's security policy forbids)";
  if (type === "scatter3d") return canv ? "" : "the 3D map drew nothing";
  return pts || (type === "scattergl" && canv) ? "" : "the map drew no points";
}
function csMount(boxes, data, timeoutMs) {   // load a chemiscope view, then check the map really drew; rejects with a readable Error
  if (!window.Chemiscope) return Promise.reject(new Error("the chemiscope library did not load from " + "https://cdn.jsdelivr.net/npm/chemiscope@1.1.0 (blocked or offline)"));
  const t0 = Date.now(), limit = timeoutMs || 6000;
  return Chemiscope.DefaultVisualizer.load(boxes, data).then(v => new Promise((res, rej) => {
    let ok = 0;                                 // success only after two clean checks 300 ms apart (Plotly draws asynchronously)
    const tick = () => { const p = csMapProblem(boxes.map);
      if (!p && ++ok >= 2) return res(v);
      if (!p) return setTimeout(tick, 300);
      ok = 0;
      if (p.indexOf("WebGL") >= 0 || Date.now() - t0 > limit) { try { v.remove(); } catch (e) {} return rej(new Error(p)); }
      setTimeout(tick, 250); };
    tick(); }));
}
function csShowError(el, what, err) {          // visible error text instead of a blank panel
  el.hidden = false; el.textContent = ""; const b = document.createElement("b"); b.textContent = what + " ";
  el.append(b, String(err && err.message ? err.message : err));
  if (err && err.stack) console.error(err);
}
function csXYZ(s) {                             // chemiscope structure -> XYZ text for a 3Dmol.js fallback viewer
  const L = [String(s.size), ""];
  for (let i = 0; i < s.size; i++) L.push(s.names[i] + " " + s.x[i] + " " + s.y[i] + " " + s.z[i]);
  return L.join("\\n") + "\\n";
}
function csFallback(host, structures, note) {   // 3Dmol.js stand-in for a chemiscope view that could not draw: frames from structures[]
  try {
    if (!window.$3Dmol) { const p = document.createElement("p"); p.className = "cs-err"; p.textContent = "The 3Dmol fallback viewer did not load either."; host.append(p); return; }
    const wrap = document.createElement("div"); wrap.className = "cs-fb"; const box = document.createElement("div"); box.className = "cs-fb-box";
    const row = document.createElement("div"); row.className = "cs-fb-row"; const sl = document.createElement("input"); sl.type = "range"; sl.min = 0; sl.max = Math.max(0, structures.length - 1); sl.value = 0; sl.setAttribute("aria-label", "Frame");
    const lab = document.createElement("span"); row.append(lab, sl); wrap.append(box, row); host.append(wrap);
    const v = $3Dmol.createViewer(box, { backgroundColor: "white" });
    const draw = () => { const i = +sl.value; v.clear(); v.addModel(csXYZ(structures[i]), "xyz"); v.setStyle({}, { stick: { radius: 0.12 }, sphere: { scale: 0.22 } }); v.zoomTo(); v.render(); lab.textContent = "Frame " + (i + 1) + " of " + structures.length + (note ? " · " + note : ""); };
    sl.addEventListener("input", draw); draw();
  } catch (e) { console.error(e); const p = document.createElement("p"); p.className = "cs-err"; p.textContent = "The fallback viewer could not draw either: " + e.message; host.append(p); }
}
</script>
<style>body{margin:0;font-family:Georgia,serif;font-size:13px}
.panes{display:flex;flex-wrap:wrap;gap:8px}.l{flex:1 1 58%;min-width:320px}.r{flex:1 1 38%;min-width:280px}
#m{height:430px}#s{height:340px}.err{color:#D55E00;font-family:sans-serif}/* chemiscope embed (CSP-safe): error text and 3Dmol fallback */ .cs-err { margin: 0; padding: 10px 12px; border: 1px solid #d9dde2; border-left: 3px solid #c0392b; border-radius: 4px; background: #ffffff; color: #14181c; font-size: 0.86rem; max-width: 95ch } .cs-fb { display: flex; flex-direction: column; gap: 6px; margin-top: 8px } .cs-fb-box { position: relative; width: 100%; height: 340px; background: #fff; border: 1px solid #d9dde2; border-radius: 4px } .cs-fb-row { display: flex; gap: 8px; align-items: center; font-size: 0.86rem; flex-wrap: wrap; color: #14181c } .cs-fb-row input { flex: 1 1 200px; min-width: 0 }</style></head>
<body><div class="panes"><div class="l"><div id="meta"></div><div id="m"></div></div>
<div class="r"><div id="s"></div><div id="i"></div></div></div>
<script>
const B = "__BLOB__", S = __SETTINGS__;
(async () => {
  let ds = null;
  try {
    const bin = Uint8Array.from(atob(B), c => c.charCodeAt(0));
    const st = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
    ds = JSON.parse(await new Response(st).text());
    for (const p of Object.values(ds.properties))
      if (p.values.every(v => v === null || typeof v === "number"))
        p.values = p.values.map(v => v === null ? NaN : v);
    if (S && Object.keys(S).length) ds.settings = S;
    await csMount({map:"m", info:"i", meta:"meta", structure:"s"}, ds, 8000);
  } catch (e) {
    const el = document.createElement("p"); el.className = "cs-err"; document.body.prepend(el);
    csShowError(el, "chemiscope could not draw this view:", e);
    if (ds && ds.structures) csFallback(document.getElementById("s"), ds.structures, "3Dmol stand-in");
  }
})();
</script></body></html>"""


def page(ds, settings=None) -> str:
    return CHEMISCOPE_PAGE.replace("__BLOB__", encode(ds)).replace("__SETTINGS__", json.dumps(settings or {}))


def page_blob(blob: str, settings=None) -> str:
    """the iframe page for a dataset that is already gzip+base64 (the program's chemiscope.json)"""
    return CHEMISCOPE_PAGE.replace("__BLOB__", blob).replace("__SETTINGS__", json.dumps(settings or {}))


def program_views(d, prefix="lrp-scan-"):
    """views of the shared chemiscope.json whose key starts with prefix, each with its blob"""
    c = _json(str(d), "chemiscope.json") or {}
    return [dict(v, blob=c["chemiscope"][v["key"]]) for v in c.get("views", [])
            if v["key"].startswith(prefix) and v["key"] in c.get("chemiscope", {})]


def settings(x, y, color=None, shape=None):
    s = {"target": "structure", "map": {"x": {"property": x}, "y": {"property": y}}}
    if color:
        s["map"]["color"] = {"property": color}
    if shape:
        s["structure"] = [{"shape": shape}]
    return s


# ------------------------------------------------------------------------------------------------ force views
def force_view(b, label, title, what, ref_name="DFT", sprops_extra=None, les_off=True, sel=None):
    """structure + atom properties for a block with reference forces F_ref and model forces F|label|on (and off)."""
    key = f"F|{label}|on"
    if b is None or key not in b:
        return None
    off = b["offsets"]
    Fm = b[key].astype(np.float32)
    Fr = b["F_ref"].astype(np.float32)
    dF = np.linalg.norm(Fm - Fr, axis=1)
    n = len(off) - 1
    mae = np.array([1e3 * np.abs(Fm[off[i]:off[i + 1]] - Fr[off[i]:off[i + 1]]).mean() for i in range(n)])
    sp = {"F MAE vs " + ref_name: (mae, "meV/A", f"mean |F_model - F_{ref_name}| over atoms and components"),
          "structure index": (np.arange(n), "", "")}
    ap = {"|F model|": (np.linalg.norm(Fm, axis=1), "eV/A", "model force, LES on"),
          f"|F {ref_name}|": (np.linalg.norm(Fr, axis=1), "eV/A", f"{ref_name} force"),
          "|F model - F ref|": (dF, "eV/A", "force error per atom")}
    arrows = {"model force": Fm, f"{ref_name} force": Fr, "force error": Fm - Fr}
    if les_off and f"F|{label}|off" in b:
        Fo = b[f"F|{label}|off"].astype(np.float32)
        sp["F MAE, LES off"] = (np.array([1e3 * np.abs(Fo[off[i]:off[i + 1]] - Fr[off[i]:off[i + 1]]).mean() for i in range(n)]),
                                "meV/A", "same weights with the LES gate at 0")
        ap["|F LES off - F ref|"] = (np.linalg.norm(Fo - Fr, axis=1), "eV/A", "force error per atom, LES off")
    for k, v in (sprops_extra or {}).items():
        sp[k] = v
    ds = dataset(b, title, what, sp, ap, arrows, sel=sel)
    return ds
