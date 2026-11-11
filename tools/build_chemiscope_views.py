#!/usr/bin/env python3
"""Build a chemiscope explorer: several pre-configured views over the scored structures, in one page.

    python tools/build_chemiscope_views.py RUNS_DIR [--data DIR] [-o OUT.html] [--open]

No model is re-evaluated. Every number comes from the canonical records and the per-term energies that
``lrbench eval --save-predictions`` already wrote, recombined through each group's own definition, so a model value
and its reference are formed identically.

Each view is a separate chemiscope instance with its own default axes, colouring and question. They share one
embedded dataset per set, so the page costs one copy of the data no matter how many views look at it, and every
view keeps chemiscope's own axis, colour, size and atom-colour controls -- the point is to hand those controls over
already pointed at something worth looking at, not to wrap them in menus of my own.

Global (per-structure) properties: class, separation, reference E_int, and per model its E_int, signed error and
absolute error. Atomic properties: fragment id, formal charge, and the DFT Löwdin charge, with an environment on
every atom so the structure panel can colour by them.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import pathlib
import sys
import webbrowser

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from lrbench.diagnostics import score_components  # noqa: E402  (after the path insert)
from lrbench import palette as pal  # noqa: E402

MEV = 1.0e3
PATHOLOGY = "dft_pathology_do_not_score"


# ------------------------------------------------------------------------------------------------- the data layer
def load_set(data: str, name: str):
    """Records and E_int/E3/ddE groups of one canonical set."""
    jl = os.path.join(data, "canonical", f"{name}.jsonl")
    gp = os.path.join(data, "canonical", f"{name}.groups.json")
    if not os.path.exists(jl):
        return [], []
    with open(jl) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    groups = []
    if os.path.exists(gp):
        with open(gp) as f:
            groups = json.load(f)
    return recs, groups


def group_value(g: dict, preds: dict):
    """The group's linear combination of predicted terms, or None if any term is missing."""
    total = 0.0
    for term, coef in g.get("terms") or []:
        vals = preds.get(term)
        if not vals:
            return None
        good = [v for v in vals if isinstance(v, (int, float))]
        if not good:
            return None
        total += float(coef) * (sum(good) / len(good))
    return total


def entries(results: dict, recs: list, setname: str, groups: list, kind: str):
    """One entry per scored group: the structure its first term names, the reference, and each model's value.

    Keying on the group rather than the record matters for ddE and E3, where several groups share a first term --
    keying on the record id would silently keep one of them and drop the rest.
    """
    by_id = {r["id"]: r for r in recs}
    gs = [g for g in groups if g.get("kind") == kind and g.get("ref") is not None
          and not any(PATHOLOGY in str(x) for x in ((g.get("meta") or {}).get("flags") or []))]
    preds = {name: (((J.get("sets") or {}).get(setname) or {}).get("predictions") or {})
             for name, J in results.items()}
    preds = {k: v for k, v in preds.items() if v}
    out = []
    for g in gs:
        rid = str((g.get("terms") or [[None]])[0][0] or "").split("#")[0]
        rec = by_id.get(rid)
        if rec is None or any(PATHOLOGY in str(f) for f in (rec.get("flags") or [])):
            continue
        vals = {}
        for name, P in preds.items():
            v = group_value(g, P)
            if v is not None:
                vals[name] = MEV * v
        if vals:
            out.append({"rec": rec, "ref": MEV * float(g["ref"]), "vals": vals,
                        "meta": g.get("meta") or {}, "name": g.get("name", "")})
    return out, sorted({m for e in out for m in e["vals"]})


def _nullify(x):
    """NaN is not valid JSON, and a browser's JSON.parse refuses it: write null and let the page restore NaN."""
    if isinstance(x, float):
        return None if (x != x or x in (float("inf"), float("-inf"))) else x
    if isinstance(x, dict):
        return {k: _nullify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nullify(v) for v in x]
    return x


def encode(payload: dict) -> str:
    return base64.b64encode(gzip.compress(
        json.dumps(_nullify(payload), allow_nan=False).encode())).decode()


def build_dataset(ents, models, setname, kind_label):
    """A chemiscope dataset: structures, global properties, atomic properties, an environment on every atom."""
    from ase import Atoms

    if not ents:
        return None
    keep = [e["rec"] for e in ents]
    frames, env = [], []
    for i, r in enumerate(keep):
        a = Atoms(r["species"], positions=np.asarray(r["positions"], float))
        if r.get("cell") is not None:
            a.set_cell(np.asarray(r["cell"], float))
            a.set_pbc(bool(np.any(r.get("pbc") or False)))
        frames.append(a)
        env += [(i, j, 3.5) for j in range(len(a))]

    def num(vals, unit, desc):
        return {"target": "structure", "values": [float(v) if v is not None else float("nan") for v in vals],
                "units": unit, "description": desc}

    def cat(vals, desc):
        return {"target": "structure", "values": [str(v) for v in vals], "description": desc}

    props = {
        "class": cat([r.get("cls") or "?" for r in keep], "interaction class of the pair"),
        "separation": num([r.get("distance") for r in keep], "Å",
                          "closest fragment–fragment contact, the canonical d_min"),
        f"{kind_label}/REF": num([e["ref"] for e in ents], "meV",
                                 f"reference {kind_label} (ωB97M-V/def2-TZVPD unless the set says otherwise)"),
        "total charge": num([r.get("charge") for r in keep], "e", "total charge of the structure"),
        "atoms": num([len(r["species"]) for r in keep], "", "number of atoms"),
    }
    for m in models:
        have = [e["vals"].get(m) for e in ents]
        props[f"{kind_label}/{m}"] = num(have, "meV", f"{m}: {kind_label}")
        props[f"err/{m}"] = num([None if v is None else v - e["ref"] for v, e in zip(have, ents)], "meV",
                                f"{m}: {kind_label} minus reference (signed)")
        props[f"abserr/{m}"] = num([None if v is None else abs(v - e["ref"]) for v, e in zip(have, ents)],
                                   "meV", f"{m}: absolute {kind_label} error")

    # ---- atomic properties: one value per atom of every kept frame, in frame order
    def atomwise(getter, unit, desc):
        vals = []
        for r in keep:
            n = len(r["species"])
            v = getter(r)
            if v is None:
                vals += [float("nan")] * n
            else:
                v = list(v) + [float("nan")] * max(0, n - len(v))
                vals += [float(x) if isinstance(x, (int, float)) else float("nan") for x in v[:n]]
        return {"target": "atom", "values": vals, "units": unit, "description": desc}

    # chemiscope plots environments when a dataset has them, and then only atom properties are valid map axes.
    # So the global and the atomic views get their own dataset over the same structures, rather than one dataset
    # that can only answer half the questions.
    atoms = {
        "fragment": atomwise(lambda r: r.get("fragment_id"), "", "which fragment the atom belongs to"),
        "q_formal": atomwise(lambda r: r.get("formal_charges"), "e", "formal charge assigned to the atom"),
        "q_lowdin": atomwise(lambda r: (r.get("refs") or {}).get("charges_lowdin"), "e",
                             "DFT Löwdin partial charge — the reference for any model's charges"),
    }
    def has_data(v):
        if v["target"] == "atom" or all(isinstance(x, (int, float)) for x in v["values"]):
            vals = [x for x in v["values"] if isinstance(x, (int, float))]
            return bool(vals) and not all(np.isnan(vals))
        return True

    atoms = {k: v for k, v in atoms.items() if has_data(v)}
    props = {k: v for k, v in props.items() if has_data(v)}
    return {"frames": frames, "properties": props, "atom_properties": atoms, "environments": env,
            "n": len(keep), "models": models}


# ------------------------------------------------------------------------------------------------- the page
HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Long-range explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chemiscope@{ver}/dist/chemiscope.min.js"></script>
<style>
:root {{
  --bg:#fcfcfb; --fg:#15171a; --mut:#6b6d70; --faint:#9a9c9f; --rule:#e4e3e0; --card:#ffffff;
  --blue:#0072B2; --verm:#D55E00; --green:#009E73; --accent:var(--blue);
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --serif:"Crimson Pro",Crimson,Charter,Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#121417; --fg:#ececea; --mut:#a2a4a6; --faint:#76787b; --rule:#282b30; --card:#191c20; }} }}
:root[data-theme="dark"] {{ --bg:#121417; --fg:#ececea; --mut:#a2a4a6; --faint:#76787b; --rule:#282b30;
  --card:#191c20; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ background:var(--bg); color:var(--fg); margin:0; font-family:var(--serif);
  font-size:18px; line-height:1.62; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1220px; margin:0 auto; padding:0 28px; }}
nav.top {{ position:sticky; top:0; z-index:60; background:color-mix(in srgb, var(--bg) 90%, transparent);
  backdrop-filter:blur(12px); border-bottom:1px solid var(--rule); }}
nav.top .wrap {{ display:flex; align-items:center; gap:20px; height:52px; overflow-x:auto; }}
nav.top b {{ font-family:var(--sans); font-size:14.5px; font-weight:600; white-space:nowrap; }}
nav.top a {{ font-family:var(--sans); font-size:13px; color:var(--mut); text-decoration:none; white-space:nowrap; }}
nav.top a:hover {{ color:var(--fg); }}
header {{ padding:72px 0 8px; }}
h1 {{ font-size:50px; font-weight:700; margin:0 0 16px; letter-spacing:-0.025em; line-height:1.05; }}
h2 {{ font-size:26px; font-weight:600; margin:0 0 4px; letter-spacing:-0.012em; }}
h2 .num {{ font-family:var(--mono); font-size:14px; font-weight:500; color:var(--faint); margin-right:10px;
  letter-spacing:0; }}
.lede {{ font-size:21.5px; max-width:66ch; }}
p {{ max-width:74ch; }}
.sub {{ color:var(--mut); font-size:16px; max-width:78ch; }}
.eq {{ margin:18px 0 14px; padding:14px 18px; background:var(--card); border:1px solid var(--rule);
  border-radius:8px; overflow-x:auto; }}
section {{ padding:46px 0 12px; border-top:1px solid var(--rule); margin-top:38px; }}
.scorecard {{ display:flex; gap:30px; flex-wrap:wrap; margin:30px 0 6px; font-family:var(--sans); }}
.scorecard div {{ border-left:3px solid var(--rule); padding-left:13px; }}
.scorecard b {{ display:block; font-size:12.5px; font-weight:600; color:var(--mut); text-transform:uppercase;
  letter-spacing:0.07em; }}
.scorecard span {{ font-size:20px; }}
.ok {{ color:var(--blue); font-weight:600; }} .bad {{ color:var(--verm); font-weight:600; }}
table.rank {{ border-collapse:collapse; width:100%; font-family:var(--sans); font-size:13.5px; margin-top:18px; }}
table.rank th {{ text-align:right; padding:9px 9px; border-bottom:1.5px solid var(--rule); cursor:pointer;
  font-weight:600; color:var(--mut); white-space:nowrap; user-select:none; font-size:12.5px;
  text-transform:uppercase; letter-spacing:0.04em; }}
table.rank th:first-child, table.rank td:first-child {{ text-align:left; }}
table.rank th.sorted {{ color:var(--fg); }}
table.rank td {{ text-align:right; padding:8px 9px; border-bottom:1px solid var(--rule);
  font-variant-numeric:tabular-nums; }}
table.rank td.name {{ font-weight:500; }}
table.rank tr.ceiling {{ background:color-mix(in srgb, var(--blue) 7%, transparent); }}
table.rank td.lrs {{ font-family:var(--mono); font-weight:500; font-size:14.5px; }}
table.rank th.vis, table.rank td.vis {{ text-align:center; padding-left:4px; padding-right:4px; width:44px; }}
table.rank tr.off td {{ opacity:0.34; }}
table.rank tr.off td.vis {{ opacity:1; }}
button.eye {{ width:17px; height:17px; padding:0; border-radius:50%; cursor:pointer;
  border:1.5px solid var(--fam); background:var(--fam); transition:background 120ms, box-shadow 120ms; }}
button.eye[aria-pressed="false"] {{ background:transparent; }}
button.eye:hover {{ box-shadow:0 0 0 3px color-mix(in srgb, var(--fam) 28%, transparent); }}
.bar {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:16px 0 10px;
  font-family:var(--sans); font-size:12.5px; }}
.bar .lbl {{ color:var(--mut); font-weight:600; text-transform:uppercase; letter-spacing:0.05em;
  font-size:11px; margin-right:1px; }}
.bar button {{ font-family:var(--sans); font-size:12.5px; padding:4px 10px; border-radius:999px;
  border:1px solid var(--rule); background:var(--card); color:var(--fg); cursor:pointer; line-height:1.5; }}
.bar button:hover {{ border-color:var(--accent); }}
.bar .sep {{ width:1px; height:18px; background:var(--rule); margin:0 5px; }}
.panes {{ display:flex; flex-wrap:wrap; gap:12px; }}
.left {{ flex:1 1 590px; min-width:320px; }} .right {{ flex:1 1 400px; min-width:290px; }}
.cs-map {{ height:490px; }} .cs-structure {{ height:380px; }}
code {{ font-family:var(--mono); font-size:13.5px;
  background:color-mix(in srgb, var(--rule) 55%, transparent); padding:1px 5px; border-radius:4px; }}
.err {{ color:var(--verm); font-family:var(--sans); }}
.fam {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px;
  vertical-align:-1px; }}
.eyebrow {{ font-family:var(--sans); font-size:11.5px; font-weight:600; letter-spacing:0.1em;
  text-transform:uppercase; color:var(--mut); margin:0 0 8px; }}
h2 .tid, .tid.sm {{ display:inline-block; font-family:var(--mono); font-size:13px; font-weight:500; color:#fff;
  background:var(--fg); border-radius:5px; padding:2px 8px; margin-right:11px; vertical-align:2px;
  letter-spacing:0.02em; }}
.tid.sm {{ font-size:11.5px; padding:1px 7px; margin:0; }}
h2 .gid {{ display:inline-block; font-family:var(--mono); font-size:17px; font-weight:500; color:#fff;
  background:var(--fg); border-radius:7px; padding:1px 13px; margin-right:13px; vertical-align:2px; }}
.grouphead {{ border-top:2px solid var(--fg); }}
table.idx td {{ padding:7px 10px; }}
table.idx td:nth-child(2), table.idx td:nth-child(3) {{ text-align:left; }}
.bar button i.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px;
  vertical-align:0; flex:none; }}
.bar button i.sw {{ width:26px; height:9px; border-radius:2px; display:inline-block; margin-right:7px;
  vertical-align:0; flex:none; border:1px solid color-mix(in srgb, var(--fg) 18%, transparent); }}
.bar button {{ display:inline-flex; align-items:center; }}
.bar button[aria-pressed="true"] {{ background:var(--on, var(--accent)); border-color:var(--on, var(--accent));
  color:#fff; font-weight:500; }}
.bar button[aria-pressed="true"] i.dot {{ box-shadow:0 0 0 1.5px #fff; }}
.chips {{ display:flex; flex-wrap:wrap; gap:7px 18px; margin:-4px 0 14px; font-family:var(--sans);
  font-size:12.5px; color:var(--mut); }}
.chips span {{ display:inline-flex; align-items:center; gap:7px; }}
.chips i {{ width:11px; height:11px; border-radius:3px; display:inline-block; flex:none; }}
.keywrap {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin:18px 0 6px; font-family:var(--sans);
  font-size:13px; }}
.keyfam {{ border-left:3px solid var(--rule); padding-left:12px; min-width:168px; }}
.keyfam b {{ display:block; font-size:11.5px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase;
  color:var(--mut); margin-bottom:4px; }}
.keyfam div {{ display:flex; align-items:center; gap:8px; line-height:1.9; }}
.keyfam svg {{ flex:none; }}{scan_css}
footer {{ margin:64px 0 72px; color:var(--mut); font-size:15px; }}
@media (max-width:760px) {{ body {{ font-size:17px; }} h1 {{ font-size:33px; }} .wrap {{ padding:0 17px; }}
  .cs-map {{ height:400px; }} .lede {{ font-size:19px; }} }}
</style></head><body>
<nav class="top"><div class="wrap"><b>Long-range explorer</b>
<a href="#scans">1 Protocol</a><a href="#score">2 Results</a><a href="#key">Models</a>
<a href="#v-decay">A1</a><a href="#v-residual">A2</a><a href="#v-parity">A3</a><a href="#v-response">B1</a>
<a href="#v-charge">C1</a><a href="#v-threebody">D1</a><a href="#notes">Method</a></div></nav>
<div class="wrap">
<header>
<p class="eyebrow">LongRangeBench · diagnostic report</p>
<h1>Evaluating Long-Range Physics in<br>Machine-Learned Interatomic Potentials</h1>
<p class="lede">A diagnostic benchmark for the behaviour of interatomic potentials <em>beyond</em> their cutoff.
Near equilibrium a model can reach an excellent force MAE while reproducing none of the long-range physics; an
RMSE on a near-equilibrium test set cannot distinguish the two. Each task below is constructed so that the correct
value is known <em>a priori</em>, which makes a single number interpretable without a baseline for comparison.</p>
<div class="scorecard">
  <div><b>A · asymptotics</b><span>tail fraction <span class="ok">1</span> / <span class="bad">0</span>,
    ε <span class="ok">0</span> / <span class="bad">−1</span></span></div>
  <div><b>B · response</b><span>ΔΔE slope <span class="ok">1</span> / <span class="bad">0</span></span></div>
  <div><b>C · charges</b><span>Σq = Q, parity against Löwdin</span></div>
  <div><b>D · many-body</b><span>three-body E₃ beyond pairs</span></div>
</div>
<p class="sub">{summary}</p>
</header>
{body}
<footer>lrbench · chemiscope {ver} · equations set with KaTeX</footer>
</div>
<script>
const DATA = {data};
const VIEWS = {views};
const RANK = {rank};
{scan_js}
/* chemiscope's palettes, as it reports them itself: a categorical property gets a categorical palette, a signed
   quantity a diverging one centred on zero, a magnitude a perceptually uniform sequential ramp. */
const PALETTES = {{
  categorical: ["tab10", "tab20", "tab20b", "tab20c"],
  diverging: ["bwr", "seismic", "rwg", "brg"],
  sequential: ["cividis", "viridis", "magma", "plasma", "inferno"],
}};
const isCategorical = (ds, p) => !!ds.properties[p] && ds.properties[p].values.some(v => typeof v === "string");
function paletteKind(ds, prop) {{
  if (isCategorical(ds, prop)) return "categorical";
  return (prop || "").startsWith("err/") ? "diverging" : "sequential";
}}

async function inflate(b64) {{
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(s).text());
}}
/* Which models are in play. The ranking table and the widgets must agree about this, so one set drives both:
   hiding a model removes it from every selector below, and moves any axis that was showing it. The points on a
   map are structures, not models, so hiding one changes what can be *plotted*, never which structures exist. */
const HIDDEN = new Set();
const STATES = [];
const visible = ms => ms.filter(m => !HIDDEN.has(m));
function onVisibilityChange() {{
  for (const st of STATES) {{
    const s = st.viz.saveSettings().map;
    const live = visible(st.models);
    const owner = p => {{ const i = (p || "").indexOf("/"); return i < 0 ? null : p.slice(i + 1); }};
    const fix = axis => {{
      const m = owner(s[axis].property);
      if (!m || !HIDDEN.has(m) || !live.length) return null;
      const kind = s[axis].property.slice(0, s[axis].property.indexOf("/"));
      const next = kind + "/" + live[0];
      return st.ds.properties[next] ? next : null;
    }};
    const patch = {{}};
    for (const axis of ["x", "y"]) {{
      const next = fix(axis);
      if (next) patch[axis] = {{ property: next }};
    }}
    const cm = owner(s.color.property);
    if (cm && HIDDEN.has(cm) && live.length) {{
      const kind = s.color.property.slice(0, s.color.property.indexOf("/"));
      if (st.ds.properties[kind + "/" + live[0]]) patch.color = {{ property: kind + "/" + live[0] }};
    }}
    if (Object.keys(patch).length) st.viz.applySettings({{ map: patch }});
    redraw(document.getElementById(st.id + "-bar"));
    setDiagonal(st);
  }}
  if (window.drawRanking) window.drawRanking();
}}
const cache = {{}};
async function dataset(key) {{
  if (!cache[key]) cache[key] = await inflate(DATA[key]);
  const ds = structuredClone(cache[key]);
  for (const p of Object.values(ds.properties))
    if (p.values.every(v => v === null || typeof v === "number"))
      p.values = p.values.map(v => v === null ? NaN : v);
  return ds;
}}
const finite = a => a.filter(v => typeof v === "number" && isFinite(v));

/* a true y = x in data coordinates, re-applied whenever a setting changes */
function diagonalShape(viz, ds) {{
  const s = viz.saveSettings().map;
  const xs = ds.properties[s.x.property], ys = ds.properties[s.y.property];
  if (!xs || !ys) return null;
  const xv = finite(xs.values), yv = finite(ys.values);
  if (!xv.length || !yv.length) return null;
  const lo = Math.max(Math.min(...xv), Math.min(...yv));
  const hi = Math.min(Math.max(...xv), Math.max(...yv));
  if (!(hi > lo)) return null;
  return {{ type: "line", xref: "x", yref: "y", x0: lo, y0: lo, x1: hi, y1: hi, layer: "below",
            line: {{ color: "#8a8a8a", width: 1.4, dash: "dot" }} }};
}}
function setDiagonal(st) {{
  try {{ st.viz.map._relayout({{ shapes: st.diag ? [diagonalShape(st.viz, st.ds)].filter(Boolean) : [] }}); }}
  catch (e) {{ console.warn("diagonal:", e); }}
}}
/* a diverging scale is only honest when it is centred on zero */
function centreColour(st) {{
  try {{
    const s = st.viz.saveSettings().map;
    const p = st.ds.properties[s.color.property];
    if (!p) return;
    const v = finite(p.values);
    if (!v.length) return;
    const m = Math.max(Math.abs(Math.min(...v)), Math.abs(Math.max(...v)));
    if (m > 0) st.viz.applySettings({{ map: {{ color: {{ min: -m, max: m }} }} }});
  }} catch (e) {{ console.warn("centre:", e); }}
}}

/* A control wears the sign of what it controls: a model button carries its family's hue, a colour button carries
   the ramp it will apply. Buttons that change a *mode* rather than an identity — log/linear, the identity line —
   stay neutral, because they refer to nothing that has a colour. */
const MODELCOL = {modelcol};
const CHEMSTOPS = {chemstops};
function rampOf(ds, prop) {{
  const kind = paletteKind(ds, prop);
  const stops = CHEMSTOPS[kind === "categorical" ? "identity" : kind === "diverging" ? "signed" : "magnitude"];
  return kind === "categorical"
    ? `linear-gradient(90deg, ${{stops.map((c, i) =>
        `${{c}} ${{(i / stops.length) * 100}}%, ${{c}} ${{((i + 1) / stops.length) * 100}}%`).join(", ")}})`
    : `linear-gradient(90deg, ${{stops.join(", ")}})`;
}}
function button(label, pressed, onclick, opts) {{
  const b = document.createElement("button");
  const o = opts || {{}};
  if (o.swatch) {{
    const sw = document.createElement("i");
    sw.className = "sw";
    sw.style.background = o.swatch;
    b.appendChild(sw);
  }} else if (o.dot) {{
    const d = document.createElement("i");
    d.className = "dot";
    d.style.background = o.dot;
    b.appendChild(d);
  }}
  b.appendChild(document.createTextNode(label));
  if (pressed && o.dot) b.style.setProperty("--on", o.dot);
  b.setAttribute("aria-pressed", pressed ? "true" : "false");
  b.onclick = onclick;
  return b;
}}
function group(bar, label, items, isOn, choose) {{
  if (!items.length) return;
  if (bar.childElementCount) bar.appendChild(Object.assign(document.createElement("span"), {{ className: "sep" }}));
  if (label) {{
    const l = document.createElement("span");
    l.className = "lbl"; l.textContent = label;
    bar.appendChild(l);
  }}
  items.forEach(it => bar.appendChild(button(it.label, isOn(it), () => {{ choose(it); redraw(bar); }},
                                             {{ swatch: it.swatch, dot: it.dot }})));
}}
const DRAW = new WeakMap();
function redraw(bar) {{ const f = DRAW.get(bar); if (f) f(); }}

function buildBar(st, v) {{
  const bar = document.getElementById(st.id + "-bar");
  const prefix = p => {{ const i = p.indexOf("/"); return i < 0 ? null : p.slice(0, i); }};
  const suffix = p => {{ const i = p.indexOf("/"); return i < 0 ? null : p.slice(i + 1); }};
  const draw = () => {{
    bar.innerHTML = "";
    const s = st.viz.saveSettings().map;
    const props = Object.keys(st.ds.properties);
    const models = visible(st.models || []);
    if (v.parity && models.length) {{
      const px = prefix(s.x.property) || "E_int", py = prefix(s.y.property) || "E_int";
      group(bar, "x model", models.map(m => ({{ label: m, m, dot: MODELCOL[m] }})),
            it => suffix(s.x.property) === it.m,
            it => {{ const p = px + "/" + it.m;
                     if (st.ds.properties[p]) st.viz.applySettings({{ map: {{ x: {{ property: p }} }} }});
                     setDiagonal(st); }});
      group(bar, "y model", models.map(m => ({{ label: m, m, dot: MODELCOL[m] }})),
            it => suffix(s.y.property) === it.m,
            it => {{ const p = py + "/" + it.m;
                     if (st.ds.properties[p]) st.viz.applySettings({{ map: {{ y: {{ property: p }} }} }});
                     setDiagonal(st); }});
      const kinds = [{{ label: "E_int", k: "E_int" }}, {{ label: "signed error", k: "err" }},
                     {{ label: "|error|", k: "abserr" }}].filter(
        it => props.includes(it.k + "/" + (suffix(s.y.property) || models[0])));
      group(bar, "quantity", kinds, it => py === it.k, it => {{
        const mx = suffix(s.x.property) || models[0], my = suffix(s.y.property) || models[0];
        const nx = it.k + "/" + mx, ny = it.k + "/" + my;
        const next = {{}};
        if (st.ds.properties[nx]) next.x = {{ property: nx }};
        if (st.ds.properties[ny]) next.y = {{ property: ny }};
        st.viz.applySettings({{ map: next }});
        setDiagonal(st);
      }});
    }} else if (models.length) {{
      const cur = suffix(s.y.property) || suffix(s.x.property);
      group(bar, "model", models.map(m => ({{ label: m, m, dot: MODELCOL[m] }})), it => it.m === cur,
            it => {{
        const k = prefix(s.y.property) || "err";
        const p = k + "/" + it.m;
        if (st.ds.properties[p]) {{
          const next = {{ y: {{ property: p }} }};
          if (prefix(s.color.property) === k) next.color = {{ property: p }};
          st.viz.applySettings({{ map: next }});
          if (v.center0) centreColour(st);
        }}
        setDiagonal(st);
      }});
    }}
    const shown = p => {{ const i = p.indexOf("/"); return i < 0 || !HIDDEN.has(p.slice(i + 1)); }};
    const colours = ["separation", "class", "total charge", "atoms", "fragment", "q_lowdin"]
      .filter(c => props.includes(c))
      .concat(props.filter(p => (p.startsWith("abserr/") || p.startsWith("err/")) && shown(p)).slice(0, 2));
    /* The palette is a consequence of what the property *is*, so it is applied, not offered: the swatch on each
       button is the ramp that choosing it will bring. */
    group(bar, "colour", colours.map(c => ({{ label: c, c, swatch: rampOf(st.ds, c) }})),
          it => it.c === s.color.property, it => {{
      const kind = paletteKind(st.ds, it.c);
      st.viz.applySettings({{ map: {{ color: {{ property: it.c, palette: PALETTES[kind][0] }} }} }});
      if (kind === "diverging") centreColour(st);
    }});
    group(bar, "x", [{{ label: "log", v: "log" }}, {{ label: "linear", v: "linear" }}],
          it => it.v === s.x.scale,
          it => {{ st.viz.applySettings({{ map: {{ x: {{ scale: it.v }} }} }}); setDiagonal(st); }});
    group(bar, "", [{{ label: "y = x", v: 1 }}], () => st.diag,
          () => {{ st.diag = !st.diag; setDiagonal(st); }});
  }};
  DRAW.set(bar, draw);
  draw();
}}

/* ---- the ranking: the score beside the bars it is made of ---- */
function scoreBars(r) {{
  const w = 92, h = 15, n = RANK.components.length, gap = 3;
  const bw = (w - gap * (n - 1)) / n;
  let svg = `<svg width="${{w}}" height="${{h}}" viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="components">`;
  RANK.components.forEach((c, i) => {{
    const v = r[c.k];
    const x = i * (bw + gap);
    svg += `<rect x="${{x}}" y="0" width="${{bw}}" height="${{h}}" fill="currentColor" opacity="0.1"/>`;
    if (typeof v === "number")
      svg += `<rect x="${{x}}" y="${{h * (1 - v)}}" width="${{bw}}" height="${{h * v}}" fill="${{c.col}}"><title>` +
             `${{c.t}}: ${{v.toFixed(2)}}</title></rect>`;
  }});
  return svg + "</svg>";
}}
(async () => {{
  const host = document.getElementById("rank-body");
  if (host && RANK.rows.length) {{
    let sortKey = RANK.sort, asc = false;
    const draw = () => {{
      window.drawRanking = draw;
      const rows = RANK.rows.slice().sort((a, b) => {{
        if (a.ceiling !== b.ceiling) return a.ceiling ? -1 : 1;
        const x = a[sortKey], y = b[sortKey];
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (asc ? 1 : -1) * (typeof x === "string" ? x.localeCompare(y) : x - y);
      }});
      host.innerHTML = "<thead><tr><th class='vis' title='show or hide this model in every widget below'>" +
        "show</th>" + RANK.cols.map(c =>
        `<th data-k="${{c.k}}" class="${{c.k === sortKey ? "sorted" : ""}}" title="${{c.h || ""}}">${{c.t}}</th>`)
        .join("") + "</tr></thead><tbody>" + rows.map(r =>
          `<tr class="${{r.ceiling ? "ceiling" : ""}}${{HIDDEN.has(r.key) ? " off" : ""}}">` +
          `<td class="vis"><button class="eye" data-m="${{r.key}}" aria-pressed="${{!HIDDEN.has(r.key)}}" ` +
          `title="${{HIDDEN.has(r.key) ? "show" : "hide"}} ${{r.key}}" ` +
          `style="--fam:${{r.fam || "#999"}}"></button></td>` +
          RANK.cols.map(c => {{
            if (c.k === "bars") return `<td>${{scoreBars(r)}}</td>`;
            const v = r[c.k];
            const cls = c.k === "model" ? "name" : (c.k === "lrs" ? "lrs" : "");
            if (c.k === "model")
              return `<td class="name" title="${{r.family || ""}}">` +
                     `<span class="fam" style="background:${{r.fam || "#999"}}"></span>${{v}}</td>`;
            return `<td class="${{cls}}">${{v === null || v === undefined ? "·" : v}}</td>`;
          }}).join("") + "</tr>").join("") + "</tbody>";
      host.querySelectorAll("th").forEach(th => th.onclick = () => {{
        const k = th.dataset.k;
        if (!k || k === "bars") return;
        if (k === sortKey) asc = !asc; else {{ sortKey = k; asc = k === "model"; }}
        draw();
      }});
      host.querySelectorAll("button.eye").forEach(b => b.onclick = () => {{
        const m = b.dataset.m;
        if (HIDDEN.has(m)) HIDDEN.delete(m); else HIDDEN.add(m);
        onVisibilityChange();
      }});
    }};
    draw();
  }}

  for (const v of VIEWS) {{
    try {{
      const ds = await dataset(v.key);
      if (v.settings) ds.settings = v.settings;
      const viz = await Chemiscope.DefaultVisualizer.load(
        {{ map: v.id + "-map", info: v.id + "-info", meta: v.id + "-meta", structure: v.id + "-structure" }}, ds);
      const st = {{ id: v.id, viz, ds, models: v.models || [], diag: !!v.diagonal }};
      STATES.push(st);
      buildBar(st, v);
      viz.onSettingChange(() => redraw(document.getElementById(v.id + "-bar")));
      setTimeout(() => {{ if (v.center0) centreColour(st); setDiagonal(st); }}, 450);
    }} catch (e) {{
      document.getElementById(v.id + "-view").innerHTML =
        '<p class="err">chemiscope could not load this view: ' + e + '</p>';
      console.error(e);
    }}
  }}
  initScans();
  if (window.renderMathInElement)
    renderMathInElement(document.body, {{ delimiters: [
      {{ left: "$$", right: "$$", display: true }}, {{ left: "\\\\[", right: "\\\\]", display: true }},
      {{ left: "\\\\(", right: "\\\\)", display: false }}], throwOnError: false }});
}})();
window.addEventListener("load", () => {{
  if (window.renderMathInElement)
    renderMathInElement(document.body, {{ delimiters: [
      {{ left: "$$", right: "$$", display: true }}, {{ left: "\\\\[", right: "\\\\]", display: true }},
      {{ left: "\\\\(", right: "\\\\)", display: false }}], throwOnError: false }});
}});
</script></body></html>
"""

SECTION = """<section id="{id}">
<p class="eyebrow" style="color:{gcol}">{group}</p>
<h2><span class="tid" style="background:{gcol}">{tid}</span>{title}</h2>
<div class="eq">$${equation}$$</div>
<div class="chips">{chips}</div>
<p>{prose}</p>
<div class="bar" id="{id}-bar"></div>
<div class="view" id="{id}-view">
  <div class="panes">
    <div class="left"><div id="{id}-meta"></div><div class="cs-map" id="{id}-map"></div></div>
    <div class="right"><div class="cs-structure" id="{id}-structure"></div><div id="{id}-info"></div></div>
  </div>
</div>
</section>
"""

PROTOCOL_A = """<div class="scanwrap">
  <div class="scanhead"><b>A · the dimer scan</b> — two fragments pulled apart along one axis. Everything the
  tail diagnostics measure happens in this picture.</div>
  <div class="ctl">
    <label>separation <i>d</i> <input type="range" id="A-d" min="2" max="25" step="0.25" value="9"><output
      id="A-d-o"></output> Å</label>
    <label>model cutoff <i>r<sub>c</sub></i> <input type="range" id="A-rc" min="3" max="26" step="0.5" value="6">
      <output id="A-rc-o"></output> Å</label>
    <label>gate <i>g</i> <input type="range" id="A-g" min="0" max="1.2" step="0.05" value="1"><output
      id="A-g-o"></output></label>
  </div>
  <div class="scanrow"><svg id="A-geo" viewBox="0 0 420 150"></svg><svg id="A-plot" viewBox="0 0 420 220"></svg></div>
  <div class="readout" id="A-read"></div>
</div>
"""

PROTOCOL_B = """<div class="scanwrap">
  <div class="scanhead"><b>B · the fixed-charge quartet</b> — a local pair of sites, and one charge far away.
  Four energies, differenced twice, so that everything except the long-range reply cancels.</div>
  <div class="ctl">
    <label>distance of the fixed charge <i>R</i> <input type="range" id="B-R" min="4" max="30" step="0.5"
      value="12"><output id="B-R-o"></output> Å</label>
    <label>model cutoff <i>r<sub>c</sub></i> <input type="range" id="B-rc" min="3" max="30" step="0.5" value="6">
      <output id="B-rc-o"></output> Å</label>
  </div>
  <div class="scanrow"><svg id="B-geo" viewBox="0 0 420 200"></svg><svg id="B-plot" viewBox="0 0 420 220"></svg></div>
  <div class="readout" id="B-read"></div>
</div>
"""

PROTOCOL_C = """<div class="scanwrap">
  <div class="scanhead"><b>C · the pathologies</b> — points the harness reports but refuses to score, and the one
  case where a metric stops being able to tell a model from a flat line.</div>
  <div class="bar" id="C-bar"></div>
  <div class="scanrow"><svg id="C-geo" viewBox="0 0 420 180"></svg><svg id="C-plot" viewBox="0 0 420 220"></svg></div>
  <div class="readout" id="C-read"></div>
</div>
"""

SCAN_CSS = """
.scanwrap {{ border:1px solid var(--rule); border-radius:10px; padding:16px 18px 10px; margin:20px 0;
  background:var(--card); }}
.scanhead {{ font-size:16.5px; margin-bottom:6px; }}
.scanhead b {{ font-family:var(--sans); font-size:14px; font-weight:600; }}
.ctl {{ display:flex; flex-wrap:wrap; gap:18px; margin:10px 0 6px; font-family:var(--sans); font-size:12.5px;
  color:var(--mut); align-items:center; }}
.ctl label {{ display:flex; align-items:center; gap:7px; }}
.ctl i {{ font-family:var(--serif); font-style:italic; font-size:15px; color:var(--fg); }}
.ctl input[type=range] {{ width:132px; accent-color:var(--blue); }}
.ctl output {{ font-family:var(--mono); font-size:12.5px; color:var(--fg); min-width:34px; }}
.scanrow {{ display:flex; flex-wrap:wrap; gap:10px; align-items:flex-start; }}
.scanrow svg {{ flex:1 1 330px; min-width:280px; height:auto; }}
.readout {{ font-family:var(--sans); font-size:13px; color:var(--mut); margin:6px 0 4px; min-height:20px; }}
.readout b {{ color:var(--fg); font-family:var(--mono); font-weight:500; }}
"""

SCAN_JS = """
/* ============================ the canonical scans =============================================================
   No data: these are the definitions themselves, drawn. Each one is the arithmetic the harness performs, with the
   model replaced by a toy whose failure you control, so a reader can see what a number of 0 or -1 is describing.
   ============================================================================================================ */
const KE = 14.399645;                       /* e^2/(4 pi eps0) in eV A */
const SEQ = {seq};                          /* crameri:batlow  — magnitude */
const DIV = {div};                          /* crameri:vik     — signed    */
const FAMCOL = {fam};
function lerp(stops, t) {{
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  const a = stops[i].slice(1), b = stops[i + 1].slice(1);
  const ch = j => Math.round(parseInt(a.substr(j, 2), 16) * (1 - f) + parseInt(b.substr(j, 2), 16) * f);
  return `rgb(${{ch(0)}},${{ch(2)}},${{ch(4)}})`;
}}
const el = id => document.getElementById(id);
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
function svg(id, body) {{ const n = el(id); if (n) n.innerHTML = body; }}
const TXT = (x, y, s, o = {{}}) =>
  `<text x="${{x}}" y="${{y}}" font-size="${{o.size || 11}}" fill="${{o.fill || "var(--mut)"}}" ` +
  `text-anchor="${{o.anchor || "start"}}" font-family="${{o.mono ? "var(--mono)" : "var(--sans)"}}"` +
  `${{o.italic ? ' font-style="italic"' : ""}}>${{esc(s)}}</text>`;

/* ---- A: the dimer scan ---- */
function toyA(r, g, rc) {{ return g * 0.5 * (1 - Math.tanh((r - rc) / 1.0)) * (KE / r); }}
function drawA() {{
  const d = +el("A-d").value, rc = +el("A-rc").value, g = +el("A-g").value;
  el("A-d-o").value = d.toFixed(2); el("A-rc-o").value = rc.toFixed(1); el("A-g-o").value = g.toFixed(2);
  const x0 = 60, scale = 13, y = 74;
  const xb = x0 + d * scale;
  let s = `<rect x="0" y="0" width="420" height="150" fill="none"/>`;
  s += `<line x1="${{x0}}" y1="${{y}}" x2="${{Math.min(xb, 404)}}" y2="${{y}}" stroke="var(--rule)" ` +
       `stroke-width="1.2" stroke-dasharray="3 3"/>`;
  s += `<circle cx="${{x0}}" cy="${{y}}" r="17" fill="${{lerp(DIV, 0.08)}}" stroke="var(--fg)" stroke-width="1"/>`;
  s += TXT(x0, y + 4, "−", {{ anchor: "middle", size: 16, fill: "#fff" }});
  if (xb < 406) {{
    s += `<circle cx="${{xb}}" cy="${{y}}" r="17" fill="${{lerp(DIV, 0.95)}}" stroke="var(--fg)" ` +
         `stroke-width="1"/>` + TXT(xb, y + 4, "+", {{ anchor: "middle", size: 16, fill: "#fff" }});
  }}
  s += TXT((x0 + Math.min(xb, 404)) / 2, y - 24, `d = ${{d.toFixed(2)}} Å`, {{ anchor: "middle", size: 11.5 }});
  const inside = d <= rc;
  s += `<line x1="${{x0}}" y1="${{y + 40}}" x2="${{x0 + rc * scale}}" y2="${{y + 40}}" ` +
       `stroke="${{lerp(SEQ, 0.45)}}" stroke-width="3"/>`;
  s += TXT(x0, y + 58, `the model sees out to r_c = ${{rc.toFixed(1)}} Å`, {{ size: 10.5 }});
  s += TXT(x0, y + 72, inside ? "this pair is inside the cutoff" : "this pair is beyond it — only a long-range " +
           "term can reach", {{ size: 10.5, fill: inside ? "var(--mut)" : lerp(DIV, 0.97) }});
  svg("A-geo", s);

  const rs = [], W = 420, H = 220, L = 52, Rr = 12, T = 14, B = 34;
  for (let r = 4; r <= 26; r += 0.25) rs.push(r);
  const ex = rs.map(r => 1e3 * KE / r), mo = rs.map(r => 1e3 * toyA(r, g, rc));
  const ymax = Math.max(...ex), sx = r => L + (r - 4) / 22 * (W - L - Rr),
        sy = v => T + (1 - v / ymax) * (H - T - B);
  let p = `<line x1="${{L}}" y1="${{sy(0)}}" x2="${{W - Rr}}" y2="${{sy(0)}}" stroke="var(--mut)" ` +
          `stroke-width="0.8"/><line x1="${{L}}" y1="${{T}}" x2="${{L}}" y2="${{sy(0)}}" ` +
          `stroke="var(--mut)" stroke-width="0.8"/>`;
  const band = rs.map((r, i) => `${{sx(r)}},${{sy(ex[i])}}`).join(" ") + " " +
               rs.slice().reverse().map((r, i) => `${{sx(r)}},${{sy(mo[rs.length - 1 - i])}}`).join(" ");
  p += `<polygon points="${{band}}" fill="${{lerp(DIV, 0.97)}}" opacity="0.22"/>`;
  p += `<polyline points="${{rs.map((r, i) => `${{sx(r)}},${{sy(ex[i])}}`).join(" ")}}" fill="none" ` +
       `stroke="var(--fg)" stroke-width="2"/>`;
  p += `<polyline points="${{rs.map((r, i) => `${{sx(r)}},${{sy(mo[i])}}`).join(" ")}}" fill="none" ` +
       `stroke="${{lerp(SEQ, 0.3)}}" stroke-width="2"/>`;
  p += `<line x1="${{sx(8)}}" y1="${{T}}" x2="${{sx(8)}}" y2="${{sy(0)}}" stroke="var(--mut)" ` +
       `stroke-width="0.8" stroke-dasharray="2 3"/>` + TXT(sx(8) + 3, sy(0) + 16, "8 Å", {{ size: 10 }});
  p += `<line x1="${{sx(d)}}" y1="${{T}}" x2="${{sx(d)}}" y2="${{sy(0)}}" stroke="${{lerp(DIV, 0.95)}}" ` +
       `stroke-width="1.2"/>`;
  p += TXT(L + 2, T - 4, "E (meV)", {{ size: 10 }});
  p += TXT(W - Rr, sy(0) + 16, "r (Å)", {{ anchor: "end", size: 10.5 }});
  const lx = W - Rr - 4;
  p += TXT(lx, T + 12, "reference  k qq / r", {{ size: 10.5, fill: "var(--fg)", anchor: "end" }});
  p += TXT(lx, T + 27, "model", {{ size: 10.5, fill: lerp(SEQ, 0.3), anchor: "end" }});
  p += TXT(lx, T + 42, "the missing tail", {{ size: 10.5, fill: lerp(DIV, 0.9), anchor: "end" }});
  svg("A-plot", p);

  const grid = [8, 9, 10, 12, 14, 18, 25];
  const dref = grid.map(r => 1e3 * (KE / r - KE / 25)), dmod = grid.map(r => 1e3 * (toyA(r, g, rc) - toyA(25, g, rc)));
  const num = dref.reduce((a, v, i) => a + v * dmod[i], 0), den = dref.reduce((a, v) => a + v * v, 0);
  const tail = den ? num / den : 0;
  const eps = dref.map((v, i) => (dmod[i] - v) / v).sort((a, b) => a - b)[Math.floor(grid.length / 2)];
  el("A-read").innerHTML = `tail fraction <b>${{tail.toFixed(3)}}</b> · ε <b>${{eps.toFixed(3)}}</b> — ` +
    (tail > 0.9 ? "the decay is reproduced." : tail > 0.05 ? "part of the tail survives the cutoff." :
     "beyond 8 Å this model is flat: both diagnostics are at the floor, and no amount of near-field accuracy " +
     "changes that.");
}}

/* ---- B: the fixed-charge quartet ---- */
function drawB() {{
  const R = +el("B-R").value, rc = +el("B-rc").value;
  el("B-R-o").value = R.toFixed(1); el("B-rc-o").value = rc.toFixed(1);
  const x0 = 46, scale = 11.5, y = 62, s1 = 0.6;
  let s = "";
  [0, 1].forEach(k => {{
    const yy = y + k * 72;
    const xs = x0 + (k ? s1 : 0) * scale * 6;
    s += TXT(10, yy + 4, k ? "s₁" : "s₀", {{ size: 12, fill: "var(--fg)" }});
    s += `<circle cx="${{xs}}" cy="${{yy}}" r="11" fill="${{lerp(SEQ, 0.25)}}" stroke="var(--fg)" ` +
         `stroke-width="0.9"/>`;
    s += `<circle cx="${{xs + 30}}" cy="${{yy}}" r="11" fill="${{lerp(SEQ, 0.55)}}" stroke="var(--fg)" ` +
         `stroke-width="0.9"/>`;
    const xq = Math.min(x0 + R * scale, 398);
    s += `<line x1="${{xs + 42}}" y1="${{yy}}" x2="${{xq - 14}}" y2="${{yy}}" stroke="var(--rule)" ` +
         `stroke-width="1" stroke-dasharray="3 3"/>`;
    s += `<circle cx="${{xq}}" cy="${{yy}}" r="13" fill="${{lerp(DIV, 0.95)}}" stroke="var(--fg)" ` +
         `stroke-width="1"/>` + TXT(xq, yy + 4, "+", {{ anchor: "middle", size: 14, fill: "#fff" }});
  }});
  s += TXT(x0, 170, `the two local arrangements differ by ${{(s1 * 6).toFixed(1)}} Å; the charge sits at ` +
           `R = ${{R.toFixed(1)}} Å`, {{ size: 10.5 }});
  s += TXT(x0, 186, "composition and total charge are identical in all four — every offset cancels",
           {{ size: 10.5 }});
  svg("B-geo", s);

  const W = 420, H = 220, L = 54, Rr = 12, T = 16, B = 34;
  const Rs = []; for (let r = 4; r <= 30; r += 0.25) Rs.push(r);
  const ex = Rs.map(r => 1e3 * KE * (1 / (r - s1) - 1 / r));
  const mo = Rs.map((r, i) => ex[i] * 0.5 * (1 - Math.tanh((r - rc) / 1.0)));
  const lo = 1e-3, hi = Math.max(...ex);
  const sx = r => L + (r - 4) / 26 * (W - L - Rr);
  const sy = v => T + (1 - (Math.log10(Math.max(v, lo)) - Math.log10(lo)) /
                       (Math.log10(hi) - Math.log10(lo))) * (H - T - B);
  let p = `<line x1="${{L}}" y1="${{H - B}}" x2="${{W - Rr}}" y2="${{H - B}}" stroke="var(--mut)" ` +
          `stroke-width="0.8"/><line x1="${{L}}" y1="${{T}}" x2="${{L}}" y2="${{H - B}}" stroke="var(--mut)" ` +
          `stroke-width="0.8"/>`;
  p += `<polyline points="${{Rs.map((r, i) => `${{sx(r)}},${{sy(ex[i])}}`).join(" ")}}" fill="none" ` +
       `stroke="var(--fg)" stroke-width="2"/>`;
  p += `<polyline points="${{Rs.map((r, i) => `${{sx(r)}},${{sy(Math.max(mo[i], lo))}}`).join(" ")}}" ` +
       `fill="none" stroke="${{lerp(SEQ, 0.3)}}" stroke-width="2"/>`;
  p += `<line x1="${{sx(R)}}" y1="${{T}}" x2="${{sx(R)}}" y2="${{H - B}}" stroke="${{lerp(DIV, 0.95)}}" ` +
       `stroke-width="1.2"/>`;
  p += TXT(L + 2, T - 4, "|ΔΔE| (meV)", {{ size: 10 }});
  p += TXT(W - Rr, H - B + 16, "R (Å)", {{ anchor: "end", size: 10.5 }});
  p += TXT(W - Rr - 4, T + 12, "reference ∝ R⁻²", {{ size: 10.5, fill: "var(--fg)", anchor: "end" }});
  p += TXT(W - Rr - 4, T + 27, "model", {{ size: 10.5, fill: lerp(SEQ, 0.3), anchor: "end" }});
  svg("B-plot", p);
  const i = Rs.findIndex(r => r >= R);
  const k = Rs.map((r, j) => r >= 8 ? j : -1).filter(j => j >= 0);
  const slope = k.reduce((a, j) => a + ex[j] * mo[j], 0) / k.reduce((a, j) => a + ex[j] * ex[j], 0);
  el("B-read").innerHTML = `at R = ${{R.toFixed(1)}} Å the reference says <b>${{ex[i].toFixed(2)}} meV</b>, ` +
    `the model <b>${{mo[i].toFixed(2)}} meV</b> · fitted slope beyond 8 Å <b>${{slope.toFixed(3)}}</b> — ` +
    (slope > 0.8 ? "the reply is there." : slope > 0.05 ? "the reply is damped." :
     "the model cannot see the charge at all; the quartet returns zero whatever its force error.");
}}

/* ---- C: the pathologies ---- */
const PATHS = [
  {{ k: "ct", t: "charge-transfer ambiguity",
     why: "Pulled apart, the reference puts a fraction of an electron on each fragment instead of settling on " +
          "integers. The interaction energy of 'two fragments' is then not a well-posed quantity, so these " +
          "points are flagged and reported but never scored — deleting them instead would hide the problem." }},
  {{ k: "unbound", t: "basis-bound anion",
     why: "The extra electron is not actually bound: it sits in a diffuse basis function, and its energy follows " +
          "the basis rather than the physics. The record is kept and flagged, because a model that reproduces " +
          "the number is reproducing an artefact." }},
  {{ k: "degenerate", t: "a metric that stops discriminating",
     why: "When the reference's own span across a scan is tiny, the offset-free error of a flat model is already " +
          "as small as the measurement: the metric can no longer tell a model from a line. The harness detects " +
          "this and reports the absolute error instead." }},
];
let pathSel = "ct";
function drawC() {{
  const bar = el("C-bar");
  bar.innerHTML = "";
  PATHS.forEach(p => {{
    const b = document.createElement("button");
    b.textContent = p.t;
    b.setAttribute("aria-pressed", p.k === pathSel ? "true" : "false");
    b.onclick = () => {{ pathSel = p.k; drawC(); }};
    bar.appendChild(b);
  }});
  const W = 420, H = 180, x0 = 56, scale = 12, y = 70;
  let s = "", p = "";
  const sel = PATHS.find(q => q.k === pathSel);
  if (pathSel === "ct") {{
    const d = 14, xb = x0 + d * scale;
    s += `<circle cx="${{x0}}" cy="${{y}}" r="19" fill="${{lerp(DIV, 0.2)}}" stroke="var(--fg)"/>`;
    s += TXT(x0, y + 5, "−0.62", {{ anchor: "middle", size: 10.5, fill: "#fff", mono: true }});
    s += `<circle cx="${{xb}}" cy="${{y}}" r="19" fill="${{lerp(DIV, 0.8)}}" stroke="var(--fg)"/>`;
    s += TXT(xb, y + 5, "−0.38", {{ anchor: "middle", size: 10.5, fill: "#fff", mono: true }});
    s += TXT((x0 + xb) / 2, y - 30, "total charge −1, split between them", {{ anchor: "middle", size: 11 }});
    s += TXT(x0 - 10, y + 48, "integers would be −1 and 0; the reference gives neither", {{ size: 10.5 }});
    const rs = []; for (let r = 4; r <= 24; r += 0.25) rs.push(r);
    const q = rs.map(r => -0.5 - 0.5 * Math.tanh((8 - r) / 3));
    const sx = r => 54 + (r - 4) / 20 * (W - 70), sy = v => 20 + (-v) * 150;
    p += `<line x1="54" y1="${{sy(-1)}}" x2="${{W - 16}}" y2="${{sy(-1)}}" stroke="var(--rule)" ` +
         `stroke-dasharray="3 3"/>` + TXT(W - 16, sy(-1) - 5, "−1", {{ anchor: "end", size: 10 }});
    p += `<line x1="54" y1="${{sy(-0.5)}}" x2="${{W - 16}}" y2="${{sy(-0.5)}}" stroke="var(--rule)"/>`;
    p += `<polyline points="${{rs.map((r, i) => `${{sx(r)}},${{sy(q[i])}}`).join(" ")}}" fill="none" ` +
         `stroke="${{lerp(DIV, 0.9)}}" stroke-width="2"/>`;
    p += TXT(56, 16, "charge on fragment A", {{ size: 10.5, fill: "var(--fg)" }});
    p += TXT(W - 16, 176, "separation →", {{ anchor: "end", size: 10.5 }});
  }} else if (pathSel === "unbound") {{
    s += `<circle cx="150" cy="${{y}}" r="22" fill="${{lerp(SEQ, 0.3)}}" stroke="var(--fg)"/>`;
    for (let i = 1; i <= 3; i++)
      s += `<circle cx="150" cy="${{y}}" r="${{22 + i * 22}}" fill="none" stroke="${{lerp(SEQ, 0.75)}}" ` +
           `stroke-width="1" stroke-dasharray="2 4" opacity="${{0.9 - i * 0.22}}"/>`;
    s += TXT(150, y + 86, "the extra electron spreads as far as the basis allows", {{ anchor: "middle",
             size: 10.5 }});
    const ns = [1, 2, 3, 4, 5];
    const e = [0.9, 0.55, 0.3, 0.14, 0.05];
    p += `<line x1="54" y1="190" x2="${{W - 16}}" y2="190" stroke="var(--mut)" stroke-width="0.8"/>`;
    ns.forEach((n, i) => {{
      const x = 70 + i * 66, h = e[i] * 150;
      p += `<rect x="${{x}}" y="${{190 - h}}" width="34" height="${{h}}" fill="${{lerp(SEQ, 0.2 + i * 0.16)}}"/>`;
      p += TXT(x + 17, 204, `+${{i}}`, {{ anchor: "middle", size: 10 }});
    }});
    p += TXT(56, 16, "binding energy against basis size — it never converges", {{ size: 10.5,
             fill: "var(--fg)" }});
  }} else {{
    const rs = []; for (let r = 8; r <= 25; r += 0.5) rs.push(r);
    const ref = rs.map(r => 1e3 * (0.002 / r - 0.002 / 25));
    const sx = r => 54 + (r - 8) / 17 * (W - 70), sy = v => 170 - v / Math.max(...ref) * 140;
    p += `<polyline points="${{rs.map((r, i) => `${{sx(r)}},${{sy(ref[i])}}`).join(" ")}}" fill="none" ` +
         `stroke="var(--fg)" stroke-width="2"/>`;
    p += `<line x1="54" y1="${{sy(0)}}" x2="${{W - 16}}" y2="${{sy(0)}}" stroke="${{lerp(SEQ, 0.3)}}" ` +
         `stroke-width="2"/>`;
    p += TXT(56, 16, "reference span across the scan", {{ size: 10.5, fill: "var(--fg)" }});
    p += TXT(56, 30, "a flat model", {{ size: 10.5, fill: lerp(SEQ, 0.3) }});
    p += TXT(W - 16, 196, "the two are within the measurement", {{ anchor: "end", size: 10.5 }});
    s += TXT(20, 40, "When the reference barely moves, 'error against a flat line'", {{ size: 11.5,
             fill: "var(--fg)" }});
    s += TXT(20, 60, "is the same size for every model, and the metric is empty.", {{ size: 11.5,
             fill: "var(--fg)" }});
    s += TXT(20, 92, "The harness checks this per class and falls back to the", {{ size: 11.5 }});
    s += TXT(20, 110, "absolute |E_int| error, which still discriminates.", {{ size: 11.5 }});
  }}
  svg("C-geo", s);
  svg("C-plot", p);
  el("C-read").innerHTML = esc(sel.why);
}}
function initScans() {{
  ["A-d", "A-rc", "A-g"].forEach(i => el(i) && el(i).addEventListener("input", drawA));
  ["B-R", "B-rc"].forEach(i => el(i) && el(i).addEventListener("input", drawB));
  if (el("A-d")) drawA();
  if (el("B-R")) drawB();
  if (el("C-bar")) drawC();
}}
"""


METHOD = """<section id="notes">
<p class="eyebrow">method</p>
<h2>Protocol and conventions</h2>
<p class="sub">No model was re-evaluated for these figures. Every point comes from the canonical records and the
per-term energies saved by <code>lrbench eval --save-predictions</code>, recombined through each group's own
definition — so a model's value and its reference are the same linear combination and a point compares like with
like. Records flagged as DFT pathologies are excluded. Energies in meV.</p>
<p class="sub"><b>Encoding.</b> A class is an identity, so it is carried by marker shape, never by a colour ramp:
chemiscope's palettes are continuous, and reading categories off a continuous bar is exactly what good practice
forbids. A signed error is carried by a diverging palette centred on zero, so white means no error and the two
hues are the two directions of being wrong. A magnitude — a separation, an absolute error — is carried by a
perceptually uniform sequential ramp.</p>
<p class="sub"><b>Levels of theory are never pooled.</b> A model trained at a different level from the reference
data is marked ⚠ in the table: its energies are not comparable with these references, its dimensionless
diagnostics are.</p>
{protocol}
</section>
"""

TASK_INDEX = """<section id="tasks">
<p class="eyebrow">benchmark tasks</p>
<h2>The task index</h2>
<p class="sub">Each task is constructed so that the correct value is known in advance. That is what makes a single
number interpretable: a result can be read against the value it should have, not against another model.</p>
<table class="rank idx"><thead><tr><th>id</th><th>task</th><th>quantity</th><th>correct</th>
<th>no long-range term</th></tr></thead><tbody>{rows}</tbody></table>
</section>
"""

GROUP_HEADER = """<section class="grouphead" id="group-{letter}">
<p class="eyebrow" style="color:{colour}">group {letter} — {sub}</p>
<h2><span class="gid" style="background:{colour}">{letter}</span>{title}</h2>
<p>{blurb}</p>
{protocol}
</section>
"""

GROUP_BLURB = {
    "A": "What a charged pair does once the fragments are far enough apart that nothing local can reach between "
         "them. The reference is not a fitted curve but an asymptotic law, so a model either reproduces it or it "
         "does not. The protocol below is the scan every task in this group is measured on.",
    "B": "A charge is placed far from a pair of local arrangements, and the energies are differenced twice. "
         "Composition and total charge are identical in all four terms, so every offset cancels and only the "
         "long-range reply survives — a quantity a Coulomb tail alone cannot produce.",
    "C": "The electrostatic variables themselves: whether a model's atomic charges sum to the charge it was given, "
         "and how they compare with the reference partitioning. A model can have a plausible energy and "
         "implausible charges.",
    "D": "What is left after every pair contribution is removed. A potential assembled from pairs has no access "
         "to this term however well it does on dimers.",
}

TASK_ROWS = [
    ("A1", "Asymptotic decay", "tail fraction", "1", "0"),
    ("A2", "Residuals against the reference", "E_int error (meV)", "0", "the DFT span"),
    ("A3", "Parity between two models", "agreement on the identity line", "—", "—"),
    ("B1", "Fixed-charge response", "ΔΔE slope", "1", "0"),
    ("C1", "Charge conservation", "Σq − Q, parity vs Löwdin", "0", "n/a if no charges"),
    ("D1", "Three-body non-additivity", "E₃ (meV)", "reference E₃", "0"),
]


KEY = """<section id="key">
<p class="eyebrow">notation</p>
<h2>Model colour key</h2>
<p class="sub">Models are grouped by family, which shares a hue; members of a family are told apart by marker.
The same colours are used wherever a model appears. Hue carries the family and shape carries the member, so
neither channel is load-bearing on its own.</p>
<div class="keywrap">{families}</div>
</section>
"""

RANKING = """<section id="score">
<h2><span class="num">score</span>The long-range score</h2>
<div class="eq">$$\\mathrm{LRS} = 100\\left(\\prod_{j} s_j\\right)^{1/n},\\qquad
s_j \\in [0,1],\\; 1 = \\text{correct}$$</div>
<p>Four quantities, each with a value that is right by construction, each mapped so that <b>1</b> is correct and
<b>0</b> is the no-long-range floor: the fraction of the \\(r\\)-dependence reproduced, how much of the physical
magnitude is captured, the reply to a distant charge, and the offset-free error measured against what a flat model
would score. The <em>geometric</em> mean is the point — a model that scores zero on any one of them has no
long-range physics, and credit elsewhere should not disguise that.</p>
<p class="sub">The bars beside each score are the four components in that order, so the number is never shown
without the thing it was built from. A score marked <b>*</b> was built from fewer than four — a task that did not
run cannot be told from one that was passed, so those are not comparable with the rest. This summarises these tests on these sets; it is not a ranking of models in
general. The ceiling row is the dataset's own multipole + induction prior, not a model. Click a column to re-sort, and the
dot beside a model to take it out of every widget below — the table and the figures always show the same set.</p>
<table class="rank" id="rank-body"></table>
</section>
"""


def payloads(runs_dir, data_dir=None, verbose: bool = True) -> dict:
    """{key: {blob, models, n, label}} — one gzipped, base64 chemiscope dataset per set, plus its atoms twin.

    Shared by the standalone explorer page and by the notebook build, so both show the same thing.
    """
    import chemiscope

    runs = pathlib.Path(runs_dir)
    data = data_dir or str(runs.parent / "lr-sets")
    results = {}
    for p in sorted(runs.glob("*.json")):
        with open(p) as f:
            J = json.load(f)
        if "sets" in J and "oracle" not in p.stem:
            results[p.stem] = J
    if not results:
        return {}

    built = {}
    for setname, kind, label in (("dimers", "E_int", "E_int"), ("trimers", "E3", "E3"),
                                 ("families_fixedq", "ddE", "ddE")):
        recs, groups = load_set(data, setname)
        if not recs:
            continue
        ents, models = entries(results, recs, setname, groups, kind)
        if not ents:
            continue
        ds = build_dataset(ents, models, setname, label)
        if ds is None:
            continue
        payload = chemiscope.create_input(
            structures=ds["frames"], properties=ds["properties"],
            metadata={"name": f"lrbench {setname}",
                      "description": f"{ds['n']} scored {setname} groups, {len(ds['models'])} models"})
        built[setname] = {"blob": encode(payload), "models": ds["models"], "n": ds["n"], "label": label}
        if verbose:
            print(f"[chemiscope] {setname}: {ds['n']} structures, {len(ds['models'])} models, "
                  f"{len(payload['properties'])} global properties")
        if ds["atom_properties"]:
            pay_a = chemiscope.create_input(
                structures=ds["frames"], properties=ds["atom_properties"], environments=ds["environments"],
                metadata={"name": f"lrbench {setname} — atoms",
                          "description": f"every atom of {ds['n']} scored {setname} groups"})
            built[f"{setname}-atoms"] = {"blob": encode(pay_a), "models": ds["models"], "n": ds["n"],
                                         "label": label}
            if verbose:
                print(f"[chemiscope] {setname}-atoms: {len(ds['environments'])} environments, "
                      f"{len(ds['atom_properties'])} atomic properties")
    return built


def pick(models, *want):
    for w in want:
        for m in models:
            if w in m:
                return m
    return models[0] if models else ""


#: The test taxonomy. Each task carries an identifier that is used in the headings, in the ranking table and in
#: any report generated from a run, so a result can be cited as "A1 on dimers/ion_pair" rather than described.
TASK_GROUPS = {
    "A": "Asymptotics — the far field",
    "B": "Response — the reply to a distant charge",
    "C": "Electrostatic variables — charges and moments",
    "D": "Many-body — what pairs cannot reach",
}

#: Colours used to annotate the terms of an equation, and then repeated as chips beneath it. Qualitative, because
#: the terms of an equation are an identity and not a magnitude.
TERM_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]


def view_specs(built: dict) -> list:
    """The benchmark tasks, each named for the physical prior it asserts.

    Encoding follows what each channel is for, with one rule learned the hard way: chemiscope draws the *symbol*
    legend in a single colour, so a class carried by shape alone produces seven black markers that are genuinely
    hard to tell apart. Class is therefore encoded by shape **and** categorical colour together on the views where
    the class is the subject (A3, D1), and left off the symbol channel entirely where colour has to carry a
    magnitude or a signed error — there it stays one click away on the colour buttons.

    A signed error gets a diverging palette used symmetrically about zero; a magnitude gets a perceptually
    uniform sequential ramp.
    """
    out = []
    dim = built.get("dimers")
    if dim:
        m_lr = pick(dim["models"], "mace-polar", "so3lr-tauri", "orbmol")
        others = [m for m in dim["models"] if m != m_lr]
        m_loc = pick(others, "mace-h", "dpa4-plus", "grace", "mace-omol") if others else ""
        c0, c1, c2 = TERM_COLOURS[0], TERM_COLOURS[1], TERM_COLOURS[2]
        out.append(dict(
            id="v-decay", key="dimers", tid="A1", group="A", models=dim["models"], diagonal=False, center0=True,
            title="Asymptotic decay of the interaction energy",
            equation=(r"E_{\mathrm{int}}(r)\;\xrightarrow{\;r\to\infty\;}\;"
                      r"\textcolor{" + c0 + r"}{\frac{q_A q_B}{4\pi\varepsilon_0 r}}"
                      r"\;+\;\textcolor{" + c1 + r"}{O(r^{-2})}"),
            terms=[(c0, "the monopole term — all that survives for a charged pair"),
                   (c1, "dipole and higher, which die away faster")],
            prose=("Beyond the cutoff a charged pair is governed by one term, and it does not decay quickly. "
                   "Separation runs along x and the signed error up y, on a diverging scale centred on zero, so "
                   "white is exact and the two hues are the two directions of being wrong. An error that "
                   "<em>grows</em> with separation is the long-range failure; one that shrinks is ordinary "
                   "short-range inaccuracy."),
            settings={"map": {"x": {"property": "separation", "scale": "log"},
                              "y": {"property": f"err/{m_lr}"},
                              "color": {"property": f"err/{m_lr}", "palette": "bwr"},
                              # the second model's channel only exists when there *is* a second model
                              **({"size": {"property": f"abserr/{m_loc}", "factor": 40}} if m_loc else {})}}))
        out.append(dict(
            id="v-residual", key="dimers", tid="A2", group="A", models=dim["models"], diagonal=True,
            center0=False,
            title="Residuals against the reference",
            equation=(r"\textcolor{" + c0 + r"}{E^{\mathrm{model}}_{\mathrm{int}}}\;\;\text{vs}\;\;"
                      r"\textcolor{" + c1 + r"}{E^{\mathrm{DFT}}_{\mathrm{int}}},\qquad"
                      r"\textcolor{" + c2 + r"}{y = x}"),
            terms=[(c0, "the model, on the vertical axis"), (c1, "the reference, on the horizontal"),
                   (c2, "the identity line: agreement")],
            prose=("A model can sit on the identity line at contact and leave it in the tail, which is exactly "
                   "what the summary statistics are built to catch. Colour is separation, so the departure "
                   "orders itself by distance; shape and colour both carry the class on the next view."),
            settings={"map": {"x": {"property": "E_int/REF"}, "y": {"property": f"E_int/{m_lr}"},
                              "color": {"property": "separation", "scale": "log", "palette": "cividis"}}}))
        out.append(dict(
            id="v-parity", key="dimers", tid="A3", group="A", models=dim["models"], diagonal=True, parity=True,
            center0=False,
            title="Parity between two models",
            equation=(r"\textcolor{" + c0 + r"}{E^{A}_{\mathrm{int}}}\;\;\text{vs}\;\;"
                      r"\textcolor{" + c1 + r"}{E^{B}_{\mathrm{int}}}"),
            terms=[(c0, "model on the x axis — choose it below"),
                   (c1, "model on the y axis — choose it below")],
            prose=("Pick the model on each axis. Points on the identity line are structures the two agree on; the "
                   "spread away from it is where they differ. Two models that fail on the same structures fail "
                   "for a reason in the data; two that fail on different ones fail for reasons of their own. "
                   "Class is carried by both shape and colour here, so the legend is readable either way."),
            settings={"map": {"x": {"property": f"E_int/{m_lr}"},
                              "y": {"property": f"E_int/{m_loc or m_lr}"},
                              "symbol": "class",
                              "color": {"property": "class", "palette": "tab10"}}}))
    if "families_fixedq" in built:
        b = built["families_fixedq"]
        m = pick(b["models"], "mace-polar", "so3lr-tauri", "orbmol")
        c0, c1 = TERM_COLOURS[0], TERM_COLOURS[1]
        out.append(dict(
            id="v-response", key="families_fixedq", tid="B1", group="B", models=b["models"], diagonal=True,
            center0=False,
            title="Fixed-charge response",
            equation=(r"\Delta\Delta E=\textcolor{" + c0 + r"}{\big[E(s_1,\text{near})-E(s_0,\text{near})\big]}"
                      r"-\textcolor{" + c1 + r"}{\big[E(s_1,\text{far})-E(s_0,\text{far})\big]}"),
            terms=[(c0, "the local energy difference with the charge nearby"),
                   (c1, "the same difference with it far away — subtracting this removes everything local")],
            prose=("Four structures of identical composition and identical total charge, so every per-element "
                   "offset, every charge-state offset and any constant cancels exactly. What survives is the "
                   "reply of a local energy difference to a charge placed far away, and it decays as "
                   "\\(R^{-2}\\) — a question a Coulomb tail alone cannot answer."),
            settings={"map": {"x": {"property": "ddE/REF"}, "y": {"property": f"ddE/{m}"},
                              "color": {"property": "atoms", "palette": "magma"}}}))
    if "dimers-atoms" in built:
        c0, c1 = TERM_COLOURS[0], TERM_COLOURS[1]
        out.append(dict(
            id="v-charge", key="dimers-atoms", tid="C1", group="C", models=[], diagonal=True, center0=False,
            title="Charge conservation and the Löwdin reference",
            equation=(r"\textcolor{" + c0 + r"}{\sum_{i\in f} q_i} \;=\; \textcolor{" + c1 + r"}{Q_f}"
                      r"\qquad\text{and}\qquad \mu \;=\; \sum_i q_i \mathbf{r}_i"),
            terms=[(c0, "the model's atomic charges, summed over a fragment"),
                   (c1, "the fragment's formal charge — an integer, by construction")],
            prose=("Every atom of every scored dimer: the DFT Löwdin charge against the formal charge it was "
                   "assigned. The formal charges are integers, so they lie on a few horizontal lines; the real "
                   "charges spread along the identity line, and that spread is the polarisation and charge "
                   "transfer a model has to reproduce. Each point is one atomic environment, so selecting one "
                   "highlights that atom."),
            settings={"map": {"x": {"property": "q_lowdin"}, "y": {"property": "q_formal"},
                              "color": {"property": "fragment", "palette": "plasma"}}}))
    if "trimers" in built:
        b = built["trimers"]
        m = pick(b["models"], "mace-polar", "so3lr-tauri", "orbmol")
        c0, c1, c2 = TERM_COLOURS[0], TERM_COLOURS[1], TERM_COLOURS[2]
        out.append(dict(
            id="v-threebody", key="trimers", tid="D1", group="D", models=b["models"], diagonal=True,
            center0=False,
            title="Three-body non-additivity",
            equation=(r"E_3 \;=\; \textcolor{" + c0 + r"}{E_{ABC}}"
                      r"\;-\;\textcolor{" + c1 + r"}{(E_{AB}+E_{AC}+E_{BC})}"
                      r"\;+\;\textcolor{" + c2 + r"}{(E_A+E_B+E_C)}"),
            terms=[(c0, "the trimer"), (c1, "every pair, removed"),
                   (c2, "every monomer, added back")],
            prose=("What is left once every pair contribution is removed. A model assembled from pairs has no "
                   "access to this term however well it does on dimers, and cooperative polarisation lives here."),
            settings={"map": {"x": {"property": "E3/REF"}, "y": {"property": f"E3/{m}"},
                              "symbol": "class",
                              "color": {"property": "class", "palette": "tab10"}}}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--data", default=None, help="dataset directory (default: alongside runs, ../lr-sets)")
    ap.add_argument("-o", "--out", default="chemiscope-views.html")
    ap.add_argument("--chemiscope-version", default="1.1.0")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    built = payloads(a.runs, a.data)
    if not built:
        sys.exit("nothing to build: no set had both references and saved predictions")
    summary = [f"{b['n']} {k}" for k, b in built.items() if not k.endswith("-atoms")]

    sections, views = [], []
    specs = view_specs(built)
    for v in specs:
        views.append({"id": v["id"], "key": v["key"], "settings": v["settings"],
                      "models": v.get("models", []), "diagonal": bool(v.get("diagonal")),
                      "parity": bool(v.get("parity")), "center0": bool(v.get("center0"))})
        chips = "".join(f'<span><i style="background:{col}"></i>{txt}</span>'
                        for col, txt in v.get("terms", []))
        sections.append((v["group"], SECTION.format(
            id=v["id"], tid=v["tid"], gcol=pal.GROUPS[v["group"]]["colour"],
            group=f"{v['group']} · {TASK_GROUPS[v['group']]}",
            title=v["title"], equation=v["equation"], chips=chips, prose=v["prose"])))

    rank_names = []
    rank = {"cols": [], "rows": [], "sort": "lrs", "components": [
        {"k": "s_decay", "t": "decay", "col": "#0072B2"},
        {"k": "s_scale", "t": "scale", "col": "#009E73"},
        {"k": "s_response", "t": "response", "col": "#CC79A7"},
        {"k": "s_accuracy", "t": "accuracy", "col": "#E69F00"}]}
    try:
        from lrbench.diagnostics import collect

        _d = collect(a.runs, with_points=False)
        rank["cols"] = [
            {"k": "model", "t": "model", "h": "colour is the model family"},
            {"k": "lrs", "t": "LRS",
             "h": "long-range score, the geometric mean of the four components; * = fewer than four ran, so it "
                  "is not comparable with a complete score"},
            {"k": "bars", "t": "decay · scale · response · accuracy"},
            {"k": "lr", "t": "LR term"},
            {"k": "tail", "t": "tail", "h": "tail fraction: 1 follows the reference, 0 flat"},
            {"k": "eps", "t": "ε", "h": "0 exact, −1 the whole tail missing"},
            {"k": "ddE", "t": "ΔΔE", "h": "fixed-charge response slope"},
            {"k": "off8", "t": "8 Å MAE", "h": "offset-free error at 8 Å, meV"},
            {"k": "far12", "t": "≥12 Å", "h": "absolute E_int error beyond 12 Å, meV"},
            {"k": "fmae", "t": "F MAE", "h": "force MAE, meV/Å"},
            {"k": "level", "t": "level"}]
        for name, r in _d["results"].items():
            S = r["sets"].get("dimers")
            if not S:
                continue
            c = S["classes"].get("ion_pair", {})
            g = lambda tp, n=3: None if not tp or tp[0] is None else round(tp[0], n)  # noqa: E731
            eps = g(c.get("eps"))
            # the fixed-charge quartets live in their own set, not in dimers
            ddE = r["sets"].get("families_fixedq", {}).get("ddE")
            sc = score_components(r)
            st = pal.style_of(name, list(_d["results"]))
            rank["rows"].append({
                "fam": st["colour"], "family": st["label"], "marker": st["marker"],
                "model": r["label"][:36], "key": name, "ceiling": "oracle" in name,
                "lrs": None if sc["lrs"] is None else
                       (f"{sc['lrs']}*" if sc.get("partial") else sc["lrs"]),
                "s_decay": sc["decay"], "s_scale": sc["scale"], "s_response": sc["response"],
                "s_accuracy": sc["accuracy"],
                "lr": "ceiling" if "oracle" in name else ("yes" if r["has_lr"] else "no"),
                "tail": g(c.get("tail")), "eps": eps, "ddE": g(ddE, 4),
                "off8": g(c.get("off8"), 1), "far12": g(c.get("far12"), 1), "fmae": g(S.get("F_mae"), 1),
                "level": r["level"][:20] + (" ⚠" if r["mismatch"] else ""),
                "abs_eps": abs(eps) if eps is not None else 9e9,
            })
        rank_names = [n for n in _d["results"]]
        print(f"[chemiscope] ranking: {len(rank['rows'])} models")
    except Exception as e:
        print(f"[chemiscope] no ranking table ({type(e).__name__}: {e})")

    out = pathlib.Path(a.out).resolve()
    fam_colours = {k: v["colour"] for k, v in pal.FAMILIES.items()}
    MARK = {"o": '<circle cx="7" cy="7" r="5.2"/>',
            "s": '<rect x="2.2" y="2.2" width="9.6" height="9.6" rx="1.4"/>',
            "D": '<rect x="2.6" y="2.6" width="8.8" height="8.8" rx="1.2" transform="rotate(45 7 7)"/>',
            "^": '<polygon points="7,1.8 12.4,11.6 1.6,11.6"/>',
            "v": '<polygon points="7,12.2 1.6,2.4 12.4,2.4"/>',
            "P": '<path d="M5.4 1.8h3.2v3.6h3.6v3.2H8.6v3.6H5.4V8.6H1.8V5.4h3.6z"/>'}
    names = sorted(rank_names)
    by_fam = {}
    for nm in names:
        st = pal.style_of(nm, names)
        by_fam.setdefault((st["family"], st["label"], st["colour"]), []).append((nm, st["marker"]))
    key_html = ""
    for (_fam, label, colour), members in sorted(by_fam.items(), key=lambda kv: kv[0][1]):
        rows = "".join(
            f'<div><svg width="14" height="14" viewBox="0 0 14 14" fill="{colour}">{MARK.get(mk, MARK["o"])}'
            f'</svg>{nm}</div>' for nm, mk in sorted(members))
        key_html += '<div class="keyfam"><b>' + label + '</b>' + rows + '</div>'
    # the labels first, then each letter group with its protocol standing beside its tasks
    idx = "".join(
        f'<tr><td class="name"><span class="tid sm" style="background:{pal.GROUPS[tid[0]]["colour"]}">{tid}'
        f'</span></td><td>{task}</td><td>{q}</td><td class="ok">{good}</td><td class="bad">{floor}</td></tr>'
        for tid, task, q, good, floor in TASK_ROWS if any(v["tid"] == tid for v in specs))
    PROTO = {"A": PROTOCOL_A, "B": PROTOCOL_B}
    body = [TASK_INDEX.format(rows=idx), RANKING if rank["rows"] else "",
            KEY.format(families=key_html) if key_html else ""]
    for letter, gspec in pal.GROUPS.items():
        mine = [sec for g, sec in sections if g == letter]
        if not mine:
            continue
        body.append(GROUP_HEADER.format(letter=letter, colour=gspec["colour"], title=gspec["title"],
                                        sub=gspec["sub"], blurb=GROUP_BLURB[letter],
                                        protocol=PROTO.get(letter, "")))
        body += mine
    body.append(METHOD.format(protocol=PROTOCOL_C))

    html = HTML.format(
        body="\n".join(body),
        scan_css=SCAN_CSS.format(),
        scan_js=SCAN_JS.format(seq=json.dumps(pal.SEQUENTIAL["batlow"]),
                               div=json.dumps(pal.DIVERGING["vik"]),
                               fam=json.dumps(fam_colours)),
        modelcol=json.dumps({n: pal.style_of(n, rank_names)["colour"] for n in rank_names}),
        chemstops=json.dumps(pal.CHEMISCOPE_STOPS),
        rank=json.dumps(rank),
        ver=a.chemiscope_version,
        summary=" · ".join(summary) + f" · {len(built.get('dimers', {}).get('models', []))} models · "
               "one embedded dataset per set, reused by every view",
        data="{" + ",".join(f'"{k}":"{v["blob"]}"' for k, v in built.items()) + "}",
        views=json.dumps(views),
    )
    out.write_text(html)
    print(f"[chemiscope] wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, {len(views)} views)")
    if a.open:
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
